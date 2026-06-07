# EGTR Visualization Script
# Generates two outputs:
#   (a) Image with bounding boxes around detected objects
#   (b) Scene Graph diagram showing relationships between objects
#
# Usage:
#   python visualize_egtr.py
#
# Change IMAGE_PATH below to point to your image.

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from glob import glob

from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from util.box_ops import box_cxcywh_to_xyxy

# ============================================================
# >>> CONFIGURATION - CHANGE THESE <<<
# ============================================================

# Path to the input image
IMAGE_PATH = "img.jpeg"   # <-- CHANGE THIS

# Path to the model artifact (contains config.json + checkpoints/)
ARTIFACT_PATH = "SenseTime"

# Output directory for results
OUTPUT_DIR = "output_visualization"

# Thresholds
OBJ_SCORE_THRESHOLD = 0.35     # Minimum confidence for object detection
REL_SCORE_THRESHOLD = 0.25     # Minimum confidence for relation prediction
MAX_OBJECTS = 15               # Maximum number of objects to display
MAX_RELATIONS = 20             # Maximum number of relations to display

# Path to Visual Genome dataset (contains train.json, rel.json)
DATA_PATH = "dataset/visual_genome"

# ============================================================
# Load categories from dataset files
# ============================================================
import json as _json

def _load_categories(data_path):
    """Load object and relation categories from VG dataset files."""
    # Object categories from train.json (COCO format)
    with open(os.path.join(data_path, "train.json"), "r") as f:
        train_data = _json.load(f)
    cats = {c["id"]: c["name"] for c in train_data["categories"]}
    # Model uses id - 1 (0-indexed), sorted by original id
    obj_categories = [cats[k] for k in sorted(cats.keys())]

    # Relation categories from rel.json (skip '__background__')
    with open(os.path.join(data_path, "rel.json"), "r") as f:
        rel_data = _json.load(f)
    rel_categories = rel_data["rel_categories"][1:]  # remove '__background__'

    return obj_categories, rel_categories

VG_OBJECT_CATEGORIES, VG_RELATION_CATEGORIES = _load_categories(DATA_PATH)

# ============================================================
# Color palette for bounding boxes (distinctive colors)
# ============================================================
BOX_COLORS = [
    "#FF6B35", "#004E89", "#1A936F", "#C5283D", "#7B2D8E",
    "#F0C808", "#00A8E8", "#E83F6F", "#2EC4B6", "#FF9F1C",
    "#5C4D7D", "#44AF69", "#F8333C", "#FCAB10", "#2B9EB3",
    "#D72638", "#3F88C5", "#F49D37", "#140F2D", "#A23B72",
]


def load_model(artifact_path):
    """Load the EGTR model from checkpoint."""
    config = DeformableDetrConfig.from_pretrained(artifact_path)
    config.logit_adjustment = False
    config.use_contrastive_decoding = getattr(config, "use_contrastive_decoding", False)

    # Initialize model directly from config (no HuggingFace download needed)
    model = DetrForSceneGraphGeneration(config=config)

    # Find the latest checkpoint
    ckpt_path = sorted(
        glob(f"{artifact_path}/checkpoints/epoch=*.ckpt"),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k[6:]] = state_dict.pop(k)  # remove "model." prefix

    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, device


def preprocess_image(image_path):
    """Load and preprocess a single image."""
    feature_extractor = DeformableDetrFeatureExtractor(
        size=800, max_size=1333
    )
    image = Image.open(image_path).convert("RGB")
    encoding = feature_extractor(images=image, return_tensors="pt")
    return image, encoding, feature_extractor


@torch.no_grad()
def run_inference(model, encoding, device, num_labels):
    """Run the model on a preprocessed image and extract predictions."""
    pixel_values = encoding["pixel_values"].to(device)
    pixel_mask = encoding["pixel_mask"].to(device)

    outputs = model(
        pixel_values=pixel_values,
        pixel_mask=pixel_mask,
        output_attentions=False,
        output_attention_states=True,
        output_hidden_states=True,
    )

    # Object detection
    pred_logits = outputs.logits[0]  # [num_queries, num_labels]
    pred_boxes = outputs.pred_boxes[0]  # [num_queries, 4] in cxcywh normalized

    # Relation prediction
    pred_rel = torch.clamp(outputs.pred_rel[0], 0.0, 1.0)  # [nq, nq, num_rel]
    if outputs.pred_connectivity is not None:
        pred_connectivity = torch.clamp(outputs.pred_connectivity[0], 0.0, 1.0)
        pred_rel = torch.mul(pred_rel, pred_connectivity)

    # Object scores and classes
    obj_probs = pred_logits.softmax(-1)[:, :num_labels]
    obj_scores, pred_classes = torch.max(obj_probs, -1)

    return obj_scores.cpu(), pred_classes.cpu(), pred_boxes.cpu(), pred_rel.cpu()


def filter_detections(obj_scores, pred_classes, pred_boxes, image_size,
                      obj_threshold, max_objects):
    """Filter high-confidence detections and rescale boxes."""
    # Filter by confidence
    keep = obj_scores > obj_threshold
    indices = torch.where(keep)[0]

    if len(indices) == 0:
        print(f"WARNING: No objects detected above threshold {obj_threshold}.")
        print(f"  Top 5 scores: {obj_scores.topk(min(5, len(obj_scores))).values.tolist()}")
        # Fallback: take top-N
        topk = min(max_objects, len(obj_scores))
        indices = obj_scores.topk(topk).indices

    # Sort by score and limit
    scores = obj_scores[indices]
    sorted_idx = scores.argsort(descending=True)[:max_objects]
    indices = indices[sorted_idx]

    scores = obj_scores[indices].numpy()
    classes = pred_classes[indices].numpy()
    boxes_cxcywh = pred_boxes[indices]

    # Convert boxes to xyxy in image coordinates
    img_w, img_h = image_size
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    boxes_xyxy = boxes_xyxy * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    boxes_xyxy = boxes_xyxy.numpy()

    return indices, scores, classes, boxes_xyxy


def extract_relations(pred_rel, obj_indices, rel_threshold, max_relations):
    """Extract top relations between detected objects."""
    n_det = len(obj_indices)
    relations = []

    for i in range(n_det):
        for j in range(n_det):
            if i == j:
                continue
            qi = obj_indices[i]
            qj = obj_indices[j]
            rel_scores = pred_rel[qi, qj]  # [num_rel]
            max_score, max_rel = rel_scores.max(0)
            if max_score.item() > rel_threshold:
                relations.append({
                    "subject_idx": i,
                    "object_idx": j,
                    "relation_id": max_rel.item(),
                    "score": max_score.item(),
                })

    # Sort by score and limit
    relations = sorted(relations, key=lambda x: x["score"], reverse=True)
    relations = relations[:max_relations]
    return relations


def draw_bboxes(image, scores, classes, boxes, output_path):
    """Draw bounding boxes on the image (part a)."""
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(image)

    for idx, (score, cls, box) in enumerate(zip(scores, classes, boxes)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        color = BOX_COLORS[idx % len(BOX_COLORS)]
        label = VG_OBJECT_CATEGORIES[cls] if cls < len(VG_OBJECT_CATEGORIES) else f"cls_{cls}"

        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Label background
        label_text = f"{label}"
        fontsize = 11
        ax.text(
            x1, y1 - 4,
            label_text,
            fontsize=fontsize,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=color,
                edgecolor=color,
                alpha=0.85,
            ),
            verticalalignment="bottom",
        )

    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved bounding box image: {output_path}")


def draw_scene_graph(scores, classes, relations, output_path):
    """Draw scene graph diagram (part b)."""
    n_objects = len(classes)

    if n_objects == 0:
        print("No objects to draw in scene graph.")
        return

    # Node labels
    node_labels = []
    for i, cls in enumerate(classes):
        label = VG_OBJECT_CATEGORIES[cls] if cls < len(VG_OBJECT_CATEGORIES) else f"cls_{cls}"
        node_labels.append(label)

    # Layout: arrange nodes in an ellipse
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_facecolor("#F5F0EB")
    fig.patch.set_facecolor("#F5F0EB")

    # Compute node positions on an ellipse
    cx, cy = 0.5, 0.5
    rx, ry = 0.35, 0.35
    angles = np.linspace(0, 2 * np.pi, n_objects, endpoint=False)
    # Start from top
    angles = angles - np.pi / 2

    node_positions = []
    for angle in angles:
        x = cx + rx * np.cos(angle)
        y = cy + ry * np.sin(angle)
        node_positions.append((x, y))

    # Draw edges (relations)
    for rel in relations:
        si = rel["subject_idx"]
        oi = rel["object_idx"]
        ri = rel["relation_id"]
        rel_name = VG_RELATION_CATEGORIES[ri] if ri < len(VG_RELATION_CATEGORIES) else f"rel_{ri}"

        x1, y1 = node_positions[si]
        x2, y2 = node_positions[oi]

        # Draw arrow
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#444444",
                lw=1.5,
                connectionstyle="arc3,rad=0.15",
                shrinkA=18,
                shrinkB=18,
            ),
        )

        # Relation label at midpoint with slight offset
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        # Offset perpendicular to the edge direction
        dx = x2 - x1
        dy = y2 - y1
        length = max(np.sqrt(dx**2 + dy**2), 1e-6)
        offset_x = -dy / length * 0.04
        offset_y = dx / length * 0.04

        ax.text(
            mid_x + offset_x, mid_y + offset_y,
            rel_name,
            fontsize=9,
            fontweight="bold",
            fontstyle="italic",
            color="#7B2D8E",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="#CCCCCC",
                alpha=0.9,
            ),
        )

    # Draw nodes
    for idx, (pos, label) in enumerate(zip(node_positions, node_labels)):
        color = BOX_COLORS[idx % len(BOX_COLORS)]
        circle = plt.Circle(
            pos, 0.025,
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            zorder=10,
        )
        ax.add_patch(circle)
        ax.text(
            pos[0], pos[1] - 0.045,
            label,
            fontsize=10,
            fontweight="bold",
            ha="center", va="center",
            color="#222222",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        0.5, 0.02,
        "(b) Scene Graph",
        fontsize=14,
        fontweight="bold",
        fontstyle="italic",
        ha="center", va="bottom",
        transform=ax.transAxes,
    )

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scene graph: {output_path}")


def draw_combined(image, scores, classes, boxes, relations, output_path):
    """Draw combined figure: (a) Image with boxes + (b) Scene Graph side by side."""
    n_objects = len(classes)

    fig = plt.figure(figsize=(20, 9))

    # ----- Part (a): Image with bounding boxes -----
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)

    for idx, (score, cls, box) in enumerate(zip(scores, classes, boxes)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        color = BOX_COLORS[idx % len(BOX_COLORS)]
        label = VG_OBJECT_CATEGORIES[cls] if cls < len(VG_OBJECT_CATEGORIES) else f"cls_{cls}"

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
        )
        ax1.add_patch(rect)
        ax1.text(
            x1, y1 - 4,
            label,
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=color,
                edgecolor=color,
                alpha=0.85,
            ),
            verticalalignment="bottom",
        )

    ax1.set_xlim(0, image.width)
    ax1.set_ylim(image.height, 0)
    ax1.axis("off")
    ax1.set_title("(a) Detected Objects", fontsize=14, fontweight="bold", fontstyle="italic", pad=10)

    # ----- Part (b): Scene Graph -----
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#F5F0EB")

    if n_objects > 0:
        # Node labels
        node_labels = []
        for cls in classes:
            label = VG_OBJECT_CATEGORIES[cls] if cls < len(VG_OBJECT_CATEGORIES) else f"cls_{cls}"
            node_labels.append(label)

        # Layout
        cx_g, cy_g = 0.5, 0.5
        rx_g, ry_g = 0.35, 0.35
        angles_g = np.linspace(0, 2 * np.pi, n_objects, endpoint=False) - np.pi / 2
        node_positions = [(cx_g + rx_g * np.cos(a), cy_g + ry_g * np.sin(a)) for a in angles_g]

        # Edges
        for rel in relations:
            si = rel["subject_idx"]
            oi = rel["object_idx"]
            ri = rel["relation_id"]
            rel_name = VG_RELATION_CATEGORIES[ri] if ri < len(VG_RELATION_CATEGORIES) else f"rel_{ri}"

            x1, y1 = node_positions[si]
            x2, y2 = node_positions[oi]

            ax2.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#444444",
                    lw=1.5,
                    connectionstyle="arc3,rad=0.15",
                    shrinkA=18,
                    shrinkB=18,
                ),
            )
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            dx = x2 - x1
            dy = y2 - y1
            length = max(np.sqrt(dx**2 + dy**2), 1e-6)
            offset_x = -dy / length * 0.04
            offset_y = dx / length * 0.04

            ax2.text(
                mid_x + offset_x, mid_y + offset_y,
                rel_name,
                fontsize=9,
                fontweight="bold",
                fontstyle="italic",
                color="#7B2D8E",
                ha="center", va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="#CCCCCC",
                    alpha=0.9,
                ),
            )

        # Nodes
        for idx, (pos, label) in enumerate(zip(node_positions, node_labels)):
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            circle = plt.Circle(
                pos, 0.025,
                facecolor=color,
                edgecolor="white",
                linewidth=2,
                zorder=10,
            )
            ax2.add_patch(circle)
            ax2.text(
                pos[0], pos[1] - 0.045,
                label,
                fontsize=10,
                fontweight="bold",
                ha="center", va="center",
                color="#222222",
            )

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("(b) Scene Graph", fontsize=14, fontweight="bold", fontstyle="italic", pad=10)

    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined visualization: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load model
    print("=" * 60)
    print("EGTR Scene Graph Generation - Visualization")
    print("=" * 60)
    model, device = load_model(ARTIFACT_PATH)
    num_labels = model.config.num_labels

    # 2. Load and preprocess image
    print(f"\nProcessing image: {IMAGE_PATH}")
    image, encoding, feature_extractor = preprocess_image(IMAGE_PATH)
    print(f"  Image size: {image.size}")

    # 3. Run inference
    print("Running inference...")
    obj_scores, pred_classes, pred_boxes, pred_rel = run_inference(
        model, encoding, device, num_labels
    )
    print(f"  Detected {(obj_scores > OBJ_SCORE_THRESHOLD).sum().item()} objects above threshold {OBJ_SCORE_THRESHOLD}")

    # 4. Filter detections
    indices, scores, classes, boxes = filter_detections(
        obj_scores, pred_classes, pred_boxes, image.size,
        OBJ_SCORE_THRESHOLD, MAX_OBJECTS
    )
    print(f"  Using top {len(scores)} objects:")
    for i, (s, c) in enumerate(zip(scores, classes)):
        label = VG_OBJECT_CATEGORIES[c] if c < len(VG_OBJECT_CATEGORIES) else f"cls_{c}"
        print(f"    [{i}] {label} (score={s:.3f})")

    # 5. Extract relations
    relations = extract_relations(pred_rel, indices, REL_SCORE_THRESHOLD, MAX_RELATIONS)
    print(f"\n  Found {len(relations)} relations above threshold {REL_SCORE_THRESHOLD}:")
    for rel in relations:
        si = rel["subject_idx"]
        oi = rel["object_idx"]
        ri = rel["relation_id"]
        sub_label = VG_OBJECT_CATEGORIES[classes[si]] if classes[si] < len(VG_OBJECT_CATEGORIES) else f"cls_{classes[si]}"
        obj_label = VG_OBJECT_CATEGORIES[classes[oi]] if classes[oi] < len(VG_OBJECT_CATEGORIES) else f"cls_{classes[oi]}"
        rel_label = VG_RELATION_CATEGORIES[ri] if ri < len(VG_RELATION_CATEGORIES) else f"rel_{ri}"
        print(f"    {sub_label} --[{rel_label}]--> {obj_label} (score={rel['score']:.3f})")

    # 6. Generate visualizations
    print("\nGenerating visualizations...")
    img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    # (a) Image with bounding boxes
    bbox_path = os.path.join(OUTPUT_DIR, f"{img_name}_bboxes.png")
    draw_bboxes(image, scores, classes, boxes, bbox_path)

    # (b) Scene graph
    sg_path = os.path.join(OUTPUT_DIR, f"{img_name}_scene_graph.png")
    draw_scene_graph(scores, classes, relations, sg_path)

    # Combined
    combined_path = os.path.join(OUTPUT_DIR, f"{img_name}_combined.png")
    draw_combined(image, scores, classes, boxes, relations, combined_path)

    print(f"\nDone! All outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
