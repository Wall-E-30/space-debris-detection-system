"""
src/data/explore_dataset.py
---
Dataset exploration and visualization for SwissCube, SPARK, and any
YOLO-format space-object dataset.

Usage:
    # SwissCube (auto-detected from swisscube_bbox.json):
    python src/data/explore_dataset.py --data /path/to/SwissCube

    # YOLO-format dataset (images/ + labels/ directories):
    python src/data/explore_dataset.py --data /path/to/spark --format yolo

    # Limit to N sample images in the grid:
    python src/data/explore_dataset.py --data /path/to/SwissCube --samples 16

    # Save outputs to a custom folder:
    python src/data/explore_dataset.py --data /path/to/SwissCube --out reports/week1

Outputs (all saved to --out, default: outputs/exploration/):
    sample_grid.png       — annotated image grid with bounding boxes
    class_balance.png     — bar chart of examples per class
    bbox_stats.png        — bbox width, height, area distributions + aspect ratio
    exploration_report.md — paste-ready summary for docs/decisions.md
"""

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Color palette — one per class, cycles if more classes than colors
# ---------------------------------------------------------------------------
PALETTE = [
    "#7F77DD",  # purple
    "#1D9E75",  # teal
    "#D85A30",  # coral
    "#378ADD",  # blue
    "#BA7517",  # amber
    "#D4537E",  # pink
    "#639922",  # green
    "#E24B4A",  # red
    "#888780",  # gray
]


# ---------------------------------------------------------------------------
# Annotation loaders
# ---------------------------------------------------------------------------

def _load_swisscube(data_dir: Path):
    """
    Load SwissCube/BOP dataset.

    Returns
    -------
    records
    class_names
    """

    records = []
    class_names = ["satellite"]

    for split in ["training", "validation", "testing"]:

        split_dir = data_dir / split

        if not split_dir.exists():
            continue

        for seq in sorted(split_dir.glob("seq_*")):

            scene_dir = next(seq.iterdir())       # 000000

            rgb_dir = scene_dir / "rgb"

            info_file = scene_dir / "scene_gt_info.json"

            if not info_file.exists():
                continue

            with open(info_file) as f:
                gt_info = json.load(f)

            for image_id, objects in gt_info.items():

                img_path = rgb_dir / f"{int(image_id):06d}.jpg"

                if not img_path.exists():
                    img_path = rgb_dir / f"{int(image_id):06d}.png"

                if not img_path.exists():
                    continue

                boxes = []

                class_ids = []

                for obj in objects:
                    bbox = obj.get("bbox_obj") or obj.get("bbox_visib")

                    if bbox is None:
                        continue

                    x, y, w, h = bbox

                    boxes.append([x, y, x + w, y + h])
                    class_ids.append(0)
                    print(
                        img_path.name,
                        len(boxes),
                        boxes[:1]
                    )
                    break

                records.append(
                    {
                        "image_path": str(img_path),
                        "boxes": boxes,
                        "class_ids": class_ids,
                        "split": split,
                    }
                )

    return records, class_names

def _swisscube_boxes_for_image(img_path: Path, bbox_json: Path):
    """
    Try to find bounding-box data for a SwissCube image.
    Priority order:
      1. Same-stem .json sidecar (pose file) with 'bb_*' keys
      2. Global swisscube_bbox.json (used as a fallback bounding region)
      3. Segmentation mask (.seg.png) → derive tight bbox from mask
    If nothing is found, returns empty lists (image will still appear in grid).
    """
    boxes, class_ids = [], []

    # 1. Sidecar pose JSON (common in SwissCube / SPEED-family datasets)
    sidecar = img_path.with_suffix(".json")
    if not sidecar.exists():
        sidecar = img_path.parent / (img_path.stem + "_pose.json")

    if sidecar.exists():
        try:
            with open(sidecar) as f:
                pose = json.load(f)
            # Keys vary by export — try common names
            for xk, yk, wk, hk in [
                ("bb_left", "bb_top", "bb_width", "bb_height"),
                ("bbox_x", "bbox_y", "bbox_w", "bbox_h"),
                ("x", "y", "w", "h"),
            ]:
                if all(k in pose for k in (xk, yk, wk, hk)):
                    x, y, w, h = pose[xk], pose[yk], pose[wk], pose[hk]
                    boxes.append([x, y, x + w, y + h])
                    class_ids.append(0)
                    break
            # COCO-style bbox array
            if not boxes and "bbox" in pose:
                x, y, w, h = pose["bbox"]
                boxes.append([x, y, x + w, y + h])
                class_ids.append(0)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # 2. Segmentation mask → tight bbox
    if not boxes:
        for mask_path in [
            img_path.with_suffix(".seg.png"),
            img_path.parent / (img_path.stem + "_mask.png"),
        ]:
            if mask_path.exists():
                try:
                    mask = np.array(Image.open(mask_path).convert("L"))
                    ys, xs = np.where(mask > 10)
                    if len(xs) > 0:
                        boxes.append([int(xs.min()), int(ys.min()),
                                      int(xs.max()), int(ys.max())])
                        class_ids.append(0)
                except Exception:
                    pass
                break

    # 3. Global bbox.json as last resort (gives model bounds, not per-image)
    if not boxes and bbox_json.exists():
        try:
            with open(bbox_json) as f:
                meta = json.load(f)
            # May be a list of [x, y, w, h] or similar
            if isinstance(meta, dict):
                if "bbox" in meta:
                    vals = meta["bbox"]
                    if len(vals) == 4:
                        x, y, w, h = vals
                        boxes.append([x, y, x + w, y + h])
                        class_ids.append(0)
        except Exception:
            pass

    return boxes, class_ids


def _load_spark_csv(data_dir: Path):
    """
    SPARK format (confirmed from actual export):

        spark/
          train.csv, val.csv   — columns: "Image name","Mask name","Class","Bounding box"
          images/<ClassName>/<Image name>
          mask/<ClassName>/<Mask name>          (optional, not used here)

    Bounding box column is a string tuple like "(397, 253, 653, 486)" — x1,y1,x2,y2 pixel coords.
    Image folders are named per-class (e.g. "Cheops", "LisaPathfinder") and may not exactly
    match the "Class" column string (spacing/casing can differ), so we build a normalized
    lookup rather than assuming an exact folder match.

    Returns:
        records     : list of dicts
        class_names : list of str (sorted, in the order class ids were assigned)
    """
    images_root = data_dir / "images"

    # Build a normalized-name -> actual folder path lookup
    # (e.g. "lisapathfinder" -> images/LisaPathfinder)
    folder_lookup = {}
    if images_root.exists():
        for d in images_root.iterdir():
            if d.is_dir():
                key = re.sub(r"[^a-z0-9]", "", d.name.lower())
                folder_lookup[key] = d

    csv_files = []
    for name, split in [("train.csv", "train"), ("val.csv", "val"), ("test.csv", "test")]:
        p = data_dir / name
        if p.exists():
            csv_files.append((p, split))

    if not csv_files:
        print(f"[warn] No train.csv/val.csv found under {data_dir}")
        return [], []

    class_to_id = {}
    records = []

    for csv_path, split in csv_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Column names can vary slightly in casing/spacing across SPARK
            # export versions, so match case-insensitively.
            fieldmap = {c.lower().strip(): c for c in reader.fieldnames}
            col_img   = fieldmap.get("image name")
            col_class = fieldmap.get("class")
            col_bbox  = fieldmap.get("bounding box")

            if not all([col_img, col_class, col_bbox]):
                print(f"[warn] {csv_path.name}: expected columns "
                     f"'Image name', 'Class', 'Bounding box' — found {reader.fieldnames}")
                continue

            for row in reader:
                img_name  = row[col_img].strip()
                cls_name  = row[col_class].strip()
                bbox_str  = row[col_bbox].strip()

                if cls_name not in class_to_id:
                    class_to_id[cls_name] = len(class_to_id)
                cid = class_to_id[cls_name]

                # Parse "(x1, y1, x2, y2)" safely
                box = None
                try:
                    box = ast.literal_eval(bbox_str)
                except (ValueError, SyntaxError):
                    nums = re.findall(r"-?\d+\.?\d*", bbox_str)
                    if len(nums) >= 4:
                        box = tuple(float(n) for n in nums[:4])

                attempts = [
                    data_dir / "images" / cls_name / split / img_name,           # ./spark/images/Cheops/train/img.png
                    data_dir / "images" / cls_name.lower() / split / img_name,   # ./spark/images/cheops/train/img.png
                    data_dir / "images" / cls_name.title() / split / img_name,   # ./spark/images/Cheops/train/img.png
                ]
                
                img_path = attempts[0]  # Default fallback
                for attempt in attempts:
                    if attempt.exists():
                        img_path = attempt
                        break
                        
                # Print a debug line for the very first missing image
                if not img_path.exists() and len(records) == 0:
                    print(f"\n[DEBUG] Path resolution failed.")
                    print(f"[DEBUG] CSV data -> Class: '{cls_name}', Split: '{split}', Image: '{img_name}'")
                    print(f"[DEBUG] Last checked absolute path: {img_path.absolute()}\n")
                
                records.append({
                    "image_path": str(img_path),
                    "boxes": [list(box)] if box else [],
                    "class_ids": [cid] if box else [],
                    "split": split,
                    "_class_name": cls_name,
                })

    # class_names ordered by assigned id
    class_names = [None] * len(class_to_id)
    for name, cid in class_to_id.items():
        class_names[cid] = name

    # Drop rows whose image file doesn't actually exist (avoids silently
    # plotting broken-image placeholders in the sample grid)
    missing = 0
    valid_records = []
    for r in records:
        if Path(r["image_path"]).exists():
            valid_records.append(r)
        else:
            missing += 1
    if missing:
        print(f"[warn] {missing:,} rows reference image files that don't exist on disk "
             f"(path resolution may need adjusting — check folder_lookup)")

    return valid_records, class_names


def _load_yolo(data_dir: Path):
    """
    YOLO format: images in images/{train,val,test}/, labels in labels/{...}/.
    Also handles flat structures where images/ and labels/ are siblings.

    Returns:
        records     : list of dicts
        class_names : list of str (from data.yaml if present)
    """
    records = []
    class_names = []

    # 1. Try to read class names from data.yaml
    yaml_path = next(data_dir.rglob("data.yaml"), None)
    if not yaml_path:
        yaml_path = next(data_dir.rglob("*.yaml"), None)
    if yaml_path:
        class_names = _read_yaml_classes(yaml_path)

    # 2. Collect all image paths
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images_root = data_dir / "images"
    if not images_root.exists():
        images_root = data_dir

    all_images = [p for p in images_root.rglob("*")
                  if p.suffix.lower() in image_exts and p.is_file()]

    for img_path in all_images:
        # Mirror path: images/train/x.jpg → labels/train/x.txt
        label_path = Path(str(img_path).replace("/images/", "/labels/")
                         ).with_suffix(".txt")
        if not label_path.exists():
            label_path = img_path.with_suffix(".txt")

        boxes_norm, class_ids = [], []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        boxes_norm.append([cx, cy, bw, bh])  # YOLO normalized
                        class_ids.append(cid)

        split = img_path.parent.name if img_path.parent.name in (
            "train", "val", "test", "training", "validation", "testing") else "all"

        records.append({
            "image_path": str(img_path),
            "boxes_norm": boxes_norm,   # YOLO normalized xywh
            "class_ids": class_ids,
            "split": split,
        })

    # Update class names from data if yaml missing
    if not class_names and records:
        max_id = max((cid for r in records for cid in r["class_ids"]), default=0)
        class_names = [f"class_{i}" for i in range(max_id + 1)]

    return records, class_names


def _read_yaml_classes(yaml_path: Path):
    """Minimal YAML parser that extracts the 'names' list without pyyaml."""
    class_names = []
    in_names = False
    with open(yaml_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("names:"):
                rest = stripped[6:].strip()
                if rest.startswith("["):
                    # Inline list: names: [a, b, c]
                    rest = rest.strip("[]")
                    class_names = [n.strip().strip("'\"") for n in rest.split(",")]
                    break
                else:
                    in_names = True
            elif in_names:
                if stripped.startswith("-"):
                    class_names.append(stripped.lstrip("- ").strip("'\""))
                elif stripped and not stripped.startswith("#"):
                    in_names = False
    return class_names


def _boxes_to_pixels(record, img_w, img_h):
    """Convert any box format to pixel [x1, y1, x2, y2]."""
    pixel_boxes = []

    if "boxes_norm" in record and record["boxes_norm"]:
        for cx, cy, bw, bh in record["boxes_norm"]:
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            pixel_boxes.append([x1, y1, x2, y2])
    elif "boxes" in record:
        pixel_boxes = record["boxes"]

    return pixel_boxes


# ---------------------------------------------------------------------------
# Auto-detect format
# ---------------------------------------------------------------------------

def detect_and_load(data_dir: Path, format_hint: str = "auto"):
    if format_hint == "yolo":
        return _load_yolo(data_dir)
    if format_hint == "swisscube":
        return _load_swisscube(data_dir)
    if format_hint == "spark_csv":
        return _load_spark_csv(data_dir)

    # Auto-detect
    if (data_dir / "train.csv").exists() or (data_dir / "val.csv").exists():
        print("[detect] SPARK CSV format (found train.csv / val.csv)")
        return _load_spark_csv(data_dir)
    if (data_dir / "swisscube_bbox.json").exists():
        print("[detect] SwissCube format (found swisscube_bbox.json)")
        return _load_swisscube(data_dir)
    if list(data_dir.rglob("labels/*.txt")):
        print("[detect] YOLO format (found labels/*.txt)")
        return _load_yolo(data_dir)
    # Fallback: try SwissCube-style subdirs
    if any((data_dir / s).exists() for s in ("training", "validation", "testing")):
        print("[detect] SwissCube-style subdirectory layout")
        return _load_swisscube(data_dir)
    # Last resort: YOLO
    print("[detect] Falling back to YOLO format")
    return _load_yolo(data_dir)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _draw_boxes_on_image(img: Image.Image, pixel_boxes, class_ids, class_names):
    """Return a copy of img with bounding boxes drawn."""
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    # Try system font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for box, cid in zip(pixel_boxes, class_ids):
        color = PALETTE[cid % len(PALETTE)]
        x1, y1, x2, y2 = [int(v) for v in box]
        # Box outline (2 px)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # Label background
        label = class_names[cid] if cid < len(class_names) else f"cls{cid}"
        bbox_text = draw.textbbox((x1, y1 - 16), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 16), label, fill="white", font=font)

    return out


def plot_sample_grid(records, class_names, n_samples, out_path):
    """Draw a grid of annotated sample images."""
    has_boxes = [r for r in records
                 if (r.get("boxes") or r.get("boxes_norm"))]
    no_boxes  = [r for r in records
                 if not (r.get("boxes") or r.get("boxes_norm"))]

    # Prefer records that have annotations
    pool = has_boxes if has_boxes else records
    chosen = random.sample(pool, min(n_samples, len(pool)))
    # Pad with no-box records if needed
    if len(chosen) < n_samples and no_boxes:
        chosen += random.sample(no_boxes, min(n_samples - len(chosen), len(no_boxes)))

    cols = min(4, n_samples)
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for ax, record in zip(axes, chosen):
        try:
            img = Image.open(record["image_path"])
            w, h = img.size
            pixel_boxes = _boxes_to_pixels(record, w, h)
            annotated = _draw_boxes_on_image(img, pixel_boxes,
                                              record["class_ids"], class_names)
            ax.imshow(annotated)
        except Exception as e:
            ax.set_facecolor("#1a1a2e")
            ax.text(0.5, 0.5, f"Load error\n{Path(record['image_path']).name}",
                    transform=ax.transAxes, ha="center", va="center",
                    color="white", fontsize=8)

        ax.axis("off")
        n_boxes = len(record.get("boxes") or record.get("boxes_norm") or [])
        stem = Path(record["image_path"]).stem
        split = record.get("split", "")
        ax.set_title(f"{stem[:20]}\n[{split}] {n_boxes} box(es)",
                     fontsize=7, pad=3)

    for ax in axes[len(chosen):]:
        ax.axis("off")

    # Legend
    patches = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=n)
               for i, n in enumerate(class_names)]
    if patches:
        fig.legend(handles=patches, loc="lower center",
                   ncol=min(len(patches), 6), fontsize=9,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Sample images with bounding boxes", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_class_balance(records, class_names, out_path):
    """Bar chart: number of annotated objects per class."""
    counts = defaultdict(int)
    for r in records:
        for cid in r["class_ids"]:
            label = class_names[cid] if cid < len(class_names) else f"cls{cid}"
            counts[label] += 1

    if not counts:
        print("  [warn] No annotations found — skipping class balance chart.")
        return

    labels = sorted(counts)
    values = [counts[l] for l in labels]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="none", width=0.6)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Number of annotations", fontsize=11)
    ax.set_title("Class balance — annotation counts", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(values) * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Imbalance flag
    if len(values) > 1:
        ratio = max(values) / max(min(values), 1)
        if ratio > 5:
            ax.text(0.98, 0.95,
                    f"⚠ Imbalance ratio {ratio:.1f}×\nConsider focal loss / oversampling",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    color="#854F0B",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FAEEDA",
                              edgecolor="#EF9F27", alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    return counts


def plot_bbox_stats(records, class_names, out_path):
    """
    2×2 grid:
      - bbox width distribution (normalized)
      - bbox height distribution (normalized)
      - bbox area distribution (normalized)
      - aspect ratio distribution
    """
    widths, heights, areas, aspects = [], [], [], []

    for r in records:
        try:
            img = Image.open(r["image_path"])
            iw, ih = img.size
        except Exception:
            iw, ih = 1, 1

        boxes = _boxes_to_pixels(r, iw, ih)
        for x1, y1, x2, y2 in boxes:
            bw = max(abs(x2 - x1), 1)
            bh = max(abs(y2 - y1), 1)
            widths.append(bw / iw)
            heights.append(bh / ih)
            areas.append((bw * bh) / (iw * ih))
            aspects.append(bw / bh)

    if not widths:
        print("  [warn] No bbox data — skipping bbox stats chart.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Bounding box statistics", fontsize=13)

    _hist(axes[0, 0], widths,  "Width (% of image)",  "#7F77DD", bins=30)
    _hist(axes[0, 1], heights, "Height (% of image)", "#1D9E75", bins=30)
    _hist(axes[1, 0], areas,   "Area (% of image)",   "#D85A30", bins=30)
    _hist(axes[1, 1], aspects, "Aspect ratio (w/h)",  "#378ADD", bins=30,
          vline=1.0, vline_label="square")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

    return widths, heights, areas, aspects


def _hist(ax, data, xlabel, color, bins=20, vline=None, vline_label=None):
    ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="none")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if vline is not None:
        ax.axvline(vline, color="#444", linestyle="--", linewidth=1,
                   label=vline_label)
        ax.legend(fontsize=9)
    # Median line
    med = float(np.median(data))
    ax.axvline(med, color=color, linestyle=":", linewidth=1.5,
               label=f"median {med:.2f}")
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(records, class_names, counts, stats, data_dir, out_path):
    """
    Writes a paste-ready markdown block for docs/decisions.md.
    stats = (widths, heights, areas, aspects) or None
    """
    total_images = len(records)
    annotated = sum(1 for r in records if r.get("class_ids"))
    total_boxes = sum(len(r["class_ids"]) for r in records)

    splits = defaultdict(int)
    for r in records:
        splits[r.get("split", "unknown")] += 1

    lines = [
        "## Dataset exploration — findings",
        "",
        f"**Dataset path:** `{data_dir}`  ",
        f"**Run date:** {__import__('datetime').date.today()}",
        "",
        "### Image inventory",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total images | {total_images:,} |",
        f"| Images with annotations | {annotated:,} |",
        f"| Total bounding boxes | {total_boxes:,} |",
    ]

    for split, n in sorted(splits.items()):
        lines.append(f"| Images in `{split}` | {n:,} |")

    if counts:
        lines += [
            "",
            "### Class distribution",
            "",
            "| Class | Annotations | % of total |",
            "|---|---|---|",
        ]
        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct = cnt / max(total_boxes, 1) * 100
            lines.append(f"| `{cls}` | {cnt:,} | {pct:.1f}% |")

        if len(counts) > 1:
            ratio = max(counts.values()) / max(min(counts.values()), 1)
            flag = f"⚠ **Imbalance ratio: {ratio:.1f}×**" if ratio > 5 else f"Imbalance ratio: {ratio:.1f}×"
            lines += ["", flag]
            if ratio > 5:
                lines.append("→ Apply focal loss and/or oversample minority classes.")

    if stats:
        widths, heights, areas, aspects = stats
        lines += [
            "",
            "### Bounding box statistics",
            "",
            "| Metric | Median | Min | Max |",
            "|---|---|---|---|",
        ]
        for name, arr in [("Width (norm.)", widths), ("Height (norm.)", heights),
                           ("Area (norm.)", areas), ("Aspect ratio", aspects)]:
            lines.append(f"| {name} | {np.median(arr):.3f} |"
                         f" {np.min(arr):.3f} | {np.max(arr):.3f} |")

    lines += [
        "",
        "### Decisions / flags",
        "",
        "- [ ] Class balance acceptable for training (no focal loss needed)",
        "- [ ] Annotation alignment verified (boxes match objects visually)",
        "- [ ] Image resolution consistent across splits",
        "- [ ] `data.yaml` written jointly with Person A",
    ]

    out_path.write_text("\n".join(lines))
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Explore and visualize a space-object detection dataset."
    )
    parser.add_argument("--data",    required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--format",  default="auto",
                        choices=["auto", "swisscube", "yolo", "spark_csv"],
                        help="Annotation format (default: auto-detect)")
    parser.add_argument("--samples", type=int, default=16,
                        help="Number of images in the sample grid (default: 16)")
    parser.add_argument("--out",     default="outputs/exploration",
                        help="Output directory (default: outputs/exploration)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"[error] Dataset directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Exploring: {data_dir}")
    print(f"  Output:    {out_dir}")
    print(f"{'='*60}\n")

    print("[1/4] Loading dataset …")
    records, class_names = detect_and_load(data_dir, args.format)
    if not records:
        print("[error] No images found. Check --data path and --format.")
        sys.exit(1)
    print(f"       {len(records):,} images found  |  classes: {class_names or ['(none)']}")

    print("\n[2/4] Generating sample grid …")
    plot_sample_grid(records, class_names, args.samples,
                     out_dir / "sample_grid.png")

    print("\n[3/4] Plotting class balance …")
    counts = plot_class_balance(records, class_names,
                                out_dir / "class_balance.png")

    print("\n[4/4] Plotting bbox statistics …")
    stats = plot_bbox_stats(records, class_names,
                            out_dir / "bbox_stats.png")

    print("\n[+]   Writing markdown report …")
    write_report(records, class_names, counts or {}, stats, data_dir,
                 out_dir / "exploration_report.md")

    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Done. Outputs:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:<30} {size_kb:>5} KB")
    print(f"{'='*60}\n")

    # Quick console summary
    total_boxes = sum(len(r["class_ids"]) for r in records)
    annotated   = sum(1 for r in records if r.get("class_ids"))
    print("  Quick summary")
    print(f"    Total images   : {len(records):,}")
    print(f"    Annotated      : {annotated:,}  ({100*annotated/max(len(records),1):.0f}%)")
    print(f"    Total boxes    : {total_boxes:,}")
    if counts:
        print(f"    Classes        : {len(counts)}")
        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"      {cls:<30} {cnt:>8,}")
    if len(records) > annotated:
        print(f"\n  ⚠  {len(records)-annotated:,} images have NO annotations.")
        print("     Check annotation files exist alongside images.")
    if counts and len(counts) > 1:
        ratio = max(counts.values()) / max(min(counts.values()), 1)
        if ratio > 5:
            print(f"\n  ⚠  Severe class imbalance (ratio {ratio:.1f}×).")
            print("     Add to docs/decisions.md — use focal loss in training.")
    print()


if __name__ == "__main__":
    main()