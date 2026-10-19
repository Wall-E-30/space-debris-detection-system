"""
scripts/convert_spark2021_to_yolo.py
---
Convert SPARK2021 Dataset Annotations to Ultralytics YOLOv8 Format.

Key Operations:
1. BBox Convention Swap:
   SPARK2021 uses row/column ordering: [R_min, C_min, R_max, C_max] == [y_min, x_min, y_max, x_max].
   Converts to YOLO normalized center format: <class_id> <x_center> <y_center> <width> <height>.
2. 11-Class Mapping:
   Maps satellite class strings and integer IDs (e.g. in test_labels.csv) to standard indices [0..10]:
   0: AcrimSat, 1: Aquarius, 2: Aura, 3: Calipso, 4: Cloudsat,
   5: CubeSat, 6: Debris, 7: Jason, 8: Sentinel-6, 9: Terra, 10: TRMM.
3. Split Organization & Collision Avoidance:
   SPARK2021 partitions images into per-class subdirectories in train_rgb.zip and validate_rgb.zip,
   meaning multiple classes have identically named files (e.g. 'image_00001_img.jpg').
   To produce a clean, flat YOLO directory without filename collisions, files are named:
   '{class_name}_{image_stem}.{ext}' in images/{train, val, test}/ and labels/{train, val, test}/.
4. Multi-mode Image Transfer:
   Supports copy, hardlink, symlink, or --labels-only / --skip-images modes.
   Fast O(1) candidate lookup finds images whether stored as .jpg or .png, flat or nested.
5. Visual Verification Samples:
   Renders bounding boxes on sample images to verify the row/column coordinate swap visually.
"""

import argparse
import ast
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CLASS_NAMES: List[str] = [
    "AcrimSat",     # 0
    "Aquarius",     # 1
    "Aura",         # 2
    "Calipso",      # 3
    "Cloudsat",     # 4
    "CubeSat",      # 5
    "Debris",       # 6
    "Jason",        # 7
    "Sentinel-6",   # 8
    "Terra",        # 9
    "TRMM",         # 10
]

CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS_NAME: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def resolve_class_id_and_name(raw_cls: Any) -> Tuple[int, str]:
    """Resolve raw class (string name or integer string/id) to (class_id, class_name)."""
    s = str(raw_cls).strip()
    if s.isdigit():
        idx = int(s)
        if idx in ID_TO_CLASS_NAME:
            return idx, ID_TO_CLASS_NAME[idx]
        raise ValueError(f"Unknown numeric class ID: {idx}")

    # Case-insensitive string match
    for name, idx in CLASS_NAME_TO_ID.items():
        if name.lower() == s.lower():
            return idx, name

    raise ValueError(f"Unknown class name: '{raw_cls}'")


def parse_bbox(bbox_raw: str) -> Optional[List[float]]:
    """Parse bounding box string '[r_min, c_min, r_max, c_max]'."""
    bbox_raw = bbox_raw.strip()
    try:
        val = json.loads(bbox_raw)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            return [float(x) for x in val]
    except Exception:
        pass

    try:
        val = ast.literal_eval(bbox_raw)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            return [float(x) for x in val]
    except Exception:
        pass

    clean = bbox_raw.strip("[]() ")
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    if len(parts) == 4:
        try:
            return [float(x) for x in parts]
        except ValueError:
            return None
    return None


def spark_bbox_to_yolo(
    r_min: float,
    c_min: float,
    r_max: float,
    c_max: float,
    img_w: float = 1024.0,
    img_h: float = 1024.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert SPARK2021 [R_min, C_min, R_max, C_max] == [y_min, x_min, y_max, x_max]
    to normalized YOLO (cx, cy, w, h).
    """
    # Swap coordinates: R is Y (rows), C is X (columns)
    y_min, y_max = float(r_min), float(r_max)
    x_min, x_max = float(c_min), float(c_max)

    # Sanity check
    if x_max <= x_min or y_max <= y_min:
        return None

    # Clip to image boundary
    x_min = max(0.0, min(x_min, img_w))
    x_max = max(0.0, min(x_max, img_w))
    y_min = max(0.0, min(y_min, img_h))
    y_max = max(0.0, min(y_max, img_h))

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w <= 0.0 or box_h <= 0.0:
        return None

    cx = (x_min + (box_w / 2.0)) / img_w
    cy = (y_min + (box_h / 2.0)) / img_h
    norm_w = box_w / img_w
    norm_h = box_h / img_h

    # Final range clamping
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return cx, cy, norm_w, norm_h


def locate_image_file(
    raw_dir: Path,
    split_name: str,
    cls_name: str,
    img_filename: str,
) -> Optional[Path]:
    """
    O(1) fast lookup for source image file across candidate extracted locations and extensions.
    """
    stem = Path(img_filename).stem
    orig_ext = Path(img_filename).suffix
    exts = [orig_ext, ".jpg", ".png", ".jpeg", ".JPG", ".PNG"]

    # Candidate directories for extracted images
    base_dirs = [
        raw_dir / f"{split_name}_rgb",
        raw_dir / f"{split_name}_rgb" / f"{split_name}_rgb",
        raw_dir / split_name,
        raw_dir / "images" / split_name,
        raw_dir / f"{split_name}_images",
    ]

    for b in base_dirs:
        if not b.exists():
            continue
        # Check class subfolder first (SPARK2021 convention for train/validate)
        class_dir = b / cls_name
        if class_dir.exists():
            for ext in exts:
                candidate = class_dir / f"{stem}{ext}"
                if candidate.is_file():
                    return candidate

        # Check directly under base (SPARK2021 convention for test_rgb)
        for ext in exts:
            candidate = b / f"{stem}{ext}"
            if candidate.is_file():
                return candidate

    return None


def link_or_copy_file(src: Path, dst: Path, mode: str = "copy") -> None:
    """Transfer file using hardlink, symlink, or copy with safe fallback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except (OSError, NotImplementedError):
            pass

    elif mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except (OSError, NotImplementedError):
            pass

    shutil.copy2(src, dst)


def render_verification_sample(
    img_path: Path,
    labels: List[Tuple[int, float, float, float, float]],
    out_path: Path,
    img_w: int = 1024,
    img_h: int = 1024,
) -> None:
    """Overlay bounding box and label on an image to visually verify coordinate transformation."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        colors = [
            "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
            "#48F90A", "#00F5D4", "#01BEFE", "#2B44FF", "#9B5DE5", "#F15BB5"
        ]

        for cls_id, cx, cy, w, h in labels:
            xmin = (cx - w / 2.0) * img_w
            ymin = (cy - h / 2.0) * img_h
            xmax = (cx + w / 2.0) * img_w
            ymax = (cy + h / 2.0) * img_h

            color = colors[cls_id % len(colors)]
            cls_name = ID_TO_CLASS_NAME.get(cls_id, f"cls_{cls_id}")
            label_text = f"{cls_id}: {cls_name}"

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            draw.rectangle([xmin, max(0, ymin - 20), xmin + len(label_text) * 9, max(20, ymin)], fill=color)
            draw.text((xmin + 4, max(0, ymin - 18)), label_text, fill="black")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
    except Exception as e:
        print(f"    [!] Failed to generate verification sample for {img_path.name}: {e}")


def process_split(
    split_name: str,
    yolo_split_name: str,
    csv_path: Path,
    raw_spark_dir: Path,
    out_dir: Path,
    img_w: float,
    img_h: float,
    skip_images: bool,
    link_mode: str,
    verify_samples_count: int,
) -> Dict[str, Any]:
    """
    Process a single split (train, validate->val, test).
    """
    print(f"\n[*] Processing split '{split_name}' -> '{yolo_split_name}'...")
    print(f"    Source CSV: {csv_path}")

    images_dst = out_dir / "images" / yolo_split_name
    labels_dst = out_dir / "labels" / yolo_split_name
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    # Key by (class_name, image_filename) to prevent collision between identically named files in different class folders
    image_annotations: Dict[Tuple[str, str], List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    total_rows = 0
    invalid_rows = 0
    class_counts: Counter = Counter()

    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            img_filename = (row.get("image") or row.get("Image name") or "").strip()
            if not img_filename:
                invalid_rows += 1
                continue

            raw_cls = row.get("class") or row.get("Class") or ""
            try:
                cls_id, cls_name = resolve_class_id_and_name(raw_cls)
            except ValueError:
                invalid_rows += 1
                continue

            bbox_raw = row.get("bbox") or ""
            parsed = parse_bbox(bbox_raw)
            if not parsed:
                invalid_rows += 1
                continue

            r_min, c_min, r_max, c_max = parsed
            yolo_box = spark_bbox_to_yolo(r_min, c_min, r_max, c_max, img_w=img_w, img_h=img_h)
            if not yolo_box:
                invalid_rows += 1
                continue

            cx, cy, w, h = yolo_box
            image_annotations[(cls_name, img_filename)].append((cls_id, cx, cy, w, h))
            class_counts[cls_name] += 1

    unique_item_count = len(image_annotations)
    print(f"    Read {total_rows} rows across {unique_item_count} unique items.")
    if invalid_rows > 0:
        print(f"    Skipped {invalid_rows} invalid/empty/degenerate bounding boxes.")

    # Write YOLO label files and optionally transfer images
    labels_written = 0
    images_transferred = 0
    missing_images = 0
    verification_done = 0
    sample_vis_dir = out_dir / "verification_samples" / yolo_split_name

    for (cls_name, img_filename), boxes in image_annotations.items():
        stem = Path(img_filename).stem
        # Use {class_name}_{stem} prefix for unambiguous flattening
        item_base_name = f"{cls_name}_{stem}"
        label_file = labels_dst / f"{item_base_name}.txt"

        with open(label_file, "w", encoding="utf-8") as lf:
            for cls_id, cx, cy, w, h in boxes:
                lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        labels_written += 1

        if not skip_images:
            src_img = locate_image_file(raw_spark_dir, split_name, cls_name, img_filename)
            if src_img:
                dst_img = images_dst / f"{item_base_name}{src_img.suffix}"
                link_or_copy_file(src_img, dst_img, mode=link_mode)
                images_transferred += 1

                if verification_done < verify_samples_count:
                    sample_out = sample_vis_dir / f"verify_{item_base_name}.jpg"
                    render_verification_sample(
                        img_path=src_img,
                        labels=boxes,
                        out_path=sample_out,
                        img_w=int(img_w),
                        img_h=int(img_h),
                    )
                    verification_done += 1
            else:
                missing_images += 1

    print(f"    [OK] Labels generated:     {labels_written}")
    if not skip_images:
        print(f"    [OK] Images linked/copied: {images_transferred}")
        if missing_images > 0:
            print(f"    [!] Images not yet found on disk: {missing_images}")
            print(f"        (Note: Raw zip archives might not be extracted yet. Labels are ready!)")
    else:
        print("    [--labels-only / --skip-images was specified: Image transfer skipped]")

    if verification_done > 0:
        print(f"    [OK] {verification_done} verification sample images written to {sample_vis_dir}")

    return {
        "split": yolo_split_name,
        "source_split": split_name,
        "csv_path": str(csv_path.resolve()),
        "total_rows": total_rows,
        "unique_images": unique_item_count,
        "labels_written": labels_written,
        "images_transferred": images_transferred,
        "missing_images": missing_images,
        "invalid_rows_skipped": invalid_rows,
        "class_counts": dict(class_counts),
    }


def convert_spark2021(
    spark_dir: str = "data/raw/SPARK2021",
    out_dir: str = "data/processed/spark2021",
    img_w: float = 1024.0,
    img_h: float = 1024.0,
    skip_images: bool = False,
    link_mode: str = "copy",
    verify_samples_count: int = 5,
) -> Dict[str, Any]:
    """
    Main conversion orchestrator.
    """
    raw_p = Path(spark_dir)
    if not raw_p.exists():
        if Path("data/raw/SPARK-2021").exists():
            raw_p = Path("data/raw/SPARK-2021")
        elif Path("data/raw/SPARK2021").exists():
            raw_p = Path("data/raw/SPARK2021")
            
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  SPARK2021 TO ULTRALYTICS YOLO FORMAT CONVERTER")
    print("=" * 80)
    print(f"  Source Directory: {raw_p.resolve()}")
    print(f"  Output Directory: {out_p.resolve()}")
    print(f"  Image Geometry:   {int(img_w)}x{int(img_h)}")
    print(f"  Link Mode:        {link_mode} (Skip Images: {skip_images})")
    print("=" * 80)

    if not raw_p.exists():
        raise FileNotFoundError(f"Source SPARK2021 directory not found: {raw_p}")

    split_configs = [
        ("train", "train", raw_p / "train_labels.csv"),
        ("validate", "val", raw_p / "validate_labels.csv"),
        ("test", "test", raw_p / "test_labels.csv"),
    ]

    splits_summary: Dict[str, Any] = {}
    total_labels = 0
    total_images_transferred = 0

    for raw_split, yolo_split, csv_file in split_configs:
        if csv_file.exists():
            res = process_split(
                split_name=raw_split,
                yolo_split_name=yolo_split,
                csv_path=csv_file,
                raw_spark_dir=raw_p,
                out_dir=out_p,
                img_w=img_w,
                img_h=img_h,
                skip_images=skip_images,
                link_mode=link_mode,
                verify_samples_count=verify_samples_count,
            )
            splits_summary[yolo_split] = res
            total_labels += res["labels_written"]
            total_images_transferred += res["images_transferred"]
        else:
            print(f"[!] Warning: CSV not found for split '{raw_split}': {csv_file}")

    manifest: Dict[str, Any] = {
        "dataset": "SPARK2021",
        "conversion_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_directory": str(raw_p.resolve()),
        "output_directory": str(out_p.resolve()),
        "total_labels_written": total_labels,
        "total_images_transferred": total_images_transferred,
        "class_mapping": {idx: name for idx, name in enumerate(CLASS_NAMES)},
        "splits": splits_summary,
    }

    manifest_file = out_p / "conversion_summary.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("  CONVERSION COMPLETE")
    print(f"  Total Labels Written:       {total_labels}")
    print(f"  Total Images Transferred:  {total_images_transferred}")
    print(f"  Summary Manifest Saved:    {manifest_file.resolve()}")
    print("=" * 80 + "\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Convert SPARK2021 Dataset Annotations to Ultralytics YOLOv8 Format"
    )
    parser.add_argument(
        "--spark-dir",
        type=str,
        default="data/raw/SPARK2021",
        help="Root path to SPARK2021 raw dataset (default: data/raw/SPARK2021)",
    )
    parser.add_argument(
        "--out",
        "--out-dir",
        dest="out",
        type=str,
        default="data/processed/spark2021",
        help="Output directory for processed YOLO dataset (default: data/processed/spark2021)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Target image dimension in pixels (default: 1024)",
    )
    parser.add_argument(
        "--labels-only",
        "--skip-images",
        dest="skip_images",
        action="store_true",
        default=False,
        help="Generate YOLO label text files only without copying images",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="File transfer strategy when copying images (default: copy)",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=5,
        help="Number of verification images to generate per split with overlaid bounding boxes (default: 5)",
    )
    args = parser.parse_args()

    convert_spark2021(
        spark_dir=args.spark_dir,
        out_dir=args.out,
        img_w=float(args.imgsz),
        img_h=float(args.imgsz),
        skip_images=args.skip_images,
        link_mode=args.link_mode,
        verify_samples_count=args.verify_samples,
    )


if __name__ == "__main__":
    main()
