"""
src/data/split_dataset.py
---
Converts SPARK (CSV) or SwissCube (sidecar-JSON) datasets into YOLO format:

    output_dir/
      images/train/*.jpg   images/val/*.jpg   images/test/*.jpg
      labels/train/*.txt   labels/val/*.txt   labels/test/*.txt
      data.yaml

Each label .txt line is: "class_id x_center y_center width height"
(all normalized 0-1, YOLO convention).

Split is done PER CLASS (stratified) so rare classes like `debris` don't
get accidentally left out of val/test — critical here since SPARK's class
balance is uneven (see docs/decisions.md).

Images are COPIED by default (safe, keeps source dataset untouched).
Use --link on Linux/Mac to symlink instead and save disk space; on
Windows, --link falls back to copy automatically (symlinks need admin
rights there).

Usage:
    # SPARK (auto-detected from train.csv / val.csv):
    python src/data/split_dataset.py --data ./spark --out data/processed

    # SwissCube (auto-detected from swisscube_bbox.json):
    python src/data/split_dataset.py --data ./SwissCube --out data/processed

    # Custom split ratios (must sum to 1.0):
    python src/data/split_dataset.py --data ./spark --out data/processed --split 0.8 0.15 0.05

    # Symlink instead of copy (Linux/Mac only):
    python src/data/split_dataset.py --data ./spark --out data/processed --link

    # Dry run — show what would happen without writing files:
    python src/data/split_dataset.py --data ./spark --out data/processed --dry-run
"""

import argparse
import ast
import csv
import json
import os
import platform
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Loaders — same parsing logic as src/data/explore_dataset.py, kept
# self-contained here so this script has no import dependency on it.
# ---------------------------------------------------------------------------

def load_spark_csv(data_dir: Path):
    """
    SPARK format (confirmed from actual export):
        train.csv, val.csv — columns: "Image name","Mask name","Class","Bounding box"
        images/<ClassName>/<split>/<Image name>

    Note: the real folder layout nests an extra split-level directory
    under each class (e.g. images/Cheops/train/img.jpg,
    images/Cheops/val/img.jpg) — the CSV itself doesn't state this split,
    but since train.csv and val.csv are separate files, we know which
    split each row belongs to and use that to build the path.

    Bounding box column is a string tuple "(x1, y1, x2, y2)" in pixel coords.

    Returns:
        records     : list of {"image_path", "box": [x1,y1,x2,y2] or None, "class_name"}
        class_names : sorted list of unique class names
    """
    images_root = data_dir / "images"

    folder_lookup = {}
    if images_root.exists():
        for d in images_root.iterdir():
            if d.is_dir():
                key = re.sub(r"[^a-z0-9]", "", d.name.lower())
                folder_lookup[key] = d

    csv_files = [(data_dir / "train.csv", "train"), (data_dir / "val.csv", "val"),
                (data_dir / "test.csv", "test")]
    csv_files = [(p, split) for p, split in csv_files if p.exists()]
    if not csv_files:
        return [], []

    records = []
    class_set = set()

    for csv_path, csv_split in csv_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldmap = {c.lower().strip(): c for c in reader.fieldnames}
            col_img   = fieldmap.get("image name")
            col_class = fieldmap.get("class")
            col_bbox  = fieldmap.get("bounding box")
            if not all([col_img, col_class, col_bbox]):
                print(f"[warn] {csv_path.name}: unexpected columns {reader.fieldnames}")
                continue

            for row in reader:
                img_name = row[col_img].strip()
                cls_name = row[col_class].strip()
                bbox_str = row[col_bbox].strip()
                class_set.add(cls_name)

                box = None
                try:
                    box = list(ast.literal_eval(bbox_str))
                except (ValueError, SyntaxError):
                    nums = re.findall(r"-?\d+\.?\d*", bbox_str)
                    if len(nums) >= 4:
                        box = [float(n) for n in nums[:4]]

                key = re.sub(r"[^a-z0-9]", "", cls_name.lower())
                folder = folder_lookup.get(key)
                class_dir = folder if folder else (images_root / cls_name)

                # Try the split-nested path first (images/<Class>/<split>/<img>),
                # which is the real layout — fall back to a flat path
                # (images/<Class>/<img>) in case a different export doesn't
                # nest by split.
                nested_path = class_dir / csv_split / img_name
                flat_path = class_dir / img_name
                img_path = nested_path if nested_path.exists() else flat_path

                records.append({
                    "image_path": img_path,
                    "box": box,
                    "class_name": cls_name,
                    "uid": f"{csv_split}_{cls_name}_{Path(img_name).stem}",
                })

    class_names = sorted(class_set)

    # Filter out rows whose image doesn't actually exist on disk
    valid, missing = [], 0
    for r in records:
        if r["image_path"].exists():
            valid.append(r)
        else:
            missing += 1
    if missing:
        print(f"[warn] {missing:,} CSV rows point to missing image files — skipped")

    return valid, class_names


def load_swisscube(data_dir: Path):
    """
    SwissCube format — confirmed BOP (Benchmark for 6D Object Pose) layout:

        training/ validation/ testing/
          seq_XXXXXX/000000/
            rgb/                   000000.jpg, 000001.jpg, ...
            scene_gt_info.json     per-frame bbox, keyed by plain int
                                    string ("0","1",...), NOT zero-padded

    scene_gt_info.json: { "1": [ { "bbox_visib": [x,y,w,h], ... } ], ... }
    bbox_visib == [-1,-1,-1,-1] means object not visible in that frame —
    those rows get box=None (same downstream handling as "no box parsed").

    Returns:
        records     : list of {"image_path", "box": [x1,y1,x2,y2] or None, "class_name"}
        class_names : ["satellite"]
    """
    records = []
    image_exts = {".jpg", ".jpeg", ".png"}

    for split in ["training", "validation", "testing"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            candidate_dirs = [seq_dir] + [d for d in seq_dir.iterdir() if d.is_dir()]

            for cdir in candidate_dirs:
                rgb_dir = cdir / "rgb"
                gt_info_path = cdir / "scene_gt_info.json"
                if not rgb_dir.exists():
                    continue

                gt_info = {}
                if gt_info_path.exists():
                    try:
                        with open(gt_info_path) as f:
                            gt_info = json.load(f)
                    except (json.JSONDecodeError, OSError) as e:
                        print(f"[warn] Could not parse {gt_info_path}: {e}")

                for img_path in sorted(rgb_dir.iterdir()):
                    if img_path.suffix.lower() not in image_exts:
                        continue

                    box = None
                    try:
                        frame_key = str(int(img_path.stem))
                    except ValueError:
                        frame_key = img_path.stem

                    entry_list = gt_info.get(frame_key)
                    if entry_list:
                        entry = entry_list[0]
                        bbox = entry.get("bbox_visib") or entry.get("bbox_obj")
                        if bbox and len(bbox) == 4 and list(bbox) != [-1, -1, -1, -1]:
                            x, y, w, h = bbox
                            if w > 0 and h > 0:
                                box = [x, y, x + w, y + h]

                    records.append({
                        "image_path": img_path,
                        "box": box,
                        "class_name": "satellite",
                        # Frame filenames restart at 000000 for EVERY sequence,
                        # so class_name+stem alone collides across sequences
                        # (confirmed bug — same filename appears in seq_000001,
                        # seq_000002, seq_000003 etc.). Sequence folder name
                        # makes this globally unique.
                        "uid": f"{split}_{seq_dir.name}_{img_path.stem}",
                    })

    return records, ["satellite"]


def detect_and_load(data_dir: Path, format_hint: str):
    if format_hint == "spark_csv":
        return load_spark_csv(data_dir)
    if format_hint == "swisscube":
        return load_swisscube(data_dir)

    # auto
    if (data_dir / "train.csv").exists() or (data_dir / "val.csv").exists():
        print("[detect] SPARK CSV format")
        return load_spark_csv(data_dir)
    if (data_dir / "swisscube_bbox.json").exists() or \
       any((data_dir / s).exists() for s in ("training", "validation", "testing")):
        print("[detect] SwissCube format")
        return load_swisscube(data_dir)

    print(f"[error] Could not detect dataset format under {data_dir}. "
         f"Pass --format spark_csv or --format swisscube explicitly.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(records, ratios, seed):
    """
    Splits records into train/val/test, stratified per class so every
    split gets a proportional share of each class — important here since
    SPARK's classes are unevenly sized (see docs/decisions.md).

    Any class with fewer than 3 examples gets everything routed to train
    (can't meaningfully split 1-2 examples three ways) and a warning is
    printed so it can be flagged in decisions.md.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class_name"]].append(r)

    train, val, test = [], [], []
    tiny_classes = []

    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        if n < 3:
            train.extend(items)
            tiny_classes.append((cls, n))
            continue

        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        # Ensure at least 1 goes to val and test where possible
        n_train = min(n_train, n - 2)
        n_val = max(1, min(n_val, n - n_train - 1))
        n_test = n - n_train - n_val

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    if tiny_classes:
        print(f"\n[warn] {len(tiny_classes)} class(es) had <3 examples and were "
             f"put entirely in train (can't split meaningfully):")
        for cls, n in tiny_classes:
            print(f"         '{cls}': {n} example(s)")
        print("       Flag this in docs/decisions.md — these classes won't be "
             "represented in validation metrics.\n")

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# YOLO conversion + file writing
# ---------------------------------------------------------------------------

def box_to_yolo_line(box, img_w, img_h, class_id):
    """Convert pixel [x1,y1,x2,y2] box to normalized YOLO 'cid cx cy w h' line."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # Clip to [0,1] in case of any off-by-a-pixel rounding at image edges
    cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_and_write(splits, class_to_id, out_dir: Path, use_link: bool, dry_run: bool):
    can_symlink = use_link and platform.system() != "Windows"
    if use_link and not can_symlink:
        print("[info] --link requested but running on Windows — falling back to copy "
             "(symlinks need admin rights there).")

    stats = defaultdict(lambda: defaultdict(int))  # split -> class -> count
    skipped_unreadable = 0
    skipped_no_box = 0

    for split_name, items in splits.items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

        for rec in items:
            src_path = rec["image_path"]
            cls_name = rec["class_name"]
            box = rec["box"]

            if box is None:
                skipped_no_box += 1
                continue

            try:
                with Image.open(src_path) as im:
                    img_w, img_h = im.size
            except Exception:
                skipped_unreadable += 1
                continue

            class_id = class_to_id[cls_name]
            stats[split_name][cls_name] += 1

            # Use each record's unique id for destination naming — plain
            # "class_name + stem" isn't safe on its own: SwissCube frame
            # filenames restart at 000000 for every sequence, so without
            # a sequence-aware uid, images from different sequences would
            # silently overwrite each other (caught via manual verification —
            # see notes in load_swisscube/load_spark_csv).
            dest_stem = rec.get("uid") or f"{cls_name}_{src_path.stem}"
            dest_img = img_dir / f"{dest_stem}{src_path.suffix}"
            dest_lbl = lbl_dir / f"{dest_stem}.txt"

            if dry_run:
                continue

            if can_symlink:
                if dest_img.exists() or dest_img.is_symlink():
                    dest_img.unlink()
                dest_img.symlink_to(src_path.resolve())
            else:
                shutil.copy2(src_path, dest_img)

            line = box_to_yolo_line(box, img_w, img_h, class_id)
            dest_lbl.write_text(line + "\n")

    return stats, skipped_unreadable, skipped_no_box


def write_data_yaml(out_dir: Path, class_names, dry_run: bool):
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for name in class_names:
        lines.append(f"  - {name}")

    content = "\n".join(lines) + "\n"
    if not dry_run:
        (out_dir / "data.yaml").write_text(content)
    return content


def print_summary(stats, class_names, skipped_unreadable, skipped_no_box):
    print("\n" + "=" * 60)
    print("  Split summary")
    print("=" * 60)

    header = f"{'Class':<20}" + "".join(f"{s:>10}" for s in ("train", "val", "test", "total"))
    print(header)
    print("-" * len(header))

    totals = defaultdict(int)
    for cls in class_names:
        row = [stats[split].get(cls, 0) for split in ("train", "val", "test")]
        total = sum(row)
        for split, v in zip(("train", "val", "test"), row):
            totals[split] += v
        print(f"{cls:<20}" + "".join(f"{v:>10,}" for v in row + [total]))

    print("-" * len(header))
    grand_total = sum(totals.values())
    print(f"{'TOTAL':<20}" + "".join(f"{totals[s]:>10,}" for s in ('train','val','test')) +
         f"{grand_total:>10,}")

    if skipped_no_box:
        print(f"\n[warn] {skipped_no_box:,} images skipped — no bounding box parsed.")
    if skipped_unreadable:
        print(f"[warn] {skipped_unreadable:,} images skipped — file unreadable/corrupt.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert SPARK/SwissCube dataset to YOLO format with a train/val/test split."
    )
    parser.add_argument("--data", required=True, help="Path to dataset root")
    parser.add_argument("--out", required=True, help="Output directory for YOLO-format data")
    parser.add_argument("--format", default="auto",
                        choices=["auto", "spark_csv", "swisscube"])
    parser.add_argument("--split", type=float, nargs=3, default=[0.80, 0.15, 0.05],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Split ratios, must sum to 1.0 (default: 0.80 0.15 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link", action="store_true",
                        help="Symlink images instead of copying (Linux/Mac only; "
                             "falls back to copy on Windows)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing any files")
    args = parser.parse_args()

    if abs(sum(args.split) - 1.0) > 1e-6:
        print(f"[error] --split ratios must sum to 1.0, got {args.split} (sum={sum(args.split)})")
        sys.exit(1)

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    if not data_dir.exists():
        print(f"[error] Dataset directory not found: {data_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Dataset:  {data_dir}")
    print(f"  Output:   {out_dir}")
    print(f"  Split:    train={args.split[0]:.0%} val={args.split[1]:.0%} test={args.split[2]:.0%}")
    print(f"  Mode:     {'DRY RUN (no files written)' if args.dry_run else 'copy' if not args.link else 'link'}")
    print(f"{'='*60}\n")

    print("[1/4] Loading dataset annotations …")
    records, class_names = detect_and_load(data_dir, args.format)
    if not records:
        print("[error] No records loaded. Check --data path and --format.")
        sys.exit(1)
    print(f"       {len(records):,} annotated rows  |  {len(class_names)} classes: {class_names}")

    class_to_id = {name: i for i, name in enumerate(class_names)}

    print("\n[2/4] Stratified split per class …")
    splits = stratified_split(records, args.split, args.seed)
    print(f"       train={len(splits['train']):,}  val={len(splits['val']):,}  test={len(splits['test']):,}")

    print("\n[3/4] Converting to YOLO format + writing files …")
    stats, skipped_unreadable, skipped_no_box = convert_and_write(
        splits, class_to_id, out_dir, args.link, args.dry_run
    )

    print("\n[4/4] Writing data.yaml …")
    yaml_content = write_data_yaml(out_dir, class_names, args.dry_run)
    if not args.dry_run:
        print(f"       Saved → {out_dir / 'data.yaml'}")
    else:
        print("       (dry run — not written)\n" + "\n".join(f"       {l}" for l in yaml_content.splitlines()))

    print_summary(stats, class_names, skipped_unreadable, skipped_no_box)

    if not args.dry_run:
        print(f"Done. YOLO-format dataset ready at: {out_dir.resolve()}")
        print(f"Pass this to training as: --data {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()