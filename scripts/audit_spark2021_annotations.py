"""
scripts/audit_spark2021_annotations.py
---
SPARK2021 Dataset CSV Label & Debris Verification Audit Utility.

Examines SPARK2021 raw annotation CSVs (train_labels.csv, validate_labels.csv, and test_labels.csv)
to compute class frequencies across all 11 classes, verify bounding box formats,
and definitively confirm presence of space debris annotations.

Supersedes Phase 1 Decision DEC-04 / Branch B (spacecraft-only) in favor of Phase 2
multi-class satellite + space debris detection.
"""

import argparse
import ast
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


CLASS_NAMES: List[str] = [
    "AcrimSat",
    "Aquarius",
    "Aura",
    "Calipso",
    "Cloudsat",
    "CubeSat",
    "Debris",
    "Jason",
    "Sentinel-6",
    "Terra",
    "TRMM",
]

CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS_NAME: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

DEBRIS_KEYWORDS = ["debris", "fragment", "rocket_body", "booster", "stage"]


def resolve_class_name(raw_cls: Any) -> str:
    """Normalize raw class value (string name or integer string/id) to official SPARK2021 class name."""
    s = str(raw_cls).strip()
    if s.isdigit():
        idx = int(s)
        if idx in ID_TO_CLASS_NAME:
            return ID_TO_CLASS_NAME[idx]
    # Check direct name match (case-insensitive fallback)
    for name in CLASS_NAMES:
        if name.lower() == s.lower():
            return name
    return s


def parse_bbox_string(bbox_str: str) -> Optional[List[float]]:
    """Safely parse bounding box string like '[296, 536, 353, 581]' into [r_min, c_min, r_max, c_max]."""
    bbox_str = bbox_str.strip()
    try:
        val = json.loads(bbox_str)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            return [float(x) for x in val]
    except Exception:
        pass

    try:
        val = ast.literal_eval(bbox_str)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            return [float(x) for x in val]
    except Exception:
        pass

    # Fallback comma-split
    clean = bbox_str.strip("[]() ")
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    if len(parts) == 4:
        try:
            return [float(x) for x in parts]
        except ValueError:
            return None
    return None


def audit_spark2021_csv(csv_path: Path) -> Dict[str, Any]:
    """
    Parse a SPARK2021 annotation CSV and compute distribution and bounding box stats.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"SPARK2021 CSV not found: {csv_path}")

    total_rows = 0
    unique_images: Set[str] = set()
    class_counter: Counter = Counter()
    invalid_bboxes = 0
    box_area_stats: List[float] = []
    illuminations: List[float] = []

    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            raw_cls = row.get("class") or row.get("Class") or ""
            cls_name = resolve_class_name(raw_cls)
            class_counter[cls_name] += 1

            img_name = (row.get("image") or row.get("Image name") or "").strip()
            if img_name:
                unique_images.add(f"{cls_name}/{img_name}")

            bbox_raw = row.get("bbox") or ""
            parsed_box = parse_bbox_string(bbox_raw)
            if parsed_box:
                r_min, c_min, r_max, c_max = parsed_box
                w = c_max - c_min
                h = r_max - r_min
                if w <= 0 or h <= 0 or r_min < 0 or c_min < 0:
                    invalid_bboxes += 1
                else:
                    box_area_stats.append(w * h)
            else:
                invalid_bboxes += 1

            illum_val = row.get("illumination")
            if illum_val is not None and illum_val.strip():
                try:
                    illuminations.append(float(illum_val.strip()))
                except ValueError:
                    pass

    debris_instances = sum(
        count for cls, count in class_counter.items()
        if any(kw in cls.lower() for kw in DEBRIS_KEYWORDS)
    )

    avg_area = (sum(box_area_stats) / len(box_area_stats)) if box_area_stats else 0.0
    illum_summary = {}
    if illuminations:
        illum_summary = {
            "min": round(min(illuminations), 4),
            "max": round(max(illuminations), 4),
            "mean": round(sum(illuminations) / len(illuminations), 4),
        }

    return {
        "csv_path": str(csv_path.resolve()),
        "total_rows": total_rows,
        "unique_images": len(unique_images),
        "class_counts": dict(class_counter),
        "debris_instances": debris_instances,
        "satellite_instances": total_rows - debris_instances,
        "invalid_bboxes": invalid_bboxes,
        "mean_bbox_area_px": round(avg_area, 2),
        "illumination_stats": illum_summary,
    }


def run_audit(
    train_csv_path: Optional[str] = None,
    val_csv_path: Optional[str] = None,
    test_csv_path: Optional[str] = None,
    spark_dir: Optional[str] = None,
    out_path: str = "docs/spark2021_class_audit.json",
) -> Dict[str, Any]:
    """
    Run full SPARK2021 dataset audit across train, val, and test splits.
    """
    print("\n" + "=" * 80)
    print("  SPARK2021 DATASET ANNOTATION AUDIT (PHASE 2 DEBRIS VERIFICATION)")
    print("=" * 80)

    # Locate CSVs
    if spark_dir and Path(spark_dir).exists():
        root_dir = Path(spark_dir)
    elif Path("data/raw/SPARK-2021").exists():
        root_dir = Path("data/raw/SPARK-2021")
    elif Path("data/raw/SPARK2021").exists():
        root_dir = Path("data/raw/SPARK2021")
    else:
        root_dir = Path(spark_dir or "data/raw/SPARK-2021")

    def find_csv(explicit: Optional[str], default_candidates: List[str]) -> Optional[Path]:
        if explicit and Path(explicit).exists():
            return Path(explicit)
        for cand in default_candidates:
            p = root_dir / cand
            if p.exists():
                return p
            p_global = Path(cand)
            if p_global.exists():
                return p_global
        return None

    train_p = find_csv(train_csv_path, ["train_labels.csv", "train.csv"])
    val_p = find_csv(val_csv_path, ["validate_labels.csv", "val_labels.csv", "val.csv"])
    test_p = find_csv(test_csv_path, ["test_labels.csv", "test.csv"])

    print(f"  Target Root: {root_dir.resolve()}")
    print(f"  Train CSV:   {train_p if train_p else '[NOT FOUND]'}")
    print(f"  Val CSV:     {val_p if val_p else '[NOT FOUND]'}")
    print(f"  Test CSV:    {test_p if test_p else '[NOT FOUND]'}")
    print("-" * 80 + "\n")

    splits_data: Dict[str, Dict[str, Any]] = {}
    total_rows = 0
    total_unique_images = 0
    all_class_counts: Counter = Counter()

    for split_name, csv_p in [("train", train_p), ("val", val_p), ("test", test_p)]:
        if csv_p and csv_p.exists():
            print(f"[*] Auditing {split_name} split from {csv_p.name}...")
            split_stat = audit_spark2021_csv(csv_p)
            splits_data[split_name] = split_stat
            total_rows += split_stat["total_rows"]
            total_unique_images += split_stat["unique_images"]
            for cls, cnt in split_stat["class_counts"].items():
                all_class_counts[cls] += cnt

    if not splits_data:
        raise FileNotFoundError(
            f"No SPARK2021 annotation CSVs found in {root_dir}. "
            "Please check --spark-dir or extract train_labels.csv, validate_labels.csv, test_labels.csv."
        )

    # Class list ordered by official 11 classes + any unexpected classes
    ordered_classes = [c for c in CLASS_NAMES if c in all_class_counts]
    for c in sorted(all_class_counts.keys()):
        if c not in ordered_classes:
            ordered_classes.append(c)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Class Name':<16} | {'Train':<10} | {'Val':<10} | {'Test':<10} | {'Total':<10} | {'Type'}")
    print("-" * 80)

    train_counts = splits_data.get("train", {}).get("class_counts", {})
    val_counts = splits_data.get("val", {}).get("class_counts", {})
    test_counts = splits_data.get("test", {}).get("class_counts", {})

    total_debris_count = 0
    total_sat_count = 0

    for cls in ordered_classes:
        t_cnt = train_counts.get(cls, 0)
        v_cnt = val_counts.get(cls, 0)
        te_cnt = test_counts.get(cls, 0)
        tot_cnt = t_cnt + v_cnt + te_cnt
        is_debris = any(kw in cls.lower() for kw in DEBRIS_KEYWORDS)
        cls_type = "DEBRIS" if is_debris else "Satellite"
        if is_debris:
            total_debris_count += tot_cnt
        else:
            total_sat_count += tot_cnt
        print(f"{cls:<16} | {t_cnt:>10} | {v_cnt:>10} | {te_cnt:>10} | {tot_cnt:>10} | {cls_type}")

    print("-" * 80)
    print(
        f"{'TOTALS':<16} | "
        f"{splits_data.get('train', {}).get('total_rows', 0):>10} | "
        f"{splits_data.get('val', {}).get('total_rows', 0):>10} | "
        f"{splits_data.get('test', {}).get('total_rows', 0):>10} | "
        f"{total_rows:>10} |"
    )
    print("=" * 80 + "\n")

    debris_present = total_debris_count > 0
    debris_pct = (total_debris_count / total_rows * 100.0) if total_rows > 0 else 0.0

    print("  VERIFICATION FINDINGS:")
    print(f"  - Total Images / Annotations:  {total_rows}")
    print(f"  - Space Debris Annotations:    {total_debris_count} ({debris_pct:.1f}% of total)")
    print(f"  - Spacecraft Annotations:      {total_sat_count} ({100.0 - debris_pct:.1f}% of total)")
    print(f"  - Debris Class Detected:       {'[YES - VERIFIED]' if debris_present else '[NO - WARNING]'}")
    if debris_present:
        print("  - Strategic Status:            DEC-04 (Branch B) SUPERSEDED. Full Debris Detection Unlocked.")
    else:
        print("  - Strategic Status:            WARNING: No debris class found.")
    print("=" * 80 + "\n")

    audit_results: Dict[str, Any] = {
        "dataset_name": "SPARK2021",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_directory": str(root_dir.resolve()),
        "debris_present": debris_present,
        "classes_detected": ordered_classes,
        "total_classes": len(ordered_classes),
        "total_rows": total_rows,
        "total_unique_images": total_unique_images,
        "debris_instances_total": total_debris_count,
        "debris_percentage": round(debris_pct, 2),
        "satellite_instances_total": total_sat_count,
        "splits": splits_data,
        "class_counts_overall": dict(all_class_counts),
        "strategic_decision": (
            "Branch A (Debris confirmed present in SPARK2021: 11-class Spacecraft + Debris system)"
            if debris_present else
            "Branch B (Debris absent - fallback)"
        ),
    }

    # Save to out_path
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"[OK] Audit report generated: {out_file.resolve()}")

    # Persistent archiving in outputs/audits/
    archive_dir = Path("outputs") / "audits"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_file = archive_dir / f"audit_spark2021_{timestamp_str}.json"
    latest_file = archive_dir / "latest_spark2021.json"

    with open(timestamped_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"[OK] Audit archived to:       {timestamped_file.resolve()}")
    print(f"[OK] Latest audit updated:    {latest_file.resolve()}\n")

    return audit_results


def main():
    parser = argparse.ArgumentParser(
        description="Audit SPARK2021 Dataset Annotation CSVs for Class Distribution & Debris Labels"
    )
    parser.add_argument(
        "--spark-dir",
        "--data",
        dest="spark_dir",
        type=str,
        default="data/raw/SPARK2021",
        help="Root directory containing SPARK2021 dataset (default: data/raw/SPARK2021)",
    )
    parser.add_argument(
        "--train-csv",
        dest="train_csv",
        type=str,
        default=None,
        help="Path to SPARK2021 train_labels.csv",
    )
    parser.add_argument(
        "--val-csv",
        "--validate-csv",
        dest="val_csv",
        type=str,
        default=None,
        help="Path to SPARK2021 validate_labels.csv",
    )
    parser.add_argument(
        "--test-csv",
        dest="test_csv",
        type=str,
        default=None,
        help="Path to SPARK2021 test_labels.csv",
    )
    parser.add_argument(
        "--out",
        "--out-json",
        dest="out",
        type=str,
        default="docs/spark2021_class_audit.json",
        help="Destination JSON report path (default: docs/spark2021_class_audit.json)",
    )
    args = parser.parse_args()

    run_audit(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        test_csv_path=args.test_csv,
        spark_dir=args.spark_dir,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
