"""
scripts/audit_spark_annotations.py
---
Day-1 SPARK Dataset CSV Label Audit Utility.
Examines full raw SPARK annotation CSVs (train.csv and val.csv) to compute
class frequencies and definitively verify whether space debris annotations exist.
Addresses Finding F1 / Decision D-01 & D-02 (DEC-04).
"""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


DEBRIS_KEYWORDS = ["debris", "fragment", "debris_small", "rocket_body", "booster", "stage"]


def audit_csv(csv_path: Path) -> Tuple[int, int, Counter, List[str]]:
    """
    Parse a SPARK annotation CSV and return statistics.

    Returns:
        total_rows: Total bounding box rows
        unique_images: Total unique image filenames
        class_counter: Counter mapping class name to row frequency
        debris_classes: List of classes matching debris keywords
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"SPARK CSV not found at: {csv_path}")

    total_rows = 0
    unique_images: Set[str] = set()
    class_counter: Counter = Counter()

    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            # Case-insensitive / alias column lookup
            img_name = (
                row.get("Image name") or row.get("image") or row.get("filename") or
                row.get("image_name") or row.get("Image") or row.get("img") or ""
            ).strip()
            if img_name:
                unique_images.add(img_name)

            cls_name = (
                row.get("Class") or row.get("class") or row.get("label") or
                row.get("class_name") or row.get("category") or row.get("name") or ""
            ).strip()
            if cls_name:
                class_counter[cls_name] += 1

    debris_classes = [
        cls for cls in class_counter
        if any(kw in cls.lower() for kw in DEBRIS_KEYWORDS)
    ]

    return total_rows, len(unique_images), class_counter, debris_classes


def run_audit(train_csv_path: str, val_csv_path: str, out_path: str) -> Dict[str, Any]:
    """
    Audit both train and validation CSVs from SPARK dataset and write structured JSON.
    """
    train_path = Path(train_csv_path)
    val_path = Path(val_csv_path)

    print("\n" + "=" * 70)
    print("  RUNNING DAY-1 SPARK DATASET ANNOTATION AUDIT (DEC-04 / F1)")
    print(f"  Train CSV: {train_path}")
    print(f"  Val CSV:   {val_path}")
    print("=" * 70 + "\n")

    train_rows, train_imgs, train_counts, train_debris = audit_csv(train_path)
    val_rows, val_imgs, val_counts, val_debris = audit_csv(val_path)

    all_classes = sorted(list(set(train_counts.keys()) | set(val_counts.keys())))
    combined_counts = {cls: train_counts.get(cls, 0) + val_counts.get(cls, 0) for cls in all_classes}
    all_debris_classes = sorted(list(set(train_debris) | set(val_debris)))

    total_debris_instances = sum(combined_counts[cls] for cls in all_debris_classes)
    total_satellite_instances = sum(
        combined_counts[cls] for cls in all_classes if cls not in all_debris_classes
    )

    decision_branch = "Branch A (Debris present in annotations)" if total_debris_instances > 0 else "Branch B (Debris absent in annotations — 10-class spacecraft detection)"

    audit_results: Dict[str, Any] = {
        "train_csv": str(train_path.resolve()),
        "val_csv": str(val_path.resolve()),
        "train_total_rows": train_rows,
        "train_unique_images": train_imgs,
        "val_total_rows": val_rows,
        "val_unique_images": val_imgs,
        "total_rows": train_rows + val_rows,
        "total_unique_images": train_imgs + val_imgs,
        "classes_detected": all_classes,
        "class_count_total": len(all_classes),
        "class_counts_train": dict(train_counts),
        "class_counts_val": dict(val_counts),
        "class_counts": combined_counts,
        "debris_classes_found": all_debris_classes,
        "debris_instances_count": total_debris_instances,
        "satellite_instances_count": total_satellite_instances,
        "decision_recommendation": decision_branch,
    }

    # Print summary table
    print(f"{'Class Name':<20} | {'Train Count':<12} | {'Val Count':<12} | {'Total Count':<12} | {'Is Debris?'}")
    print("-" * 72)
    for cls in all_classes:
        is_deb = "YES" if cls in all_debris_classes else "No"
        print(f"{cls:<20} | {train_counts.get(cls, 0):>11} | {val_counts.get(cls, 0):>11} | {combined_counts[cls]:>11} | {is_deb}")
    print("-" * 72)
    print(f"{'TOTALS':<20} | {train_rows:>11} | {val_rows:>11} | {train_rows + val_rows:>11} |")
    print("\n" + "=" * 70)
    print(f"  Total Debris Instances:     {total_debris_instances}")
    print(f"  Total Spacecraft Instances: {total_satellite_instances}")
    print(f"  Resolution Recommendation:  {decision_branch}")
    print("=" * 70 + "\n")

    # -------------------------------------------------------------
    # Persistent Audit Archiving (outputs/audits/)
    # -------------------------------------------------------------
    import datetime
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path("outputs") / "audits"
    archive_dir.mkdir(parents=True, exist_ok=True)

    timestamped_file = archive_dir / f"audit_{now_str}.json"
    latest_file = archive_dir / "latest.json"

    with open(timestamped_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"[OK] Audit archived to: {timestamped_file.resolve()}")
    print(f"[OK] Latest audit updated at: {latest_file.resolve()}")

    if out_path and Path(out_path).resolve() != timestamped_file.resolve():
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2)
        print(f"[OK] Audit findings also exported to: {out_file.resolve()}\n")

    return audit_results


def main():
    parser = argparse.ArgumentParser(
        description="Audit SPARK Dataset Annotation CSVs for Class Distribution & Debris Labels"
    )
    parser.add_argument(
        "--spark-dir",
        "--data",
        "--dir",
        dest="spark_dir",
        type=str,
        default=None,
        help="Root directory containing SPARK dataset (searches for train.csv and val.csv)",
    )
    parser.add_argument(
        "--train_csv",
        "--train-csv",
        dest="train_csv",
        type=str,
        default=None,
        help="Path to raw SPARK train.csv",
    )
    parser.add_argument(
        "--val_csv",
        "--val-csv",
        dest="val_csv",
        type=str,
        default=None,
        help="Path to raw SPARK val.csv",
    )
    parser.add_argument(
        "--out",
        "--out-json",
        dest="out",
        type=str,
        default="docs/spark_class_audit.json",
        help="Destination JSON report path",
    )
    args = parser.parse_args()

    train_csv = args.train_csv
    val_csv = args.val_csv

    if args.spark_dir:
        spark_dir = Path(args.spark_dir)
        if not spark_dir.exists():
            raise FileNotFoundError(f"Provided --spark-dir does not exist: {spark_dir}")

        train_matches = (
            list(spark_dir.glob("**/train_labels.csv")) +
            list(spark_dir.glob("**/train.csv")) +
            list(spark_dir.glob("*train*.csv"))
        )
        val_matches = (
            list(spark_dir.glob("**/validate_labels.csv")) +
            list(spark_dir.glob("**/val_labels.csv")) +
            list(spark_dir.glob("**/val.csv")) +
            list(spark_dir.glob("*val*.csv")) +
            list(spark_dir.glob("*validate*.csv"))
        )

        if not train_csv and train_matches:
            train_csv = str(train_matches[0])
        if not val_csv and val_matches:
            val_csv = str(val_matches[0])

    # Fallbacks if still not specified
    if not train_csv:
        for candidate in ["data/raw/spark/train.csv", "data/raw/SPARK/train.csv", "spark/train.csv"]:
            if Path(candidate).exists():
                train_csv = candidate
                break
        if not train_csv:
            train_csv = "data/raw/spark/train.csv"

    if not val_csv:
        for candidate in ["data/raw/spark/val.csv", "data/raw/SPARK/val.csv", "spark/val.csv"]:
            if Path(candidate).exists():
                val_csv = candidate
                break
        if not val_csv:
            val_csv = "data/raw/spark/val.csv"

    run_audit(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
