"""
scripts/extract_spark2021.py
---
Extract SPARK2021 dataset zip archives (train_rgb.zip, validate_rgb.zip, test_rgb.zip)
with progress reporting and path auto-detection.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive to dest_dir with progress tracking."""
    if not zip_path.exists():
        print(f"[!] Archive not found: {zip_path}")
        return

    print(f"\n[*] Extracting: {zip_path.name} ({zip_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"    Destination: {dest_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_files = len(members)
        print(f"    Total files to extract: {total_files}")

        # Check if already extracted
        first_few = [m.filename for m in members if not m.is_dir()][:5]
        already_extracted = all((dest_dir / f).exists() for f in first_few) if first_few else False
        if already_extracted:
            print(f"    [INFO] Files from {zip_path.name} appear to already be extracted. Overwriting/verifying...")

        extracted_count = 0
        last_reported = 0

        for member in members:
            zf.extract(member, dest_dir)
            extracted_count += 1

            if extracted_count - last_reported >= 5000 or extracted_count == total_files:
                pct = (extracted_count / total_files) * 100.0
                elapsed = time.time() - start_time
                print(f"    [{pct:5.1f}%] {extracted_count}/{total_files} files extracted ({elapsed:.1f}s elapsed)")
                last_reported = extracted_count

    elapsed_total = time.time() - start_time
    print(f"[OK] Finished extracting {zip_path.name} in {elapsed_total:.1f} seconds.\n")


def main():
    parser = argparse.ArgumentParser(description="Extract SPARK2021 RGB Archives")
    parser.add_argument(
        "--spark-dir",
        type=str,
        default=None,
        help="Path to SPARK2021 directory containing zip files (auto-detected by default)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "validate", "test"],
        default="all",
        help="Which split to extract (default: all)",
    )
    args = parser.parse_args()

    # Auto-detect directory
    candidate_dirs = [
        args.spark_dir,
        "data/raw/SPARK-2021",
        "data/raw/SPARK2021",
        "data/raw/spark2021",
    ]

    target_dir = None
    for cand in candidate_dirs:
        if cand and Path(cand).exists():
            target_dir = Path(cand).resolve()
            break

    if not target_dir:
        print("[!] Error: Could not find SPARK2021 directory. Please specify --spark-dir.")
        sys.exit(1)

    print("=" * 70)
    print("  SPARK2021 ARCHIVE EXTRACTOR")
    print(f"  Target Directory: {target_dir}")
    print("=" * 70)

    zips_to_extract = []
    if args.split in ("all", "train"):
        zips_to_extract.append(target_dir / "train_rgb.zip")
    if args.split in ("all", "validate"):
        zips_to_extract.append(target_dir / "validate_rgb.zip")
    if args.split in ("all", "test"):
        zips_to_extract.append(target_dir / "test_rgb.zip")

    for zp in zips_to_extract:
        extract_zip(zp, target_dir)

    print("=" * 70)
    print("  ALL REQUESTED ARCHIVES EXTRACTED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
