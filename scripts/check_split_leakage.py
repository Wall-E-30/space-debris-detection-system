"""
scripts/check_split_leakage.py
---
Audit for near-duplicate frame leakage between train and test splits.
Uses 64-bit difference hashing (dHash) via standard PIL (zero external dependencies).
A Hamming distance <= 3 indicates near-duplicate orbital frames (sequence leakage).
"""

import argparse
import datetime
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image


def compute_dhash(img_path: Path, hash_size: int = 8) -> int:
    """
    Computes 64-bit difference hash (dHash) for an image.
    Robust to minor lighting shifts and compression artifacts.
    """
    with Image.open(img_path) as im:
        # Resize to (width + 1, height) and convert to grayscale
        resized = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(resized.getdata())

    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            diff.append(pixel_left > pixel_right)

    # Convert boolean array to 64-bit integer
    decimal_val = 0
    for idx, bit in enumerate(diff):
        if bit:
            decimal_val |= 1 << idx
    return decimal_val


def hamming_distance(h1: int, h2: int) -> int:
    """Calculate the Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


from concurrent.futures import ThreadPoolExecutor


def audit_splits(
    dataset_dir: Path,
    samples_per_class: int = 50,
    train_samples: int = 5000,
    threshold: int = 3,
    threads: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    random.seed(seed)
    train_dir = dataset_dir / "images" / "train"
    test_dir = dataset_dir / "images" / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Images directories not found under {dataset_dir}. Expected images/train and images/test."
        )

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_train = [p for p in train_dir.iterdir() if p.suffix.lower() in image_exts]
    all_test = [p for p in test_dir.iterdir() if p.suffix.lower() in image_exts]

    # Sample train images if requested to keep audit under 30-45 seconds
    if 0 < train_samples < len(all_train):
        train_pool = random.sample(all_train, train_samples)
    else:
        train_pool = all_train

    sample_size = min(len(all_test), samples_per_class * 11)
    sampled_test = random.sample(all_test, sample_size)

    print("\n" + "=" * 70)
    print("  SPARK-2021 TRAIN/TEST SPLIT LEAKAGE & DUPLICATE AUDIT")
    print(f"  Dataset Directory:    {dataset_dir}")
    print(f"  Total Train Pool:     {len(all_train):,} (Auditing sample of {len(train_pool):,})")
    print(f"  Total Test Pool:      {len(all_test):,} (Auditing sample of {len(sampled_test):,})")
    print(f"  Parallel Workers:     {threads} threads")
    print(f"  Hamming Threshold:    <= {threshold} bits (out of 64)")
    print("=" * 70 + "\n")

    def hash_single(path: Path) -> Tuple[str, int]:
        try:
            return path.name, compute_dhash(path)
        except Exception:
            return path.name, -1

    # Parallel hashing
    print(f"Computing perceptual dHashes for {len(train_pool):,} train images ({threads} threads)...")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        train_hashes = [item for item in executor.map(hash_single, train_pool) if item[1] != -1]
    print(f"[OK] Hashed {len(train_hashes):,} train images.")

    print(f"Computing perceptual dHashes for {len(sampled_test):,} test images ({threads} threads)...")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        test_hashes = [item for item in executor.map(hash_single, sampled_test) if item[1] != -1]
    print(f"[OK] Hashed {len(test_hashes):,} test images.\n")

    print(f"Comparing test hashes against train pool (searching for near-duplicates <= {threshold} bits)...")
    near_duplicates = []
    min_distances = []

    for idx, (test_name, th) in enumerate(test_hashes, 1):
        best_dist = 64
        best_match = None

        for train_name, trh in train_hashes:
            dist = hamming_distance(th, trh)
            if dist < best_dist:
                best_dist = dist
                best_match = train_name
                if best_dist == 0:
                    break

        min_distances.append(best_dist)
        if best_dist <= threshold:
            near_duplicates.append({
                "test_image": test_name,
                "train_match": best_match,
                "hamming_distance": best_dist,
            })

        if idx % 100 == 0 or idx == len(test_hashes):
            print(f"  [{idx}/{len(test_hashes)}] Checked... ({len(near_duplicates)} near-duplicates detected so far)")

    leakage_rate = (len(near_duplicates) / len(test_hashes)) * 100.0 if test_hashes else 0.0
    avg_min_distance = sum(min_distances) / len(min_distances) if min_distances else 0.0

    print("\n" + "=" * 70)
    print(f"  LEAKAGE AUDIT RESULTS:")
    print(f"  Audited Test Samples:   {len(test_hashes):,}")
    print(f"  Audited Train Samples:  {len(train_hashes):,}")
    print(f"  Near-Duplicates Found:  {len(near_duplicates)}")
    print(f"  Estimated Leakage Rate: {leakage_rate:.2f}%")
    print(f"  Mean Min Hamming Dist:  {avg_min_distance:.2f} bits (random image distance is ~32 bits)")
    print("=" * 70 + "\n")

    results = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "total_train_images": len(all_train),
        "total_test_images": len(all_test),
        "audited_train_samples": len(train_hashes),
        "audited_test_samples": len(test_hashes),
        "hamming_threshold": threshold,
        "near_duplicates_count": len(near_duplicates),
        "leakage_rate_pct": round(leakage_rate, 2),
        "avg_min_distance": round(avg_min_distance, 2),
        "near_duplicate_samples": near_duplicates[:20],
    }

    out_dir = Path("outputs") / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "split_leakage_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Saved audit results to {out_file.resolve()}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset for train/test split leakage.")
    parser.add_argument("--data-dir", type=str, default="data/processed/spark2021", help="Path to processed dataset directory")
    parser.add_argument("--samples-per-class", type=int, default=50, help="Test images sampled per class (default: 50 -> 550 test images)")
    parser.add_argument("--train-samples", type=int, default=5000, help="Train images sampled for pool (default: 5000, 0 for all 90k)")
    parser.add_argument("--threshold", type=int, default=3, help="Max Hamming distance to flag near-duplicate (default: 3)")
    parser.add_argument("--threads", type=int, default=8, help="Parallel worker threads (default: 8)")
    args = parser.parse_args()

    audit_splits(
        dataset_dir=Path(args.data_dir),
        samples_per_class=args.samples_per_class,
        train_samples=args.train_samples,
        threshold=args.threshold,
        threads=args.threads
    )
