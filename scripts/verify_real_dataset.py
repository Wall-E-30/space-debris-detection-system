"""
scripts/verify_real_dataset.py
---
Batch inference validator across the curated real-world space imagery dataset
(data/real_validation/) using the trained ONNX Runtime model.

Verifies:
1. Batch execution across all 25 operational space scenes without crashes.
2. Formats results to outputs/inference/real_validation_results.json.
3. Computes detection count, latency distribution, and class attribution.
"""

import json
import os
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np

from src.infer import init_onnx_session, load_class_names, run_onnx_inference


def main():
    repo_root = Path(__file__).resolve().parent.parent
    real_val_dir = repo_root / "data" / "real_validation"
    metadata_path = real_val_dir / "METADATA.json"
    weights_path = repo_root / "experiments" / "spark2021_yolov8n_50e" / "weights" / "best_int8.onnx"
    data_config = repo_root / "configs" / "spark2021.yaml"
    out_dir = repo_root / "outputs" / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at: {metadata_path}. Run scripts/build_real_validation_set.py first.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = load_class_names(str(data_config))
    print(f"\n[INFO] Loaded {len(class_names)} classes from {data_config.name}")
    print(f"[INFO] Initializing inference session with: {weights_path.name}")

    session = init_onnx_session(str(weights_path))

    results = []
    latencies = []

    print(f"[INFO] Running batch inference across {len(metadata['scenes'])} scenes...\n")

    for scene in metadata["scenes"]:
        img_file = real_val_dir / scene["name"]
        t0 = time.perf_counter()
        detections, latency_ms = run_onnx_inference(
            weights=str(weights_path),
            image_path=str(img_file),
            imgsz=640,
            conf=0.25,
            iou=0.45,
            session=session,
            class_names=class_names,
        )
        total_time = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        results.append({
            "id": scene["id"],
            "file_name": scene["name"],
            "target": scene["target"],
            "challenge": scene["challenge"],
            "detections_count": len(detections),
            "detections": detections,
            "inference_ms": round(latency_ms, 2),
            "total_roundtrip_ms": round(total_time, 2)
        })

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    total_det = sum(r["detections_count"] for r in results)

    output_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_evaluated": str(weights_path.name),
        "total_scenes": len(results),
        "scenes_with_detections": sum(1 for r in results if r["detections_count"] > 0),
        "total_objects_detected": total_det,
        "latency_metrics": {
            "mean_ms": round(mean_lat, 2),
            "median_ms": round(median_lat, 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "min_ms": round(float(np.min(latencies)), 2),
            "max_ms": round(float(np.max(latencies)), 2)
        },
        "detailed_results": results
    }

    out_json = out_dir / "real_validation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print("=" * 65)
    print("  REAL-WORLD VALIDATION BATCH INFERENCE SUMMARY")
    print("=" * 65)
    print(f"Total Scenes Evaluated:  {len(results)}")
    print(f"Total Detections:        {total_det}")
    print(f"Mean Inference Latency:  {mean_lat:.2f} ms")
    print(f"Median Latency:          {median_lat:.2f} ms")
    print(f"Detailed Results Saved:  {out_json}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
