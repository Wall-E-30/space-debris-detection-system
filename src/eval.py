"""
src/eval.py
---
Standalone evaluation CLI and harness for YOLOv8 models.
Wraps Ultralytics model.val() to produce a standardized metrics.json artifact
containing mAP, precision, recall, speed, parameters, FLOPs, and disk size.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from ultralytics import YOLO


def evaluate_model(
    weights: str,
    data: str,
    split: str = "val",
    imgsz: int = 1024,
    batch: int = 8,
    device: Optional[str] = None,
    out_json: Optional[str] = "metrics.json",
) -> Dict[str, Any]:
    """
    Run independent validation on a model checkpoint and export standardized metrics.

    Args:
        weights: Path to weights file (.pt, .onnx, etc.)
        data: Path to dataset YAML configuration file
        split: Dataset split ('val', 'test', 'train')
        imgsz: Evaluation image resolution
        batch: Evaluation batch size
        device: Compute device ('0', 'cpu', None for auto)
        out_json: Path to output JSON file (if None, does not write to disk)

    Returns:
        Dictionary of standardized metrics conforming to the EVAL-02 schema.
    """
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")

    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {data_path}")

    # Auto-resolve device if not specified (default to cpu for safe execution)
    if device is None:
        device = "cpu"

    print("\n" + "=" * 65)
    print("  RUNNING STANDALONE EVALUATION HARNESS")
    print(f"  Weights: {weights_path}")
    print(f"  Data:    {data_path}")
    print(f"  Split:   {split}")
    print(f"  Imgsz:   {imgsz}")
    print(f"  Batch:   {batch}")
    print(f"  Device:  {device}")
    print("=" * 65 + "\n")

    model = YOLO(str(weights_path))

    # Run validation without writing training-like plots unless needed
    results = model.val(
        data=str(data_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=False,
        save_json=False,
        verbose=True,
    )

    box = results.box
    speed = results.speed

    # Extract per-class metrics
    per_class = {}
    class_indices = getattr(box, "ap_class_index", [])
    if class_indices is not None and len(class_indices) > 0:
        names = results.names
        for idx, cls_id in enumerate(class_indices):
            cls_name = names.get(cls_id, str(cls_id))
            p_val = float(box.p[idx]) if idx < len(box.p) else 0.0
            r_val = float(box.r[idx]) if idx < len(box.r) else 0.0
            ap50_val = float(box.ap50[idx]) if idx < len(box.ap50) else 0.0
            ap_val = float(box.ap[idx]) if idx < len(box.ap) else 0.0
            per_class[cls_name] = {
                "class_id": int(cls_id),
                "p": round(p_val, 4),
                "r": round(r_val, 4),
                "map50": round(ap50_val, 4),
                "map50_95": round(ap_val, 4),
            }

    metrics: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": str(weights_path.resolve()),
        "data_path": str(data_path.resolve()),
        "imgsz": int(imgsz),
        "split": str(split),
        "device": str(device),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "mp": round(float(box.mp), 4),
        "mr": round(float(box.mr), 4),
        "speed": {
            "preprocess_ms": round(float(speed.get("preprocess", 0.0)), 2),
            "inference_ms": round(float(speed.get("inference", 0.0)), 2),
            "postprocess_ms": round(float(speed.get("postprocess", 0.0)), 2),
            "total_ms": round(
                float(speed.get("preprocess", 0.0))
                + float(speed.get("inference", 0.0))
                + float(speed.get("postprocess", 0.0)),
                2,
            ),
        },
        "per_class": per_class,
        "size_mb": round(weights_path.stat().st_size / (1024 * 1024), 2),
        "params": None,
        "gflops": None,
    }

    # Extract parameters and FLOPs for PyTorch models
    if str(weights_path).endswith(".pt"):
        try:
            _, params, _, gflops = model.info(verbose=False)
            metrics["params"] = int(params)
            metrics["gflops"] = round(float(gflops), 2)
        except Exception as e:
            print(f"[warn] Could not extract model.info: {e}")

    # Extract confusion matrix array
    try:
        if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
            cm = results.confusion_matrix.matrix
            if cm is not None:
                metrics["confusion_matrix"] = cm.tolist()
                print(f"[OK] Extracted confusion matrix: {cm.shape[0]}x{cm.shape[1]}")
    except Exception as e:
        print(f"[warn] Could not extract confusion matrix: {e}")

    # Output to stdout summary
    print("\n" + "-" * 50)
    print("  STANDALONE EVALUATION SUMMARY")
    print("-" * 50)
    print(f"  mAP@0.5:      {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    print(f"  Precision:    {metrics['mp']:.4f}")
    print(f"  Recall:       {metrics['mr']:.4f}")
    print(f"  Latency:      {metrics['speed']['inference_ms']:.2f} ms")
    print(f"  Model Size:   {metrics['size_mb']:.2f} MB")
    if metrics["params"] is not None:
        print(f"  Parameters:   {metrics['params']:,}")
        print(f"  GFLOPs:       {metrics['gflops']}")
    print("-" * 50 + "\n")

    # -------------------------------------------------------------
    # Persistent Results Archiving (outputs/evaluations/)
    # -------------------------------------------------------------
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path("outputs") / "evaluations"
    archive_dir.mkdir(parents=True, exist_ok=True)

    timestamped_file = archive_dir / f"eval_{now_str}.json"
    latest_file = archive_dir / "latest.json"
    history_file = archive_dir / "history.jsonl"

    # 1. Save timestamped archive
    with open(timestamped_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2. Update latest.json
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 3. Append to history.jsonl ledger
    summary_entry = {
        "timestamp": metrics["timestamp"],
        "model": str(weights_path.name),
        "split": split,
        "imgsz": imgsz,
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "mp": metrics["mp"],
        "mr": metrics["mr"],
        "inference_ms": metrics["speed"]["inference_ms"],
        "size_mb": metrics["size_mb"],
        "archive_file": str(timestamped_file.as_posix()),
    }
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_entry) + "\n")

    print(f"[OK] Evaluation results archived to: {timestamped_file.resolve()}")
    print(f"[OK] Latest evaluation updated at:   {latest_file.resolve()}")

    # 4. Save to explicit out_json if requested
    if out_json and Path(out_json).resolve() != timestamped_file.resolve():
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[OK] Evaluation metrics also written to: {out_path.resolve()}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Evaluation Harness for Space Debris Detection Models"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pt or exported format)",
    )
    parser.add_argument(
        "--data",
        "--data-config",
        dest="data",
        type=str,
        default="configs/spark.yaml",
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Image resolution (default: 1024)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (default: 'cpu' for safe execution)",
    )
    parser.add_argument(
        "--out",
        "--out-json",
        "--output",
        dest="out",
        type=str,
        default="metrics.json",
        help="Destination path for output JSON (default: 'metrics.json')",
    )

    args = parser.parse_args()

    evaluate_model(
        weights=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        out_json=args.out,
    )


if __name__ == "__main__":
    main()
