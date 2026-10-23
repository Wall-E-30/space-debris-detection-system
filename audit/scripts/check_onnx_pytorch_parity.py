"""
audit/scripts/check_onnx_pytorch_parity.py
---
Compares PyTorch (best.pt) and ONNX (best.onnx) on 50 space images.
Measures:
- Box count agreement
- Box IoU agreement
- Class label agreement
- Confidence delta (PyTorch vs ONNX)
"""

import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    boxBArea = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def main():
    img_dir = Path("data/processed/spark/images/test")
    pt_path = Path("experiments/spark2021_yolov8n_50e/weights/best.pt")
    onnx_path = Path("experiments/spark2021_yolov8n_50e/weights/best.onnx")

    if not img_dir.exists():
        print(f"Error: {img_dir} not found.")
        return

    images = list(img_dir.glob("*.jpg"))[:50]
    print(f"Testing parity across {len(images)} images...")

    pt_model = YOLO(str(pt_path))
    onnx_model = YOLO(str(onnx_path), task="detect")

    total_pt_boxes = 0
    total_onnx_boxes = 0
    matched_boxes = 0
    ious = []
    conf_diffs = []
    class_agreements = []

    for img_p in images:
        # Run inference via Ultralytics predict pipeline for both models on CPU
        res_pt = pt_model.predict(source=str(img_p), imgsz=1024, conf=0.25, device='cpu', verbose=False)[0]
        res_onnx = onnx_model.predict(source=str(img_p), imgsz=1024, conf=0.25, device='cpu', verbose=False)[0]

        b_pt = res_pt.boxes.xyxy.cpu().numpy()
        c_pt = res_pt.boxes.cls.cpu().numpy()
        conf_pt = res_pt.boxes.conf.cpu().numpy()

        b_on = res_onnx.boxes.xyxy.cpu().numpy()
        c_on = res_onnx.boxes.cls.cpu().numpy()
        conf_on = res_onnx.boxes.conf.cpu().numpy()

        total_pt_boxes += len(b_pt)
        total_onnx_boxes += len(b_on)

        for i, box_p in enumerate(b_pt):
            best_iou = 0.0
            best_j = -1
            for j, box_o in enumerate(b_on):
                iou = box_iou(box_p, box_o)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= 0.5 and best_j >= 0:
                matched_boxes += 1
                ious.append(best_iou)
                conf_diffs.append(abs(conf_pt[i] - conf_on[best_j]))
                class_agreements.append(int(c_pt[i] == c_on[best_j]))

    parity_results = {
        "num_images_tested": len(images),
        "total_pytorch_detections": int(total_pt_boxes),
        "total_onnx_detections": int(total_onnx_boxes),
        "matched_detections_iou50": int(matched_boxes),
        "mean_iou_matched": round(float(np.mean(ious)), 4) if ious else None,
        "min_iou_matched": round(float(np.min(ious)), 4) if ious else None,
        "mean_conf_delta": round(float(np.mean(conf_diffs)), 6) if conf_diffs else None,
        "max_conf_delta": round(float(np.max(conf_diffs)), 6) if conf_diffs else None,
        "class_agreement_pct": round(float(np.mean(class_agreements)) * 100.0, 2) if class_agreements else None,
    }

    out_file = Path("audit/onnx_parity_check.json")
    with open(out_file, "w") as f:
        json.dump(parity_results, f, indent=2)

    print("\n=== ONNX vs PyTorch Parity Results ===")
    print(json.dumps(parity_results, indent=2))

if __name__ == "__main__":
    main()
