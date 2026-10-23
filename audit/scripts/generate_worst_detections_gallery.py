"""
audit/scripts/generate_worst_detections_gallery.py
---
Runs best.pt on images in data/processed/spark/images/test/
Finds challenging, low-confidence, or multi-box detections,
draws bounding boxes with labels and confidences, and saves a gallery of 20 images
to audit/label_overlays/worst_detections/ for visual inspection.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    img_dir = Path("data/processed/spark/images/test")
    out_dir = Path("audit/label_overlays/worst_detections")
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_p = Path("experiments/spark2021_yolov8n_50e/weights/best.pt")
    if not weights_p.exists() or not img_dir.exists():
        print("Missing weights or test image directory.")
        return

    model = YOLO(str(weights_p))
    images = list(img_dir.glob("*.jpg"))[:100]

    collected = []

    for img_p in images:
        results = model.predict(source=str(img_p), imgsz=1024, conf=0.15, device="cpu", verbose=False)[0]
        boxes = results.boxes

        if len(boxes) == 0:
            # Missed detection (target present in image but missed at conf 0.15)
            collected.append((img_p, results, "MISS (0 detections)"))
        elif len(boxes) > 1:
            # Over-detection / false positive splitting
            collected.append((img_p, results, f"MULTI-DET ({len(boxes)} boxes)"))
        else:
            conf = float(boxes.conf[0])
            if conf < 0.70:
                # Low confidence detection
                collected.append((img_p, results, f"LOW-CONF ({conf:.2f})"))

        if len(collected) >= 20:
            break

    # If fewer than 20 anomalies found, fill with lowest confidence samples
    if len(collected) < 20:
        for img_p in images[len(collected):]:
            results = model.predict(source=str(img_p), imgsz=1024, conf=0.15, device="cpu", verbose=False)[0]
            collected.append((img_p, results, "SAMPLE"))
            if len(collected) >= 20:
                break

    for idx, (img_p, results, tag) in enumerate(collected[:20], 1):
        rendered = results.plot() # Ultralytics renders annotated boxes
        out_p = out_dir / f"sample_{idx:02d}_{img_p.stem}.jpg"
        cv2.imwrite(str(out_p), rendered)

    print(f"[OK] Saved 20 detection images to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
