"""
audit/scripts/verify_bbox_conversion.py
---
Verifies the SPARK bounding box row/column convention:
SPARK format is [R_min, C_min, R_max, C_max] -> [y_min, x_min, y_max, x_max].
Draws converted bounding boxes onto sample images from data/raw/spark/images
and saves overlay images to audit/label_overlays/ to visually confirm fit.
"""

import ast
import csv
from pathlib import Path
import cv2
import numpy as np

def main():
    csv_path = Path("data/raw/spark/train.csv")
    img_dir = Path("data/raw/spark/images")
    out_dir = Path("audit/label_overlays")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or not img_dir.exists():
        print("Dataset files not found for bbox test.")
        return

    # Sample images per class from train.csv
    samples_per_class = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row["Class"]
            if cls not in samples_per_class:
                samples_per_class[cls] = []
            if len(samples_per_class[cls]) < 2:
                samples_per_class[cls].append(row)

    drawn_count = 0
    overlay_paths = []

    for cls, rows in samples_per_class.items():
        for row in rows:
            img_name = row["Image name"]
            img_path = img_dir / cls / img_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Parse [R_min, C_min, R_max, C_max]
            bbox_raw = row["Bounding box"].strip()
            bbox = ast.literal_eval(bbox_raw)
            r_min, c_min, r_max, c_max = bbox

            # Row is Y, Col is X:
            # Correct conversion:
            y1, x1, y2, x2 = int(r_min), int(c_min), int(r_max), int(c_max)

            # Draw green box for [R=y, C=x]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{cls} [y1:{y1}, x1:{x1}]",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Also draw inverted red box [R=x, C=y] to show what failure would look like
            y1_bad, x1_bad, y2_bad, x2_bad = int(c_min), int(r_min), int(c_max), int(r_max)
            cv2.rectangle(img, (x1_bad, y1_bad), (x2_bad, y2_bad), (0, 0, 255), 1)

            out_file = out_dir / f"{cls}_{img_name}"
            cv2.imwrite(str(out_file), img)
            overlay_paths.append(str(out_file))
            drawn_count += 1

    print(f"Generated {drawn_count} overlay images in {out_dir}.")
    print("Green box = Correct [R=y, C=x] convention.")
    print("Red box = Inverted [R=x, C=y] failure comparison.")

if __name__ == "__main__":
    main()
