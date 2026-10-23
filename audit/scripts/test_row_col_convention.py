"""
audit/scripts/test_row_col_convention.py
---
Tests whether [R_min, C_min, R_max, C_max] is [y_min, x_min, y_max, x_max] or [x_min, y_min, x_max, y_max]
by checking pixel brightness/foreground overlap of the spacecraft object on Cheops_image_00001_img.jpg.
"""

import cv2
import numpy as np
from pathlib import Path

def main():
    img_path = Path("data/processed/spark/images/train/Cheops_image_00001_img.jpg")
    out_dir = Path("audit/label_overlays")
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        print("Image not found.")
        return

    # From train.csv row 2: Cheops, Bounding box: "(397, 253, 653, 486)"
    r_min, c_min, r_max, c_max = 397, 253, 653, 486

    # Convention A: R=y, C=x (The SPARK specification claim)
    # y1=397, y2=653, x1=253, x2=486
    boxA_crop = img[r_min:r_max, c_min:c_max]
    boxA_energy = np.mean(boxA_crop)

    # Convention B: R=x, C=y (The inverted/mistaken Phase 1 label convention)
    # y1=253, y2=486, x1=397, x2=653
    boxB_crop = img[c_min:c_max, r_min:r_max]
    boxB_energy = np.mean(boxB_crop)

    print(f"Convention A (R=y, C=x) Mean Brightness: {boxA_energy:.2f}")
    print(f"Convention B (R=x, C=y) Mean Brightness: {boxB_energy:.2f}")

    # Draw both on image
    # Green: Convention A [R=y, C=x]
    cv2.rectangle(img, (c_min, r_min), (c_max, r_max), (0, 255, 0), 3)
    cv2.putText(img, "R=y, C=x (Correct SPARK)", (c_min, r_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Red: Convention B [R=x, C=y]
    cv2.rectangle(img, (r_min, c_min), (r_max, c_max), (0, 0, 255), 2)
    cv2.putText(img, "R=x, C=y (Inverted Phase 1)", (r_min, c_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out_file = out_dir / "convention_test_Cheops_00001.jpg"
    cv2.imwrite(str(out_file), img)
    print(f"Saved visual overlay to {out_file}")

if __name__ == "__main__":
    main()
