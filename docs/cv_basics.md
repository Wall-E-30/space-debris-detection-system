# Computer Vision & Object Detection Fundamentals

A concise reference guide summarizing core object detection concepts for Space Situational Awareness (SSA) and YOLO baseline training.

---

## 1. Anchor Boxes & Anchor-Free Detection
- **Concept:** Traditional detectors (like YOLOv3/v4 or Faster R-CNN) use predefined bounding box shapes called **anchor boxes** placed across feature map grid cells as initial templates for objects.
- **YOLOv8 Approach:** YOLOv8 uses an **anchor-free (center-based)** detection head. Instead of predicting offsets from fixed anchor box templates, it directly predicts the distance from target grid points to the 4 boundaries (left, top, right, bottom) of an object. This simplifies configuration, improves small object detection, and reduces hyperparameters.

---

## 2. Intersection over Union (IoU)
- **Definition:** IoU measures the spatial overlap between a predicted bounding box ($B_p$) and a ground truth bounding box ($B_{gt}$):
  $$\text{IoU} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$
- **Interpretation:**
  - $\text{IoU} = 1.0$: Perfect overlap.
  - $\text{IoU} \ge 0.5$: Commonly considered a positive detection (True Positive) in object detection benchmarks.
  - $\text{IoU} = 0.0$: No overlap.

---

## 3. Non-Maximum Suppression (NMS)
- **Problem:** Object detectors output multiple candidate bounding boxes surrounding the same object across nearby grid cells.
- **Algorithm:**
  1. Filter out all bounding boxes below a minimum confidence threshold ($\text{conf} < \text{conf\_thresh}$).
  2. Sort remaining candidate boxes by confidence score in descending order.
  3. Select the highest confidence box and suppress (discard) any other candidate box that has an $\text{IoU} > \text{nms\_thresh}$ (typically 0.45 – 0.70) with the selected box.
  4. Repeat until all candidate boxes are either retained or suppressed.

---

## 4. Evaluation Metrics: mAP@0.5 vs mAP@0.5:0.95

### Precision & Recall
- **Precision:** Fraction of detected objects that are correct predictions:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Recall:** Fraction of actual ground truth objects that were successfully detected:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### Precision-Recall Curve & AP
- For a given class, plotting Precision vs Recall across varying confidence thresholds produces the Precision-Recall (PR) curve. **Average Precision (AP)** is the area under this PR curve.

### mAP Metrics
- **mAP@0.5:** Mean Average Precision calculated at a fixed IoU threshold of 0.50. This evaluates whether objects are correctly located and classified, without being overly strict on bounding box alignment.
- **mAP@0.5:0.95 (mAP@50-95):** The primary COCO metric. It averages the mAP computed at 10 distinct IoU thresholds from 0.50 to 0.95 in steps of 0.05 ($0.50, 0.55, \dots, 0.95$). This rewards high localization accuracy (tight bounding boxes).

---

## 5. Space Debris Detection & SSA Key Takeaways

1. **High Contrast & Scale Variability:** Spacecraft and debris range from large solar panel arrays to tiny tumbling fragments ($<10\text{px}$). Anchor-free detectors like YOLOv8 excel at multi-scale detection.
2. **Synthetic-to-Real Domain Gap:** Space optics suffer from harsh solar illumination, specular reflections, and deep black space backgrounds. Augmentation (contrast jitter, Gaussian blur, sensor noise) is essential for generalization.
3. **Class Imbalance:** Debris fragments far outnumber whole satellites in operational scenarios. Loss functions like Focal Loss / CIoU loss are key to preventing dominant background classes from over-fitting.
