# Project Decision Log

This document records major technical, architectural, and deployment decisions for the Space Debris Detection System.

---

## 1. Edge Deployment Framework (TFLite vs ONNX vs NCNN)

**TFLite** and **ONNX Runtime** are both inference runtimes — you export your trained YOLOv8n weights into their format and run them locally. **Edge Impulse** is a full MLOps platform (train + optimize + deploy) built primarily around its own EON compiler for microcontroller-class TinyML models (FOMO, small CNNs) rather than full YOLO detection heads.

| | TFLite | ONNX Runtime | Edge Impulse |
|---|---|---|---|
| **What it is** | Google's mobile/embedded inference runtime | Cross-platform inference runtime | End-to-end MLOps platform (train + deploy) |
| **YOLOv8 export support** | Native `.tflite` export via Ultralytics | Native `.onnx` export via Ultralytics | Requires "Bring Your Own Model" workaround |
| **RPi 4 CPU speed** | Finicky without manual multi-threading/XNNPACK configuration | Comparable or better than baseline TFLite on ARM CPU | Not designed for full YOLO detector backbones |
| **Ultralytics Benchmark** | Included in standard export list | Included as standard portable format | Not part of Ultralytics' benchmarked export list |
| **Ecosystem fit** | Good — well-documented, offline | Good — cross-platform interchange format | Poor fit for YOLOv8n detection on RPi4 |
| **License** | Apache 2.0 | MIT | Free tier + paid tiers |

### Recommendation
**Export to NCNN as primary, keep ONNX as portable fallback, drop Edge Impulse for this deployment.**

- **NCNN:** Primary deployment runtime for Raspberry Pi 4 demo (`model.export(format="ncnn")`). Achieves up to ~5x speedup over standard ONNX on ARM CPU.
- **ONNX:** Interchange format kept as fallback for cross-platform hardware validation.
- **Edge Impulse:** Excluded for YOLOv8n detector backbone (suitable only if adding TinyML pre-filters later).

---

## 2. Baseline Model Architecture & Training Optimization

### Decisions
1. **Model Selection:** `YOLOv8n` (Nano) selected as baseline due to lightweight parameter footprint (~3.0M parameters, 8.2 GFLOPs) ideal for real-time edge execution.
2. **GPU Acceleration & Dataloader Optimization:**
   - Default CPU training (~34 hrs/5 epochs) migrated to **NVIDIA RTX 2050 GPU (CUDA)**.
   - Enabled dataset RAM caching (`cache=True`) to eliminate disk I/O bottleneck (`Slow image access detected`).
   - Mixed precision (`amp=True`) and 4 dataloader workers (`workers=4`) enabled.
   - Reduced 5-epoch training time to **~8–10 minutes**.

---

## 3. Week 1 Smoke Test Results Summary

- **Run ID:** `experiments/week1_smoke_test`
- **Dataset:** SwissCube (17,289 train images, 3,242 val images)
- **Metrics Achieved (5 Epochs):**
  - **mAP@0.5:** `0.9903`
  - **mAP@0.5:0.95:** `0.9211`
  - **Precision:** `0.974`
  - **Recall:** `0.962`
- **Checkpoints Saved:** `experiments/week1_smoke_test/weights/best.pt` and `last.pt`.

---

## 4. Object Taxonomy & Dataset Strategy

*(Detailed taxonomy analysis and Superclass mapping log available in [`data/decisions.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/data/decisions.md))*

- **Target Classes:** `satellite`, `debris`, `rocket_body`.
- **v1 Taxonomy Strategy:** Collapsed multi-spacecraft classes into `satellite` superclass; `debris` mapped 1:1. Deferred `rocket_body` synthesis to v2/stretch goals.
- **Class Imbalance Mitigation:** Carried over ~10x satellite to debris ratio; focal loss / class-weighted loss scheduled for Week 2-3 training.

---

## Related References
- [CV Basics & Object Detection Fundamentals](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/cv_basics.md)
- [Dataset Shortlist & Exploration Notes](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/data/README.md)
- [Object Classes & Taxonomy Decisions](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/data/decisions.md)