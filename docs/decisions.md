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

## 5. Photometric Augmentation Pipeline (Domain Gap Mitigation)

### Context & Design
Synthetic satellite imagery (SPARK, SwissCube) exhibits clean rendering artifacts (uniform lighting, zero sensor noise, perfect optical focus) that do not reflect real-world space domain telescope or star tracker conditions.

### Decisions
1. **Photometric-Only Transforms:** Implemented `src/data/augment.py` using composable photometric adjustments (brightness/contrast jitter, Gaussian blur, sensor read noise). Because pixel positions are unchanged, bounding box annotations remain 100% invariant and require no coordinate recalculations.
2. **Strict Split Isolation:** Augmentations are applied exclusively to training splits (`train/`); validation and test sets remain unaltered to guarantee uncorrupted metric evaluation.
3. **Cross-Team Handoff:** Verified pipeline and handed off to Person A with parameterized CLI and programmatic `SpaceImageAugmenter` class.

---

## 6. Stage 3 Orbit Propagation & Conjunction Distance Estimation

### Architecture & Mechanics
Integrated a real-time orbit propagation pipeline (`src/data/tle_pipeline.py`) downstream of YOLO object detection to calculate relative Euclidean distances between space objects in 3D space.

```
+-------------------+      Class Mapping       +----------------------+
|  YOLOv8 Detection | -----------------------> |  NORAD ID Resolution |
+-------------------+                          +----------------------+
                                                          |
                                                          v
+-------------------+      SGP4 Propagation    +----------------------+
|  CelesTrak TLEs   | -----------------------> |  3D Geocentric XYZ   |
|   (Live / GP)     |                          |   Coordinates (km)   |
+-------------------+                          +----------------------+
                                                          |
                                                          v
                                               +----------------------+
                                               |  Euclidean Distance  |
                                               |   & Conjunction Risk |
                                               +----------------------+
```

### Decisions
1. **Live Ephemeris Fetching:** Skyfield fetches live General Perturbations (GP) Two-Line Element Sets (TLEs) directly from CelesTrak (`GROUP=active&FORMAT=tle`), indexing into an in-memory NORAD catalog.
2. **NORAD ID Mapping & Historical Proxies:** Created a lookup table mapping detected classes (e.g., `Cheops`, `LisaPathfinder`, `Proba2`, `ObservationSat1`) to catalog IDs, utilizing active LEO proxies for deep space / historical assets to ensure valid SGP4 orbit propagation.
3. **Relative Distance Calculation:** Resolved geocentric position vectors $(X, Y, Z)$ in kilometers at UTC timestamps and calculated Euclidean separation:
   $$\text{Distance} = \|\mathbf{P}_{\text{satA}} - \mathbf{P}_{\text{satB}}\|_2$$

---

## 7. Edge Export Pipeline & Optimization

### Implementation
Developed `edge/export.py` to automate conversion of trained PyTorch weights (`best.pt`) into optimized edge formats:

1. **ONNX Export (`best.onnx`):**
   - Exported at FP32 with Opset 12 and ONNX graph simplification enabled (`simplify=True`).
   - Serves as portable cross-platform fallback.
2. **NCNN Export (`best_ncnn_model/`):**
   - Generates `.param` and `.bin` binaries optimized for ARM CPU instruction sets (NEON, fp16 math).
   - Serves as the primary deployment format for the Raspberry Pi 4 edge target.
3. **Automated Sanity Verification:**
   - Script performs automated forward-pass verification using `onnxruntime` with synthetic tensors to validate graph structure and output shapes before edge transfer.

---

## 8. Empirical Edge Benchmarking & Latency Analysis

### Benchmark Methodology (`edge/benchmark.py`)
Inference throughput and latency were empirically evaluated on local CPU (640x640 resolution, 5 warmup passes, 30-50 iterations) evaluating thread scaling (`threads=1`, `threads=4`) and process memory footprint via `psutil`.

### Empirical Results Summary

| Engine | Thread Config | Mean Latency | p95 Latency | Throughput (FPS) | Memory Footprint (RAM) |
|---|---|---|---|---|---|
| **ONNX Runtime** | `threads=1` | 223.20 ms | 385.64 ms | 4.5 FPS | ~0.0 MB RSS |
| **ONNX Runtime** | `threads=4` | 209.71 ms | 226.12 ms | 4.8 FPS | ~0.0 MB RSS |
| **NCNN** | `threads=4` | **165.21 ms** | **178.48 ms** | **6.1 FPS** | **~10.9 MB RSS** |

### Key Takeaways & Validation
- **NCNN Performance Lead:** NCNN with 4 threads achieved a **~21% latency reduction** (165.21 ms vs 209.71 ms) and **~27% higher FPS** (6.1 vs 4.8) compared to 4-thread ONNX Runtime.
- **Latency Stability:** NCNN delivered substantially tighter p95 variance (178.48 ms vs 226.12 ms in ONNX multi-thread and 385.64 ms in ONNX single-thread).
- **Architecture Validation:** Empirically confirms the Section 1 recommendation to establish NCNN as the primary edge deployment runtime for ARM CPU SBCs (RPi 4).

---

## DEC-04: SPARK Dataset Annotation Audit & Debris Taxonomy Resolution

**Date:** 2026-10-14  
**Author:** Data & Edge Lead (Role 2)  
**Status:** Decided & Locked (`one-way` reversibility)  
**Evidence Artifact:** `docs/spark_class_audit.json`

### Empirical Audit Findings
On Day 1 of Phase 1, `scripts/audit_spark_annotations.py` executed a complete row-by-row audit across all annotation rows in the raw SPARK dataset (`data/raw/spark/train.csv` and `data/raw/spark/val.csv`).

| Split | Total Image Files | Total Label Rows | Detected Classes | Debris Rows |
| :--- | :--- | :--- | :--- | :--- |
| **Train (`train.csv`)** | 60,000 | 60,000 | 10 classes (6,000 per class) | **0** |
| **Val (`val.csv`)** | 20,000 | 20,000 | 10 classes (2,000 per class) | **0** |
| **Total** | 80,000 | 80,000 | 10 classes (8,000 per class) | **0** |

**Detected Class Frequencies (Exactly 8,000 rows each across train+val):**
1. `Cheops` (8,000)
2. `LisaPathfinder` (8,000)
3. `ObservationSat1` (8,000)
4. `Proba2` (8,000)
5. `Proba3` (8,000)
6. `Proba3ocs` (8,000)
7. `Smart1` (8,000)
8. `Soho` (8,000)
9. `VenusExpress` (8,000)
10. `XMM Newton` (8,000)

### Decision & Project Alignment (Adoption of Branch B)
1. **Finding F1 Confirmed:** The raw SPARK dataset contains **zero annotated debris fragments**. It is strictly a 10-class spacecraft dataset featuring 10 operational satellites and deep-space probes under varied space lighting conditions.
2. **Adoption of Branch B:**
   - **Primary Quantitative Model (Phases 2–7):** Formally defined as **10-class spacecraft detection and classification** (`yolov8n` trained on balanced Subset B at 640px). All quantitative evaluation metrics (mAP@0.5, mAP@0.5:0.95, per-class P/R, edge latency on Raspberry Pi 4) are grounded on this real 10-class dataset.
   - **Debris Coverage (Phase 8–9):** True space debris (small fragments, mission-related debris, rocket bodies) will be demonstrated **qualitatively** using real-world public orbital imagery (SPEED+ / PRISMA rendezvous imagery).
   - **Transparency & Integrity:** The project report and evaluation presentations will document this empirical finding openly, highlighting that real-world edge deployment requires multi-target tracking across active payloads and defunct assets.

---

## Related References
- [CV Basics & Object Detection Fundamentals](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/cv_basics.md)
- [Dataset Shortlist & Exploration Notes](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/data/README.md)
- [Object Classes & Taxonomy Decisions](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/data/decisions.md)
- [Photometric Augmentation Pipeline](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/data/augment.py)
- [SGP4 Orbital Tracker & TLE Pipeline](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/data/tle_pipeline.py)
- [Edge Model Exporter](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/edge/export.py)
- [Edge Benchmark Suite](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/edge/benchmark.py)