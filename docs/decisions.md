# Project Decision Log

This document records major technical, architectural, and deployment decisions for the Space Debris Detection System.

---

## 1. Edge Deployment Framework (TFLite vs ONNX vs NCNN)

**TFLite** and **ONNX Runtime** are both inference runtimes — you export your trained YOLOv8n weights into their format and run them locally. **Edge Impulse** is a full MLOps platform (train + optimize + deploy) built primarily around its own EON compiler for microcontroller-class TinyML models (FOMO, small CNNs) rather than full YOLO detection heads.

|                           | TFLite                                                       | ONNX Runtime                                         | Edge Impulse                                     |
| ------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------ |
| **What it is**            | Google's mobile/embedded inference runtime                   | Cross-platform inference runtime                     | End-to-end MLOps platform (train + deploy)       |
| **YOLOv8 export support** | Native `.tflite` export via Ultralytics                      | Native `.onnx` export via Ultralytics                | Requires "Bring Your Own Model" workaround       |
| **RPi 4 CPU speed**       | Finicky without manual multi-threading/XNNPACK configuration | Comparable or better than baseline TFLite on ARM CPU | Not designed for full YOLO detector backbones    |
| **Ultralytics Benchmark** | Included in standard export list                             | Included as standard portable format                 | Not part of Ultralytics' benchmarked export list |
| **Ecosystem fit**         | Good — well-documented, offline                              | Good — cross-platform interchange format             | Poor fit for YOLOv8n detection on RPi4           |
| **License**               | Apache 2.0                                                   | MIT                                                  | Free tier + paid tiers                           |

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
   - Default CPU training (~34 hrs/5 epochs) migrated to **NVIDIA RTX 2050 GPU (CUDA)** for preliminary prototyping.
   - Enabled dataset RAM caching (`cache=True`) to eliminate disk I/O bottlenecks.
   - Mixed precision (`amp=True`) and 4 dataloader workers (`workers=4`) enabled.
   - *[Historical Correction: Initial speculative estimate was ~8–10 minutes; actual measured RTX 2050 5-epoch training duration was 5,441 seconds (~90.7 minutes). In Phase 2, deep 50-epoch training was transitioned to the departmental 32 GB NVIDIA Quadro GV100 workstation (DEC-06 / DEC-10).]*

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

> [!NOTE]
> **SUPERSEDED by DEC-10:** The preliminary collapsed 3-class proposal (`satellite`, `debris`, `rocket_body`) and focal loss plan was superseded by the SPARK-2021 11-class ground-truth taxonomy (10 satellites + 1 synthetic debris class). See DEC-10.

- **Preliminary Strategy (Superseded):** Collapsed multi-spacecraft classes into `satellite` superclass; `debris` mapped 1:1. Deferred `rocket_body` synthesis to v2/stretch goals.
- **Active Strategy (DEC-10):** Full 11-class training on SPARK-2021 (10 satellite classes + consolidated synthetic debris class across 25,000 ground truth annotations).

---

## 5. Photometric Augmentation Pipeline (Domain Gap Mitigation)

### Context & Design

Synthetic satellite imagery (SPARK, SwissCube) exhibits clean rendering artifacts (uniform lighting, zero sensor noise, perfect optical focus) that do not reflect real-world space domain telescope or star tracker conditions.

### Decisions

1. **Photometric-Only Transforms:** Implemented `src/data/augment.py` using composable photometric adjustments (brightness/contrast jitter, Gaussian blur, sensor read noise). Because pixel positions are unchanged, bounding box annotations remain 100% invariant and require no coordinate recalculations.
2. **Strict Split Isolation:** Augmentations are applied exclusively to training splits (`train/`); validation and test sets remain unaltered to guarantee uncorrupted metric evaluation.
3. **Distinction from Online Training Augmentations:** The offline photometric pipeline (`src/data/augment.py`) is decoupled from the online geometric augmentations in `src/train.py` (`degrees=180`, `flipud=0.5`, `erasing=0.4`), where coordinate transforms are handled on-the-fly by the Ultralytics training engine.

---

## 6. Stage 3 Orbit Propagation & Conjunction Distance Estimation

### Architecture & Mechanics

An independent orbital propagation pipeline (`src/data/tle_pipeline.py`) operates as a decoupled catalog layer alongside the vision detector.

```
+---------------------------------------+      +------------------------------------------+
|       Optical Vision Subsystem        |      |   Independent Orbital Catalog Subsystem  |
|  YOLOv8 Detection (BBox + Class)      |      |   Live CelesTrak TLEs (Active Catalog)   |
+---------------------------------------+      +------------------------------------------+
                   |                                                 |
                   v                                                 v
         Detected Class Label                      SGP4 Orbit Propagation (Known Satellites)
                   |                                                 |
                   +---------> [ NORAD Lookup ] -------------------> 3D Geocentric XYZ (km)
                                     |                                       |
                   +-----------------+-------------------+                   v
                   |                                     |           Euclidean Distance
                   v                                     v           & Conjunction Risk
          Known Spacecraft Class                   Debris Class
          (e.g., AcrimSat, Sentinel-6)             (UNCATALOGED)
                   |                                     |
                   v                                     v
          Catalog ID Resolved                      NO NORAD ID Exists
          SGP4 Position Calculated                 SGP4 Propagation Unsupported
```

### Decisions

1. **Decoupled Catalog Architecture:** The vision detector cannot deterministically resolve orbital ephemeris from a 2D bounding box alone. The TLE module is an **independent catalog layer** that maps identified spacecraft models to candidate NORAD IDs.
2. **Uncataloged Space Debris Limitation:** **Synthetic space debris has NO catalog ID in CelesTrak.** The TLE layer explicitly cannot propagate or predict orbits for uncataloged debris fragments without prior radar or multi-station optical state vector estimation.
3. **SPARK-2021 Class Mapping:** Updated the catalog mapping from deprecated Phase 1 classes to active SPARK-2021 spacecraft assets (`AcrimSat`: 26032, `Aquarius`: 37673, `Aura`: 28376, `Calipso`: 29108, `Cloudsat`: 29107, `CubeSat`: 25544 proxy, `Jason`: 26997, `Sentinel-6`: 46984, `Terra`: 25994, `TRMM`: 25063).

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

> [!WARNING]
> **SUPERSEDED by Phase 1 Rigorous Benchmark Suite & DEC-01:** The preliminary 30–50 iteration smoke test below with an uncalibrated memory measurement (~0.0 MB RSS) has been superseded by the research-grade `edge/benchmark.py` harness enforcing $\ge 10$ discarded warmup runs, $\ge 100$ timed runs, CPU thread pinning, and whole-process peak RSS profiling via `psutil`.

### Historical Smoke Test (Superseded)

| Engine           | Thread Config | Mean Latency  | p95 Latency   | Throughput (FPS) | Memory Footprint (RAM) | Status |
| ---------------- | ------------- | ------------- | ------------- | ---------------- | ---------------------- |:---:|
| **ONNX Runtime** | `threads=1`   | 223.20 ms     | 385.64 ms     | 4.5 FPS          | ~0.0 MB RSS            | *Superseded* |
| **ONNX Runtime** | `threads=4`   | 209.71 ms     | 226.12 ms     | 4.8 FPS          | ~0.0 MB RSS            | *Superseded* |
| **NCNN**         | `threads=4`   | **165.21 ms** | **178.48 ms** | **6.1 FPS**      | **~10.9 MB RSS**       | *Superseded* |

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

| Split                   | Total Image Files | Total Label Rows | Detected Classes             | Debris Rows |
| :---------------------- | :---------------- | :--------------- | :--------------------------- | :---------- |
| **Train (`train.csv`)** | 60,000            | 60,000           | 10 classes (6,000 per class) | **0**       |
| **Val (`val.csv`)**     | 20,000            | 20,000           | 10 classes (2,000 per class) | **0**       |
| **Total**               | 80,000            | 80,000           | 10 classes (8,000 per class) | **0**       |

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

### Decision & Project Alignment (Historical Adoption of Branch B)

1. **Finding F1 Confirmed:** The initial SPARK release evaluated in Phase 1 contained **zero annotated debris fragments**. It was strictly a 10-class spacecraft dataset.
2. **Phase 1 Branch B Lock:** Formally defined as 10-class spacecraft detection for Phase 1 baseline reproducibility.
3. **Superseded by DEC-10:** Ground-truth space debris detection was subsequently restored via the SPARK-2021 dataset.

---

## DEC-05: TLE Independent Catalog Layer & Uncataloged Debris Limitation

**Date:** 2026-10-14  
**Author:** Software Architecture Lead  
**Status:** Decided & Locked  

1. **Decoupled Architecture:** Optical 2D bounding box detection cannot deterministically resolve 3D orbital state vectors. The TLE subsystem operates as an **independent catalog layer** mapping identified operational satellite classes to candidate NORAD IDs.
2. **Debris Limitation:** **Synthetic space debris has NO catalog ID in CelesTrak.** The TLE pipeline cannot perform SGP4 propagation or conjunction analysis for uncataloged debris fragments without prior multi-station orbit determination.

---

## DEC-06: Compute Hardware Allocation Strategy

**Date:** 2026-10-16  
**Author:** Systems Lead  
**Status:** Decided & Locked  

1. **Local Host (NVIDIA RTX 2050 4GB / Intel Core i5-11400H):** Dedicated to dataset audits, preprocessing, unit test harnesses, and local development proxies.
2. **Compute Workstation (NVIDIA Quadro GV100 32GB / 16-Core Xeon):** Dedicated to high-resolution (1024 px) multi-class deep training (50+ epochs) and high-throughput batch evaluation.
3. **Target Edge Hardware (Raspberry Pi 4B 4GB / Cortex-A72):** Dedicated to physical embedded verification against the PRD target ($\le 1000\text{ ms}$).

---

## DEC-07: Model Input Resolution & Multi-Scale Trade-off Strategy

**Date:** 2026-10-17  
**Author:** ML & Edge Systems Team  
**Status:** Decided & Locked  

1. **Primary High-Resolution Training (1024x1024 px):** Adopted for deep training on the 32 GB Quadro GV100 to maximize feature extraction for small orbital bodies (CubeSat, debris fragments).
2. **Multi-Scale Evaluation Trade-off:** While 1024 px is primary, the model is evaluated across multiple resolutions (640 px, 800 px, 1024 px) to measure the explicit accuracy-vs-latency trade-off without mandatory retraining.
3. **Edge Feasibility Constraint:** Recognizing that 1024 px requires $\sim 2.5\times$ the FLOPs of 640 px, the physical Raspberry Pi 4 edge benchmark will evaluate both 640 px and 1024 px to assess compliance with the $\le 1000\text{ ms}$ latency gate.

---

## DEC-10: Transition to SPARK-2021 11-Class Dataset (Synthetic Debris Integration)

**Date:** 2026-10-19  
**Author:** ML & Edge Systems Team  
**Status:** Decided & Executed (`supersedes DEC-04 Branch B`)  
**Evidence Artifacts:**
- Dataset Audit: `outputs/audits/latest_spark2021.json`
- Training Run: `experiments/spark2021_yolov8n_50e`
- Evaluation Reports: `outputs/evaluations/spark2021_val_metrics.json`, `outputs/evaluations/spark2021_test_metrics.json`
- Edge Benchmark: `outputs/benchmarks/spark2021_onnx_benchmark.json`

### Empirical Findings:
1. **Resolution of Debris Deficiency:** While the initial release of SPARK evaluated in Phase 1 contained zero debris annotations (DEC-04), the official **SPARK-2021 dataset** contains **25,000 confirmed debris annotations** across 5 synthetic geometric debris types alongside 125,000 satellite annotations across 10 spacecraft classes (150,000 images total).
2. **11-Class Model Convergence:** A 50-epoch deep training run on an NVIDIA Quadro GV100 (`spark2021_yolov8n_50e`) converged cleanly to:
   - Overall Test mAP@50: **99.02%**
   - Overall Test mAP@50-95: **86.98%**
   - **Synthetic Debris Class (Class 6) Test Precision:** **97.75%**
   - **Synthetic Debris Class (Class 6) Test Recall:** **97.34%**
   - **Synthetic Debris Class (Class 6) Test mAP@50:** **98.34%**
   - **Synthetic Debris Class (Class 6) Test mAP@50-95:** **85.39%**
3. **Edge Optimization:** The model was successfully compiled to ONNX FP32 (67.54 ms latency / 14.8 FPS on 4 CPU threads on 16-core workstation) and NCNN (`best_ncnn_model/`), providing dual runtime deployment readiness for the Raspberry Pi 4.

### Architectural Decision:
- **Supersession of Branch B:** The project returns to its primary original objective: quantitative detection and edge tracking of both **operational spacecraft AND synthetic space debris**.
- **Dataset Contract:** `configs/spark2021.yaml` (11 classes) is established as the primary dataset contract moving forward.

---

## Related References

- [SPARK-2021 Comprehensive Report](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md)
- [Phase 1 Verification & Results](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/PHASE_1_VERIFICATION_AND_RESULTS.md)
- [CV Basics & Object Detection Fundamentals](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/cv_basics.md)
- [Photometric Augmentation Pipeline](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/data/augment.py)
- [SGP4 Orbital Tracker & TLE Pipeline](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/data/tle_pipeline.py)
- [Edge Model Exporter](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/edge/export.py)
- [Edge Benchmark Suite](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/edge/benchmark.py)
