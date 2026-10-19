# Space Object Detection System — Phase 1 Verification & Results

**Project:** ISRO Hardware & Edge Vision Challenge (Problem Statement #05)  
**Phase:** 01 — Foundations, Evaluation Harness & Baseline Findings  
**Date:** October 19, 2026  
**Status:** **Verified & Complete (Foundations & Evaluation Harness Established; Physical Pi 4 Hardware Execution Tracked)**

---

## 1. Executive Summary

Phase 1 established the foundational evaluation infrastructure, resolved critical dataset ambiguities, implemented photometric data augmentation, and built research-grade edge benchmarking protocols for the Edge-Deployable Space Object Detection System.

Before Phase 1:
- `src/eval.py` was an empty 0-byte stub.
- SPARK dataset composition was assumed to contain debris, but unverified.
- Benchmark timings were ad-hoc, mixing 1-thread and 4-thread measurements on laptop CPUs without memory profiling.
- Edge dependencies and dataset configurations were intertwined.

At the conclusion of Phase 1:
- A standalone, training-independent evaluation CLI (`src/eval.py`) exports standardized JSON metric artifacts (`metrics.json`).
- An exhaustive audit of all 80,000 raw SPARK labels verified zero debris instances, formalizing Decision **DEC-04 (10-Class Spacecraft Detection)**.
- Edge benchmarking (`edge/benchmark.py`) enforces strict warmups, 100 timed iterations, thread pinning, and whole-process peak RSS tracking.
- All 5 user acceptance testing (UAT) criteria in `01-UAT.md` are verified.

---

## 2. GSD Phase 1 UAT Verification Matrix (`01-UAT.md`)

All Phase 1 requirements were validated through conversational User Acceptance Testing (UAT) recorded in `.planning/phases/01-foundations-evaluation-harness-day-1-audit/01-UAT.md`:

| # | UAT Test Name | Expected Behavior | Verification Evidence & Outcome | Status |
|---|---|---|---|:---:|
| **1** | **Standalone Evaluation Harness** (`src/eval.py`) | Runs independently of training; evaluates model checkpoints (`best.pt`) on validation/test splits with safe device handling (CPU/GPU); outputs standardized JSON metrics to `outputs/evaluations/`. | Verified with SwissCube and SPARK models. Produces `map50`, `map50_95`, `precision`, `recall`, params, FLOPs, and file size. | **PASS** |
| **2** | **SPARK Dataset Label Audit** (`scripts/audit_spark_annotations.py`) | Audits all 80,000 annotation rows across `train.csv` and `val.csv`; checks for debris keywords; archives results to `docs/spark_class_audit.json`. | Verified 80,000 rows across 10 spacecraft classes (8,000 instances each). Found **0 debris rows**, definitively locking DEC-04 Branch B. | **PASS** |
| **3** | **Edge Testbed Verification** (`edge/verify_pi_env.py`) | Checks Python version ($\ge 3.10$), architecture (`aarch64` / system host), memory, CPU core count, and ONNX Runtime imports; archives health check to `outputs/env_checks/latest.json`. | Verified environment sanity checker runs without crashing and outputs JSON status with clear PASS/FAIL flags. | **PASS** |
| **4** | **Edge Benchmark Harness** (`edge/benchmark.py`) | Discards $\ge 10$ warmup runs; measures $\ge 100$ timed runs; pins thread counts; tracks whole-process peak Resident Set Size (RSS) memory via `psutil`; writes metrics to `outputs/benchmarks/`. | Verified on Laptop CPU Proxy (i5-11400H). Measured 85.94 ms (4 threads, 1024 px) and 46.44 ms (4 threads, 640 px). Physical Pi 4 deployment tracked as pending. | **PASS** |
| **5** | **Repository Hygiene & Config Isolation** | Isolates edge dependencies in `requirements-edge.txt`; separates configs into `configs/swisscube.yaml` and `configs/spark.yaml`; gitignores binary weights and transient scripts (`gp.php`). | Clean git status verified; configs load cleanly in CLI scripts without mutual interference. | **PASS** |

---

## 3. Comprehensive Inventory of What Was Done

### Workstream 1: Standalone Evaluation Harness (`src/eval.py`)
- Created `src/eval.py` wrapping Ultralytics `YOLO.val()`.
- Supports CLI arguments: `--weights`, `--data`, `--split`, `--imgsz`, `--batch`, `--device`, and `--out`.
- Automatically computes and exports:
  - `map50`: Mean Average Precision at IoU 0.50
  - `map50_95`: Mean Average Precision across IoU 0.50:0.95 (primary headline metric)
  - `precision` ($P$) and `recall` ($R$)
  - Model parameter count, FLOPs (GFLOPs), and on-disk file size (MB)
  - Latency breakdown: preprocess, inference, loss, postprocess (ms per frame)
- Automatically archives evaluation runs with timestamps into `outputs/evaluations/` and maintains `outputs/evaluations/latest.json`.

### Workstream 2: Ground-Truth Dataset Audit (`scripts/audit_spark_annotations.py`)
- Built an automated CSV audit utility supporting case-insensitive column headers and aliases.
- Scanned all 80,000 bounding box rows in SPARK `train.csv` (60,000 rows) and `val.csv` (20,000 rows).
- Resolved **Finding F1 / Decision D-01 & D-02**: Confirmed that generic debris does not exist in the official SPARK dataset.
- Formally established Architecture Decision **DEC-04 (Branch B)**: The project is framed as an edge-deployable multi-spacecraft recognition system across 10 classes, with qualitative debris tracking demonstrated via SPEED+ in later phases.
- Results archived to `docs/spark_class_audit.json` and synchronized in `docs/decisions.md`.

### Workstream 3: Space-Domain & Photometric Augmentation Pipeline (`src/data/augment.py`)
- Built `SpaceImageAugmenter` class implementing domain-specific space perturbations:
  - Brightness/contrast scaling ($0.7\times - 1.3\times$) for solar flare and eclipse transitions.
  - Gaussian blur ($\sigma \in [0.0, 2.0]$) for motion blur during orbital tumbling.
  - Gaussian noise ($\sigma \in [2.0, 12.0]$) for space-grade camera sensor noise in low-light LEO conditions.
- Added visual progress bar with ETA and rate tracking using `tqdm`.
- Integrated seamless handoff into `src/train.py`, generating 4,030 augmented training images (`aug1_*`).

### Workstream 4: Edge Testbed & Rigorous Benchmark Suite (`edge/benchmark.py`)
- Overhauled `edge/benchmark.py` to meet strict scientific standards:
  - **Warmup:** Discards first $\ge 10$ runs to eliminate engine initialization, disk cache warmup, and memory allocation overhead.
  - **Sample Size:** Benchmarks over $\ge 100$ timed runs for statistical confidence.
  - **Thread Control:** Explicitly pins CPU threads (e.g. 1 thread vs 4 threads) to prevent OS scheduler jitter.
  - **Memory Profiling:** Integrates `psutil` to track whole-process peak Resident Set Size (RSS) in MB.
  - **Metrics Computed:** Mean latency, median latency, 95th-percentile (p95) latency, standard deviation, and FPS.
- Built `edge/verify_pi_env.py` to inspect target platforms for Python $\ge 3.10$, ARM architecture, RAM capacity, and library dependencies.
- Authored `edge/README.md` providing step-by-step physical Raspberry Pi 4 setup instructions.

### Workstream 5: Repository Hygiene & Config Separation
- Created `requirements-edge.txt` containing only lightweight runtime dependencies (`onnxruntime`, `opencv-python-headless`, `numpy`, `psutil`, `pyyaml`).
- Updated root `requirements.txt` with developer and training tools.
- Separated dataset configurations: `configs/swisscube.yaml` and `configs/spark.yaml`.
- Updated `.gitignore` to prevent committing binary model checkpoints (`*.pt`, `*.onnx`), zip archives, and cached orbital files (`gp.php`).

---

## 4. Quantitative Results & Metrics Summary

### 4.1 Standalone Model Evaluation (`src/eval.py`)

#### Baseline SwissCube (YOLOv8-Nano, 1-Class):
- **mAP@0.5:** 0.9903 (99.03%)
- **mAP@0.5:0.95:** 0.9211 (92.11%)
- **Precision:** 0.9741 (97.41%)
- **Recall:** 0.9621 (96.21%)
- **Parameters:** 3,005,843 (~3.0M)
- **FLOPs:** 8.1 GFLOPs
- **Model Footprint:** 6.03 MB (`best.pt`)

#### SPARK Baseline Model (YOLOv8-Nano, 10-Class Spacecraft):
- **mAP@0.5:** 0.9950 (99.50% — Saturated)
- **mAP@0.5:0.95:** 0.9413 (94.13% — **Primary Headline Metric**)
- **Precision:** 0.9996 (99.96%)
- **Recall:** 0.9986 (99.86%)
- **Parameters:** 3,013,000 (~3.0M)
- **Model Footprint:** 6.29 MB (`best.pt`)
- **Inference Latency (GPU):** 18.50 ms (RTX 2050) / 3.44 ms (GV100)

### 4.2 SPARK Annotation Audit Breakdown (80,000 Ground Truth Labels)

| Class Name | Train Annotations | Validation Annotations | Total Annotations | Debris Keyword Match? | Taxonomy Determination |
|---|---|---|---|:---:|---|
| `Cheops` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| `LisaPathfinder` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| `ObservationSat1` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| `Proba2` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| `Proba3` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| `Proba3ocs` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| `Smart1` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| `Soho` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| `VenusExpress` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| `XMM Newton` | 6,000 | 2,000 | 8,000 | No | Spacecraft |
| **Totals** | **60,000** | **20,000** | **80,000** | **0** | **10 Spacecraft Classes (DEC-04)** |

### 4.3 Benchmark Latency & Memory Profile (ONNX Runtime)

To avoid misleading claims, all timings are strictly attributed to their execution platform:

| Model & Resolution | Hardware Platform | CPU Threads | Mean Latency | Median Latency | p95 Latency | Throughput (FPS) | Peak RSS | Hardware Note |
|---|---|:---:|---|---|---|:---:|---|---|
| SwissCube (640x640) | Laptop CPU Proxy (i5-11400H) | 1 Thread | 170.92 ms | 165.23 ms | 219.49 ms | 5.9 FPS | 118.6 MB | Explains earlier "210 ms" single-thread observation |
| SwissCube (640x640) | Laptop CPU Proxy (i5-11400H) | 4 Threads | 46.44 ms | 45.10 ms | 54.69 ms | 21.5 FPS | 121.8 MB | Laptop proxy baseline |
| SPARK (1024x1024) | Laptop CPU Proxy (i5-11400H) | 1 Thread | 182.38 ms | 178.60 ms | 191.49 ms | 5.5 FPS | 204.7 MB | Legacy 1024 px single-thread |
| **SPARK (1024x1024)** | **Laptop CPU Proxy (i5-11400H)** | **4 Threads** | **85.94 ms** | **84.72 ms** | **97.62 ms** | **11.6 FPS** | **207.1 MB** | **Laptop proxy (NOT a Raspberry Pi 4)** |
| Physical Pi 4 (Cortex-A72) | Raspberry Pi 4B (4 GB) | 4 Threads | *Pending* | *Pending* | *Pending* | *--* | *--* | **PRD Gate: $\le 1000\text{ ms}$ (Pending hardware test)** |

---

## 5. Scientific Rigor & Caveats

1. **Host Proxy vs. Physical Edge:**
   - The 85.94 ms latency was achieved on an Intel Core i5-11400H laptop CPU with 4 threads, which serves as a development proxy.
   - An ARM Cortex-A72 (Raspberry Pi 4) is substantially slower. The official PRD requirement is **$\le 1000\text{ ms}$** on real edge hardware, which remains the binding project target.
2. **Metric Saturation:**
   - mAP@0.5 is at 99.50%, which leaves virtually no dynamic range to measure improvements or degradation during quantization. All future model evaluations will report **mAP@0.5:0.95 (currently 94.13%)** as the primary figure of merit.
3. **Data Splitting & Leakage Risk:**
   - The initial SPARK split was generated per-image. If consecutive orbital frames from the same video sequence were placed into both train and validation sets, metrics may be optimistic. Phase 2 introduces trajectory-aware sequence splitting.

---

---

## 6. Next Milestone: The SPARK-2021 Breakthrough

Following the completion and verification of Phase 1, the official **SPARK-2021 dataset (150,000 images)** was ingested and audited (`outputs/audits/latest_spark2021.json`).

Crucially, **SPARK-2021 contains 25,000 confirmed debris annotations across 5 geometric debris types**, resolving the zero-debris limitation of Phase 1 and restoring the project's original primary mission scope.

A 50-epoch deep training run on an NVIDIA Quadro GV100 GPU (`spark2021_yolov8n_50e`) successfully converged:
- **Test Set Accuracy:** **99.02% mAP@50**, **86.98% mAP@50-95** (across all 11 classes).
- **Debris Class Performance:** **98.34% mAP@50**, **85.39% mAP@50-95**, **97.75% Precision**, **97.34% Recall**.
- **Edge ONNX Latency (1024px, 4 CPU threads):** **67.54 ms (14.8 FPS)**, **207.18 MB peak RSS memory**.
- **Edge Compilations:** Successfully exported to `best.onnx` and `best_ncnn_model/` for direct Raspberry Pi 4 edge deployment.

👉 **Full Report:** See [`docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md) and [`docs/decisions.md` (DEC-07)](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md).

---

## 7. Artifact & File Reference

- **SPARK-2021 Comprehensive Report:** [`docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md)
- **Phase 1 Comprehensive Report:** [`docs/PHASE_1_COMPREHENSIVE_REPORT.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/PHASE_1_COMPREHENSIVE_REPORT.md)
- **Technical Decision Log (DEC-01 to DEC-07):** [`docs/decisions.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md)
- **Workflow Playbook:** [`docs/WORKFLOW_PLAYBOOK.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/WORKFLOW_PLAYBOOK.md)
- **Evaluation CLI:** [`src/eval.py`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py)
- **Dataset Audit Script:** [`scripts/audit_spark_annotations.py`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/scripts/audit_spark_annotations.py)
- **Phase 1 UAT Record:** [`.planning/phases/01-foundations-evaluation-harness-day-1-audit/01-UAT.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/.planning/phases/01-foundations-evaluation-harness-day-1-audit/01-UAT.md)
