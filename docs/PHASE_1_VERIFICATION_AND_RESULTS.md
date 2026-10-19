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
| **2** | **SPARK Dataset Label Audit** (`scripts/audit_spark_annotations.py`) | Audits all 80,000 annotation rows across `train.csv` and `val.csv`; checks for debris keywords; archives results to `docs/spark_class_audit.json`. | Verified 80,000 rows across 10 spacecraft classes (8,000 instances each). Found **0 debris rows in the initial 80k subset**, establishing interim Decision DEC-04 Branch B (later superseded by DEC-10 upon acquiring SPARK-2021). | **PASS** |
| **3** | **Edge Testbed Verification** (`edge/verify_pi_env.py`) | Checks Python version ($\ge 3.10$), architecture (`aarch64` / system host), memory, CPU core count, and ONNX Runtime imports; archives health check to `outputs/env_checks/latest.json`. | Verified environment sanity checker runs without crashing and outputs JSON status with clear PASS/FAIL flags. | **PASS** |
| **4** | **Edge Benchmark Harness** (`edge/benchmark.py`) | Discards $\ge 10$ warmup runs; measures $\ge 100$ timed runs; pins thread counts; tracks whole-process peak Resident Set Size (RSS) memory via `psutil`; writes metrics to `outputs/benchmarks/`. | Verified on 16-Core Xeon CPU. Measured 23.70 ms (4 threads, 640 px) and 65.20 ms (4 threads, 1024 px). Physical Pi edge deployment tracked as open gate. | **PASS** |
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
- Scanned all 80,000 bounding box rows in the SPARK subset `train.csv` (60,000 rows) and `val.csv` (20,000 rows).
- Resolved **Finding F1 / Decision D-01 & D-02**: Confirmed that generic debris did not exist in the 80k subset we had.
- Formally established interim Architecture Decision **DEC-04 (Branch B)**: The project was framed as an edge-deployable multi-spacecraft recognition system across 10 classes for Phase 1 baseline reproducibility (subsequently superseded by DEC-10 with SPARK-2021).
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

### 4.3 Benchmark Latency & Memory Profile (ONNX Runtime, Single-Machine Workstation CPU)

To ensure scientific reproducibility, all benchmarks were executed on the **identical 16-Core Intel Xeon CPU Workstation** using `edge/benchmark.py` (10 warmup, 100 timed runs, FP32 precision):

| Model & Resolution | Hardware Platform | CPU Threads | Mean Latency | Median Latency | p95 Latency | Throughput (FPS) | Peak RSS | Scaling Note |
|---|---|:---:|---|---|---|:---:|---|---|
| SPARK-2021 (640x640) | Intel Xeon (Model 85, 16 Cores) | 1 Thread | **61.54 ms** | 60.81 ms | 65.35 ms | 16.3 FPS | 126.14 MB | Single-thread baseline |
| **SPARK-2021 (640x640)** | **Intel Xeon (Model 85, 16 Cores)** | **4 Threads** | **23.70 ms** | **22.32 ms** | **28.70 ms** | **42.2 FPS** | **129.34 MB** | **$2.75\times$ speedup vs 1024px** |
| SPARK-2021 (1024x1024) | Intel Xeon (Model 85, 16 Cores) | 1 Thread | 167.64 ms | 165.98 ms | 175.38 ms | 6.0 FPS | 204.92 MB | $2.72\times$ linear compute scaling |
| **SPARK-2021 (1024x1024)** | **Intel Xeon (Model 85, 16 Cores)** | **4 Threads** | **65.20 ms** | **64.33 ms** | **72.80 ms** | **15.3 FPS** | **207.36 MB** | **Native 1024px multi-thread** |
| Physical Pi Edge (640 / 1024) | Raspberry Pi 4 / Pi 5 | 4 Threads | *Pending* | *Pending* | *Pending* | *--* | *--* | **PRD Gate: $\le 1000\text{ ms}$ (Open Hardware Gate)** |

---

## 5. Scientific Rigor & Caveats

1. **Workstation Proxy vs. Physical Edge:**
   - The 23.70 ms (640px) and 65.20 ms (1024px) latencies were achieved on a 16-core Intel Xeon workstation CPU, serving as an upper-bound compute proxy.
   - Low-power ARM SBCs (Raspberry Pi 4 / Pi 5) are substantially more constrained. The official PRD requirement is **$\le 1000\text{ ms}$** on real physical edge hardware, which remains the binding open gate.
2. **Metric Saturation:**
   - mAP@0.5 is saturated at $>98.7\%$, which leaves virtually no dynamic range to measure improvements or degradation during quantization. All future model evaluations report **mAP@0.5:0.95** as the primary headline metric.
3. **Data Splitting & Leakage Risk (Split Leakage NOT Ruled Out):**
   - The initial SPARK split was generated per-image. In synthetic orbital video trajectories, adjacent frames may appear across splits if not grouped by trajectory ID. Because global perceptual difference hashing (dHash) collapsed due to uniform solid black space backgrounds (>95% black pixels), the audit could not verify sequence independence. **Split leakage has not been ruled out.** Trajectory-level sequence splitting and bounding-box ROI perceptual hashing remain required.

---

---

## 6. Next Milestone: The SPARK-2021 Breakthrough

Following the completion and verification of Phase 1, the official **SPARK-2021 dataset (150,000 images)** was ingested and audited (`outputs/audits/latest_spark2021.json`).

Crucially, **SPARK-2021 contains 25,000 confirmed debris annotations across 5 geometric debris types**, resolving the zero-debris limitation of Phase 1 and restoring the project's original primary mission scope.

A 50-epoch deep training run on an NVIDIA Quadro GV100 GPU (`spark2021_yolov8n_50e`) successfully converged:
- **Test Set Accuracy:** **99.02% mAP@50**, **86.98% mAP@50-95** (at 1024px across all 11 classes).
- **Debris Class Performance:** **98.34% mAP@50**, **85.39% mAP@50-95** (at 1024px), **82.81% mAP@50-95** (at 640px).
- **Multi-Scale Pareto Optimization:** Evaluated across resolutions without retraining. 640px retains **85.06% headline mAP@50-95** (only -1.92% drop) while cutting pixel compute by 61%.
- **Single-Machine Edge ONNX Benchmarks (16-Core Xeon Workstation):**
  - **640x640 (4 CPU Threads):** **23.70 ms (42.2 FPS)**, **129.34 MB peak RSS memory** ($2.75\times$ speedup, -37.6% RAM).
  - **1024x1024 (4 CPU Threads):** **65.20 ms (15.3 FPS)**, **207.36 MB peak RSS memory**.
- **Edge Compilations:** Successfully exported to `best.onnx` and `best_ncnn_model/` for direct Raspberry Pi 4 edge deployment.
- **Split Leakage Audit:** Audited via 64-bit dHash; identified deep-space black-background false duplicate bias alongside sequence temporal correlation.

👉 **Full Report:** See [`docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md) and [`docs/decisions.md` (DEC-07 & DEC-10)](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md).

---

## 7. Artifact & File Reference

- **SPARK-2021 Comprehensive Report:** [`docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md)
- **Phase 1 Comprehensive Report:** [`docs/PHASE_1_COMPREHENSIVE_REPORT.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/PHASE_1_COMPREHENSIVE_REPORT.md)
- **Technical Decision Log (DEC-01 to DEC-07):** [`docs/decisions.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md)
- **Workflow Playbook:** [`docs/WORKFLOW_PLAYBOOK.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/WORKFLOW_PLAYBOOK.md)
- **Evaluation CLI:** [`src/eval.py`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py)
- **Dataset Audit Script:** [`scripts/audit_spark_annotations.py`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/scripts/audit_spark_annotations.py)
- **Phase 1 UAT Record:** [`.planning/phases/01-foundations-evaluation-harness-day-1-audit/01-UAT.md`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/.planning/phases/01-foundations-evaluation-harness-day-1-audit/01-UAT.md)
