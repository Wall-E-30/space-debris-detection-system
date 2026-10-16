# Space Object Detection System (Edge-Deployable)
## Phase 1 Comprehensive Engineering Report & Baseline Findings
**Project:** ISRO Hardware & Edge Vision Challenge (Problem Statement #05)  
**Phase:** 01 — Foundations, Evaluation Harness & Baseline Findings  
**Date:** October 16, 2026  
**Status:** **Phase 1 Complete (Foundations Established, Real Hardware Edge Verification Pending)**

---

> [!WARNING]
> ### Critical Scope & Dataset Disclosure
> The project dataset currently contains **10 spacecraft classes and ZERO debris annotations** (confirmed by an exhaustive 80,000 label audit). This project is strictly evaluated as an **edge-deployable multi-spacecraft recognition system**. Neither the codebase, metrics, nor reports claim debris detection capabilities until true debris annotations are introduced.

---

## 1. Executive Summary & Objective Status

Phase 1 established the evaluation infrastructure and resolved critical dataset ambiguities for the Edge-Deployable Space Object Detection System. Prior to Phase 1, the standalone evaluation harness was an empty 0-byte stub, SPARK dataset class composition was unverified, and edge benchmarking lacked standardization.

### Core Outcomes:
1. **Evaluation Harness (`src/eval.py`):** Built a standalone evaluation CLI independent of training loops, exporting standardized `metrics.json` artifacts adhering to the EVAL-02 metric contract.
2. **Taxonomy & Class Audit (DEC-04):** Audited 80,000 SPARK annotations, discovering that generic debris does not exist in the dataset. Formally locked **Decision DEC-04 (Branch B: 10-Class Spacecraft Recognition)**.
3. **Rigorous Edge Benchmark Harness (`edge/benchmark.py`):** Upgraded the benchmark suite to enforce $\ge 10$ discarded warmup runs, $\ge 100$ timed runs, CPU thread pinning, and whole-process peak Resident Set Size (RSS) profiling via `psutil`.
4. **Hardware Attribution & Edge Reality:** Evaluated proxy inference across hardware tiers while acknowledging that **the Thursday milestone ("a model running on the physical Raspberry Pi 4") is currently PENDING**.

---

## 2. Hardware Tiers & Proxy Attribution

To maintain scientific integrity, all benchmark measurements are attributed to their exact hardware execution environments without conflating proxy numbers with physical edge results:

| Dimension | Compute Workstation (Quadro GV100) | Host Laptop (CPU Proxy & RTX 2050) | Physical Edge Hardware (Raspberry Pi 4) |
|---|---|---|---|
| **Assigned Role** | **ML Lead** (Deep Training & Export) | **Data & Edge Lead** (Audits, Augment, Local Proxy) | **Target Edge Deployment** |
| **Processor** | High-Core Workstation CPU (16 Cores) | Intel Core i5-11400H (6C/12T) | ARM Cortex-A72 Quad-Core @ 1.5 GHz |
| **GPU / Accelerator**| NVIDIA Quadro GV100 (32 GB HBM2) | NVIDIA GeForce RTX 2050 (4 GB GDDR6) | Broadcom VideoCore VI (CPU execution) |
| **Edge Role** | Compute Server (Training & Export) | **Host CPU Proxy (4 threads)** | **Target Deployment Environment** |
| **Status** | Active | Active | **PENDING (Target: $\le 1000\text{ ms}$ per PRD)** |

---

## 3. Evaluation Harness & Metric Analysis (`src/eval.py`)

The standalone CLI `src/eval.py` evaluates model checkpoints independently of training scripts. On the initial 50-epoch SPARK model, the standalone evaluation yielded:

```
--------------------------------------------------
  STANDALONE EVALUATION SUMMARY (SPARK 10-Class)
--------------------------------------------------
  mAP@0.5:0.95: 0.9413  (94.13% — HEADLINE METRIC)
  mAP@0.5:      0.9950  (99.50% — Saturated)
  Precision:    0.9996  (99.96%)
  Recall:       0.9986  (99.86%)
  GPU Latency:  18.50 ms (RTX 2050) / 3.44 ms (GV100)
  Resolution:   1024x1024 (Legacy Smoke Test Size)
  Model Size:   6.29 MB (best.pt)
--------------------------------------------------
```

> [!IMPORTANT]
> ### Validation Integrity & Inflation Caveats
> 1. **Metric Saturation:** At $\text{mAP@0.5} = 0.9950$, the metric is at ceiling and cannot show meaningful differences during INT8 quantization or 416 px tradeoff analysis. **$\text{mAP@0.5:0.95}$ (94.13%) must serve as the primary headline metric.**
> 2. **Sequence Leakage Risk:** The current SPARK train/val split is likely random per-image rather than trajectory-aware. If adjacent video frames from the same orbital sequence appear in both splits, metrics are artificially optimistic. Phase 2 must enforce strict trajectory-aware splitting.
> 3. **Validation Selection Bias:** `best.pt` was checkpointed on the same validation set it was evaluated on. A frozen, held-out test split must be evaluated before claiming production generalization.

---

## 4. Day-1 SPARK Annotation Audit & Decision DEC-04

Using `scripts/audit_spark_annotations.py`, we parsed all 80,000 ground-truth labels across `train.csv` and `val.csv`:

| Class Name | Train Count | Validation Count | Total Count | Debris Flag | Audit Finding |
|---|---|---|---|---|---|
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
| **TOTALS** | **60,000** | **20,000** | **80,000** | **0 (0.00%)** | **10-Class Lock (DEC-04)** |

> [!NOTE]
> ### Phase 2 Subsampling Plan is Stale
> The earlier plan called for 2:1 majority-class subsampling under the assumption of severe imbalance between spacecraft and debris. Because all 10 spacecraft classes have exactly 8,000 images each, there is **zero class imbalance**. With the 32 GB Quadro GV100, training on the full 60k training split at 640 px is practical; artificial subsampling is unnecessary.

---

## 5. Latency Benchmark Analysis: Host CPU Proxy vs Edge Reality

We profiled ONNX Runtime across thread counts and resolutions. The comparison below clarifies previous confusion between 1-thread p95 latency and 4-thread mean latency:

| Model / Resolution | Hardware Tier | Threads | Mean Latency | p95 Latency | FPS | Peak RSS | Target Status |
|---|---|---|---|---|---|---|---|
| SwissCube (640x640) | Laptop CPU Proxy (i5-11400H) | 1 Thread | 170.92 ms | 219.49 ms | 5.9 | 118.6 MB | Reference (Explains "210 ms" p95) |
| SwissCube (640x640) | Laptop CPU Proxy (i5-11400H) | 4 Threads | 46.44 ms | 54.69 ms | 21.5 | 121.8 MB | Laptop Proxy Baseline |
| SPARK (1024x1024) | Laptop CPU Proxy (i5-11400H) | 1 Thread | 182.38 ms | 191.49 ms | 5.5 | 204.7 MB | Legacy 1024 px Single-Thread |
| **SPARK (1024x1024)** | **Laptop CPU Proxy (i5-11400H)** | **4 Threads** | **85.94 ms** | **97.62 ms** | **11.6** | **207.1 MB** | **Laptop Proxy (NOT a Pi 4)** |
| Physical Pi 4 (Cortex-A72)| Raspberry Pi 4B (4 GB) | 4 Threads | *Pending* | *Pending* | *--* | *--* | **PRD Gate: $\le 1000\text{ ms}$ (PENDING)** |

> [!IMPORTANT]
> ### Resolution Realignment (DEC-06)
> The 1024 px model was an early smoke-test configuration. The primary production model must be trained and benchmarked at **640x640 px** per DEC-06 so that accuracy, memory, and edge latency are evaluated on identical input tensors.

---

## 6. Phase 1 Deliverables & Phase 2 Action Plan

| Workstream | Deliverable | Demonstrated Reality | Status |
|---|---|---|---|
| Evaluation CLI | `src/eval.py` | Standalone metric calculation; exports standardized JSON artifacts. | **Complete** |
| Taxonomy Audit | `scripts/audit_spark_annotations.py` | 80k labels audited; confirmed 0 debris; DEC-04 locked. | **Complete** |
| Augmentation | `src/data/augment.py` | Photometric jitter, blur, noise with ETA tracking; 4,030 images generated. | **Complete** |
| Benchmark Protocol | `edge/benchmark.py` | $\ge 10$ warmup, $\ge 100$ runs, thread control, peak RSS profiling. | **Complete** |
| Physical Edge Run | Model running on physical Pi 4 | Measured on laptop CPU proxy; physical deployment to Pi 4 pending. | **Pending Physical Hardware Run** |

### Immediate Phase 2 Action Plan:
1. **Retrain at 640 px (DEC-06):** Retire 1024 px as legacy; train primary YOLOv8-Nano on full 60k training split at 640 px.
2. **Trajectory-Aware Splitting:** Prevent sequence leakage by ensuring all frames of any sequence reside entirely in train or validation.
3. **Carve-out Frozen Test Split:** Create an untouched test split to report honest, uninflated mAP@0.5:0.95.
4. **Physical Pi 4 Deployment:** Transfer 640 px ONNX/NCNN exports to physical Raspberry Pi 4 and measure actual ARM latency against the PRD target ($\le 1000\text{ ms}$).
