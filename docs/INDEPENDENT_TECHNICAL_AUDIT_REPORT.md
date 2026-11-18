# Independent Technical Audit Report
**Target Repository:** `space-debris-detection-system` (ISRO Problem Statement #05)  
**Role:** Independent Technical Auditor  
**Audit Baseline:** Source Code, Training Logs, Checkpoint Binaries, Benchmark Exports, Evaluation JSONs, and Technical Documentation

---

## Executive Audit Summary

The deep learning and edge deployment engineering conducted for this project—specifically the **50-epoch deep training on the NVIDIA Quadro GV100**, the **180,000-image augmented training corpus**, the **sub-millisecond PyTorch-to-ONNX numerical parity**, and the **empirical ONNX Runtime benchmarks on Raspberry Pi 5**—is authentic, mathematically grounded, and reproducible from raw artifacts.

However, prior to formal submission to ISRO evaluators, several critical discrepancies, hardware requirement mismatches, and narrative embellishments must be quarantined and remediated. Most notably, an earlier draft of the Master Report introduced **artificially smoothed per-class metrics to mask the geometric degradation of small CubeSats**, the physical edge benchmarks were executed on a **Raspberry Pi 5 (Cortex-A76) rather than the PRD-mandated Raspberry Pi 4 (Cortex-A72)**, and the astrodynamics tracking script contains a **hardcoded proxy mapping all generic CubeSats to the International Space Station (NORAD 25544)**.

---

## Detailed Task Verification & Evidence Analysis

### Task 1: Cross-Reference of Per-Class mAP Metrics (Master Report vs. Raw JSONs)

* **Raw Artifact:** `outputs/evaluations/spark2021_test_metrics.json`
* **Historical Report:** `docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md`
* **Target Document:** `docs/MASTER_PROJECT_REPORT.md` (and patch `audit/proposed_patches/01_master_project_report.patch`)

#### Empirical Discrepancy Matrix
A line-by-line audit revealed that the Master Report originally contained altered numbers that smoothed the variance across classes while preserving the exact overall mean of **86.98% mAP@50-95**:

| Class Name | Raw JSON mAP@50-95 | Unpatched Master Report | Discrepancy ($\Delta$) | Audit Assessment |
|---|:---:|:---:|:---:|---|
| `AcrimSat` | **95.09%** | 90.07% | -5.02% | **Artificially Deflated** |
| `Aquarius` | **92.97%** | 90.96% | -2.01% | Minor smoothing |
| `Aura` | **93.80%** | 89.70% | -4.10% | Artificially Deflated |
| `Calipso` | **91.33%** | 83.21% | -8.12% | **Artificially Deflated** |
| `Cloudsat` | **86.25%** | 89.87% | +3.62% | Artificially Inflated |
| `CubeSat` | **63.41%** | **82.17%** | **+18.76%** | **CRITICAL MANIPULATION (Masked Failure)** |
| **`Debris`** | **85.39%** | **85.39%** | **0.00%** | **Accurate** (Verified Target Class) |
| `Jason` | **91.73%** | 89.70% | -2.03% | Minor smoothing |
| `Sentinel-6` | **89.93%** | 90.99% | +1.06% | Minor smoothing |
| `Terra` | **89.16%** | 89.80% | +0.64% | Minor smoothing |
| `TRMM` | **77.76%** | 74.88% | -2.88% | Deflated to create narrative scapegoat |
| **Mean (All)** | **86.98%** | **86.98%** | **0.00%** | **Mean preserved to evade superficial review** |

#### Root Cause Analysis
1. **The CubeSat Compression:** In SPARK-2021, CubeSats are miniature spacecraft ($<100 \times 100\text{ px}$ at native 1024px; $<30 \times 30\text{ px}$ at 640px). While classification accuracy remained high (95% diagonal on the confusion matrix), boundary regression errors of only 3–5 pixels cause the bounding box IoU to drop below $0.75$, collapsing mAP across the upper evaluation bins ($0.75-0.95$).
2. **Fabricated Narrative Scapegoat:** The unpatched draft falsely attributed the lowest score to `TRMM` (74.88%), asserting that *"analysis of the confusion matrix reveals TRMM false positives under dark Earth-limb illumination."* This claim is scientifically invalid: a confusion matrix records only predicted vs. ground-truth counts and carries zero environmental, illumination, or celestial geometry metadata.
3. **Current State:** The patch (`audit/proposed_patches/01_master_project_report.patch`) has restored the authentic raw numbers (CubeSat = 63.41%, TRMM = 77.76%, AcrimSat = 95.09%) in `MASTER_PROJECT_REPORT.md`. All future project slide decks and defense materials must strictly use the verified JSON numbers.

---

### Task 2: Training Provenance & Augmentation Verification

* **Config Artifact:** `experiments/spark2021_yolov8n_50e/args.yaml`
* **Convergence Log:** `experiments/spark2021_yolov8n_50e/results.csv`
* **Visual Artifacts:** `train_batch450000.jpg`, `labels.jpg`
* **Leakage Log:** `outputs/audits/split_leakage_audit.json`

#### Provenance Verification Verdict: **CONFIRMED & AUTHENTIC**
1. **Dataset Scaling:** Raw SPARK-2021 contains 90,000 training images (`train_labels.csv`). Running `src/data/augment.py` produced 90,000 offline photometric augmented images (`aug1_*`), establishing an exact training corpus of **180,000 images**. `labels.jpg` verifies 15,000 instances per satellite class and 30,000 debris instances ($10 \times 15,000 + 30,000 = 180,000$).
2. **Iteration Count Arithmetic:**
   $$\text{Batches per Epoch} = \frac{180,000 \text{ images}}{\text{Batch Size } 16} = 11,250 \text{ batches/epoch}$$
   $$\text{Total Iterations} = 11,250 \times 50 \text{ epochs} = \mathbf{562,500 \text{ iterations}}$$
3. **Mosaic Closure Proof:** `args.yaml` sets `close_mosaic: 10`. Mosaic augmentation was closed at epoch 40:
   $$11,250 \times 40 = \mathbf{450,000 \text{ iterations}}$$
   The physical checkpoint image `train_batch450000.jpg` exists in the repository, and `results.csv` records a steep loss reduction between epoch 40 (box loss: 0.710) and epoch 41 (box loss: 0.593), directly confirming the mosaic disablement schedule.
4. **Compute Duration:** Total execution logged in `results.csv` (line 51) is **122,183 seconds (~33.94 hours)** on an NVIDIA Quadro GV100 (32 GB VRAM).
5. **CRITICAL CAVEAT (Split Leakage NOT Ruled Out):** While zero `aug1_*` images exist in `val/` or `test/`, the perceptual difference hashing audit (`scripts/check_split_leakage.py`) collapsed because space imagery features $>95\%$ uniform solid black backgrounds ($\text{RGB} = [0, 0, 0]$), yielding a false 61.27% duplicate match rate. **Temporal sequence leakage between consecutive rendering frames across splits has NOT been ruled out.** The project must not claim evaluations are "guaranteed unpolluted."

---

### Task 3: PyTorch vs. ONNX Export Parity Verification

* **Audit Script:** `audit/scripts/check_onnx_pytorch_parity.py`
* **Audit Record:** `audit/onnx_parity_check.json`
* **Models Evaluated:** `experiments/spark2021_yolov8n_50e/weights/best.pt` vs. `best.onnx`

#### Export Parity Verdict: **100% NUMERICAL PASS**

```
+----------------------------------------------------------------------------------------+
|                            PYTORCH VS. ONNX PARITY AUDIT                               |
+----------------------------------------------------------------------------------------+
| Metric Parameter                        | Target Threshold       | Measured Outcome    |
+-----------------------------------------+------------------------+---------------------+
| Sample Images Audited                   | 50 test images         | 50 images           |
| PyTorch Total Detections                | Baseline               | 46 detections       |
| ONNX Total Detections                   | Exact Count Match      | 46 detections (PASS)|
| Matched Detections (IoU >= 0.50)        | 100.0%                 | 46 / 46 (100.0%)    |
| Mean Bounding Box IoU                   | >= 0.999               | 1.0000 (Exact)      |
| Minimum Bounding Box IoU                | >= 0.990               | 1.0000 (Exact)      |
| Class Classification Agreement          | 100.0%                 | 100.0% (PASS)       |
| Mean Confidence Score Delta (|Δconf|)   | <= 1e-4                | 1.0e-06             |
| Maximum Confidence Score Delta          | <= 1e-3                | 4.0e-06             |
+----------------------------------------------------------------------------------------+
```

Export graph simplification (`simplify=True`) and Opset 12 translation preserved floating-point activations with an absolute maximum confidence drift of only $0.000004$.

---

### Task 4: Hardware Benchmarks Audit (Raspberry Pi 5 vs. Pi 4)

* **Raw SBC Benchmarks:** `outputs/benchmarks/pi5_onnx_640_benchmark.json`, `pi5_onnx_800_benchmark.json`, `pi5_onnx_1024_benchmark.json`
* **Host CPU Benchmarks:** `outputs/benchmarks/spark2021_onnx_*_benchmark.json`
* **Hardware Requirement:** PRD Target $\le 1000\text{ ms}$ on Raspberry Pi 4 (Cortex-A72)

#### Hardware Verification Findings

```
               Physical SBC Benchmarks vs. PRD Hardware Mandate
   Mean Latency (ms)
    1600 │                                                 ▲ Projected Pi 4 Fail
    1400 │                                                 │ (~1300 - 1500 ms)
    1200 │                                                 │
    1000 ├─────────────────────────────────────────────────┼── PRD LATENCY GATE (1000 ms)
     800 │                                                 │
     600 │                                   ● 515.19 ms   │
     400 │                     ● 285.15 ms   (Pi 5 1024px) │
     200 │       ● 144.83 ms   (Pi 5 800px)                │
       0 └───────(Pi 5 640px)──────────────────────────────┴────────────────────────►
```

1. **Hardware Discrepancy (HIGH AUDIT RISK):** The raw JSON artifacts confirm that physical testing was conducted on a **Raspberry Pi 5 (`aarch64`, ARM Cortex-A76 @ 2.4 GHz)**. The PRD mandate and Problem Statement #05 specifically designated the **Raspberry Pi 4B (ARM Cortex-A72 @ 1.5 GHz)**.
2. **Performance Delta & Failure Projection:** Cortex-A76 achieves $2.5\times - 3.0\times$ higher compute throughput than Cortex-A72:
   * **640px:** Pi 5 achieves **144.83 ms (6.9 FPS)**. Projected Pi 4 latency: **~360–430 ms (PASS)**.
   * **800px:** Pi 5 achieves **285.15 ms (3.5 FPS)**. Projected Pi 4 latency: **~710–850 ms (BORDERLINE)**.
   * **1024px:** Pi 5 achieves **515.19 ms (1.9 FPS)**. Projected Pi 4 latency: **~1,290–1,550 ms (HARD FAILURE)**.
3. **Thread Scaling & Bottlenecks:** On Pi 5, scaling from 1 thread (378.56 ms) to 4 threads (144.83 ms) at 640px yields a **$2.61\times$ speedup**. This falls short of the ideal $4.0\times$ theoretical linear limit due to memory bus contention in dense convolution layers, but indicates healthy multi-core thread scaling.
4. **Thermal Throttling Evaluation:** The 100-run benchmark exhibited standard deviation of only **2.80 ms** at 640px, with the 95th-percentile latency (150.42 ms) tightly tracking the mean (144.83 ms). There is **zero evidence of thermal throttling** under the tested profile.

---

### Task 5: Astrodynamics Layer Audit (`src/data/tle_pipeline.py`)

* **Source File:** `src/data/tle_pipeline.py` (lines 15–86)
* **Architectural Decision:** `docs/decisions.md` (DEC-05)

#### Logic Assessment & Discrepancies
1. **Decoupling from 2D Vision:** The code **does not erroneously project 2D bounding boxes into 3D orbits**. `get_position()` accepts only a class label string (e.g., `"AcrimSat"`), ignoring bounding box coordinates and camera focal parameters.
2. **The Hardcoded CubeSat ISS Proxy (LOGICAL DEFECT):**
   ```python
   # Line 21 in src/data/tle_pipeline.py:
   "CubeSat": 25544,  # LEO CubeSat/ISS deployment proxy
   ```
   **NORAD ID 25544 is the International Space Station (ISS).** If the YOLO model detects a CubeSat, the pipeline queries CelesTrak for the ISS ephemeris and returns the ISS's position. This proxy will be immediately flagged by orbital dynamics reviewers.
3. **Uncataloged Space Debris Handling:**
   ```python
   # Line 26 & 46 in src/data/tle_pipeline.py:
   "Debris": None,
   if detected_class == "Debris":
       return {"error": "Debris objects are uncataloged and have no deterministic NORAD IDs in CelesTrak..."}
   ```
   Debris correctly returns an error because space debris has no deterministic NORAD ID in CelesTrak.
4. **The `calculate_distance` Conjunction Flaw:**
   `calculate_distance(class_a, class_b)` calculates the Euclidean distance between two satellites using geocentric ECI coordinates. If one object is `Debris`, it returns `None`. If two satellites are detected in the same camera frame, it calculates their separation based on independent global orbits, ignoring camera relative geometry.
5. **Architectural Reconciliation:** The documentation must formally present the TLE module as an **independent catalog awareness layer for known assets**, not an end-to-end visual conjunction assessment tool.

---

## Prioritized List of Audit Findings

```
+----------------------------------------------------------------------------------------------------+
|                                    AUDIT FINDINGS REGISTER                                         |
+----------------------------------------------------------------------------------------------------+
| ID       | Severity | Domain           | Summary of Finding                                        |
+----------+----------+------------------+-----------------------------------------------------------+
| AUD-01   | CRITICAL | Evaluation Data  | CubeSat 1024px mAP@50-95 was altered (82.17% vs 63.41% raw)|
| AUD-02   | HIGH     | Hardware Gate    | Latency benchmarks run on Pi 5; Pi 4 will fail at 1024px  |
| AUD-03   | HIGH     | Astrodynamics    | CubeSat hardcoded to ISS (NORAD 25544); Debris has no TLE |
| AUD-04   | HIGH     | Data Splitting   | Split leakage not ruled out due to black-space dHash collapse|
| AUD-05   | MEDIUM   | Model Complexity | Unqualified 8.2 GFLOPs reported; 1024px native is 22.4 GFLOPs|
| AUD-06   | MEDIUM   | Git Provenance   | git tag 'model-freeze' and commit hash provenance check   |
| AUD-07   | MEDIUM   | Documentation    | DEC numbers inconsistently redefined across Phase 1 & 2   |
+----------------------------------------------------------------------------------------------------+
```

### Finding Details

#### [AUD-01] CRITICAL: Per-Class Metric Manipulation in Early Reports
* **Impact:** Immediate disqualification if reviewed against raw JSON logs.
* **Evidence:** `outputs/evaluations/spark2021_test_metrics.json` lines 58–59 record CubeSat mAP@50-95 as `0.6341`. Early draft of `docs/MASTER_PROJECT_REPORT.md` reported `82.17%`.

#### [AUD-02] HIGH: Hardware Target Bait-and-Switch (Pi 5 vs. Pi 4)
* **Impact:** Accusation of failing to meet the mandated hardware gate ($\le 1000\text{ ms}$ on Raspberry Pi 4).
* **Evidence:** `outputs/benchmarks/pi5_onnx_*.json` confirms the physical testbed was an ARM Cortex-A76 (Raspberry Pi 5). At 1024px (515 ms on Pi 5), a Pi 4 is projected to take $>1300\text{ ms}$.

#### [AUD-03] HIGH: Incoherent Astrodynamics Coupling & ISS Proxy
* **Impact:** Technical embarrassment during astrodynamics oral examination.
* **Evidence:** `src/data/tle_pipeline.py` maps `CubeSat` to NORAD 25544 (ISS) and aborts on `Debris`. 2D bounding boxes cannot physically determine 3D orbital state vectors.

#### [AUD-04] HIGH: Split Leakage Not Empirically Ruled Out
* **Impact:** Accusation of over-fitting and reporting artificially inflated test metrics.
* **Evidence:** `outputs/audits/split_leakage_audit.json` records a 61.27% dHash match rate due to empty black background collapse. Orbital video sequence overlap between splits cannot be disproven.

#### [AUD-05] MEDIUM: Misattributed Compute Complexity (FLOPs)
* **Impact:** Reviewers calculating FLOPs will flag discrepancies between COCO defaults and custom resolution models.
* **Evidence:** The model has 3.14M parameters and runs at **8.8 GFLOPs at 640px**, **13.7 GFLOPs at 800px**, and **22.4 GFLOPs at 1024px**. Documents reporting a flat "8.2 GFLOPs" are quoting standard 640px 80-class COCO defaults.

---

## Exact Remediation Steps (Prioritized Action Plan)

### Step 1: Reconcile Evaluation Metrics (Immediate)
1. Ensure all active reports (`MASTER_PROJECT_REPORT.md`, slides, executive summaries) reflect the verified JSON values:
   * **CubeSat mAP@50-95:** **63.41%**
   * **TRMM mAP@50-95:** **77.76%**
   * **AcrimSat mAP@50-95:** **95.09%**
   * **Debris mAP@50-95:** **85.39%**
   * **Overall Headline mAP@50-95:** **86.98%**
2. Explain the physical root cause honestly in documentation: CubeSat is the smallest target in the dataset ($<100\times100$ px), where 3–5 pixel boundary errors severely depress IoU above the 0.75 threshold.

### Step 2: Formally Disclose Hardware Target & Define Pareto Operating Point
1. In all engineering reports, explicitly state:
   > *"Empirical hardware benchmarking was conducted on physical Raspberry Pi 5 (ARM Cortex-A76 @ 2.4 GHz), passing the $\le 1000\text{ ms}$ gate across all resolutions (144.83 ms at 640px, 285.15 ms at 800px, 515.19 ms at 1024px). Physical testing on Raspberry Pi 4 (Cortex-A72) remains an open gate, where 640px (~380 ms) is designated as the primary compliant operating point."*
2. Establish **640x640 px as the primary edge deployment baseline** to guarantee that the model stays under the 1000 ms ceiling on older Raspberry Pi 4 hardware while retaining 85.06% mAP@50-95 (82.81% on Debris).

### Step 3: Cleanse Astrodynamics Logic & Documentation
1. **Remove the ISS Proxy:** In `src/data/tle_pipeline.py`, change `"CubeSat": 25544` to `"CubeSat": None` or add an explicit disclaimer that generic CubeSats require on-orbit tracklet determination.
2. **Re-label Subsystem in Architecture Diagrams:** Present the TLE pipeline strictly as an **independent catalog awareness layer for known assets**, not an end-to-end conjunction predictor.
3. State explicitly in the submission deck: *"Optical 2D vision provides line-of-sight bearing and classification. Uncataloged space debris fragments have no NORAD IDs and trigger visual proximity tracking alerts."*

### Step 4: Accurately Report Compute Complexity & Data Leakage Caveats
1. Update all specifications to report FLOPs relative to resolution:
   * **640x640 px:** 8.8 GFLOPs (144.83 ms on Pi 5)
   * **800x800 px:** 13.7 GFLOPs (285.15 ms on Pi 5)
   * **1024x1024 px:** 22.4 GFLOPs (515.19 ms on Pi 5)
2. State the split leakage caveat transparently: *"Offline photometric augmentations were isolated to train splits. Because orbital rendering sequences feature identical black space backgrounds, difference hashing collapsed; sequence-level leakage is acknowledged as an open research limitation."*

### Step 5: Lock Model Provenance & Tag Git Freeze
1. Confirm git tag `model-freeze` is anchored to the commit containing frozen candidate weights:
   ```bash
   git tag -a model-freeze -m "Model freeze: 11-class YOLOv8n trained on SPARK-2021"
   ```
2. Embed the immutable cryptographic hashes into project documentation:
   * **`best.pt` SHA-256:** `34fe3d10b045849fef33e29f10028e878fbd7745a71e2839aa02875e47c8795d`
   * **`best.onnx` SHA-256:** `df2cde420af209acddcc9aceb5e9d9ca909d75cdca16474bc7f71aca60260912`

---

## Final Auditor Verdict

**Conditionally Approved for Submission.**  
The technical core of this project (training convergence, 98.34% debris mAP@50, ONNX parity, and 6.9 FPS edge inference on Raspberry Pi 5) is robust and defensible. Executing the 5 remediation steps above will eliminate exposure to scientific scrutiny, aligning the submission with ISRO technical standards.

