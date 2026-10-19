# Roadmap: Edge-Deployable Space Debris Detection System

## Overview

This roadmap defines the 4-week execution trajectory (12 Oct – 8 Nov 2026) for the Edge-Deployable Space Debris Detection System (Problem Statement #05). Starting from an audited brownfield repository with a SwissCube smoke test, the project systematically resolves critical evidence gaps (evaluation harness, SPARK class audit, resolution alignment), executes reproducible multi-class YOLOv8-Nano training on a local 4GB GPU, applies INT8 post-training quantization, benchmarks inference latency and peak memory on physical Raspberry Pi 4 hardware, and integrates an end-to-end CLI and interactive demonstrator with an independent SGP4 orbital awareness layer.

## Phases

- [x] **Phase 1: Foundations, Evaluation Harness & Day-1 Audit** - Standalone `eval.py`, SPARK CSV audit, repo hygiene, and Pi environment verification.
- [ ] **Phase 2: SPARK Dataset Preparation & Stratified Subsetting** - Build balanced Subset B (1024ox, subsampled 2–3:1) with train-split photometric augmentation.
- [ ] **Phase 3: Baseline Edge Validation on Raspberry Pi** - Refactor benchmark protocol (≥100 runs, peak RSS) and measure SwissCube model on physical Pi 4.
- [ ] **Phase 4: SPARK Model Training & Optimization** - Execute overnight YOLOv8-Nano 1024ox training on local RTX 2050 GPU and train 416px variant.
- [ ] **Phase 5: Model Evaluation, Error Analysis & Model Freeze** - Independent test-split evaluation, confusion matrix/failure analysis, and official model freeze by Sun 25 Oct.
- [ ] **Phase 6: Edge Quantization & INT8 Accuracy Retention** - Static INT8 quantization (ONNX Runtime & NCNN) and evaluate accuracy retention loss.
- [ ] **Phase 7: Physical Raspberry Pi Benchmark Suite & Tradeoff Analysis** - Full edge benchmark matrix on Pi 4, runtime decision lock (DEC-01), and tradeoff charts.
- [ ] **Phase 8: End-to-End Inference CLI, Demo & Astrodynamics Layer** - Multi-backend `src/infer.py`, interactive demo app, independent SGP4 panel, and real-image collection.
- [ ] **Phase 9: Real-Image Validation, Final Reporting & Project Delivery** - Qualitative domain-gap report, FPGA feasibility note, slides, fallback video, and `v1.0` tag.

---

## Phase Details

### Phase 1: Foundations, Evaluation Harness & Day-1 Audit

**Goal**: Establish rigorous independent evaluation, audit SPARK data access and class balance, set up the Raspberry Pi testbed, and clean repository hygiene.  
**Depends on**: Nothing (first phase)  
**Requirements**: DATA-01, EVAL-01, EVAL-02, EDGE-01, DELV-01  
**Success Criteria** (what must be TRUE):

1. `src/eval.py` exists, executes independently of training, and writes standardized `metrics.json` reproducing SwissCube `best.pt` metrics (0.9903 mAP@0.5, 0.9211 mAP@0.5:0.95).
2. SPARK CSV label audit is complete; class row counts (including debris) are verified and taxonomy decision is logged in `docs/decisions.md`.
3. Physical Raspberry Pi 4 is powered, reachable via network, and has Python 3.10+, `onnxruntime`, and `ncnn` installed.
4. Repository hygiene is resolved: `requirements-edge.txt` created, missing packages (`skyfield`, `psutil`, `ncnn`) added to host requirements, separate `swisscube.yaml` and `spark.yaml` configured, and cached TLE file (`gp.php`) gitignored.
   **Plans**: 3 plans

Plans:

- [x] 01-01: Implement `src/eval.py` and verify against SwissCube `best.pt`.
- [x] 01-02: Count SPARK CSV class rows, audit debris presence, and record taxonomy in `docs/decisions.md`.
- [x] 01-03: Prepare Raspberry Pi 4 environment and resolve repository hygiene (`requirements-edge.txt`, YAML configs, `.gitignore`).

---

### Phase 2: SPARK Dataset Preparation & Stratified Subsetting

**Goal**: Create a frozen, leak-free, class-balanced SPARK subset formatted for YOLO at 1024ox, with strict training-only photometric augmentation.  
**Depends on**: Phase 1  
**Requirements**: DATA-02, DATA-03  
**Success Criteria** (what must be TRUE):

1. SPARK Subset B (~24k train, 3k val, plus held-out test split) is generated at 1024ox resolution with majority classes subsampled to roughly 2–3:1.
2. Train/val/test splits have zero sequence leakage or near-duplicates across splits.
3. Photometric augmentation (`src/data/augment.py`) applies exclusively to the train split, generating visual preview grids without altering validation or test sets.
   **Plans**: 2 plans

Plans:

- [ ] 02-01: Generate balanced SPARK Subset B at 1024ox with sequence/near-duplicate validation and frozen test split.
- [ ] 02-02: Wire photometric augmentation pipeline with configuration logging and train-only isolation checks.

---

### Phase 3: Baseline Edge Validation on Raspberry Pi

**Goal**: Measure actual edge hardware inference latency and memory on physical Raspberry Pi 4 using the existing SwissCube models.  
**Depends on**: Phase 1  
**Requirements**: EDGE-02, EDGE-03  
**Success Criteria** (what must be TRUE):

1. `edge/benchmark.py` is upgraded to execute $\ge 10$ discarded warm-ups, $\ge 100$ timed runs, 4 fixed threads, and measures peak process Resident Set Size (RSS).
2. Existing SwissCube ONNX and NCNN models run on the physical Raspberry Pi 4 by Thu 15 Oct, logging real baseline latency (mean, median, p95) and memory.
   **Plans**: 2 plans

Plans:

- [ ] 03-01: Refactor `edge/benchmark.py` to enforce the rigorous edge evaluation protocol.
- [ ] 03-02: Deploy SwissCube model exports to Raspberry Pi 4, execute benchmarks, and log initial baseline edge numbers.

---

### Phase 4: SPARK Model Training & Optimization

**Goal**: Train the primary YOLOv8-Nano detector on SPARK Subset B at 1024ox and train the 416px comparison variant on the local RTX 2050 GPU.  
**Depends on**: Phase 2  
**Requirements**: MODL-01, MODL-02  
**Success Criteria** (what must be TRUE):

1. A 2–3 epoch dry run on local GPU measures real epoch duration and validates RAM/disk cache behavior.
2. Primary YOLOv8-Nano model completes 30–40 epochs overnight at 1024ox (`degrees=0, flipud=0`) with per-epoch checkpointing.
3. 416px comparison variant completes training on the identical subset and splits.
   **Plans**: 2 plans

Plans:

- [ ] 04-01: Execute dry run, calibrate cache mode, and run full overnight 1024ox SPARK training on local RTX 2050 GPU.
- [ ] 04-02: Train 416px comparison variant for latency-vs-accuracy tradeoff analysis.

---

### Phase 5: Model Evaluation, Error Analysis & Model Freeze

**Goal**: Perform independent evaluation and deep error analysis on the SPARK models, locking the candidate model by Sun 25 Oct.  
**Depends on**: Phase 4  
**Requirements**: EVAL-03, MODL-03  
**Success Criteria** (what must be TRUE):

1. `src/eval.py` evaluates 1024ox and 416px models on validation and test splits, producing standardized `metrics.json`.
2. Error analysis report is generated: per-class precision/recall, object-size bins, confusion matrices, and failure gallery.
3. Candidate model is officially frozen by Sun 25 Oct with Git tag `model-freeze`, weights archived with SHA256 hashes, and metrics committed.
   **Plans**: 2 plans

Plans:

- [ ] 05-01: Run `src/eval.py` across trained SPARK models and generate detailed error analysis documentation.
- [ ] 05-02: Select best model, record verification hashes, and tag `model-freeze` in Git.

---

### Phase 6: Edge Quantization & INT8 Accuracy Retention

**Goal**: Quantize the frozen model to INT8 across ONNX Runtime and NCNN routes, measuring accuracy retention loss on the test split.  
**Depends on**: Phase 5  
**Requirements**: EDGE-04, EDGE-05  
**Success Criteria** (what must be TRUE):

1. Post-training static INT8 quantization is implemented for ONNX Runtime using 300–500 representative calibration images.
2. NCNN INT8 quantization pipeline is built using NCNN calibration tools.
3. `src/eval.py` evaluates FP32 vs INT8 models on the held-out test split, confirming retention loss is $\le 3\text{ points}$ mAP@0.5 (or documenting fallbacks).
   **Plans**: 2 plans

Plans:

- [ ] 06-01: Build INT8 calibration and quantization pipeline for ONNX Runtime and NCNN.
- [ ] 06-02: Measure test-split accuracy retention between FP32 and INT8 exports using `src/eval.py`.

---

### Phase 7: Physical Raspberry Pi Benchmark Suite & Tradeoff Analysis

**Goal**: Execute comprehensive multi-model edge benchmarking on the Raspberry Pi 4 and generate accuracy-vs-latency-vs-size tradeoff charts.  
**Depends on**: Phase 6  
**Requirements**: EVAL-04, EDGE-06, EDGE-07  
**Success Criteria** (what must be TRUE):

1. Raspberry Pi 4 executes $\ge 100$ timed runs for FP32 and INT8 models across NCNN and ONNX Runtime at 1024ox and 416px.
2. Full edge benchmark table is populated (p50/p95 latency, peak RAM, file size $\le 5\text{ MB}$, throughput FPS).
3. Decision DEC-01 (NCNN vs ONNX Runtime) is officially locked based on Pi 4 data.
4. Automated script generates accuracy vs latency vs size tradeoff tables and publication-ready charts.
   **Plans**: 2 plans

Plans:

- [ ] 07-01: Run full edge benchmark suite on Raspberry Pi 4 across models, resolutions, and runtimes.
- [ ] 07-02: Lock DEC-01 and generate automated accuracy-vs-latency-vs-size tradeoff charts.

---

### Phase 8: End-to-End Inference CLI, Demo & Astrodynamics Layer

**Goal**: Build a unified inference CLI, an interactive demo application, integrate the independent SGP4 orbital tracking layer, and collect real space images.  
**Depends on**: Phase 7  
**Requirements**: DATA-04, PROD-01, PROD-02, PROD-03  
**Success Criteria** (what must be TRUE):

1. `src/infer.py` runs seamlessly with PyTorch, ONNX, and NCNN backends, outputting annotated visual overlays and structured JSON.
2. Interactive demo (Gradio or notebook) allows users to upload images and inspect detections, class confidence, inference latency, and model size.
3. Independent SGP4 TLE orbital tracking module (`src/data/tle_pipeline.py`) is integrated into the demo as an informational catalog awareness panel (DEC-07).
4. Small real-world space imagery set (~20–30 public images) is assembled with recorded licenses.
   **Plans**: 2 plans

Plans:

- [ ] 08-01: Develop `src/infer.py` with multi-backend execution and assemble qualitative real-image set.
- [ ] 08-02: Build interactive demo interface integrating detector inference and independent SGP4 awareness panel.

---

### Phase 9: Real-Image Validation, Final Reporting & Project Delivery

**Goal**: Execute qualitative real-image domain-gap analysis, author FPGA feasibility note, finalize report and presentation, and deliver tagged v1.0 release.  
**Depends on**: Phase 8  
**Requirements**: PROD-04, DELV-02, DELV-03, DELV-04, DELV-05  
**Success Criteria** (what must be TRUE):

1. Qualitative real-image validation report documents synthetic-to-real domain gap and failure modes.
2. Analytical 1–2 page FPGA feasibility note is authored using measured FLOPs and published TinyML results.
3. Comprehensive technical report, presentation slide deck, one-page summary, and fallback video are finalized.
4. Repository is cleaned, verified from a clean clone, and tagged `v1.0` in Git with reproduction instructions.
   **Plans**: 2 plans

Plans:

- [ ] 09-01: Perform real-image qualitative validation and draft FPGA feasibility note.
- [ ] 09-02: Finalize technical report, slide deck, demo video, and tag repository `v1.0`.

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

| Phase                                                        | Plans Complete | Status      | Completed |
| ------------------------------------------------------------ | -------------- | ----------- | --------- |
| 1. Foundations, Evaluation Harness & Day-1 Audit             | 0/3            | Not started | -         |
| 2. SPARK Dataset Preparation & Stratified Subsetting         | 0/2            | Not started | -         |
| 3. Baseline Edge Validation on Raspberry Pi                  | 0/2            | Not started | -         |
| 4. SPARK Model Training & Optimization                       | 0/2            | Not started | -         |
| 5. Model Evaluation, Error Analysis & Model Freeze           | 0/2            | Not started | -         |
| 6. Edge Quantization & INT8 Accuracy Retention               | 0/2            | Not started | -         |
| 7. Physical Raspberry Pi Benchmark Suite & Tradeoff Analysis | 0/2            | Not started | -         |
| 8. End-to-End Inference CLI, Demo & Astrodynamics Layer      | 0/2            | Not started | -         |
| 9. Real-Image Validation, Final Reporting & Project Delivery | 0/2            | Not started | -         |

---

_Roadmap created: 2026-10-12_  
_Last updated: 2026-10-12 after initialization_
