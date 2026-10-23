# Roadmap: Edge-Deployable Space Debris Detection System

## Overview

This roadmap defines the 4-week execution trajectory (12 Oct – 8 Nov 2026) for the Edge-Deployable Space Debris Detection System (Problem Statement #05). Starting from an audited brownfield repository with a SwissCube baseline, the project systematically resolved critical evidence gaps (independent evaluation harness, SPARK-2021 ground-truth debris audit, split leakage investigation), executed reproducible 11-class YOLOv8-Nano deep training (180,000 training images at 1024px) on an NVIDIA Quadro GV100 workstation GPU ([DEC-06](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-06-compute-hardware-allocation-strategy)), evaluated multi-scale Pareto trade-offs (640px vs 800px vs 1024px) on a single-machine 16-Core Xeon CPU ([DEC-07](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-07-model-input-resolution--multi-scale-trade-off-strategy)), prepares INT8 post-training quantization, benchmarks inference latency and peak memory on physical edge hardware (Raspberry Pi 4 / Raspberry Pi 5), and integrates an end-to-end CLI and interactive demonstrator with an independent SGP4 orbital catalog awareness layer ([DEC-05](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-05-tle-independent-catalog-layer--uncataloged-debris-limitation)).

## Phases

- [x] **Phase 1: Foundations, Evaluation Harness & Day-1 Audit** - Standalone [src/eval.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py), SPARK CSV audit ([DEC-04](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-04-spark-dataset-annotation-audit--debris-taxonomy-resolution)), repo hygiene, and Pi environment verification.
- [x] **Phase 2: SPARK-2021 Dataset Preparation & Stratified Subsetting** - Ingest 150k images with 25k debris annotations ([DEC-10](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-10-transition-to-spark-2021-11-class-dataset-synthetic-debris-integration)), apply train-only photometric augmentation (180k images), and audit split leakage.
- [x] **Phase 3: Baseline Edge Validation & Multi-Scale Single-Machine Benchmarking** - Upgraded [edge/benchmark.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/edge/benchmark.py) (≥100 runs, peak RSS), single-machine 16-Core Xeon CPU benchmarks at 640px and 1024px, and Pi 4 / Pi 5 testbed preparation.
- [x] **Phase 4: SPARK-2021 Model Training & Optimization** - Execute 50-epoch YOLOv8-Nano deep training on 32GB Quadro GV100 at 1024px resolution across 180,000 augmented images.
- [x] **Phase 5: Multi-Scale Evaluation, Error Analysis & Model Freeze** - Multi-scale test evaluation (640px, 800px, 1024px), confusion matrix, 5-CAD synthetic debris analysis, and model freeze.
- [ ] **Phase 6: Edge Quantization & INT8 Accuracy Retention** - Static INT8 post-training quantization (ONNX Runtime & NCNN) and evaluate accuracy retention loss on test split.
- [ ] **Phase 7: Physical Raspberry Pi 4 / Pi 5 Benchmark Suite & Tradeoff Analysis** - Full edge benchmark matrix on physical Raspberry Pi 4 and Pi 5, runtime decision lock ([DEC-01](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#1-edge-deployment-framework-tflite-vs-onnx-vs-ncnn)), and tradeoff charts.
- [ ] **Phase 8: End-to-End Inference CLI, Demo & Astrodynamics Layer** - Multi-backend [src/infer.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/infer.py), interactive demo application, independent SGP4 awareness panel ([DEC-05](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-05-tle-independent-catalog-layer--uncataloged-debris-limitation)), and real-image collection.
- [ ] **Phase 9: Real-Image Validation, Final Reporting & Project Delivery** - Qualitative synthetic-to-real domain-gap report, FPGA feasibility note, slides, fallback video, and `v1.0` tag.

---

## Phase Details

### Phase 1: Foundations, Evaluation Harness & Day-1 Audit

**Goal**: Establish rigorous independent evaluation, audit SPARK data access and class balance, set up the Raspberry Pi testbed, and clean repository hygiene.  
**Depends on**: Nothing (first phase)  
**Requirements**: DATA-01, EVAL-01, EVAL-02, EDGE-01, DELV-01  
**Success Criteria** (what must be TRUE):

1. [src/eval.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py) exists, executes independently of training, and writes standardized `metrics.json` reproducing SwissCube `best.pt` metrics (0.9903 mAP@0.5, 0.9211 mAP@0.5:0.95).
2. SPARK CSV label audit is complete; class row counts (including debris) are verified and taxonomy decision is logged in [docs/decisions.md](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md).
3. Physical Raspberry Pi 4 environment is verified with Python 3.10+, `onnxruntime`, and `ncnn` installation requirements documented.
4. Repository hygiene is resolved: `requirements-edge.txt` created, missing packages (`skyfield`, `psutil`, `ncnn`) added to host requirements, separate `swisscube.yaml` and `spark2021.yaml` configured, and cached TLE file (`gp.php`) gitignored.

**Plans**: 3 plans (Complete)

- [x] 01-01: Implement [src/eval.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py) and verify against SwissCube `best.pt`.
- [x] 01-02: Count SPARK CSV class rows, audit debris presence, and record taxonomy in [docs/decisions.md](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md) ([DEC-04](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-04-spark-dataset-annotation-audit--debris-taxonomy-resolution)).
- [x] 01-03: Prepare Raspberry Pi environment specifications and resolve repository hygiene (`requirements-edge.txt`, YAML configs, `.gitignore`).

---

### Phase 2: SPARK-2021 Dataset Preparation & Stratified Subsetting

**Goal**: Ingest official SPARK-2021 11-class dataset with 25k synthetic debris annotations ([DEC-10](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-10-transition-to-spark-2021-11-class-dataset-synthetic-debris-integration)), apply train-only photometric augmentation, and audit split leakage across orbital imagery.  
**Depends on**: Phase 1  
**Requirements**: DATA-02, DATA-03  
**Success Criteria** (what must be TRUE):

1. SPARK-2021 dataset (150,000 raw images: 90k train, 30k val, 30k test) is extracted, audited, and converted to normalized YOLO format in `data/processed/spark2021/` with verified 25,000 synthetic debris annotations across 5 geometric shapes.
2. Photometric augmentation pipeline ([src/data/augment.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/data/augment.py)) expands the training set to 180,000 images (`aug1_*`) with zero augmentation leakage into validation or test splits.
3. Perceptual hash audit ([scripts/check_split_leakage.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/scripts/check_split_leakage.py)) evaluates train vs test similarity, documents solid black background hash collapse, and formally records that split leakage is not ruled out pending bounding-box ROI hashing.

**Plans**: 3 plans (Complete)

- [x] 02-01: Extract SPARK-2021 archives, audit all 150,000 labels ([scripts/audit_spark2021_annotations.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/scripts/audit_spark2021_annotations.py)), and convert row/col coordinates to YOLO format ([scripts/convert_spark2021_to_yolo.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/scripts/convert_spark2021_to_yolo.py)).
- [x] 02-02: Run photometric data augmentation on `train/` split only ([src/data/augment.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/data/augment.py)), expanding training corpus to 180,000 images while maintaining strict val/test isolation.
- [x] 02-03: Execute perceptual dHash audit ([scripts/check_split_leakage.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/scripts/check_split_leakage.py)), analyze black-background collision dynamics, and export structured verification findings to `outputs/audits/split_leakage_audit.json`.

---

### Phase 3: Baseline Edge Validation & Multi-Scale Single-Machine Benchmarking

**Goal**: Enforce rigorous benchmark protocol (≥10 warmup runs, ≥100 timed runs, whole-process peak RSS, fixed threads) and measure multi-scale CPU inference performance on an identical 16-Core Xeon host while preparing physical Pi 4 / Pi 5 testbeds.  
**Depends on**: Phase 1, Phase 2  
**Requirements**: EDGE-02, EDGE-03  
**Success Criteria** (what must be TRUE):

1. [edge/benchmark.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/edge/benchmark.py) enforces ≥10 warmups, ≥100 timed iterations, 4 CPU threads, peak RSS tracking via `psutil`, and auto-detects static ONNX tensor shapes.
2. Single-machine 16-Core Xeon CPU benchmark establishes the multi-scale baseline: 640px ONNX FP32 runs at 23.70 ms (129.34 MB RSS) and 1024px ONNX FP32 runs at 65.20 ms (207.36 MB RSS).
3. Physical edge SBCs (Raspberry Pi 4 Model B and Raspberry Pi 5) are targeted for physical hardware deployment to validate the PRD latency gate ($\le 1000\text{ ms}$).

**Plans**: 2 plans (Complete)

- [x] 03-01: Upgrade [edge/benchmark.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/edge/benchmark.py) to research-grade protocol and execute single-machine 16-Core Xeon benchmarks for 640px, 800px, and 1024px ONNX models.
- [x] 03-02: Deploy ONNX models to physical Raspberry Pi 5 testbed and validate PRD $\le 1000\text{ ms}$ gate (640px: 144.83 ms, 800px: 285.15 ms, 1024px: 515.19 ms).

---

### Phase 4: SPARK-2021 Model Training & Optimization

**Goal**: Train the primary 11-class YOLOv8-Nano detector on the 180,000-image SPARK-2021 training set at 1024px on the NVIDIA Quadro GV100 workstation GPU ([DEC-06](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-06-compute-hardware-allocation-strategy)).  
**Depends on**: Phase 2  
**Requirements**: MODL-01, MODL-02  
**Success Criteria** (what must be TRUE):

1. Primary YOLOv8-Nano completes 50 epochs on Quadro GV100 (32GB VRAM) at 1024px (`batch=16`, `imgsz=1024`, `mosaic=0.5` closing at epoch 40, `amp=True`) over ~34 hours (122,183 seconds).
2. Per-epoch checkpointing records validation loss and saves `experiments/spark2021_yolov8n_50e/weights/best.pt` and `last.pt`.
3. Checkpoint exports verified for FP32 ONNX (`best.onnx`) and NCNN (`best_ncnn_model/`) formats via [edge/export.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/edge/export.py).

**Plans**: 2 plans (Complete)

- [x] 04-01: Execute 50-epoch deep training run on NVIDIA Quadro GV100 at 1024px with checkpointing and learning rate scheduling.
- [x] 04-02: Export weights to FP32 ONNX and NCNN formats with automated forward-pass sanity checks.

---

### Phase 5: Multi-Scale Evaluation, Error Analysis & Model Freeze

**Goal**: Perform independent evaluation across multi-scale resolutions (640px, 800px, 1024px) on the held-out 30,000-image test split, conduct detailed confusion matrix analysis, document synthetic debris characteristics, and freeze the candidate model.  
**Depends on**: Phase 4  
**Requirements**: EVAL-03, MODL-03  
**Success Criteria** (what must be TRUE):

1. Standalone evaluation with [src/eval.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py) evaluates the held-out test split (30,000 images) across 640px, 800px, and 1024px, establishing the empirical Pareto frontier ([DEC-07](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-07-model-input-resolution--multi-scale-trade-off-strategy)):
   - 1024px: 99.02% mAP@50, **86.98% mAP@50-95** (Debris: 85.39%)
   - 800px: 98.94% mAP@50, **86.47% mAP@50-95** (Debris: 84.31%)
   - 640px: 98.74% mAP@50, **85.06% mAP@50-95** (Debris: 82.81%)
2. Error analysis report documents confusion matrix, median bounding-box dimensions, TRMM background misclassifications, and the 5-CAD model synthetic debris representation.
3. Candidate model is officially frozen with weights archived in `experiments/spark2021_yolov8n_50e/weights/` and documentation committed.

**Plans**: 2 plans (Complete)

- [x] 05-01: Run multi-scale evaluation (640px, 800px, 1024px) on 30k test split using [src/eval.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py) and generate comprehensive technical report ([docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md)).
- [x] 05-02: Complete confusion matrix analysis, document synthetic debris geometric constraints, and freeze model weights.

---

### Phase 6: Edge Quantization & INT8 Accuracy Retention

**Goal**: Quantize the frozen model to INT8 across ONNX Runtime and NCNN routes, measuring accuracy retention loss on the held-out test split.  
**Depends on**: Phase 5  
**Requirements**: EDGE-04, EDGE-05  
**Success Criteria** (what must be TRUE):

1. Post-training static INT8 quantization is implemented for ONNX Runtime using 300–500 representative calibration images from the training split.
2. NCNN INT8 quantization pipeline is built using NCNN calibration tools (`ncnn2table`).
3. [src/eval.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py) evaluates FP32 vs INT8 models on the held-out test split, confirming retention loss is $\le 3\text{ points}$ mAP@0.5 (or documenting fallbacks).

**Plans**: 2 plans

- [ ] 06-01: Build INT8 calibration and quantization pipeline for ONNX Runtime and NCNN.
- [ ] 06-02: Measure test-split accuracy retention between FP32 and INT8 exports using [src/eval.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py).

---

### Phase 7: Physical Raspberry Pi 4 / Pi 5 Benchmark Suite & Tradeoff Analysis

**Goal**: Execute comprehensive multi-model edge benchmarking on physical Raspberry Pi 4 and Raspberry Pi 5 hardware, lock runtime decision [DEC-01](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#1-edge-deployment-framework-tflite-vs-onnx-vs-ncnn), and generate accuracy-vs-latency-vs-size tradeoff charts.  
**Depends on**: Phase 6  
**Requirements**: EVAL-04, EDGE-06, EDGE-07  
**Success Criteria** (what must be TRUE):

1. Raspberry Pi 4 (Cortex-A72) and Raspberry Pi 5 (Cortex-A76) execute $\ge 100$ timed runs for FP32 and INT8 models across NCNN and ONNX Runtime at 640px and 1024px.
2. Full edge benchmark table is populated (p50/p95 latency, peak RAM, file size $\le 5\text{ MB}$, throughput FPS), validating against the PRD $\le 1000\text{ ms}$ gate.
3. Decision [DEC-01](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#1-edge-deployment-framework-tflite-vs-onnx-vs-ncnn) (NCNN vs ONNX Runtime) is officially locked based on physical SBC data.
4. Publication-ready accuracy vs latency vs size tradeoff tables and Pareto charts are generated.

**Plans**: 2 plans (1 Complete, 1 In Progress)

- [x] 07-01: Run full edge benchmark suite on physical Raspberry Pi 5 across resolutions (640px: 144.83 ms, 800px: 285.15 ms, 1024px: 515.19 ms).
- [ ] 07-02: Lock [DEC-01](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#1-edge-deployment-framework-tflite-vs-onnx-vs-ncnn) and generate automated accuracy-vs-latency-vs-size tradeoff charts.

---

### Phase 8: End-to-End Inference CLI, Demo & Astrodynamics Layer

**Goal**: Build a unified inference CLI, an interactive demo application, integrate the independent SGP4 orbital tracking catalog layer ([DEC-05](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-05-tle-independent-catalog-layer--uncataloged-debris-limitation)), and assemble a qualitative real-image set.  
**Depends on**: Phase 7  
**Requirements**: DATA-04, PROD-01, PROD-02, PROD-03  
**Success Criteria** (what must be TRUE):

1. [src/infer.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/infer.py) runs seamlessly across PyTorch, ONNX, and NCNN backends, outputting annotated visual overlays and structured JSON.
2. Interactive demo (Gradio or notebook) allows users to upload space imagery and inspect detections, class confidence, inference latency, and model size.
3. Independent SGP4 TLE orbital tracking module ([src/data/tle_pipeline.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/data/tle_pipeline.py)) is integrated into the demo as an informational catalog awareness panel for known spacecraft ([DEC-05](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md#dec-05-tle-independent-catalog-layer--uncataloged-debris-limitation)).
4. Small real-world space imagery set (~20–30 public images) is assembled with recorded source licenses.

**Plans**: 2 plans

- [ ] 08-01: Develop [src/infer.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/infer.py) with multi-backend execution and assemble qualitative real-image set.
- [ ] 08-02: Build interactive demo interface integrating detector inference and independent SGP4 awareness panel.

---

### Phase 9: Real-Image Validation, Final Reporting & Project Delivery

**Goal**: Execute qualitative real-image domain-gap analysis, author FPGA feasibility note, finalize report and presentation, and deliver tagged v1.0 release.  
**Depends on**: Phase 8  
**Requirements**: PROD-04, DELV-02, DELV-03, DELV-04, DELV-05  
**Success Criteria** (what must be TRUE):

1. Qualitative real-image validation report documents synthetic-to-real domain gap and visual failure modes.
2. Analytical 1-2 page FPGA feasibility note is authored using measured FLOPs and published TinyML results.
3. Comprehensive technical report, presentation slide deck, one-page summary, and fallback video are finalized.
4. Repository is cleaned, verified from a clean clone, and tagged `v1.0` in Git with reproduction instructions.

**Plans**: 2 plans

- [ ] 09-01: Perform real-image qualitative validation and draft FPGA feasibility note.
- [ ] 09-02: Finalize technical report, slide deck, demo video, and tag repository `v1.0`.

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

| Phase                                                                        | Plans Complete | Status      | Completed    |
| ---------------------------------------------------------------------------- | -------------- | ----------- | ------------ |
| 1. Foundations, Evaluation Harness & Day-1 Audit                             | 3/3            | Complete    | 2026-10-14   |
| 2. SPARK-2021 Dataset Preparation & Stratified Subsetting                     | 3/3            | Complete    | 2026-10-19   |
| 3. Baseline Edge Validation & Multi-Scale Single-Machine Benchmarking        | 2/2            | Complete    | 2026-10-23   |
| 4. SPARK-2021 Model Training & Optimization                                  | 2/2            | Complete    | 2026-10-19   |
| 5. Multi-Scale Evaluation, Error Analysis & Model Freeze                     | 2/2            | Complete    | 2026-10-19   |
| 6. Edge Quantization & INT8 Accuracy Retention                               | 0/2            | Planned     | -            |
| 7. Physical Raspberry Pi 4 / Pi 5 Benchmark Suite & Tradeoff Analysis        | 1/2            | In Progress | -            |
| 8. End-to-End Inference CLI, Demo & Astrodynamics Layer                      | 0/2            | Planned     | -            |
| 9. Real-Image Validation, Final Reporting & Project Delivery                 | 0/2            | Planned     | -            |

---

_Roadmap created: 2026-10-12_  
_Last updated: 2026-10-23 after Raspberry Pi 5 physical edge benchmark validation_
