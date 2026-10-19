# Requirements: Edge-Deployable Space Debris Detection System

**Defined:** 2026-10-12  
**Core Value:** An honest, reproducible accuracy-versus-deployability tradeoff (mAP, latency, model size, FLOPs) measured on actual edge hardware rather than isolated leaderboard metrics.

## v1 Requirements

Requirements for the 4-week delivery window (12 Oct – 8 Nov 2026). Each requirement is testable and maps directly to roadmap phases.

### Data & Preprocessing (DATA)

- [ ] **DATA-01**: Complete Day-1 audit of full SPARK CSV label rows for debris class; confirm 2-class vs 10-class taxonomy in `docs/decisions.md` (FR-D3, F1).
- [ ] **DATA-02**: Build balanced SPARK Subset B (~24k train, 3k val, plus held-out test split) resized to 1024ox with sequence/near-duplicate checks (FR-D1, DEC-06).
- [ ] **DATA-03**: Verify photometric augmentation pipeline (`src/data/augment.py`) applies strictly to training data, logging parameters per run (FR-D2, DEC-03).
- [ ] **DATA-04**: Curate small qualitative real-world space imagery set (~20–30 public / SPEED+ images) with licenses documented for Week 4 validation (FR-D4).

### Evaluation & Tooling (EVAL)

- [ ] **EVAL-01**: Implement standalone evaluation harness `src/eval.py` outputting `metrics.json` (mAP50, mAP50-95, per-class P/R, size, params, FLOPs) (FR-M3, F2).
- [ ] **EVAL-02**: Verify `src/eval.py` on existing SwissCube `best.pt` checkpoint to confirm it reproduces documented baseline metrics (0.9903 / 0.9211) (F2).
- [ ] **EVAL-03**: Execute structured error analysis on SPARK validation split (per-class P/R, object-size bins, confusion matrix, failure gallery) (FR-M4).
- [ ] **EVAL-04**: Implement automated reporting script to generate results tables and accuracy-vs-latency-vs-size tradeoff charts from `metrics.json` and benchmark logs (FR-P4).

### Model Training & Architecture (MODL)

- [ ] **MODL-01**: Train baseline YOLOv8-Nano on SPARK Subset B at 1024ox (`degrees=0, flipud=0`) on local RTX 2050 GPU with overnight checkpointing (FR-M1, F4, F7).
- [ ] **MODL-02**: Train comparison variant at 416px on the same subset and test split to evaluate latency vs accuracy scaling (FR-M2, DEC-06).
- [ ] **MODL-03**: Execute official model freeze by Sun 25 Oct (Git tag `model-freeze`, weights archived on shared storage with SHA256 hashes, `metrics.json` recorded) (FR-M5, DEC-10).

### Edge Optimization & Benchmarking (EDGE)

- [ ] **EDGE-01**: Prepare physical Raspberry Pi 4 environment (OS, Python 3.10+, `onnxruntime`, `ncnn` wheels) and confirm hardware availability (FR-E3, F6).
- [ ] **EDGE-02**: Refactor `edge/benchmark.py` protocol to enforce $\ge 10$ warm-up runs, $\ge 100$ timed runs, fixed 4 threads, mean/median/p95 latency, and peak resident RSS memory profiling (FR-E3, F10).
- [ ] **EDGE-03**: Benchmark SwissCube baseline ONNX and NCNN exports on physical Raspberry Pi 4 by Thu 15 Oct to establish real edge numbers (FR-E3, F6).
- [ ] **EDGE-04**: Implement INT8 post-training quantization pipeline (ONNX Runtime static INT8 & NCNN calibration route) using 300–500 calibration images (FR-E2, DEC-09).
- [ ] **EDGE-05**: Evaluate FP32 vs INT8 accuracy retention on held-out test split using `src/eval.py`, verifying retention drop is within $\le 3\text{ points}$ mAP@0.5 (FR-E4, NFR-2).
- [ ] **EDGE-06**: Execute full Raspberry Pi 4 benchmark suite on frozen model (p50/p95 latency, peak RAM, file size $\le 5\text{ MB}$, NCNN vs ONNX Runtime) (FR-E3, FR-E5, NFR-3, NFR-4).
- [ ] **EDGE-07**: Lock edge runtime decision DEC-01 (NCNN vs ONNX Runtime) based on physical Raspberry Pi 4 benchmark evidence (DEC-01).

### Product, Demo & Contextual Layer (PROD)

- [ ] **PROD-01**: Implement end-to-end CLI inference tool `src/infer.py` with multi-backend switching (`--backend pytorch|onnx|ncnn`) outputting annotated images and structured JSON (FR-P1).
- [ ] **PROD-02**: Build interactive web demo (Gradio or Jupyter notebook) displaying uploaded images, bounding boxes, class confidences, inference latency, and model size (FR-P2).
- [ ] **PROD-03**: Maintain independent SGP4 orbital propagation module (`src/data/tle_pipeline.py`) as an optional contextual SSA catalog awareness panel (FR-X1, DEC-07).
- [ ] **PROD-04**: Record fallback demonstration video in case of live hardware failure during final evaluation (FR-P3).

### Delivery, Documentation & Hygiene (DELV)

- [ ] **DELV-01**: Resolve repository hygiene: create `requirements-edge.txt`, add missing packages (`skyfield`, `psutil`, `ncnn`), create separate `swisscube.yaml` and `spark.yaml`, and gitignore cached TLE file (F11).
- [ ] **DELV-02**: Perform qualitative real-image validation analysis on the frozen model, detailing domain-gap failure modes (FR-V1).
- [ ] **DELV-03**: Author 1–2 page analytical FPGA feasibility note based on measured FLOPs and published TinyML-on-FPGA literature (FR-E6).
- [ ] **DELV-04**: Deliver comprehensive technical report, presentation slide deck, and one-page executive summary (FR-R1, FR-R2).
- [ ] **DELV-05**: Tag clean Git repository milestones (`week1-checkpoint`, `model-freeze`, `core-complete`, `v1.0`) with verifiable reproduction commands in `README.md` (FR-R1, NFR-5, NFR-9).

## v2 Requirements (Deferred)

- **V2-01**: Synthetic rocket body class generation via Blender 3D CAD rendering and star-field compositing.
- **V2-02**: Multi-frame optical flow tracking for temporal velocity and motion consistency.
- **V2-03**: Hardware synthesis and deployment on AMD/Xilinx Zynq FPGA platform.
- **V2-04**: Knowledge distillation and structured network pruning for extreme sub-1MB models.

## Out of Scope

| Feature                                      | Reason                                                                                                                                           |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Direct image-to-orbit conjunction claims** | 2D bounding boxes do not measure 3D range, state vectors, or velocity; SGP4 is kept strictly as an independent catalog awareness layer (DEC-07). |
| **Optical flow & multi-frame tracking**      | Out of scope for 4-week detect→quantize→benchmark critical path.                                                                                 |
| **Frame-based collision-risk heuristics**    | Omitted to focus on core detector optimization and physical edge deployment.                                                                     |
| **Physical FPGA hardware build**             | Toolchains and RTL design exceed 4-week window; short analytical feasibility note only (FR-E6).                                                  |
| **Training on real SSN radar/optical data**  | Restricted access; synthetic training with qualitative real validation only.                                                                     |
| **Heavy model backbones (YOLOv8x, DETR)**    | Prohibitive training time on 4GB GPU and exceeds edge power/memory budgets.                                                                      |

## Traceability

| Requirement | Phase   | Status  |
| ----------- | ------- | ------- |
| DATA-01     | Phase 1 | Pending |
| DATA-02     | Phase 2 | Pending |
| DATA-03     | Phase 2 | Pending |
| DATA-04     | Phase 8 | Pending |
| EVAL-01     | Phase 1 | Pending |
| EVAL-02     | Phase 1 | Pending |
| EVAL-03     | Phase 5 | Pending |
| EVAL-04     | Phase 7 | Pending |
| MODL-01     | Phase 4 | Pending |
| MODL-02     | Phase 4 | Pending |
| MODL-03     | Phase 5 | Pending |
| EDGE-01     | Phase 1 | Pending |
| EDGE-02     | Phase 3 | Pending |
| EDGE-03     | Phase 3 | Pending |
| EDGE-04     | Phase 6 | Pending |
| EDGE-05     | Phase 6 | Pending |
| EDGE-06     | Phase 7 | Pending |
| EDGE-07     | Phase 7 | Pending |
| PROD-01     | Phase 8 | Pending |
| PROD-02     | Phase 8 | Pending |
| PROD-03     | Phase 8 | Pending |
| PROD-04     | Phase 9 | Pending |
| DELV-01     | Phase 1 | Pending |
| DELV-02     | Phase 9 | Pending |
| DELV-03     | Phase 9 | Pending |
| DELV-04     | Phase 9 | Pending |
| DELV-05     | Phase 9 | Pending |

**Coverage:**

- v1 requirements: 26 total
- Mapped to phases: 26
- Unmapped: 0 ✓

---

_Requirements defined: 2026-10-12_  
_Last updated: 2026-10-12 after initial definition_
