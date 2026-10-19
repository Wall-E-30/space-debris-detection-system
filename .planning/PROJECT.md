# Edge-Deployable Space Debris Detection System

## What This Is

An end-to-end computer vision and edge intelligence pipeline designed to detect, classify, and evaluate space objects (satellites and space debris) in simulated space imagery. Built for Space Situational Awareness (Problem Statement #05), the system fine-tunes an ultralightweight YOLOv8-Nano detector, quantizes it to INT8, and benchmarks inference latency, memory, and accuracy directly on physical edge hardware (Raspberry Pi 4) alongside an independent catalog-based SGP4 orbital propagation layer.

## Core Value

An honest, reproducible accuracy-versus-deployability tradeoff (mAP, latency, model size, FLOPs) measured on actual edge hardware rather than isolated leaderboard metrics.

## Requirements

### Validated

- ✓ SwissCube BOP dataset to YOLO format conversion and stratified splitter (`src/data/make_labels.py`, `src/data/split_dataset.py`) — existing
- ✓ Photometric space-domain augmentation pipeline with strict train-split isolation (`src/data/augment.py`) — existing
- ✓ YOLOv8-Nano baseline training harness with space-domain geometric transforms and GPU acceleration (`src/train.py`, SwissCube 5-epoch smoke test) — existing
- ✓ Edge model export harness for FP32 ONNX and ARM-optimized NCNN formats with shape validation (`edge/export.py`) — existing
- ✓ Host CPU latency and memory benchmark suite (`edge/benchmark.py`) — existing
- ✓ Independent catalog-based SGP4 orbit propagation module using live CelesTrak TLEs (`src/data/tle_pipeline.py`) — existing

### Active

- [ ] Count label rows in full SPARK CSV to audit debris class presence and resolve 2-class vs space-object taxonomy (FR-D3, F1)
- [ ] Implement standalone evaluation harness `src/eval.py` logging standardized `metrics.json` (mAP50, mAP50-95, per-class P/R, size, params, FLOPs) and verify on SwissCube `best.pt` (FR-M3, F2)
- [ ] Build balanced SPARK subset B (~24k train, 3k val) resized to 1024ox with sequence/near-duplicate checks and populate test split (FR-D1, DEC-06)
- [ ] Prepare physical Raspberry Pi 4 environment (OS, Python, onnxruntime, ncnn wheels) and establish first baseline edge numbers (FR-E3, F6, F10)
- [ ] Resolve repo hygiene: create `requirements-edge.txt`, add missing dependencies (`skyfield`, `psutil`, `ncnn`), create separate `swisscube.yaml` and `spark.yaml`, and gitignore TLE cache (F11)
- [ ] Train YOLOv8-Nano on SPARK at 1024ox (degrees=0, flipud=0) on local RTX 2050 GPU with overnight checkpointing (FR-M1, F4, F7)
- [ ] Train 416px comparison variant on subset B for edge tradeoff analysis (FR-M2, DEC-06)
- [ ] Perform detailed error analysis on SPARK validation split (per-class P/R, object-size bins, confusion matrix, failure gallery) (FR-M4)
- [ ] Freeze model by Sun 25 Oct (Git tag `model-freeze`, weights hashed and archived, metrics logged) (FR-M5, DEC-10)
- [ ] Implement INT8 quantization pipeline (ONNX Runtime static INT8 & NCNN calibration) tested on SwissCube then applied to frozen SPARK model (FR-E2, DEC-09)
- [ ] Measure FP32 vs INT8 accuracy retention on held-out test split (FR-E4)
- [ ] Execute formal Raspberry Pi 4 benchmark suite (≥100 timed runs, mean, p50, p95, peak RSS, FP32 vs INT8, both runtimes) (FR-E3, FR-E5, F10)
- [ ] Build end-to-end CLI inference tool `src/infer.py` supporting PyTorch, ONNX, and NCNN backends (FR-P1)
- [ ] Build interactive demo (Gradio or notebook) with bounding boxes, confidence, latency, and model size; optional independent TLE panel (FR-P2, FR-X1)
- [ ] Generate automated results table and accuracy-vs-latency-vs-size tradeoff charts from experiment metrics (FR-P4)
- [ ] Evaluate qualitative performance on small real-image validation set (~20–30 images) to assess synthetic-to-real domain gap (FR-V1, FR-D4)
- [ ] Write 1–2 page FPGA feasibility note based on measured FLOPs and published TinyML results (FR-E6)
- [ ] Deliver technical report, slide deck, one-page summary, and fallback video (FR-R1, FR-R2, FR-P3)

### Out of Scope

- Direct detection-to-orbit conjunction measurement — 2D bounding boxes carry no range or velocity; SGP4 orbit propagation kept strictly as an independent catalog awareness layer / demo panel (DEC-07)
- Optical flow and multi-frame tracking — omitted to protect 4-week edge deployment critical path
- Frame-based collision-risk heuristic — omitted for 4-week scope
- Real FPGA hardware synthesis — short written feasibility note only
- Training on real Space Surveillance Network (SSN) radar/optical data — restricted access; synthetic training with qualitative real validation only
- Large-model comparisons (e.g. DETR, YOLOv8x) — out of scope for lightweight edge deployment focus
- Knowledge distillation and network pruning — deferred to future work

## Context

- **Timeline**: 4-week compressed execution window (12 Oct – 8 Nov 2026), team of 2: Person A (ML/CV Lead) and Person B (Data/Edge Lead).
- **Hard Deadline**: End of Week 3 (Fri 30 Oct internal / Sun 1 Nov latest) for core pipeline end-to-end completion; Week 4 reserved for real-image validation, report, slides, and buffer.
- **Hardware Profile**: Local training conducted on NVIDIA RTX 2050 GPU (4GB VRAM) laptop workstation; free Kaggle/Colab GPU as secondary backup. Target deployment platform is a physical Raspberry Pi 4 Model B (Quad-core ARM Cortex-A72).
- **Current State**: Initial pipeline validation established on SwissCube dataset (0.9903 mAP@0.5), but gaps exist in independent evaluation (`src/eval.py`), SPARK dataset multi-class training, INT8 quantization, and verified physical Pi benchmarks.

## Constraints

- **Compute**: 4GB GPU VRAM caps training batch size and resolution; requires training at 1024ox and balanced subset selection (~27k images) rather than raw 150k dataset.
- **Edge Budget**: Model size target ≤ 5 MB (NFR-3, evaluated on INT8); Raspberry Pi 4 inference latency target ≤ 1000 ms (NFR-4).
- **Evaluation Discipline**: All headline numbers must come from the held-out test split, evaluated with the standardized `src/eval.py` script.

## Key Decisions

| Decision                                                              | Rationale                                                                                                                                       | Outcome                     |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **DEC-01**: NCNN primary hypothesis, ONNX Runtime fallback            | NCNN showed ~21% latency lead on CPU proxy, but must be validated on physical Pi before locking                                                 | — Pending (Gate Thu 15 Oct) |
| **DEC-02**: YOLOv8-Nano architecture                                  | Lightweight parameter footprint (~3.0M params, ~8 GFLOPs) suited for real-time edge execution                                                   | ✓ Good                      |
| **DEC-03**: Photometric + minimal geometric augmentation              | Prevents box distortion from large rotations; run SPARK baseline with degrees=0, flipud=0                                                       | — Pending                   |
| **DEC-04**: Class balance via subsampling                             | Subsample majority class to 2–3:1 when building subset B rather than tuning complex focal loss                                                  | — Pending                   |
| **DEC-05**: RAM cache / disk cache + AMP                              | Accelerated local GPU training on RTX 2050; verify cache mode based on available RAM                                                            | ✓ Good                      |
| **DEC-06**: 1024ox training/deployment size, 416px comparison variant | Aligns training resolution with edge benchmark resolution; eliminates 1024px vs 1024ox mismatch                                                 | — Pending                   |
| **DEC-07**: TLE module framed as independent catalog layer            | 2D bounding boxes cannot physically measure 3D distance; keep SGP4 module as contextual SSA demo panel without claiming detection-to-orbit link | ✓ Good                      |
| **DEC-08**: NVIDIA GV100 primary training machine                   | Avoids blocking on external GPU allocation; train subset B overnight with margin                                                                | ✓ Good                      |
| **DEC-09**: Build INT8 pipeline on SwissCube checkpoint first         | De-risks the edge quantization toolchain in Week 1 before waiting for SPARK model freeze                                                        | — Pending                   |
| **DEC-10**: Model freeze on Sun 25 Oct                                | Protects Week 3 edge integration and delivery deadlines                                                                                         | — Pending                   |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):

1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):

1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---

_Last updated: 2026-10-12 after initialization_
