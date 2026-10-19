# Project Research Summary

**Project:** Edge-Deployable Space Debris Detection System  
**Domain:** Space Situational Awareness / Edge AI Object Detection & Optimization  
**Researched:** 2026-10-12  
**Confidence:** HIGH

## Executive Summary

The Edge-Deployable Space Debris Detection System addresses Space Situational Awareness (Problem Statement #05) by fine-tuning an ultralightweight YOLOv8-Nano detector on simulated space imagery, quantizing to INT8, and benchmarking physical edge inference on a Raspberry Pi 4 Model B SBC. The system is delivered by a 2-person team (Person A: ML/CV Lead, Person B: Data/Edge Lead) over a compressed 4-week window (12 Oct – 8 Nov 2026).

Initial validation on the SwissCube dataset demonstrated functional pipeline viability (0.9903 mAP@0.5 on host GPU, 165ms NCNN inference on CPU proxy). However, technical auditing revealed critical gaps that must be resolved to ensure project credibility:

1. `src/eval.py` is currently an empty 0-byte file, leaving metric reporting dependent on internal training CSV logs.
2. Training and edge latency were previously measured at mismatched resolutions ($1024\text{px}$ vs $640\text{px}$).
3. The SPARK training subset must be audited for the explicit presence of debris classes.
4. SPARK training must be planned around the local 4GB RTX 2050 GPU using a balanced subset (~27k images at 1024ox) rather than waiting for external cluster allocation.
5. The 2D bounding box detector must not over-claim physical 3D conjunction measurement; SGP4 orbital tracking is strictly maintained as an independent catalog awareness layer.

The core deliverable of this project is not an isolated leaderboard mAP score, but an honest, reproducible accuracy-vs-deployability tradeoff curve across model architectures, image resolutions (1024ox vs 416px), quantization formats (FP32, FP16, INT8), and edge runtimes (NCNN, ONNX Runtime) measured on physical Raspberry Pi 4 hardware.

## Key Findings

### Recommended Stack

- **Core Frameworks:** Ultralytics YOLOv8-Nano (`yolov8n.pt`) fine-tuned via PyTorch with CUDA acceleration on a local RTX 2050 GPU (AMP + dataset caching).
- **Edge Runtimes:** Tencent NCNN as the primary edge runtime hypothesis for ARM Cortex-A72 CPU execution; ONNX Runtime as the robust cross-platform fallback.
- **Orbital Mechanics:** Skyfield for SGP4 propagation of live CelesTrak Two-Line Element (TLE) ephemeris sets.
- **Edge Hardware:** Raspberry Pi 4 Model B (Quad-core ARM Cortex-A72 @ 1.5/1.8 GHz, 4GB/8GB RAM) targeting an inference SLA $\le 1000\text{ ms}$.

### Expected Features

**Must Have (Table Stakes):**

- Standardized `src/eval.py` validation harness outputting `metrics.json` (mAP50, mAP50-95, P/R, size, params, FLOPs).
- Balanced SPARK subset B (~24k train, 3k val) resized to 1024ox with sequence/near-duplicate filtering.
- Dual edge export harness (`edge/export.py`) producing FP32 ONNX and NCNN binaries.
- Physical Raspberry Pi 4 benchmarking harness (`edge/benchmark.py`) with $\ge 100$ timed iterations, warm-up discard, and peak RSS memory profiling.
- Command-line inference script `src/infer.py` with multi-backend switching (PyTorch, ONNX, NCNN).

**Should Have (Competitive Differentiators):**

- Post-training INT8 quantization with measured accuracy retention on the held-out test split (retention drop $\le 3\text{ points}$).
- Multi-resolution tradeoff comparison ($640\text{px}$ vs $416\text{px}$).
- Photometric augmentation pipeline (`src/data/augment.py`) with blur, noise, and exposure jitter applied strictly to training splits.
- Qualitative evaluation on small real-world space imagery set (~20–30 images) to assess synthetic-to-real transfer.
- Analytical FPGA feasibility note (1–2 pages) mapping measured FLOPs to radiation-hardened space platforms.

**Defer (Out of Scope):**

- Direct image-to-orbit conjunction claims (mathematically invalid without range/velocity sensors).
- Optical flow, multi-frame velocity tracking, and frame-based collision heuristics.
- Physical FPGA hardware synthesis.

### Architecture Approach

- **Separation of Workstations:** Host workstation performs data preparation, photometric augmentation, model training, and quantization calibration; edge SBC runs benchmarking and inference validation.
- **Decoupled Metric Contract:** Standardized `metrics.json` decouples model evaluation from training engines and feeds automated tradeoff visualization scripts.
- **Dual-Runtime Edge Fallback:** NCNN is evaluated for maximum ARM throughput, while ONNX Runtime ensures zero-risk delivery if NCNN toolchains encounter issues on ARM Linux.
- **Isolated Astrodynamics Service:** Skyfield SGP4 orbit propagator functions as an independent contextual catalog layer rather than an integrated vision sensor head.

### Critical Pitfalls

1. **Resolution Mismatch:** Enforce DEC-06 ($640\text{px}$ deployment and training standard; $416\text{px}$ comparison variant).
2. **Astrodynamics Over-Claiming:** Enforce DEC-07 (frame SGP4 as independent catalog awareness layer).
3. **GPU Compute Stalling:** Enforce DEC-08 (train balanced Subset B on local 4GB RTX 2050 overnight).
4. **Uncalibrated Benchmarks:** Enforce $\ge 100$ timed runs, warm-ups discarded, 4 threads fixed, and peak RSS memory reporting.
5. **Missing Debris Class:** Audit SPARK CSV immediately on Day 1 (F1); reframe taxonomy honestly if absent.

## Implications for Roadmap

To satisfy the 4-week timeline and enforce fine granularity (7–9 focused phases), the roadmap should be structured into clear chronological workstreams:

### Suggested Phase Breakdown

1. **Phase 1: Foundations, Evaluation Harness & Day-1 Audit**
   - Implement `src/eval.py` and verify against SwissCube `best.pt`.
   - Conduct Day-1 SPARK CSV label audit (F1) and confirm taxonomy.
   - Set up Raspberry Pi 4 environment and verify onnxruntime/ncnn installation.
   - Resolve repository hygiene (`requirements-edge.txt`, gitignore TLE cache, separate configs).

2. **Phase 2: SPARK Dataset Preparation & Stratified Subsetting**
   - Build balanced SPARK Subset B (~24k train, 3k val, plus test split) at $640\text{px}$ with subsampling (2–3:1).
   - Verify photometric augmentation isolation and generate preview sample grids.

3. **Phase 3: Baseline Edge Validation on Raspberry Pi**
   - Refactor `edge/benchmark.py` for $\ge 100$ runs, warm-up, and peak RSS tracking.
   - Benchmark exported SwissCube ONNX and NCNN models on physical Raspberry Pi 4 to establish first real edge numbers.

4. **Phase 4: SPARK Model Training & Optimization**
   - Run short dry run on local RTX 2050 GPU to establish exact epoch timings.
   - Execute main overnight YOLOv8-Nano training run on SPARK Subset B at $640\text{px}$ (`degrees=0, flipud=0`).
   - Train $416\text{px}$ comparison variant for latency tradeoff analysis.

5. **Phase 5: Model Evaluation, Error Analysis & Model Freeze**
   - Evaluate SPARK models using `src/eval.py` on validation and held-out test splits.
   - Perform structured error analysis (per-class P/R, size bins, confusion matrices, failure gallery).
   - Freeze candidate model by Sun 25 Oct (Git tag `model-freeze`, weights archived).

6. **Phase 6: Edge Quantization & INT8 Accuracy Retention**
   - Build INT8 quantization pipeline (ONNX Runtime static INT8 & NCNN calibration route) on calibration subset.
   - Measure FP32 vs INT8 accuracy retention on held-out test split.

7. **Phase 7: Physical Raspberry Pi Benchmark Suite & Tradeoff Analysis**
   - Benchmark frozen FP32 and INT8 models across NCNN and ONNX Runtime on Raspberry Pi 4.
   - Generate automated accuracy vs latency vs size tradeoff tables and charts.

8. **Phase 8: End-to-End Inference CLI & Interactive Demo**
   - Implement `src/infer.py` with multi-backend switching (PyTorch, ONNX, NCNN).
   - Build interactive demo (Gradio / notebook) with bounding boxes, confidence, latency, and independent SGP4 TLE dashboard.

9. **Phase 9: Real-Image Validation, Reporting & Final Delivery**
   - Run qualitative validation on real space imagery set (~20–30 images) and document domain gap.
   - Author 1–2 page FPGA feasibility note.
   - Finalize technical report, presentation slide deck, one-page summary, and demo video.

## Confidence Assessment

| Area             | Confidence | Notes                                                                                            |
| ---------------- | ---------- | ------------------------------------------------------------------------------------------------ |
| **Stack**        | HIGH       | Verified locally; YOLOv8, NCNN, ONNX Runtime, and PyTorch fully operational.                     |
| **Features**     | HIGH       | Explicitly scoped against PRD v2.0 and Status Review v2.1 addendum.                              |
| **Architecture** | HIGH       | Clean pipeline boundaries established with isolated astrodynamics service.                       |
| **Pitfalls**     | HIGH       | Concrete failure modes identified and mitigated by updated decision register (DEC-01 to DEC-10). |

**Overall Confidence:** HIGH

## Sources

### Primary (HIGH confidence)

- Status Review & Revised 4-Week Plan (v2.1, October 2026)
- Product Requirements Document (PRD v2.0, Problem Statement #05)
- Repository code and decision logs (`docs/decisions.md`, `PROJECT_DOCUMENTATION.md`)

### Secondary (MEDIUM confidence)

- Ultralytics YOLOv8 Edge Export Guidelines
- CelesTrak GP Satellite Catalog and Skyfield Astrodynamics Engine

---

_Research completed: 2026-10-12_  
_Ready for roadmap: yes_
