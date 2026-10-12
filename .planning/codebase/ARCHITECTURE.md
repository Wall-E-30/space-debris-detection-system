---
last_mapped_commit: ac3c85127588eab2407698aba0ad22f3abe8167d
last_mapped_at: 2026-10-12
---
# Architecture

**Analysis Date:** 2026-10-12

## System Overview

The Space Debris Detection System is designed as a multi-stage, edge-deployable pipeline spanning three primary operational stages:
1. **Data Ingestion & Augmentation:** Ingesting raw synthetic space imagery (BOP / SPARK CSV formats), normalizing into YOLO coordinates, and applying domain-gap photometric augmentations (sensor noise, optical blur, exposure jitter).
2. **Object Detection:** Fine-tuning and running inference using an ultralightweight YOLOv8-Nano detector backbone, outputting bounding boxes and classification tags (`satellite`, `debris`).
3. **Orbital Mechanics & Conjunction Analysis:** Mapping detections to NORAD catalog IDs and running real-time SGP4 orbit propagation to calculate relative 3D Euclidean distances in kilometers.
4. **Edge Optimization:** Converting PyTorch weights into ONNX and NCNN binaries, and benchmarking on ARM multi-threaded CPU environments.

## Architectural Pattern

- **Pipeline / Staged Processing Pattern:** Loose coupling between stages via standardized disk artifacts (YOLO dataset layout, PyTorch checkpoint `.pt`, ONNX `.onnx`, NCNN `.param`/`.bin`, CelesTrak TLE ephemeris).
- **Separation of Concerns:**
  - `src/data/`: Data conversion, augmentation, dataset exploration, and orbital propagation.
  - `src/`: Training entry points and future evaluation scripts.
  - `edge/`: Quantization, model export, and latency benchmarking.

## Key Abstractions

**SpaceImageAugmenter:**
- Purpose: Composable photometric augmentation pipeline for synthetic space imagery to simulate real star tracker / telescope noise without distorting bounding boxes.
- Examples: `src/data/augment.py`
- Pattern: Object-oriented pipeline with configurable probability, range parameters, and immutable bounding box guarantee.

**OrbitalTracker:**
- Purpose: Translates detected visual object classes into physical 3D geocentric coordinates via SGP4 propagation and computes pairwise Euclidean distances.
- Examples: `src/data/tle_pipeline.py`
- Pattern: Service object holding cached timescale, catalog dictionary, and SGP4 propagation methods.

**Model Exporter & Verifier:**
- Purpose: Multi-target model export converting PyTorch weights into simplified FP32 ONNX and ARM-optimized NCNN formats, with automated forward-pass verification.
- Examples: `edge/export.py`
- Pattern: Script harness with verification step.

## Entry Points

**Training Pipeline:**
- Location: `src/train.py` (and legacy `train_gpu_opt_a.py`)
- Triggers: CLI invocation (`python src/train.py --epochs 5 --imgsz 1024 --batch 8`)
- Responsibilities: Verifies dataset paths, runs training-split photometric augmentation handoff, initializes YOLOv8-Nano with space-domain hyperparameter augmentations (360° rotation, zero-g flips, random erasing), launches GPU training.

**Edge Model Export:**
- Location: `edge/export.py`
- Triggers: CLI invocation (`python edge/export.py`)
- Responsibilities: Loads `best.pt`, exports to ONNX and NCNN, validates graph execution via `onnxruntime`.

**Edge Benchmarking:**
- Location: `edge/benchmark.py`
- Triggers: CLI invocation (`python edge/benchmark.py --onnx ... --ncnn ...`)
- Responsibilities: Measures mean/median/p95 latency, FPS throughput, and RSS memory deltas across thread counts (1, 4).

**Dataset Conversion & Split:**
- Location: `src/data/split_dataset.py`, `src/data/make_labels.py`
- Triggers: CLI invocation
- Responsibilities: Converts BOP JSON or SPARK CSV datasets into YOLO format with stratified class splits.

**Orbital Conjunction Simulation:**
- Location: `src/data/tle_pipeline.py`
- Triggers: Direct script execution or downstream import
- Responsibilities: Resolves detected classes to NORAD IDs, pulls live TLEs, calculates geocentric XYZ vectors, and assesses distance.

## Architectural Constraints

- **Threading:** Multi-threaded dataloading (`workers=4` in training); multi-threaded inference benchmarking (`threads=4` intra-op threads in ONNX and NCNN).
- **Global state:** Ephemeris data cached in-memory inside `OrbitalTracker.catalog`; hardcoded dataset paths in `spark.yaml`.
- **Circular imports:** None observed; clean hierarchical imports from `src.data.augment`.

## Anti-Patterns

### Hardcoded Absolute Paths in Configuration

**What happens:** `spark.yaml` hardcodes `path: C:/Users/Sharanya Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/data/processed/Swisscube`.
**Why it's wrong:** Breaks portability across different workstations, CI runners, or edge platforms (e.g. Raspberry Pi).
**Do this instead:** Use repository-relative paths or environment-derived base paths, as partially handled in `src/train.py`.

### Redundant Top-Level Scripts

**What happens:** `train_gpu_opt_a.py` exists at the root, in `notebooks/train_gpu_opt_a.py`, and is rewritten as `src/train.py`.
**Why it's wrong:** Violates DRY, creates drift between ad-hoc experimentation and production code.
**Do this instead:** Consolidate training logic strictly under `src/train.py`.

## Error Handling

**Strategy:** Fail-fast on missing files with clear console feedback; fallback handling in data parsing and orbital tracking.

**Patterns:**
- `FileNotFoundError` raised when required model checkpoints or configuration files are missing (`src/train.py`, `edge/export.py`).
- Error dictionaries returned when catalog lookups fail (`src/data/tle_pipeline.py`).
- Fallback from symlinking to copying on Windows systems (`src/data/split_dataset.py`).

## Cross-Cutting Concerns

**Logging:** Standard Python `print()` statements formatted with visual dividers (`"=" * 60`). No structured logging library (e.g. `logging` module or `loguru`).
**Validation:** CLI argument parsing with `argparse` type enforcement and defaults.
**Authentication:** Not applicable; public astronomical endpoints used.

---

*Architecture analysis: 2026-10-12*
