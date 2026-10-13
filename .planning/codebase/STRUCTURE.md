---
last_mapped_commit: ac3c85127588eab2407698aba0ad22f3abe8167d
last_mapped_at: 2026-10-12
---
# Codebase Structure

**Analysis Date:** 2026-10-12

## Directory Layout

```
space-debris-detection-system/
├── .agents/                    # GSD workflows, skills, and agent tools
├── .venv/                      # Python virtual environment
├── data/                       # Datasets, taxonomy documentation, raw/processed splits
│   ├── processed/              # Formatted YOLO datasets (images/ & labels/ for train/val/test)
│   ├── raw/                    # Raw downloads (SwissCube BOP, SPARK)
│   ├── README.md               # Dataset overview and notes
│   └── decisions.md            # Object taxonomy and dataset decisions log
├── docs/                       # Project engineering documentation and decision logs
│   ├── cv_basics.md            # Theoretical guide to computer vision & object detection
│   ├── decisions.md            # Major project decision records (runtimes, models, edge results)
│   └── meeting_notes.md        # Collaboration and status logs
├── edge/                       # Edge model conversion, optimization, and benchmarking
│   ├── benchmark.py            # Latency (mean/p95), throughput (FPS), RAM benchmarking harness
│   └── export.py               # Exporter for ONNX (FP32) and NCNN runtimes with forward-pass verification
├── experiments/                # Experiment checkpoints and training outputs
│   ├── week1_smoke_test/       # YOLOv8n 5-epoch baseline experiment weights and results.csv
│   └── week1_smoke_test.zip    # Archived baseline experiment artifacts
├── notebooks/                  # Prototyping and exploration notebooks
│   ├── 00_colab_setup.ipynb    # Cloud GPU environment setup
│   ├── 01_data_exploration.ipynb # Dataset inspection and visualization
│   ├── 02_train_baseline.ipynb # Interactive training notebook
│   └── train_gpu_opt_a.py      # Standalone training script copy
├── outputs/                    # Output visual inspection images, confusion matrices, logs
├── src/                        # Core production source code
│   ├── data/                   # Data processing and domain-specific pipelines
│   │   ├── augment.py          # Photometric space-domain augmentation engine
│   │   ├── explore_dataset.py  # Dataset statistics and validation utility
│   │   ├── make_labels.py      # BOP format to YOLO label converter
│   │   ├── split_dataset.py    # Stratified dataset train/val/test splitter
│   │   └── tle_pipeline.py     # SGP4 orbit propagator and conjunction calculator
│   ├── eval.py                 # Evaluation pipeline entry point (currently 0 bytes / empty)
│   └── train.py                # Main YOLOv8 training CLI with space augmentations
├── gp.php                      # Cached CelesTrak active satellite ephemeris TLE file
├── PROJECT_DOCUMENTATION.md    # Comprehensive project reference, architecture, and roadmap
├── README.md                   # Project summary
├── requirements.txt            # Python dependencies manifest
├── spark.yaml                  # YOLO dataset configuration for SwissCube/SPARK
└── train_gpu_opt_a.py          # Legacy/standalone GPU training script at root
```

## Directory Purposes

**`src/`:**
- Purpose: Production Python application modules for dataset processing, training, and evaluation.
- Contains: Executable Python scripts and modular classes (`SpaceImageAugmenter`, `OrbitalTracker`).
- Key files: `src/train.py`, `src/eval.py`, `src/data/augment.py`, `src/data/tle_pipeline.py`.

**`edge/`:**
- Purpose: Optimization, cross-compilation, and latency evaluation for edge hardware (Raspberry Pi).
- Contains: Model export scripts and benchmarking harnesses.
- Key files: `edge/export.py`, `edge/benchmark.py`.

**`data/`:**
- Purpose: Storage for raw and preprocessed orbital datasets, dataset documentation, and taxonomy logs.
- Contains: Label definitions, image splits, documentation files.
- Key files: `data/decisions.md`, `data/README.md`.

**`docs/`:**
- Purpose: Engineering documentation, research notes, and architectural decision records.
- Contains: Markdown technical documentation.
- Key files: `docs/decisions.md`, `docs/cv_basics.md`.

**`experiments/`:**
- Purpose: Artifacts generated during model training runs (weights, confusion matrices, metrics logs).
- Contains: `.pt` checkpoints, `results.csv`, training curve plots.
- Key files: `experiments/week1_smoke_test/weights/best.pt`, `experiments/week1_smoke_test/results.csv`.

**`notebooks/`:**
- Purpose: Exploratory analysis, Colab experimentation, and interactive validation.
- Contains: Jupyter notebooks and quick-test scripts.
- Key files: `notebooks/01_data_exploration.ipynb`, `notebooks/02_train_baseline.ipynb`.

## Key File Locations

**Entry Points:**
- `src/train.py`: Primary CLI for model training.
- `edge/export.py`: Model conversion to ONNX and NCNN.
- `edge/benchmark.py`: Empirical edge latency & resource benchmarking.
- `src/data/tle_pipeline.py`: Real-time SGP4 orbit propagation & distance calculation.

**Configuration:**
- `spark.yaml`: YOLO dataset directory mappings and class definitions.
- `requirements.txt`: Python package dependency list.

**Core Logic:**
- `src/data/augment.py`: Photometric jitter, blur, and noise implementation.
- `src/data/tle_pipeline.py`: SGP4 ephemeris parsing and 3D Euclidean distance math.

**Testing:**
- Not detected / None centralized (ad-hoc inline execution via `if __name__ == '__main__':` blocks).

## Naming Conventions

**Files:**
- Snake_case for Python scripts: `train.py`, `split_dataset.py`, `tle_pipeline.py`.
- Numbered prefixes for exploratory notebooks: `00_colab_setup.ipynb`, `01_data_exploration.ipynb`.
- Snake_case or kebab-case for documentation: `cv_basics.md`, `decisions.md`.

**Directories:**
- Lowercase single words or snake_case: `src`, `data`, `edge`, `docs`, `notebooks`, `experiments`.

## Where to Add New Code

**New Feature (e.g. Inference CLI):**
- Primary code: `src/infer.py`
- Tests: `tests/test_infer.py`

**New Edge Optimization (e.g. INT8 Quantization):**
- Implementation: `edge/quantize.py`
- Tests: `edge/test_quantization.py`

**Utilities & Data Processors:**
- Shared helpers: `src/data/` or `src/utils/`

## Special Directories

**`experiments/`:**
- Purpose: Generated training weights and evaluation outputs.
- Generated: Yes.
- Committed: Partially (best.pt / results.csv tracked, large datasets gitignored).

**`.planning/`:**
- Purpose: GSD workflow planning artifacts, state, and codebase maps.
- Generated: Yes.
- Committed: Yes.

---

*Structure analysis: 2026-10-12*
