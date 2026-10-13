---
last_mapped_commit: ac3c85127588eab2407698aba0ad22f3abe8167d
last_mapped_at: 2026-10-12
---
# Technology Stack

**Analysis Date:** 2026-10-12

## Languages

**Primary:**
- Python 3.10+ - Core pipeline (`src/train.py`, `src/data/`, `edge/`, `notebooks/`)

**Secondary:**
- YAML - Model and dataset configuration (`spark.yaml`)
- Markdown - Documentation, decision logs, and specifications (`docs/`, `data/`, `PROJECT_DOCUMENTATION.md`)

## Runtime

**Environment:**
- Python 3.10 virtual environment (`.venv/`)
- CUDA execution backend (NVIDIA RTX 2050 Laptop GPU, compute capability support with PyTorch CUDA runtime)

**Package Manager:**
- pip (`requirements.txt`)
- Lockfile: missing (no `poetry.lock`, `Pipfile.lock`, or `requirements.lock`)

## Frameworks

**Core:**
- Ultralytics YOLOv8 (`ultralytics`) - Object detection framework and training engine (`src/train.py`, `train_gpu_opt_a.py`)
- PyTorch (`torch`, `torchvision`) - Deep learning framework with CUDA acceleration backend (`src/train.py`)
- OpenCV (`opencv-python`) & Pillow (`PIL`) - Image reading, preprocessing, photometric transforms (`src/data/augment.py`, `src/data/split_dataset.py`)
- Skyfield (`skyfield`) - SGP4 orbital mechanics and satellite ephemeris propagator (`src/data/tle_pipeline.py`)

**Testing:**
- Not detected / None configured (no pytest or unittest suites configured; testing is done ad-hoc via inline scripts `edge/benchmark.py`, `src/data/tle_pipeline.py`)

**Build/Dev:**
- Jupyter Notebook (`notebooks/`) - Interactive exploration and prototyping (`00_colab_setup.ipynb`, `01_data_exploration.ipynb`, `02_train_baseline.ipynb`)
- ONNX (`onnx`) - Open Neural Network Exchange graph exporter (`edge/export.py`)
- ONNX Runtime (`onnxruntime`) - Portable CPU/GPU inference execution engine (`edge/export.py`, `edge/benchmark.py`)
- Tencent NCNN (`ncnn`) - High-performance mobile/embedded neural network inference engine (`edge/export.py`, `edge/benchmark.py`)
- TensorBoard (`tensorboard`) - Experiment metrics tracking (`requirements.txt`)

## Key Dependencies

**Critical:**
- `ultralytics` - Provides YOLOv8 architecture, CLI training interface, dataset caching, export hooks (`src/train.py`, `edge/export.py`)
- `torch` & `torchvision` - Core tensor compute and CUDA memory management
- `skyfield` - Orbital propagation using SGP4 and CelesTrak ephemeris (`src/data/tle_pipeline.py`)
- `onnxruntime` - Evaluates inference latency and validates exported ONNX models (`edge/benchmark.py`, `edge/export.py`)
- `numpy` & `matplotlib` - Numerical operations, bounding box calculations, visual inspection grids (`src/data/augment.py`, `src/data/split_dataset.py`)

**Infrastructure:**
- `psutil` - Process memory and CPU footprint profiling during edge benchmarks (`edge/benchmark.py`)
- `pyyaml` - Dataset and hyperparameter configuration parsing (`src/train.py`, `spark.yaml`)

## Configuration

**Environment:**
- Local execution configured with environment variables / Python paths (`sys.path.insert(0, str(repo_root))`)
- No `.env` files detected; ephemeris data fetched via public HTTP endpoint (`https://celestrak.org/`)

**Build:**
- Dataset config: `spark.yaml`
- Dependency manifest: `requirements.txt`

## Platform Requirements

**Development:**
- Windows / Linux / macOS with Python 3.10+
- NVIDIA CUDA-capable GPU (tested on NVIDIA RTX 2050 4GB VRAM) for accelerated training; CPU fallback supported

**Production:**
- Raspberry Pi 4 Model B (Quad-core ARM Cortex-A72 @ 1.5/1.8 GHz, 4GB/8GB RAM) targeting ARM NEON CPU execution via NCNN FP32/INT8 or ONNX Runtime CPUExecutionProvider

---

*Stack analysis: 2026-10-12*
