---
last_mapped_commit: ac3c85127588eab2407698aba0ad22f3abe8167d
last_mapped_at: 2026-10-12
---
# External Integrations

**Analysis Date:** 2026-10-12

## APIs & External Services

**Orbital Ephemeris & Satellite Catalog:**
- CelesTrak GP API (`https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle`) - Fetches active Two-Line Element (TLE) satellite ephemeris data for real-time SGP4 propagation (`src/data/tle_pipeline.py`)
  - SDK/Client: `skyfield.api.load`
  - Auth: None (public open scientific data endpoint)
  - Caching: Downloaded cached raw response found at project root (`gp.php`)

**Model Hub:**
- Ultralytics Assets Hub - Auto-downloads pretrained model checkpoints (e.g. `yolov8n.pt`) on first invocation (`src/train.py`, `train_gpu_opt_a.py`)
  - SDK/Client: `ultralytics.YOLO`
  - Auth: None (public GitHub release downloads)

## Data Storage

**Databases:**
- Not detected / None used (all dataset metadata, bounding boxes, and checkpoints are stored flat on disk)

**File Storage:**
- Local filesystem only (`data/raw/`, `data/processed/`, `experiments/`, `outputs/`)
- YOLO annotation format (`*.txt` files with normalized coordinates `class_id x y w h`)
- BOP JSON format (`scene_gt_info.json` for SwissCube data)
- CSV metadata (`train.csv`, `val.csv` for SPARK dataset)

**Caching:**
- RAM Caching enabled in training dataloader (`cache=True` in `src/train.py`)
- Ephemeris cache: Local `gp.php` file at repository root

## Authentication & Identity

**Auth Provider:**
- Not applicable (offline edge computing & local data processing pipelines)

## Monitoring & Observability

**Error Tracking:**
- Not detected / None (standard Python tracebacks and error handling)

**Logs:**
- TensorBoard event logs generated under `experiments/` (`requirements.txt`)
- CSV training metrics (`experiments/week1_smoke_test/results.csv`)
- Console stdout/stderr profiling in benchmarking harness (`edge/benchmark.py`)

## CI/CD & Deployment

**Hosting:**
- Edge target: Raspberry Pi 4 SBC (ARM64 Linux)
- Training host: Local NVIDIA GPU laptop workstation

**CI Pipeline:**
- None detected (no `.github/workflows/`, GitLab CI, or CircleCI configurations present)

---

*Integrations analysis: 2026-10-12*
