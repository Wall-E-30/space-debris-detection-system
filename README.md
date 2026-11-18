# Autonomous On-Orbit Space Debris Detection & Situational Awareness System

[![ISRO Problem Statement #05](https://img.shields.io/badge/ISRO-Problem%20Statement%20%2305-blue.svg)](https://www.isro.gov.in/)
[![Model Status](https://img.shields.io/badge/Model-Frozen%20(DEC--09)-brightgreen.svg)](docs/decisions.md)
[![mAP@50](https://img.shields.io/badge/mAP%4050-99.02%25-success.svg)](docs/MASTER_PROJECT_REPORT.md)
[![Edge Footprint](https://img.shields.io/badge/INT8%20Footprint-3.59%20MB-blue.svg)](docs/RAW_OUTPUTS_AND_BENCHMARKS.md)
[![Edge Target](https://img.shields.io/badge/Edge%20Platform-Raspberry%20Pi%204%2F5-orange.svg)](docs/MASTER_PROJECT_REPORT.md)

An embedded computer vision and astrodynamics situational awareness pipeline engineered for autonomous, real-time on-orbit detection and tracking of space debris and Resident Space Objects (RSOs) under severe SWaP-C (Size, Weight, Power, and Cost) constraints.

---

## Architecture Overview

```
+-----------------------------------------------------------------------------------------+
|                                    OPTICAL SENSOR STREAM                                 |
+-----------------------------------------------------------------------------------------+
                                             |
                                    (Letterbox Preprocessing)
                                             v
+-----------------------------------------------------------------------------------------+
|                      ON-ORBIT COMPUTER VISION LAYER (YOLOv8-Nano)                        |
|                                                                                         |
|   * Input: 640x640 / 1024x1024 RGB Sensor Frame                                         |
|   * Classes: 11 Classes (10 Satellites + 1 Consolidated Debris across 5 CAD Models)     |
|   * Engine: ONNX Runtime CPU Execution Provider / PyTorch FP16                           |
|   * Quantization: Static INT8 (3.59 MB footprint, 96.47% mAP@50 retention)              |
+-----------------------------------------------------------------------------------------+
                                             |
                             (2D Bounding Boxes + Classes + Confidences)
                                             v
+-----------------------------------------------------------------------------------------+
|                     DECOUPLED ASTRODYNAMICS LAYER (SGP4 / Skyfield)                     |
|                                                                                         |
|   * Ground-Truth NORAD Catalog Mapping (CelesTrak Two-Line Elements)                    |
|   * SGP4 Ephemeris Propagation: Resolves 3D ECI State Vectors for known assets           |
|   * Strict Decoupling (DEC-05): 2D optical detections are NEVER directly inverted into   |
|     3D orbits; serves as independent situational awareness catalog layer.                |
+-----------------------------------------------------------------------------------------+
```

---

## Key Performance & Verification Metrics

All metrics verified against the 11-class SPARK-2021 test set (15,465 ground-truth targets) following 50 epochs of training on an NVIDIA Quadro GV100 GPU (180,000 augmented images, 562,500 iterations):

| Metric | PyTorch FP32 Baseline | ONNX FP32 Export | ONNX Static INT8 (Calibrated) | Edge Target Gate |
| :--- | :---: | :---: | :---: | :---: |
| **Model Size** | 6.25 MB | 12.33 MB | **3.59 MB** | $\le 4.0\text{ MB}$ (PASSED) |
| **Parameters** | 3.14M | 3.14M | 3.14M (INT8 Weights) | Lightweight Nano |
| **Overall mAP@0.50** | **99.02%** | **99.02%** | **96.47%** | $\ge 90.0\%$ (PASSED) |
| **Overall mAP@0.50:0.95** | **86.98%** | **86.98%** | **84.12%** | $\ge 80.0\%$ (PASSED) |
| **Debris Class mAP@0.50** | **98.34%** | **98.34%** | **95.80%** | $\ge 90.0\%$ (PASSED) |
| **Inference Latency (Pi 5 @ 640px)** | — | 144.83 ms | ~92 ms | $\le 1000\text{ ms}$ (PASSED) |
| **Inference Latency (Pi 5 @ 1024px)**| — | 515.19 ms | ~310 ms | $\le 1000\text{ ms}$ (PASSED) |
| **Inference Latency (Pi 4 @ 640px)** | — | ~360 ms | **~190 ms** | $\le 1000\text{ ms}$ (PASSED) |

> **Note on Hardware Operating Point:** Raspberry Pi 5 easily satisfies real-time latencies across all resolutions (up to 1024px). For the PRD-mandated **Raspberry Pi 4**, **640x640 px with INT8 quantization** is the officially qualified safe operating point.

---

## Repository Structure

```
space-debris-detection-system/
├── configs/                          # Dataset definitions
│   ├── spark2021.yaml                # Verified 11-class dataset manifest
│   └── spark.yaml                    # Legacy 10-class manifest
├── data/
│   ├── processed/                    # Processed YOLO format train/val/test splits
│   └── tle/                          # Offline NORAD Two-Line Element (TLE) ephemeris
│       └── celestrak_active.tle      # Cached active satellite catalog
├── docs/                             # Engineering reports and architecture ledgers
│   ├── MASTER_PROJECT_REPORT.md      # Primary technical engineering report
│   ├── RAW_OUTPUTS_AND_BENCHMARKS.md # Complete empirical benchmark tables
│   ├── decisions.md                  # Architectural Decisions Ledger (DEC-01 to DEC-10)
│   └── CONCEPTUAL_SYSTEM_SUMMARY.md  # Core principles (YOLO, SGP4, Edge CV)
├── edge/                             # Edge optimization & benchmarking
│   ├── benchmark.py                  # Standardized 100-run latency & RSS memory profiler
│   ├── export.py                     # ONNX (Opset 12) and NCNN export harness
│   └── quantize_onnx.py              # Static INT8 post-training quantizer
├── experiments/
│   └── spark2021_yolov8n_50e/
│       └── weights/
│           ├── best.pt               # Frozen PyTorch weights (SHA-256: 34fe3d10...)
│           ├── best.onnx             # Exported ONNX FP32 (SHA-256: df2cde42...)
│           └── best_int8.onnx        # Calibrated INT8 model (3.59 MB)
├── src/                              # Source code modules
│   ├── infer.py                      # Multi-backend CLI (PyTorch & ONNX Runtime)
│   ├── train.py                      # Training wrapper with space photometric hooks
│   ├── eval.py                       # Standalone evaluation & metrics generator
│   └── data/
│       ├── augment.py                # Photometric transform pipeline (blur, noise, jitter)
│       ├── split_dataset.py          # Stratified train/val/test dataset partitioner
│       └── tle_pipeline.py           # SGP4 orbital tracking & catalog lookup
├── requirements.txt                  # Pinned core dependencies
└── requirements-edge.txt             # Minimal edge runtime dependencies (Raspberry Pi)
```

---

## Quickstart Guide

### 1. Installation

#### Host Workstation (PyTorch & CUDA):
```bash
git clone https://github.com/<org>/space-debris-detection-system.git
cd space-debris-detection-system
python -m venv .venv
source .venv/bin/activate  # Or: .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

#### Edge Platform (Raspberry Pi 4/5):
```bash
pip install -r requirements-edge.txt
```

---

### 2. Running Inference

The modular CLI (`src/infer.py`) automatically selects between the **PyTorch backend** and the high-efficiency **ONNX Runtime backend** based on the weights extension:

#### PyTorch FP32 Inference:
```bash
python src/infer.py \
  --source data/processed/spark2021/images/test/sample.jpg \
  --weights experiments/spark2021_yolov8n_50e/weights/best.pt \
  --imgsz 640
```

#### Native ONNX INT8 Edge Inference (Optimized for Raspberry Pi):
```bash
python src/infer.py \
  --source data/processed/spark2021/images/test/sample.jpg \
  --weights experiments/spark2021_yolov8n_50e/weights/best_int8.onnx \
  --imgsz 640
```

Results (annotated bounding boxes, class labels, and confidence metrics) are saved to `outputs/inference/`.

---

### 3. Astrodynamics & SGP4 Catalog Awareness

Query real-time 3D orbital state vectors for detected cataloged satellites using the offline TLE catalog:

```bash
python src/data/tle_pipeline.py
```

Outputs:
```text
[INFO] Loading TLEs from offline cache: data/tle/celestrak_active.tle
[INFO] Loaded 2584 active satellites into memory.
Object 1 (AcrimSat): Position (km) [  -692.14  -5114.32   4812.51 ], Velocity: 7.42 km/s
Object 2 (Sentinel-6): Position (km) [  3120.45   6102.12  -2914.88 ], Velocity: 7.21 km/s
CubeSat Query: uncataloged (Multi-manifest / no single NORAD ID)
Debris Query: uncataloged (Non-cooperative / SGP4 unsupported)
[CATALOG AWARENESS] Distance between cataloged assets: 12,042.81 km
```

---

### 4. Edge Benchmarking

Benchmark latency and RSS memory consumption on the target hardware:

```bash
python edge/benchmark.py \
  --model experiments/spark2021_yolov8n_50e/weights/best.onnx \
  --imgsz 640 \
  --threads 4 \
  --runs 100
```

---

## Verified Model Checkpoints

Model checkpoints are permanently frozen under **DEC-09**:

| Artifact | File Path | Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **PyTorch Weights** | `experiments/spark2021_yolov8n_50e/weights/best.pt` | 6.25 MB | `34fe3d10b045849fef33e29f10028e878fbd7745a71e2839aa02875e47c8795d` |
| **ONNX FP32** | `experiments/spark2021_yolov8n_50e/weights/best.onnx` | 12.33 MB | `df2cde420af209acddcc9aceb5e9d9ca909d75cdca16474bc7f71aca60260912` |
| **ONNX Static INT8**| `experiments/spark2021_yolov8n_50e/weights/best_int8.onnx`| 3.59 MB | `50e7bba8bbdc068a49cbe778b47ecff1129f79ca02ce4c7810ff4c15502c34d3` |

---

## Technical Documentation & References

- [Master Technical Engineering Report](docs/MASTER_PROJECT_REPORT.md)
- [Raw Outputs, Metrics & Hardware Benchmarks](docs/RAW_OUTPUTS_AND_BENCHMARKS.md)
- [Independent Technical Audit Report](docs/INDEPENDENT_TECHNICAL_AUDIT_REPORT.md)
- [Architectural Decisions Ledger (DEC-01 to DEC-10)](docs/decisions.md)
- [Conceptual System Summary & Theoretical Foundations](docs/CONCEPTUAL_SYSTEM_SUMMARY.md)
- [Workflow Playbook & Operational Guide](docs/WORKFLOW_PLAYBOOK.md)

---

## License & Attribution
Developed for the **Indian Space Research Organisation (ISRO) Problem Statement #05**.  
Dataset based on **Stanford / ESA SPARK-2021 Spacecraft Detection Challenge**.
