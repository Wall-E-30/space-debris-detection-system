# Phase 6: Edge Quantization, Hardware Qualification & Astrodynamics Cleansing — Research

**Gathered:** 2026-11-04  
**Status:** Complete  
**Role:** Person B: Data/Edge Lead  
**Governing Requirements:** EDGE-04, EDGE-05, PROD-01, PROD-03, DATA-02  

---

## 1. Executive Context & Scope Overview

Phase 6 executes the edge deployment, static quantization, and architectural hygiene mandates for the space debris detection system:
1. **Astrodynamics Logic Cleansing (`src/data/tle_pipeline.py` & architecture docs):**
   - Eliminate illegitimate conflation between 2D optical bounding box detections and 3D Keplerian orbit determination.
   - Remove hardcoded ISS proxy (`"CubeSat": 25544`).
   - Redraw all system architecture diagrams to present the TLE pipeline strictly as an independent catalog awareness layer for known satellite assets.
2. **Raspberry Pi Hardware Gates Qualification:**
   - Update all engineering reports to explicitly confirm that the $\le 1000\text{ ms}$ real-time gate was validated on a physical **Raspberry Pi 5** (ARM Cortex-A76 @ 2.4 GHz).
   - Formally designate **640px** as the safe operating point projected to pass on the PRD-mandated **Raspberry Pi 4** (ARM Cortex-A72 @ 1.5 GHz).
3. **Data Leakage Caveats Disclosure:**
   - Disclose the perceptual hash collapse on space backgrounds (>95% black pixels) and document that sequence-level trajectory leakage across splits is not ruled out.
4. **ONNX Runtime Static INT8 Quantization (`edge/quantize_onnx.py`):**
   - Quantize `best.onnx` using `onnxruntime.quantization.quantize_static` with a 300-image calibration dataset.
   - Measure test split accuracy retention loss using `src/eval.py`, ensuring degradation is $\le 3.0\text{ points}$ mAP@50.
5. **Inference CLI Integration (`src/infer.py`):**
   - Author `src/infer.py` supporting both PyTorch and ONNX Runtime backends with letterbox preprocessing, NMS postprocessing, and visual/JSON annotation.

---

## 2. Technical Findings & Architectural Specifications

### 2.1 Astrodynamics Pipeline Cleansing (`src/data/tle_pipeline.py`)

#### Problem Analysis
In `src/data/tle_pipeline.py`:
- Line 21 previously contained: `"CubeSat": 25544, # LEO CubeSat/ISS deployment proxy`
- Conflating CubeSats with NORAD 25544 (the International Space Station) creates severe physical inaccuracies: CubeSats deployed across diverse orbital inclinations do not share ISS ephemeris.
- Debris fragments (Class 6) have no NORAD entries in CelesTrak.
- Single mono-camera 2D bounding boxes $[cx, cy, w, h]$ cannot resolve 3D range or orbital state vectors $[X, Y, Z, \dot{X}, \dot{Y}, \dot{Z}]$.

#### Remediation Architecture
- Modify `self.class_to_norad`:
  ```python
  "CubeSat": None,  # UNCATALOGED: Multi-manifest CubeSats lack fixed single NORAD ID
  "Debris": None,   # UNCATALOGED: SGP4 propagation unsupported
  ```
- Update `get_position()`:
  When queried with uncataloged objects (`CubeSat`, `Debris`, or unmapped classes), return an explicit structured status:
  ```python
  if detected_class in ["Debris", "CubeSat"] or norad_id is None:
      return {
          "status": "uncataloged",
          "error": f"Class '{detected_class}' is uncataloged or has no single fixed NORAD ID in CelesTrak. Direct 2D optical detections cannot determine 3D orbital state vectors without multi-station radar/optical tracklet orbit determination."
      }
  ```
- Decouple `calculate_distance()`: Clarify that relative distance calculations apply exclusively between two verified cataloged assets (e.g., `AcrimSat` vs `Sentinel-6`) as a parallel ground-catalog awareness tool, never between a camera detection and an orbital body.

---

### 2.2 Raspberry Pi Hardware Gates & Resolution Pareto Designation

#### Physical Hardware Delta
- **Raspberry Pi 5:** Quad-core ARM Cortex-A76 @ 2.4 GHz (Crypto/DotProd/NEON).
  - Measured 640px ONNX FP32 (4 threads): **144.83 ms** (6.9 FPS, $6.90\times$ faster than PRD 1000 ms gate).
  - Measured 800px ONNX FP32 (4 threads): **285.15 ms** (3.5 FPS, $3.51\times$ faster than PRD 1000 ms gate).
  - Measured 1024px ONNX FP32 (4 threads): **515.19 ms** (1.9 FPS, $1.94\times$ faster than PRD 1000 ms gate).
- **Raspberry Pi 4 (PRD Target):** Quad-core ARM Cortex-A72 @ 1.5 GHz.
  - IPC and clock frequency difference yields an empirical $\sim 2.2\times–2.5\times$ latency multiplier for dense CNN matrix operations.
  - Projected 640px on Pi 4: $144.83\text{ ms} \times 2.4 \approx 348\text{ ms}$ (comfortably **PASS** $\le 1000\text{ ms}$).
  - Projected 800px on Pi 4: $285.15\text{ ms} \times 2.4 \approx 684\text{ ms}$ (conditional **PASS** $\le 1000\text{ ms}$).
  - Projected 1024px on Pi 4: $515.19\text{ ms} \times 2.4 \approx 1236\text{ ms}$ (**FAIL** / exceeds 1000 ms ceiling).
- **Designation:** Formally designate **640px** as the safe, real-time operating point for Raspberry Pi 4 edge deployment, offering $>60\%$ latency margin against the PRD gate.

---

### 2.3 Data Leakage Caveats & Transparent Disclosure

#### Background Hash Collapse
- Orbital synthetic imagery features $>95\%$ uniform solid black pixels ($\text{RGB} = [0, 0, 0]$).
- Global 64-bit difference hashing (dHash) reduces images to an $8 \times 8$ grayscale thumbnail. For small orbital objects (e.g. CubeSat, distant debris fragments occupying $<1\%$ image area), thumbnail averaging renders all 64 blocks uniformly black, yielding identical hashes (`0x0000000000000000`) and false-positive duplicate collisions across completely distinct classes.
- Consequently, global dHash cannot guarantee that training trajectories and testing trajectories are strictly disjoint.
- Documentation must state: *"Because global difference hashing collapsed on uniform space backgrounds, sequence-level trajectory leakage across splits is not ruled out; test set evaluations demonstrate high-accuracy synthetic scene detection but should be validated against physically distinct trajectory sequences."*

---

### 2.4 ONNX Runtime Static INT8 Quantization Strategy

#### Quantization Mechanics
- **Model Target:** `experiments/spark2021_yolov8n_50e/weights/best.onnx` (12.91 MB FP32).
- **Input Tensor:** `'images'` shape `[batch, 3, height, width]`, type float32 normalized $[0.0, 1.0]$.
- **Calibration Pipeline:**
  - Calibration reader class inheriting from `onnxruntime.quantization.CalibrationDataReader`.
  - Samples 300 representative images from training split (`data/processed/spark/images/train/`).
  - Preprocessing applies letterboxing, color conversion (BGR to RGB), normalization ($x / 255.0$), and transpose ($H \times W \times C \to 1 \times C \times H \times W$).
  - Quantization parameters:
    - `quant_format=QuantFormat.QDQ` (Quantize-Dequantize representation for cross-platform hardware).
    - `activation_type=QuantType.QInt8` (or `QUInt8`).
    - `weight_type=QuantType.QInt8`.
    - `calibrate_method=CalibrationMethod.MinMax` (robust for bounded sensor dynamic ranges).
- **Verification Criterion (EDGE-05):**
  - Evaluate accuracy retention on test split via `src/eval.py`.
  - Baseline FP32 test mAP@50: $99.02\%$.
  - Retention threshold: $\le 3.0\text{ points}$ drop (mAP@50 $\ge 96.0\%$).

---

### 2.5 Multi-Backend Inference CLI Architecture (`src/infer.py`)

#### Pipeline Design
`src/infer.py` provides an end-to-end user-facing CLI:
1. **Input Ingestion:** Accepts single image (`--source path/to/image.jpg`) or folder (`--source path/to/dir/`).
2. **Backend Engine:**
   - `--backend pytorch`: Wraps `ultralytics.YOLO(weights)` for native PyTorch inference.
   - `--backend onnx`: Initializes `onnxruntime.InferenceSession(weights, providers=['CPUExecutionProvider'])`.
3. **ONNX Preprocessing:**
   - Letterbox resize to target resolution (default 640 or 1024), preserving aspect ratio with black-border padding.
   - Channel swap (BGR to RGB), float32 scaling ($1/255$), tensor transpose (`1, 3, H, W`).
4. **ONNX Postprocessing:**
   - Raw output shape: `[1, 15, 8400]`.
   - Transpose to `[8400, 15]`.
   - Columns 0..3: Bounding box center coordinates `[cx, cy, w, h]`.
   - Columns 4..14: Class confidence logits / probabilities for 11 classes.
   - Compute max class score and class ID per anchor.
   - Filter by confidence threshold (`--conf 0.25`).
   - Convert `[cx, cy, w, h]` to `[x1, y1, x2, y2]` in padded letterbox coordinates.
   - Apply Non-Maximum Suppression (NMS) (`--iou 0.45`).
   - Invert letterbox scaling to restore coordinates to original source image dimensions.
5. **Output Generation:**
   - Draw bounding boxes with distinct class colors, labels, and confidence percentages.
   - Save annotated image to `--out-dir`.
   - Optional `--json` flag prints or writes structured detection array with latency metrics.

