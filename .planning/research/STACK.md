# Stack Research

**Domain:** Edge AI / Space Situational Awareness (SSA) Object Detection & Orbital Tracking  
**Researched:** 2026-10-12  
**Confidence:** HIGH  

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Python** | 3.10+ | Primary runtime environment | Universal ecosystem standard for PyTorch, OpenCV, Ultralytics, and edge conversion scripts. |
| **Ultralytics YOLOv8** | 8.0+ | Object detection architecture and training engine | State-of-the-art accuracy/speed Pareto frontier; native export pipelines to ONNX and NCNN; anchor-free detection head. |
| **YOLOv8-Nano (`yolov8n.pt`)** | v8n | Baseline model backbone | Ultra-low parameter footprint (~3.0M params, ~8.2 GFLOPs at 640px) capable of real-time or near-real-time CPU edge inference. |
| **PyTorch & Torchvision** | 2.0+ (CUDA 11.8/12.x) | Deep learning framework & GPU acceleration | Hardware acceleration on local NVIDIA RTX 2050 GPU (AMP, RAM caching); core model training engine. |
| **NCNN** | Latest (2024+) | Primary edge inference runtime (ARM CPU) | High-performance mobile/embedded inference engine heavily optimized for ARM NEON vector instructions and multi-threading on Raspberry Pi 4. |
| **ONNX Runtime** | 1.15+ | Secondary/fallback edge inference runtime & graph verification | Universal cross-platform interchange standard; robust static INT8 quantization via calibration reader. |
| **Skyfield** | 1.46+ | High-precision SGP4 orbital mechanics | Fast, accurate propagation of NORAD Two-Line Element (TLE) ephemerides into 3D geocentric coordinates without requiring heavy C extensions. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **OpenCV (`opencv-python`)** | 4.8+ | Image decoding, resizing, letterboxing | High-speed pre-processing and bounding box rendering in CLI (`src/infer.py`). |
| **Pillow (`PIL`)** | 10.0+ | Photometric transformations | Brightness, contrast, blur, and noise injections (`src/data/augment.py`). |
| **NumPy** | 1.24+ | Vectorized numerical operations & Euclidean distance math | Bounding box normalization, array operations, and 3D coordinate distance calculations. |
| **psutil** | 5.9+ | Process resource profiling | Measuring peak Resident Set Size (RSS) memory footprint during edge benchmarking (`edge/benchmark.py`). |
| **PyYAML** | 6.0+ | Configuration parsing | Reading and writing dataset configurations (`spark.yaml`, `swisscube.yaml`). |

### Development & Benchmarking Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **NCNN Tools (`ncnn2mem`, `ncnnoptimize`)** | Graph optimization & quantization | Eliminates redundant layers, fuses convolutions, and builds INT8 calibration tables for ARM. |
| **ONNX Simplifier (`onnxsim`)** | Graph simplification | Cleans up constant folding and shape inference before deploying ONNX graphs. |
| **Gradio / Jupyter** | Interactive demonstrator | UI for uploading test images, rendering detections, confidence, latency, and model size. |

## Installation

```bash
# Core Training & Pipeline (Host Workstation)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy matplotlib pyyaml onnx onnxruntime skyfield psutil

# Edge Environment (Raspberry Pi 4 - requirements-edge.txt)
pip install onnxruntime opencv-python-headless numpy psutil
# Install precompiled ncnn wheel for aarch64 Linux
pip install ncnn
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **NCNN** | **TFLite** | TFLite is standard on mobile Android devices, but on ARM Linux SBCs (Raspberry Pi 4), NCNN delivers superior raw CPU performance with NEON tuning. |
| **NCNN** | **ONNX Runtime** | ONNX Runtime is easier to install and more portable, which is why it serves as the official fallback if NCNN wheels encounter issues on the Pi. |
| **YOLOv8-Nano** | **YOLOv5-Nano** | YOLOv5n has slightly fewer parameters, but YOLOv8n has superior mAP@0.5:0.95 due to its anchor-free head and decoupled classification/regression branches. |
| **YOLOv8-Nano** | **DETR / RT-DETR** | Transformer-based detectors require high FLOPs and high memory bandwidth; completely unsuitable for 4-week delivery on a 4GB GPU and Raspberry Pi 4. |
| **Subsampling (2-3:1)** | **Focal Loss** | Ultralytics v8 does not expose focal loss via clean flags; tuning custom loss weights introduces instability in a short 4-week window. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **Edge Impulse** | EON compiler is built for TinyML microcontrollers (FOMO, micro-CNNs), not full YOLO detector heads. | NCNN / ONNX Runtime. |
| **1024px Resolution on Edge** | Compute scales quadratically with resolution; 1024px costs 2.6x more work than 640px, causing Raspberry Pi latency to exceed the 1000ms SLA. | 640px deployment standard, with 416px comparison variant. |
| **Large Geometric Rotations (`degrees=180`)** | In axis-aligned box detection, rotating an oblong satellite causes the bounding box to expand artificially, degrading mAP@0.5:0.95. | Offline photometric augmentation (`src/data/augment.py`) with `degrees=0, flipud=0`. |
| **Image-based SGP4 Conjunction Claims** | 2D bounding boxes carry no physical range, radial velocity, or attitude; resolving a class name to a NORAD ID is a catalog lookup, not a sensor measurement. | Present TLE SGP4 module as an independent contextual catalog awareness layer. |

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `ultralytics@8.x` | `torch>=2.0.0` | PyTorch 2.x recommended for optimized export and memory management. |
| `onnx@1.14+` | `opset=12` | Opset 12 ensures broad compatibility across both ONNX Runtime and NCNN tools. |
| `skyfield@1.46` | `numpy>=1.20` | Seamless timescale calculations and TLE parsing. |

## Sources

- Ultralytics YOLOv8 Documentation (`https://docs.ultralytics.com/modes/export/`)
- Tencent NCNN Wiki & Raspberry Pi Optimization Benchmarks
- CelesTrak General Perturbation (GP) API & Skyfield Astronomical Library
- Empirical local benchmarks from `docs/decisions.md` (NCNN: 165ms vs ONNX Runtime: 210ms)

---
*Stack research for: Edge-Deployable Space Debris Detection System*  
*Researched: 2026-10-12*
