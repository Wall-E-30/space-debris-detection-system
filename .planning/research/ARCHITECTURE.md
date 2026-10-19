# Architecture Research

**Domain:** Edge AI Computer Vision & Embedded Inference Architecture  
**Researched:** 2026-10-12  
**Confidence:** HIGH

## Architecture Overview

The system architecture cleanly separates development/training workflows on a GPU host workstation from deployment/benchmarking pipelines on edge SBC hardware (Raspberry Pi 4), connected via standardized model and data serialization formats.

```
+----------------------------------------------------------------------------------------------------+
|                                    HOST WORKSTATION (DEVELOPMENT)                                  |
|                                                                                                    |
|  +--------------------+     Offline Augmentation     +------------------------------------------+  |
|  | SPARK / SwissCube  | ---------------------------> | Processed YOLO Dataset                   |  |
|  | Raw Data           |   (Noise, Blur, Exposure)    | (train: 1024ox / val: 1024ox / test: 1024ox) |  |
|  +--------------------+                              +------------------------------------------+  |
|                                                                    |                               |
|                                                                    v                               |
|                                                      +------------------------------------------+  |
|                                                      | YOLOv8-Nano GPU Training Loop            |  |
|                                                      | (Local RTX 2050 CUDA, AMP, RAM Cache)    |  |
|                                                      +------------------------------------------+  |
|                                                                    |                               |
|                                                                    v                               |
|                                                      +------------------------------------------+  |
|                                                      | Trained Checkpoint (best.pt, FP16)       |  |
|                                                      +------------------------------------------+  |
|                                                                    |                               |
|                         +------------------------------------------+--------------------+          |
|                         |                                                               |          |
|                         v                                                               v          |
|          +-----------------------------+                                 +----------------------+  |
|          | edge/export.py              |                                 | src/eval.py          |  |
|          | (Opset 12 FP32 ONNX & NCNN) |                                 | (Validation Harness) |  |
|          +-----------------------------+                                 +----------------------+  |
|                         |                                                               |          |
|                         v                                                               v          |
|          +-----------------------------+                                 +----------------------+  |
|          | Quantization Engine         |                                 | metrics.json         |  |
|          | (ONNX Static / NCNN INT8)   |                                 | (mAP, P/R, FLOPs)    |  |
|          +-----------------------------+                                 +----------------------+  |
+-------------------------|--------------------------------------------------------------------------+
                          | (Deploy binaries & test split via SCP/Drive)
                          v
+----------------------------------------------------------------------------------------------------+
|                                     EDGE HARDWARE (RASPBERRY PI 4)                                 |
|                                                                                                    |
|          +-------------------------------------------------------------+                           |
|          | edge/benchmark.py Harness                                   |                           |
|          | - ≥100 timed runs, 10 warmups                               |                           |
|          | - Multithreaded execution (4 threads, ARM NEON)             |                           |
|          | - Peak RSS Memory (psutil / /usr/bin/time -v)               |                           |
|          +-------------------------------------------------------------+                           |
|                         |                                                                          |
|                         +-------------------------------+                                          |
|                         v                               v                                          |
|          +-----------------------------+ +-----------------------------+                           |
|          | NCNN FP32 & INT8 Engine     | | ONNX Runtime FP32 & INT8    |                           |
|          | (.param / .bin)             | | (.onnx)                     |                           |
|          +-----------------------------+ +-----------------------------+                           |
|                         |                               |                                          |
|                         +---------------+---------------+                                          |
|                                         |                                                          |
|                                         v                                                          |
|                         +-------------------------------+                                          |
|                         | Edge Benchmark Results JSON   |                                          |
|                         | (Mean/p50/p95 latency, FPS)   |                                          |
|                         +-------------------------------+                                          |
+----------------------------------------------------------------------------------------------------+

+----------------------------------------------------------------------------------------------------+
|                               INDEPENDENT CONTEXTUAL AWARENESS LAYER                               |
|                                                                                                    |
|  +--------------------+      Skyfield SGP4       +----------------------+      3D Distance Calc    |
|  | CelesTrak TLEs     | -----------------------> | Geocentric Positions | -----------------------> |
|  | (Live / Cached)    |                          | (X, Y, Z in km)      |                          |
|  +--------------------+                          +----------------------+                          |
+----------------------------------------------------------------------------------------------------+
```

## Architectural Patterns

### 1. Offline Photometric Augmentation Handoff

- **Pattern:** Offline preprocessing with frozen split boundaries.
- **Rationale:** Synthetic space renders have perfect pixel contrast and zero sensor grain. Augmenting pixel intensities (noise, blur, contrast) offline prior to training avoids runtime GPU bottlenecks while strictly preserving 2D bounding box coordinate geometry. Val and test splits remain completely unaugmented to prevent metric corruption.

### 2. Dual-Engine Edge Deployment Strategy (Hypothesis & Fallback)

- **Pattern:** Primary runtime hypothesis paired with standard fallback.
- **Rationale:** Tencent NCNN provides aggressive ARM NEON assembly optimizations for embedded processors, delivering ~21% lower latency than ONNX Runtime on host tests. However, NCNN wheels and INT8 toolchains can be finicky on specific Linux OS distributions. Keeping ONNX Runtime as a fully validated fallback guarantees that the project delivers edge numbers without getting blocked by compiler issues.

### 3. Decoupled Evaluation & Metric Standardization

- **Pattern:** Standardized JSON contract for evaluation.
- **Rationale:** Moving from Ultralytics ad-hoc CSV logs to `src/eval.py` writing a structured `metrics.json` contract allows downstream analysis scripts to programmatically plot accuracy-vs-latency-vs-size curves across PyTorch, ONNX, and NCNN backends without manual transcription.

### 4. Catalog-Based Orbit Awareness (Architectural Isolation)

- **Pattern:** Separate contextual domain service.
- **Rationale:** 2D computer vision bounding boxes cannot physically measure 3D orbital state vectors. Isolating the SGP4 module as an independent catalog awareness layer preserves technical credibility with judges while still demonstrating end-to-end SSA domain context.

---

_Architecture research for: Edge-Deployable Space Debris Detection System_  
_Researched: 2026-10-12_
