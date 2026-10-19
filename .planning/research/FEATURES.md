# Feature Research

**Domain:** Edge AI / Space Debris Object Detection & Edge Optimization  
**Researched:** 2026-10-12  
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features expected for Problem Statement #05 and rigorous edge evaluation. Missing these = system feels incomplete or unconvincing to judges.

| Feature                                                 | Why Expected                                                                                                   | Complexity | Notes                                                                                                |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------- |
| **Independent Evaluation Harness (`src/eval.py`)**      | PRD requires reproducible accuracy logging decoupled from training logs; needed for INT8 retention validation. | LOW        | Outputs `metrics.json` containing mAP50, mAP50-95, per-class P/R, model size, parameters, FLOPs.     |
| **Reproducible SPARK Multi-Class Dataset Pipeline**     | Moving beyond single-object SwissCube to multi-class spacecraft detection on SPARK.                            | MEDIUM     | Frozen stratified train/val/test splits, near-duplicate checks, majority-class subsampling to 2-3:1. |
| **Edge Export Pipeline (FP32 ONNX & NCNN)**             | Translating research PyTorch `.pt` checkpoints into edge-deployable formats.                                   | LOW        | Handled via `edge/export.py` with automated dummy forward-pass validation.                           |
| **Physical Raspberry Pi 4 Latency & Memory Benchmarks** | Demonstrates real edge feasibility, avoiding misleading host CPU proxy numbers.                                | MEDIUM     | ≥100 timed runs after warm-up, reporting mean, p50, p95 latency, and peak resident memory (RSS).     |
| **End-to-End Inference CLI (`src/infer.py`)**           | Standard developer interface: input image or directory → outputs bounding box overlays + structured JSON.      | LOW        | Multi-backend support: switches between PyTorch, ONNX, and NCNN.                                     |
| **Complete Metrics Logging & Tradeoff Analysis**        | Comparing accuracy, latency, model size, and FLOPs together across runtimes and resolutions.                   | LOW        | Script-driven generation of accuracy vs latency vs size charts from `metrics.json`.                  |

### Differentiators (Competitive Advantage)

Features that elevate the submission above typical student computer vision projects.

| Feature                                                     | Value Proposition                                                                                                                   | Complexity | Notes                                                                                                           |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------- |
| **INT8 Post-Training Quantization with Measured Retention** | Compresses model file size to ≤ 5 MB (NFR-3) and verifies mAP drop is within ≤ 3 points (NFR-2).                                    | MEDIUM     | Uses 300–500 representative calibration images; evaluates on held-out test split.                               |
| **Multi-Resolution Tradeoff Comparison (1024ox vs 416px)**  | Shows engineering maturity by evaluating latency vs accuracy scaling on edge hardware.                                              | LOW        | 416px variant reduces FLOPs by ~58%, serving as a low-latency edge candidate.                                   |
| **Photometric Domain-Gap Augmentation Pipeline**            | Narrows the synthetic-to-real domain gap without distorting bounding box coordinates.                                               | LOW        | `src/data/augment.py` injects sensor read noise, optical defocus blur, and exposure jitter on train split only. |
| **Qualitative Real-Image Validation**                       | Honest qualitative evaluation on a small real-image test set (~20–30 public/SPEED+ images) to document where synthetic models fail. | LOW        | Establishes domain boundaries without claiming false high accuracy on unlabelled real data.                     |
| **FPGA Feasibility Note**                                   | Provides forward-looking systems architecture analysis for radiation-hardened space hardware (e.g. Xilinx Zynq / Microsemi RTG4).   | LOW        | 1–2 page analytical note mapping measured FLOPs to FPGA DSP slices and power envelopes.                         |
| **Independent SGP4 Catalog Awareness Demo**                 | Shows broader Space Situational Awareness (SSA) system context by calculating 3D orbit separation from live CelesTrak TLEs.         | LOW        | Preserved as an independent contextual dashboard layer (DEC-07).                                                |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem appealing in initial proposals but derail delivery or undermine credibility.

| Feature                                        | Why Requested                                                 | Why Problematic                                                                                                                 | Alternative                                                          |
| ---------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Image-to-Orbit Conjunction Claims**          | Desired to show "automated collision avoidance from cameras". | Mathematically invalid: 2D bounding boxes do not carry physical 3D range, orbital state vectors, or velocities.                 | Frame SGP4 orbit tracking as an independent catalog awareness layer. |
| **Heavy Models (YOLOv8x, DETR, Transformers)** | Desired for higher raw leaderboard mAP.                       | Completely unviable on 4GB local GPU and will freeze or drop to <0.5 FPS on Raspberry Pi 4.                                     | Stick firmly to YOLOv8-Nano.                                         |
| **Optical Flow & Multi-Frame Tracking**        | Desired for velocity estimation across frames.                | Explodes development scope, adds frame synchronization dependencies, and is not on the detect→quantize→benchmark critical path. | Omit for 4-week scope; list as future work.                          |
| **Hardware FPGA Build**                        | Desired for "space-grade hardware demonstration".             | Toolchains (Vivado/Vitis) and RTL development take months; high probability of failure in 4 weeks.                              | Short written FPGA feasibility note based on measured FLOPs.         |

## Feature Dependencies

```
[Frozen SPARK Dataset (Subset B)]
       └──requires──> [Class Audit / Taxonomy Resolution (F1)]
       └──requires──> [Photometric Augmentation Pipeline]

[YOLOv8-Nano Training (1024ox & 416px)]
       └──requires──> [Frozen SPARK Dataset (Subset B)]

[Independent Evaluation (eval.py)]
       └──requires──> [Model Checkpoint (.pt / .onnx / ncnn)]
       └──requires──> [Held-out Test Split]

[INT8 Quantization]
       └──requires──> [Model Checkpoint]
       └──requires──> [Calibration Dataset Subset (300-500 images)]

[Physical Raspberry Pi Benchmark]
       └──requires──> [Exported Models (FP32 & INT8)]
       └──requires──> [benchmark.py (≥100 runs, RSS memory)]

[CLI & Interactive Demo]
       └──requires──> [Exported Backends (PyTorch, ONNX, NCNN)]
```

---

_Feature research for: Edge-Deployable Space Debris Detection System_  
_Researched: 2026-10-12_
