# Phase 08 Context: End-to-End Inference CLI, Demo & Astrodynamics Layer

## Phase Objective
Build a unified inference CLI, curate a qualitative real-world space imagery validation dataset with recorded licenses, develop an interactive demonstration application with real-time inference telemetry, and integrate the decoupled SGP4 orbital tracking catalog layer (DEC-05).

## Key Requirements
- **DATA-04**: Assemble a small qualitative real-world space imagery set (~20–30 open-access images) from public NASA, ESA, Roscosmos, and ISRO archives to benchmark domain gap.
- **PROD-01**: Multi-backend inference tool supporting PyTorch (`.pt`) and ONNX Runtime (`.onnx` FP32 and INT8) with dynamic tensor shape routing.
- **PROD-02**: Interactive demonstration web application with real-time bounding box visualization, confidence filtering, latency, and model size telemetry.
- **PROD-03**: Decoupled astrodynamics panel integrating `src/data/tle_pipeline.py` displaying live NORAD catalog data and 3D ECI orbital state vectors for known assets while maintaining physical honesty for uncataloged debris.

## Dependencies & Architecture
- Uses frozen weights: `experiments/spark2021_yolov8n_50e/weights/best.pt`, `best.onnx`, and `best_int8.onnx`.
- Uses dataset configuration: `configs/spark2021.yaml` (11 classes).
- Uses offline TLE cache: `data/tle/celestrak_active.tle`.
