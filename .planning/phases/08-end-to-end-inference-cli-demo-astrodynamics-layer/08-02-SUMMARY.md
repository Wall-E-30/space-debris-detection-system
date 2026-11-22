# Plan 08-02 Summary: Interactive Demonstration Application & Astrodynamics Panel

## Overview
Constructed a lightweight, self-contained interactive web demonstrator (`demo/app.py`) for live space object detection, edge telemetry HUD monitoring, and decoupled SGP4 orbital tracking awareness.

## Key Deliverables
1. **Interactive Web Application (`demo/app.py`):**
   - Built using Python's standard library `http.server` with zero heavy external dependencies (no Gradio/Streamlit requirements), enabling native execution on both developer workstations and Raspberry Pi 4 / Pi 5 SBCs.
   - Modern dark-mode UI with preset scene selection (real-world validation and SPARK test scenes) and custom image upload.
   - Model runtime selector: ONNX Static INT8 (3.59 MB) and ONNX FP32 (12.33 MB).
   - Dynamic confidence gate (0.10 to 0.95) and NMS IoU threshold (0.10 to 0.80) sliders.
   - Real-time Telemetry HUD: Latency (ms), Throughput (FPS), Model Footprint (MB), and Localized Object Count.
   - Decoupled Astrodynamics Panel:
     - For verified cataloged spacecraft (e.g. Sentinel-6, Terra, Calipso): displays live CelesTrak NORAD ID, epoch timestamp, and 3D ECI coordinates ($X, Y, Z$ in km) computed via SGP4.
     - For uncataloged space debris fragments and multi-manifest CubeSats: explicitly presents the physical honesty disclosure (DEC-05).
2. **Automated Unit Test Suite (`tests/test_demo_app.py`):**
   - 5 comprehensive automated tests covering preset discovery, bounding box rendering, `/api/health`, `/api/presets`, and `/api/infer` endpoints.
   - Tests execute with proxy-bypass loopback handler.

## Verification
- `pytest tests/test_demo_app.py` -> 5 passed in 3.17s.
- Full pytest suite -> 39 passed in 5.16s (100% pass rate).
- Flake8 lint -> 0 errors.
