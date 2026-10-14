---
phase: "01"
plan: "03"
status: complete
completed_at: "2026-10-14"
requirements:
  - EDGE-01
  - DELV-01
files_produced:
  - requirements-edge.txt
  - requirements.txt
  - configs/swisscube.yaml
  - swisscube.yaml
  - configs/spark.yaml
  - .gitignore
  - edge/benchmark.py
  - edge/verify_pi_env.py
  - edge/README.md
---

# Plan 01-03 Summary: Repo Hygiene & Edge Benchmark Harness

## Overview
Resolved repository hygiene and configuration separation by creating `requirements-edge.txt`, updating root `requirements.txt`, isolating dataset configs (`configs/swisscube.yaml` and `configs/spark.yaml`), and ignoring `gp.php`. Upgraded `edge/benchmark.py` to research-grade standards (>=10 warmup, >=100 timed runs, 4 threads, peak RSS memory via `psutil`), and created testbed environment verification utility `edge/verify_pi_env.py`.

## Key Deliverables
- `requirements-edge.txt`: Isolated dependencies for Raspberry Pi 4 edge deployment (`onnxruntime`, `opencv-python-headless`, `numpy`, `psutil`, `pyyaml`).
- `requirements.txt`: Root requirements updated with `psutil`, `skyfield`, and `ncnn`.
- `configs/swisscube.yaml` & `swisscube.yaml`: Clean dataset configuration for SwissCube.
- `configs/spark.yaml`: Clean 10-class dataset configuration for SPARK.
- `.gitignore`: Updated to ignore `gp.php`, binary weights (`*.bin`, `*.param`, `*.onnx`, `*.zip`), and cache folders.
- `edge/benchmark.py`: Standardized benchmark harness measuring latency (mean, median, p95, std, fps) and peak RSS memory.
- `edge/verify_pi_env.py`: Testbed verification script checking Python >= 3.10, ARM aarch64, RAM/cores, and module imports.
- `edge/README.md`: Step-by-step Raspberry Pi 4 deployment and benchmarking guide.

## Requirements Satisfied
- **EDGE-01**: Raspberry Pi 4 edge environment setup and verification protocol established.
- **DELV-01**: Repository hygiene resolved, configuration files separated, and transient download scripts ignored.
