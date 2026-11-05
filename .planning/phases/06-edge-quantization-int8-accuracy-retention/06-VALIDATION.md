---
phase: "06"
slug: "edge-quantization-int8-accuracy-retention"
status: ready
nyquist_compliant: true
wave_0_complete: true
created: "2026-11-04"
---

# Phase 06 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution of Phase 6.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Python verification scripts & test harnesses (`src/data/tle_pipeline.py`, `edge/quantize_onnx.py`, `src/eval.py`, `src/infer.py`) |
| **Config file** | `configs/spark2021.yaml` |
| **Quick run command** | `.venv\Scripts\python.exe -c "from src.data.tle_pipeline import OrbitalTracker; t=OrbitalTracker(); assert t.class_to_norad['CubeSat'] is None; print('QUICK CHECK OK')"` |
| **Full suite command** | `.venv\Scripts\python.exe edge/quantize_onnx.py --help && .venv\Scripts\python.exe src/infer.py --help` |
| **Estimated runtime** | ~15–30 seconds |

---

## Sampling Rate

- **After every task commit:** Run task-specific automated verification command
- **After every plan wave:** Run full validation verification
- **Before `/gsd-verify-work`:** All Phase 6 deliverables must possess verifiable artifacts
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---|:---:|:---:|---|---|---|---|---|:---:|:---:|
| 06-01-01 | 01 | 1 | PROD-03 | — | N/A | logic | `.venv\Scripts\python.exe -c "from src.data.tle_pipeline import OrbitalTracker; t = OrbitalTracker(); assert t.class_to_norad['CubeSat'] is None; print('OK')"` | ✅ | ⬜ ready |
| 06-01-02 | 01 | 1 | PROD-03 | — | N/A | doc | `.venv\Scripts\python.exe -c "for p in ['docs/MASTER_PROJECT_REPORT.md', 'docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md']: assert 'decoupled' in open(p, encoding='utf-8').read().lower()"` | ✅ | ⬜ ready |
| 06-01-03 | 01 | 1 | EDGE-02 | — | N/A | doc | `.venv\Scripts\python.exe -c "for p in ['docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md', 'docs/decisions.md']: assert 'Raspberry Pi 5' in open(p, encoding='utf-8').read()"` | ✅ | ⬜ ready |
| 06-01-04 | 01 | 1 | DATA-02 | — | N/A | doc | `.venv\Scripts\python.exe -c "assert 'not ruled out' in open('docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md', encoding='utf-8').read().lower()"` | ✅ | ⬜ ready |
| 06-02-01 | 02 | 2 | EDGE-04 | — | N/A | script | `.venv\Scripts\python.exe edge/quantize_onnx.py --help` | ⬜ | ⬜ ready |
| 06-02-02 | 02 | 2 | EDGE-04 | — | N/A | artifact | `.venv\Scripts\python.exe -c "import os; assert os.path.exists('experiments/spark2021_yolov8n_50e/weights/best_int8.onnx')"` | ⬜ | ⬜ ready |
| 06-02-03 | 02 | 2 | EDGE-05 | — | N/A | retention | `.venv\Scripts\python.exe -c "import json; m=json.load(open('outputs/evaluations/spark2021_int8_test_metrics.json')); assert (0.9874 - m['map50'])*100 <= 3.0"` | ⬜ | ⬜ ready |
| 06-03-01 | 03 | 3 | PROD-01 | — | N/A | cli | `.venv\Scripts\python.exe src/infer.py --help` | ⬜ | ⬜ ready |
| 06-03-02 | 03 | 3 | PROD-01 | — | N/A | unit | `.venv\Scripts\python.exe -c "import src.infer as infer; assert hasattr(infer, 'run_onnx_inference')"` | ⬜ | ⬜ ready |
| 06-03-03 | 03 | 3 | PROD-01 | — | N/A | e2e | `.venv\Scripts\python.exe src/infer.py --source data/processed/spark/images/test/Cheops_image_00005_img.jpg --weights experiments/spark2021_yolov8n_50e/weights/best.onnx --backend onnx"` | ⬜ | ⬜ ready |

*Status: ⬜ ready · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `experiments/spark2021_yolov8n_50e/weights/best.onnx` — frozen candidate FP32 ONNX model
- [x] `configs/spark2021.yaml` — dataset configuration
- [x] `src/data/tle_pipeline.py` — orbital tracker module
- [x] `docs/MASTER_PROJECT_REPORT.md` — audited master technical report

