# Phase 1: Foundations, Evaluation Harness & Day-1 Audit - Context

**Gathered:** 2026-10-13  
**Status:** Ready for planning  
**Collaboration Role:** Data & Edge Lead (Role 2) working in parallel with ML & CV Lead (Role 1), governed by Version 3.0 Work Distribution.

<domain>
## Phase Boundary

Phase 1 establishes the foundational verification and audit assets across both collaborative lanes:
- **ML & CV Lead Lane (Role 1):** Develops standalone `src/eval.py`, reproduces SwissCube baseline metrics (0.9903 mAP@0.5, 0.9211 mAP@0.5:0.95) to `metrics.json`, and merges training scripts.
- **Data & Edge Lead Lane (Role 2 / User):** Executes full SPARK dataset class audit (F1, B-01), confirms hardware availability & installs runtimes on physical Raspberry Pi 4 (B-03), refactors `edge/benchmark.py` for benchmark rigor (B-04), and creates `requirements-edge.txt`.
- **Shared Gate (Thu 15 Oct):** Working `eval.py`, SPARK debris question resolved in `docs/decisions.md`, and first model running on the Raspberry Pi 4.

</domain>

<decisions>
## Implementation Decisions

### SPARK Dataset Audit & Taxonomy Fallback (F1, B-01, S-01)
- **D-01:** Automated row-counting script directly inspecting the full SPARK CSV download to log exact per-class frequencies and definitively verify whether explicit `debris` labels exist.
- **D-02:** Dual-path conditional decision in `docs/decisions.md`:
  - *If debris rows > 0:* Lock 2-class taxonomy (`satellite` / `debris`) with 2–3:1 subsampling on the majority class.
  - *If debris rows = 0:* Reframe project to 10-class spacecraft detection, keep debris coverage strictly qualitative (using SPEED+ or public domain imagery), and explicitly document the absence in reports (F1).  
  — **Reversibility:** `one-way` — downstream Subset B dataset creation and model classification heads directly depend on this taxonomy lock.

### Raspberry Pi 4 Hardware & Runtime Strategy (B-03, F6)
- **D-03:** Target edge platform is confirmed as a physical Raspberry Pi 4 Model B running 64-bit OS (Raspberry Pi OS Debian Bookworm aarch64) with Python 3.10/3.11.
- **D-04:** Dual-runtime installation on the Pi: install `onnxruntime` and precompiled aarch64 `ncnn` wheels on Day 1–2. If `ncnn` Python bindings fail or block by Wed 14 Oct, fall back cleanly to ONNX Runtime as the primary edge target on the Pi, recording NCNN numbers as host proxy (DEC-01).  
  — **Reversibility:** `reversible` — both runtimes are benchmarked in parallel.

### Benchmark Protocol & Metric Contract (B-04, H3, F10)
- **D-05:** Protocol rigor for `edge/benchmark.py`:
  - $\ge 10$ warm-up iterations discarded.
  - $\ge 100$ timed runs with fixed 4 threads.
  - Measure whole-process peak Resident Set Size (RSS memory in MB).
  - Report inference-only latency and end-to-end latency separately.
- **D-06:** Standardized JSON output schema for `edge/benchmark.py` (H3 handoff to Person A):
  ```json
  {
    "model": "yolov8n",
    "format": "ncnn|onnx",
    "precision": "fp32|fp16|int8",
    "imgsz": 640,
    "device": "Raspberry Pi 4",
    "threads": 4,
    "runs": 100,
    "inference_mean_ms": 165.2,
    "inference_median_ms": 162.0,
    "inference_p95_ms": 178.5,
    "inference_std_ms": 7.3,
    "fps": 6.1,
    "peak_rss_mb": 42.5
  }
  ```

### Repository Hygiene (DELV-01, A-02, B-03)
- **D-07:** Create dedicated `requirements-edge.txt` for the Raspberry Pi containing minimal dependencies (`onnxruntime`, `ncnn`, `opencv-python-headless`, `numpy`, `psutil`).
- **D-08:** Separate dataset configs: split `spark.yaml` and create a dedicated `swisscube.yaml` pointing to local processed SwissCube data. Add cached TLE file (`gp.php`) to `.gitignore`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing:**

### Work Distribution & Project Specifications
- `Work Distribution v3.0` (PDF) — Role cards (Role 1: ML & CV, Role 2: Data & Edge), 64-hour task allocation, handoffs (H1, H2, H3), and gates.
- `.planning/PROJECT.md` — Core value, active requirements, and decision register (DEC-01 to DEC-10).
- `.planning/REQUIREMENTS.md` — Traceability for DATA-01, EVAL-01, EVAL-02, EDGE-01, DELV-01.
- `docs/decisions.md` — Active decision log containing Section 2 findings (F1–F12).

### Codebase & Benchmark References
- `edge/export.py` — Current ONNX and NCNN export implementation.
- `edge/benchmark.py` — Baseline benchmarking harness to be refactored for rigor (B-04).
- `src/train.py` — Active YOLOv8 training CLI.
- `PROJECT_DOCUMENTATION.md` §11–§13 — Empirical SwissCube baseline figures and hardware profiles.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `edge/export.py`: Fully functional PyTorch → FP32 ONNX and NCNN exporter with shape verification.
- `edge/benchmark.py`: Existing timing loops with `onnxruntime` and `ncnn` inference extractors; ready for $\ge 100$ runs and peak RSS integration.
- `src/data/explore_dataset.py`: Existing inspection routines for SwissCube and SPARK datasets.

### Established Patterns
- High-precision timing using `time.perf_counter()`.
- Thread configuration via `SessionOptions.intra_op_num_threads` (ONNX) and `Net.opt.num_threads` (NCNN).
- Dataset specification via standard YAML files (`spark.yaml`).

### Integration Points
- `src/eval.py`: Writes `experiments/week1_smoke_test/metrics.json` reproducing SwissCube metrics.
- `edge/benchmark.py`: Outputs structured JSON files into `experiments/` conforming to the D-06 schema.

</code_context>

<specifics>
## Specific Ideas

- Ensure `requirements-edge.txt` does not pull heavy packages like `torch` or `torchvision` onto the Raspberry Pi 4 to save SD card space and compile time.
- Standardize on `640x640` input size for the first Raspberry Pi benchmark run (B-05) to eliminate resolution mismatches.

</specifics>

<deferred>
## Deferred Ideas

- **Subset B Generation (B-02):** Moving raw images, subsampling majority classes to 2–3:1, and freezing train/val/test splits belongs in Phase 2.
- **INT8 Quantization Pipeline (B-06):** Scheduled for Week 2 (Phase 6), tested on SwissCube first before freezing SPARK.

</deferred>

---
*Phase: 01-foundations-evaluation-harness-day-1-audit*  
*Context gathered: 2026-10-13*
