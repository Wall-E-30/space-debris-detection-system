# Phase 01: Foundations, Evaluation Harness & Day-1 Audit - Research

**Researched:** 2026-10-13  
**Domain:** Model Evaluation Harness, Dataset Auditing & Embedded SBC Benchmarking  
**Confidence:** HIGH  

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Scripted audit of local/mounted SPARK CSV inspecting full label counts directly to definitively verify whether explicit `debris` labels exist (B-01).
- **D-02:** Dual-path taxonomy fallback policy in `docs/decisions.md`:
  - If debris rows > 0: Lock 2-class (`satellite` / `debris`) with 2–3:1 subsampling.
  - If debris rows = 0: Reframe project openly to 10-class spacecraft detection, show debris purely qualitatively (SPEED+), and state limits transparently in reports (F1).
- **D-03:** Raspberry Pi 4 Model B running 64-bit OS (Raspberry Pi OS Debian Bookworm aarch64) with Python 3.10/3.11 confirmed as target edge platform.
- **D-04:** Install `onnxruntime` and precompiled aarch64 `ncnn` wheels on Day 1–2. If `ncnn` wheels block on the Pi by Wed 14 Oct, fall back cleanly to ONNX Runtime as the primary edge target on the Pi, recording NCNN host numbers as proxy (DEC-01).
- **D-05:** Protocol rigor for `edge/benchmark.py`: $\ge 10$ warm-up iterations discarded, $\ge 100$ timed runs with fixed 4 threads, whole-process peak Resident Set Size (`peak_rss_mb`), and separate inference-only vs end-to-end latency.
- **D-06:** Standardized JSON output schema for `edge/benchmark.py` (H3 handoff to Person A):
  `{"model", "format", "precision", "imgsz", "device", "threads", "runs", "inference_mean_ms", "inference_median_ms", "inference_p95_ms", "inference_std_ms", "fps", "peak_rss_mb"}`.
- **D-07:** Create dedicated `requirements-edge.txt` for the Raspberry Pi containing minimal dependencies (`onnxruntime`, `ncnn`, `opencv-python-headless`, `numpy`, `psutil`).
- **D-08:** Separate dataset configs: split `spark.yaml` and create dedicated `swisscube.yaml` pointing to local processed SwissCube data. Add cached TLE file (`gp.php`) to `.gitignore`.

### the agent's Discretion
- Exact CLI flags for `src/eval.py` (`--weights`, `--data`, `--split`, `--imgsz`, `--device`, `--out`).
- Internal CSV parsing mechanisms for the SPARK class auditing script.

### Deferred Ideas (OUT OF SCOPE)
- Generating Subset B images and freezing splits belongs in Phase 2 (B-02).
- Building INT8 calibration and quantization pipeline belongs in Phase 6 (B-06).
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Model Evaluation (`src/eval.py`) | Model / Host Workstation | Edge SBC | Independent verification script run on GPU host workstation against `.pt`, `.onnx`, and `ncnn` models. |
| SPARK CSV Audit & Taxonomy | Data Engineering / Host | Documentation | Row frequency analysis on host filesystem committed directly to `docs/decisions.md`. |
| Edge Runtime & Benchmark | Edge Hardware (Pi 4) | Host Proxy | Latency and peak RSS profiling executed on Raspberry Pi 4 CPU with local fallback proxy. |
| Repo Hygiene & Configs | Repository / Infrastructure | — | Clean separation of environment manifests and dataset YAML pointers. |
</architectural_responsibility_map>

<research_summary>
## Summary

Phase 1 establishes the empirical foundation for the 4-week project by tackling the four critical review findings (F1, F2, F6, F11):
1. **Independent Evaluation (`src/eval.py`):** Currently, metric tracking relies entirely on internal Ultralytics training logs (`results.csv`). Implementing a standalone evaluation script wrapping `YOLO.val()` that extracts mAP@0.5, mAP@0.5:0.95, per-class P/R, parameter counts, and GFLOPs to `metrics.json` provides an uncorrupted ground truth. Verifying this script on `experiments/week1_smoke_test/weights/best.pt` must reproduce the documented SwissCube baseline metrics (0.9903 mAP@0.5 / 0.9211 mAP@0.5:0.95).
2. **SPARK CSV Class Audit (F1, B-01):** The project title claims "space debris detection", yet the initial audit of the 80k SPARK subset showed only 10 satellite spacecraft classes. An automated audit of the full SPARK CSV will verify whether explicit debris instances exist. If absent, DEC-04 reframes the project transparently to satellite/space-object detection, avoiding disqualified over-claims.
3. **Raspberry Pi 4 Setup & Benchmark Rigor (F6, F10, B-03, B-04):** The previous 165ms NCNN measurement was conducted on a laptop CPU proxy. Refactoring `edge/benchmark.py` for $\ge 100$ timed runs, 4 threads, and peak resident memory (RSS), and installing wheels on the physical Raspberry Pi 4 ensures that edge numbers are credible and reproducible.

**Primary recommendation:** Implement `src/eval.py` first and verify baseline reproduction; concurrently run the automated SPARK CSV audit to resolve taxonomy before dataset subsetting in Phase 2.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `ultralytics` | 8.0+ | Model evaluation and inference | Provides `YOLO(weights).val(data=..., split=..., plots=True)` returning structured metric boxes (`map50`, `map50_95`, `mp`, `mr`). |
| `onnxruntime` | 1.15+ | Edge inference and graph execution | Universal cross-platform runtime for validation on host and ARM64 Raspberry Pi. |
| `ncnn` | Latest aarch64 | ARM NEON optimized edge inference | Tencent's high-speed neural network library for mobile/embedded platforms. |
| `psutil` | 5.9+ | Process memory tracking | Retrieves peak Resident Set Size (`rss`) to accurately gauge physical edge RAM usage. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pyyaml` | 6.0+ | Configuration parsing | Loading and rewriting `swisscube.yaml` and `spark.yaml`. |
| `json` | Built-in | Structured metrics logging | Saving `metrics.json` and benchmark results. |

**Installation:**
```bash
# Host Workstation
pip install ultralytics pyyaml psutil skyfield

# Raspberry Pi 4 (from requirements-edge.txt)
pip install onnxruntime opencv-python-headless numpy psutil
pip install ncnn
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Evaluation Harness Contract (`src/eval.py`)
```python
# Standard evaluation workflow outputting metrics.json
import json, argparse, pathlib
from ultralytics import YOLO

def evaluate(weights_path, data_config, split="val", imgsz=640, device="cpu", out_json="metrics.json"):
    model = YOLO(weights_path)
    res = model.val(data=data_config, split=split, imgsz=imgsz, device=device, plots=True)
    b = res.box
    
    metrics = {
        "weights": str(weights_path),
        "split": split,
        "imgsz": imgsz,
        "map50": float(b.map50),
        "map50_95": float(b.map),
        "precision": float(b.mp),
        "recall": float(b.mr),
        "per_class": {
            res.names[c]: {
                "p": float(b.p[i]),
                "r": float(b.r[i]),
                "ap50": float(b.ap50[i])
            }
            for i, c in enumerate(b.ap_class_index)
        }
    }
    
    w = pathlib.Path(weights_path)
    if w.is_file():
        metrics["size_mb"] = round(w.stat().st_size / (1024 * 1024), 2)
    if str(weights_path).endswith(".pt"):
        _, params, _, gflops = model.info(verbose=False)
        metrics["params"] = params
        metrics["gflops"] = gflops
        
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
```

### Benchmark Memory Measurement Pattern
Avoid measuring Python heap delta (`mem_after - mem_before`), which frequently reports 0.0 MB for C++ runtimes like ONNX and NCNN. Instead, measure whole-process peak Resident Set Size:
```python
import psutil, os
process = psutil.Process(os.getpid())
peak_rss_mb = process.memory_info().rss / (1024 * 1024)
```
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| mAP & PR Curve Calculation | Custom IoU/mAP calculation | `model.val()` from Ultralytics | Standardized COCO metric computation with interpolation and area under curve math. |
| CSV Parsing | Raw line-splitting strings | Python `csv.DictReader` or `pandas` | Handles quoted columns, commas inside fields, and escape characters robustly. |
| Thread Affinity on ARM | OS-level taskset hacks | `opts.intra_op_num_threads = 4` & `net.opt.num_threads = 4` | Direct runtime thread pool management is cleaner and avoids process priority collisions. |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: False Verification of `eval.py`
**What goes wrong:** `src/eval.py` is executed, but on a different split, different resolution, or different data YAML than `experiments/week1_smoke_test`, producing mismatched mAP numbers that cause confusion.
**How to avoid:** Specifically evaluate `experiments/week1_smoke_test/weights/best.pt` using `swisscube.yaml` at `imgsz=1024` on `split=val` to verify that it reproduces 0.9903 mAP@0.5 and 0.9211 mAP@0.5:0.95 within rounding tolerance.

### Pitfall 2: Silent Debris Omission in SPARK Subsets
**What goes wrong:** Subsetting code only iterates through subfolders or classes it expects, silently dropping debris classes if they use a different naming convention.
**How to avoid:** Script inspects all unique values in the class column of the full raw CSV rather than filtering by known satellite names.

### Pitfall 3: NCNN Python Wheel Dependency Conflicts on Pi OS Bookworm
**What goes wrong:** Debian 12 (Bookworm) enforces PEP 668 (externally managed environment), blocking raw `pip install`.
**How to avoid:** Run installation inside a dedicated Python virtual environment (`.venv/`) on the Raspberry Pi 4.
</common_pitfalls>

<sources>
## Sources
### Primary (HIGH confidence)
- Status Review & Revised 4-Week Plan (v2.1, October 2026) — Sections 1–3, F1–F12.
- Work Distribution v3.0 (PDF) — ML & CV Lead vs Data & Edge Lead task allocations.
- Ultralytics YOLOv8 Documentation — Validation mode (`model.val`) and export attributes.

### Secondary (MEDIUM confidence)
- Raspberry Pi OS Debian 12 Bookworm aarch64 Python packaging standards.
</sources>

---
*Phase: 01-foundations-evaluation-harness-day-1-audit*  
*Research completed: 2026-10-13*  
*Ready for planning: yes*
