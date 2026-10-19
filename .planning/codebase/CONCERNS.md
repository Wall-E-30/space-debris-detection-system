---
last_mapped_commit: ac3c85127588eab2407698aba0ad22f3abe8167d
last_mapped_at: 2026-10-12
---

# Codebase Concerns

**Analysis Date:** 2026-10-12

## Tech Debt

**Empty Evaluation Module (`src/eval.py`):**

- Issue: `src/eval.py` exists as a 0-byte blank file. Metric extraction currently depends entirely on internal training logs (`results.csv`) rather than an independent, repeatable verification script outputting standardized `metrics.json`.
- Files: `src/eval.py`
- Impact: Blocks automated model evaluation on holdout test sets and independent post-quantization validation.
- Fix approach: Implement an evaluation harness in `src/eval.py` that computes mAP@0.5, mAP@0.5:0.95, per-class metrics, and confusion matrices against `spark.yaml`.

**Hardcoded Absolute Paths:**

- Issue: `spark.yaml` contains a hardcoded absolute Windows directory path (`path: C:/Users/Sharanya Nagar/...`).
- Files: `spark.yaml`, `train_gpu_opt_a.py`
- Impact: Breaks execution on other machines, Linux environments, or edge platforms (Raspberry Pi).
- Fix approach: Update `spark.yaml` and dataset loaders to use relative paths resolved against the repository root.

**Redundant / Duplicate Training Scripts:**

- Issue: `train_gpu_opt_a.py` is duplicated at the root and inside `notebooks/train_gpu_opt_a.py`, while `src/train.py` serves as a third variant.
- Files: `train_gpu_opt_a.py`, `notebooks/train_gpu_opt_a.py`, `src/train.py`
- Impact: Maintenance overhead and risk of divergence between experimental runs.
- Fix approach: Consolidate all training configurations into `src/train.py` and remove duplicate root scripts.

## Known Bugs

**Unlisted Dependency (`skyfield`):**

- Symptoms: `ModuleNotFoundError: No module named 'skyfield'` when running `src/data/tle_pipeline.py`.
- Files: `src/data/tle_pipeline.py`, `requirements.txt`
- Trigger: Installing from `requirements.txt` into a clean environment and running the TLE pipeline.
- Workaround: Manually running `pip install skyfield`. Needs addition to `requirements.txt`.

**CelesTrak Ephemeris Fetch Failure without Internet:**

- Symptoms: `src/data/tle_pipeline.py` crashes if the external network is unavailable when invoking `load.tle_file('https://celestrak.org/...')`.
- Files: `src/data/tle_pipeline.py`
- Trigger: Running edge inference in an offline or air-gapped environment.
- Workaround: Use the local cached `gp.php` file when network calls fail.

## Security Considerations

**Unauthenticated External HTTP Calls:**

- Risk: `src/data/tle_pipeline.py` downloads ephemeris data directly from an external HTTP/HTTPS URL (`https://celestrak.org/...`) without integrity checksums or pinned certificate validation.
- Files: `src/data/tle_pipeline.py`
- Current mitigation: Standard SSL via Python requests/urllib.
- Recommendations: Implement caching with cryptographic checksum verification and offline fallback to local TLE files.

## Performance Bottlenecks

**GPU VRAM Limitation on Local Workstation:**

- Problem: Training batch size is capped at 8 for $1024 \times 1024$ images on the 4GB RTX 2050 GPU.
- Files: `src/train.py`, `train_gpu_opt_a.py`
- Cause: High activation memory consumption at $1024 \times 1024$ image resolutions.
- Improvement path: Lower resolution to 1024ox for edge training or migrate multi-class SPARK training (~150k images) to cloud GPU instances.

**Synthetic-to-Real Domain Gap:**

- Problem: SwissCube smoke test achieved near-perfect mAP@0.5 (0.9903) due to synthetic render uniformity; will not generalize to multi-target orbital clutter or real telescope sensor noise.
- Files: `src/data/augment.py`, `docs/decisions.md`
- Cause: Clean 3D CAD renders against dark backgrounds without atmospheric turbulence or sensor artifacts.
- Improvement path: Continue tuning photometric transforms and transition to the multi-class SPARK dataset with star streaks.

## Fragile Areas

**Orbital Tracker Class-to-NORAD Lookup Table:**

- Files: `src/data/tle_pipeline.py`
- Why fragile: Hardcoded dictionary mapping specific SPARK classes to NORAD IDs. Any dataset taxonomy change or missing class returns an error dictionary.
- Safe modification: Externalize mapping to a configuration file (e.g., `config/norad_mapping.yaml`) with robust default fallbacks.
- Test coverage: No automated unit tests cover this mapping.

## Scaling Limits

**Local Dataset Processing:**

- Current capacity: ~20,000 SwissCube images handled comfortably in local RAM and storage.
- Limit: Full multi-class SPARK dataset (~150,000 images, >50 GB) exceeds local RAM caching capabilities (`cache=True`).
- Scaling path: Disable RAM caching (`cache=False` or `cache='disk'`) and use memory-mapped tensors.

## Dependencies at Risk

**`ncnn` Python Bindings:**

- Risk: `ncnn` Python wheels often require manual compilation or specific OS wheels on ARM/Windows.
- Files: `edge/benchmark.py`
- Impact: Benchmark falls back to ONNX Runtime if `ncnn` package is absent.
- Migration plan: Document pre-compiled wheel installation instructions or use C++ binary benchmarking harness on Raspberry Pi.

## Missing Critical Features

**End-to-End Evaluation Script:**

- Problem: `src/eval.py` is empty (0 bytes).
- Blocks: Automated calculation of validation metrics, mAP@0.5, and post-export drift verification.

**Inference & Visualization CLI:**

- Problem: No standalone inference script (`src/infer.py`) to process single images or video streams and output annotated bounding boxes.
- Blocks: Quick qualitative testing and demonstration.

**INT8 Quantization Pipeline:**

- Problem: Current exports are FP32 only; INT8 calibration subset pass is missing.
- Blocks: Meeting edge memory (<5 MB) and latency goals on physical Raspberry Pi hardware.

## Test Coverage Gaps

**Zero Automated Unit/Integration Tests:**

- What's not tested: Data split stratification, label format conversion, photometric transform boundaries, SGP4 math calculations.
- Files: `src/data/split_dataset.py`, `src/data/make_labels.py`, `src/data/augment.py`, `src/data/tle_pipeline.py`
- Risk: Breaking changes introduced during refactoring or SPARK dataset onboarding will go unnoticed.
- Priority: High.

---

_Concerns audit: 2026-10-12_
