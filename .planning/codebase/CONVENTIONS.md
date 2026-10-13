---
last_mapped_commit: ac3c85127588eab2407698aba0ad22f3abe8167d
last_mapped_at: 2026-10-12
---
# Coding Conventions

**Analysis Date:** 2026-10-12

## Naming Patterns

**Files:**
- Lowercase snake_case: `train.py`, `augment.py`, `export.py`, `benchmark.py`.
- Numbered sequential prefixes for notebooks: `00_colab_setup.ipynb`, `01_data_exploration.ipynb`.

**Functions:**
- Snake_case: `export_and_verify()`, `benchmark_onnx()`, `run_augmentation()`, `calculate_distance()`.

**Variables:**
- Snake_case for standard variables: `dataset_dir`, `avg_latency`, `dummy_input`, `pos_km`.
- UPPERCASE for configuration constants: `IMG_WIDTH`, `IMG_HEIGHT`.

**Types & Classes:**
- PascalCase: `SpaceImageAugmenter`, `OrbitalTracker`.

## Code Style

**Formatting:**
- PEP 8 standard formatting in Python modules (`src/data/augment.py`, `edge/benchmark.py`).
- 4-space indentation throughout.
- Section dividers using comment bars (`# ---------------------------------------------------------------------------` or `# ============================================================`).

**Linting:**
- Not configured (no flake8, black, ruff, or pylint configuration files present).

## Import Organization

**Order:**
1. Standard library imports (`argparse`, `os`, `sys`, `time`, `pathlib`, `shutil`, `json`, `random`, `collections`)
2. Third-party library imports (`numpy`, `torch`, `yaml`, `PIL`, `ultralytics`, `skyfield`, `onnxruntime`, `psutil`)
3. Internal / local module imports (`from src.data.augment import SpaceImageAugmenter, run_augmentation`)

**Path Aliases:**
- Standard absolute project imports rooted at `src` (e.g. `from src.data.augment import ...`).
- Dynamic path adjustment using `sys.path.insert(0, str(repo_root))` observed in scripts to support invocation from subdirectories.

## Error Handling

**Patterns:**
- Explicit validation with standard exceptions: `raise FileNotFoundError(f"Config file not found at {spark_yaml}")` (`src/train.py`, `edge/export.py`).
- Guard clauses with return values / dictionaries: `return {"error": f"Class '{detected_class}' not in mapping dictionary."}` (`src/data/tle_pipeline.py`).
- Try-except with graceful fallback or warnings: `try: import ncnn except ImportError:` (`edge/benchmark.py`).

## Logging

**Framework:** `print()` with visual banners. No dedicated logging framework (e.g., `logging` or `loguru`).

**Patterns:**
- Section demarcation:
  ```python
  print("\n" + "=" * 60)
  print("  [Step 1/2] Wiring & Running Photometric Augmentation Pipeline")
  print("=" * 60)
  ```
- Informational tags: `print("[INFO] Loading active satellites...")`, `print("[WARN] ncnn not installed...")`.

## Comments

**When to Comment:**
- Module docstrings providing purpose, mathematical rationale, domain explanation, and CLI usage instructions (`src/data/augment.py`, `src/train.py`).
- Inline commentary explaining domain-specific constraints (e.g., why rotation invariance is set to 180° for tumbling objects in microgravity).

**Type Annotations:**
- Partial type hinting in modern modules (`weights_path: str, imgsz: int = 640`), but not consistently enforced across legacy scripts.

## Function Design

**Size:**
- Moderate, focused functions (20–60 lines). Long procedural blocks in CLI entry points (`main()`).

**Parameters:**
- Explicit keyword arguments with defaults (`epochs: int = 5, imgsz: int = 1024, batch: int = 8`).

**Return Values:**
- Structured dictionaries returned from analysis / benchmark functions (`edge/benchmark.py`).
- Tuple counts from batch processing (`n_imgs, n_lbls` in `run_augmentation()`).

## Module Design

**Exports:**
- Standalone execution block pattern: Every key module includes an `if __name__ == "__main__":` block with testing simulations or CLI argument parsing.
- Core classes (`SpaceImageAugmenter`, `OrbitalTracker`) designed to be imported cleanly by other modules.

**Barrel Files:**
- Not used (`__init__.py` files are minimal or absent).

---

*Convention analysis: 2026-10-12*
