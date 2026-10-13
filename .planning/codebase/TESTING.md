---
last_mapped_commit: ac3c85127588eab2407698aba0ad22f3abe8167d
last_mapped_at: 2026-10-12
---
# Testing Patterns

**Analysis Date:** 2026-10-12

## Test Framework

**Runner:**
- Not configured / Ad-hoc execution (No pytest or unittest framework configured in repository)
- Config: None (`pytest.ini` or `setup.cfg` missing)

**Assertion Library:**
- Built-in Python statements and verification scripts

**Run Commands:**

```bash

# Run standalone module self-tests:

python src/data/augment.py --preview-only           # Visual preview of augmentation transforms
python src/data/tle_pipeline.py                     # SGP4 propagation & distance verification
python edge/export.py                               # Model export & ONNX forward-pass verification
python edge/benchmark.py --runs 10                  # Latency and throughput benchmark
```

## Test File Organization

**Location:**
- Inline in script modules under `if __name__ == "__main__":` blocks. No dedicated `tests/` directory exists.

**Naming:**
- Not applicable (no `test_*.py` files present).

**Structure:**
- Functional execution and sanity check scripts.

## Test Structure

**Suite Organization:**

```python

# Self-test pattern in module __main__ (from src/data/tle_pipeline.py)

if __name__ == "__main__":
    tracker = OrbitalTracker()
    obj1 = tracker.get_position("Cheops")
    obj2 = tracker.get_position("ObservationSat1")
    if "error" not in obj1 and "error" not in obj2:
        dist = tracker.calculate_distance("Cheops", "ObservationSat1")
        print(f"[ALERT] Distance between objects: {dist:,.2f} km")
```

**Patterns:**
- Synthetic dummy tensor generation (`dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)` in `edge/export.py`).
- Warmup passes before benchmarking (`edge/benchmark.py`).

## Mocking

**Framework:**
- None / Manual proxies.

**Patterns:**
- Proxy NORAD IDs used for historical/deep-space probes (e.g. `ObservationSat1` mapped to NORAD 25544 / ISS proxy) to ensure SGP4 math functions for orbital conjunction simulation (`src/data/tle_pipeline.py`).

**What to Mock:**
- High-rate orbital telemetry when offline.

**What NOT to Mock:**
- Image array transforms; ONNX graph execution.

## Fixtures and Factories

**Test Data:**
- Synthetic dummy inputs created via `numpy.random.randn()` for inference validation.
- Smoke test dataset: SwissCube dataset subset in `data/processed/Swisscube/`.

**Location:**
- Generated in memory or read from `data/processed/`.

## Coverage

**Requirements:** None enforced.

**View Coverage:**
- Not configured.

## Test Types

**Unit Tests:**
- Not formalized into unit test suites.

**Integration Tests:**
- `edge/export.py` acts as an integration test verifying end-to-end model export and ONNX runtime forward execution.
- `src/train.py` verifies the photometric augmentation integration step before launching training.

**E2E Tests:**
- `edge/benchmark.py` measures complete inference cycle throughput and latency under multi-threaded execution.

## Common Patterns

**Inference Verification:**

```python

# Forward-pass verification with dummy tensor (from edge/export.py)

session = ort.InferenceSession(str(onnx_file), providers=['CPUExecutionProvider'])
dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
output = session.run(None, {input_name: dummy_input})
```

**Latency Profiling:**

```python

# Latency measurement over warm iterations (from edge/benchmark.py)

for _ in range(num_runs):
    t0 = time.perf_counter()
    _ = session.run(None, {input_name: dummy_input})
    latencies.append((time.perf_counter() - t0) * 1000)
```

---

*Testing analysis: 2026-10-12*
