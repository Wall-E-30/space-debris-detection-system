# Edge Deployment & Benchmarking Guide (Raspberry Pi 4)

This directory contains the deployment assets, export scripts, and benchmarking harnesses for the **Raspberry Pi 4 Model B (4GB / 8GB)** target edge platform.

---

## 1. Raspberry Pi 4 Environment Setup

### Prerequisites
- **Hardware:** Raspberry Pi 4 Model B
- **Operating System:** Raspberry Pi OS (64-bit Debian 12 Bookworm, `aarch64`)
- **Python:** Python 3.10+

### Setup Instructions on the Raspberry Pi
1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/Wall-E-30/space-debris-detection-system.git
   cd space-debris-detection-system
   ```

2. Create and activate a dedicated virtual environment (PEP 668 compliant):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install lightweight edge dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-edge.txt
   ```

4. Verify the edge environment:
   ```bash
   python edge/verify_pi_env.py
   ```

---

## 2. Benchmark Protocol (`edge/benchmark.py`)

Per Decision **D-05 / D-06** and **EDGE-02**, all edge benchmarks adhere to strict measurement standards:
- **Warmup:** $\ge 10$ discarded warmup passes
- **Timed runs:** $\ge 100$ timed inference iterations
- **Threads:** Fixed at 4 threads (utilizing all 4 Cortex-A72 cores)
- **Memory profiling:** Whole-process peak Resident Set Size (`peak_rss_mb` via `psutil`)
- **Output:** Standardized JSON report (`edge_benchmark.json`)

### Running the Benchmark
```bash
python edge/benchmark.py \
  --onnx experiments/week1_smoke_test/weights/best.onnx \
  --ncnn experiments/week1_smoke_test/weights/best_ncnn_model \
  --imgsz 640 \
  --runs 100 \
  --warmup 10 \
  --threads 4 \
  --out edge_benchmark.json
```

---

## 3. Runtimes Evaluated

| Runtime | Target Format | Thread Config | Fallback Policy (DEC-01 / D-04) |
| :--- | :--- | :--- | :--- |
| **NCNN** | `.param` + `.bin` | 4 threads | Primary ARM CPU deployment runtime. |
| **ONNX Runtime** | `.onnx` | 1 and 4 threads | Portable fallback runtime across all edge platforms. |
