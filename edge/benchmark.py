"""
edge/benchmark.py
---
Standardized Edge Inference Benchmark Harness.
Evaluates ONNX Runtime and NCNN inference engines under research-grade rigor:
- >=10 warmup iterations discarded
- >=100 timed runs
- Fixed CPU threads (default 4 for Raspberry Pi 4 Cortex-A72 cores)
- Measures latency statistics (mean, median, p95, std, fps)
- Profiles peak Resident Set Size (RSS memory in MB via psutil)
- Emits standardized JSON adhering to DEC-01 / D-06 contract.
"""

import argparse
import datetime
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil


def benchmark_onnx(
    model_path: str,
    imgsz: int = 640,
    num_runs: int = 100,
    warmup_runs: int = 10,
    threads: int = 4,
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """
    Benchmark ONNX Runtime inference latency and peak memory usage.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[warn] ONNX model not found at {model_file}. Skipping.")
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime not installed. Skipping ONNX benchmark.")
        return None

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    is_cuda = str(device).lower() in ("cuda", "0", "gpu")
    available_providers = ort.get_available_providers()

    if is_cuda and "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        engine_name = "ONNX Runtime (CUDA / GV100)"
    elif is_cuda:
        print("[warn] CUDAExecutionProvider requested but not available in onnxruntime. Falling back to CPU.")
        providers = ["CPUExecutionProvider"]
        engine_name = f"ONNX Runtime (CPU, threads={threads})"
    else:
        providers = ["CPUExecutionProvider"]
        engine_name = f"ONNX Runtime (CPU, threads={threads})"

    session = ort.InferenceSession(
        str(model_file),
        sess_options=opts,
        providers=providers,
    )
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    # Auto-detect and respect static ONNX input dimensions
    if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
        expected_h, expected_w = input_shape[2], input_shape[3]
        if imgsz != expected_h:
            print(f"[INFO] ONNX model has static input resolution {expected_w}x{expected_h}. "
                  f"Auto-adjusting benchmark tensor from {imgsz}x{imgsz} to {expected_w}x{expected_h}.")
            imgsz = expected_h

    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # 1. Warmup runs
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_input})

    # 2. Timed benchmark runs
    latencies: List[float] = []
    process = psutil.Process(os.getpid())

    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    # Measure whole-process peak Resident Set Size in MB
    peak_rss_mb = process.memory_info().rss / (1024.0 * 1024.0)

    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    std_ms = float(np.std(latencies))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "engine": engine_name,
        "format": "onnx",
        "precision": "fp32",
        "model": str(model_file.name),
        "device": "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu",
        "imgsz": imgsz,
        "threads": threads if not is_cuda else 1,
        "runs": num_runs,
        "warmup": warmup_runs,
        "inference_mean_ms": round(mean_ms, 2),
        "inference_median_ms": round(median_ms, 2),
        "inference_p95_ms": round(p95_ms, 2),
        "inference_std_ms": round(std_ms, 2),
        "fps": round(fps, 1),
        "peak_rss_mb": round(peak_rss_mb, 2),
    }


def benchmark_ncnn(
    model_dir: str,
    imgsz: int = 640,
    num_runs: int = 100,
    warmup_runs: int = 10,
    threads: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Benchmark NCNN inference latency and peak memory usage.
    """
    model_path = Path(model_dir)
    param_path = model_path / "model.ncnn.param"
    bin_path = model_path / "model.ncnn.bin"

    if not param_path.exists() or not bin_path.exists():
        print(f"[warn] NCNN param/bin files missing in {model_path}. Skipping.")
        return None

    try:
        import ncnn
    except ImportError:
        print("[warn] ncnn python bindings not installed. Skipping NCNN benchmark.")
        return None

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = threads

    net.load_param(str(param_path))
    net.load_model(str(bin_path))

    dummy_input = np.random.randn(imgsz, imgsz, 3).astype(np.float32)
    in_mat = ncnn.Mat.from_pixels(
        dummy_input.tobytes(),
        ncnn.Mat.PixelType.PIXEL_RGB,
        imgsz,
        imgsz,
    )

    # 1. Warmup runs
    for _ in range(warmup_runs):
        ex = net.create_extractor()
        ex.input("in0", in_mat)
        _ = ex.extract("out0")

    # 2. Timed benchmark runs
    latencies: List[float] = []
    process = psutil.Process(os.getpid())

    for _ in range(num_runs):
        t0 = time.perf_counter()
        ex = net.create_extractor()
        ex.input("in0", in_mat)
        _ = ex.extract("out0")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    # Measure whole-process peak Resident Set Size in MB
    peak_rss_mb = process.memory_info().rss / (1024.0 * 1024.0)

    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    std_ms = float(np.std(latencies))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "engine": f"NCNN (threads={threads})",
        "format": "ncnn",
        "precision": "fp32",
        "model": str(model_path.name),
        "imgsz": imgsz,
        "threads": threads,
        "runs": num_runs,
        "warmup": warmup_runs,
        "inference_mean_ms": round(mean_ms, 2),
        "inference_median_ms": round(median_ms, 2),
        "inference_p95_ms": round(p95_ms, 2),
        "inference_std_ms": round(std_ms, 2),
        "fps": round(fps, 1),
        "peak_rss_mb": round(peak_rss_mb, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Edge Inference Benchmark Harness (DEC-01 / D-05 / D-06)")
    parser.add_argument(
        "--model",
        "--onnx",
        dest="onnx",
        default="experiments/week1_smoke_test/weights/best.onnx",
        help="Path to ONNX weights (.onnx) or model file",
    )
    parser.add_argument(
        "--ncnn",
        default="experiments/week1_smoke_test/weights/best_ncnn_model",
        help="Path to NCNN directory containing model.ncnn.param and model.ncnn.bin",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on: 'cpu' or 'cuda'/'0' (default: cpu)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of timed benchmark runs (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Fixed CPU thread count (default: 4)",
    )
    parser.add_argument(
        "--out",
        "--out-json",
        dest="out",
        type=str,
        default="edge_benchmark.json",
        help="Destination JSON report file path",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  RUNNING EDGE BENCHMARK SUITE (RIGOROUS PROTOCOL)")
    print(f"  Model:             {args.onnx}")
    print(f"  Device:            {args.device}")
    print(f"  Warmup Iterations: {args.warmup}")
    print(f"  Timed Iterations:  {args.runs}")
    print(f"  CPU Threads:       {args.threads}")
    print(f"  Resolution:        {args.imgsz}x{args.imgsz}")
    print("=" * 65)

    results: List[Dict[str, Any]] = []

    # Benchmark ONNX Runtime (Single-thread & Multi-thread, or CUDA)
    if Path(args.onnx).exists():
        is_cuda = str(args.device).lower() in ("cuda", "0", "gpu")
        threads_to_test = [args.threads] if is_cuda else [1, args.threads]
        for t in threads_to_test:
            res = benchmark_onnx(
                model_path=args.onnx,
                imgsz=args.imgsz,
                num_runs=args.runs,
                warmup_runs=args.warmup,
                threads=t,
                device=args.device,
            )
            if res:
                results.append(res)

    # Benchmark NCNN (Multi-thread)
    if Path(args.ncnn).exists():
        res_ncnn = benchmark_ncnn(
            model_dir=args.ncnn,
            imgsz=args.imgsz,
            num_runs=args.runs,
            warmup_runs=args.warmup,
            threads=args.threads,
        )
        if res_ncnn:
            results.append(res_ncnn)

    # Print summary table
    print("\nBenchmark Results Summary:")
    print(f"{'Engine':<28} | {'Mean Latency':<12} | {'p95 Latency':<12} | {'FPS':<8} | {'Peak RSS'}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['engine']:<28} | {r['inference_mean_ms']:>8.2f} ms | "
            f"{r['inference_p95_ms']:>8.2f} ms | {r['fps']:>6.1f} | ~{r['peak_rss_mb']:.1f} MB"
        )
    print("=" * 65 + "\n")

    # Save standardized JSON report
    report: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
        },
        "config": {
            "imgsz": args.imgsz,
            "runs": args.runs,
            "warmup": args.warmup,
            "threads": args.threads,
        },
        "benchmarks": results,
    }

    # -------------------------------------------------------------
    # Persistent Benchmark Archiving (outputs/benchmarks/)
    # -------------------------------------------------------------
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path("outputs") / "benchmarks"
    archive_dir.mkdir(parents=True, exist_ok=True)

    timestamped_file = archive_dir / f"benchmark_{now_str}.json"
    latest_file = archive_dir / "latest.json"
    history_file = archive_dir / "history.jsonl"

    # 1. Save timestamped archive
    with open(timestamped_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 2. Update latest.json
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 3. Append to history ledger
    history_entry = {
        "timestamp": report["timestamp"],
        "platform": report["platform"]["machine"],
        "threads": args.threads,
        "runs": args.runs,
        "imgsz": args.imgsz,
        "benchmarks": [
            {
                "engine": b["engine"],
                "mean_ms": b["inference_mean_ms"],
                "p95_ms": b["inference_p95_ms"],
                "fps": b["fps"],
                "peak_rss_mb": b["peak_rss_mb"],
            }
            for b in results
        ],
        "archive_file": str(timestamped_file.as_posix()),
    }
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry) + "\n")

    print(f"[OK] Benchmark archived to: {timestamped_file.resolve()}")
    print(f"[OK] Latest benchmark updated at: {latest_file.resolve()}")

    # 4. Save to explicit out path
    out_file = Path(args.out)
    if out_file.resolve() != timestamped_file.resolve():
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] Benchmark report also saved to: {out_file.resolve()}\n")


if __name__ == "__main__":
    main()