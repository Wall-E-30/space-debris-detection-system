# edge/benchmark.py
import argparse
import os
import time
from pathlib import Path
import numpy as np
import psutil

def benchmark_onnx(model_path: str, imgsz: int = 640, num_runs: int = 50, threads: int = 4):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark run
    latencies = []
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)

    mem_after = process.memory_info().rss / (1024 * 1024)

    return {
        "engine": f"ONNX Runtime (threads={threads})",
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "fps": 1000.0 / np.mean(latencies),
        "ram_delta_mb": max(0.0, mem_after - mem_before)
    }

def benchmark_ncnn(model_dir: str, imgsz: int = 640, num_runs: int = 50, threads: int = 4):
    try:
        import ncnn
    except ImportError:
        print("[warn] ncnn python bindings not installed. Skipping NCNN benchmark.")
        return None

    param_path = Path(model_dir) / "model.ncnn.param"
    bin_path = Path(model_dir) / "model.ncnn.bin"

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = threads

    net.load_param(str(param_path))
    net.load_model(str(bin_path))

    dummy_input = np.random.randn(imgsz, imgsz, 3).astype(np.float32)
    in_mat = ncnn.Mat.from_pixels(dummy_input.tobytes(), ncnn.Mat.PixelType.PIXEL_RGB, imgsz, imgsz)

    # Warmup
    for _ in range(5):
        ex = net.create_extractor()
        ex.input("in0", in_mat)
        _ = ex.extract("out0")

    latencies = []
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        ex = net.create_extractor()
        ex.input("in0", in_mat)
        _ = ex.extract("out0")
        latencies.append((time.perf_counter() - t0) * 1000)

    mem_after = process.memory_info().rss / (1024 * 1024)

    return {
        "engine": f"NCNN (threads={threads})",
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "fps": 1000.0 / np.mean(latencies),
        "ram_delta_mb": max(0.0, mem_after - mem_before)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Inference Benchmark Harness")
    parser.add_argument("--onnx", default="experiments/week1_smoke_test/weights/best.onnx")
    parser.add_argument("--ncnn", default="experiments/week1_smoke_test/weights/best_ncnn_model")
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  RUNNING EDGE BENCHMARK SUITE")
    print("=" * 65)

    results = []

    # Test thread scaling (Single-thread vs Multi-thread)
    for t in [1, 4]:
        if Path(args.onnx).exists():
            res = benchmark_onnx(args.onnx, num_runs=args.runs, threads=t)
            results.append(res)

    if Path(args.ncnn).exists():
        res_ncnn = benchmark_ncnn(args.ncnn, num_runs=args.runs, threads=4)
        if res_ncnn:
            results.append(res_ncnn)

    print("\nBenchmark Results Summary:")
    print(f"{'Engine':<28} | {'Mean Latency':<12} | {'p95 Latency':<12} | {'FPS':<8} | {'RAM Footprint'}")
    print("-" * 75)
    for r in results:
        print(f"{r['engine']:<28} | {r['mean_ms']:>8.2f} ms | {r['p95_ms']:>8.2f} ms | {r['fps']:>6.1f} | ~{r['ram_delta_mb']:.1f} MB")
    print("=" * 65 + "\n")