# edge/export.py
from pathlib import Path
import time
import numpy as np
from ultralytics import YOLO

def export_and_verify(weights_path: str, imgsz: int = 640):
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights.resolve()}")

    print(f"\n{'='*60}")
    print(f"  [1/3] Loading PyTorch Model: {weights.name}")
    print(f"{'='*60}")
    model = YOLO(str(weights))

    # --- 1. Export to ONNX (Fallback runtime) ---
    print("\n[2/3] Exporting to ONNX format...")
    onnx_file = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        half=False # Keep FP32 for broad CPU compatibility
    )
    print(f"      Saved ONNX -> {onnx_file}")

    # --- 2. Export to NCNN (Primary Raspberry Pi runtime) ---
    print("\n[3/3] Exporting to NCNN format...")
    ncnn_file = model.export(
        format="ncnn",
        imgsz=imgsz,
        half=False
    )
    print(f"      Saved NCNN -> {ncnn_file}")

    # --- 3. Verify Inference with ONNX Runtime ---
    print("\n--- Testing Forward Pass via ONNX Runtime ---")
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_file), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"      ONNX Input node: '{input_name}', expected shape: {input_shape}")

    # Create synthetic blank frame (Batch=1, Channels=3, H=imgsz, W=imgsz)
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # Measure CPU latency over 10 warm runs
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = np.mean(latencies)
    print(f"\n[SUCCESS] Local CPU ONNX Forward pass verified.")
    print(f"          Average Latency (CPU): {avg_latency:.2f} ms")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Point directly to your Week 1 smoke test checkpoint
    checkpoint_path = "experiments/week1_smoke_test/weights/best.pt"
    export_and_verify(checkpoint_path, imgsz=640)