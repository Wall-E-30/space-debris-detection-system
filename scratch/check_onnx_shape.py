import onnxruntime as ort
from pathlib import Path

models = [
    Path("experiments/spark2021_yolov8n_50e/weights/best.onnx"),
    Path("experiments/spark_yolov8n_50e/weights/best.onnx"),
    Path("best.onnx"),
]

for m in models:
    if m.exists():
        print(f"\nModel: {m}")
        sess = ort.InferenceSession(str(m), providers=["CPUExecutionProvider"])
        for inp in sess.get_inputs():
            print(f"  Input: name={inp.name}, shape={inp.shape}, type={inp.type}")
        for out in sess.get_outputs():
            print(f"  Output: name={out.name}, shape={out.shape}, type={out.type}")
