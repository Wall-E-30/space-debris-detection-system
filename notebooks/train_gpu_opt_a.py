from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# 1. Check CUDA / GPU availability
device = 0 if torch.cuda.is_available() else "cpu"
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Using device: {device} ({device_name})")

# 2. Locate repository root and dataset config
current_dir = Path.cwd()
repo_root = current_dir if (current_dir / "spark.yaml").exists() else current_dir.parent
spark_yaml = repo_root / "spark.yaml"
dataset_dir = repo_root / "data" / "processed" / "SwissCube"

# 3. Ensure spark.yaml has the absolute path for this environment
with open(spark_yaml, "r") as f:
    cfg = yaml.safe_load(f) or {}

cfg["path"] = str(dataset_dir.resolve().as_posix())
cfg["train"] = "images/train"
cfg["val"] = "images/val"
cfg["test"] = "images/test"
cfg["names"] = {0: "swisscube"}

with open(spark_yaml, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)

# 4. Initialize YOLOv8-Nano model
model = YOLO("yolov8n.pt")

# 5. Run GPU Optimization A (imgsz=1024, batch=8, cache=True, device=0)
results = model.train(
    data=str(spark_yaml.resolve().as_posix()),
    epochs=5,
    imgsz=1024,
    batch=8,
    cache=True,
    workers=4,
    amp=True,
    project=str((repo_root / "experiments").resolve().as_posix()),
    name="week1_smoke_test",
    device=device
)

print("Week 1 Smoke Test Complete!")
