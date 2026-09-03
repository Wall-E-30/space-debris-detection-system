from pathlib import Path
import sys
import yaml
import torch
from ultralytics import YOLO

# 1. Locate repository root and add to sys.path
current_dir = Path.cwd()
repo_root = current_dir if (current_dir / "spark.yaml").exists() else current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.augment import SpaceImageAugmenter, run_augmentation

# 2. Check CUDA / GPU availability
device = 0 if torch.cuda.is_available() else "cpu"
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Using device: {device} ({device_name})")

# 3. Locate dataset config
spark_yaml = repo_root / "spark.yaml"
dataset_dir = repo_root / "data" / "processed" / "Swisscube"

with open(spark_yaml, "r") as f:
    cfg = yaml.safe_load(f) or {}

cfg["path"] = str(dataset_dir.resolve().as_posix())
cfg["train"] = "images/train"
cfg["val"] = "images/val"
cfg["test"] = "images/test"
cfg["names"] = {0: "swisscube"}

with open(spark_yaml, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)

# 4. Step 1: Run Photometric Augmentation (Noise, Blur, Exposure Jitter)
img_dir = dataset_dir / "images" / "train"
existing_aug = list(img_dir.glob("aug1_*")) if img_dir.exists() else []

if len(existing_aug) == 0:
    print("Wiring & applying photometric augmentation (noise, Gaussian blur, brightness shift)...")
    augmenter = SpaceImageAugmenter(
        brightness_range=(0.7, 1.3),
        contrast_range=(0.7, 1.3),
        blur_radius_range=(0.0, 2.0),
        noise_sigma_range=(2.0, 12.0),
        p_brightness_contrast=0.6,
        p_blur=0.5,
        p_noise=0.5,
        seed=42
    )
    n_imgs, n_lbls = run_augmentation(dataset_dir, "train", 1, augmenter)
    print(f"Augmentation complete: {n_imgs} photometric augmented images added to training set.")
else:
    print(f"Photometric augmentation verified: {len(existing_aug)} augmented images present.")

# 5. Step 2: Initialize YOLOv8-Nano and run space-domain training loop
model = YOLO("yolov8n.pt")

results = model.train(
    data=str(spark_yaml.resolve().as_posix()),
    epochs=5,
    imgsz=1024,
    batch=8,
    cache=True,
    workers=4,
    amp=True,
    # Space domain hyperparameter augmentations
    degrees=180.0,   # Full 360-degree rotation invariance
    flipud=0.5,      # Vertical flip (zero-g environment)
    fliplr=0.5,      # Horizontal flip
    hsv_h=0.015,     # Color jitter
    hsv_s=0.7,       # Saturation jitter
    hsv_v=0.4,       # Brightness jitter
    erasing=0.4,     # Occlusion erasing
    project=str((repo_root / "experiments").resolve().as_posix()),
    name="week1_smoke_test",
    device=device
)

print("Training Run Complete with Photometric Augmentations!")
