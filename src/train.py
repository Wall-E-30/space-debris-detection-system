"""
src/train.py
---
Main training entry point for Space Debris Detection System.

Features:
1. Photometric Augmentation Handoff: Automatically runs src/data/augment.py
   on the training split before training (brightness/contrast jitter,
   Gaussian blur, Gaussian noise) to narrow the synthetic-to-real gap.
2. Space-Domain Augmentations: Configures YOLOv8 hyperparameter augmentations
   suited for orbital objects (360° rotation invariance, vertical/horizontal flips,
   random erasing for panel/antenna occlusions).
3. Acceleration: Supports CUDA GPU execution with dataset RAM caching.
"""

import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

from src.data.augment import SpaceImageAugmenter, run_augmentation


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Space Debris Detection Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--augment-copies", type=int, default=1, help="Photometric augmented copies per train image")
    parser.add_argument("--name", type=str, default="space_debris_baseline", help="Experiment name")
    parser.add_argument("--data-config", type=str, default="spark.yaml", help="Path to data YAML config")
    args = parser.parse_args()

    repo_root = Path.cwd()
    spark_yaml = repo_root / args.data_config

    if not spark_yaml.exists():
        raise FileNotFoundError(f"Config file not found at {spark_yaml}")

    # Read config to get dataset path
    with open(spark_yaml, "r") as f:
        cfg = yaml.safe_load(f) or {}

    dataset_dir = Path(cfg.get("path", repo_root / "data" / "processed" / "Swisscube"))

    # Step 1: Run Photometric Augmentation Handoff (Noise, Blur, Brightness Shift)
    print("\n" + "=" * 60)
    print(" [Step 1/2] Wiring & Running Photometric Augmentation Pipeline")
    print("=" * 60)

    img_dir = dataset_dir / "images" / "train"
    existing_aug = list(img_dir.glob("aug1_*")) if img_dir.exists() else []

    if len(existing_aug) == 0 and args.augment_copies > 0:
        print(f"Applying photometric augmentation ({args.augment_copies} copy/copies per image)...")
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
        n_imgs, n_lbls = run_augmentation(dataset_dir, "train", args.augment_copies, augmenter)
        print(f"Photometric Augmentation Complete: {n_imgs} augmented images generated in {img_dir}.")
    else:
        print(f"Photometric augmentation verified. Found {len(existing_aug)} existing augmented images in dataset.")

    # Step 2: Configure GPU Device & YOLOv8 Space Domain Training
    print("\n" + "=" * 60)
    print(" [Step 2/2] Launching Space-Domain YOLOv8 Training Loop")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training Device: {device} ({device_name})")

    model = YOLO("yolov8n.pt")

    # Space-domain hyperparameter augmentations
    results = model.train(
        data=str(spark_yaml.resolve().as_posix()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cache=True,
        workers=4,
        amp=True,
        # Space-domain hyperparameter augmentations
        degrees=180.0,   # Full 360-degree rotation invariance for tumbling space objects
        flipud=0.5,      # Vertical flip (zero-gravity environment)
        fliplr=0.5,      # Horizontal flip
        hsv_h=0.015,     # Color jitter
        hsv_s=0.7,       # Saturation jitter
        hsv_v=0.4,       # Brightness/exposure jitter
        erasing=0.4,     # Random erasing for self-occlusion & antenna/panel obscuration
        project=str((repo_root / "experiments").resolve().as_posix()),
        name=args.name,
        device=device
    )

    print("\nTraining Run Complete!")
    return results


if __name__ == "__main__":
    main()
