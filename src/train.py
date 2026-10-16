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
import sys
from pathlib import Path

# Add repository root to sys.path so 'src.*' imports resolve regardless of cwd or invocation directory
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import yaml
import torch
from ultralytics import YOLO

try:
    from src.data.augment import SpaceImageAugmenter, run_augmentation
except (ImportError, ModuleNotFoundError):
    try:
        from data.augment import SpaceImageAugmenter, run_augmentation
    except (ImportError, ModuleNotFoundError):
        SpaceImageAugmenter = None
        run_augmentation = None


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Space Debris Detection Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--augment-copies", type=int, default=1, help="Photometric augmented copies per train image")
    parser.add_argument("--name", type=str, default="space_debris_baseline", help="Experiment name")
    parser.add_argument("--device", type=str, default=None, help="Device to train on ('0', 'cpu', default: auto)")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Initial model weights (.pt checkpoint or yolov8n.pt, default: yolov8n.pt)",
    )
    parser.add_argument(
        "--data-config",
        "--data",
        dest="data_config",
        type=str,
        default="configs/swisscube.yaml",
        help="Path to dataset YAML configuration file (default: configs/swisscube.yaml)",
    )
    parser.add_argument("--exist-ok", action="store_true", default=False, help="Overwrite existing experiment directory")
    args = parser.parse_args()

    repo_root = Path.cwd()
    data_config = Path(args.data_config)
    if not data_config.is_absolute():
        data_config = repo_root / data_config

    if not data_config.exists():
        fallback_root = repo_root / "swisscube.yaml"
        if fallback_root.exists():
            data_config = fallback_root
        else:
            raise FileNotFoundError(f"Config file not found at {data_config}")

    # Read config to get dataset path
    with open(data_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_path = Path(cfg.get("path", "data/processed/Swisscube"))
    if not raw_path.is_absolute():
        candidate_from_config = (data_config.parent / raw_path).resolve()
        candidate_from_root = (repo_root / raw_path).resolve()
        if candidate_from_config.exists():
            dataset_dir = candidate_from_config
        elif candidate_from_root.exists():
            dataset_dir = candidate_from_root
        else:
            dataset_dir = candidate_from_config
    else:
        dataset_dir = raw_path

    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    if not train_img_dir.exists() and not val_img_dir.exists():
        print("\n" + "=" * 70)
        print(" [ERROR] Dataset images not found on disk!")
        print(f" Checked directory: {dataset_dir}")
        print(f" Missing path:      {val_img_dir}")
        print("-" * 70)
        print(" Why this happens:")
        print("  - Dataset folders ('data/') are gitignored, so images are not")
        print("    automatically transferred via git clone/pull.")
        print(" How to fix:")
        print("  Option 1: If SwissCube is already on this machine, update 'path:' in")
        print(f"            {data_config.name} to its absolute location.")
        print("  Option 2: Copy 'data/processed/Swisscube' from your laptop to:")
        print(f"            {dataset_dir}")
        print("=" * 70 + "\n")
        raise FileNotFoundError(f"Dataset images not found in {dataset_dir}")

    # Step 1: Run Photometric Augmentation Handoff (Noise, Blur, Brightness Shift)
    print("\n" + "=" * 60)
    print(" [Step 1/2] Wiring & Running Photometric Augmentation Pipeline")
    print("=" * 60)

    img_dir = dataset_dir / "images" / "train"
    existing_aug = list(img_dir.glob("aug1_*")) if img_dir.exists() else []

    if len(existing_aug) == 0 and args.augment_copies > 0:
        print(f"Applying photometric augmentation ({args.augment_copies} copy/copies per image)...")
        print(" [Tip: To bypass offline image generation and train immediately, use '--augment-copies 0']")
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

    # Step 2: Configure Device & YOLOv8 Space Domain Training
    print("\n" + "=" * 60)
    print(" [Step 2/2] Launching Space-Domain YOLOv8 Training Loop")
    print("=" * 60)

    if args.device is not None:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if str(device) == "0" and torch.cuda.is_available() else "CPU"
    print(f"Training Device: {device} ({device_name})")
    print(f"Base Weights:    {args.weights}")

    model = YOLO(args.weights)

    # Space-domain hyperparameter augmentations
    results = model.train(
        data=str(data_config.resolve().as_posix()),
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
        exist_ok=args.exist_ok,
        device=device
    )

    # -------------------------------------------------------------
    # Persistent Training Run Archiving (outputs/training/)
    # -------------------------------------------------------------
    import datetime, json
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = repo_root / "outputs" / "training"
    archive_dir.mkdir(parents=True, exist_ok=True)

    train_record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiment_name": args.name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "data_config": str(data_config.as_posix()),
        "device": str(device),
        "save_dir": str(getattr(results, "save_dir", "")),
    }

    timestamped_file = archive_dir / f"train_{now_str}.json"
    latest_file = archive_dir / "latest.json"
    history_file = archive_dir / "history.jsonl"

    with open(timestamped_file, "w", encoding="utf-8") as f:
        json.dump(train_record, f, indent=2)

    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(train_record, f, indent=2)

    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(train_record) + "\n")

    print(f"[OK] Training run archived to: {timestamped_file.resolve()}")
    print(f"[OK] Latest training run updated at: {latest_file.resolve()}\n")

    return results


if __name__ == "__main__":
    main()
