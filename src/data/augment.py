"""
src/data/augment.py
---
Photometric augmentation pipeline for synthetic space imagery.

Purpose: synthetic renders (SPARK, SwissCube, etc.) are too clean —
uniform lighting, no sensor noise, no optical blur. Real telescope/camera
imagery has all three. This pipeline narrows that synthetic-to-real gap
by injecting:

    1. Brightness / contrast jitter  — simulates exposure & lighting variation
    2. Gaussian blur                 — simulates optical defocus / motion blur
    3. Gaussian noise                — simulates sensor read noise

These are all PHOTOMETRIC transforms (pixel values change, pixel
POSITIONS don't) — so bounding boxes are unaffected and label files are
simply copied alongside each augmented image, unchanged.

This is a skeleton, not tuned: default ranges are reasonable starting
points, not validated against real telescope imagery yet. Tuning comes
after Person A has baseline training results to compare against (see
Week 1 plan) — don't burn time calibrating strength before there's a
baseline to measure improvement against.

Usage:
    # Generate 2 augmented copies of every train image (val/test untouched
    # by default — augmenting eval data would corrupt your metrics):
    python src/data/augment.py --data data/processed --copies 2

    # Preview grid only, no files written — check the augmentations look
    # reasonable before generating thousands of images:
    python src/data/augment.py --data data/processed --preview-only

    # Adjust strength (see --help for all ranges):
    python src/data/augment.py --data data/processed --copies 3 \\
        --brightness 0.6 1.4 --blur-radius 0 2.5 --noise-sigma 2 15

    # Use as a library in training code:
    from src.data.augment import SpaceImageAugmenter
    aug = SpaceImageAugmenter(seed=42)
    augmented_img = aug.apply(pil_image)
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Core augmenter — importable independently of the CLI below
# ---------------------------------------------------------------------------

class SpaceImageAugmenter:
    """
    Composable photometric augmenter for synthetic space imagery.

    Each transform has its own probability and range, applied
    independently and in random order per call to `apply()`. Geometry
    (image size, object positions) is never touched, so this is safe to
    use on already-labeled YOLO data without adjusting bounding boxes.

    Parameters
    ----------
    brightness_range : (float, float)
        Multiplicative brightness factor. 1.0 = unchanged. Default (0.7, 1.3)
        covers moderate under/over-exposure.
    contrast_range : (float, float)
        Multiplicative contrast factor. 1.0 = unchanged. Default (0.7, 1.3).
    blur_radius_range : (float, float)
        Gaussian blur radius in pixels. 0 = no blur. Default (0, 2.0) —
        upper bound kept small so small/dim debris objects aren't erased.
    noise_sigma_range : (float, float)
        Standard deviation of additive Gaussian noise, in 0-255 pixel
        value units. Default (2, 12) approximates typical sensor read noise.
    p_brightness_contrast, p_blur, p_noise : float
        Independent probability each transform is applied per image.
        Defaults all 0.5 — on average each image gets ~1.5 of the 3
        transforms, giving varied but not uniformly maximal degradation.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self,
                 brightness_range=(0.7, 1.3),
                 contrast_range=(0.7, 1.3),
                 blur_radius_range=(0.0, 2.0),
                 noise_sigma_range=(2.0, 12.0),
                 p_brightness_contrast=0.5,
                 p_blur=0.5,
                 p_noise=0.5,
                 seed=None):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_radius_range = blur_radius_range
        self.noise_sigma_range = noise_sigma_range
        self.p_brightness_contrast = p_brightness_contrast
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # -- individual transforms, each usable standalone -----------------

    def brightness_contrast_jitter(self, img: Image.Image) -> Image.Image:
        """Randomly jitter brightness then contrast, independently."""
        b_factor = self.rng.uniform(*self.brightness_range)
        c_factor = self.rng.uniform(*self.contrast_range)
        img = ImageEnhance.Brightness(img).enhance(b_factor)
        img = ImageEnhance.Contrast(img).enhance(c_factor)
        return img

    def gaussian_blur(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian blur with a randomly sampled radius."""
        radius = self.rng.uniform(*self.blur_radius_range)
        if radius <= 0.01:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def add_gaussian_noise(self, img: Image.Image) -> Image.Image:
        """Add zero-mean Gaussian noise, sigma sampled from noise_sigma_range."""
        sigma = self.rng.uniform(*self.noise_sigma_range)
        if sigma <= 0.01:
            return img
        arr = np.asarray(img).astype(np.float32)
        noise = self._np_rng.normal(0, sigma, arr.shape).astype(np.float32)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode=img.mode)

    # -- composed pipeline -----------------------------------------------

    def apply(self, img: Image.Image, force_all=False) -> Image.Image:
        """
        Apply the full augmentation pipeline to a single image.

        Parameters
        ----------
        force_all : bool
            If True, ignore per-transform probabilities and apply all
            three (useful for generating a clear before/after preview).
        """
        img = img.convert("RGB")

        if force_all or self.rng.random() < self.p_brightness_contrast:
            img = self.brightness_contrast_jitter(img)
        if force_all or self.rng.random() < self.p_blur:
            img = self.gaussian_blur(img)
        if force_all or self.rng.random() < self.p_noise:
            img = self.add_gaussian_noise(img)

        return img


# ---------------------------------------------------------------------------
# Dataset-level pipeline runner (operates on YOLO-format images/labels)
# ---------------------------------------------------------------------------

def find_image_label_pairs(data_dir: Path, split: str):
    """
    Given a YOLO-format dataset root (images/<split>/, labels/<split>/),
    return list of (image_path, label_path_or_None) pairs.
    """
    img_dir = data_dir / "images" / split
    lbl_dir = data_dir / "labels" / split

    if not img_dir.exists():
        print(f"[error] {img_dir} does not exist. "
             f"Run src/data/split_dataset.py first to produce YOLO-format data.")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        pairs.append((img_path, lbl_path if lbl_path.exists() else None))

    return pairs


def run_augmentation(data_dir: Path, split: str, n_copies: int, augmenter: SpaceImageAugmenter,
                     dry_run: bool = False):
    """
    For every image in images/<split>/, generate n_copies augmented
    versions named 'aug{i}_<original_stem>.jpg', with the matching label
    file copied unchanged alongside each.
    """
    pairs = find_image_label_pairs(data_dir, split)
    if not pairs:
        print(f"[warn] No images found in images/{split}/")
        return 0, 0

    img_dir = data_dir / "images" / split
    lbl_dir = data_dir / "labels" / split

    written_imgs, written_lbls, missing_labels = 0, 0, 0

    for img_path, lbl_path in pairs:
        try:
            with Image.open(img_path) as im:
                im.load()
                for i in range(1, n_copies + 1):
                    aug_img = augmenter.apply(im)
                    dest_img = img_dir / f"aug{i}_{img_path.stem}{img_path.suffix}"
                    if not dry_run:
                        aug_img.save(dest_img, quality=95)
                    written_imgs += 1

                    dest_lbl = lbl_dir / f"aug{i}_{img_path.stem}.txt"
                    if lbl_path is not None:
                        if not dry_run:
                            shutil.copy2(lbl_path, dest_lbl)
                        written_lbls += 1
                    else:
                        missing_labels += 1
        except Exception as e:
            print(f"[warn] Failed to process {img_path.name}: {e}")

    if missing_labels:
        print(f"[warn] {missing_labels} augmented image(s) have no matching label "
             f"file (source image had none either) — check for unlabeled images.")

    return written_imgs, written_lbls


# ---------------------------------------------------------------------------
# Preview grid (before / after) — sanity check before generating N copies
# ---------------------------------------------------------------------------

def save_preview_grid(data_dir: Path, split: str, augmenter: SpaceImageAugmenter,
                      out_path: Path, n_samples: int = 6, seed: int = 0):
    import matplotlib.pyplot as plt

    pairs = find_image_label_pairs(data_dir, split)
    if not pairs:
        print(f"[warn] No images found in images/{split}/ — skipping preview.")
        return

    rng = random.Random(seed)
    chosen = rng.sample(pairs, min(n_samples, len(pairs)))

    fig, axes = plt.subplots(2, len(chosen), figsize=(len(chosen) * 3, 6.5))
    if len(chosen) == 1:
        axes = axes.reshape(2, 1)

    for col, (img_path, _) in enumerate(chosen):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            axes[0, col].imshow(im)
            axes[0, col].axis("off")
            axes[0, col].set_title(img_path.stem[:18], fontsize=8)

            aug_im = augmenter.apply(im, force_all=True)
            axes[1, col].imshow(aug_im)
            axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Augmented", fontsize=10)
    fig.suptitle("Augmentation preview — original (top) vs. augmented (bottom)",
                fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Preview saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Photometric augmentation pipeline for synthetic space imagery."
    )
    parser.add_argument("--data", required=True,
                        help="YOLO-format dataset root (output of split_dataset.py)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                        help="Which split to augment (default: train — augmenting "
                             "val/test would corrupt evaluation metrics)")
    parser.add_argument("--copies", type=int, default=2,
                        help="Number of augmented copies to generate per image (default: 2)")
    parser.add_argument("--brightness", type=float, nargs=2, default=[0.7, 1.3],
                        metavar=("MIN", "MAX"), help="Brightness factor range")
    parser.add_argument("--contrast", type=float, nargs=2, default=[0.7, 1.3],
                        metavar=("MIN", "MAX"), help="Contrast factor range")
    parser.add_argument("--blur-radius", type=float, nargs=2, default=[0.0, 2.0],
                        metavar=("MIN", "MAX"), help="Gaussian blur radius range (pixels)")
    parser.add_argument("--noise-sigma", type=float, nargs=2, default=[2.0, 12.0],
                        metavar=("MIN", "MAX"), help="Gaussian noise sigma range (0-255 scale)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-only", action="store_true",
                        help="Only save a before/after preview grid — don't generate "
                             "the full augmented dataset")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip preview grid generation and go straight to full run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be generated without writing files")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"[error] Dataset directory not found: {data_dir}")
        sys.exit(1)

    augmenter = SpaceImageAugmenter(
        brightness_range=tuple(args.brightness),
        contrast_range=tuple(args.contrast),
        blur_radius_range=tuple(args.blur_radius),
        noise_sigma_range=tuple(args.noise_sigma),
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print(f"  Dataset:   {data_dir}")
    print(f"  Split:     {args.split}")
    print(f"  Copies:    {args.copies} per image")
    print(f"  Brightness range: {args.brightness}   Contrast range: {args.contrast}")
    print(f"  Blur radius range: {args.blur_radius}  Noise sigma range: {args.noise_sigma}")
    print(f"{'='*60}\n")

    if not args.no_preview:
        preview_path = data_dir / "augment_preview.png"
        print("[preview] Generating before/after preview grid …")
        save_preview_grid(data_dir, args.split, augmenter, preview_path)
        print()

    if args.preview_only:
        print("Preview-only mode — stopping here. Check the preview grid, then "
             "re-run without --preview-only to generate the full augmented set.")
        return

    print(f"[run] Generating {args.copies} augmented copies per image in "
         f"images/{args.split}/ …")
    n_imgs, n_lbls = run_augmentation(data_dir, args.split, args.copies, augmenter,
                                      dry_run=args.dry_run)

    mode = "would write" if args.dry_run else "wrote"
    print(f"\nDone. {mode} {n_imgs:,} augmented images and {n_lbls:,} label files "
         f"to images/{args.split}/ and labels/{args.split}/.")
    if not args.dry_run:
        print(f"Original images are untouched — augmented files are prefixed 'aug1_', 'aug2_', etc.")


if __name__ == "__main__":
    main()