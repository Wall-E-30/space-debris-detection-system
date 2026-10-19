# Phase 2: SPARK Dataset Preparation & Stratified Subsetting — Research

**Gathered:** 2026-10-19  
**Status:** Complete  

---

## 1. SPARK-2021 Dataset Domain Analysis

The official SPARK-2021 dataset (*SPAcecraft Recognition leveraging Knowledge of space environment*, released by the CVI² group at SnT, University of Luxembourg for the IEEE ICIP 2021 Challenge) provides the definitive space object detection benchmark:
- **Total Images:** 150,000 images (90,000 train, 30,000 validation, 30,000 test).
- **Modality:** 1024x1024 RGB images with paired 16-bit depth maps.
- **Classes:** 11 distinct classes (10 NASA 3D satellite models + 1 consolidated Debris class).
- **Debris Composition:** 25,000 debris images across 5 distinct space debris CAD geometries (space shuttle external tank, orbital docking system, damaged communication dish, thermal protection tiles, and connector ring).

### Why the Original Subsampling Plan was Rejected
The initial Phase 2 roadmap anticipated severe class imbalance (e.g., 90% satellite vs 10% debris) requiring 2:1 or 3:1 majority-class subsampling.
The empirical audit proved:
- Each satellite class contains exactly 12,500 images (7,500 train, 2,500 val, 2,500 test).
- Debris contains exactly 25,000 images (15,000 train, 5,000 val, 5,000 test).
Because classes are evenly balanced and the department Quadro GV100 has 32 GB VRAM, **artificial subsampling is unnecessary and counterproductive**. The model trains on the complete corpus.

---

## 2. Bounding Box Transformation Mechanics

### Coordinate Conventions

SPARK-2021 employs row/column ordering in its ground-truth annotations:
$$\mathbf{b}_{\text{SPARK}} = [R_{\min}, C_{\min}, R_{\max}, C_{\max}] \equiv [y_{\min}, x_{\min}, y_{\max}, x_{\max}]$$

Ultralytics YOLOv8 strictly requires normalized center coordinates:
$$\mathbf{b}_{\text{YOLO}} = [\text{class\_id}, c_x, c_y, w, h]$$

Where for an image of dimensions $(W, H)$:
$$c_x = \frac{x_{\min} + x_{\max}}{2 \cdot W} = \frac{C_{\min} + C_{\max}}{2 \cdot W}$$
$$c_y = \frac{y_{\min} + y_{\max}}{2 \cdot H} = \frac{R_{\min} + R_{\max}}{2 \cdot H}$$
$$w = \frac{x_{\max} - x_{\min}}{W} = \frac{C_{\max} - C_{\min}}{W}$$
$$h = \frac{y_{\max} - y_{\min}}{H} = \frac{R_{\max} - R_{\min}}{H}$$

### Edge Boundary & Collision Guardrails
1. **Coordinate Clamping:** $c_x, c_y, w, h \in [0.0, 1.0]$. Box boundaries clipped to $[0, W]$ and $[0, H]$.
2. **Degenerate Bounding Boxes:** Any bounding box with $w \le 0$ or $h \le 0$ or $R_{\max} \le R_{\min}$ is discarded.
3. **Collision-Free Namespace:** Raw archive files repeat basenames across class directories (`train_rgb/AcrimSat/image_00001_img.jpg` vs `train_rgb/Debris/image_00001_img.jpg`). Standardized destination filenames prefix the class name: `{class_name}_{image_stem}.{ext}`.

---

## 3. Photometric Space-Domain Augmentation Pipeline

### Synthetic-to-Real Domain Gap
Synthetic space renders exhibit characteristic rendering artifacts that fail in orbit:
1. **Perfect Contrast / Linear Light:** No sensor bloom, solar glint, or Earth-albedo wash.
2. **Zero Sensor Read Noise:** Real CMOS/CCD orbital sensors produce dark-current and read noise under low-flux orbital conditions.
3. **Infinite Depth of Field:** Real star trackers and close-proximity sensors suffer optical defocus and tumbling motion blur.

### Transformation Mathematics
All applied transforms are strictly **photometric** (coordinate-invariant):
1. **Brightness / Contrast Scaling:**
   $$I_{\text{scaled}}(x,y) = \alpha \cdot I(x,y) + \beta, \quad \alpha \in [0.7, 1.3], \ \beta \in [-20, 20]$$
2. **Gaussian Optical Defocus Blur:**
   $$I_{\text{blur}} = I * G_\sigma, \quad \sigma \in [0.0, 2.0]$$
3. **Gaussian Sensor Read Noise:**
   $$I_{\text{noisy}}(x,y) = \text{clip}(I(x,y) + \mathcal{N}(0, \sigma_{\text{noise}}^2), 0, 255), \quad \sigma_{\text{noise}} \in [2.0, 12.0]$$

---

## 4. Perceptual Hash Leakage Audit & Black-Space Hash Collapse

### Difference Hash (dHash) Principle
The 64-bit difference hash downsamples an image to $9 \times 8$ grayscale pixels and computes horizontal gradient signs:
$$H(x, y) = \begin{cases} 1 & \text{if } I(x+1, y) > I(x, y) \\ 0 & \text{otherwise} \end{cases}$$
Near-duplicates have small Hamming distances ($\le 3$ bits).

### The Global Black-Space Collapse Finding
In orbital space imagery, **$>95\%$ of pixels are solid black space ($[0,0,0]$)**. When a $1024 \times 1024$ image with a small spacecraft ($<5\%$ area) is downsampled to an $8 \times 8$ thumbnail:
- The spacecraft details are averaged out.
- The thumbnail gradient collapses to near-zero everywhere.
- Completely distinct spacecraft (e.g. `Terra` vs `AcrimSat`) yield identical 64-bit hashes (Hamming distance 0).

**Key Takeaway:** Split leakage audits on orbital imagery must use **bounding-box cropped regions of interest (ROI)** rather than full-frame thumbnail hashes. Full-frame dHash is fundamentally blind to foreground space objects.
