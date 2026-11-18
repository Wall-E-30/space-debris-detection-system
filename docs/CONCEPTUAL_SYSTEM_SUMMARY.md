# Conceptual Summary: Space Debris Detection & SSA System
**Project:** ISRO Hardware & Edge Vision Challenge (Problem Statement #05)  
**Target Hardware:** Raspberry Pi 4B (ARM Cortex-A72) / Raspberry Pi 5 (ARM Cortex-A76)  
**Timeline & Budget:** 4-Week Rapid Engineering Sprint | Power $\le 5\text{W} - 15\text{W}$ | Latency Gate $\le 1000\text{ ms}$

---

## Executive Overview

Deploying an autonomous Computer Vision (CV) payload into Low Earth Orbit (LEO) presents an unforgiving engineering intersection: **extreme orbital dynamics ($>7.5\text{ km/s}$)**, **severe radiometric and illumination challenges** (zero atmospheric scatter, deep specular shadows, $>95\%$ black space background), and **strict edge compute envelopes** (sub-watt to 15W power budgets, rad-hard memory limits, and zero active GPU cooling).

Within our 4-week sprint, we delivered an end-to-end, edge-deployable pipeline built on **YOLOv8-Nano**, compiled to hardware-portable **ONNX** and ARM-vectorized **NCNN**, and coupled with a decoupled **SGP4 orbital catalog layer**.

```
+----------------------------------------------------------------------------------------------------+
|                                    ISRO PS #05 SYSTEM PIPELINE                                     |
+----------------------------------------------------------------------------------------------------+
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
   [ OPTICAL VISION SUBSYSTEM ]                                      [ ASTRODYNAMICS SUBSYSTEM ]
   Wide-Field Mono-Camera Frame                                      Space-Track / CelesTrak TLEs
                 │                                                                 │
                 ▼                                                                 ▼
   Scale Adaptation (640 / 800 / 1024 px)                            SGP4 Analytical Propagator
                 │                                                                 │
                 ▼                                                                 ▼
   YOLOv8-Nano (Anchor-Free + DFL)                                   3D ECI Geocentric State Vectors
   Runtime: ONNX Runtime / NCNN Engine                               (X, Y, Z, Vx, Vy, Vz in km & km/s)
                 │                                                                 │
                 ▼                                                                 ▼
   Class ID + Bounding Box Centroids                                 Known Operational Spacecraft
                 │                                                   (NORAD Ephemeris Lookup)
                 │                                                                 │
                 ├──────────────────────────────┬──────────────────────────────────┤
                 ▼                              ▼                                  ▼
        [ Known Satellite ]            [ Synthetic Debris ]             [ Conjunction Assessment ]
     Resolved to NORAD ID            Uncataloged Fragment             Relative Euclidean Separation
     Cross-referenced with SGP4      Immediate Proximity Alert        Distance (km) to Host Vehicle
```

---

## 1. Object Detection Architecture: YOLOv8-Nano vs. Alternatives

### 1.1 The Concept
YOLOv8-Nano is a single-stage, anchor-free convolutional object detector. It replaces legacy anchor-based heads (which predicted box offsets relative to predefined anchor boxes) with an **anchor-free, center-based decoupled head**. It extracts spatial features through a modified CSPDarknet53 backbone with **C2f modules** (Cross-Stage Partial Bottleneck with two convolutions) and aggregates multi-scale contextual features via a **Path Aggregation Network (PANet)** neck.

### 1.2 Alternatives Considered & Decision Rationale

| Architecture Class | Candidate Models | Parameters & GFLOPs | Why Rejected / Trade-off Analysis |
|---|---|---|---|
| **Two-Stage Detectors** | Faster R-CNN, Cascade R-CNN | ~41M–60M params<br>$\ge 150\text{ GFLOPs}$ | **Computational Infeasibility:** Uses an explicit Region Proposal Network (RPN) followed by RoI Pooling. The multi-stage pipeline introduces non-deterministic graph latency and severe memory bottlenecks, taking $>3,000\text{ ms}$ on an ARM Cortex-A72 CPU. Exceeds the 1000 ms real-time gate by $>3\times$. |
| **Vision Transformers** | RT-DETR, ViT-based detectors | ~30M–50M params<br>$\ge 100\text{ GFLOPs}$ | **Quadratic Token Complexity:** Self-attention mechanisms scale quadratically ($O(N^2)$) with token sequence length. On high-resolution orbital imagery ($1024\times1024$), memory bandwidth saturates edge SBC cache. Furthermore, embedded runtimes (NCNN/TFLite) lack native, optimized micro-kernels for transformer attention layers on ARM. |
| **Larger Single-Stage** | YOLOv8-Medium / X, YOLOv9, YOLOv10 | ~25M–65M params<br>$\ge 80\text{ GFLOPs}$ | **Diminishing Returns & Latency:** While offering marginally higher mAP on complex terrestrial datasets (COCO), the high parameter volume causes massive thermal throttling on passively cooled edge hardware. In our 4-week timeline, training larger models would take weeks of compute without passing the edge latency gate. |
| **TinyML Classifiers** | MobileNetV3-SSD, FOMO (Edge Impulse) | <1.5M params<br><2.0 GFLOPs | **Spatial Blindness on Irregular Debris:** MobileNet-SSD suffers from low multi-scale feature resolution. FOMO completely drops bounding box regression (predicting only object centroids), making it incapable of estimating object size, solar panel extent, or bounding-box area. |

### 1.3 Why YOLOv8-Nano Won for PS #05
1. **Ultra-Lightweight Footprint:** With only **3.01M to 3.14M parameters** and **8.8 GFLOPs at 640px** (22.4 GFLOPs at native 1024px), the uncompressed FP32 model compiles to just **12.9 MB**, comfortably fitting within the strict spaceborne storage limits ($\le 15\text{ MB}$).
2. **Anchor-Free Superiority in Space:** Space objects possess extreme aspect ratio variance—from tumbling polygonal debris shards ($1:1$) to elongated deployable solar wings on satellites like TRMM ($1:8$). Anchor-based architectures require manual clustering of predefined anchor ratios that consistently fail on off-nominal debris orientations. YOLOv8 directly regresses the distance from grid points to bounding box edges.
3. **Maturity & 4-Week Delivery Sprint:** Ultralytics provides a hardened, battle-tested export pipeline to ONNX and NCNN. It minimized deployment debugging and eliminated operator-compatibility failures on ARM.

---

## 2. Evaluation Metrics: mAP@0.5 vs. mAP@0.5:0.95 & Small-Object IoU Sensitivity

### 2.1 The Concept of mAP Metrics
Object detection performance is quantified via Average Precision (AP), the area beneath the Precision-Recall curve:
$$\text{AP} = \int_{0}^{1} P(R) \, dR$$

* **mAP@0.5 (PASCAL VOC Metric):** Mean AP calculated at a single, coarse Intersection over Union threshold ($\text{IoU} \ge 0.50$). It evaluates whether the detector recognized the object and roughly localized its region.
* **mAP@0.5:0.95 (COCO Primary Metric / Headline Metric):** The mean of AP values calculated across 10 distinct IoU thresholds from $0.50$ to $0.95$ in increments of $0.05$:
  $$\text{mAP@[0.5:0.95]} = \frac{1}{10} \sum_{k=0}^{9} \text{AP}_{\text{IoU} = 0.50 + 0.05k}$$
  It strictly penalizes sloppy, loose bounding box predictions, rewarding sub-pixel edge alignment.

### 2.2 Why mAP@0.5 Saturates in Space Imagery
In space situational awareness, imagery consists of an illuminated spacecraft against a pitch-black cosmic backdrop ($>95\%$ zero-pixel intensity). 
* At $\text{IoU} \ge 0.50$, the detector effortlessly segregates foreground targets from background sky, causing **metric saturation** (**99.02% overall mAP@0.5**, **98.34% on Debris**).
* Relying exclusively on mAP@0.5 provides a false sense of security; a model could have bounding boxes over-extended by 20 pixels and still score a perfect 1.0. To ensure our detections are precise enough for proximity operations and collision avoidance, **mAP@0.5:0.95 (86.98%) was enforced as the scientific headline metric**.

### 2.3 Mathematical Proof: Why IoU Severely Penalizes Small Objects (CubeSats)

Intersection over Union is defined as:
$$\text{IoU}(B_p, B_{gt}) = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$

Consider an absolute boundary regression error of $\delta = 4\text{ pixels}$ along the boundary edges:

```
LARGE SPACECRAFT (e.g., Terra / AcrimSat)        SMALL TARGET (e.g., CubeSat / Debris Fragment)
Ground Truth: 400 x 400 px (Area = 160,000 px²)  Ground Truth: 20 x 20 px (Area = 400 px²)
Predicted:    408 x 408 px (4-pixel edge shift)  Predicted:    28 x 28 px (4-pixel edge shift)
Overlap:      160,000 px²                        Overlap:      400 px²
Union:        166,464 px²                        Union:        784 px²
-----------------------------------------------  -----------------------------------------------
IoU = 160,000 / 166,464 ≈ 0.961 (PASSES ALL IoU) IoU = 400 / 784 ≈ 0.510 (FAILS ALL IoU ≥ 0.55!)
```

#### Empirical Impact in Our Results
This mathematical property directly explains our empirical findings in `SPARK-2021`:
* **Large Spacecraft (`AcrimSat`, `Aura`):** Achieved **$95.09\%$** and **$93.80\%$** mAP@0.5:0.95.
* **Miniature Spacecraft (`CubeSat`):** While its classification recall was high ($95\%$ true positives), its mAP@0.5:0.95 collapsed to **$63.41\%$** (and $59.20\%$ at 640px).
* Because CubeSats occupy $<100\times100$ px (and $<30\times30$ px at 640px), a minute 3-pixel localization error collapses the IoU below $0.75$, completely zeroing out precision in the upper five evaluation bins of the COCO metric.

---

## 3. Edge Deployment & Quantization: FP32 vs. INT8, and NCNN vs. ONNX Runtime

### 3.1 The Concept: FP32 vs. Post-Training Quantization (PTQ) INT8
* **FP32 (Single Precision Floating Point):** Uses 1 sign bit, 8 exponent bits, and 23 mantissa bits (32 bits = 4 bytes per weight). High dynamic range, zero truncation error, but high memory bandwidth consumption and slower floating-point ALU operations on embedded CPUs.
* **INT8 Quantization:** Maps the continuous dynamic range of weights and activations $[\alpha, \beta]$ to an 8-bit integer lattice $[-128, 127]$ using an affine mapping:
  $$q = \text{round}\left(\frac{x}{S}\right) + Z$$
  where $S$ is the scale factor ($S = \frac{\beta - \alpha}{255}$) and $Z$ is the integer zero-point.

### 3.2 The Quantization Trade-off in the Space Domain
* **The Benefit:** Shrinks memory footprint by **$4\times$** (from $12.9\text{ MB}$ to $\approx 3.4\text{ MB}$) and slashes L1/L2 cache misses. Allows leveraging ARM Cortex-A76 integer vector units (e.g., `SDOT`/`UDOT` NEON instructions).
* **The Space Domain Failure Mode (Dynamic Range Collapse):** In orbital optical images, dynamic range is extreme: $>95\%$ of pixels have near-zero intensity, while solar panel specular reflections peak at 255. In uniform affine quantization, clipping thresholds can inadvertently quantize tiny, low-contrast debris fragments into the zero-bin, causing catastrophic recall drops on small debris.
* **Our Project Decision (DEC-08 / DEC-11):** 
  Empirical benchmarking proved that **FP32 ONNX Runtime on Raspberry Pi 5 already achieves 144.83 ms (6.9 FPS) at 640px**, beating the $\le 1000\text{ ms}$ PRD gate by **$6.9\times$** while consuming only **138.67 MB RAM** (<5.5% of total system RAM). Therefore, FP32 was frozen as the production baseline, leaving INT8 as a Phase 6 optimization once a dedicated representative calibration dataset is curated.

### 3.3 Deployment Framework: Why NCNN & ONNX Runtime (and not TFLite / Edge Impulse)?

```
+--------------------------------------------------------------------------------------------------+
|                                    RUNTIME SELECTION MATRIX                                      |
+--------------------------------------------------------------------------------------------------+
| Metric / Feature           | ONNX Runtime            | Tencent NCNN            | Google TFLite   |
+----------------------------+-------------------------+-------------------------+-----------------+
| Export Pipeline            | Native Ultralytics      | Native (PNNX compiler)  | Finicky Python  |
| Target Architecture        | Cross-Platform (x86/ARM)| Highly Optimized ARM    | Mobile/Android  |
| ARM NEON / fp16 Assembly   | Generic via Eigen       | Hand-tuned ASM kernels  | via XNNPACK     |
| Measured Pi 4/Proxy RAM    | ~120 MB – 200 MB RSS    | ~10.9 MB RSS            | ~80 MB – 150 MB |
| Multi-threading Scaling    | Thread pool / OpenMP    | OpenMP with affinity    | pthread-based   |
| Role in Our Pipeline       | Portable Baseline & UAT | Production ARM Target   | Discarded       |
+--------------------------------------------------------------------------------------------------+
```

1. **Why ONNX Runtime?** It serves as the universal hardware-independent interchange format. It allowed the team to benchmark identical graph models across our 16-Core Xeon workstation, laptop developer proxies, and physical Raspberry Pi SBCs with 100% reproducible numerical parity.
2. **Why NCNN for Embedded Edge?** Developed by Tencent, NCNN has no third-party dependencies and is specifically written for ARM platforms with custom assembly implementations for ARM NEON vector instructions. In our empirical edge tests, NCNN achieved a **21% latency reduction** (165.2 ms vs 209.7 ms) and reduced peak RAM footprint to just **10.9 MB RSS**, freeing system memory for flight software.
3. **Why TFLite and Edge Impulse Were Rejected:**
   - **TFLite:** Requires complex build-time dependencies, finicky XNNPACK multi-threading flags on aarch64 Linux, and struggles with anchor-free split detection heads.
   - **Edge Impulse:** Primarily an end-to-end TinyML SaaS platform centered around microcontrollers and its proprietary EON compiler. It lacks native support for modern YOLOv8 backbone graphs without unwieldy "Bring Your Own Model" wrappers.

---

## 4. Class Imbalance Handling: Subsampling vs. Focal Loss

### 4.1 The Dual Imbalance in Space Situational Awareness
Our space debris dataset faces two distinct structural imbalances:
1. **Extreme Foreground-to-Background Imbalance:** In a $1024 \times 1024$ frame ($1,048,576\text{ pixels}$), an approaching debris fragment might occupy only a $20 \times 20$ patch ($400\text{ pixels}$—less than $0.04\%$ of the image). Dense detectors evaluate thousands of candidate feature map cells, resulting in a negative-to-positive ratio exceeding $10,000 : 1$.
2. **Inter-Class Imbalance:** Operational spacecraft comprise 125,000 instances, whereas debris comprises 25,000 instances across 5 synthetic CAD variations.

### 4.2 Why Subsampling Was Rejected
* **Information Destruction:** Majority-class downsampling discards tens of thousands of valid operational spacecraft images, forfeiting diverse lighting angles, Earth albedo backgrounds, and sensor glare conditions.
* **Overfitting Hazard:** Minority-class oversampling (duplicating debris frames) artificially concentrates training on the 5 synthetic CAD debris models, causing the network to memorize specific polygons rather than learning invariant specular edge gradients.
* **4-Week Timeline Cost:** Manual rebalancing and dataset curation introduce engineering drag and disk I/O overhead.

### 4.3 Why Focal Loss & Distribution Focal Loss (DFL) Were Chosen
To solve the foreground-to-background imbalance at the loss level without modifying the dataset distribution, we rely on the **Focal Loss** formulation (Lin et al.):
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
* $p_t \in [0, 1]$ is the model's estimated probability for the ground truth class.
* $\gamma \ge 0$ is the tunable focusing parameter (standard $\gamma = 2.0$).
* $\alpha_t$ is the class-balancing weight.

```
                    Focal Loss vs. Cross-Entropy Dynamics
    Loss
     ▲
 5.0 │─── Standard Cross-Entropy (All samples contribute heavily)
     │
 3.0 │
     │       ╭── Hard, Ambiguous Targets (Debris Fragments, pt ≈ 0.2)
 1.0 │       │   Modulating factor (1 - pt)^γ ≈ 0.64 (Grad Signal Preserved!)
     │       ▼
 0.1 │               ╰── Easy Background Black Pixels (pt ≥ 0.95)
     │                   Modulating factor (1 - pt)^γ ≤ 0.0025 (Loss Suppressed ~1000x!)
     └────────────────────────────────────────────────────────────────────────► pt
     0.0                    0.5                                              1.0
```

#### How YOLOv8 Extends This (Distribution Focal Loss)
YOLOv8 utilizes a compound loss combining **Complete IoU (CIoU)**, **Binary Cross-Entropy (BCE)**, and **Distribution Focal Loss (DFL)**:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$

* **DFL (Li et al.):** Treats bounding box coordinate regression not as a single Dirac delta point estimate, but as a continuous probability distribution across neighboring feature grid bins.
* When debris boundaries are blurred or semi-shadowed in space, DFL allows the network to focus on the sharpest probability density around the object’s true physical extremities, drastically improving small-scale bounding box convergence.

---

## 5. Astrodynamics (SGP4/TLE): Why Decoupled from the 2D Vision Pipeline

### 5.1 The Mathematical Reality of Monocular Orbital Vision
A monocular spaceborne camera captures an optical projection of the celestial sphere onto a 2D focal plane $(u, v)$. 
* This provides strictly **two angular degrees of freedom**: visual line-of-sight bearing (Right Ascension $\alpha$ and Declination $\delta$, or Azimuth and Elevation).
* It provides **zero range ($r$) and zero range-rate ($\dot{r}$)**.

In orbital mechanics, propagating an orbit requires a full 6-state vector in the Earth-Centered Inertial (ECI) frame:
$$\mathbf{X}(t) = \begin{bmatrix} x & y & z & v_x & v_y & v_z \end{bmatrix}^T$$

To determine an orbit optically from scratch (**Initial Orbit Determination / IOD** via Gauss or Gooding algorithms), a sensor requires a minimum of **three separate, time-stamped visual tracklets** along an orbital arc, combined with observer ephemeris.

```
                               THE DECOUPLED ARCHITECTURE
                               
     ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
     │     OPTICAL VISION SUBSYSTEM      │       │    ORBITAL CATALOG SUBSYSTEM      │
     │  - Input: Raw Camera Frame (2D)   │       │  - Input: CelesTrak TLE Database  │
     │  - Output: Class + 2D BBox Centroid│       │  - Propagator: SGP4 Analytical    │
     │  - Computes: Line-of-Sight Bearing│       │  - Computes: 3D ECI State Vector  │
     └─────────────────┬─────────────────┘       └─────────────────┬─────────────────┘
                       │                                           │
                       │    Detected Class Name                    │
                       └───────────────────┬───────────────────────┘
                                           ▼
                                 [ NORAD Catalog Lookup ]
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   ▼                                               ▼
          [ Known Satellite ]                             [ Space Debris ]
       Class ID: e.g., AcrimSat                        Class ID: `Debris`
       NORAD ID Resolved: #26032                       NO NORAD ID EXISTS IN CELESTRAK!
                   │                                               │
                   ▼                                               ▼
       SGP4 Propagation Enabled                        SGP4 Propagation Mathematically Blocked
       3D Geocentric XYZ Calculated                    Generates Local Visual Proximity Alert
       True Conjunction Risk Evaluated                 (Triggers Multi-Frame Tracking Mode)
```

### 5.2 The Danger of "End-to-End" Deep Learning Conflation
Attempting to train an end-to-end deep neural network to predict 3D distance or Keplerian elements directly from a single 2D bounding box is an unphysical fallacy:
1. **Spurious Scale Correlation:** The model simply learns that "small box = far away, big box = close." In space, a $10\text{ cm}$ CubeSat at $100\text{ meters}$ subtends the exact same pixel area as a $15\text{ meter}$ rocket body at $15\text{ km}$. Under off-nominal scale variations, an end-to-end model produces catastrophic range errors.
2. **Failure Under Non-Keplerian Manoeuvres:** Spacecraft actively adjust orbits; hardcoding range inside visual weights prevents ground-station ephemeris updates from propagating to the model.

### 5.3 Architectural Justification for the Decoupled Layer (DEC-05)
1. **The Uncataloged Debris Physical Boundary:** **Space debris fragments have NO catalog ID in CelesTrak.** By cleanly decoupling the systems, our pipeline explicitly acknowledges that while known spacecraft (`Sentinel-6`, `AcrimSat`) can be matched to active NORAD ephemeris to calculate 3D Euclidean conjunction distances, detected debris triggers a **pure visual tracking alert**, prompting the chaser spacecraft to capture sequential tracklets for IOD.
2. **Decoupled Engineering Velocity:** In a 4-week sprint, decoupling allowed the vision team to optimize YOLOv8 edge inference on ARM CPUs simultaneously while the astrodynamics lead independently built and verified the `Skyfield`/`SGP4` propagation harness (`src/data/tle_pipeline.py`).

---

## Curated Foundational Reading Materials

### Area 1: Object Detection Architecture & Anchor-Free Design
* **YOLOv8 Technical Documentation:**  
  *Jocher, G., Chaurasia, A., & Qiu, J. (2023).* "Ultralytics YOLOv8."  
  [Official Documentation](https://docs.ultralytics.com/) — *Essential for understanding C2f modules, PANet neck scaling, and export switches.*
* **Anchor-Free State-of-the-Art:**  
  *Feng, C., Zhong, Y., Gao, Y., et al. (2021).* "TOOD: Task-aligned One-stage Object Detection." *IEEE/CVF International Conference on Computer Vision (ICCV).*  
  [arXiv:2108.07755](https://arxiv.org/abs/2108.07755) — *Explains the task-aligned head that replaces legacy anchor-based heuristics in modern YOLO detectors.*
* **Spacecraft Recognition Dataset (SPARK):**  
  *Del Prete, A., et al. (2022).* "SPARK: Spacecraft Recognition for Space Situational Awareness." *European Conference on Computer Vision (ECCV).*  
  [Springer Link](https://link.springer.com/) — *Foundational paper behind our dataset, detailing 3D CAD rendering conditions and optical space illumination.*

### Area 2: Detection Metrics & Spatial Sensitivity
* **COCO Evaluation Benchmark Protocol:**  
  *Lin, T.-Y., et al. (2014).* "Microsoft COCO: Common Objects in Context." *European Conference on Computer Vision (ECCV).*  
  [arXiv:1405.0312](https://arxiv.org/abs/1405.0312) — *Defines the mathematical 10-threshold formulation of mAP@0.5:0.95.*
* **Bounding Box Regression Losses & IoU Sensitivity:**  
  *Rezatofighi, H., et al. (2019).* "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression." *IEEE/CVF CVPR.*  
  [arXiv:1902.09630](https://arxiv.org/abs/1902.09630) — *Provides mathematical proofs for IoU vanishing gradients and scale sensitivity on small objects.*
* **Distance & Complete IoU Loss:**  
  *Zheng, Z., et al. (2020).* "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression." *AAAI Conference on Artificial Intelligence.*  
  [arXiv:1911.08287](https://arxiv.org/abs/1911.08287) — *Details CIoU (used in YOLOv8) which accounts for aspect ratio and centroid distance.*

### Area 3: Edge Quantization & Embedded Runtimes
* **Integer Arithmetic & Post-Training Quantization:**  
  *Jacob, B., et al. (2018).* "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *IEEE/CVF CVPR.*  
  [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) — *The seminal paper establishing the affine scale-and-zero-point integer quantization mathematics.*
* **Neural Network Quantization Survey:**  
  *Gholami, A., et al. (2021).* "A Survey of Quantization Methods for Efficient Neural Network Inference."  
  [arXiv:2103.13630](https://arxiv.org/abs/2103.13630) — *In-depth exploration of Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT).*
* **NCNN Embedded Inference Engine:**  
  *Tencent Corporation (2020).* "NCNN High-Performance Neural Network Inference Framework."  
  [GitHub Repository](https://github.com/Tencent/ncnn) / [Wiki Documentation](https://github.com/Tencent/ncnn/wiki) — *Details ARM NEON assembly optimizations, fp16 vector math, and zero-dependency embedded design.*

### Area 4: Class Imbalance & Loss Dynamics
* **The Focal Loss Formulation:**  
  *Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).* "Focal Loss for Dense Object Detection." *IEEE International Conference on Computer Vision (ICCV).*  
  [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) — *Mandatory reading on dynamically down-weighting easy background samples to resolve extreme class imbalance.*
* **Generalized & Distribution Focal Loss:**  
  *Li, X., et al. (2020).* "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection." *NeurIPS 2020.*  
  [arXiv:2006.04388](https://arxiv.org/abs/2006.04388) — *Introduces Distribution Focal Loss (DFL), directly illuminating YOLOv8's continuous box distribution mechanics.*

### Area 5: Astrodynamics, SGP4, and Orbit Propagation
* **Standard Astrodynamics Reference Text:**  
  *Vallado, D. A. (2013).* "Fundamentals of Astrodynamics and Applications" (4th Edition). *Microcosm Press / Springer.*  
  *Chapters 2 & 9 are essential reading for coordinate frames (ECI, ECEF, Topocentric) and Initial Orbit Determination (Gauss & Laplace methods).*
* **The Definitive SGP4 / TLE Mathematical Foundation:**  
  *Vallado, D. A., Crawford, P., Hujsak, R., & Kelso, T. S. (2006).* "Revisiting Spacetrack Report #3: Rev 2." *AIAA/AAS Astrodynamics Specialist Conference.*  
  [AIAA-2006-6753 PDF](https://celestrak.org/publications/AIAA/2006-6753/AIAA-2006-6753.pdf) — *The foundational paper detailing exact algorithmic propagation and error bounds of NORAD Two-Line Element (TLE) sets.*
* **Python Astrodynamics & Space Ephemeris:**  
  *Rhodes, B. (2020).* "Skyfield: High Precision Astrometry for Python."  
  [Skyfield Documentation](https://rhodesmill.org/skyfield/) — *The official documentation for our decoupled catalog propagation module, explaining SGP4 vector routines and timescale synchronization.*

