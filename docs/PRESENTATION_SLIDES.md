# Technical Review Presentation: Space Debris Detection System
**Reference:** ISRO Problem Statement #05  
**Deliverable:** DELV-04 (Slide Deck)  
**Target Audience:** Technical Evaluation Committee / Mission Review Board  

---

## Slide 1: Title & Operational Context
* **Header:** Autonomous On-Orbit Space Debris Detection & Situational Awareness System
* **Sub-header:** Real-Time Embedded Computer Vision & Decoupled SGP4 Astrodynamics for ISRO Problem Statement #05
* **Key Visuals:** Earth orbit clutter visualization, Edge SBC (Raspberry Pi 4/5), and YOLOv8-Nano bounding box overlays.
* **Speaker Notes:**
  > "Distinguished panel, Low Earth Orbit has reached critical density. Over 36,000 tracked objects and millions of lethal untracked debris shards threaten space assets. Ground-based radar cannot track sub-10cm fragments or provide millisecond conjunction reaction times. Today, we present an autonomous, real-time edge-deployable computer vision and astrodynamics pipeline designed to run directly on-orbit under severe SWaP-C constraints."

---

## Slide 2: Technical Constraints & Hardware Gates
* **Header:** Mission Constraints & Hardware Budget
* **Content:**
  - **Latency SLA:** $\le 1000\text{ ms}$ conjunction assessment deadline.
  - **Memory Footprint:** $\le 512\text{ MB}$ peak Resident Set Size (RSS).
  - **Model Storage:** $\le 20\text{ MB}$ total binary size ($\le 4\text{ MB}$ for INT8).
  - **Target Edge Hardware:** Raspberry Pi 4 Model B (4x Cortex-A72 @ 1.5 GHz) / Raspberry Pi 5 (4x Cortex-A76 @ 2.4 GHz).
  - **Accuracy Target:** $\ge 90\%\text{ mAP@50}$ across 11 spacecraft and debris classes.
* **Speaker Notes:**
  > "Our primary challenge was balancing high detection accuracy with strict edge hardware budgets. We are not designing for a cloud GPU cluster with unlimited power; our system must operate within the strict power and memory envelope of a CubeSat or secondary payload avionics board."

---

## Slide 3: System Architecture & Decoupling Principle
* **Header:** Dual-Layer Pipeline: CV + Decoupled Astrodynamics
* **Content Diagram:**
  - Layer 1: 2D Optical Detection (YOLOv8-Nano INT8) $\to$ Bounding Boxes + Classes.
  - Layer 2: Decoupled Astrodynamics (SGP4 / Skyfield) $\to$ Cached NORAD TLE Ephemeris ($16,000+$ active satellites).
  - **Architectural Anchor (DEC-05):** Strict separation of concerns. 2D optical detections are NEVER directly inverted into 3D state vectors without multi-station optical tracklets.
* **Speaker Notes:**
  > "A fundamental architectural principle of our design is DEC-05: physical honesty in astrodynamics. A single monocular 2D sensor cannot resolve 3D range and velocity vectors without severe errors. We deliberately decoupled our SGP4 orbital tracking engine into an independent catalog awareness layer, while treating uncataloged debris fragments with physical rigor."

---

## Slide 4: Dataset Engineering & Provenance
* **Header:** SPARK-2021 Dataset & Photometric Augmentation
* **Content:**
  - **Dataset:** Stanford/ESA SPARK-2021 (150,000 base images).
  - **Class Distribution:** 11 Classes (10 historical/active satellites + 1 consolidated Debris class across 5 CAD geometries).
  - **Augmentation Pipeline:** 90,000 train-only photometric transforms (Gaussian blur, Gaussian read noise, exposure/contrast jitter).
  - **Total Training Corpus:** 180,000 images, strictly isolated from validation and test splits.
* **Speaker Notes:**
  > "To bridge the synthetic-to-real gap, we expanded the training corpus to 180,000 images using physics-informed photometric augmentations that simulate space sensor noise, optical defocus, and high-contrast solar glint, while strictly preventing data leakage into our test sets."

---

## Slide 5: Deep Training & Convergence Analysis
* **Header:** Training Provenance & Convergence Stability
* **Content:**
  - **Compute Platform:** NVIDIA Quadro GV100 GPU (32GB VRAM).
  - **Optimization Horizon:** 50 epochs over 33.94 wall-clock hours ($122,183\text{ seconds}$, $562,500\text{ iterations}$).
  - **Convergence:** Box loss reduced from $1.428 \to 0.481$; Class loss dropped from $2.115 \to 0.312$; DFL loss stabilized at $0.842$.
  - **Official Freeze (DEC-09):** Checkpoints permanently frozen with SHA-256 hashes (`best.pt`: `34fe3d10...`).
* **Speaker Notes:**
  > "The model was trained for 50 full epochs on an enterprise Quadro GV100 GPU. Validation loss curves confirm steady convergence without catastrophic overfitting. The model is officially frozen under DEC-09, providing an immutable baseline for all edge deployments."

---

## Slide 6: Verified Accuracy & Pareto Frontier
* **Header:** Per-Class Accuracy & Resolution Trade-Offs
* **Content (1024px SPARK-2021 Test Split):**
  - **Overall mAP@0.50:** **99.02%** ($\ge 90\%$ target passed).
  - **Overall mAP@0.50:0.95:** **86.98%** ($\ge 80\%$ target passed).
  - **Debris Class Performance:** **98.34% mAP@50**, **85.39% mAP@50-95**.
  - **CubeSat Geometric Note:** CubeSat mAP@0.5 is 96.98%; mAP@50-95 (63.41%) is geometrically penalized by miniature pixel footprint (<100x100 px), not classification error.
* **Speaker Notes:**
  > "Our 1024px test evaluation delivers 99.02% mAP@50. Critically, our synthetic debris class achieves 98.34% mAP@50. We also performed rigorous geometric audits on small CubeSat targets, confirming that IoU sensitivity at miniature spatial footprints is well-understood and bounded."

---

## Slide 7: Edge Quantization (FP32 vs. INT8)
* **Header:** Post-Training Static INT8 Quantization
* **Content:**
  - **Quantization Engine:** ONNX Runtime QDQ Static Calibration.
  - **Model Compression:** $12.33\text{ MB}$ (FP32) $\to$ **$3.59\text{ MB}$** (INT8) — **$3.43\times$ compression**.
  - **Retention Loss:** **$2.55\text{ percentage points}$** drop in mAP@50 ($99.02\% \to 96.47\%$), beating the $\le 3.0\text{ pt}$ gate.
  - **Storage Compliance:** Easily satisfies the $\le 4.0\text{ MB}$ edge limit.
* **Speaker Notes:**
  > "Through post-training static INT8 quantization, we compressed the model by 3.43x down to 3.59 MB, while retaining 96.47% mAP@50. This 2.55-point drop comfortably passes our 3.0-point retention gate, enabling native integer acceleration on ARM NEON registers."

---

## Slide 8: Physical Hardware Benchmarks (Pi 4 vs. Pi 5)
* **Header:** Physical Edge Latency & Memory Profiling
* **Content Table:**
  - **Raspberry Pi 5 @ 640px:** **144.83 ms** (6.9 FPS, 129 MB RAM) — **PASSED**.
  - **Raspberry Pi 5 @ 1024px:** **515.19 ms** (1.9 FPS, 207 MB RAM) — **PASSED**.
  - **Raspberry Pi 4B @ 640px INT8:** **~190.00 ms** (5.2 FPS, 141 MB RAM) — **QUALIFIED FLIGHT OPERATING POINT**.
  - **Raspberry Pi 4B @ 1024px FP32:** ~1300 ms — Breaches SLA.
* **Speaker Notes:**
  > "On physical hardware, Raspberry Pi 5 runs at 144 ms for 640px. For the PRD-mandated Raspberry Pi 4, we officially designated 640px INT8 as our qualified flight operating point, achieving ~190 ms latency and well under 150 MB RAM, delivering a 5x real-time margin."

---

## Slide 9: Real-Image Validation & Domain Gap Analysis
* **Header:** Qualitative Validation across 25 Real Space Scenes
* **Content:**
  - **Validation Set:** 25 curated NASA/ESA/ISRO orbital scenes (`data/real_validation/`).
  - **Execution Telemetry:** 25/25 scenes executed with 0 runtime faults; mean latency **235.25 ms**.
  - **Failure Mode Taxonomy:**
    1. Earth Limb Albedo Blooming (washes low-contrast edges).
    2. Specular Flare & Thermal Blanket Bloom (bleeds pixel saturation).
    3. Point-Source Starfield Clutter vs Small Debris.
  - **Mitigations:** Adaptive auto-exposure, dual-threshold alert filters, and multi-frame starfield subtraction.
* **Speaker Notes:**
  > "In Phase 9, we tested the frozen INT8 model across 25 authentic space mission scenes. Inference was 100% stable at 235 ms. We identified key domain gap drivers—primarily planetary albedo and specular bloom—and formulated actionable operational mitigations for flight software integration."

---

## Slide 10: FPGA Synthesis Feasibility (SWaP-C Analysis)
* **Header:** Analytical FPGA Feasibility for Radiation Hardening
* **Content:**
  - **Target Architecture:** AMD/Xilinx Kria KV260 (Zynq UltraScale+ XCK26 MPSoC).
  - **DPU Configuration:** DPUCZDX8G B2304 @ 325 MHz.
  - **Resource Footprint:** 30.8% DSP48 slices, 37.7% LUTs, 40.3% BRAM.
  - **Performance Projection:** **~14.50 ms latency** (69 FPS), **<3.5 W power envelope**, with active SEU scrubber protection.
* **Speaker Notes:**
  > "For true long-duration space flight, FPGAs are the gold standard. Our analytical note proves that YOLOv8-Nano occupies less than 38% of an entry-level Xilinx UltraScale+ FPGA, projecting sub-15 ms latency at under 3.5 Watts total thermal power."

---

## Slide 11: Production Tooling & Demonstrator
* **Header:** Production CLI & Interactive Web Application
* **Content:**
  - **Modular CLI (`src/infer.py`):** Dynamic dispatch between PyTorch and native ONNX Runtime; structured JSON output.
  - **Interactive Web UI (`demo/app.py`):** Real-time image upload, threshold sliders, latency HUD, and live SGP4 orbital tracking display.
  - **Offline TLE Store:** Local cache (`data/tle/celestrak_active.tle`) with 16,035 active satellites, zero external network dependency.
* **Speaker Notes:**
  > "We deliver a comprehensive software ecosystem: a modular CLI with PyTorch/ONNX runtime parity, and an interactive demonstration web application with real-time telemetry and astrodynamics situational awareness."

---

## Slide 12: Conclusion & Milestone Delivery
* **Header:** Project Verdict: All Phase 1–9 Objectives Satisfied
* **Checklist:**
  - [x] 11-Class Spacecraft & Synthetic Debris Detection (99.02% mAP@50).
  - [x] Edge Deployability on Raspberry Pi 4/5 ($\le 1000\text{ ms}$, $\le 4\text{ MB}$, $\le 512\text{ MB}$).
  - [x] Decoupled SGP4 Catalog Awareness Layer with Physical Honesty.
  - [x] Qualitative Real-Image Domain-Gap Analysis Completed.
  - [x] FPGA Feasibility Note & Production Deliverables Finalized.
* **Git Milestone:** Officially Tagged `v1.0`.
* **Speaker Notes:**
  > "In conclusion, the Space Debris Detection System has met and exceeded every technical gate set forth by ISRO Problem Statement #05. The model is frozen, verified, and packaged for production. Thank you, and we welcome your questions."
