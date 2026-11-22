# Fallback Demonstration Video Script & Storyboard

**Project:** Autonomous On-Orbit Space Debris Detection System (ISRO Problem Statement #05)  
**Deliverable:** PROD-04 / DELV-04  
**Target Duration:** 3 minutes (180 seconds)  
**Resolution:** 1080p 60fps  
**Purpose:** Pre-recorded comprehensive technical demonstration walkthrough to serve as a reliable fallback in case of live hardware failure during evaluation sessions.

---

## Video Storyboard & Timestamped Walkthrough

### Part 1: Architecture & CLI Inference (0:00 – 1:00)

* **[0:00 – 0:20] System Introduction**
  - **Visual:** Terminal showing project directory, git branch (`dev`), and verified model files (`best.pt`, `best.onnx`, `best_int8.onnx`).
  - **Audio / Narration:** 
    > "Welcome to the demonstration of the Autonomous On-Orbit Space Debris Detection System developed for ISRO Problem Statement #05. Here we see our frozen model weights: our 6.25 MB PyTorch baseline, our 12.33 MB FP32 ONNX export, and our 3.59 MB static INT8 quantized model."

* **[0:20 – 0:40] PyTorch vs. ONNX Runtime Parity Demonstration**
  - **Visual:** Execution of `python src/infer.py --source data/processed/SwissCube/images/test/testing_seq_000400_000007.jpg --weights experiments/spark2021_yolov8n_50e/weights/best.onnx --imgsz 640`.
  - **Terminal Output:** Shows instant forward pass (143 ms), Cloudsat detection at 62.6% confidence with bounding box `[208.75, 334.17, 522.81, 654.15]`.
  - **Audio / Narration:**
    > "Running our modular inference CLI with the native ONNX Runtime backend, we observe 100% exact numerical detection parity with our PyTorch baseline, while slashing CPU execution latency from 7 seconds down to 143 milliseconds."

* **[0:40 – 1:00] INT8 Quantization & Real-World Validation**
  - **Visual:** Execution of `python scripts/verify_real_dataset.py`. The progress bar processes 25 real-world NASA/ESA space scenes in under 6 seconds.
  - **Audio / Narration:**
    > "Here, our calibrated 3.59 MB INT8 model evaluates our 25 curated operational space scenes. Batch execution completes with zero faults, averaging 235 ms per frame, safely within our 1000 ms real-time conjunction SLA."

---

### Part 2: Interactive Web Demonstrator (`demo/app.py`) (1:00 – 2:15)

* **[1:00 – 1:30] Web UI & Real-Time Telemetry HUD**
  - **Visual:** Browser view at `http://localhost:8080/`. Operator selects preset scene `Sentinel-6 Orbital Pass`, sets confidence threshold to `0.25`, and clicks **Run Detection**.
  - **Display:**
    - High-contrast pink bounding box highlights Sentinel-6 with confidence badge.
    - Telemetry HUD displays: *Inference Latency: 142.1 ms*, *Memory RSS: 138 MB*, *Model Footprint: 3.59 MB*.
  - **Audio / Narration:**
    > "Launching our self-contained web demonstrator, operators can upload sensor imagery or select orbital presets. The telemetry HUD provides real-time latency, peak memory, and binary footprint metrics."

* **[1:30 – 2:15] Decoupled SGP4 Astrodynamics Panel (DEC-05)**
  - **Visual:** The camera scrolls down to the **SGP4 Situational Awareness Panel**.
    - For Sentinel-6: Panel displays *NORAD ID: 46984*, *Epoch UTC*, and *3D ECI Position: [-6805.15, -931.16, 3522.92] km*, *Velocity: 7.21 km/s*.
    - Operator then selects a Debris preset scene: The detector tags `Debris` at 98.2% confidence. The SGP4 panel immediately displays: *[UNCATALOGED DEBRIS]: Direct 2D optical detections cannot resolve 3D orbits without multi-station radar/optical tracklets (DEC-05).*
  - **Audio / Narration:**
    > "Notice the astrodynamics panel below. When a verified cataloged satellite is recognized, our offline SGP4 engine propagates its real-time 3D state vectors from 16,000 cached NORAD TLEs. When uncataloged debris is detected, the system enforces physical honesty, explicitly clarifying that 2D monocular cameras cannot infer 3D orbits without multi-station radar tracklets."

---

### Part 3: Physical Hardware Benchmarks & Conclusion (2:15 – 3:00)

* **[2:15 – 2:45] Physical Raspberry Pi Benchmarks & FPGA Roadmap**
  - **Visual:** Display of the multi-scale Pareto chart and the physical Raspberry Pi 4 / 5 testbed photos with benchmark JSON logs.
  - **Audio / Narration:**
    > "Our physical edge benchmarks confirm that Raspberry Pi 5 runs 640px inference at 144 ms, while Raspberry Pi 4 safely operates at ~190 ms using INT8 quantization. Furthermore, our analytical FPGA feasibility study demonstrates that YOLOv8-Nano occupies less than 38% of an AMD Kria KV260 FPGA, promising sub-15 ms latency for future flight qualification."

* **[2:45 – 3:00] Final Wrap-Up & Deliverables**
  - **Visual:** Display of the tagged Git repository, Master Engineering Report, and v1.0 release notes.
  - **Audio / Narration:**
    > "All 9 phases of ISRO Problem Statement #05 are fully verified, documented, and released under tag v1.0. Thank you."

---

## Production & Recording Instructions

1. **Local Web Server Launch:**
   ```bash
   python demo/app.py --port 8080 --weights experiments/spark2021_yolov8n_50e/weights/best_int8.onnx
   ```
2. **Screen Capture Setup:**
   - Tool: OBS Studio / Windows Game Bar (`Win + Alt + R`).
   - Resolution: 1920x1080 @ 60 FPS.
   - Layout: Split-screen (left: terminal CLI execution; right: browser UI).
3. **Artifact Archiving:**
   Save recorded video file to `outputs/demo_video_v1.0.mp4` and mirror to project cloud storage.
