# Phase 6: Edge Quantization, Hardware Qualification & Astrodynamics Cleansing — Context

**Gathered:** 2026-11-04  
**Role:** Person B: Data/Edge Lead  
**Focus:** Hardware benchmarks, orbital logic, data pipelines, and ONNX/INT8 deployment.  
**Governing Decisions:** DEC-01, DEC-05, DEC-07, DEC-09, DEC-10, DEC-11.  

---

<domain>
## Phase Boundary & Work Distribution

Phase 6 addresses the core Edge Deployment and Quantization milestone (EDGE-04, EDGE-05) alongside critical P1 audit remediations in astrodynamics logic, hardware qualification reporting, and data leakage disclosure:

### Priority 1 (P1): Audit Remediations & Architectural Rigor
1. **Qualify Raspberry Pi Hardware Gates:**
   - Explicitly update engineering reports (`docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md`, `docs/PHASE_1_VERIFICATION_AND_RESULTS.md`, `docs/PHASE_1_COMPREHENSIVE_REPORT.md`, `docs/WORKFLOW_PLAYBOOK.md`, etc.) to state that the $\le 1000\text{ ms}$ real-time gate was validated on a physical **Raspberry Pi 5** (ARM Cortex-A76 @ 2.4 GHz).
   - Formally designate **640px** as the safe operating point projected to pass on the PRD-mandated **Raspberry Pi 4** (ARM Cortex-A72 @ 1.5 GHz) based on empirical scaling factors ($2.2\times–2.5\times$ Cortex-A76 to A72 delta $\implies 144.83\text{ ms} \times 2.5 \approx 362\text{ ms} \ll 1000\text{ ms}$, whereas 1024px at $515.19\text{ ms} \times 2.5 \approx 1288\text{ ms}$ risks exceeding the budget).
2. **Cleanse Astrodynamics Logic:**
   - Remove the hardcoded ISS proxy (`"CubeSat": 25544`) from `src/data/tle_pipeline.py`. `CubeSat` must return an uncataloged/no-TLE status consistent with other uncataloged classes.
   - Redraw the system architecture diagram across all documentation to present the TLE subsystem strictly as an independent catalog awareness layer for known satellites, completely decoupled from visual 2D detections and eliminating claims of automated conjunction risk calculation or direct orbit determination from 2D bounding boxes.
3. **Disclose Data Leakage Caveats:**
   - Document the dataset split leakage caveat in relevant technical reports: because global perceptual hashing (dHash) collapsed on the uniform solid black space backgrounds (>95% black pixels), sequence-level / trajectory-level leakage between train, val, and test splits is possible and not ruled out.

### Priority 2 (P2): Edge Quantization & Inference Integration
4. **Execute INT8 Quantization:**
   - Implement static post-training INT8 quantization using ONNX Runtime quantization tooling (`onnxruntime.quantization.quantize_static`) with a dedicated calibration data reader.
   - Sample 300 representative images from the training split (`data/processed/spark/images/train/`).
   - Produce the quantized INT8 model (`best_int8.onnx`).
   - Evaluate accuracy retention loss on the test split using `src/eval.py`, verifying that the mAP drop remains within acceptable bounds ($\le 3.0\text{ points}$ mAP@50 per EDGE-05).
5. **Integrate ONNX into CLI:**
   - Implement or update `src/infer.py` to support multi-backend inference (`--backend pytorch|onnx`), with complete preprocessing (letterbox scaling, RGB normalization, NCHW transpose) and postprocessing (box scaling, confidence thresholding, NMS) for ONNX Runtime.

</domain>

<decisions>
## Implementation Decisions

### 1. Hardware Qualification Standard (DEC-11 Alignment)
- Clear distinction between **validated platform** (Raspberry Pi 5) and **target gate platform** (Raspberry Pi 4).
- 640px is designated as the recommended deployment resolution for Cortex-A72 hardware, offering ~360 ms projected latency and >6 FPS headroom under the 1000 ms ceiling.

### 2. Astrodynamics Boundary (DEC-05 Alignment)
- Remove `"CubeSat": 25544`. CubeSats in LEO do not share a single ephemeris with the ISS (NORAD 25544).
- TLE pipeline returns an informational error/notice when queried with `CubeSat` or `Debris`, enforcing physical integrity.
- Architecture diagrams strictly portray a dual-track parallel monitoring system: optical bearing awareness vs. ground-catalog tracking.

### 3. ONNX Runtime Static Quantization Strategy (EDGE-04 / EDGE-05)
- Use `CalibrationDataReader` in `onnxruntime.quantization` with `QuantType.QInt8` (or `QUInt8` per activation/weight convention).
- Calibration batch size = 1, total calibration samples = 300 images.
- Quantized model output path: `experiments/spark2021_yolov8n_50e/weights/best_int8.onnx`.

### 4. CLI Inference Design (`src/infer.py` - PROD-01)
- Supports single image, directory of images, or batch inference.
- Backends: `pytorch` (Ultralytics YOLO wrapper) and `onnx` (Direct ONNX Runtime InferenceSession).
- Output: Annotated output image saved to disk + structured detection dictionary (JSON output support).
</decisions>

<canonical_refs>
## Canonical References

- [MASTER_PROJECT_REPORT.md](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/MASTER_PROJECT_REPORT.md)
- [SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md)
- [PHASE_1_VERIFICATION_AND_RESULTS.md](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/PHASE_1_VERIFICATION_AND_RESULTS.md)
- [PHASE_1_COMPREHENSIVE_REPORT.md](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/PHASE_1_COMPREHENSIVE_REPORT.md)
- [decisions.md (DEC-01, DEC-05, DEC-07, DEC-09, DEC-11)](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/decisions.md)
- [src/data/tle_pipeline.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/data/tle_pipeline.py)
- [src/eval.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/eval.py)
- [edge/benchmark.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/edge/benchmark.py)
</canonical_refs>

