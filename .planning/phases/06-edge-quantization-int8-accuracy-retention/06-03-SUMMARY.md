# Plan 06-03: Multi-Backend Inference CLI Integration (`src/infer.py`) — Summary

**Executed:** 2026-11-05  
**Status:** Completed & Verified  
**Requirements Addressed:** `PROD-01`  
**Commit:** `cec8dcd`: `feat(cli): multi-backend inference CLI supporting PyTorch and ONNX Runtime (06-03)`

---

## Accomplishments

1. **Multi-Backend Standalone CLI Architecture (`src/infer.py`):**
   - Implemented an unified command-line tool for inference supporting PyTorch (`.pt`) and ONNX Runtime (`.onnx`, both FP32 and INT8 QDQ formats) without modifying any frozen weights.
   - Built full CLI argument parser with `--source`, `--weights`, `--backend` (`auto`, `pytorch`, `onnx`), `--imgsz`, `--conf`, `--iou`, `--out-dir`, `--json`, and `--device`.
   - Incorporated `CLASS_NAMES` mapping (IDs 0..10) matching [`configs/spark2021.yaml`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/configs/spark2021.yaml).

2. **Preprocessing, Postprocessing & Math Utilities:**
   - Pure-NumPy `letterbox` image padding with aspect-ratio preservation and exact pad coordinate offsets `(dw, dh)`.
   - Bounding box coordinate transformation `xywh2xyxy`.
   - Pure-NumPy Non-Maximum Suppression (`nms`) without dependencies on torchvision or C++ compilation.
   - Exact coordinate unscaling and unpadding:
     $$x_{\text{orig}} = \frac{x_{\text{canvas}} - dw}{r}, \quad y_{\text{orig}} = \frac{y_{\text{canvas}} - dh}{r}$$
     projecting normalized letterbox detections accurately back to original image coordinates.

3. **Inference Engines:**
   - `run_pytorch_inference`: Ultralytics YOLO loader and predictor with confidence filtering and bounding box extraction.
   - `run_onnx_inference`: Direct ONNX Runtime session execution, anchor decoding, dynamic tensor handling, threshold filtering, and pure-NumPy NMS.
   - Verified exact numerical prediction parity between PyTorch and ONNX Runtime backends on sample space imagery:
     - Cloudsat detected at identical bounding box coordinates `[522.19, 517.33, 777.59, 745.24]` with identical confidence (76.01%).
     - ONNX Runtime CPU latency: ~58.79 ms.

4. **Visual Annotation & Structured JSON Export:**
   - `draw_detections`: High-contrast class-specific color bounding boxes and legible labeled banners with confidence percentages.
   - Structured JSON export containing image metadata, backend info, weights path, resolution, latency (ms), and detection lists with coordinates and class IDs.
   - Verified outputs created in [`outputs/inference/`](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/outputs/inference/).

---

## Verification Evidence

- `python src/infer.py --help`: Verified exit code 0 and all argument flags present.
- Unit inspect test: `run_onnx_inference`, `run_pytorch_inference`, `letterbox`, `nms` present and verified.
- End-to-end execution on test frame `data/processed/spark/images/test/Cheops_image_00005_img.jpg`:
  - Output images: `outputs/inference/Cheops_image_00005_img_annotated.jpg` and `outputs/inference/Cheops_image_00005_img.jpg`.
  - Output JSON: `outputs/inference/Cheops_image_00005_img.json` and `outputs/inference/Cheops_image_00005_img_detections.json`.
