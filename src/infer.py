"""
src/infer.py
---
Multi-backend standalone CLI for edge-deployable space debris and spacecraft detection.
Supports both PyTorch (.pt) and ONNX Runtime (.onnx, FP32 and INT8) backends.

Provides complete letterbox preprocessing, forward pass execution, anchor box decoding,
pure-NumPy Non-Maximum Suppression (NMS), bounding box annotation, and structured JSON export.

Requirement: PROD-01
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Class names mapping matching configs/spark2021.yaml (IDs 0..10)
CLASS_NAMES: List[str] = [
    "AcrimSat",
    "Aquarius",
    "Aura",
    "Calipso",
    "Cloudsat",
    "CubeSat",
    "Debris",
    "Jason",
    "Sentinel-6",
    "Terra",
    "TRMM",
]

# High-contrast color palette for visualization (BGR format)
PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 0),       # AcrimSat - Green
    (255, 128, 0),     # Aquarius - Orange/Blue
    (0, 165, 255),     # Aura - Orange
    (255, 0, 255),     # Calipso - Magenta
    (255, 255, 0),     # Cloudsat - Cyan
    (0, 255, 255),     # CubeSat - Yellow
    (0, 0, 255),       # Debris - Red (High alert)
    (200, 200, 0),     # Jason - Teal
    (147, 20, 255),    # Sentinel-6 - Pink
    (0, 128, 255),     # Terra - Coral
    (255, 0, 128),     # TRMM - Purple
]


def letterbox(
    img: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image and pad to new_shape maintaining aspect ratio.

    Args:
        img: Input BGR image as numpy array [H, W, C].
        new_shape: Target (height, width) or int.
        color: Padding border color (default gray 114).

    Returns:
        blob: Transposed, normalized float32 tensor [1, 3, H, W] in RGB order.
        r: Scale ratio applied to original image.
        (dw, dh): Left/right and top/bottom padding offsets in pixels.
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    th, tw = new_shape
    h, w = img.shape[:2]

    # Compute aspect-preserving scale ratio
    r = min(tw / w, th / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw = (tw - nw) // 2
    dh = (th - nh) // 2

    # Resize image
    if (w, h) != (nw, nh):
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img

    # Construct padded canvas
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    canvas[dh : dh + nh, dw : dw + nw] = resized

    # Convert BGR -> RGB, scale to [0.0, 1.0], transpose to [1, 3, H, W]
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]
    return np.ascontiguousarray(blob), r, (dw, dh)


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from [cx, cy, w, h] format to [x1, y1, x2, y2].

    Args:
        boxes: Array of shape [N, 4] with (cx, cy, w, h).

    Returns:
        Array of shape [N, 4] with (x1, y1, x2, y2).
    """
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0  # y2
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """
    Pure-NumPy Non-Maximum Suppression (NMS).

    Args:
        boxes: Array of shape [N, 4] in (x1, y1, x2, y2) format.
        scores: Array of shape [N] with confidence scores.
        iou_threshold: Intersection over Union suppression threshold.

    Returns:
        List of selected detection indices in descending score order.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def run_pytorch_inference(
    weights: str,
    image_path: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    model: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Execute PyTorch inference using Ultralytics YOLO engine.

    Args:
        weights: Path to PyTorch model weights (.pt).
        image_path: Path to input image file.
        imgsz: Inference image size.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        device: Execution device ('cpu', 'cuda', '0').
        model: Optional pre-instantiated YOLO model.

    Returns:
        detections: List of detection dictionaries.
        latency_ms: Inference execution time in milliseconds.
    """
    from ultralytics import YOLO

    if model is None:
        model = YOLO(weights)

    t0 = time.perf_counter()
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    detections: List[Dict[str, Any]] = []
    if results and len(results) > 0:
        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes_data = res.boxes.xyxy.cpu().numpy()
            scores_data = res.boxes.conf.cpu().numpy()
            classes_data = res.boxes.cls.cpu().numpy().astype(int)

            for b, s, c in zip(boxes_data, scores_data, classes_data):
                cls_id = int(c)
                cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": float(round(float(s), 4)),
                    "bbox": [float(round(float(v), 2)) for v in b],
                })

    return detections, latency_ms


def run_onnx_inference(
    weights: str,
    image_path: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    session: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Execute ONNX Runtime inference supporting both FP32 and INT8 models.

    Args:
        weights: Path to ONNX model weights (.onnx).
        image_path: Path to input image file.
        imgsz: Inference resolution.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        device: Target device ('cpu', 'cuda', '0').
        session: Optional pre-instantiated ort.InferenceSession.

    Returns:
        detections: List of detection dictionaries.
        latency_ms: Inference execution time in milliseconds.
    """
    import onnxruntime as ort

    if session is None:
        providers = ["CPUExecutionProvider"]
        if device.lower() in ("cuda", "0", "gpu"):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(weights, providers=providers)

    input_node = session.get_inputs()[0]
    input_name = input_node.name

    # Read original image
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Failed to read image at: {image_path}")
    orig_h, orig_w = raw_img.shape[:2]

    # Preprocessing with letterbox
    blob, scale_r, (dw, dh) = letterbox(raw_img, new_shape=(imgsz, imgsz))

    # Inference forward pass
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: blob})
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    raw_output = outputs[0]  # Expected shape [1, 15, 8400] for YOLOv8 (11 classes + 4 coords)
    if raw_output.ndim == 3 and raw_output.shape[1] == 15:
        preds = np.transpose(raw_output[0], (1, 0))  # [8400, 15]
    elif raw_output.ndim == 3 and raw_output.shape[2] == 15:
        preds = raw_output[0]  # [8400, 15]
    else:
        # Fallback transpose if shape is [1, num_channels, num_anchors]
        preds = np.transpose(raw_output[0], (1, 0))

    # Box coordinates (columns 0..3: cx, cy, w, h in canvas space)
    boxes_cxcywh = preds[:, :4]
    # Class probability scores (columns 4..14: 11 classes)
    cls_scores = preds[:, 4:]

    max_scores = np.max(cls_scores, axis=1)
    class_ids = np.argmax(cls_scores, axis=1)

    # Filter by confidence threshold
    score_mask = max_scores >= conf
    if not np.any(score_mask):
        return [], latency_ms

    filtered_boxes = boxes_cxcywh[score_mask]
    filtered_scores = max_scores[score_mask]
    filtered_classes = class_ids[score_mask]

    # Convert cxcywh -> xyxy in padded canvas coordinates
    canvas_xyxy = xywh2xyxy(filtered_boxes)

    # Apply pure-NumPy NMS
    keep_indices = nms(canvas_xyxy, filtered_scores, iou_threshold=iou)
    if not keep_indices:
        return [], latency_ms

    final_boxes = canvas_xyxy[keep_indices]
    final_scores = filtered_scores[keep_indices]
    final_classes = filtered_classes[keep_indices]

    # Unpad and scale coordinates back to original image dimensions
    # x_orig = (x_canvas - dw) / scale_r
    # y_orig = (y_canvas - dh) / scale_r
    detections: List[Dict[str, Any]] = []
    for b, s, c in zip(final_boxes, final_scores, final_classes):
        x1 = (b[0] - dw) / scale_r
        y1 = (b[1] - dh) / scale_r
        x2 = (b[2] - dw) / scale_r
        y2 = (b[3] - dh) / scale_r

        # Clip coordinates to image boundary
        x1 = max(0.0, min(float(orig_w), float(x1)))
        y1 = max(0.0, min(float(orig_h), float(y1)))
        x2 = max(0.0, min(float(orig_w), float(x2)))
        y2 = max(0.0, min(float(orig_h), float(y2)))

        cls_id = int(c)
        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": float(round(float(s), 4)),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })

    return detections, latency_ms


def draw_detections(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    out_path: Optional[str] = None,
) -> np.ndarray:
    """
    Annotate image with high-contrast bounding boxes, class labels, and confidences.

    Args:
        img: Input BGR image array.
        detections: List of detection dictionaries.
        out_path: Optional path to save annotated image.

    Returns:
        Annotated BGR image array.
    """
    annotated = img.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        conf = det["confidence"]

        # Select color
        color = PALETTE[cls_id % len(PALETTE)]

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label formatting
        label = f"{cls_name} {conf:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw background banner for label readability
        banner_y1 = max(0, y1 - txt_h - baseline - 4)
        banner_y2 = y1
        banner_x1 = x1
        banner_x2 = min(annotated.shape[1], x1 + txt_w + 6)

        cv2.rectangle(annotated, (banner_x1, banner_y1), (banner_x2, banner_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (banner_x1 + 3, banner_y2 - baseline - 1),
            font,
            font_scale,
            (0, 0, 0) if (color[0] + color[1] + color[2]) > 380 else (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), annotated)

    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Multi-backend CLI for space debris and spacecraft detection (PyTorch / ONNX Runtime).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image file or directory containing images.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="experiments/spark2021_yolov8n_50e/weights/best.onnx",
        help="Path to model weights file (.pt or .onnx, including INT8).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "pytorch", "onnx"],
        default="auto",
        help="Inference engine backend. 'auto' determines backend from weights file extension.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference resolution (square image dimension).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence score threshold for filtering detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for Non-Maximum Suppression (NMS).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/inference/",
        help="Directory where annotated output images and reports are saved.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Export detection metadata and latencies to structured JSON files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target execution device ('cpu', 'cuda', '0').",
    )

    args = parser.parse_args()

    # Determine backend
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    backend = args.backend
    if backend == "auto":
        suffix = weights_path.suffix.lower()
        if suffix in (".pt", ".pth"):
            backend = "pytorch"
        elif suffix == ".onnx":
            backend = "onnx"
        else:
            raise ValueError(f"Cannot auto-detect backend for weights file extension: {suffix}")

    print("=" * 65)
    print("  EDGE-DEPLOYABLE SPACE DEBRIS & SPACECRAFT INFERENCE")
    print(f"  Backend:       {backend.upper()}")
    print(f"  Weights:       {weights_path}")
    print(f"  Source:        {args.source}")
    print(f"  Resolution:    {args.imgsz}x{args.imgsz}")
    print(f"  Conf / IoU:    {args.conf} / {args.iou}")
    print(f"  Output Dir:    {args.out_dir}")
    print(f"  Device:        {args.device}")
    print("=" * 65 + "\n")

    # Collect source images
    source_p = Path(args.source)
    if source_p.is_file():
        image_files = [source_p]
    elif source_p.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [p for p in source_p.iterdir() if p.suffix.lower() in valid_exts]
        if not image_files:
            raise FileNotFoundError(f"No valid image files found in directory: {source_p}")
    else:
        raise FileNotFoundError(f"Source path not found: {source_p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize shared model/session for batch efficiency
    shared_session = None
    shared_model = None
    if backend == "onnx":
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        if args.device.lower() in ("cuda", "0", "gpu"):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        shared_session = ort.InferenceSession(str(weights_path), providers=providers)
    elif backend == "pytorch":
        from ultralytics import YOLO
        shared_model = YOLO(str(weights_path))

    total_images = len(image_files)
    all_results: List[Dict[str, Any]] = []

    for idx, img_file in enumerate(image_files, 1):
        raw_img = cv2.imread(str(img_file))
        if raw_img is None:
            print(f"[{idx}/{total_images}] WARNING: Skipping unreadable image {img_file.name}")
            continue

        if backend == "onnx":
            detections, latency_ms = run_onnx_inference(
                weights=str(weights_path),
                image_path=str(img_file),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                session=shared_session,
            )
        else:
            detections, latency_ms = run_pytorch_inference(
                weights=str(weights_path),
                image_path=str(img_file),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                model=shared_model,
            )

        # Draw and save annotated visual image
        annotated_out_path = out_dir / f"{img_file.stem}_annotated.jpg"
        draw_detections(raw_img, detections, out_path=str(annotated_out_path))

        # Also save as exact original filename in out-dir if requested by standard pipelines
        direct_out_path = out_dir / img_file.name
        cv2.imwrite(str(direct_out_path), cv2.imread(str(annotated_out_path)))

        result_entry: Dict[str, Any] = {
            "image": img_file.name,
            "source_path": str(img_file.resolve()),
            "backend": backend,
            "weights": str(weights_path.resolve()),
            "imgsz": args.imgsz,
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "inference_latency_ms": round(latency_ms, 2),
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image": str(annotated_out_path.resolve()),
        }
        all_results.append(result_entry)

        # Print detection summary
        print(f"[{idx}/{total_images}] {img_file.name} | Latency: {latency_ms:.2f} ms | Detections: {len(detections)}")
        for d in detections:
            print(f"    - {d['class_name']}: {d['confidence']:.2%} at bbox {d['bbox']}")

        if args.json:
            # Save per-image JSON output
            json_file = out_dir / f"{img_file.stem}_detections.json"
            json_file_simple = out_dir / f"{img_file.stem}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result_entry, f, indent=2)
            with open(json_file_simple, "w", encoding="utf-8") as f:
                json.dump(result_entry, f, indent=2)

    if args.json and total_images > 1:
        batch_summary_file = out_dir / "batch_inference_summary.json"
        with open(batch_summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_images": total_images,
                "backend": backend,
                "weights": str(weights_path.resolve()),
                "results": all_results,
            }, f, indent=2)
        print(f"\n[INFO] Batch JSON summary written to: {batch_summary_file}")

    print("\n" + "-" * 55)
    print(f"  INFERENCE COMPLETED: {len(all_results)} images processed")
    print(f"  Annotated outputs saved in: {out_dir}")
    print("-" * 55 + "\n")


if __name__ == "__main__":
    main()

