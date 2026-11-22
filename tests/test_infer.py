"""
tests/test_infer.py
---
Unit tests for multi-backend CLI inference pipeline:
- Pure-NumPy Non-Maximum Suppression (NMS)
- YOLOv8 output postprocessing and class remapping
- CLI argument parsing and backend resolution
- Forward pass sanity and detection drawing
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from src.infer import (
    CLASS_NAMES,
    draw_detections,
    infer_onnx,
    init_onnx_session,
    nms,
    postprocess,
)


class TestNonMaximumSuppression:
    """Tests for pure-NumPy Non-Maximum Suppression logic."""

    def test_nms_empty_inputs(self):
        """Verify NMS returns empty list when given zero candidate boxes."""
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        keep = nms(boxes, scores, iou_threshold=0.45)
        assert keep == [], "Empty input should return empty keep list"

    def test_nms_single_box(self):
        """Verify single box is unconditionally kept."""
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        scores = np.array([0.95])
        keep = nms(boxes, scores, iou_threshold=0.45)
        assert keep == [0]

    def test_nms_disjoint_boxes_all_kept(self):
        """Verify non-overlapping boxes are all preserved."""
        boxes = np.array([
            [0.0, 0.0, 50.0, 50.0],
            [100.0, 100.0, 150.0, 150.0],
            [200.0, 200.0, 250.0, 250.0],
        ])
        scores = np.array([0.80, 0.90, 0.70])
        keep = nms(boxes, scores, iou_threshold=0.45)
        # Should keep all 3, ordered by descending score: index 1, 0, 2
        assert keep == [1, 0, 2]

    def test_nms_overlapping_suppression(self):
        """Verify redundant overlapping box with lower score is suppressed."""
        boxes = np.array([
            [100.0, 100.0, 200.0, 200.0],  # Box A
            [105.0, 105.0, 205.0, 205.0],  # Box B (high IoU with A)
        ])
        scores = np.array([0.92, 0.85])
        keep = nms(boxes, scores, iou_threshold=0.45)
        assert keep == [0], f"Expected only highest-score box kept, got {keep}"

    def test_nms_iou_threshold_sensitivity(self):
        """Verify modifying iou_threshold suppresses or keeps borderline overlaps."""
        # Two boxes overlapping by ~50%
        boxes = np.array([
            [0.0, 0.0, 100.0, 100.0],
            [30.0, 0.0, 130.0, 100.0],
        ])
        scores = np.array([0.90, 0.80])

        # At strict threshold 0.30, overlap (~0.53) should suppress box 1
        keep_strict = nms(boxes, scores, iou_threshold=0.30)
        assert keep_strict == [0]

        # At lenient threshold 0.70, both should be kept
        keep_lenient = nms(boxes, scores, iou_threshold=0.70)
        assert keep_lenient == [0, 1]


class TestPostprocessing:
    """Tests for postprocess function decoding raw YOLOv8 prediction tensors."""

    def test_postprocess_confidence_filtering(self):
        """Verify detections with scores below confidence threshold are omitted."""
        # Create synthetic tensor [1, 15, 2] (2 anchors, 15 channels: 4 bbox coords + 11 classes)
        raw_output = np.zeros((1, 15, 2), dtype=np.float32)

        # Anchor 0: center (320, 320), w=100, h=100, Class 6 (Debris) score = 0.85
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]
        raw_output[0, 4 + 6, 0] = 0.85

        # Anchor 1: center (100, 100), w=50, h=50, Class 0 (AcrimSat) score = 0.15 (below 0.25)
        raw_output[0, :4, 1] = [100.0, 100.0, 50.0, 50.0]
        raw_output[0, 4 + 0, 1] = 0.15

        orig_shape = (640, 640)
        detections = postprocess(
            raw_output=raw_output,
            orig_shape=orig_shape,
            scale_r=1.0,
            pad=(0, 0),
            conf=0.25,
            iou=0.45,
        )

        assert len(detections) == 1, f"Expected 1 detection above 0.25 conf, got {len(detections)}"
        assert detections[0]["class_id"] == 6
        assert detections[0]["class_name"] == "Debris"
        assert detections[0]["confidence"] == pytest.approx(0.85, abs=1e-3)
        assert detections[0]["bbox"] == [270.0, 270.0, 370.0, 370.0]

    def test_postprocess_boundary_clipping(self):
        """Verify bounding boxes extending beyond image boundaries are clamped."""
        raw_output = np.zeros((1, 15, 1), dtype=np.float32)
        # Box centered near edge with size spilling over 0 and 640
        raw_output[0, :4, 0] = [10.0, 10.0, 100.0, 100.0]  # x1 would be -40, y1 would be -40
        raw_output[0, 4 + 5, 0] = 0.90  # CubeSat

        detections = postprocess(
            raw_output=raw_output,
            orig_shape=(640, 640),
            scale_r=1.0,
            pad=(0, 0),
            conf=0.25,
            iou=0.45,
        )

        assert len(detections) == 1
        bbox = detections[0]["bbox"]
        assert bbox[0] >= 0.0, f"x1 should be clamped to >= 0, got {bbox[0]}"
        assert bbox[1] >= 0.0, f"y1 should be clamped to >= 0, got {bbox[1]}"
        assert bbox[2] <= 640.0
        assert bbox[3] <= 640.0


class TestCLIAndInferenceSanity:
    """Tests for CLI arguments, model initialization, and visual drawing."""

    def test_class_names_mapping(self):
        """Verify 11-class mapping matches SPARK-2021 specification."""
        assert len(CLASS_NAMES) == 11
        assert CLASS_NAMES[0] == "AcrimSat"
        assert CLASS_NAMES[6] == "Debris"
        assert CLASS_NAMES[10] == "TRMM"

    def test_draw_detections_annotation(self):
        """Verify visual bounding box drawer renders and outputs valid image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {
                "class_id": 6,
                "class_name": "Debris",
                "confidence": 0.945,
                "bbox": [100.0, 120.0, 200.0, 240.0],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_annotated.jpg"
            annotated = draw_detections(img, detections, out_path=str(out_file))

            assert annotated.shape == img.shape
            assert out_file.exists(), "Annotated image should be written to disk"
            read_back = cv2.imread(str(out_file))
            assert read_back is not None
            # Pixel in the box border area should no longer be purely black
            box_area = read_back[120:240, 100:200]
            assert np.any(box_area > 0), "Bounding box should be visibly drawn"

    def test_onnx_forward_pass_sanity(self):
        """Verify ONNX forward pass execution with frozen model if available."""
        model_path = Path("experiments/spark2021_yolov8n_50e/weights/best.onnx")
        if not model_path.exists():
            pytest.skip("best.onnx model weights not present in repository")

        session = init_onnx_session(model_path, device="cpu")
        dummy_tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)

        # Run forward pass and verify return format
        detections = infer_onnx(session, dummy_tensor, orig_shape=(640, 640))
        assert isinstance(detections, list), f"Expected list of detections, got {type(detections)}"

    def test_run_onnx_inference_numpy_array(self):
        """Verify run_onnx_inference accepts in-memory BGR numpy array without disk writes."""
        from src.infer import run_onnx_inference
        model_path = Path("experiments/spark2021_yolov8n_50e/weights/best_int8.onnx")
        if not model_path.exists():
            pytest.skip("best_int8.onnx model weights not present in repository")

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)

        detections, latency_ms = run_onnx_inference(
            weights=str(model_path),
            image_path=img,
            conf=0.10,
            iou=0.45,
            imgsz=640,
            device="cpu",
        )

        assert isinstance(detections, list)
        assert isinstance(latency_ms, float)
        assert latency_ms > 0

