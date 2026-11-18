"""
tests/test_cv_preprocessing.py
---
Unit tests for computer vision preprocessing utilities:
- Aspect ratio preserving letterbox resizing and padding
- Coordinate reversibility between canvas space and original image space
- Bounding box coordinate transformations (xywh to xyxy)
"""

import numpy as np
import pytest

from src.infer import letterbox, xywh2xyxy


class TestLetterboxPreprocessing:
    """Tests for the letterbox aspect-ratio padding and normalization pipeline."""

    def test_letterbox_output_contract(self):
        """Verify output tensor dimensions, data type, and normalization range."""
        h, w = 1024, 1024
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        target_size = (640, 640)

        blob, r, (dw, dh) = letterbox(img, new_shape=target_size)

        # Shape contract: [1, 3, H, W]
        assert blob.shape == (1, 3, target_size[0], target_size[1]), (
            f"Expected shape (1, 3, {target_size[0]}, {target_size[1]}), got {blob.shape}"
        )
        assert blob.dtype == np.float32, f"Expected float32 dtype, got {blob.dtype}"
        assert blob.min() >= 0.0, f"Pixel minimum should be >= 0.0, got {blob.min()}"
        assert blob.max() <= 1.0, f"Pixel maximum should be <= 1.0, got {blob.max()}"
        assert r == pytest.approx(640 / 1024, rel=1e-3)
        assert dw == 0 and dh == 0, f"Square to square should have zero padding, got dw={dw}, dh={dh}"

    def test_landscape_letterbox_padding(self):
        """Verify horizontal/landscape images get symmetric top/bottom padding."""
        h, w = 1080, 1920
        img = np.zeros((h, w, 3), dtype=np.uint8)
        target_size = (640, 640)

        blob, r, (dw, dh) = letterbox(img, new_shape=target_size)

        expected_r = 640 / 1920
        expected_nh = int(round(1080 * expected_r))  # 360
        expected_dh = (640 - expected_nh) // 2       # 140

        assert r == pytest.approx(expected_r, rel=1e-3)
        assert dw == 0, f"Landscape image should have zero horizontal padding, got {dw}"
        assert dh == expected_dh, f"Expected dh={expected_dh}, got {dh}"

    def test_portrait_letterbox_padding(self):
        """Verify vertical/portrait images get symmetric left/right padding."""
        h, w = 1920, 1080
        img = np.zeros((h, w, 3), dtype=np.uint8)
        target_size = (640, 640)

        blob, r, (dw, dh) = letterbox(img, new_shape=target_size)

        expected_r = 640 / 1920
        expected_nw = int(round(1080 * expected_r))  # 360
        expected_dw = (640 - expected_nw) // 2       # 140

        assert r == pytest.approx(expected_r, rel=1e-3)
        assert dw == expected_dw, f"Expected dw={expected_dw}, got {dw}"
        assert dh == 0, f"Portrait image should have zero vertical padding, got {dh}"

    def test_pad_border_color(self):
        """Verify padding regions are filled with exact standard gray color (114, 114, 114)."""
        h, w = 400, 800
        # Create solid white input image
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        target_size = (640, 640)
        pad_color = (114, 114, 114)

        blob, r, (dw, dh) = letterbox(img, new_shape=target_size, color=pad_color)

        # Transposed tensor shape is [1, 3, H, W] (RGB channels)
        # Check top padding region: row 0 to dh
        expected_norm = 114.0 / 255.0
        top_pad_region = blob[0, :, :dh, :]
        assert np.allclose(top_pad_region, expected_norm, atol=1e-4), (
            "Top padded region does not match expected border color"
        )


class TestCoordinateReversibility:
    """Tests verifying coordinate transformations can be inverted losslessly."""

    @pytest.mark.parametrize("orig_shape", [(1024, 1024), (1080, 1920), (1920, 1080), (720, 1280)])
    def test_coordinate_roundtrip(self, orig_shape):
        """Verify unscaling projected box coordinates reconstructs original box coordinates."""
        orig_h, orig_w = orig_shape
        dummy_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        target_size = (640, 640)

        _, scale_r, (dw, dh) = letterbox(dummy_img, new_shape=target_size)

        # Arbitrary bounding box in original image coordinates [x1, y1, x2, y2]
        x1_orig = 120.5
        y1_orig = 85.25
        x2_orig = 450.75
        y2_orig = 380.0

        # Forward projection to letterbox canvas:
        x1_canvas = x1_orig * scale_r + dw
        y1_canvas = y1_orig * scale_r + dh
        x2_canvas = x2_orig * scale_r + dw
        y2_canvas = y2_orig * scale_r + dh

        # Inverse projection back to original image space:
        x1_reconstructed = (x1_canvas - dw) / scale_r
        y1_reconstructed = (y1_canvas - dh) / scale_r
        x2_reconstructed = (x2_canvas - dw) / scale_r
        y2_reconstructed = (y2_canvas - dh) / scale_r

        assert x1_reconstructed == pytest.approx(x1_orig, abs=1e-4)
        assert y1_reconstructed == pytest.approx(y1_orig, abs=1e-4)
        assert x2_reconstructed == pytest.approx(x2_orig, abs=1e-4)
        assert y2_reconstructed == pytest.approx(y2_orig, abs=1e-4)


class TestBoxMathTransformations:
    """Tests for pure-NumPy coordinate conversions."""

    def test_xywh2xyxy_basic(self):
        """Verify cx, cy, w, h converts correctly to x1, y1, x2, y2."""
        boxes = np.array([
            [100.0, 100.0, 40.0, 60.0],
            [50.0, 50.0, 20.0, 20.0],
        ])
        expected = np.array([
            [80.0, 70.0, 120.0, 130.0],
            [40.0, 40.0, 60.0, 60.0],
        ])
        result = xywh2xyxy(boxes)
        assert np.allclose(result, expected, atol=1e-5)

    def test_xywh2xyxy_area_preservation(self):
        """Verify converted box dimensions preserve original width and height."""
        boxes = np.array([[250.0, 300.0, 75.0, 120.0]])
        result = xywh2xyxy(boxes)

        w_converted = result[0, 2] - result[0, 0]
        h_converted = result[0, 3] - result[0, 1]

        assert w_converted == pytest.approx(75.0, abs=1e-5)
        assert h_converted == pytest.approx(120.0, abs=1e-5)
