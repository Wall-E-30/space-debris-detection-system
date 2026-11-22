"""
tests/test_split_dataset.py
Unit tests for src/data/split_dataset.py coordinate conversion and utilities.
"""

import pytest
from src.data.split_dataset import box_to_yolo_line, write_data_yaml


def test_box_to_yolo_line_xyxy():
    # 1000x1000 image, box [100, 200, 300, 600] -> x1=100, x2=300, y1=200, y2=600
    # cx = 200/1000 = 0.2, cy = 400/1000 = 0.4, w = 200/1000 = 0.2, h = 400/1000 = 0.4
    line = box_to_yolo_line([100, 200, 300, 600], 1000, 1000, 0, coord_format="xyxy")
    parts = line.split()
    assert parts[0] == "0"
    assert pytest.approx(float(parts[1]), abs=1e-5) == 0.2
    assert pytest.approx(float(parts[2]), abs=1e-5) == 0.4
    assert pytest.approx(float(parts[3]), abs=1e-5) == 0.2
    assert pytest.approx(float(parts[4]), abs=1e-5) == 0.4


def test_box_to_yolo_line_row_col():
    # SPARK-2021 ordering: [r_min, c_min, r_max, c_max] == [y_min, x_min, y_max, x_max]
    # box = [200, 100, 600, 300] -> y1=200, y2=600, x1=100, x2=300
    # cx = 200/1000 = 0.2, cy = 400/1000 = 0.4, w = 200/1000 = 0.2, h = 400/1000 = 0.4
    line = box_to_yolo_line([200, 100, 600, 300], 1000, 1000, 5, coord_format="row_col")
    parts = line.split()
    assert parts[0] == "5"
    assert pytest.approx(float(parts[1]), abs=1e-5) == 0.2
    assert pytest.approx(float(parts[2]), abs=1e-5) == 0.4
    assert pytest.approx(float(parts[3]), abs=1e-5) == 0.2
    assert pytest.approx(float(parts[4]), abs=1e-5) == 0.4


def test_box_to_yolo_line_clamping():
    # Box exceeds image boundaries
    line = box_to_yolo_line([-50, -50, 1200, 1200], 1000, 1000, 1, coord_format="xyxy")
    parts = [float(p) for p in line.split()]
    assert parts[0] == 1
    # Clamped within [0, 1]
    assert 0.0 <= parts[1] <= 1.0
    assert 0.0 <= parts[2] <= 1.0
    assert 0.0 <= parts[3] <= 1.0
    assert 0.0 <= parts[4] <= 1.0


def test_write_data_yaml(tmp_path):
    class_names = ["Debris", "Satellite"]
    content = write_data_yaml(tmp_path, class_names, dry_run=False)
    assert (tmp_path / "data.yaml").exists()
    assert "nc: 2" in content
    assert "- Debris" in content
    assert "- Satellite" in content
