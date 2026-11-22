"""
tests/test_demo_app.py
---
Unit tests for the interactive demonstration application (demo/app.py).
Verifies preset loading, pipeline initialization, image annotation rendering,
and HTTP endpoints (/api/health, /api/presets, /api/infer).
"""

import base64
import json
import threading
import time
import urllib.request
import cv2
import numpy as np
import pytest

from demo.app import (
    init_demo_pipeline,
    get_available_presets,
    render_detections_on_image,
    DemoRequestHandler,
    HTTPServer,
)


@pytest.fixture(scope="module")
def live_server():
    """Starts the demo server on a free loopback port in a background thread."""
    init_demo_pipeline()
    server = HTTPServer(("127.0.0.1", 0), DemoRequestHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    base_url = f"http://127.0.0.1:{port}"
    # Opener that bypasses proxy for local loopback testing
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    yield base_url, opener
    server.shutdown()
    server.server_close()


def test_demo_presets_loading():
    presets = get_available_presets()
    assert isinstance(presets, list)
    if len(presets) > 0:
        assert "name" in presets[0]
        assert "path" in presets[0]


def test_render_detections_on_image():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    dummy_dets = [
        {"box": [10, 10, 50, 50], "class_name": "Debris", "confidence": 0.95},
        {"box": [60, 60, 120, 120], "class_name": "Sentinel-6", "confidence": 0.88},
    ]
    annotated = render_detections_on_image(img, dummy_dets)
    assert annotated.shape == (200, 200, 3)
    # Check that drawing actually modified some pixels
    assert np.any(annotated > 0)


def test_api_health_endpoint(live_server):
    base_url, opener = live_server
    url = f"{base_url}/api/health"
    req = opener.open(url, timeout=5)
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert data["status"] == "healthy"
    assert "space-debris-demo" in data["service"]


def test_api_presets_endpoint(live_server):
    base_url, opener = live_server
    url = f"{base_url}/api/presets"
    req = opener.open(url, timeout=5)
    assert req.status == 200
    presets = json.loads(req.read().decode("utf-8"))
    assert isinstance(presets, list)


def test_api_infer_endpoint(live_server):
    base_url, opener = live_server
    # Create small test image
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    # Draw simple satellite-like shape
    cv2.rectangle(img, (80, 80), (240, 240), (200, 200, 200), -1)
    _, enc = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(enc).decode("utf-8")

    payload = json.dumps({
        "image_base64": b64,
        "model_type": "int8",
        "conf": 0.20,
        "iou": 0.45,
        "imgsz": 640
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/infer",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    resp = opener.open(req, timeout=10)
    assert resp.status == 200
    res_data = json.loads(resp.read().decode("utf-8"))

    assert "latency_ms" in res_data
    assert "model_footprint_mb" in res_data
    assert "detections" in res_data
    assert "astrodynamics" in res_data
    assert "annotated_image" in res_data
    assert res_data["model_footprint_mb"] == 3.59

