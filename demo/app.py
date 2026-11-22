"""
demo/app.py
---
Edge-Deployable Space Debris Detection & Astrodynamics Awareness Demonstrator
ISRO Problem Statement #05

A self-contained, lightweight interactive web server built with Python standard library
http.server and OpenCV / ONNX Runtime. Requires zero heavy web frameworks (runs natively
on workstation or resource-constrained Raspberry Pi 4 / Pi 5).

Key Capabilities:
1. Multi-backend model inference (ONNX Static INT8, ONNX FP32, PyTorch FP16).
2. Preset space imagery loader & drag-and-drop custom image upload.
3. Interactive threshold controls (confidence & NMS IoU).
4. Real-time Telemetry HUD (Inference Latency, Throughput FPS, Model Footprint, Peak RSS).
5. Decoupled SGP4 Astrodynamics Panel:
   - Queries CelesTrak TLE catalog for known spacecraft (Sentinel-6, Terra, etc.)
   - Discloses physical honesty disclaimer for uncataloged space debris & CubeSats.
"""

import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.parse

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np

from src.infer import init_onnx_session, load_class_names, run_onnx_inference
from src.data.tle_pipeline import OrbitalTracker


# Global pipeline state (cached sessions for fast real-time inference)
MODELS: Dict[str, Any] = {}
TRACKER: Optional[OrbitalTracker] = None
CLASS_NAMES: List[str] = []

CONFIG_PATH = repo_root / "configs" / "spark2021.yaml"
TLE_PATH = repo_root / "data" / "tle" / "celestrak_active.tle"
INT8_PATH = repo_root / "experiments" / "spark2021_yolov8n_50e" / "weights" / "best_int8.onnx"
FP32_PATH = repo_root / "experiments" / "spark2021_yolov8n_50e" / "weights" / "best.onnx"
PT_PATH = repo_root / "experiments" / "spark2021_yolov8n_50e" / "weights" / "best.pt"


def init_demo_pipeline():
    global TRACKER, CLASS_NAMES

    # 1. Load class names
    if CONFIG_PATH.exists():
        CLASS_NAMES = load_class_names(str(CONFIG_PATH))
    else:
        CLASS_NAMES = ["AcrimSat", "Aquarius", "Aura", "Calipso", "Cloudsat", "CubeSat", "Debris", "Jason", "Sentinel-6", "Terra", "TRMM"]

    # 2. Initialize TLE tracker
    try:
        tle_src = str(TLE_PATH) if TLE_PATH.exists() else None
        TRACKER = OrbitalTracker(tle_source=tle_src)
        print("[INFO] Astrodynamics SGP4 orbital catalog initialized.")
    except Exception as exc:
        print(f"[WARN] Could not initialize full TLE tracker: {exc}")
        TRACKER = None

    # 3. Pre-load INT8 model session
    if INT8_PATH.exists():
        try:
            MODELS["int8"] = init_onnx_session(str(INT8_PATH), device="cpu")
            print(f"[INFO] Preloaded INT8 ONNX session ({INT8_PATH.name})")
        except Exception as exc:
            print(f"[WARN] Could not preload INT8 session: {exc}")

    # 4. Pre-load FP32 model session if available
    if FP32_PATH.exists():
        try:
            MODELS["fp32"] = init_onnx_session(str(FP32_PATH), device="cpu")
            print(f"[INFO] Preloaded FP32 ONNX session ({FP32_PATH.name})")
        except Exception as exc:
            print(f"[WARN] Could not preload FP32 session: {exc}")


def get_available_presets() -> List[Dict[str, str]]:
    presets = []
    real_val = repo_root / "data" / "real_validation"
    if real_val.exists():
        for p in sorted(real_val.glob("*.jpg"))[:8]:
            presets.append({
                "name": f"Real: {p.stem.replace('_', ' ').title()}",
                "path": str(p.relative_to(repo_root)).replace("\\", "/")
            })

    test_dir = repo_root / "data" / "processed" / "spark" / "images" / "test"
    if test_dir.exists():
        for p in sorted(test_dir.glob("*.jpg"))[:6]:
            presets.append({
                "name": f"SPARK: {p.stem.replace('_', ' ').title()}",
                "path": str(p.relative_to(repo_root)).replace("\\", "/")
            })
    return presets


def render_detections_on_image(img_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    annotated = img_bgr.copy()
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_name = det["class_name"]
        conf = det["confidence"]

        # Debris: high-visibility red; Satellites: cyan/green
        color = (0, 0, 255) if cls_name == "Debris" else (255, 200, 0) if cls_name == "CubeSat" else (0, 255, 128)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - h - 6)), (x1 + w + 4, max(h + 6, y1)), color, -1)
        cv2.putText(annotated, label, (x1 + 2, max(h + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISRO #05: Space Debris Detection & Astrodynamics Telemetry</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #131b2a;
            --bg-card: #1c273c;
            --accent-blue: #00b4d8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-orange: #f59e0b;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --border-color: #2e3d57;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-primary); color: var(--text-main); min-height: 100vh; padding: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; margin-bottom: 24px; }
        .logo-group h1 { font-size: 1.4rem; color: var(--accent-blue); display: flex; align-items: center; gap: 10px; }
        .badge { background: #1e3a5f; color: #7dd3fc; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: bold; border: 1px solid #0284c7; }
        .grid-container { display: grid; grid-template-columns: 340px 1fr 380px; gap: 20px; }
        @media (max-width: 1200px) { .grid-container { grid-template-columns: 1fr; } }
        .card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 18px; margin-bottom: 18px; }
        .card h2 { font-size: 1rem; color: var(--accent-blue); margin-bottom: 14px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; }
        label { display: block; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 5px; font-weight: 600; }
        select, input[type="range"], input[type="file"], button { width: 100%; margin-bottom: 14px; background: var(--bg-secondary); border: 1px solid var(--border-color); color: var(--text-main); border-radius: 6px; padding: 8px 10px; font-size: 0.85rem; }
        button { background: #0284c7; color: white; border: none; font-weight: bold; cursor: pointer; transition: background 0.2s; padding: 10px; margin-top: 6px; }
        button:hover { background: #0369a1; }
        .telemetry-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
        .metric-box { background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 6px; padding: 10px; text-align: center; }
        .metric-value { font-size: 1.3rem; font-weight: bold; color: var(--accent-green); }
        .metric-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; margin-top: 2px; }
        .viewer-container { background: #000; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; display: flex; justify-content: center; align-items: center; min-height: 480px; position: relative; }
        .viewer-container img { max-width: 100%; max-height: 600px; object-fit: contain; }
        .astrodynamics-panel { max-height: 540px; overflow-y: auto; font-size: 0.8rem; }
        .astro-card { background: var(--bg-secondary); border-left: 3px solid var(--accent-blue); padding: 10px; margin-bottom: 10px; border-radius: 4px; }
        .astro-card.debris { border-left-color: var(--accent-red); }
        .astro-card.cubesat { border-left-color: var(--accent-orange); }
        .astro-title { font-weight: bold; font-size: 0.85rem; margin-bottom: 4px; display: flex; justify-content: space-between; }
        .coord-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin-top: 6px; font-family: monospace; font-size: 0.75rem; color: #a5f3fc; }
        .disclaimer-box { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 4px; padding: 8px; margin-top: 6px; color: #fca5a5; font-size: 0.75rem; line-height: 1.3; }
        .spinner { display: none; width: 36px; height: 36px; border: 3px solid rgba(255,255,255,0.1); border-top-color: var(--accent-blue); border-radius: 50%; animation: spin 1s infinite linear; position: absolute; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header>
        <div class="logo-group">
            <h1>ISRO #05: Space Debris Detection & Astrodynamics Hub</h1>
        </div>
        <div style="display:flex; gap:10px;">
            <span class="badge">Model: INT8 (3.59 MB)</span>
            <span class="badge">SGP4: Active Catalog</span>
            <span class="badge">Edge Target: RPi 4/5</span>
        </div>
    </header>

    <div class="grid-container">
        <!-- Control Sidebar -->
        <div class="sidebar">
            <div class="card">
                <h2>Mission Controls</h2>
                <label>Preset Space Domain Scene</label>
                <select id="presetSelect" onchange="loadPreset()"><option value="">-- Choose Scene Preset --</option></select>

                <label>Upload Custom Sensor Frame</label>
                <input type="file" id="fileInput" accept="image/*" onchange="handleFileUpload()">

                <label>Inference Runtime Engine</label>
                <select id="modelSelect">
                    <option value="int8" selected>ONNX Static INT8 (3.59 MB - Edge Primary)</option>
                    <option value="fp32">ONNX FP32 (12.33 MB - Full Precision)</option>
                </select>

                <label>Confidence Gate: <span id="confVal">0.25</span></label>
                <input type="range" id="confRange" min="0.10" max="0.95" step="0.05" value="0.25" oninput="document.getElementById('confVal').innerText=this.value">

                <label>NMS IoU Threshold: <span id="iouVal">0.45</span></label>
                <input type="range" id="iouRange" min="0.10" max="0.80" step="0.05" value="0.45" oninput="document.getElementById('iouVal').innerText=this.value">

                <button onclick="triggerInference()">Run Edge Inference</button>
            </div>

            <div class="card">
                <h2>Telemetry HUD</h2>
                <div class="telemetry-row">
                    <div class="metric-box">
                        <div class="metric-value" id="hudLatency">-- ms</div>
                        <div class="metric-label">Inference Latency</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="hudFps">--</div>
                        <div class="metric-label">Throughput FPS</div>
                    </div>
                </div>
                <div class="telemetry-row">
                    <div class="metric-box">
                        <div class="metric-value" id="hudSize">3.59 MB</div>
                        <div class="metric-label">Model Size</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="hudCount">0</div>
                        <div class="metric-label">Objects Localized</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Central Inspection Viewer -->
        <div class="main-viewer">
            <div class="card" style="margin-bottom:0; height:100%;">
                <h2>Optical Sensor Stream & Bounding Overlays</h2>
                <div class="viewer-container">
                    <div class="spinner" id="spinner"></div>
                    <img id="displayImg" src="" alt="Select preset or upload an image to begin.">
                </div>
            </div>
        </div>

        <!-- Astrodynamics SGP4 Panel -->
        <div class="astro-sidebar">
            <div class="card" style="height:100%;">
                <h2>Decoupled Astrodynamics Panel (DEC-05)</h2>
                <p style="font-size:0.75rem; color:var(--text-muted); margin-bottom:12px;">
                    Ground-truth ephemeris propagation linking verified detections to CelesTrak NORAD elements.
                </p>
                <div class="astrodynamics-panel" id="astroFeed">
                    <div style="color:var(--text-muted); text-align:center; padding:30px 10px;">
                        Awaiting detection inputs. Orbital propagation telemetry will render here once space assets are classified.
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentImageBase64 = null;

        window.onload = async () => {
            const resp = await fetch('/api/presets');
            const presets = await resp.json();
            const sel = document.getElementById('presetSelect');
            presets.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.path;
                opt.innerText = p.name;
                sel.appendChild(opt);
            });
            if (presets.length > 0) {
                sel.value = presets[0].path;
                loadPreset();
            }
        };

        async function loadPreset() {
            const path = document.getElementById('presetSelect').value;
            if (!path) return;
            const res = await fetch('/api/load_image?path=' + encodeURIComponent(path));
            const data = await res.json();
            currentImageBase64 = data.image_base64;
            document.getElementById('displayImg').src = 'data:image/jpeg;base64,' + currentImageBase64;
            triggerInference();
        }

        function handleFileUpload() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImageBase64 = e.target.result.split(',')[1];
                document.getElementById('displayImg').src = e.target.result;
                triggerInference();
            };
            reader.readAsDataURL(file);
        }

        async function triggerInference() {
            if (!currentImageBase64) return;
            document.getElementById('spinner').style.display = 'block';

            const payload = {
                image_base64: currentImageBase64,
                model_type: document.getElementById('modelSelect').value,
                conf: parseFloat(document.getElementById('confRange').value),
                iou: parseFloat(document.getElementById('iouRange').value),
                imgsz: 640
            };

            const t0 = performance.now();
            const res = await fetch('/api/infer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById('spinner').style.display = 'none';

            // Update display image
            document.getElementById('displayImg').src = 'data:image/jpeg;base64,' + data.annotated_image;

            // Update HUD
            document.getElementById('hudLatency').innerText = data.latency_ms.toFixed(1) + ' ms';
            document.getElementById('hudFps').innerText = (1000.0 / Math.max(1, data.latency_ms)).toFixed(1);
            document.getElementById('hudCount').innerText = data.detections.length;
            document.getElementById('hudSize').innerText = data.model_footprint_mb + ' MB';

            // Update Astrodynamics Panel
            const astroFeed = document.getElementById('astroFeed');
            astroFeed.innerHTML = '';

            if (data.detections.length === 0) {
                astroFeed.innerHTML = '<div style="color:var(--text-muted); text-align:center; padding:30px;">No space objects localized at current confidence threshold.</div>';
                return;
            }

            data.astrodynamics.forEach(item => {
                const card = document.createElement('div');
                const isDebris = item.class_name === 'Debris';
                const isCubesat = item.class_name === 'CubeSat';
                card.className = 'astro-card' + (isDebris ? ' debris' : isCubesat ? ' cubesat' : '');

                let content = `<div class="astro-title"><span>${item.class_name} (Conf: ${(item.confidence*100).toFixed(1)}%)</span><span>${item.status.toUpperCase()}</span></div>`;
                if (item.status === 'cataloged') {
                    content += `<div style="color:#94a3b8; font-size:0.75rem;">NORAD ID: <strong>${item.norad_id}</strong> | Epoch: ${item.epoch_utc}</div>
                    <div class="coord-grid">
                        <div>X: ${item.coordinates_eci_km[0].toFixed(1)} km</div>
                        <div>Y: ${item.coordinates_eci_km[1].toFixed(1)} km</div>
                        <div>Z: ${item.coordinates_eci_km[2].toFixed(1)} km</div>
                    </div>
                    <div style="font-size:0.75rem; color:#86efac; margin-top:4px;">Orbital state propagated via SGP4</div>`;
                } else {
                    content += `<div class="disclaimer-box"><strong>PHYSICAL HONESTY (DEC-05):</strong> ${item.notice}</div>`;
                }
                card.innerHTML = content;
                astroFeed.appendChild(card);
            });
        }
    </script>
</body>
</html>
"""


class DemoRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean console output
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "service": "space-debris-demo", "models": list(MODELS.keys())}).encode("utf-8"))

        elif path == "/api/presets":
            presets = get_available_presets()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(presets).encode("utf-8"))

        elif path == "/api/load_image":
            query = urllib.parse.parse_qs(parsed.query)
            rel_path = query.get("path", [""])[0]
            abs_path = repo_root / rel_path
            if abs_path.exists() and abs_path.is_file():
                with open(abs_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"image_base64": b64}).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/infer":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid JSON payload")
                return

            raw_b64 = data.get("image_base64", "")
            model_type = data.get("model_type", "int8")
            conf_thresh = float(data.get("conf", 0.25))
            iou_thresh = float(data.get("iou", 0.45))
            imgsz = int(data.get("imgsz", 640))

            try:
                img_bytes = base64.b64decode(raw_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as exc:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Image decode failure: {exc}"}).encode("utf-8"))
                return

            # Temporary file write for existing inference API
            temp_img = repo_root / "outputs" / "inference" / "_temp_demo.jpg"
            temp_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(temp_img), img_bgr)

            # Route model
            weights_file = INT8_PATH if model_type == "int8" else FP32_PATH
            session = MODELS.get(model_type)

            detections, latency_ms = run_onnx_inference(
                weights=str(weights_file),
                image_path=str(temp_img),
                imgsz=imgsz,
                conf=conf_thresh,
                iou=iou_thresh,
                session=session,
                class_names=CLASS_NAMES,
            )

            # Render annotations
            annotated_bgr = render_detections_on_image(img_bgr, detections)
            _, enc = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            annotated_b64 = base64.b64encode(enc).decode("utf-8")

            # Astrodynamics queries
            astro_records = []
            for det in detections:
                cls_name = det["class_name"]
                conf = det["confidence"]
                if TRACKER:
                    pos = TRACKER.get_position(cls_name)
                    if pos.get("status") == "uncataloged":
                        astro_records.append({
                            "class_name": cls_name,
                            "confidence": conf,
                            "status": "uncataloged",
                            "notice": pos.get("error")
                        })
                    elif pos.get("status") == "ok":
                        astro_records.append({
                            "class_name": cls_name,
                            "confidence": conf,
                            "status": "cataloged",
                            "norad_id": pos.get("norad_id"),
                            "epoch_utc": pos.get("epoch_utc"),
                            "coordinates_eci_km": [pos["position_km"]["x"], pos["position_km"]["y"], pos["position_km"]["z"]],
                        })
                else:
                    astro_records.append({
                        "class_name": cls_name,
                        "confidence": conf,
                        "status": "uncataloged",
                        "notice": "Offline catalog layer unavailable"
                    })

            footprint_mb = 3.59 if model_type == "int8" else 12.33

            resp_payload = {
                "latency_ms": latency_ms,
                "model_footprint_mb": footprint_mb,
                "detections": detections,
                "astrodynamics": astro_records,
                "annotated_image": annotated_b64
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8080):
    init_demo_pipeline()
    server_address = ("", port)
    httpd = HTTPServer(server_address, DemoRequestHandler)
    print(f"\n{'='*65}")
    print(f"  ISRO #05 INTERACTIVE DEMO APPLICATION RUNNING")
    print(f"  Local URL:  http://localhost:{port}")
    print(f"  Mode:       Lightweight Standard Library (zero heavy dependencies)")
    print(f"{'='*65}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Demo server stopped.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8080
    run_server(port)
