## TFLite vs ONNX Runtime

**TFLite** and **ONNX Runtime** are both just inference runtimes — you export your trained YOLOv8n weights into their format and run them locally. **Edge Impulse** is a different category entirely: a full MLOps platform (train + optimize + deploy) that's built primarily around its own EON compiler and is most commonly used for microcontroller-class TinyML models (FOMO, small CNNs) rather than full YOLO detection heads.

| | TFLite | ONNX Runtime | Edge Impulse |
|---|---|---|---|
| **What it is** | Google's mobile/embedded inference runtime | Cross-platform inference runtime | End-to-end MLOps platform (train + deploy) |
| **YOLOv8 export support** | Native `.tflite` export via Ultralytics | Native `.onnx` export via Ultralytics | No native YOLOv8 detection export — requires "Bring Your Own Model" workaround |
| **RPi 4 CPU speed** | Documented as underwhelming without correctly configured acceleration — one developer saw inference speed improve from 1500ms to 750ms per frame only after manually enabling multi-threading, since XNNPACK delegation didn't kick in automatically | Comparable or better than baseline TFLite on ARM CPU in published cross-runtime benchmarks | Not designed for this workload — best-in-class for microcontroller-scale models, not full detector backbones |
| **Where Ultralytics is trending** | Ultralytics' official Raspberry Pi guide now recommends exporting to NCNN for the fastest inference on ARM, benchmarking ten formats including TFLite, ONNX, and NCNN head-to-head | Included as one of the standard benchmarked export formats, but not the top performer on ARM | Not part of Ultralytics' benchmarked export list at all |
| **Ecosystem fit for your project** | Good — well-documented, works offline, no cloud dependency | Good — cross-platform, useful if you also target non-RPi hardware later | Poor fit specifically for YOLOv8n detection on RPi4; better suited if you later shrink to a TinyML classifier on a microcontroller |
| **License** | Apache 2.0 | MIT | Free tier + paid tiers for larger projects/teams |

## The finding that changes the decision

Independent of the original three-way framing, current data points to a fourth option that outperforms both: NCNN is a high-performance inference framework specifically optimized for mobile/ARM platforms, with no third-party dependencies, and delivers the best inference performance on Raspberry Pi devices among Ultralytics' supported export formats. In one documented case, a model that took ~400ms per frame in ONNX-adjacent formats on a Pi dropped to ~80ms after exporting to NCNN — a 5x speedup, on the same hardware, same model.

## My recommendation

**Export to NCNN as primary, keep ONNX as the portable fallback, drop Edge Impulse for this specific deployment.**

- **NCNN** — this is what should power your actual Raspberry Pi 4 demo. It's a direct `model.export(format="ncnn")` call from Ultralytics, so it doesn't cost you extra pipeline complexity, and the speed headroom matters a lot for a real-time debris-tracking demo where every extra frame per second makes the panel demo look more "real."
- **ONNX** — keep this as your interchange format in `docs/decisions.md`. If you ever target a different edge board (Jetson Nano via TensorRT, an Intel NUC via OpenVINO), ONNX is the common entry point for all of those toolchains, so it's worth exporting and benchmarking even if it's not your primary deployment target.
- **Edge Impulse** — don't use it for the YOLOv8n detector itself. It's the wrong tool for this model class. If your project later adds a lightweight "is there a bright anomaly in this frame" pre-filter to run on an even smaller microcontroller before triggering the full detector, that's where Edge Impulse would genuinely shine — flag it as a stretch-goal idea, not a Week 1 decision.
- **Skip TFLite** unless you hit an NCNN compatibility issue — it works, but the XNNPACK delegate configuration is finicky in practice and the speed ceiling is lower than NCNN's on this exact hardware.