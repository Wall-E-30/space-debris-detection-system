# Space Debris Detection System — Final Output Metrics & Benchmarks
**ISRO Problem Statement #05** | **Phase 1 & Audit Verified Outputs**

---

### 1. Verified 1024px SPARK-2021 Per-Class Accuracy

* **Split:** Held-Out Test Set (30,000 images)
* **Model Checkpoint:** `spark2021_yolov8n_50e/weights/best.pt`
* **Input Resolution:** $1024 \times 1024$ px
* **Evidence Source:** `outputs/evaluations/spark2021_test_metrics.json`

| Class ID | Class Name | Category | Precision ($P$) | Recall ($R$) | mAP@50 | mAP@50-95 |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 0 | `AcrimSat` | Spacecraft | 0.9953 | 1.0000 | 0.9950 | 0.9509 |
| 1 | `Aquarius` | Spacecraft | 0.9940 | 0.9900 | 0.9948 | 0.9297 |
| 2 | `Aura` | Spacecraft | 0.9960 | 0.9961 | 0.9950 | 0.9380 |
| 3 | `Calipso` | Spacecraft | 0.9758 | 0.9920 | 0.9948 | 0.9133 |
| 4 | `Cloudsat` | Spacecraft | 0.9981 | 0.9460 | 0.9934 | 0.8625 |
| 5 | `CubeSat` | Spacecraft | 0.9622 | 0.9368 | 0.9698 | 0.6341 |
| 6 | `Debris` | Synthetic Debris (5 CAD Models) | 0.9775 | 0.9734 | 0.9834 | 0.8539 |
| 7 | `Jason` | Spacecraft | 0.9941 | 0.9988 | 0.9950 | 0.9173 |
| 8 | `Sentinel-6` | Spacecraft | 0.9960 | 0.9920 | 0.9950 | 0.8993 |
| 9 | `Terra` | Spacecraft | 0.9846 | 0.9452 | 0.9812 | 0.8916 |
| 10 | `TRMM` | Spacecraft | 0.9924 | 0.9951 | 0.9948 | 0.7776 |
| **--** | **OVERALL (Mean)** | **All 11 Classes** | **0.9878** | **0.9787** | **0.9902** | **0.8698** |

---

### 2. Hardware Deployability Metrics

* **Architecture:** YOLOv8-Nano
* **Parameter Count:** 3,140,433
* **PyTorch FP16 Weights Size (`best.pt`):** 6.29 MB (6,596,906 bytes)
* **ONNX FP32 Model Size (`best.onnx`):** 12.33 MiB / 12.93 MB (12,933,088 bytes)
* **`best.pt` SHA-256:** `34fe3d10b045849fef33e29f10028e878fbd7745a71e2839aa02875e47c8795d`
* **`best.onnx` SHA-256:** `df2cde420af209acddcc9aceb5e9d9ca909d75cdca16474bc7f71aca60260912`
* **Evidence Source:** `audit/AUDIT_REPORT.md` (AUD-07), `docs/decisions.md` (DEC-09)

| Input Resolution | Parameters | Compute Complexity (GFLOPs) | On-Disk Model Size |
|---|:---:|:---:|:---:|
| **640 x 640 px** | 3.14M | 8.8 GFLOPs | 12.93 MB (ONNX) / 6.29 MB (FP16) |
| **800 x 800 px** | 3.14M | 13.7 GFLOPs | 12.93 MB (ONNX) / 6.29 MB (FP16) |
| **1024 x 1024 px** | 3.14M | 22.4 GFLOPs | 12.93 MB (ONNX) / 6.29 MB (FP16) |

---

### 3. Raspberry Pi 5 Latency & Memory Benchmarks

* **Hardware Platform:** Raspberry Pi 5 (ARM Cortex-A76, 4 cores, `aarch64`)
* **Execution Engine:** ONNX Runtime (CPU Execution Provider)
* **Thread Allocation:** 4 Threads
* **Precision:** FP32
* **Benchmark Protocol:** 10 discarded warmup passes, 100 timed iterations
* **Evidence Sources:** `outputs/benchmarks/pi5_onnx_640_benchmark.json`, `pi5_onnx_800_benchmark.json`, `pi5_onnx_1024_benchmark.json`

| Input Resolution | Mean Latency | Median Latency | p95 Latency | Latency Std | Throughput (FPS) | Peak RAM (RSS) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **640 x 640 px** | 144.83 ms | 143.77 ms | 150.42 ms | 2.80 ms | 6.90 FPS | 138.67 MB |
| **800 x 800 px** | 285.15 ms | 284.24 ms | 311.33 ms | 15.29 ms | 3.51 FPS | 175.30 MB |
| **1024 x 1024 px** | 515.19 ms | 515.45 ms | 540.82 ms | 14.33 ms | 1.94 FPS | 216.73 MB |

---

### 4. PyTorch vs. ONNX Parity Check Results

* **Script:** `audit/scripts/check_onnx_pytorch_parity.py`
* **Evidence Source:** `audit/onnx_parity_check.json`

| Metric Field | Measured Value | Status |
|---|:---:|:---:|
| **Sample Images Tested** | 50 | Verified |
| **Total PyTorch Detections** | 46 | Reference |
| **Total ONNX Detections** | 46 | Reference |
| **Matched Detections ($\text{IoU} \ge 0.50$)** | 46 / 46 | 100.0% Matched |
| **Mean IoU (Matched)** | 1.0000 | Numerical Parity |
| **Minimum IoU (Matched)** | 1.0000 | Numerical Parity |
| **Class Agreement Percentage** | 100.0% | Zero Discrepancy |
| **Mean Confidence Score Delta** | $1.0 \times 10^{-6}$ | $< 0.0001$ Threshold |
| **Maximum Confidence Score Delta** | $4.0 \times 10^{-6}$ | $< 0.0001$ Threshold |

---

### 5. Dataset Split Counts (SPARK-2021 Ground Truth)

* **Evidence Source:** `outputs/audits/latest_spark2021.json`

#### Split Summary
| Split | Total Image Files | Spacecraft Instances | Debris Instances | Total Rows | % Debris |
|---|:---:|:---:|:---:|:---:|:---:|
| **Train (`train_labels.csv`)** | 90,000 | 75,000 | 15,000 | 90,000 | 16.67% |
| **Validation (`validate_labels.csv`)** | 30,000 | 25,000 | 5,000 | 30,000 | 16.67% |
| **Test (`test_labels.csv`)** | 30,000 | 25,000 | 5,000 | 30,000 | 16.67% |
| **Total Raw Ground Truth** | **150,000** | **125,000** | **25,000** | **150,000** | **16.67%** |
| **Total Augmented Training Set** | **180,000** | **150,000** | **30,000** | **180,000** | **16.67%** |

#### Class Breakdown (Raw 150,000 Instance Rows)
| Class ID | Class Name | Train Rows | Val Rows | Test Rows | Total Verified Rows | Object Type |
|:---:|---|:---:|:---:|:---:|:---:|---|
| 0 | `AcrimSat` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |
| 1 | `Aquarius` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |
| 2 | `Aura` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |
| 3 | `Calipso` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |
| 4 | `Cloudsat` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |
| 5 | `CubeSat` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |
| 6 | `Debris` | 15,000 | 5,000 | 5,000 | 25,000 | Synthetic Debris (5 CAD Models) |
| 7 | `Jason` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |
| 8 | `Sentinel-6` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |
| 9 | `Terra` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |
| 10 | `TRMM` | 7,500 | 2,500 | 2,500 | 12,500 | Spacecraft |

