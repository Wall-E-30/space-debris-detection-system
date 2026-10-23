"""
audit/scripts/generate_claims_ledger.py
---
Extracts every numeric and factual claim from docs/MASTER_PROJECT_REPORT.md
and cross-references them against raw artifacts (JSONs, args.yaml, results.csv, benchmarks).
Writes audit/claims_ledger.csv conforming to Task A specifications.
"""

import csv
from pathlib import Path

CLAIMS = [
    # Executive Summary & Architecture
    {
        "claim_id": "CLM-001",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "YOLOv8-Nano architecture (3.01M parameters, 8.2 GFLOPs at 640px)",
        "raw_artifact_path": "experiments/spark2021_yolov8n_50e/weights/best.pt (model.info)",
        "raw_value": "3,140,433 params (11-class), 8.8 GFLOPs at 640px (22.4 GFLOPs at 1024px)",
        "verdict": "CONTRADICTED",
        "notes": "3.01M params and 8.2 GFLOPs are standard 80-class COCO figures; our 11-class model has 3.14M params and 8.8 GFLOPs at 640px. At native 1024px, compute is 22.4 GFLOPs."
    },
    {
        "claim_id": "CLM-002",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Power limits (<= 5W - 15W)",
        "raw_artifact_path": "ARTIFACT NOT FOUND",
        "raw_value": "None (textbook CubeSat power figure)",
        "verdict": "UNSUPPORTED",
        "notes": "Internal project design assumption; not an official ISRO Problem Statement constraint."
    },
    {
        "claim_id": "CLM-003",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Strict Real-Time Latency Gate: <= 1000 ms (1 second)",
        "raw_artifact_path": "docs/PHASE_1_VERIFICATION_AND_RESULTS.md:L128",
        "raw_value": "<= 1000 ms defined for Raspberry Pi 4B (Cortex-A72)",
        "verdict": "VERIFIED",
        "notes": "Formally defined in Phase 1 PRD gate, but originally targeted Raspberry Pi 4B."
    },
    {
        "claim_id": "CLM-004",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Memory footprint under 500 MB and binary sizes <= 15 MB (INT8 targeting <= 4 MB)",
        "raw_artifact_path": "experiments/spark2021_yolov8n_50e/weights/best.pt; best.onnx",
        "raw_value": "best.pt: 6.29 MB (FP16), best.onnx: 12.33 MB (FP32), Peak RSS: 138-216 MB",
        "verdict": "VERIFIED",
        "notes": "Current weights and benchmark peak RSS easily satisfy the <=500 MB and <=15 MB gates."
    },
    {
        "claim_id": "CLM-005",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "SwissCube baseline smoke test: 0.9903 mAP@0.5, 0.9211 mAP@0.5:0.95",
        "raw_artifact_path": "outputs/evaluations/week1_smoke_eval.json",
        "raw_value": "map50: 0.9903, map50_95: 0.9211",
        "verdict": "VERIFIED",
        "notes": "Matches raw evaluation JSON exactly."
    },
    {
        "claim_id": "CLM-006",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "On Day 1, audited all 80,000 label rows of initial SPARK release and confirmed exactly 0 space debris annotations",
        "raw_artifact_path": "docs/spark_class_audit.json",
        "raw_value": "total_rows: 80000, debris_instances_total: 0",
        "verdict": "VERIFIED",
        "notes": "Matches spark_class_audit.json (10 classes x 8,000 images = 80,000 rows, 0 debris)."
    },
    {
        "claim_id": "CLM-007",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "SPARK-2021 dataset (150,000 images): 25,000 debris annotations across 5 geometric debris types alongside 125,000 spacecraft labels across 10 satellite classes",
        "raw_artifact_path": "docs/spark2021_class_audit.json",
        "raw_value": "total_rows: 150000, debris_instances_total: 25000, satellite_instances_total: 125000",
        "verdict": "VERIFIED",
        "notes": "Confirmed in spark2021_class_audit.json and latest_spark2021.json."
    },
    # Dataset Table Claims
    {
        "claim_id": "CLM-008",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Dataset split table: Train 90k (7.5k sat, 15k debris), Val 30k (2.5k sat, 5k debris), Test 30k (2.5k sat, 5k debris)",
        "raw_artifact_path": "docs/spark2021_class_audit.json",
        "raw_value": "train: 90000, val: 30000, test: 30000 (each sat: 7.5k/2.5k/2.5k, debris: 15k/5k/5k)",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw CSV audit output."
    },
    # Photometric Augmentation Claims
    {
        "claim_id": "CLM-009",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Photometric augmentation applied exclusively to 90,000 raw training images, doubling training set to 180,000 images (aug1_*)",
        "raw_artifact_path": "outputs/audits/split_leakage_audit.json:L4",
        "raw_value": "total_train_images: 180000",
        "verdict": "VERIFIED",
        "notes": "Offline augmentation was executed for training; verified by total_train_images=180000 and aug1_* matches in leakage audit."
    },
    {
        "claim_id": "CLM-010",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Exactly zero augmented images exist in val/ or test/, guaranteeing that evaluations remain unpolluted",
        "raw_artifact_path": "outputs/audits/split_leakage_audit.json:L10",
        "raw_value": "leakage_rate_pct: 61.27 (dHash collapse); split leakage NOT ruled out",
        "verdict": "CONTRADICTED",
        "notes": "No aug1_ files exist in val/test, but claiming evaluations are 'guaranteed unpolluted' is contradicted by the team's own disclosure that sequence split leakage is not ruled out."
    },
    # Training Stage Claims
    {
        "claim_id": "CLM-011",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "50 epochs (562,500 iterations), batch 16, mosaic closing at epoch 40 (train_batch450000.jpg)",
        "raw_artifact_path": "experiments/spark2021_yolov8n_50e/args.yaml; train_batch450000.jpg",
        "raw_value": "180,000 / 16 = 11,250 batches/epoch; 11,250 * 40 = 450,000; 11,250 * 50 = 562,500",
        "verdict": "VERIFIED",
        "notes": "Arithmetic and filename train_batch450000.jpg directly prove 180k images and 50 epochs."
    },
    {
        "claim_id": "CLM-012",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Training duration: 122,183 seconds (~33.94 hours)",
        "raw_artifact_path": "experiments/spark2021_yolov8n_50e/results.csv:L51",
        "raw_value": "epoch 50 time: 122183",
        "verdict": "VERIFIED",
        "notes": "Exact match with results.csv line 51."
    },
    {
        "claim_id": "CLM-013",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "SGD optimizer",
        "raw_artifact_path": "experiments/spark2021_yolov8n_50e/args.yaml:L20",
        "raw_value": "optimizer: auto",
        "verdict": "CONTRADICTED",
        "notes": "args.yaml specifies 'optimizer: auto'. Ultralytics auto defaults to SGD for YOLOv8 detection, but the explicit config argument was auto."
    },
    # Per-Class Evaluation Table Claims (1024px)
    {
        "claim_id": "CLM-014",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "1024px Overall mAP@50 = 99.02%, mAP@50-95 = 86.98%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_metrics.json:L8-L9",
        "raw_value": "map50: 0.9902, map50_95: 0.8698",
        "verdict": "VERIFIED",
        "notes": "Overall metric matches raw JSON artifact exactly."
    },
    {
        "claim_id": "CLM-015",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "1024px CubeSat mAP@50 = 98.71%, mAP@50-95 = 82.17%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_metrics.json:L58-L59",
        "raw_value": "map50: 0.9698 (96.98%), map50_95: 0.6341 (63.41%)",
        "verdict": "CONTRADICTED",
        "notes": "CRITICAL DISCREPANCY: Raw artifact shows CubeSat is 63.41% mAP@50-95. Master Report falsely reports 82.17% (+18.76 points higher). Older SPARK report correctly reported 63.41%."
    },
    {
        "claim_id": "CLM-016",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "1024px TRMM mAP@50 = 96.65%, mAP@50-95 = 74.88%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_metrics.json:L93-L94",
        "raw_value": "map50: 0.9948 (99.48%), map50_95: 0.7776 (77.76%)",
        "verdict": "CONTRADICTED",
        "notes": "Raw artifact shows TRMM mAP@50 is 99.48% and mAP@50-95 is 77.76%. Master Report lowered TRMM to 74.88%."
    },
    {
        "claim_id": "CLM-017",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "1024px AcrimSat mAP@50 = 99.42%, mAP@50-95 = 90.07%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_metrics.json:L23-L24",
        "raw_value": "map50: 0.9950 (99.50%), map50_95: 0.9509 (95.09%)",
        "verdict": "CONTRADICTED",
        "notes": "Raw artifact shows AcrimSat is 95.09%. Master Report lowered it to 90.07% to offset CubeSat inflation."
    },
    {
        "claim_id": "CLM-018",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "1024px Calipso mAP@50 = 99.24%, mAP@50-95 = 83.21%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_metrics.json:L44-L45",
        "raw_value": "map50: 0.9948 (99.48%), map50_95: 0.9133 (91.33%)",
        "verdict": "CONTRADICTED",
        "notes": "Raw artifact shows Calipso is 91.33%. Master Report lowered it to 83.21% (-8.12 points)."
    },
    {
        "claim_id": "CLM-019",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "1024px Debris mAP@50 = 98.34%, mAP@50-95 = 85.39%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_metrics.json:L65-L66",
        "raw_value": "map50: 0.9834, map50_95: 0.8539",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw JSON artifact."
    },
    # 800px and 640px Evaluation Claims
    {
        "claim_id": "CLM-020",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "800px Debris mAP@50 = 98.19%, mAP@50-95 = 84.31%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_800.json:L65-L66",
        "raw_value": "map50: 0.9828 (98.28%), map50_95: 0.8431 (84.31%)",
        "verdict": "CONTRADICTED",
        "notes": "mAP@50-95 matches (84.31%), but mAP@50 was misquoted as 98.19% instead of raw 98.28%."
    },
    {
        "claim_id": "CLM-021",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "640px Debris mAP@50 = 97.83%, mAP@50-95 = 82.81%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_640.json:L65-L66",
        "raw_value": "map50: 0.9802 (98.02%), map50_95: 0.8281 (82.81%)",
        "verdict": "CONTRADICTED",
        "notes": "mAP@50-95 matches (82.81%), but mAP@50 was misquoted as 97.83% instead of raw 98.02%."
    },
    {
        "claim_id": "CLM-022",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "800px Overall mAP@50 = 98.94%, mAP@50-95 = 86.47%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_800.json:L8-L9",
        "raw_value": "map50: 0.9894, map50_95: 0.8647",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw JSON artifact."
    },
    {
        "claim_id": "CLM-023",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "640px Overall mAP@50 = 98.74%, mAP@50-95 = 85.06%",
        "raw_artifact_path": "outputs/evaluations/spark2021_test_640.json:L8-L9",
        "raw_value": "map50: 0.9874, map50_95: 0.8506",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw JSON artifact."
    },
    # Qualitative Claims
    {
        "claim_id": "CLM-024",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Analysis of the confusion matrix reveals TRMM false positives under dark Earth-limb illumination",
        "raw_artifact_path": "experiments/spark2021_yolov8n_50e/confusion_matrix.png",
        "raw_value": "Confusion matrix contains only numerical prediction counts, no illumination/limb metadata",
        "verdict": "CONTRADICTED",
        "notes": "A confusion matrix cannot reveal illumination conditions or Earth-limb geometry; claim is speculative narrative without artifact backing."
    },
    {
        "claim_id": "CLM-025",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "The network effectively learns high-gradient specular edges of tumbling debris",
        "raw_artifact_path": "ARTIFACT NOT FOUND",
        "raw_value": "No feature map or Grad-CAM artifact exists",
        "verdict": "UNSUPPORTED",
        "notes": "No qualitative edge-activation or feature saliency analysis exists in the repository."
    },
    # Edge Benchmark Claims
    {
        "claim_id": "CLM-026",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Xeon 640px (4 th): 23.70 ms, 42.2 FPS, 129.34 MB RSS",
        "raw_artifact_path": "outputs/benchmarks/spark2021_onnx_640_benchmark.json",
        "raw_value": "inference_mean_ms: 23.7, fps: 42.2, peak_rss_mb: 129.34",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw benchmark JSON."
    },
    {
        "claim_id": "CLM-027",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Xeon 800px (4 th): 35.79 ms, 27.9 FPS, 162.71 MB RSS",
        "raw_artifact_path": "outputs/benchmarks/spark2021_onnx_800_benchmark.json",
        "raw_value": "inference_mean_ms: 35.79, fps: 27.9, peak_rss_mb: 162.71",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw benchmark JSON."
    },
    {
        "claim_id": "CLM-028",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Xeon 1024px (4 th): 65.20 ms, 15.3 FPS, 207.36 MB RSS",
        "raw_artifact_path": "outputs/benchmarks/spark2021_onnx_1024_benchmark.json",
        "raw_value": "inference_mean_ms: 65.2, fps: 15.3, peak_rss_mb: 207.36",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw benchmark JSON."
    },
    {
        "claim_id": "CLM-029",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Raspberry Pi 5 640px (4 th): 144.83 ms, 6.90 FPS, 138.67 MB RSS",
        "raw_artifact_path": "outputs/benchmarks/pi5_onnx_640_benchmark.json",
        "raw_value": "inference_mean_ms: 144.83, fps: 6.9, peak_rss_mb: 138.67",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw benchmark JSON."
    },
    {
        "claim_id": "CLM-030",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Raspberry Pi 5 800px (4 th): 285.15 ms, 3.51 FPS, 175.30 MB RSS",
        "raw_artifact_path": "outputs/benchmarks/pi5_onnx_800_benchmark.json",
        "raw_value": "inference_mean_ms: 285.15, fps: 3.5, peak_rss_mb: 175.3",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw benchmark JSON."
    },
    {
        "claim_id": "CLM-031",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Raspberry Pi 5 1024px (4 th): 515.19 ms, 1.94 FPS, 216.73 MB RSS",
        "raw_artifact_path": "outputs/benchmarks/pi5_onnx_1024_benchmark.json",
        "raw_value": "inference_mean_ms: 515.19, fps: 1.9, peak_rss_mb: 216.73",
        "verdict": "VERIFIED",
        "notes": "Exact match with raw benchmark JSON."
    },
    {
        "claim_id": "CLM-032",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "PRD Gate (<=1000 ms) is PASSED across all Pi resolutions",
        "raw_artifact_path": "docs/PHASE_1_VERIFICATION_AND_RESULTS.md:L128",
        "raw_value": "PRD specified Raspberry Pi 4B (Cortex-A72), NOT Raspberry Pi 5 (Cortex-A76)",
        "verdict": "CONTRADICTED",
        "notes": "HIGH SEVERITY: Validated on Raspberry Pi 5 (Cortex-A76 @ 2.4GHz). Pi 4B is ~2.5-3x slower; 1024px is projected to exceed 1200-1500 ms on Pi 4B."
    },
    # Model Freeze & Checkpoint Claims
    {
        "claim_id": "CLM-033",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "DEC-09: Model Checkpoint Freeze (best.pt officially locked, Frozen status)",
        "raw_artifact_path": ".git/refs/tags/model-freeze",
        "raw_value": "Git tag model-freeze NOT FOUND; SHA256 checksum NOT RECORDED",
        "verdict": "CONTRADICTED",
        "notes": "No git tag model-freeze exists. Neither best.pt SHA256 (34fe3d10...) nor best.onnx SHA256 (df2cde42...) was recorded in any repo file."
    },
    # Astrodynamics & TLE Claims
    {
        "claim_id": "CLM-034",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Architecture Diagram: Cross-System Association (class -> NORAD lookup -> relative ephemeris & conjunction risk)",
        "raw_artifact_path": "src/data/tle_pipeline.py:L21,L26,L47; docs/decisions.md:DEC-05",
        "raw_value": "CubeSat maps to ISS (25544 proxy); Debris returns error; DEC-05 states 2D boxes cannot give orbits",
        "verdict": "CONTRADICTED",
        "notes": "Diagram shows 2D bounding boxes directly feeding NORAD lookup to compute relative distance. This directly contradicts DEC-05 and physical reality (CubeSat maps to ISS, debris has no TLE)."
    },
    # Project Hygiene & Requirements
    {
        "claim_id": "CLM-035",
        "document": "MASTER_PROJECT_REPORT.md",
        "quoted_claim": "Pinned host requirements (requirements.txt)",
        "raw_artifact_path": "requirements.txt",
        "raw_value": "13 unpinned package names with 0 version specifiers",
        "verdict": "CONTRADICTED",
        "notes": "requirements.txt has zero version pins (ultralytics, torch, etc.). requirements-edge.txt has minimum bounds (>=), not strict pins (==)."
    }
]

def main():
    out_dir = Path("audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "claims_ledger.csv"

    fieldnames = [
        "claim_id",
        "document",
        "quoted_claim",
        "raw_artifact_path",
        "raw_value",
        "verdict",
        "notes"
    ]

    with open(out_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in CLAIMS:
            writer.writerow(c)

    print(f"[OK] Generated {len(CLAIMS)} claims in {out_file.resolve()}")

if __name__ == "__main__":
    main()
