# Independent Technical Audit Report
**Repository:** `space-debris-detection-system` (ISRO Problem Statement #05)  
**Auditor:** Independent Technical Auditor  
**Date:** October 23, 2026  
**Audit Scope:** Codebase, Documentation (`docs/`, `.planning/`), Training Logs, Model Weights, Evaluation JSONs, and Hardware Benchmarks  

---

## 1. Executive Trustworthiness Summary

1. The underlying deep learning engineering—50-epoch training on the Quadro GV100, 180,000 augmented training images, and physical benchmarks on Raspberry Pi 5—is technically authentic and verifiable from raw logs and checkpoints.
2. However, the untracked [docs/MASTER_PROJECT_REPORT.md](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/MASTER_PROJECT_REPORT.md) contains a **critical data manipulation**: the per-class performance table was artificially altered to inflate `CubeSat` mAP@50-95 from **63.41% to 82.17%** (+18.76 points), while reducing other classes to preserve the exact overall mean of **86.98%**.
3. The older report ([docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md)) is **scientifically honest** and matches the raw JSON artifacts to the fourth decimal place; the Master Report must be immediately corrected.
4. Qualitative explanations are frequently ungrounded: the claim that the confusion matrix proves "TRMM false positives under dark Earth-limb illumination" is completely fabricated storytelling, as confusion matrices contain no environmental metadata.
5. The astrodynamics layer ([src/data/tle_pipeline.py](file:///c:/Users/Sharanya%20Nagar/Desktop/ISRO_Hardware/space-debris-detection-system/src/data/tle_pipeline.py)) is conceptually disconnected from vision: `CubeSat` is hardcoded to NORAD 25544 (the International Space Station), debris propagation returns an error, and 2D bounding boxes cannot physically determine 3D orbits or ranges.
6. The PRD real-time gate ($\le 1000\text{ ms}$) was validated exclusively on a **Raspberry Pi 5 (Cortex-A76)**, but the PRD mandated a **Raspberry Pi 4 (Cortex-A72)**; on a Pi 4, native 1024px inference will likely fail the gate (>1200 ms).
7. Decision DEC-09 claims the model is "Frozen", but **no git tag `model-freeze` exists**, and neither `best.pt` nor `best.onnx` SHA-256 checksums were recorded in the repository.
8. The 150,000-image SPARK-2021 dataset resides exclusively on the departmental Quadro GV100 workstation (`D:\...`) and was not transferred to this development machine, preventing local re-evaluation of the 30k test split.
9. Documentation inconsistently renumbered Architectural Decisions (DEC-01, DEC-03, DEC-05, DEC-06) across Phase 1 and Phase 2, and `.planning/STATE.md` has been abandoned since October 19.
10. **Overall Verdict:** The project has achieved a viable edge model and genuine hardware benchmarks, but the reporting layer contains severe narrative embellishments and smoothed metrics that must be stripped before ISRO evaluation.

---

## 2. Table of Audit Findings

| Finding ID | Area | Quoted / Documented Claim | Verdict | Severity | Evidence (File + Key / Command) | Recommended Remediation |
|---|---|---|:---:|:---:|---|---|
| **AUD-01** | Evaluation Metrics | Master Report lists CubeSat 1024px mAP@50-95 as **82.17%** (and alters AcrimSat, Calipso, TRMM) | **CONTRADICTED** | **CRITICAL** | `outputs/evaluations/spark2021_test_metrics.json` lines 58-59: CubeSat `map50_95 = 0.6341`. `SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md` line 116 records `0.6341`. | Revert Master Report table to match raw JSON: CubeSat 63.41%, AcrimSat 95.09%, Calipso 91.33%, TRMM 77.76%. Apply patch `audit/proposed_patches/01_master_project_report.patch`. |
| **AUD-02** | Edge Hardware Gate | "PRD Gate ($\le 1000\text{ ms}$) is PASSED across all resolutions" | **CONTRADICTED** | **HIGH** | `docs/PHASE_1_VERIFICATION_AND_RESULTS.md` line 128: PRD gate specifies **Raspberry Pi 4 (Cortex-A72)**. Raw benchmarks in `outputs/benchmarks/pi5_onnx_*` were run on **Raspberry Pi 5 (Cortex-A76)**. | Explicitly disclose that validation was conducted on Raspberry Pi 5. State clearly that Pi 4 validation remains an open gate and is projected to fail at 1024px. |
| **AUD-03** | Model Freeze & Provenance | DEC-09: Model Checkpoint Freeze (best.pt officially locked, status "Frozen") | **CONTRADICTED** | **HIGH** | Shell command `git tag -l` returns empty. `git grep -i "34fe3d10"` returns empty. Checksum of `best.pt` is `34fe3d10b045849fef33e29f10028e878fbd7745a71e2839aa02875e47c8795d`. | Create git tag `model-freeze` pointing to commit `f7b3011`. Commit SHA-256 hashes of `best.pt` and `best.onnx` into `docs/decisions.md`. |
| **AUD-04** | Evaluation Methodology | Operating point (800px "sweet spot") chosen from test split Pareto curve | **CONTRADICTED** | **HIGH** | `outputs/evaluations/`: only `spark2021_test_800.json` and `test_640.json` exist; no `val_800.json` or `val_640.json` were generated. | Selection on test split violates evaluation hygiene. Disclose that multi-resolution evaluation was conducted post-freeze across splits, or generate multi-scale validation benchmarks. |
| **AUD-05** | Astrodynamics & Conjunction | Architecture diagram shows BBox -> Class -> NORAD lookup -> Relative Ephemeris & Conjunction Risk | **CONTRADICTED** | **HIGH** | `src/data/tle_pipeline.py`: line 21 maps CubeSat to ISS (25544 proxy); line 47 blocks Debris with error; `docs/decisions.md` DEC-05 notes 2D boxes cannot yield orbits. | Remove Conjunction Risk block from diagram. Propose dual-track monitoring diagram where vision provides bearing/classification and SGP4 provides catalog tracking for known satellites only. |
| **AUD-06** | Coordinate Conventions | SPARK bounding box row/col conversion: [R_min, C_min, R_max, C_max] == [y_min, x_min, y_max, x_max] | **CANNOT-CHECK** (Locally) | **MEDIUM** | Binary mask ground truth on `Cheops_image_00001` in SPARK 2019 proves coordinates are `(x_min, y_min, x_max, y_max)`. SPARK-2021 raw CSVs are missing locally on drive C:. | Visually inspect converted SPARK-2021 boxes on the Quadro GV100 workstation using `scripts/convert_spark2021_to_yolo.py --verify-samples`. |
| **AUD-07** | Compute FLOPs | YOLOv8-Nano has "8.2 GFLOPs" | **CONTRADICTED** | **MEDIUM** | Model profiling via `model.info(imgsz=...)`: 640px = **8.8 GFLOPs** (3.14M params); 800px = **13.7 GFLOPs**; native 1024px = **22.4 GFLOPs**. | Update all documents to label FLOPs with resolution: 8.8 GFLOPs (640px), 13.7 GFLOPs (800px), 22.4 GFLOPs (1024px). |
| **AUD-08** | Narrative Attribution | "Analysis of confusion matrix reveals TRMM false positives under dark Earth-limb illumination" | **CONTRADICTED** | **MEDIUM** | `experiments/spark2021_yolov8n_50e/confusion_matrix.png`: contains only an 11x11 matrix of counts. No lighting/illumination or Earth-limb metadata exists. | Remove speculative narrative from reports. Attribute TRMM degradation to high aspect ratio / elongated solar wings. |
| **AUD-09** | Taxonomy Terminology | Claims "~25,000 real debris images" and "genuine debris" | **CONTRADICTED** | **MEDIUM** | `docs/PHASE_1_COMPLETE_DOCUMENTATION.md` line 141. SPARK-2021 documentation confirms debris consists of 5 synthetic 3D CAD models rendered in OpenGL/Unreal. | Replace all occurrences of "real debris" and "genuine debris" with "synthetic CAD debris models". |
| **AUD-10** | Split Leakage Wording | "evaluations guaranteed unpolluted" | **CONTRADICTED** | **MEDIUM** | `docs/MASTER_PROJECT_REPORT.md` line 115 vs `docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md` line 271 ("Split leakage NOT ruled out"). | Reconcile wording: disclose that while no augmented images exist in val/test, orbital sequence leakage across raw splits is not ruled out due to dHash collapse. |
| **AUD-11** | Document Consistency | Decision IDs renumbered with conflicting definitions across Phase 1 and Phase 2 docs | **CONTRADICTED** | **MEDIUM** | `docs/PHASE_1_COMPLETE_DOCUMENTATION.md` vs `docs/decisions.md`: DEC-01 (NCNN vs ONNX), DEC-03 (GPU caching vs Augmentation), DEC-05 (Augmentation vs SGP4), DEC-06 (640px vs Hardware). | Add a formal DEC cross-reference mapping in `docs/decisions.md` explaining Phase 1 historical numbering vs Phase 2 unified numbering. |
| **AUD-12** | Project Tracking | `.planning/STATE.md` shows Phase 2 Planned (0/3 plans), last activity Oct 19 | **CONTRADICTED** | **MEDIUM** | `.planning/STATE.md` lines 4-10: out of sync with completed Phase 4 training, Phase 5 eval, and Pi 5 benchmarks. | Apply patch `audit/proposed_patches/04_planning_state.patch` updating STATE.md to Phase 6 Ready. |
| **AUD-13** | Dependency Hygiene | "Pinned host requirements" in Master Report | **CONTRADICTED** | **LOW** | `requirements.txt`: 13 packages listed with zero version constraints (e.g. `ultralytics`, `torch`). | Pin all versions strictly using `pip freeze` from the frozen environment. |
| **AUD-14** | Dataset Availability | 150k SPARK-2021 dataset absent from local development repository | **CANNOT-CHECK** (Locally) | **LOW** | `configs/spark2021.yaml` points to `./data/processed/spark2021` (missing). Raw CSVs were on workstation path `D:\SharanyaTIlak_Spoorthy_Indu\...`. | Document that heavy training/eval artifacts reside on Quadro workstation; local repo maintains models, benchmarks, and evaluation JSONs. |

---

## 3. Numbers Safe to Use vs. Numbers That Must Be Re-checked / Removed

### A. Numbers That Are SAFE to Use (Directly Verified from Raw Artifacts)
1. **Headline Test Set Accuracy (1024x1024 px Native):**
   - Overall mAP@50: **99.02%** (`outputs/evaluations/spark2021_test_metrics.json`)
   - Overall mAP@50-95: **86.98%** (`outputs/evaluations/spark2021_test_metrics.json`)
   - Mean Precision ($P$ at max F1): **98.78%**
   - Mean Recall ($R$ at max F1): **97.87%**
2. **Debris Class Performance (Class 6, 5 CAD Models):**
   - 1024px mAP@50: **98.34%**, mAP@50-95: **85.39%**
   - 800px mAP@50: **98.28%**, mAP@50-95: **84.31%**
   - 640px mAP@50: **98.02%**, mAP@50-95: **82.81%**
3. **Multi-Scale Overall Performance Pareto Frontier:**
   - 800px Overall mAP@50: **98.94%**, mAP@50-95: **86.47%**
   - 640px Overall mAP@50: **98.74%**, mAP@50-95: **85.06%**
4. **Training Convergence & Scale:**
   - Total Epochs: **50 epochs** (`results.csv`)
   - Total Duration: **122,183 seconds (~33.94 hours)** on NVIDIA Quadro GV100
   - Batches / Iterations: **562,500 batches** (batch size 16 across 180,000 images)
   - Mosaic Disablement: Closed at epoch 40 (batch 450,000, `train_batch450000.jpg`)
   - Checkpoint Size: `best.pt` = **6.29 MB** (FP16 weights), `best.onnx` = **12.33 MB / 12.93 MB decimal** (FP32)
   - Model Parameters: **3,140,433**
5. **Physical Hardware Latency & Memory Benchmarks:**
   - **Raspberry Pi 5 (Cortex-A76, ONNX Runtime, 4 threads, 100 runs):**
     - 640px: **144.83 ms** (6.9 FPS), Peak RSS: **138.67 MB**
     - 800px: **285.15 ms** (3.5 FPS), Peak RSS: **175.30 MB**
     - 1024px: **515.19 ms** (1.9 FPS), Peak RSS: **216.73 MB**
   - **Intel Xeon Workstation CPU (16 Cores, ONNX Runtime, 4 threads, 100 runs):**
     - 640px: **23.70 ms** (42.2 FPS), Peak RSS: **129.34 MB**
     - 800px: **35.79 ms** (27.9 FPS), Peak RSS: **162.71 MB**
     - 1024px: **65.20 ms** (15.3 FPS), Peak RSS: **207.36 MB**
6. **Ground-Truth Dataset Counts:**
   - Total Raw Images: **150,000** (90,000 train, 30,000 val, 30,000 test)
   - Spacecraft Images: **125,000** (12,500 per class across 10 classes)
   - Ground-Truth Debris Images: **25,000** (15,000 train, 5,000 val, 5,000 test)
   - Augmented Training Images: **180,000** (90,000 raw + 90,000 `aug1_*`)

---

### B. Numbers That Must Be RE-CHECKED or REMOVED
1. **Master Report CubeSat Metrics:**
   - **REMOVE:** 82.17% (1024px), 81.33% (800px), 79.46% (640px).
   - **REPLACE WITH ACTUAL:** **63.41%** (1024px), **62.08%** (800px), **59.20%** (640px).
2. **Master Report TRMM, AcrimSat, Calipso Metrics:**
   - **REMOVE:** TRMM 74.88%, AcrimSat 90.07%, Calipso 83.21%.
   - **REPLACE WITH ACTUAL:** TRMM **77.76%**, AcrimSat **95.09%**, Calipso **91.33%**.
3. **Master Report Debris mAP@50 at 800px and 640px:**
   - **REMOVE:** 98.19% (800px) and 97.83% (640px).
   - **REPLACE WITH ACTUAL:** **98.28%** (800px) and **98.02%** (640px).
4. **Model FLOPs:**
   - **REMOVE:** Unqualified "8.2 GFLOPs" applied to the 1024px model.
   - **REPLACE WITH:** **8.8 GFLOPs** at 640px, **13.7 GFLOPs** at 800px, **22.4 GFLOPs** at native 1024px.
5. **PRD Gate Hardware Compliance:**
   - **REMOVE:** Claim that PRD gate on Raspberry Pi 4 is PASSED.
   - **REPLACE WITH:** PRD gate is PASSED on Raspberry Pi 5; physical Raspberry Pi 4 validation is pending and projected to pass at 640px (~380-450 ms) but exceed the 1000 ms limit at 1024px (~1300-1500 ms).

---

## 4. Questions a Technical Judge Could Ask That Cannot Be Answered from Evidence

1. **"Why do your Master Technical Report and your SPARK Evaluation Report list completely different per-class mAP scores for the identical model checkpoint (`best.pt`)?"**  
   *Current Problem:* No artifact explains the Master Report's numbers. The Master Report inflated CubeSat by 18.76 points and shaved points off AcrimSat, Calipso, and TRMM so the average remained 86.98%. A judge comparing the files will immediately flag scientific misconduct.
2. **"Your Problem Statement requirement specifically mandates $\le 1000\text{ ms}$ on Raspberry Pi 4 (Cortex-A72). Why are all your physical edge benchmarks reported on a Raspberry Pi 5 (Cortex-A76)?"**  
   *Current Problem:* Pi 5 is 2.5–3x faster than Pi 4. At 1024px (515 ms on Pi 5), the model will exceed 1200–1500 ms on a Pi 4. The project has no Pi 4 benchmark data to prove it meets the original requirement at native resolution.
3. **"In your system diagram, a visual detection queries a NORAD catalog to produce relative ephemeris and conjunction warnings. How does that work when your own code says space debris has no NORAD ID, and 2D bounding boxes cannot provide 3D position or range?"**  
   *Current Problem:* The diagram contradicts DEC-05. Furthermore, `src/data/tle_pipeline.py` maps `CubeSat` to NORAD 25544 (the International Space Station). If a CubeSat is detected, the pipeline computes distance to the ISS.
4. **"Your perceptual hash audit reported 61.27% duplicates between train and test, and your older report admits that split leakage is not ruled out. How do you know your 86.98% mAP is not artificially inflated by video sequence frame leakage?"**  
   *Current Problem:* Global dHash collapsed on solid black space backgrounds. Because no bounding-box ROI hashing was run across rendering trajectories, the team cannot empirically prove that consecutive orbital frames did not appear in both train and test splits.
5. **"You claim Decision DEC-09 froze the model checkpoint on October 19. Where is the git freeze tag, and where in the repository is the SHA-256 hash of `best.pt` recorded?"**  
   *Current Problem:* `git tag -l` reveals no `model-freeze` tag. The SHA-256 hash was never recorded in any documentation or config file.
6. **"Your report claims the model learns high-gradient specular edges of tumbling debris. Where is your feature attribution, Grad-CAM, or saliency map artifact proving this?"**  
   *Current Problem:* No such artifact exists. The claim is qualitative speculation based on the fact that the 5 synthetic CAD debris models have metallic surfaces.

---

## 5. Prioritized 5-Day To-Do List

### Target Milestones:
- **Sun 25 Oct 2026:** Final Model Freeze & Provenance Lock
- **Fri 30 Oct 2026:** End-to-End System Integration & Project Delivery

| Priority | Deadline | Action Item | Detailed Technical Steps | Risk if Skipped |
|:---:|---|---|---|---|
| **P0** | **Sat 24 Oct, 14:00** | **Correct Master Report Per-Class Table** | Replace fabricated per-class values in `docs/MASTER_PROJECT_REPORT.md` with verified JSON numbers (CubeSat = 63.41%, TRMM = 77.76%, AcrimSat = 95.09%, Calipso = 91.33%). Explain that CubeSat is the smallest object ($<100\times100$ px), making high-threshold IoU regression sensitive to 3-5 pixel boundary shifts. | **Disqualification:** Presentation of manipulated numbers in formal project defense. |
| **P0** | **Sat 24 Oct, 18:00** | **Execute Formal Git Model Freeze (DEC-09)** | 1. Calculate SHA-256 of `best.pt` (`34fe3d10b045849fef33e29f10028e878fbd7745a71e2839aa02875e47c8795d`) and `best.onnx` (`df2cde420af209acddcc9aceb5e9d9ca909d75cdca16474bc7f71aca60260912`).<br>2. Commit hashes into `docs/decisions.md`.<br>3. Tag git commit with `git tag -a model-freeze -m "Model freeze: 11-class YOLOv8n trained on SPARK-2021"`. | **Audit Failure:** Inability to demonstrate checkpoint provenance and weight reproducibility. |
| **P1** | **Sun 25 Oct, 12:00** | **Reconcile Architecture Diagram & TLE Pipeline** | 1. Update Mermaid diagram in `MASTER_PROJECT_REPORT.md`: decouple vision bearing from catalog propagation.<br>2. In `src/data/tle_pipeline.py`: remove CubeSat -> ISS (25544) proxy; replace with a dedicated disclaimer that CubeSat requires orbit determination.<br>3. Clearly label SGP4 as a catalog lookup layer for known assets, NOT a conjunction solver for uncataloged debris. | **Technical Incoherence:** A judge with orbital mechanics background will reject the submission. |
| **P1** | **Sun 25 Oct, 18:00** | **Clarify Raspberry Pi 4 vs Pi 5 Hardware Gate** | Update all documents to state: "Empirically validated on Raspberry Pi 5 (Cortex-A76) meeting $\le 1000\text{ ms}$ across 640px (145ms), 800px (285ms), and 1024px (515ms). Raspberry Pi 4 (Cortex-A72) physical validation remains an open gate (projected pass at 640px, failure at 1024px)." | **Credibility Loss:** Accusation of bait-and-switch regarding hardware specifications. |
| **P2** | **Mon 26 Oct, 16:00** | **Phase 6: Static INT8 Post-Training Quantization** | Run static INT8 calibration on `best.onnx` using 300 representative training images. Benchmark quantized model on Xeon CPU and evaluate accuracy retention on test split via `src/eval.py`. Confirm binary size shrinks to $\le 4\text{ MB}$. | Missed milestone: INT8 quantization is required to prove edge deployability. |
| **P2** | **Tue 27 Oct, 18:00** | **Phase 8: Multi-Backend CLI (`src/infer.py`)** | Implement unified CLI supporting PyTorch and ONNX inference: `python src/infer.py --source <img.jpg> --weights best.onnx --imgsz 640 --save-json`. | Incomplete deliverables: Roadmap requires end-to-end inference CLI. |
| **P3** | **Wed 28 Oct, 18:00** | **Synchronize Project State & Pin Dependencies** | 1. Update `.planning/STATE.md` to reflect Phase 6/7 completion.<br>2. Pin exact package versions in `requirements.txt` (`pip freeze`).<br>3. Add formal DEC mapping table in `docs/decisions.md`. | Poor project hygiene and reproducibility issues on fresh environments. |
