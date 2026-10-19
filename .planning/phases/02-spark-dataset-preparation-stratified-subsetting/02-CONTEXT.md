# Phase 2: SPARK Dataset Preparation & Stratified Subsetting — Context

**Gathered:** 2026-10-19  
**Status:** Ready for planning  
**Collaboration Roles:** Data & Edge Lead (Role 2) and ML & CV Lead (Role 1), governed by Version 3.0 Work Distribution.  

---

<domain>
## Phase Boundary

Phase 2 establishes the complete, production-grade dataset ingestion, formatting, augmentation, and validation pipeline for the space debris detection model:
- **SPARK-2021 Transition (DEC-10 superseding DEC-04):** Integrates the official SPARK-2021 dataset containing 150,000 images across 11 classes (10 satellites + 1 consolidated Debris class with ~25,000 debris images across 5 geometric models).
- **Format Conversion & Bounding Box Transformation:** Translates SPARK-2021 row/column ordered coordinates `[R_min, C_min, R_max, C_max] == [y_min, x_min, y_max, x_max]` into normalized YOLO center format `<class_id> <cx> <cy> <w> <h>`, mapping splits into `data/processed/spark2021/{images,labels}/{train,val,test}`.
- **Photometric Space-Domain Augmentation:** Implements domain-specific transforms (solar flare exposure jitter, optical motion blur, sensor read noise) isolated strictly to the training split (`train/`), generating 90,000 augmented copies (`aug1_*`) to double the training corpus to 180,000 images while protecting validation/test split integrity.
- **Split Leakage Audit & Verification:** Executes perceptual hash auditing (`scripts/check_split_leakage.py`) across train and test splits, documenting the black space background hash collapse phenomenon and recording honest leakage audit findings.

</domain>

<decisions>
## Implementation Decisions

### 1. Dataset Selection & Class Mapping (DEC-10)
- **Decision:** Shift from old SPARK (spacecraft-only) to official **SPARK-2021** (11 classes).
- **Class ID Mapping [0..10]:**
  - `0: AcrimSat`, `1: Aquarius`, `2: Aura`, `3: Calipso`, `4: Cloudsat`
  - `5: CubeSat`, `6: Debris` (Consolidated ~25k instances across 5 CAD models)
  - `7: Jason`, `8: Sentinel-6`, `9: Terra`, `10: TRMM`
- **Reversibility:** `one-way` — all downstream model weights, configs, and metric evaluations depend on this 11-class definition.

### 2. Coordinate Transformation (BBox Convention Swap)
- **Decision:** Implement deterministic coordinate swap in `scripts/convert_spark2021_to_yolo.py`:
  $$\text{cx} = \frac{c_{\min} + c_{\max}}{2 \cdot W}, \quad \text{cy} = \frac{r_{\min} + r_{\max}}{2 \cdot H}$$
  $$w = \frac{c_{\max} - c_{\min}}{W}, \quad h = \frac{r_{\max} - r_{\min}}{H}$$
- **Collision-Free Flattening:** To resolve identical image filenames across class subfolders in raw ZIPs (`image_00001_img.jpg`), flattened filenames are prefixed as `{class_name}_{image_stem}.jpg` and `{class_name}_{image_stem}.txt`.

### 3. Photometric Augmentation Isolation (DEC-05)
- **Decision:** Photometric transforms (brightness/contrast scaling, Gaussian blur, Gaussian noise) alter pixel values without modifying pixel coordinates.
- **Isolation Gate:** Augmentations execute **exclusively** on `train/`. `val/` and `test/` remain strictly untouched to ensure uncorrupted metric evaluation.
- **Output:** Generates `aug1_*` images, yielding 180,000 total training samples (90k raw + 90k augmented).

### 4. Split Leakage Audit Protocol
- **Decision:** Execute difference hashing (dHash) to check for near-duplicates across train and test splits.
- **Finding Disclosure:** Document the black-space background hash collapse (>95% black pixels cause full-frame dHash collapse), transparently reporting that sequence leakage cannot be ruled out by global thumbnail hashing alone.

</decisions>

<canonical_refs>
## Canonical References

- [WORKFLOW_PLAYBOOK.md](file:///c:/Users/tilak/Desktop/Repos/space-debris-detection/docs/WORKFLOW_PLAYBOOK.md) — Operational instructions for SPARK-2021 extraction, audit, conversion, and augmentation.
- [SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md](file:///c:/Users/tilak/Desktop/Repos/space-debris-detection/docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md) — Empirical 50-epoch GV100 training results and multi-resolution Pareto profile.
- [decisions.md (DEC-10, DEC-05, DEC-07)](file:///c:/Users/tilak/Desktop/Repos/space-debris-detection/docs/decisions.md) — Architectural decision log.
- [scripts/convert_spark2021_to_yolo.py](file:///c:/Users/tilak/Desktop/Repos/space-debris-detection/scripts/convert_spark2021_to_yolo.py) — Bounding box coordinate conversion utility.
- [scripts/audit_spark2021_annotations.py](file:///c:/Users/tilak/Desktop/Repos/space-debris-detection/scripts/audit_spark2021_annotations.py) — Ground-truth annotation audit tool.
- [scripts/check_split_leakage.py](file:///c:/Users/tilak/Desktop/Repos/space-debris-detection/scripts/check_split_leakage.py) — Difference hash cross-split leakage audit.
- [src/data/augment.py](file:///c:/Users/tilak/Desktop/Repos/space-debris-detection/src/data/augment.py) — Photometric space-domain data augmentation pipeline.
</canonical_refs>
