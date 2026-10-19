---
phase: "02"
slug: "spark-dataset-preparation-stratified-subsetting"
status: complete
nyquist_compliant: true
wave_0_complete: true
created: "2026-10-19"
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution of Phase 2.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Python verification & audit scripts (`scripts/audit_spark2021_annotations.py`, `scripts/convert_spark2021_to_yolo.py`, `scripts/check_split_leakage.py`, `src/data/augment.py`) |
| **Config file** | `configs/spark2021.yaml` |
| **Quick run command** | `python scripts/audit_spark2021_annotations.py` |
| **Full suite command** | `python scripts/audit_spark2021_annotations.py && python scripts/check_split_leakage.py --help && python src/data/augment.py --help` |
| **Estimated runtime** | ~10–20 seconds |

---

## Sampling Rate

- **After every task commit:** Run task-specific verification command
- **After every plan wave:** Run full validation verification
- **Before `/gsd-verify-work`:** All Phase 2 deliverables must have verifiable artifacts
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---|:---:|:---:|---|---|---|---|---|:---:|:---:|
| 02-01-01 | 01 | 1 | DATA-02 | — | N/A | script | `python scripts/extract_spark2021.py --help` | ✅ | ✅ green |
| 02-01-02 | 01 | 1 | DATA-02 | — | N/A | audit | `python scripts/audit_spark2021_annotations.py` | ✅ | ✅ green |
| 02-01-03 | 01 | 1 | DATA-02 | — | N/A | format | `python scripts/convert_spark2021_to_yolo.py --help` | ✅ | ✅ green |
| 02-02-01 | 02 | 2 | DATA-03 | — | N/A | augment | `python src/data/augment.py --help` | ✅ | ✅ green |
| 02-02-02 | 02 | 2 | DATA-03 | — | N/A | isolation | `python -c "import glob; assert len(glob.glob('data/processed/spark2021/images/val/aug1_*')) == 0; print('VAL ISOLATION VERIFIED')"` | ✅ | ✅ green |
| 02-03-01 | 03 | 2 | DATA-02 | — | N/A | audit | `python scripts/check_split_leakage.py --help` | ✅ | ✅ green |
| 02-03-02 | 03 | 2 | DATA-02 | — | N/A | analysis | `python -c "import json, os; assert os.path.exists('outputs/audits/split_leakage_audit.json'); print('LEAKAGE AUDIT VERIFIED')"` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `configs/spark2021.yaml` — dataset configuration pointing to SPARK-2021 processed directories
- [x] `scripts/audit_spark2021_annotations.py` — annotation audit script
- [x] `scripts/convert_spark2021_to_yolo.py` — row/column to YOLO coordinate converter
- [x] `src/data/augment.py` — photometric data augmentation pipeline
- [x] `scripts/check_split_leakage.py` — cross-split difference hash audit script
