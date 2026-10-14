---
phase: "01"
slug: "foundations-evaluation-harness-day-1-audit"
status: draft
nyquist_compliant: true
wave_0_complete: false
created: "2026-10-13"
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution of Phase 1.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Python standalone verification scripts (`src/eval.py`, audit scripts) |
| **Config file** | `swisscube.yaml`, `spark.yaml` |
| **Quick run command** | `python src/eval.py --help` |
| **Full suite command** | `python src/eval.py --weights experiments/week1_smoke_test/weights/best.pt --data swisscube.yaml --split val --imgsz 1024 --out experiments/week1_smoke_test/metrics.json` |
| **Estimated runtime** | ~15–30 seconds on CPU / GPU |

---

## Sampling Rate

- **After every task commit:** Run task-specific verification command
- **After every plan wave:** Run full validation verification
- **Before `/gsd-verify-work`:** All Phase 1 deliverables must have verifiable artifacts
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | EVAL-01 | — | N/A | unit/script | `python src/eval.py --weights experiments/week1_smoke_test/weights/best.pt --data swisscube.yaml --split val --imgsz 1024 --out experiments/week1_smoke_test/metrics.json` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | EVAL-02 | — | N/A | integration | `python -c "import json; m=json.load(open('experiments/week1_smoke_test/metrics.json')); assert abs(m['map50'] - 0.9903) < 0.005; print('BASELINE VERIFIED')"` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | DATA-01 | — | N/A | audit/script | `python -c "import os; assert os.path.exists('docs/decisions.md'); print('AUDIT DOCUMENTED')"` | ✅ | ⬜ pending |
| 01-03-01 | 03 | 1 | DELV-01 | — | N/A | hygiene/file | `python -c \"import yaml; assert os.path.exists('swisscube.yaml') and os.path.exists('requirements-edge.txt'); print('HYGIENE VERIFIED')\"` | ❌ W0 | ⬜ pending |
| 01-03-02 | 03 | 1 | EDGE-02 | — | N/A | unit/script | `python edge/benchmark.py --help` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `swisscube.yaml` — dataset configuration pointing to local SwissCube data
- [ ] `src/eval.py` — implementation of the evaluation CLI
- [ ] `requirements-edge.txt` — lightweight edge dependency manifest

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Raspberry Pi 4 Network Reachability & Wheel Import | EDGE-01 | Physical hardware deployment | SSH into Pi 4, activate venv, run `python -c 'import onnxruntime, ncnn; print(onnxruntime.__version__)'` |
| Class Row Count Verification in Full Download | DATA-01 | Data file location on external drive/disk | Run class inspection script against raw download directory and confirm non-zero counts |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending 2026-10-13
