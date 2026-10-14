# Centralized Experiment & Execution Output Registry

All execution results, benchmark numbers, evaluation metrics, and environment audit artifacts are automatically saved here with full persistence and timestamps.

---

## Directory Organization

```
outputs/
├── evaluations/         # Model accuracy, mAP, and parameter evaluations
│   ├── eval_YYYYMMDD_HHMMSS.json  # Full evaluation snapshot per run
│   ├── latest.json                # Pointer to the most recent evaluation
│   └── history.jsonl              # Chronological append-only ledger of all evaluation runs
│
├── benchmarks/          # Edge inference latency, FPS, and peak RAM benchmarks
│   ├── benchmark_YYYYMMDD_HHMMSS.json  # Comprehensive benchmark run (mean, p95, peak RSS)
│   ├── latest.json                     # Pointer to the most recent benchmark
│   └── history.jsonl                   # Chronological append-only ledger of all benchmarks
│
├── audits/              # Dataset class distribution and label verification reports
│   ├── audit_YYYYMMDD_HHMMSS.json      # SPARK / dataset class audit record
│   └── latest.json                     # Pointer to the most recent audit
│
└── env_checks/          # Testbed hardware and runtime verification reports
    ├── env_check_YYYYMMDD_HHMMSS.json  # Raspberry Pi 4 environment checklist snapshot
    └── latest.json                     # Pointer to the most recent environment check
```

---

## How to Share Results with the AI

When you run any script locally (e.g., on your GPU or on the Raspberry Pi 4):
1. **You don't need to copy-paste terminal logs:** Every script automatically updates `outputs/*/latest.json` and creates a timestamped JSON file.
2. **In our future conversations:** You can simply say:
   - *"Check latest benchmark"* (I will inspect `outputs/benchmarks/latest.json`)
   - *"Check latest evaluation"* (I will inspect `outputs/evaluations/latest.json`)
   - *"Compare history"* (I will inspect `outputs/*/history.jsonl`)
