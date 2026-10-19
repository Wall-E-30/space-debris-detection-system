# Pitfalls Research

**Domain:** Edge AI / Space Situational Awareness Object Detection  
**Researched:** 2026-10-12  
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Resolution & Compute Mismatch between Training and Edge Benchmarks (F3)

**What goes wrong:**
The model is trained and evaluated at $1024 \times 1024$ resolution (achieving 0.99 mAP), but latency is benchmarked on edge hardware at $640 \times 640$ (reporting 165ms). Putting 0.99 mAP and 165ms side by side is an invalid comparison that technical evaluators will immediately challenge. Compute scales with pixel count ($1024^2 \approx 2.6 \times 640^2$).

**Why it happens:**
High resolutions give higher detection accuracy on small objects, while lower resolutions are chosen to make edge benchmarks run faster.

**How to avoid:**
Enforce DEC-06: Standardize deployment and training resolution at $640 \times 640$. Train a $416 \times 416$ variant as a lower-resolution comparison point. Label every results table row with input size, model format, and hardware platform.

**Warning signs:**
Results tables reporting latency and mAP without explicitly specifying image dimensions.

**Phase to address:**
Phase 1 & Phase 2 (SPARK baseline training configuration).

---

### Pitfall 2: Over-Claiming Detection-to-Orbit Conjunction (F5)

**What goes wrong:**
Claiming that a 2D bounding box from a single camera frame "resolves to a NORAD ID" and calculates real-time 3D collision risk. A 2D box has no range, radial velocity, or attitude information; mapping a class label to a catalog ID is a dictionary lookup, not a sensor measurement.

**Why it happens:**
Attempting to connect visual AI directly to astrodynamics without physical sensors (radar/LiDAR/stereo baseline).

**How to avoid:**
Enforce DEC-07: Keep the Skyfield SGP4 module as an independent contextual catalog layer. Redraw architecture diagrams without direct causal arrows from detections to TLE propagation. Present it as an operational awareness demo panel.

**Warning signs:**
Architecture diagrams showing "Bounding Box → NORAD ID → 3D Distance".

**Phase to address:**
Phase 1 (Documentation and architectural framing).

---

### Pitfall 3: Blocking Training on Departmental GPU Allocations (F4)

**What goes wrong:**
Planning training around waiting for external cluster/departmental server access (A100/H100), stalling project progress for weeks.

**Why it happens:**
Assuming the full multi-class SPARK dataset (~150,000 images) must be trained for 50 epochs.

**How to avoid:**
Plan training around the local NVIDIA RTX 2050 (4GB VRAM) laptop GPU: create balanced Subset B (~24k train, 3k val) at 1024ox, which trains in ~3.2–4.3 hours (~5–6.5h with safety margin). Run overnight. Treat departmental GPU access as a bonus.

**Warning signs:**
Workstream status marked "Blocked on GPU access".

**Phase to address:**
Phase 1 (Dataset subsampling) & Phase 2 (Overnight training run).

---

### Pitfall 4: Flawed Edge Memory and Latency Measurements (F6, F10)

**What goes wrong:**
Benchmarking edge inference with too few runs (e.g. 30 runs), no warm-up discarded, uncalibrated thread counts, or reporting memory deltas that show "0.0 MB".

**Why it happens:**
`psutil` memory deltas often measure only Python heap changes rather than the underlying C++ runtime engine allocations.

**How to avoid:**
Use a rigorous benchmark protocol: ≥10 warm-up runs discarded, ≥100 timed iterations, reporting mean, median, and p95 latency. Measure peak Resident Set Size (RSS) for the whole process. Benchmark on cool, idle hardware with 4 threads fixed.

**Warning signs:**
Latency reported as a single average number without variance/p95, or memory delta showing 0 MB.

**Phase to address:**
Phase 1 (Benchmark script refactoring) & Phase 3 (Raspberry Pi execution).

---

### Pitfall 5: Debris Class Absence in Training Data (F1)

**What goes wrong:**
The project is titled "Space Debris Detection", but the downloaded SPARK dataset subset contains 10 satellite spacecraft classes with zero debris class instances.

**Why it happens:**
The SPARK dataset has multiple subsets; some splits contain only spacecraft pose estimation targets.

**How to avoid:**
Day-1 label row count in full SPARK CSV. If debris is missing: honestly state the project is "space-object and satellite detection", train the 10-class model, and show debris only qualitatively from SPEED+ or public domain imagery. Never claim synthetic debris metrics without verified debris data.

**Warning signs:**
Class lists containing only named satellites (`Cheops`, `Proba-2`, etc.) without `debris`.

**Phase to address:**
Phase 1 (Day-1 SPARK audit).

---

_Pitfalls research for: Edge-Deployable Space Debris Detection System_  
_Researched: 2026-10-12_
