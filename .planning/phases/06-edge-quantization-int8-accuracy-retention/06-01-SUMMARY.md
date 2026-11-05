# Plan 06-01: Astrodynamics Logic Cleansing, Hardware Gate Qualification & Data Leakage Disclosure — Summary

**Executed:** 2026-11-05  
**Status:** Completed & Verified  
**Requirements Addressed:** `PROD-03`, `EDGE-02`, `DATA-02`  
**Commits:**
- `5cce31b`: `fix(tle): remove hardcoded ISS proxy from CubeSat and enforce uncataloged error handling (06-01-01)`
- `62b39fd`: `docs(audit): qualify Pi 5 benchmarks, designate 640px for Pi 4, and cleanse TLE architecture (06-01)`

---

## Accomplishments

1. **Astrodynamics Logic Cleansing (`src/data/tle_pipeline.py`):**
   - Removed hardcoded ISS proxy (`"CubeSat": 25544`). Set `"CubeSat": None` to reflect that multi-manifest CubeSats lack a single deterministic NORAD ID in CelesTrak.
   - Enforced uncataloged status in `get_position()` for `CubeSat` and `Debris`, explicitly documenting that 2D optical detections cannot determine 3D orbital state vectors without multi-station orbit determination.
   - Clarified that `calculate_distance()` operates strictly between verified active catalog assets (DEC-05) for parallel ground-tracking awareness.

2. **System Architecture Diagrams Cleansing:**
   - Decoupled visual detection pipelines from orbital catalog layers across `docs/MASTER_PROJECT_REPORT.md` and `docs/decisions.md`.
   - Removed automated conjunction risk prediction nodes from detector outputs.

3. **Raspberry Pi Hardware Gates Qualification:**
   - Explicitly updated `docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md`, `docs/PHASE_1_VERIFICATION_AND_RESULTS.md`, `docs/WORKFLOW_PLAYBOOK.md`, and `docs/decisions.md` to state that the $\le 1000\text{ ms}$ real-time gate was validated on physical **Raspberry Pi 5** (144.83 ms at 640px, 285.15 ms at 800px, 515.19 ms at 1024px).
   - Formally designated **640px** as the safe real-time operating point projected to pass on **Raspberry Pi 4** (~348 ms projected, $>60\%$ margin under 1000 ms ceiling).

4. **Data Leakage Caveats Disclosure:**
   - Documented the physical mechanism of perceptual difference hash (dHash) collapse on $>95\%$ uniform solid black space backgrounds.
   - Transparently disclosed that sequence-level trajectory correlation between train, val, and test splits cannot be ruled out by global thumbnail hashing.

---

## Verification Evidence

- `src/data/tle_pipeline.py`: Verified `OrbitalTracker().class_to_norad['CubeSat'] is None`, and `get_position("CubeSat")` and `get_position("Debris")` return uncataloged notices.
- `docs/MASTER_PROJECT_REPORT.md` & `docs/decisions.md`: Verified decoupled architecture and catalog overlay.
- `docs/SPARK_2021_TRAINING_AND_EVALUATION_REPORT.md`: Verified Raspberry Pi 5 measurements and Raspberry Pi 4 640px safe operating point designation.
- Data leakage caveats verified in report texts.

