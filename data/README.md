# Dataset Shortlist — Space Debris Detection

## Summary

| Dataset | Size | Classes | License | Access friction |
|---|---|---|---|---|
| SPARK (2022) | ~150k synthetic images (Stream-1) | 11: 10 named spacecraft + 1 generic `debris` | Restricted (registration required) | **High — register today, approval can take days** |
| SPEED+ | 59,960 synthetic + 9,531 real testbed images | 1 (Tango spacecraft, pose only) | CC BY-NC-SA 4.0 | Low — instant download, no approval wait |
| SwissCube | 50,000 synthetic images | 1 (CubeSat, pose only) | MIT | None — instant download via Hugging Face |

## 1. SPARK (SPAcecraft Recognition leveraging Knowledge of space environment)

- **Publisher:** SnT, University of Luxembourg
- **Size:** ~150,000 annotated multi-modal images in Stream-1 (detection stream); Stream-2 holds trajectory sequences for tracking.
- **Format:** 1080×1080 RGB images with bounding-box annotations; depth and segmentation masks available for a subset. Real images also included from the Zero-G Lab facility in later editions (2022/2024).
- **Classes (11):** `lisa_pathfinder`, `proba_data3_csc`, `smart_1`, `xmm_newton`, `soho`, `earth_observation_sat_1`, `debris`, `proba_2`, `proba_3_ocs`, `cheops`, `double_star`.
- **License / access:** Record is publicly listed on Zenodo but **files are restricted**. Access requires registering at the official challenge site (cvi2.uni.lu/spark2022 or the current year's challenge page) before a request can be approved. This is a manual review process — **not instant**.
- **Why shortlisted:** This is the only one of the three with an actual `debris` class baked in alongside multiple satellite types, which most directly matches our 3-class target (satellite / debris / rocket body). Published baselines report strong YOLOv3 performance (mAP@0.5 ≈ 97%) on this exact class set, so it's a proven detector-friendly dataset.
- **Gap to flag:** No dedicated `rocket_body` class exists in SPARK — see `docs/decisions.md`.

## 2. SPEED+ (Next-Gen Spacecraft Pose Estimation Dataset)

- **Publisher:** Stanford Space Rendezvous Laboratory (SLAB), released with ESA's Advanced Concepts Team
- **Size:** 59,960 labeled synthetic images (80:20 train/val split) + 9,531 real Hardware-in-the-Loop testbed images (lightbox + sunlamp domains).
- **Format:** Grayscale-rendered images with full 6D pose labels (position + attitude), not class labels — this is a single-target pose dataset, not multi-class detection.
- **License:** CC BY-NC-SA 4.0 — free for non-commercial/academic use, share-alike.
- **Access:** Publicly downloadable now via Stanford Digital Repository / Zenodo, no approval queue.
- **Why shortlisted:** Not useful for our classification head (it's one spacecraft, Tango, across domains), but directly useful for the synthetic-to-real domain gap problem the project report calls out — the lightbox/sunlamp real-image domains are exactly the kind of validation set we need to measure generalization honestly.

## 3. SwissCube

- **Publisher:** EPFL CVLAB
- **Size:** 50,000 synthetic images (500 scenes × 100-frame trajectories), 1024×1024 resolution.
- **Format:** Physically-rendered RGB images (Mitsuba 2 ray tracer) with pose and segmentation mask annotations. Mirrored on Hugging Face.
- **License:** MIT — fully permissive, no usage restriction.
- **Access:** Zero friction — direct `git clone` / Hugging Face download, no account or registration needed.
- **Why shortlisted:** Best fallback if SPARK approval is delayed past our internal deadlines. Also useful for stress-testing the detector on high-fidelity lighting/occlusion (Earth/Sun reflection effects) that SPARK's simpler renders may not cover.

## Recommended strategy

Use **SPARK** as the primary training set for the multi-class detector once access is approved (it's the only one with a real debris class). Start downloading **SPEED+** and **SwissCube** today since they require no approval wait — use them immediately for pipeline development, augmentation testing, and the synthetic-to-real validation step, so the team isn't blocked waiting on SPARK approval.