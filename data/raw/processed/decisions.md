# Decision Log — Object Classes

## Target taxonomy (per project report, Section 3.2 / Stage 2)

The system needs to classify detections into three semantically meaningful categories for downstream collision-risk flagging:

1. **satellite** — active or inactive spacecraft, including both operational and defunct platforms
2. **debris** — fragments, paint flecks, explosion/collision byproducts, anything without a recognizable spacecraft structure
3. **rocket_body** — spent upper stages / boosters left in orbit

## Mapping from shortlisted datasets

| Our class | SPARK | SPEED+ | SwissCube |
|---|---|---|---|
| `satellite` | All 10 named classes (`lisa_pathfinder`, `proba_2`, `cheops`, etc.) collapse into this superclass | `Tango` (single target) | The CubeSat target |
| `debris` | Direct 1:1 match — SPARK's `debris` class | Not present | Not present |
| `rocket_body` | **Not present** | Not present | Not present |

## Open decision: rocket_body has no source data

None of the three shortlisted datasets contain a dedicated rocket-body class — they're either spacecraft-recognition sets (SPARK) or single-target pose sets (SPEED+, SwissCube). This is a real gap, not an oversight, and needs a call before training starts.

**Options considered:**

- **A — Collapse to 2 classes for v1 (satellite / debris).** Defer rocket_body to a stretch goal. Lowest risk, fastest to a working pipeline, but understates the project's stated scope.
- **B — Synthesize a rocket_body class.** Render a generic cylinder/booster mesh (Blender) and composite it into star-field backgrounds the same way SPARK was built, similar in spirit to our planned synthetic streak-generation approach. Gives us a real third class but costs build time we don't have today.
- **C — Source a handful of real labeled images for a small validation-only rocket_body set** (e.g. from public astrophotography of known reentry/booster events), used only to qualitatively report on this class rather than train on it — consistent with how the report already treats real images as validation-only, not primary training data.

**Recommendation:** Go with **A** as the default for the Week 1–2 baseline (keeps us on the Week 6 hard checkpoint), with **C** added opportunistically in Weeks 5–6 if time allows, and **B** only as a stretch goal if the core pipeline lands early. Flagging this to Person A before the next sync since it affects the classification head's output dimension from day one.

## Class imbalance note

Carried over from SPARK's published baseline: even within its 11 classes, the dataset is roughly balanced (~13–14k images/class), but our collapsed `satellite` superclass will end up with ~10x more images than `debris` once classes are merged. Plan to apply focal loss / class-weighted sampling from the first training run, not as a later fix — consistent with the project report's stated risk mitigation for synthetic-to-real gap and the general SSA class-imbalance problem.

## Next decision point

Revisit this taxonomy after the first SPARK-trained baseline (end of Week 2) — if `rocket_body` data becomes available via option B or C, the classification head and label files will need to be regenerated, so better to confirm early rather than mid-training.