# October plan (agreed with Aman 2026-09-29)

**Goal:** win both tracks. Everyone else will improve in October too, so there's no fixed target: maximise our own score. The leaderboard closes 2026-10-31. There are 3 official slots in October, shared by both tracks.

**Standing (2026-09-29):**
- PAI: PCS 2143, rank 2/58 (leader 2224).
- nuPlan: PCS 1296, rank 19/71. About 4th among likely-eligible teams; the top 10 is mostly the host team py123d-garage.

## Principles

- **Levers are parts of an architecture, not standalone models.** Report each as "pipeline X plus or minus part Y". "Didn't work" always means "didn't work in configuration X".
- **Design the full system from a failure analysis** (one part per failure cause), then **ablate**: remove one part at a time from the full system.
- **Measurement:** one run per configuration on ~120 scenes per track, paired per scene (`src/compare_runs.py`).
  - No repeated runs (too costly).
  - Noise was ±0.03 at 40 scenes, so ~±0.017 is expected at 120.
- **No hand-tuning to the official set.** Official results are only used to calibrate local vs official.
- **Clear negatives** (reasoning text -0.165, K=5 medoid -0.082 at 40 scenes) come back only with a fix for why they failed.

## Weeks

1. **Test bench and failure analysis (Sep 29 to Oct 5)**
   - PAI: 120 curated-val scenes. nuPlan: 120 navtest scenes. Download in batches; delete scene data we no longer need.
   - Per-scene failure taxonomy (collision type, corridor exit at intersection, offroad, progress/speed) from logs and rollout videos.
   - Local-vs-official calibration: PAI starter 0.349 local vs 0.140 official; v0.3 ~0.68 vs 0.611.
2. **Full system (Oct 6 to 12)**
   - Candidates: Alpamayo 1.5 samples, plus a fine-tuned action expert (NuRec/PhysicalAI-AV data for PAI; nuPlan data for nuPlan).
   - A learned chooser scoring safety, corridor and progress (the nuReasoning "menu and waiter"), instead of the medoid.
   - Route input where the failure analysis says intersections cost us.
   - Controller gains tuned jointly.
   - nuPlan: add the nuReasoning-borrowed models as candidates (fine-tuned Alpamayo on the Motional rig, nuVLA), read-only per BORROWED.md.
3. **Ablations plus the first October submission (Oct 13 to 19):** one bundled step-up per track.
4. **Fix what the official results reveal; last two slots (Oct 20 to 28),** well before the Oct 31 close.

## Out of scope here

- JEPA models are a separate thread run by another agent (`../alpasim-jepa`). This repo stays on the Alpamayo 1.5 line for both tracks.

## Constraints

- Driver ≤ 16 GiB (v0.3 peaks at 15,052 MiB in closed loop), and within the per-track time limit (PAI 14,400 s, v0.3 used 9,474; nuPlan 3,000 s, v0.3 used 1,268).
- Cluster:
  - Fair-share 0.547 on 2026-09-29.
  - Local PAI runs need Blackwell nodes (driver 575); nuPlan runs on L40S/H100.
  - Per-job port blocks are mandatory.
- Tell Aman before big GPU runs. Aman submits; Claude prepares verified images.
