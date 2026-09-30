# October plan: PAI track only (scope set by Aman 2026-09-30)

**Goal:** win the PAI track with the Alpamayo 1.5 line. Everyone else will improve in October too, so there's no fixed target: maximise our own score. The leaderboard closes 2026-10-31.

**Standing (2026-09-29):** PAI PCS 2143, rank 2/58 (leader acmlab-dow 2224). Scene score 0.611, at-fault incident spacing 0.795.

**Scope:**
- **This repo:** PAI only.
- **nuPlan:** moved to the JEPA thread (`../alpasim-jepa`, another agent) on 2026-09-30. No nuPlan runs here. Our nuPlan notes stay in LEDGER.md for that thread to read.
- **`../alpasim-jepa`** and its cluster folder are read-only for this repo.
- **Slots:** October has 3 official slots shared by both tracks. PAI plans for up to 2 of them; the JEPA thread takes 1 nuPlan slot and 1 nuPlan warmup early in October. Nothing is submitted without Aman's go-ahead; Aman submits.

## Principles

- **Levers are parts of an architecture, not standalone models.** Report each as "pipeline X plus or minus part Y". "Didn't work" always means "didn't work in configuration X".
- **Design the full system from a failure analysis** (one part per failure cause), then **ablate**: remove one part at a time from the full system.
- **Measurement:** one run per configuration on the 120-scene PAI bench, paired per scene (`src/compare_runs.py`).
  - No repeated runs (too costly).
  - Noise was ±0.03 at 40 scenes, so ~±0.017 is expected at 120.
- **Driver checks come first:** every run starts with the driver regression tests and ends with `driver_health.txt`.
- **No hand-tuning to the official set.** Official results are only used to calibrate local vs official.
- **Clear negatives** (reasoning text -0.165, K=5 medoid -0.082 at 40 scenes) come back only with a fix for why they failed.

## What the PAI bench says (120 scenes, v0.3 = 0.638 local vs 0.611 official)

| Failure | Scenes | Score lost | What it looks like |
|---|---|---|---|
| Left the 4 m corridor | 24 | 24.0 | Mostly lane choice on straight roads: the recorded driver changes lane, ours stays, ending 4.0-4.3 m off. A few junction turns. |
| Low progress | 24 | 10.4 | Too slow; some nearly stop on long routes. |
| At-fault collision | 5 | 5.0 | |
| Offroad | 4 | 4.0 | |

## Weeks

1. **Bench and failure analysis (to Oct 5)**
   - Done: 120-scene PAI baseline, taxonomy, corridor-exit videos.
   - Running: route chooser (K=6) on all 120 scenes.
   - Next:
     - per-scene look at lane-change corridor exits (does the route show the lane change? do candidates include it?);
     - the low-progress scenes (why the car slows or stops);
     - the PAI time budget for any K>1 chooser (v0.3 used 9,474 of 14,400 s).
2. **Full system (Oct 6 to 12)**, one part per failure cause:
   - route-consistent candidate choice for lane choice;
   - speed / progress handling (chooser feature or a fine-tuned action expert on PAI-domain data);
   - controller gains tuned jointly.
   - Low priority: a JEPA plan as an extra candidate (the per-scene best of JEPA and v0.3 on 40 PAI scenes was 0.762 vs 0.668). Only via the JEPA thread's published outputs, read-only.
3. **Ablations and the first October PAI submission (Oct 13 to 19):** one bundled step-up, after Aman's go-ahead.
4. **Fix what the official result reveals; second PAI slot (Oct 20 to 28),** well before the Oct 31 close.

## Constraints

- Driver ≤ 16 GiB (v0.3 peaks at 15,052 MiB in closed loop); PAI time limit 14,400 s (v0.3 used 9,474).
- Cluster:
  - GPU time, fair-share and storage are shared with the JEPA thread (department filesystem ~95% full). Keep PAI runs modest: one Blackwell GPU at a time (`--dependency=singleton`).
  - Local PAI runs need Blackwell nodes (driver 575). Per-job port blocks are mandatory.
- Tell Aman before big GPU runs. Aman submits; Claude prepares verified images.
