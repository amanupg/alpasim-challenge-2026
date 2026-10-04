# October plan: PAI track only (scope set by Aman 2026-09-30)

**Goal:** win the PAI track with the Alpamayo 1.5 line. Everyone else will improve in October too, so there's no fixed target: maximise our own score. The leaderboard closes 2026-10-31.

**Standing (2026-09-29):** PAI PCS 2143, rank 2/58 (leader acmlab-dow 2224). Scene score 0.611, at-fault incident spacing 0.795.

**Scope:**
- **This repo:** PAI only.
- **nuPlan:** moved to the JEPA thread (`../alpasim-jepa`, another agent) on 2026-09-30. No nuPlan runs here. Our nuPlan notes stay in LEDGER.md for that thread to read.
- **`../alpasim-jepa`** and its cluster folder are read-only for this repo.
- **Slots (Aman, 2026-10-04): PAI gets exactly ONE official October slot.**
  - nuPlan (JEPA thread) gets one; the third is Aman's call.
  - No "submit, learn, fix, resubmit": the one PAI entry must be our best verified system, decided on the local bench.
  - Nothing is submitted without Aman's go-ahead; Aman submits.

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

## Weeks (one PAI slot)

1. **Bench and failure analysis (done Oct 3)**
   - 120-scene baseline (0.638), taxonomy, videos.
   - Chooser v2 +0.054 [+0.005, +0.105] (kept); v3 same as v2; speed scaling rejected; route text on PAI being measured.
2. **Full system (Oct 4 to 17)**, one part per failure cause, each measured on the 120-scene bench against the current best:
   - lane choice: chooser v2 (in);
   - slow driving (26 scenes, ~10 points): needs a model-side fix, since candidates agree on speed. Options: a fine-tuned action expert on PAI-domain data; stop/go handling at lights and queues;
   - remaining corridor exits (14, all 4.0-4.2 m: knife-edge): controller / lateral tracking with the chooser in place;
   - low priority: a JEPA plan as an extra candidate, only via the JEPA thread's published outputs, read-only.
3. **Freeze and verify (Oct 18 to 24)**
   - Ablate the full system (remove one part at a time) on the 120-scene bench.
   - Build the `v5` image from the frozen commit and run that exact image locally on the bench (it must reproduce the bench numbers).
   - Spend the one remaining PAI warmup on it to check form factor and per-wave time against the 14,400 s limit. (The two September warmups free up again around Oct 28.)
4. **One official PAI submission (target Oct 25 to 27),** after Aman's go-ahead. That leaves days of buffer before the Oct 31 close, since a failed official run still uses the slot.

## Constraints

- Driver ≤ 16 GiB (v0.3 peaks at 15,052 MiB in closed loop); PAI time limit 14,400 s (v0.3 used 9,474).
- Cluster:
  - GPU time, fair-share and storage are shared with the JEPA thread (department filesystem ~95% full). Keep PAI runs modest: one Blackwell GPU at a time (`--dependency=singleton`).
  - Local PAI runs need Blackwell nodes (driver 575). Per-job port blocks are mandatory.
- Tell Aman before big GPU runs. Aman submits; Claude prepares verified images.
- **Image tags are neutral version numbers** (`v5`, `v6`, ...), never method words: the leaderboard shows tags publicly.
  - One sequence is shared with the JEPA thread, since both push to `teams/amanupg`. Public so far: `v0.1-starter`, `v0.3-a15int8-1hz` (already public, can't be renamed), and `v4` (JEPA thread, nuPlan). PAI's next is `v5` or the next free number; check the tag is unused before pushing.
  - What each tag contains is recorded only in LEDGER.md (private repo). Local build names are neutral too.
