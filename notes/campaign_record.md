# AlpaSim Challenge 2026, PAI track: complete campaign record

Aman Upganlawar (Columbia University), team `amanupg`. Sep 26 – Oct 31, 2026.

This is the narrative record, kept for resume bullets, interviews, study and talking about the work. Every number comes from `LEDGER.md` (the day-by-day log with job IDs). Lessons in short form are in `notes/learnings_pai_campaign.md`, and the cross-agent lessons are in `../LESSONS_FOR_AGENTS.md`.

---

## 1. Outcome in one paragraph

We entered the AlpaSim end-to-end closed-loop driving challenge (NVIDIA, NeurIPS 2026) on the PAI track. The entry was NVIDIA's own 10B vision-language-action model, Alpamayo 1.5, running inside a thin closed-loop driver that we built.

The driver had several parts:
- int8 weight-only quantization to fit the 16 GiB per-driver VRAM cap;
- replanning at 1 Hz;
- a geometric "route snap" correction for lane changes;
- a forced trajectory-start token;
- route text in the prompt;
- a start rule for stalled plans;
- a fine-tuned and weight-averaged action expert, trained on simulator-rendered data with recovery labels.

Six official PAI submissions in total:

| Entry | Final PCS | Final rank | Avg scene score | Notes |
|---|---|---|---|---|
| starter | 1430 | 132 | 0.175 | |
| v0.3 | 2187 | 22 | 0.630 | rank 2 of 58 when submitted |
| **v5** | **2226** | **16** | 0.671 | best entry; about 10th team on the final board |
| v7 | 2151 | 28 | **0.696** | |

PCS is the leaderboard's Policy Capability Score; all numbers are from the board of Oct 31.

v7 improved the average scene score, the distance between at-fault incidents and the distance to the recorded path, but its PCS was lower than v5's. PCS is an item-response (zero-one-inflated beta) fit over every team's per-scene scores, and we had validated only the mean scene score. We finished outside the top 3 needed for the finals.

The technical report was written (4 pages, official template) but not submitted, by Aman's decision.

---

## 2. The challenge (rules that shaped every decision)

- **Task:** closed-loop driving in NVIDIA's AlpaSim simulator.
  - The PAI track renders 987 rollouts from neural reconstructions (NuRec) of real PhysicalAI-AV drives, using NVIDIA's NRE renderer.
  - A driver container receives camera frames, ego poses and a route, and returns trajectories. A controller then tracks those trajectories in physics.
- **Scoring:**
  - Per-scene score: 0 for an at-fault collision, offroad or a corridor exit (more than 4 m from the recorded path). Otherwise progress-based, saturating at 80% of the recorded distance.
  - The leaderboard ranks by **PCS**: a zero-one-inflated beta item-response model fitted jointly over all submissions' per-scene scores and mapped to an anchor scale. It is re-fitted as entries arrive, so a submission's PCS moves over time.
  - Ties break on at-fault incident spacing.
  - The top-3 PAI entries get final reruns with the renderer's diffusion "Harmonizer" ON. The board's initial runs have it OFF.
- **Contract:**
  - The image is pushed to the team's ECR repo.
  - Each driver instance uses at most 16 GiB of VRAM.
  - The PAI throughput limit is 14,400 s of wall time for the whole run on H100s, with 4 replicas per GPU.
  - Submissions are under Apache-2.0, plus a broad licence to NVIDIA.
- **Quota:**
  - 3 official submissions per UTC calendar month, shared across tracks; unused slots expire.
  - Warmups: 3 per track per rolling 30 days.
  - A failed submission still counts.
- **Deadlines:**
  - PAI submissions: Oct 31 AoE (CLI: 2026-11-01T12:10Z).
  - Report: originally Nov 17. The challenge page later listed it with the PAI deadline.
  - nuPlan was extended to Nov 16. That track belonged to a separate thread (`alpasim-jepa`).

---

## 3. Infrastructure we built

### 3.1 Running the official simulator on a university cluster (Columbia Insomnia, Slurm)

The cluster had no docker, no enroot/pyxis and no subuid mapping. What we did:
- **Container conversion:** wrote `src/compose_apptainer.py`, which turns the simulator wizard's docker-compose file into Apptainer launches.
- **Sandboxes:** unpacked the images once into `sandbox/` (alpasim-base 20 GB, NRE 27 GB), because without squashfuse every `exec *.sif` unpacked 10 GB into RAM.
- **Launch fixes:**
  - the apt sandbox user, so unprivileged builds work;
  - Hydra's working directory;
  - Apptainer ignoring WORKDIR (`--pwd`);
  - compose bind mounts that hid compiled protos;
  - quoted env values;
  - host-backed writable caches.
- **Renderer on old NVIDIA drivers:** NRE 26.04 needs CUDA 12.8 (driver ≥ 570), but the L40S/H100 nodes ran driver 550. We first ran PAI on the Blackwell nodes. Then we shipped NVIDIA's `cuda-compat-12-8` package into every simulator service and bound Slurm's `TMPDIR` into the containers. That second step was the real blocker, because Warp's NVRTC compile needed a writable temp dir.
- **Per-job port blocks:** two co-located runs once silently shared gRPC ports through SO_REUSEPORT, mixing drivers and controllers between runs. We found this from session counts, invalidated the affected runs and gave every job its own port block derived from its job ID.
- **Storage:**
  - raw logs on node-local scratch, copied back without `.asl` files;
  - `LOCAL_FS=1` stages scene files node-local (each NuRec scene is ~1.8 GB);
  - a shared uv wheel cache cut the per-job env build from ~2,400 s to seconds;
  - a node-local rebuild of the Python env, byte-equivalent to the old one (trajectories matched to 0.000000 m), after another session deleted the env we borrowed.
- **Health checks:** each run writes `driver_health.txt` (fallback steps, INCOMPLETE flag), runs driver unit tests before starting, and traps TERM/USR1 to save partial outputs.
- **Determinism discovery:** with the Harmonizer off, closed-loop runs on one GPU type repeat bit for bit. The ±0.05 per-batch noise we had fought for weeks came from the Harmonizer. This made exact paired comparisons possible.
- **Insomnia H100 fault:** "CUDA unspecified launch failure" lost 10 of ~35 H100 batches. It was specific to this cluster's driver, as the official evaluator ran everything cleanly. We moved screening to L40S/R6000 and showed L40S agrees with H100 (A/A −0.002 [−0.020, +0.016]).

### 3.2 The driver (`src/a15_driver`)

- **Model:** Alpamayo 1.5 (Qwen3-VL-8B backbone plus a 2.3B flow-matching action expert, about 11B parameters).
- **Memory, measured on L40S (peak GiB / latency per plan):**

| Config | Peak GiB | Latency |
|---|---|---|
| bf16 | 22.1 | 0.75 s |
| **int8 weight-only (torchao)** | **15.6** | **0.88 s** |
| fp8 | 15.9 | 1.41 s |
| int4 | 12.4 | 3.18 s, worse ADE |

  int8 cost no measurable accuracy (ADE 1.96 vs 1.98).
- **Headroom:** the token-embedding table on CPU saved 1.2 GB, giving about 14.2–15.2 GB in closed loop.
- **One GPU thread:** every model call runs on one dedicated thread. On H100, gRPC worker threads each built their own CUDA state, and memory climbed 14.6 → 17.3 GB.
- **Reasoning off:** the chain-of-causation text cost −0.165 scene score.
- **1 Hz replanning:** 2 Hz was −0.045 to −0.097, and 0.5 Hz was −0.072 on nuPlan.
- **Cached plan between replans,** re-anchored to the current pose.
- **Inputs:** 4 cameras × 4 frames at 576×320. The input contract was verified against NVIDIA's loader.
- **Speed of the wave:** on the official warmup, model work was only about 16 s per 32-rollout wave out of ~540 s. Rendering dominates.

### 3.3 Evaluation bench

- **Screening set:** 120 curated-val PAI scenes. Paired per-scene comparison with bootstrap 90% intervals (`src/compare_runs.py`).
- **Failure taxonomy** (`src/failure_taxonomy.py`): at-fault, offroad, corridor, low progress, pass. Plus per-timestep metrics and BEV/camera videos.
- **Held-out sets (from Oct 24):** 321 never-used scenes, made of a fresh 120 plus a 201-scene lockbox (seeded split).
- **OFFICIAL_MIMIC=1:** Harmonizer off and `skip_driver_during_force_gt=true`, exactly as in the official initial runs.
- **Calibration of local to official:**
  - v0.3: local H100 0.657 vs official 0.611.
  - v5: local gain +0.071 vs official +0.041 (~0.58×).
  - v7: local +0.057 vs official +0.025 (~0.44×), and PCS fell.

---

## 4. Timeline

### Week 0 (Sep 26–29): feasibility to rank 2
- One-day feasibility test passed: the simulator runs on Insomnia, the submission pipeline works, and Alpamayo 1.5 int8 fits 16 GiB.
- **Starter official** (straight line at current speed): PCS 1491, rank 38/52. It used 9,250 of 14,400 s, so 64% of the time budget is pure simulator overhead.
- **v0.2 (2 Hz) warmup:** 552 s per wave. A naive extrapolation said it would exceed the limit. Fitting a second warmup (v0.3 at 1 Hz, 536 s) showed nearly all the time is fixed startup, giving a ~10,000 s estimate. The actual was 9,474 s.
- **Sprint:** reasoning −0.165, K=5 medoid −0.082. The port-contamination incident invalidated three variants. A/A noise was ±0.03 at 40 scenes, and no lever beat it.
- **v0.3 official:** PCS 2143, **rank 2 of 58**, scene 0.611, and the best at-fault spacing in the top 6. nuPlan v0.3: PCS 1296, rank 19/71.

### Week 1 (Sep 29 – Oct 4): bench, failure taxonomy, route chooser
- **Taxonomy** on 120 PAI scenes: corridor exits are more than half of the lost score (24.0 of 43.4). Low progress comes second (10.4).
- **Corridor exits:** on PAI these are mostly lane choices on straight roads. On nuPlan they are junction branches.
- **Low progress:** mostly not-at-fault rear-ends. The car falls behind the recorded pace and the replayed follower hits it.
- **Route chooser** (K diffusion samples; drive the one closest to the route): nuPlan +0.028.
  - PAI v1: +0.013 (noise).
  - PAI v2, with a 1 m margin and a speed guard: **+0.054 [+0.005, +0.105]**, the first part to clear noise.
  - v3, with a graded guard: +0.007 over v2.
- **Plan-storage bug:** chooser logging accidentally stopped plans being stored. We caught it from the plan counts, voided the affected runs and added a 16-combination regression test.
- **Zero effect:** route text v1–v3 (which also fixed a distance bug), diffusion temperature 0.6, early replanning.
- **Speed scaling ×1.15:** −0.307.
- **Scope change (Sep 30):** this repo became PAI-only. nuPlan went to the `alpasim-jepa` thread.

### Week 2 (Oct 5–12): rear-ends, six cameras, route snap
- **Six cameras** (adding the rears): rear-ends 25 → 11, but own failures rose, for net −0.023. A hybrid that took the path from 4 cameras and the speed from 6 scored −0.052.
- **Lesson: speed is not a free lever.** Keeping pace trades rear-ends for our own crashes.
- **Analysis:** in 11 of 14 remaining corridor exits, all 6 model samples kept the lane, so no chooser could fix them. The recorded driver changed lane or took an exit.
- **Route snap:** when the plan deviates more than 2 m from the route, replace its geometry with a cubic Hermite onto the route, keeping the model's speed profile.
  - Unguarded, it broke turns.
  - **The 15° parallel guard** (snap only when the route is near-straight) kept 6 of 8 fixes and removed the turn breakage.
  - H100, 120 scenes: **+0.053 [+0.004, +0.101]**; corridor exits 16 → 9.
- **Not better:** 1.5 m threshold, a min-offset rule, chooser margin 0.5.
- **Hardware differences:** Blackwell A/A within 0.005 (deterministic there). H100 reads about 0.03 lower than Blackwell, and the chooser's gain vanished on H100 (+0.005).

### Week 3 (Oct 13–21): v5, waypoint memory, fine-tune pipeline
- **v5 = v0.3 + guarded snap** (K=1): H100 **+0.071 [+0.011, +0.131]** over v0.3.
- **Waypoint-memory snap:** remember earlier routes so turns can be followed. Four variants: v1 buggy, v2, v3 with a curvature speed cap, and geometry-only. All tied with or lost to v5. Dropped.
- **Env loss:** `nureasoning/ext/a15env` was deleted by another session's approved cleanup. We rebuilt it node-local and proved it equivalent (trajectories identical to 0.000000 m).
- **v5 official (Oct 21):** PCS 2184, scene 0.652, **4th team**. The local gain converted at ~0.58×.
- **Fine-tune pipeline:**
  - capture mode forces the ego onto the recording and saves 4-camera samples every 0.5 s;
  - a DAgger mode lets the policy drive and labels each state with a smooth path back to the recording;
  - the trainer is adapted from NVIDIA alpamayo-recipes.
- **bf16 AdamW silently dropped updates** at lr 1e-5: 50 steps of −1e-5 left 0.02 at 0.02002. We wrote SRAdamW (stochastic rounding) with a unit test that runs before every training job.
- **Forced `<traj_future_start>`:** the model skipped the token in 30–42% of closed-loop plans. Forcing it was neutral alone, but it aligned inference with training.
- **Runs A/B** (force-GT data only): open-loop ADE −21% / −24%, but neutral closed loop. Better imitation on recording states does not fix closed-loop drift.
- **Rejected:** time warp (−0.121), fast chooser (−0.046). Stall rescue was neutral.

### Week 4 (Oct 22–25): run C, weight averaging, route text, held-out confirmation
- **Run C** (force-GT ~7.8k + DAgger 1.4k samples): L40S +0.040; a replicate gave +0.047.
- **WiSE-FT averaging,** 0.5 base + 0.5 C (`ftC_w50`): 0.742. It restored pace and halved at-fault collisions.
- **Research sweep** (4 parallel agents) found RoaD (NVIDIA's closed-loop SFT), Alpamayo 2-Super, the Harmonizer-on finals, the innovation prize, FlashDrive, the exact scoring code and the controller gains.
- **w50 + route text:** **+0.084 [+0.027, +0.142]**, the first significant fine-tune result. Route text did nothing with the released expert but helped with the fine-tuned one, so **components interact**.
- **Rejected or not adopted:**
  - run C seed 1: 0.686;
  - run D (more DAgger);
  - soup C+C2;
  - R (RoaD) and T (A2 teacher), both tested under Harmonizer noise and later retested;
  - controller idx_start_penalty 5 and 15, i.e. both directions from the default 10;
  - CFG on route text: −0.03;
  - SNAP_MIN_T: neutral.
- **"Deterministic gains only" rule** (Aman, Oct 24), and held-out splits created.
- **Fileset quota incident (my fault):** new-scene runs fetched 150 GB of scenes into shared storage. Fix: `LOCAL_FS=1`.
- **v7 image bug caught before any push:** the expert file was root-only readable, which would have scored zero. Fixed, and the rule became: run image checks as the runtime user.
- **Held-out verdict:** v5 → v7 on 321 scenes, **+0.057 [+0.023, +0.093]**, W/L 65/48. Corridor exits 37 → 22, at-fault 9 → 11.
- **Finals condition** (Harmonizer on, 2 runs): v7 ≈ v5 (+0.006).
- **Also rejected:** run E, run N (nav-conditioned fine-tune), 2 Hz, 20 flow steps.

### Week 5 (Oct 26–31): teacher, rental, audit, push
- **Wait-then-go start rule:** when stopped, a plan that keeps postponing the start for 3 plans in a row has its waiting prefix dropped. Held-out 321: **+0.0037 [+0.0001, +0.0089], 4 wins / 0 losses**. Adopted.
- **T2 (Alpamayo 2-Super teacher + RoaD):**
  - v7 drove 240 unused training scenes, giving 2,466 states with 16 own samples each.
  - Labelling 2,466 states with the 34B teacher (K=8, route text) took about 55 minutes on a rented 4× RTX PRO 6000 (RunPod, ~$12).
  - Finding: the teacher is no closer to the recording than v7 on v7's own states (0.83 vs 0.82 m per sample). Its value is diversity.
- **T2b:** held-out +0.005, not significant. **The audit then found T2b was under-trained** (flat loss, my lr choice).
- **Fair-dose retests:**

| Run | What | Result |
|---|---|---|
| T2c | teacher + RoaD | −0.020 |
| T2d | outcome-weighted | −0.021 |
| R2 | RoaD | −0.058 |
| Cs2, Cs3 | seeds 2 and 3 of run C's recipe | −0.034, −0.026 |
| seed soup | average of the C seeds | +0.003 |
| lateral chooser | NVIDIA's CLOSEST_LATERAL | −0.073 |

- **Pattern:** every fine-tune improved pace but added net zero-score lane/branch flips.
- **Push (Oct 30, supervised):** digest `sha256:05e650bd…`.
- **Warmup:** 32/32 rollouts, 539 s per wave.
- **Official v7 (Oct 31):** 987/987 rollouts, 9,535 / 14,400 s. Scene score 0.696 (v5 0.671), but PCS 2151 vs v5's 2226.

---

## 5. Final system (v7)

| Part | Setting | Evidence |
|---|---|---|
| Model | Alpamayo 1.5 10B, int8 weight-only, CPU embedding, one GPU thread | fits 16 GiB, no accuracy loss |
| Replanning | 1 Hz, cached plan re-anchored | 2 Hz and 0.5 Hz worse |
| Route snap | 2.0 m, 15° guard, cubic Hermite, model's speed | H100 +0.053; official +0.041 (v5) |
| Forced start token | `A15_FORCE_TRAJ_START=1` | token missing in 30–42% of plans |
| Route text | "Turn left/right in X m", intersect distance | +0.035 on top of w50 |
| Action expert | 0.5 released + 0.5 run C (force-GT + DAgger) | held-out +0.057 with the above |
| Start rule | `A15_WAITGO_K=3`, far 5 m | held-out +0.004, 4/0 |

Fine-tuning details:
- Only `expert.*`, `action_in_proj`, `action_out_proj` (2.28B of 11.08B params); VLM frozen.
- Teacher-forced flow-matching MSE, 4 noise draws per VLM prefill.
- SRAdamW (betas 0.9/0.95, wd 0), lr 3e-5, 20-step warmup, grad clip 1.0, accumulation 8, about 2 epochs (~2k steps), on 48 GB GPUs (~42 GB peak).

---

## 6. Numbers worth remembering

- Local held-out v5 → v7: +0.057 [+0.023, +0.093] over 321 scenes. Official: scene +0.025, PCS −75.
- Official v0.3 → v5: scene +0.041, PCS +34 at the time.
- Throughput: starter 9,250 s; v0.3 9,474; v5 9,522; v7 9,535 (limit 14,400). Warmup waves: 552 / 536 / 539 s.
- Memory: int8 15.6 GiB standalone; closed loop 14.2–15.2 GB; H100 thread growth 14.6 → 17.3 GB before the fix.
- Experiments: about 60 closed-loop variants, 14 expert fine-tunes (A, B, C, C2, D, E, R, T, N, T2a–d, R2, Cs2, Cs3) and 6 official submissions.
- Data: ~7.8k force-GT samples (400 train scenes), 3.2k DAgger (180 scenes), 1.1k on-policy DAgger, 14.8k with route text, and 2,466 T2 states labelled by the A2 teacher.
- Compute: Insomnia L40S/H100/R6000/A6000. Rental: ~$12 for teacher labelling.

---

## 7. What went wrong (honest list)

1. **Optimised the wrong target for the last decision.** PCS rewards which scenes change, not the mean. We had per-scene data for dozens of our runs and could have fitted a local IRT proxy. We didn't. The "+28 PCS" projection was never tested.
2. **Verdicts on weak tests:**
   - T2b was under-dosed;
   - run T had only 30% teacher labels;
   - R and T were "rejected" inside Harmonizer noise.
   Each correction came only after Aman probed.
3. **The research method grew outward from the current approach** (choosers, snaps, fine-tunes of Alpamayo) instead of starting from the problem. See `../LESSONS_FOR_AGENTS.md`.
4. **Infrastructure slips:**
   - port contamination;
   - the plan-storage bug;
   - a Slurm singleton dependency that idled a lane for 5 days;
   - filling the shared quota with scene downloads;
   - a socket-gate slip (a piped check that didn't gate);
   - reading "Slurm down" when `source /etc/profile` was missing.
5. **Noise:** weeks of Harmonizer-on comparisons at ±0.05 per batch before we found that Harmonizer-off runs are deterministic.

---

## 8. Raw material for resume bullets (pick and edit)

- Built an end-to-end closed-loop driving entry for NVIDIA's AlpaSim challenge (NeurIPS 2026), using a 10B vision-language-action model (Alpamayo 1.5). It reached **rank 2 of 58** at first submission and finished with a best PCS of 2226 among 150+ entries.
- Fitted a 10B VLA into a 16 GiB per-instance VRAM contract with int8 weight-only quantization (torchao), CPU-resident embeddings and single-thread CUDA execution. Accuracy loss was zero, memory 14.6 GB, and the whole 987-rollout evaluation ran in 9.5k of the 14.4k-second budget.
- Ported NVIDIA's closed-loop simulator stack (docker compose, neural renderer, physics, gRPC driver) to a Slurm/Apptainer HPC cluster without docker. Solved CUDA forward-compatibility, port isolation, node-local staging and reproducible environments, and found that one renderer setting makes closed-loop evaluation bit-for-bit deterministic.
- Designed a route-snap trajectory correction (cubic Hermite onto route waypoints with an angle guard) that cut corridor exits 44% and raised the official scene score 0.611 → 0.652.
- Fine-tuned the 2.3B flow-matching action expert on simulator-rendered data with DAgger recovery labels and WiSE-FT weight averaging. Also found and fixed silent bf16 AdamW update loss with stochastic rounding. Gain: +0.057 [+0.023, +0.093] mean scene score on 321 held-out scenes.
- Distilled from a 34B teacher (Alpamayo 2-Super) with RoaD-style closed-loop targets on rented multi-GPU hardware. The analysis showed the teacher added diversity, not accuracy, on the student's own states.
- Ran a paired, held-out evaluation protocol (deterministic simulation, bootstrap intervals, lockbox split) across about 60 closed-loop variants and 14 fine-tunes.

Keep the PCS outcome honest if asked: v7 improved every visible component metric, but the ranking metric fell, because we validated against the mean rather than the item-response model.

---

## 9. Study list (what this campaign touched)

- **Models:** VLA architectures (Qwen3-VL + flow-matching action expert); flow matching and Euler sampling; classifier-free guidance; reasoning traces in driving VLAs.
- **Fine-tuning:**
  - teacher-forced flow-matching loss;
  - bf16 numerics and stochastic rounding;
  - WiSE-FT and model soups;
  - DAgger, RoaD (closed-loop SFT) and outcome-weighted self-imitation;
  - knowledge distillation from larger teachers.
- **Quantization:** int8/fp8/int4 weight-only (torchao), and memory accounting (torch allocator vs CUDA context vs per-thread state).
- **Evaluation:**
  - paired bootstrap;
  - A/A tests;
  - winner's curse and lockbox splits;
  - determinism and noise sources;
  - **item-response theory and zero-one-inflated beta models (how PCS works)**;
  - open-loop vs closed-loop metrics.
- **Systems:** Slurm (MaxMemPerCPU, dependencies), Apptainer, CUDA forward compatibility, gRPC, ECR, node-local staging.
- **Driving:** log-replay artifacts (replayed followers causing rear-ends), lane-change and branch decisions, controller/MPC gains, route representations.

---

## 10. Talking points (how to tell the story)

1. **Engineering under a contract:** a 40 GB model running in 16 GiB, on a cluster that couldn't run the official stack, inside a fixed time budget. All met on the first official try.
2. **Measurement is the product:** most early "gains" were noise. Finding the determinism switch, and building held-out lockbox confirmation, changed what we could trust.
3. **What actually moved the score:** geometry (route snap) and training on recovery states (DAgger), not larger samplers, reasoning text or speed hacks.
4. **The hard lesson:** v7 was better on every number we could see and worse on the one that ranked us. Know exactly how the leaderboard aggregates, and build a local proxy for it before the last decision.
