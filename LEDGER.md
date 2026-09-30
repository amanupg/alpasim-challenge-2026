# AlpaSim ledger

Measurements and decisions, newest at the bottom. Numbers are only written here once verified.

## 2026-09-27 Feasibility test (Aman: "go ahead with the 1-day feasibility test")

Questions:
1. Can the simulator run on Insomnia?
2. Can Alpamayo 1.5 meet the 16 GiB and latency contract?

### Access and data (verified)
- **HF account `amanupg`:**
  - Can list `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`, but downloading its files gives **403 GatedRepoError**. Aman must accept the dataset terms on HF before the PAI track can be run locally.
  - `OpenDriveLab/AlpasimChallenge2026_nuplan_track` is open. Total 491.5 GB: 15 MTGS asset shards of 29-35 GB each, trajdata cache 2.17 GB, configs.
- **NRE renderer image** `nvcr.io/nvidia/nre/nre-ga:26.04`: anonymous pull works (manifest HTTP 200). 14.29 GB compressed, 41 layers.
- **Insomnia tooling:** no docker, no enroot/pyxis (`srun --container-image` unavailable); `apptainer/1.4.5` module; rootless podman; no `/etc/subuid` entry for au2327.
- **Shared storage:** the `/insomnia001` filesystem has 1017 GB free of 5.0 TB. The full nuPlan navtest set (491 GB) would take half of that, so only fetch subsets.

### Newer models (rule: check before using any model)
- **`nvidia/Alpamayo2-Super`** was released on HF on 2026-08-04 and is not gated.
  - 32B backbone (Cosmos 3 Super Reasoner) plus a 2.3B action expert; 71.6 GB of weights.
  - Measured peak 72 GB on an H100 at bf16.
  - Reported: AlpaSim score 1.50 ± 0.13 on 913 NuRec scenarios; minADE_6 at 6.4 s of 0.911 m on 1434 PhysicalAI-AV samples.
  - Needs transformers ≥ 4.57.1 and torch ≥ 2.8.
- **For AlpaSim it cannot fit the 16 GiB cap:** even 4-bit weights are about 17 GB. At most it's a teacher for distillation.
- No smaller Alpamayo 2 exists. NVIDIA's list: Alpamayo-R1-10B, Alpamayo-1.5-10B, Alpamayo2-Super.
- **Relevance to nuReasoning** (open loop, no VRAM cap) is recorded in the nuReasoning LEDGER.

### Upstream facts that shape the plan
- The `e2e_challenge` branch (commit 2023534, 2026-09-18) ships an Alpamayo 1.5 driver: `src/driver/.../alpamayo1_5_model.py`.
  - Config: bf16, 4 cameras × 4 frames, batch 1.
  - The config comments say "~40 GB standard", with reasoning text generated before every trajectory. That's too big for the 16 GiB cap as-is.
- The wizard can deploy via docker compose or via SLURM with Pyxis/enroot. On Insomnia we use `run_method=NONE` (it writes `docker-compose.yaml`) plus our own compose-to-Apptainer launcher.

### Jobs
| Job | What | Status |
|---|---|---|
| 12946961 | pull NRE SIF | running |
| 12946962 | fetch nuPlan smoke subset (cache + configs + shard 001, streamed extract) | running |
| 12946964 | build `alpasim-base.sif` (retry: apt sandbox fix) | queued (QOS CPU cap) |
| 12946966 | `a15_fit`: Alpamayo 1.5 memory/latency/quality × {bf16, int8, fp8, fp8-dyn, int4} × {reasoning, no reasoning}, 1 H100 | queued (QOS CPU cap) |

The first base-image build failed. Apptainer's unprivileged build (no subuid, so a "root-mapped namespace") cannot run apt's `_apt` sandbox. The fix is `-o APT::Sandbox::User=root`, plus `set -e` in `%post`.

### Result 1: the official closed-loop simulator runs on Insomnia (verified 2026-09-27)
- **How:** Apptainer sandboxes plus our `src/compose_apptainer.py` in place of docker compose.
- **Test:** nuPlan dev preset, one scene, starter-kit driver (straight line at current speed). Job 12947998 on an L40S (ins057), 4 min 36 s wall.
- **Result:** scene `2021.05.25.14.16.10_veh-35_01100_01664-f02b15cc225b5d9a`:
  - PASS, score 1.0;
  - collision_at_fault 0, offroad 0, left_corridor 0;
  - progress_rel 0.953; dist_to_gt_trajectory 0.15 m; 56.9 m driven;
  - 10 Drive calls, mean 1.2 ms.
- **Fixes needed on the way:**
  1. **No squashfuse**, so `exec *.sif` unpacks 10 GB into node RAM each time. We unpack once into `sandbox/` instead: alpasim-base 20 GB, nre 27 GB. `--writable-tmpfs` is ignored on GPFS; harmless.
  2. **Hydra writes to the working directory:** the wizard now runs from the run directory.
  3. **Apptainer ignores Docker WORKDIR:** `--pwd /repo`.
  4. **Compose bind-mounts `/repo/src`,** which would hide the protos compiled into the image: skipped.
  5. **The nuPlan preset pins `TORCH_CUDA_ARCH_LIST=8.9;9.0+PTX`,** so the MTGS renderer (gsplat, compiled at startup in about 250 s, cached in `~/.cache/torch_extensions`) fails on an A6000 (sm86) with "no kernel image". Use L40S or H100.
- **Still untested locally:** the PAI track (NRE renderer), which is blocked on the gated NuRec scenes.

### Correction: official submission quota is per CALENDAR MONTH (2026-09-27)
- **Earlier notes said "3 per rolling 30 days" for official submissions. That's wrong.** Only warmups are rolling-30-day.
- **Evidence:**
  - The CLI's `limits` prints "Monthly submission limit / Submitted this month / Remaining this month".
  - NVlabs/alpasim#171: teams losing "remaining August submissions" before month end, with the "September submission limit reduced to 3".
  - The organiser in #170/#158: "3 per month"; soft-open allocations were reset.
- **Consequence:** the 3 September slots expire at month end (Sep 30), and October brings 3 more. Unused slots don't carry over.
- **Other facts from the upstream issues:**
  - #185: warmups were wrongly counted against the official quota; fixed 2026-09-18.
  - #158: a failed submission (typo'd image) still counts. The CLI now checks that the image exists first.
  - #170: one team measured 25% of the time budget locally with 4 replicas on a similar GPU, but the server reported ~80%. So local latency underestimates the server's throughput accounting; only the server's "observed throughput wall time vs limit" is authoritative.
  - #166: PAI rank = IRT posterior rank interval (better upper bound first), tie-break on at-fault incident spacing. Hard braking inflates incident spacing but not scene score. The top ~5 PAI entries get a final rerun.
  - #188: the terms require submissions under Apache 2.0, plus a perpetual, broad, sublicensable license to NVIDIA (including making submissions open source). Teams that don't accept get their submission data deleted at month end. Accepting is Aman's decision.
  - The Space says PAI evaluations use the larger final suite and can take ~10x longer.

### Result 2: Alpamayo 1.5 against the 16 GiB contract (job 12947478, L40S, 2026-09-27)
- **Setup:**
  - Inputs: 4 camera slots × 4 frames, with the nuReasoning front camera also standing in for the telephoto; 1 sample, seed 42.
  - Clips: nuReasoning validation key frames (frame 100). ADE/FDE are against the logged 6.4 s future.
  - Quantization: torchao 0.13 on 7.40B params (language model + expert linears). Vision, embeddings and lm_head stay bf16.
- **Results** (latency per plan on the L40S; the H100 is faster):

| Config | Peak GiB | Latency s | VLM s | Diffusion s | ADE (same 30 clips) | FDE |
|---|---|---|---|---|---|---|
| bf16, reasoning (~17 tok) | 22.1 | 1.15 | 0.90 | 0.24 | 3.14 | 7.46 |
| bf16, no reasoning | 22.1 | 0.75 | 0.50 | 0.24 | 1.98 | 6.55 |
| **int8 weight-only, no reasoning** | **15.6** | **0.88** | 0.57 | 0.30 | **1.96** | 6.49 |
| int8 weight-only, reasoning | 15.7 | 1.60 | 1.28 | 0.31 | 3.14 | 7.45 |
| fp8 weight-only, no reasoning | 15.9 | 1.41 | 1.08 | 0.33 | 1.97 | 6.50 |
| int4 weight-only, no reasoning | 12.4 | 3.18 | 2.35 | 0.83 | 2.26 | 7.64 |
| fp8 dynamic activation+weight | 16.0 | 52-75 | - | - | 1.98 | - |

- **fp8 dynamic** apparently took a fallback path (tens of seconds per plan); discarded.
- **"No reasoning"** closes the reasoning block in the prompt (`<|cot_start|><|cot_end|>`). The model emits `<traj_future_start>` itself in 87% of cases; the rest fall back to the last token, with similar ADE (1.82 on those clips).
- **Reasoning hurts here:** on the same clips it gives worse open-loop ADE (3.14 vs 1.98). Single sample at temperature 0.6; closed-loop effect unknown.
- **int8 costs nothing measurable in accuracy** (1.96 vs 1.98). int4 costs about 0.3 m ADE and is slower (unoptimised kernels on the L40S).
- **Memory is tight:** int8 peaks at 15.6 GiB plus the CUDA context, about 0.5 GiB outside PyTorch. Headroom plan: int8 embeddings and lm_head (about 1.3 GB bf16 saved), and keep the 4-camera input.

### Result 3: first server warmup SUCCEEDED (2026-09-27)
- Submission `951885e7-86f7-4607-83a9-af8ca7c6617e`, track nuplan-warmup, image `teams/amanupg:v0.1-starter` (starter kit, 100 MB, digest `5d93a152…`).
- 32/32 valid scored rollouts; evaluation 21:56:19 to 22:02:23 UTC.
- Throughput: observed 269 s vs limit 2700 s (passed). A near-zero-cost driver shows about 270 s of pure simulator overhead per 32-scene wave.
- Warmups aren't scored and their limit isn't the official one, so this doesn't settle the official budget.
- **Accounts:**
  - Challenge: HF `amanupg-columbia`, team `amanupg`, Aman is captain; terms `official-2026-08` accepted.
  - Cluster token for AlpaSim: `alpasim/.cache/hf/token`. NuRec downloads verified (2.0 GB sample scene).

### Verdict of the one-day feasibility test: GO
- The simulator runs locally on Insomnia (nuPlan verified; PAI data is now accessible).
- The submission pipeline works end to end.
- Alpamayo 1.5 fits 16 GiB at int8 with no loss in open-loop accuracy.
- **Open risk:** the official throughput limit with 4 replicas per H100. The first official submission reports it.

### Result 4: container driver in local closed loop, nuPlan (2026-09-27)
- **Driver:** `src/a15_driver` (Alpamayo 1.5, int8 weight-only on 7.70B params, no reasoning, replan every 0.5 s, cached plan in between). Run natively from the nureasoning a15env plus AlpaSim overlays: the same code that goes in the container.
- **Token embedding table on CPU** (`A15_EMBED_CPU=1`, default):
  - trajectories identical (max diff 0.000000 m, 6 clips);
  - nvidia-smi process memory 15,456 → 14,236 MiB standalone (job 12950505), 14,528 MiB peak in closed loop.
  - That leaves ~1.8 GiB margin to the 16 GiB (16,384 MiB) cap.
- **Same 30 nuPlan navtest scenes** (shard 001), 2 concurrent rollouts, L40S:

| Driver | Passed | Mean scene score | At-fault coll. | Offroad | Left corridor | Progress | Dist to GT |
|---|---|---|---|---|---|---|---|
| starter (straight, current speed), job 12950496 | 23/30 | 0.687 | 0.03 | 0.07 | 0.13 | 0.80 | 2.10 m |
| a15 int8 no-reasoning, job 12950497 | 23/30 | 0.728 | 0.03 | 0.17 | 0.07 | 0.86 | 1.37 m |

- **Timing:** 300 Drive calls with an inference on every call (nuPlan calls at 2 Hz); mean 0.83 s; 0 inference errors, 0 fallbacks.
- **Caveats:**
  - nuPlan is off-domain for Alpamayo: Motional rig, CAM_L0/F0/R0 mapped to slots 0/1/2, frames 0.5 s apart instead of 0.1 s, no telephoto.
  - 30 scenes is a small sample.
- **Image `v0.2-a15int8`** (built on the Mac, linux/amd64, 21.5 GB):
  - verified offline inside the image: vocab 155697 / traj offset 151669; frames resize to 576x320;
  - starts read-only with no network; the model needs a GPU.
  - Pushing to ECR.

### Result 5: first OFFICIAL submission, PAI track, starter image (September slot 1 of 3)
- **Submission:** `0f1f124a-ca30-4804-aebb-db10cff56267`, image `v0.1-starter` (straight line at current speed). Evaluated 2026-09-27 22:08 to 2026-09-28 01:11 UTC.
- **Coverage:** 987/987 valid scored rollouts.
- **Leaderboard:** PCS **1491** (interval 1440-1543), rank 38/52. Average scene score **0.140**; at-fault incident spacing 0.143; dist_to_gt 2.16.
- **Above** every VAVAM entry (PCS ~1013-1082). **Top:** jongam 2135 (scene score 0.561). **Host "internal" (Alpamayo 1 reference):** 2000 (scene score 0.560).
- **Throughput:** observed **9,250 s of the 14,400 s limit** (margin 5,150 s, passed).
  - A near-zero-cost driver already uses 64% of the PAI budget; our whole model cost must fit in about 5,150 s.
  - Rough estimate for the int8 Alpamayo at 2 Hz: about 40 plans per rollout × 987 ≈ 39.5k plans over 4 H100s × 4 replicas. At ~10 plans/s aggregate that's ~3,950 s. It fits, but not with much room.
  - **Unknown:** H100 throughput with 4 replicas contending per GPU. A PAI warmup reports one 32-rollout wave's wall time; compare with the starter's ~300 s/wave (9,250 s / ~31 waves) and extrapolate.
  - **Levers if tight:** replan at 1 Hz, CUDA graphs for the diffusion expert (they collapsed the diffusion time in the fp8 test), drop the telephoto camera.

### Result 6: Alpamayo image on the real evaluator, pai-warmup (2026-09-28)
- **Submission:** `ff87a2af-5ebf-401a-ae0c-dafbe9b0361a`, image `teams/amanupg:v0.2-a15int8`, digest `sha256:877117daffdbc5f9fd4299d22fa1d048504b6e2d97083ca3f080a77b740b1e0b`.
  - Config: int8, no reasoning, CPU embedding, replan every 0.5 s.
  - Queued ~3 h; evaluation 04:26:19 to 04:44:59 UTC.
- **Result:** SUCCEEDED, 32/32 valid scored rollouts. So the image loads, fits and drives on the evaluator's H100s with 16 replicas.
- **Timing:** observed **552 s** for one 32-rollout wave (warmup limit 2,700 s).
- **Extrapolation to the official PAI run** (ESTIMATE, not measured):
  - Starter official: 9,250 s / (987/32 ≈ 30.8 waves) ≈ 300 s per wave. The Alpamayo wave is ~250 s longer.
  - If none of the 552 s is one-off startup: 9,250 + 30.8 × 252 ≈ **17,000 s > 14,400 s limit**.
  - If ~120 s of it is startup (16 replicas loading the 21 GB model), it's ≈ 13,300 s: just under, too thin to risk a slot.
- **Implied effective cost:** ~250 s / (2 rollouts × ~40 plans per replica) ≈ 3 s per plan under 4-replica contention per H100, vs 0.83 s alone on an L40S.
- **Decision:** don't spend an official slot on v0.2 at 2 Hz. Next is v0.3:
  - replan at 1 Hz (halves the model load);
  - CUDA graphs for the diffusion expert, if exact-equivalent (0.30 of the 0.83 s is diffusion);
  - measure 1 Hz vs 2 Hz driving quality locally on the same PAI scenes before choosing.

### Local PAI simulation on Insomnia: renderer driver requirement (2026-09-28)
- **Launcher fixes along the way:**
  - per-image working dir (NRE has no /repo);
  - host-backed writable dirs for NRE's `ptx_cache` and physics' `/mnt/warp`;
  - strip the shell-style quotes the wizard writes in env values (`WARP_CACHE_PATH='/mnt/warp'` had become a relative path).
- **Blocker:** NRE 26.04 is built on CUDA 12.8 and JIT-compiles kernels (NVRTC). On nodes with NVIDIA driver 550.54 (L40S, H100; CUDA ≤ 12.4) every render fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
- **Measured drivers:** L40S 550.54.14; H100 NVL 550.54.14; RTX PRO 6000 Blackwell (ins046/047) **575.57.08**.
- **Consequence:** local PAI runs go on the R6000 (Blackwell) nodes. nuPlan (MTGS) runs on L40S/H100. The official evaluator is unaffected (its drivers are new).
- The CUDA-graph option for the diffusion expert costs +1.2 GB GPU memory (15,438 vs 14,236 MiB) for -6% latency (0.898 vs 0.956 s). Kept off.

### Result 7: local PAI closed loop, same 10 curated-val scenes (Blackwell node ins047, 2 concurrent rollouts)

| Driver | Passed | Mean scene score | At-fault coll. | Offroad | Left corridor | Progress | Dist to GT |
|---|---|---|---|---|---|---|---|
| starter, job 12952258 | 5/10 | 0.349 | 0.00 | 0.30 | 0.20 | 0.57 | 2.38 m |
| a15 int8, replan 2 Hz, job 12952257 | 8/10 | 0.776 | 0.10 | 0.00 | 0.10 | 0.92 | 3.16 m |
| a15 int8, replan 1 Hz, job 12952259 | 8/10 | **0.800** | 0.10 | 0.00 | 0.10 | 0.95 | 3.43 m |

- **Alpamayo more than doubles the starter's mean scene score on PAI.** Replanning at 1 Hz is no worse than 2 Hz on these scenes, and halves the mean Drive-call time (0.133 vs 0.262 s).
- **Load:** 199 Drive calls and 40 plans per rollout at 2 Hz (20 at 1 Hz), so the budget estimate's "40 plans per rollout" is confirmed.
- **Caveats:**
  - 10 scenes only; local scenes are easier than the hidden set (starter 0.349 here vs 0.140 officially).
  - Renderer harmonizer config differs from the official EC2 preset.
- **Memory flag:** on Blackwell the driver process showed 19.2 GB in nvidia-smi (vs 14.2 GB on L40S). Being measured standalone on H100 (the official GPU) and Blackwell: jobs 12952759 and 12952760.

### Result 8: driver GPU memory by GPU type, and v0.3 image (2026-09-28)
- **Standalone** (`src/check_policy.py`, 4 camera slots × 4 frames, 12 nuReasoning clips, int8 + CPU embedding):

| GPU | Driver | nvidia-smi (process) | Peak reserved | Latency/plan |
|---|---|---|---|---|
| H100 PCIe (official GPU family), job 12952759 | 550.54.14 | 14,620 MiB | 13.41 GiB | 0.784 s |
| RTX PRO 6000 Blackwell, job 12952760 | 575.57.08 | 14,432 MiB | 13.37 GiB | 0.696 s |
| L40S, job 12950505 | 550.54.14 | 14,236 MiB | 13.37 GiB | 0.959 s |

- **Closed loop** (PAI run 12952259): the driver process (the first GPU pid, present before the simulator) peaked at **15,052 MiB**, leaving ~1.3 GB under the 16,384 MiB cap.
  - The 19.3 GB figure reported earlier was NVIDIA's renderer process sharing that GPU (the logger took the max over all pids; now fixed to report the driver pid only).
  - Physics used ~2.0 GB.
- **Image `v0.3-a15int8-1hz`:** same layers as v0.2 except the last.
  - `A15_INFERENCE_INTERVAL_US=1000000`, `A15_GIT_HASH=e7a75d6`, `A15_EMBED_CPU=1`, `A15_CUDA_GRAPH=0` (verified inside the image).
  - Pushing to ECR.
- **v0.3 pushed:** digest `sha256:29a4f9c2f340bf3ee2914df131627eb8895684f2df67d6667c27a817cfebe424`. The push took 2 h 26 min, not minutes.
  - Cause: Docker Desktop's build cache had been cleared, so `pip install torch` re-ran and gave a new layer digest. Every later layer (including the 21 GB weights) changed identity.
  - Fix going forward: build new versions `FROM teams/amanupg:v0.3-a15int8-1hz` so only new code layers upload.

### Result 9: v0.3 pai-warmup, and nuPlan at 1 Hz (2026-09-28)
- **pai-warmup `db872692-56f9-497e-85cb-93ca6595223d`** (v0.3, digest `29a4f9c2…`): SUCCEEDED, 32/32. Observed **535.7 s** vs v0.2's 552.1 s.
  - Halving the plans saved only ~16 s per wave, so almost all of the ~540 s is fixed (16 replicas starting up and loading 21 GB).
  - Two-point linear fit (ESTIMATE): model time ≈ 32 s/wave at 2 Hz and 16 s/wave at 1 Hz; fixed part ≈ 520 s.
  - Official PAI estimate for v0.3: 9,250 (starter) + ~200 (one-off model load) + 30.8 × ~16 ≈ **~10,000 s of 14,400**. The earlier 17,000 s extrapolation wrongly treated startup as per-wave cost.
- **nuPlan, same 30 scenes as Result 4** (job 12953375, L40S): a15 at 1 Hz scored 24/30, mean scene score **0.784**.
  - Compare: a15 at 2 Hz 0.728 (23/30); starter 0.687 (23/30).
  - At-fault collisions 0.07, offroad 0.10, left corridor 0.07, progress 0.87.
  - Mean Drive call 0.42 s; driver peak GPU memory **14,528 MiB**.
- **Recommendation to Aman:**
  - September slot 2 = v0.3 on PAI.
  - Slot 3 = v0.3 on nuPlan. No nuPlan warmup: the image is proven on the server, and the same code passed locally on nuPlan.

### Sprint before the last two September slots (Aman approved 2026-09-28; slots held until Sep 30)
- **Levers**, all switches in the v0.3 code (defaults unchanged):
  - `A15_NUM_SAMPLES=K`: K diffusion samples per plan, keep the xy medoid. The VLM prompt runs once and the KV cache is reused.
    - K=5 check (job 12957818, Blackwell): nvidia-smi 14,432 MiB (unchanged); 1.71 s vs 0.70 s per plan.
    - Estimated official ≈ 10,700 of 14,400 s at 1 Hz.
  - `A15_NAV=1`: "Turn left/right in Xm" from the route waypoints (first heading change > 30° within 100 m). Unit-tested on synthetic routes.
  - `A15_REASONING=1`: chain-of-causation text on.
  - Controller gains (submission-time JSON, no image change):
    - A: lat_position 3.0, heading 2.0;
    - B: rel_front_steering_angle 10.0, rel_acceleration 3.0.
- **Evaluation:**
  - The first 40 curated-val PAI scenes (prefetched once: 90 usdz files, 70 GB).
  - One Blackwell GPU per variant, 4 concurrent rollouts (same for all variants).
  - Paired per-scene comparison against the base run with `src/compare_runs.py` (bootstrap 90% interval).
  - Jobs 12959800-805: base, k5, nav, gainsA, gainsB, cot.
- **Paired re-read of Result 7:** 1 Hz vs 2 Hz on 10 scenes = +0.024 (1 win, 0 losses, 9 ties). Equivalent, not better.
- **Rule:** only variants with a clear paired gain go into v0.4; otherwise submit v0.3.

### Sprint round 1: 40 PAI scenes, paired against base (jobs 12959800-805, Blackwell, 4 concurrent rollouts)

| Variant | Mean | Paired diff [90%] | Wins/losses/ties | Passed |
|---|---|---|---|---|
| base (v0.3: int8, 1 Hz, K=1) | 0.661 | - | - | 30/40 |
| reasoning on | 0.496 | **-0.165** [-0.308, -0.024] | 7/14/19 | 21 |
| K=5 medoid | 0.579 | **-0.082** [-0.179, +0.007] | 5/9/26 | 27 |
| gains A (lat 3, heading 2) | 0.701 | +0.040 [-0.012, +0.104] | 5/5/30 | 32 |
| nav text | 0.696 | +0.036 [-0.013, +0.095] | 5/7/28 | 33 |
| gains B (steer 10, acc 3) | 0.672 | +0.011 [-0.080, +0.100] | 8/4/28 | 31 |

- **Dropped:** reasoning and the medoid (both lose).
- **Surprising:** the medoid's consensus trajectory drives worse. Plausibly it averages away decisive manoeuvres. Note it, don't chase it now.
- **Suggestive but not significant:** gains A and nav.
- **Driver peak GPU memory 15,052-15,134 MiB in every variant.**
- **gains B had 2 rollouts end on `UNAVAILABLE: Stream removed (Connection reset by peer)`** during the runtime's submit_image to our driver.
  - The driver kept running with no errors. Likely load: 4 concurrent rollouts per driver locally vs 2 officially.
  - Hardening for v0.4: more gRPC workers, larger message limits.
- **Round 2** (same 40 scenes):
  - base rerun (A/A noise calibration), 12964410;
  - nav + gains A, 12964411;
  - gains A strong (lat 5, heading 3), 12964412.
- **Round 3:** the winner vs base on 30 fresh scenes (curated val 41-70, prefetch 12964413) to avoid winner's curse.

### INVALIDATION: co-located local runs shared ports (found 2026-09-29 00:40 UTC)
- **Cause:** every local run used fixed ports (services 6000+, driver 6789). gRPC binds with SO_REUSEPORT, so two runs on one node silently split connections between each other's drivers and services.
- **Evidence (driver sessions served, 40 expected):**
  - ins046: nav 40 (3 with nav text), gains A **54**, gains B **22**. Round 2 base rerun **0**, nav+gains A 8.
  - ins047: base, k5 and cot each exactly 40, with per-variant latencies (0.17 / 0.42 / 0.32 s per Drive call) matching their own drivers.
- **Verdict:**
  - Round 1 nav, gains A and gains B are **invalid** (drivers and controllers, which hold the gains, mixed on ins046). Round 2 base rerun and nav+gains A were cancelled.
  - base, k5 and cot (ins047) stand. They all used default gains, so a shared controller couldn't change results, and each driver served its own 40 sessions.
  - All earlier comparisons (nuPlan starter/a15 on different nodes; the sequential 10-scene PAI chain; nuPlan 1 Hz alone) are unaffected.
- **Fix:** `sim_local.sbatch` now uses a per-job port block (`wizard.baseport`, driver `PORT_BASE+9`, from the Slurm job id) and refuses to start if the block is busy.
- **Resubmitted:** nav 12965003, gains A 12965004, nav+gains A 12965005, base rerun (A/A) 12965006. Gains A strong (12964412) ran alone on ins047 and is kept.
- The fresh-scene prefetch first failed on CRLF line endings in `sim_suites_curated.csv` (scene ids ended in `\r`); rerun as 12964995.

### Sprint result (clean reruns with per-job ports, 2026-09-29): no lever beats noise; submit v0.3 unchanged
- **Validity:** every driver served exactly its own 40 sessions (jobs 12965003-006 on ins046 with distinct port blocks; 12964412 alone on ins047).
- **Noise floor (A/A):** base rerun 0.693 vs base 0.661 on the same 40 scenes, config identical. That's +0.032 [-0.013, +0.092], 4 wins / 6 losses. The simulator is not deterministic run to run, so differences of about ±0.03 are noise at 40 scenes.
- **Against the base rerun (0.693):**

| Variant | Mean | Diff [90%] | Wins/losses |
|---|---|---|---|
| nav | 0.691 | -0.002 [-0.079, +0.075] | 7/7 |
| gains A (lat 3, heading 2) | 0.696 | +0.002 [-0.051, +0.056] | 8/4 |
| nav + gains A | 0.672 | -0.021 [-0.072, +0.007] | 4/6 |
| gains A strong (lat 5, heading 3) | 0.678 | -0.015 [-0.052, +0.007] | 5/4 |

- **Nav text triggered in only 2 of 40 sessions** (first turn within 100 m), so it can barely matter on these scenes.
- **Still-valid negatives** (ins047, beyond the noise floor): reasoning -0.165 [-0.308, -0.024]; K=5 medoid -0.082 [-0.179, +0.007].
- **Decision:**
  - Both remaining September slots use v0.3 (`v0.3-a15int8-1hz`, digest 29a4f9c2…) with default controller gains; no v0.4.
  - The fresh-scene confirmation (curated val 41-70, prefetched: 150 usdz, 120 GB total) is not needed now; kept for October.
- **Lesson:** always run an A/A (same-config) rerun before believing a local gain. Here a +0.03 "gain" was exactly the size of rerun noise.
- **October candidates:**
  - Reduce local noise: more scenes; 2 seeds per scene.
  - Fine-tune the action expert on NuRec/PAI data (the nuReasoning fine-tune fixed Alpamayo's speed profile).
  - Selective (not uniform) conservatism for the at-fault collisions.
  - The gRPC robustness hardening.

### Official submissions: September slots 2 and 3 (2026-09-29)
- PAI: `5aab2ba2-ada8-4df5-91ab-24b03bb86234`, image `v0.3-a15int8-1hz` (digest 29a4f9c2…), default gains.
- nuPlan: `491ce993-ed61-4c43-ad08-a9968aa3bce9`, same image, default gains.
- The first submit attempt was refused before submission: the Docker ECR token had expired (12 h). The CLI checks the image first, so no slot was used; `ecr-login` fixed it.
- September quota is now used up (3/3). October brings 3 more.

### Result 10: v0.3 official results (2026-09-29)

| Track | Submission | Coverage | Wall time / limit | PCS [interval] | Rank | Scene score | At-fault spacing |
|---|---|---|---|---|---|---|---|
| PAI | 5aab2ba2 | 987/987 | 9,474 / 14,400 s | **2143** [2096-2189] | **2 / 58** | 0.611 | 0.795 |
| nuPlan | 491ce993 | 1000/1000 | 1,268 / 3,000 s | 1296 [1235-1358] | 19 / 71 | 0.827 | 0.234 |

- **PAI:**
  - The only entry above us is acmlab-dow, 2224 (scene 0.665). Next: jongam 2123/2119, fox-srdc 2108, team-unofficial (slim-fp8) 2081. The host's Alpamayo 1 reference is 2000.
  - Our at-fault incident spacing (0.795) is the best in the top 6.
  - Our starter entry now shows 1468 (rank 43); scores are re-fit as entries arrive.
  - My pre-submission ESTIMATE was PCS ~1900-1990 (rank ~10-16): too pessimistic. The local-to-hidden scene-score shrink measured on the starter (0.349 → 0.140) did not transfer to a driver that turns and reacts. Local 0.661-0.693 on 40 scenes vs official 0.611.
  - The wall-time estimate (~10,000 s) held: 9,474 s.
- **nuPlan:**
  - The top 10 are almost all py123d-garage (recorded on day one as the host framework; the leaderboard has no host flag).
  - Likely-eligible teams ahead of us: team-ready 1463 (scene 0.875), yax-nam 1469, team-lucifer 1467. Host go_straight_with_delay_v2: 1333 (scene 0.799, spacing 0.258).
  - Our scene score beats go-straight, but our incident spacing is lower (0.234), and the IRT fit weights it.
  - nuPlan is off-domain for Alpamayo (Motional rig, 2 Hz frames). The fix is more data-side work than tuning.
- **Quota:** September's 3/3 used. October: 3.

## October, week 1: test bench (started 2026-09-29, PLAN.md)
- **Storage:** the department filesystem is at **90%** (541 GB free of 5.0 TB; ~1 TB free on 2026-09-27). AlpaSim holds ~210 GB (scene data 162 GB, sandboxes 47 GB).
  - The nuPlan shard-2 fetch was cancelled before extracting. nuPlan bench = the 100 scenes of shard 001.
  - The PAI bench grows to 120 curated-val scenes (prefetch 71-120, job 12991537, ~100 GB).
- **v0.3 baselines with rollout videos** (VIDEO=true) for the failure taxonomy:
  - nuPlan 100 scenes: 12991555 (L40S, 2 rollouts).
  - PAI 1-40: 12991558 (third same-config run on these scenes).
  - PAI 41-80: 12991556.
  - PAI 81-120: 12991557.
  - PAI runs: Blackwell, 4 rollouts, per-job ports.
