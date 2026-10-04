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
  - NVlabs/alpasim#170 (2026-08-31): a team asked whether "unused August submissions" would carry over, noting "the September submission limit is planned to be reduced to 3". The organiser replied "No... all team allocations will be reset". (#171 is the same question, closed by the asker with no reply.)
  - NVlabs/alpasim#158 (organiser, 2026-08-24): "submissions per month(3)".
  - Contestant Guide: official limits "are shared across tracks"; only warmups are "rolling 30-day". It never states the official period itself; re-checked 2026-09-29.
  - Still unconfirmed by us: that the counter resets on Oct 1. Check `limits` on Oct 1.
  - `limits` on 2026-09-29 (Aman): official 3 used / 0 remaining; pai-warmup 2 used / 1 left; nuplan-warmup 1 used / 2 left.
  - Worst case if it is really rolling 30 days: slots free up on Oct 27 (starter, sent Sep 27) and Oct 29/30 (the two v0.3 entries), still before the Oct 31 close.
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

### Failure taxonomy, first pass (`src/failure_taxonomy.py`; PAI scenes 1-40; two same-config v0.3 runs, 12959800 and 12965006)
- **Primary categories** (run 1 / run 2):
  - at-fault collision 2/2;
  - offroad 0/0;
  - left corridor 8/6;
  - low progress (progress_clipped_rel < 0.8) 7/9;
  - pass 23/23.
- **Stability:** 37/40 scenes land in the same category in both runs, so failures are mostly systematic, not simulator noise.
- **Reliable failure set** (same category in both runs):
  - **Low progress, 7 scenes:** dist_to_gt 0.02-0.35 m (on the human path) but progress 0.16-0.75. The car is too slow or stops. That matches the nuReasoning finding that Alpamayo's path is good and its speed is weak (their stage-2 fine-tune fixed the speed profile).
  - **Left corridor, 6 scenes:** dist_to_gt 4.06-4.25 m, just past the 4 m corridor (`max_dist_to_gt_trajectory: 4.0`). The car diverges from the recorded route (lane or branch choice).
  - **At-fault collision, 2 scenes.**
- **Implications for the full system:**
  1. Speed / progress is the largest reliable bucket, and the driver has no speed handling at all. Candidates: a fine-tuned action expert, or speed as a chooser feature.
  2. Route adherence: nav text fired in only 2/40 sessions; check how the route arrives (start offset 40 m) before judging nav.
- **Next:** view rollout videos (bench run 12991558 renders them) for these 15 scenes.
- **Nav text v1 was misplaced, not refuted.**
  - The runtime's route has 20 waypoints over 80 m, but the challenge trims the first 40 m (`route_start_offset_m: 40`, `route_generator.py`).
  - v1 measured turns against the route's own first segment, which already points the new way when the turn is inside the trimmed window. So it fired in only 2/40 sessions.
  - v2 measures route headings against the car's heading and logs each session's first route. Unit tests: straight none; turn inside the window "Turn left in 29m"; turn at 60 m "Turn right in 60m"; gentle 10° curve none.
  - Nav gets re-tested in the week-2 full system.

## 2026-09-29 Week-1 bench: nuPlan baseline (v0.3, 100 navtest scenes, job 12991555, L40S, 37 min)
- **Mean scene score 0.759**, 82 passed. Official v0.3 nuPlan scene score was 0.827, so these local scenes are harder or differently mixed.
- **Taxonomy:** pass 63, low progress 19, left corridor 9, offroad 6, at-fault collision 3.
- **Score lost** (sum of 1 - score): corridor 9.0, low progress 6.1, offroad 6.0, collision 3.0.
- **Rollout videos (BEV panel, all 18 corridor/offroad/collision scenes):**
  - **Corridor exits and offroad are almost all wrong-branch choices at junctions:** straight where the recorded route turns, a turn where it goes straight, or the wrong exit on a curve. So 15 of the 24 lost points are route intent, not control.
  - **All 3 collisions are rear-ends** on straight multi-lane roads (ego closes on the lead car).
  - **Low progress:** short-route scenes (GT 7-20 m in the window); in two the car never moves (progress 0.00 and 0.03).
- **Route chooser written** (`src/a15_driver/route_choice.py`, `A15_CHOOSER=route` with `A15_NUM_SAMPLES=K`): K diffusion samples, keep sample 0 (the K=1 plan) unless another is closer to the route polyline [car, waypoints 40-80 m] by more than 2 m. Unit tests on synthetic junctions: picks the turning sample for a left-turn route, the straight one for a straight route, and keeps sample 0 under jitter, short plans or no route. Off by default. Not yet run closed loop.
- Three nuPlan runs of it (nav v2, chooser K=6, both) are queued on the same 100 scenes; the first attempt (12995295-97) was cancelled before starting and requeued as 12995325 (nav v2), 12995326 (chooser K=6), 12995327 (both).

### nuPlan route runs (same 100 scenes, paired vs bench 12991555 = 0.759)
| Variant | Mean | Diff [90%] | W/L/T | Notes |
|---|---|---|---|---|
| nav v2 text (12995325) | 0.739 | -0.020 [-0.047, -0.000] | 2/4/94 | text alone slightly hurts |
| route chooser K=6 (12995326) | 0.787 | **+0.028 [+0.005, +0.059]** | 3/5/92 | 66 plans rerouted over 100 sessions; offroad 6 to 4, corridor 9 to 8, low progress 19 to 22 |
| nav v2 + chooser (12995327) | 0.767 | +0.007 [-0.005, +0.027] | 4/5/91 | text cancels most of the chooser gain |
- All drivers served their own 100 sessions. 92-94 ties: closed loop on nuPlan is near-deterministic, so paired diffs are sharp.
- **Reading:** the chooser helps a little but fixes few of the 15 wrong-branch scenes; most samples probably agree on the wrong branch. Next: look at the per-scene candidate spread at the junction scenes before going further.

### Storage cleanup (2026-09-30, approved by Aman per item)
- Deleted only the raw `rollout.asl` logs (~60 GB) of:
  - the 5 port-contaminated runs: 12959802, 12959803, 12959804, 12964410, 12964411;
  - the three superseded 10-scene PAI runs: 12952257, 12952258, 12952259.
- Kept each run's aggregate, configs and logs. The two runs cancelled mid-way (12964410, 12964411) never had a summary.
- Everything else stays: scene data (exactly the 120 PAI + 100 nuPlan bench scenes), simulator sandboxes, valid runs.

### Why the route chooser fixes so few wrong-branch scenes (15 nuPlan corridor/offroad scenes, `A15_LOG_CHOICES=1`)
- Jobs 12996572 (K=6) and 12996573 (K=16). Pass 2/15 in both, vs 0/15 at baseline. K=16 is no better than K=6.
- **The candidates are diverse:** at a scene's first plan the 16 end points span up to ±40 m laterally, so the right branch is usually on the menu.
- **The chooser does switch** (3-4 of 5 plans per scene), yet 13/15 still fail. Suspected reasons:
  1. nuPlan scenes are short: 10 steps at 2 Hz, so only 5 plans at 1 Hz. The first plan is by far the most uncertain (widest spread, highest route cost), and the car commits during that first second.
  2. Picking a different candidate each replan gives inconsistent plans (left-ish, then right-ish).
  3. Alpamayo expects 4 frames 0.1 s apart. On nuPlan (2 Hz cameras) it gets frames 0.5 s apart, plus a short pose history at the session start.
- **Cost:** K=6 takes 2.37 s per plan vs 0.80 s at K=1 (L40S). v0.3 used 1,268 of the 3,000 s nuPlan budget, so K=6 is borderline and K=16 (5.8 s) is out.
- **Next:** log the first plan's inputs (frames and poses available) and try a temporally consistent chooser (stay near the previous plan unless the route clearly disagrees).

## 2026-09-30 Week-1 bench: PAI baseline complete (v0.3, 120 curated-val scenes; jobs 12991558, 12991556, 12991557; 40 sessions each, own drivers)
- **Mean scene score 0.638** (batches 0.668 / 0.579 / 0.667). Official v0.3 PAI scene score 0.611, so the local bench tracks the hidden set closely.
- **Taxonomy (120):** pass 63, left corridor 24, low progress 24, at-fault collision 5, offroad 4.
- **Score lost (43.4 in total):** corridor **24.0**, low progress 10.4, collision 5.0, offroad 4.0. Corridor exits are more than half of what PAI loses.
- **Corridor exits (videos, 17 in 41-120):** mostly lane choice on straight roads (the recorded driver changes lane, ours stays, ending 4.0-4.3 m off); a few are junction turns. Unlike nuPlan, where they are nearly all junction branches.
- **Low progress:** progress 0.01-0.78 over GT distances of 12-275 m; several nearly stop on long routes (0.01-0.2).
- Scenes 1-40 vs the two earlier same-config runs: 35/40 in the same category in all three; the new run is -0.025 [-0.100, +0.049] vs the A/A rerun (noise).
- **PAI route chooser (K=6, logged)** on all 120 scenes is queued one batch at a time: 12996677-79.

## 2026-09-30 nuPlan first-plan inputs (job 12996681, `A15_LOG_CHOICES=1`)
- At every session's first plan: **1 frame per camera** (the 4-frame input is one frame repeated) and **2 poses 0.02 s apart**; the 1.5 s history is extrapolated from that 20 ms segment.
- That plan drives for 1 s = 20% of a 5 s nuPlan scene, and it is the least certain plan (widest candidate spread).
- **Test:** `A15_EARLY_REPLAN=1` replans on each new frame until 4 distinct frames exist, then 1 Hz. 100 nuPlan scenes, job 12996703.
- ~~Result (12996703): early replan -0.083~~ **VOID, see the plan-storage bug below.**
  - It made 10 plans per scene (one per step), not the ~7 expected; the trigger behaved differently from its design, not yet explained. Either way it replanned more often.
  - Same direction as the earlier nuPlan test (2 Hz 0.728 vs 1 Hz 0.784 on 30 scenes): **on nuPlan, replanning more often hurts**, plausibly because each replan draws a fresh sample and the car jitters between them.
- **Next probe:** replan every 2 s (job 12996906). If fewer replans help, plan consistency (blending with or staying near the previous plan) becomes a full-system part for both tracks.

## 2026-09-30 BUG: plans not stored when chooser logging was off (commit be21c0c to c722a2a)
- **What:** adding the chooser log (be21c0c) nested `s.cached_plan = new` inside `if log_choices and costs`. Any run without chooser costs computed plans but never stored them, so the car drove the whole scene on the straight constant-velocity fallback ("cached 0, fallback 10").
- **Found by:** the early-replan run made 10 plans per scene instead of 7; a 3-scene debug log (12996909) showed plan_nframes stuck at 0 and no cached plan.
- **Void results:** early replan 12996703 (the -0.083); 2 s interval 12996906 (cancelled); PAI chooser 12996677-79 (cancelled; its plans were dropped whenever no route had arrived yet). The first-plan input finding (12996681) stands, because it was logged before storing.
- **Unaffected (verified "cached 10, fallback 0" or older code):** the week-1 bench (12991555-58), nuPlan route tests 12995325-27, the 15-scene K=6/K=16 runs, and every submitted image (v0.3 is e7a75d6).
- **Fix and guards:**
  - plan storage moved back inside the stats lock, logging after it;
  - `tests/test_driver_plan_storage.py` drives the real `_maybe_infer` with a fake policy under all 16 flag combinations (log, costs, early, nav): plan stored every step, and 5 plans per 5 s scene (7 with early replan). PASS (job 12996940);
  - `sim_local.sbatch` now runs that test before starting (exit 4 on failure) and writes `driver_health.txt` (sessions with >1 fallback step; good runs have 0).

## 2026-09-30 Replan timing on nuPlan, fixed driver (100 scenes, paired vs bench 0.759; both "0 sessions with fallback")
| Variant | Plans/scene | Mean | Diff [90%] | W/L/T |
|---|---|---|---|---|
| early replan until 4 frames (12996944) | 7 | 0.767 | +0.008 [-0.025, +0.043] | 10/15/75 |
| replan every 2 s (12996945) | 3 | 0.688 | **-0.072 [-0.123, -0.022]** | 7/22/71 |
- **1 Hz stays.** Early replanning is noise: the one-frame first plan is real, but it is not what loses the junction scenes. Replanning less often clearly hurts.
- With the 2 Hz vs 1 Hz result (0.728 vs 0.784, 30 scenes), 1 Hz is the best of 0.5, 1 and 2 Hz on nuPlan.
- The earlier line "replanning more often hurts" came from the void run and is withdrawn.

## 2026-09-30 Assessment: borrowing nuReasoning's fine-tuned Alpamayo for nuPlan (read-only look, nothing run)
- `nureasoning/runs_a15ft/all/checkpoint-30573`: alpamayo-recipes stage 2 on nuScenes train (VLM frozen, action expert trained with flow matching), 3 epochs, loss 4.15 to 3.34. The h0/h1 halves are the same recipe on half the data.
- **Only the expert changed:** 396 `expert.layers` tensors plus `action_in_proj` / `action_out_proj` (~2.3B trainable params). So the swap is: base Alpamayo, same int8 VLM, fine-tuned expert. The image stays small (only the expert weights are added).
- **Mismatches to resolve before a fair nuPlan test:**
  1. Prompt format: trained with `include_camera_ids: false`, `include_frame_nums: false`; our driver sends camera indices.
  2. Image size: trained at 896x592; we feed 576x320 to fit 16 GiB. The larger size means more vision tokens, so the memory cap must be re-measured.
  3. Frame timing and camera set: nuScenes Motional rig (older vehicle) vs nuPlan's Motional rig.
- **Licence question for Aman before any submission:** nuScenes is CC BY-NC-SA. A model fine-tuned on it, submitted under the challenge terms (Apache-2.0 plus a broad NVIDIA licence), may not be allowed. Local tests are fine.
- **Plan:** first a 100-scene nuPlan test of the expert swap with the prompt format matched (camera ids off), at our 576x320; then 896x592 if memory allows.
- **Update after reading `nureasoning/src/a15ft_infer.py`:** the fine-tune is a recipes `TrainableAlpamayoR1` checkpoint with the recipes' VLA processor (its own prompt build at 896x592) and the R1 sampler, not a drop-in expert for our `Alpamayo1_5` path. A faithful port means bringing the recipes processor and model class into the driver.
- On nuReasoning, in-domain, it beat zero-shot by only +0.002 to +0.0085. And it calibrates speed, while nuPlan loses mostly to wrong-branch choices.
- **Decision:** deprioritised. nuPlan effort goes to route intent (chooser geometry and consistency) first.

## 2026-09-30 What the chooser sees at nuPlan junctions (job 12997073: 15 wrong-branch scenes, K=6, route and candidates logged)
- Plots of the first two plans per scene (candidates, pick, route):
  - **About a third of scenes have no candidate turning the route's way** at the first plan. The chooser can't fix those; the model has to propose the turn.
  - **Where a turning candidate exists, the chooser picks it** (88a82b, aa575f, 11939e, 558367, 7923af).
- **Videos:** in f357ff and f95146 the car now takes the right branch but turns wider and later than the recorded driver, ending 4.1-4.5 m off. With a 2 m margin, later plans fall back to sample 0 (the wide turn).
- **Route text v2 misstates the distance:** it said "Turn left in 40m" for turns about 15 m ahead, because it measured to the first route waypoint, which is always ≥40 m out. v3 (`A15_NAV_DIST=intersect`) uses where the turned road, extended back, meets the car's forward axis. On the logged 88a82b route that gives 15 m instead of 42 m.
- **Some offroad failures look like map artifacts:** e65021 sits on the recorded path (dist 0.0-0.1 m) and is flagged offroad from the first frame.
- Routes arrive as 20 slots with about 10 real points and NaN padding; the chooser and route text both drop the NaNs (now covered by a test).
- **Queued (100 scenes, paired vs 0.759):** chooser margin 0.5 m (12997155); route text v3 (12997156); both (12997157).

## 2026-09-30 nuPlan route runs, round 2 (100 scenes, paired vs bench 0.759; all "0 sessions with fallback")
| Variant | Mean | Diff vs bench [90%] | W/L/T |
|---|---|---|---|
| chooser K=6, margin 0.5 m (12997155) | 0.786 | +0.027 [-0.015, +0.070] | 7/8/85 |
| route text v3, turn distance fixed (12997156) | 0.749 | -0.010 [-0.039, +0.019] | 3/4/93 |
| text v3 + chooser margin 0.5 (12997157) | 0.762 | +0.002 [-0.034, +0.039] | 6/9/85 |
- **Margin 0.5 vs 2.0:** -0.001 [-0.032, +0.030], identical. The chooser's gain is stable across two runs (+0.028, +0.027); the margin doesn't matter.
- **Route text doesn't help Alpamayo on nuPlan, even with the distance fixed:** 93/100 scenes tie. The model mostly ignores the route text, and adding it to the chooser costs ~0.025.
- Text v3 fired in 39/100 first routes, with distances 26-68 m.
- **Where nuPlan stands:** the chooser (K=6, +0.028) is the one validated part. The remaining junction failures need the model to propose the turn, which neither the text input nor more samples (K=16) achieves.

## 2026-09-30 PAI chooser batch 1 lost to a renderer crash (12996946)
- 26/40 rollouts errored: the NRE renderer hit "CUDA error: an illegal memory access" in its own camera kernel (`generate_image_points`), after which every rollout failed.
- Not memory pressure: the driver peaked at 15.1 GB and the renderer at 20.6 GB, on a 96 GB card. Our driver logged no errors.
- The batch is rerun after batches 2-3 (12996947 running, clean so far).

## 2026-09-30 Scope change (Aman): PAI only
- This repo now works on the PAI track only, with Alpamayo 1.5. The nuPlan track moved to the JEPA thread (`../alpasim-jepa`, another agent). That folder and its cluster folder are read-only for us; the thread reads this repo read-only.
- **Aman's figures from that thread, not measured here:**
  - WA-JEPA 0.958 vs v0.3 0.759 on our 100-scene nuPlan bench, and 0.729 vs 0.510 on a held-out shard;
  - zero-shot WA-JEPA on PAI (our bench1 scenes): 0.25-0.30 vs v0.3 0.668. The per-scene best of both is 0.762, a possible low-priority extra candidate later.
- **Slots:** October's 3 official slots are shared. PAI plans for up to 2; the JEPA thread takes 1 nuPlan warmup and 1 official nuPlan slot early in October. Aman submits; nothing goes in without his go-ahead.
- **nuPlan wrap-up here:**
  - no nuPlan jobs were queued or running;
  - deleted the raw `rollout.asl` logs of the 21 nuPlan experiment runs (~30 GB), as allowed for our own scratch outputs;
  - kept every run's aggregate, videos, configs and driver logs, plus the whole bench baseline `local_nuplan_a15_bench_12991555` (the JEPA thread pairs against it);
  - scene data (`data/nuplan_track`, 43 GB, shard 001) is untouched; deleting it needs Aman's go-ahead.

## 2026-10-01 Quota reset confirmed
- `limits` on Oct 1 (Aman): official **0 used / 3 remaining**, so the counter resets per calendar month. Warmups unchanged (rolling 30 days): pai-warmup 2 used / 1 left, nuplan-warmup 1 used / 2 left.
- October split per the scope note: PAI up to 2 official; nuPlan (JEPA thread) 1 official and 1 warmup. Aman submits.

## 2026-10-01 PAI route chooser v1, all 120 scenes (K=6, margin 2 m; 12997585 rerun of 1-40, 12996947, 12996948; all "0 sessions with fallback")
- **Paired vs the v0.3 bench: 0.651 vs 0.638, +0.013 [90% -0.040, +0.066]**, wins 19, losses 22. Noise-level.
  - By batch: 1-40 +0.055, 41-80 -0.065, 81-120 +0.048.
- **Categories (base to chooser):** corridor 24 to **14**, offroad 4 to 8, low progress 24 to 29, collision 5 to 6, pass 63 to 63.
- **Transitions:**
  - corridor to pass: 5 (+5.0);
  - pass to offroad: 4 (-4.0);
  - pass to low progress: 4 (-3.0);
  - corridor to collision/offroad: 5 (no score change).
- **Switching (rerouted plans out of 20 per scene):**
  - scenes that pass both ways: mostly 0, so the chooser leaves normal driving alone;
  - pass to low progress: 4-16 (mean 11), i.e. the scenes where it switches most;
  - remaining corridor exits: 8 of 13 had 0 switches. A lane change moves the route only ~3.5 m sideways, under the 2 m switching margin.
- **Cost:** Drive RPC mean 0.44 s vs 0.17 s; batch wall time ~2.5-3.2 h vs ~2 h. PAI's time limit (v0.3 used 9,474 of 14,400 s) needs checking before any K=6 image.
- **v2 queued (13060758-60, one GPU at a time):** margin 1.0 m plus a speed guard (`A15_ROUTE_MIN_LEN=0.9`: only samples whose path is at least 90% of sample 0's are eligible).

## 2026-10-01 PAI low-progress scenes (videos of the 8 worst, baseline bench)
- Progress 0.01-0.28 of 29-275 m recorded distance.
- **Seen:**
  - waiting at red lights or in queues, then setting off later and slower than the recorded driver;
  - general slowness on open roads;
  - one night intersection where the car never moves for the whole 20 s (the recorded driver turned left);
  - two scenes where the car drifts off the mapped road late on.
- Consistent with the nuReasoning finding (Alpamayo's path is good, its speed weak).
- **Probe queued:** `A15_SPEED_SCALE=1.15`, the same planned path driven 15% faster (time-stretched; horizon 6.4 s / 1.15). Runs after chooser v2, one GPU at a time (13061277-79). The progress score saturates at 80% of the recorded distance, so only slow scenes can gain; the risk is more collisions and corridor exits.

## 2026-10-01 PAI chooser v2 and speed probe (120-scene bench, paired vs v0.3 0.638; "0 sessions with fallback")
| Variant | n | Mean | Diff [90%] | W/L | Categories |
|---|---|---|---|---|---|
| **chooser v2**: K=6, margin 1 m, speed guard 0.9 (13060758-60) | 120 | **0.692** | **+0.054 [+0.005, +0.105]** | 21/18 | pass 69, low progress 24, corridor 14, offroad 8, collision 5 |
| speed scale 1.15, K=1 (13061278-79) | 80 | 0.316 | **-0.307 [-0.418, -0.199]** | 14/33 | collision 19, corridor 25, offroad 9, pass 25 |
- **Chooser v2 is the first PAI part to clear the noise floor.** Against v1 (+0.013), the speed guard removed the extra slow scenes (29 back to 24) and the 1 m margin kept the corridor gain (24 to 14).
- **Speed scaling is rejected:** driving the planned path 15% faster causes collisions and corridor exits. Speed has to come from the model's own judgement (chooser feature or fine-tune), not a global stretch.
- **Renderer crash again:** speed batch 1-40 (13061277) lost 28/40 rollouts to the NRE "illegal memory access" (105 log lines). That's now 2 of the last 12 PAI batches. Not rerun, given the verdict.
- Timing: v2 batches took 2.3 h each (v1 2.5-3.2 h, baseline ~2 h).

## 2026-10-03 Disk-full incident and storage cleanup (approved by Aman: A, B, C)
- The shared /insomnia001 hit 100% on 2026-10-02 ~22:17 UTC (heads-up from the alpasim-jepa thread). **None of our jobs ran then:** the last ended 2026-10-01 14:05 UTC, and all outputs were checked (health files, summaries).
- Our folder was 595 GB: runs 304, data 246, sandboxes 46. Free space was 185 GB, under the ~200 GB everyone keeps.
- **Deleted the raw `rollout.asl` logs only** (kept aggregate, configs, driver logs, videos):
  - A: speed probe 13061277-79, crashed 12996946, buggy-driver 12996677;
  - B: chooser v1 12996947, 12996948, 12997585;
  - C: September tuning runs 12959800, 12959801, 12959805, 12964412, 12965003-06.
- After: runs 97 GB; filesystem **391 GB free**.
- Kept raw logs: baseline bench (12991556-58) and chooser v2 (13060758-60).

## 2026-10-03 Chooser v2 diagnostics and v3
- **Memory:** v2's driver peaks at 15,052 MiB in all three batches, identical to K=1. Only the flow sampler repeats, so the 16 GiB cap holds.
- **Time (estimate, not measured on official hardware):** the warmups showed halving model work (2 Hz to 1 Hz) changed wave time only 552 to 536 s, so model work is ~16 s per wave and rendering dominates. K=6 multiplies model work ~2.5-3.5x, which puts the official PAI run near 10,000-10,500 s of 14,400 (v0.3: 9,474). To confirm with the last PAI warmup, on the full-system image only.
- **Faster candidates are not on offer:** in low-progress scenes the longest of the 6 candidates is a median 1.06x sample 0's length (p90 1.28x; pass scenes 1.00x / 1.13x). The samples agree on speed, mostly on stopping. A "pick the fastest" chooser would gain little.
- **Remaining corridor exits are knife-edge:** all 14 end 4.01-4.24 m off (limit 4.0). In several, a candidate >1 m closer to the route was refused because it was slower (7 plans in b8f636; 4 each in 52364e, 7ca7da, e7c20a). The speed guard that fixed v1's slow scenes also blocks some lane-change fixes.
- **v3 (13141881-83, one GPU at a time):** v2 plus a graded guard. A sample more than 3 m closer to the route (lane-sized) may be as short as 0.75x sample 0; otherwise 0.9x as before. Tests cover both thresholds.
- `sim_local.sbatch` now deletes raw `rollout.asl` logs once the summary exists (`KEEP_ASL=1` keeps them), approved by Aman.

## 2026-10-03 Chooser v3 progress and cluster events
- **Cluster unreachable ~2.5 h** (2026-10-03, SSH to insomnia timed out from the Mac; DNS fine). Jobs kept running on Slurm. The watchers now report reachability changes; the old ones hid SSH failures.
- **v3 batch 41-80 (13141882) FAILED:** loading the model shards took 4 min 36 s (normally ~14 s), so the driver missed the 20-minute readiness window. Slow storage, not our code. Requeued as 13146500 after batch 81-120 (13141883, running).
- **v3 batch 1-40 (13141881):** 0.795 vs bench 0.668, +0.127 [+0.006, +0.250]. vs v2 on the same scenes: **+0.002 [-0.050, +0.055]** (7/4/29). Corridor 0, pass 28, low progress 8, collision 2, offroad 2. Clean driver health, no renderer crash.
- **Auto-delete works:** the batch run dir is 130 MB (raw logs removed after the summary).

## 2026-10-03 Chooser v3 verdict (120 PAI scenes; 13141881, 13146500 rerun of 41-80, 13141883; all clean, no renderer crash)
| Comparison | Means | Diff [90%] | W/L |
|---|---|---|---|
| v3 vs bench | 0.638 to 0.699 | **+0.061 [+0.008, +0.117]** | 25/19 |
| v3 vs v2 | 0.692 to 0.699 | +0.007 [-0.025, +0.039] | 17/14 |
| v2 vs bench | 0.638 to 0.692 | +0.054 [+0.005, +0.105] | 21/18 |
- v3 categories: pass 69, low progress 26, corridor 14, offroad 7, collision 4.
- **The graded guard adds nothing measurable over v2;** corridor exits stay at 14. Both clear the bench noise floor.
- **Keep v2** (fewer knobs) as the chooser part of the full system. v3 stays available as an option.
- **Next (13150825-27, one GPU at a time):** route text v3 (`A15_NAV=1`, `A15_NAV_DIST=intersect`) plus chooser v2, on 120 PAI scenes.
  - On nuPlan the model ignored route text, but PAI is Alpamayo 1.5's training domain, where its navigation input was trained.
  - The only earlier PAI nav test used v1, which fired in 2/40 scenes.

## 2026-10-04 Image tag policy (Aman)
- Submission tags are neutral version numbers, never method names; the leaderboard shows tags publicly. This applies to every future submission.
- Tag register (one sequence for both threads, all in `teams/amanupg`):

| Tag | Track | Contents (private) |
|---|---|---|
| v0.1-starter | PAI, nuPlan warmup | organisers' starter driver (straight line, current speed) |
| v0.2-a15int8 | PAI warmup | Alpamayo 1.5 int8, 2 Hz |
| v0.3-a15int8-1hz | PAI, nuPlan | Alpamayo 1.5 int8, 1 Hz (git e7a75d6) |
| v4 | nuPlan | JEPA thread's entry (see `../alpasim-jepa`) |
| v5 | PAI | reserved for the first PAI full-system image |

## 2026-10-04 Slot plan change (Aman)
- **PAI gets exactly one official October slot.** nuPlan (JEPA thread) gets one; the third is Aman's call.
- So no learn-and-resubmit loop for PAI: the single entry must be the best system verified on the local bench.
  - Freeze and verify Oct 18-24 (ablations, exact-image local run, the last PAI warmup for timing).
  - Submit Oct 25-27, with buffer before the Oct 31 close, because a failed official run still counts.

## 2026-10-04 Untried Alpamayo 1.5 sampler settings (from its own nav notebook and code, read-only)
- **Diffusion temperature:** the flow sampler scales its initial noise by `temperature` (default 1.0, what we have always run). NVIDIA's nav notebook uses **0.6**: "more stable but less diverse" trajectories. Now `A15_DIFF_TEMP`.
  - Lower temperature also narrows the chooser's candidate spread, so it is tested alone and with chooser v2.
  - GPU check job 13153714 (runs, memory, trajectory change), then 120-scene runs.
- **Classifier-free guidance on navigation** (`sample_trajectories_from_data_with_vlm_rollout_cfg_nav`, weight alpha > 1 amplifies the route instruction).
  - It builds a second, nav-free prompt cache; the notebook warns 60 GB+ for its 16-sample demo. Memory at 1 sample is untested.
  - Only matters where route text fires (9/40 PAI scenes). Parked until the plain route-text result is in.
- **GPU check (13153714, L40S, 12 validation clips):** temperature 0.6 runs at the same memory (14,236 MiB) and latency (0.955 s per plan), and changes the plans by up to 3.0 m. With 6 candidates, candidate 0 at 0.6 equals the 1-candidate plan at 0.6 (0.000 m), as designed.
- **Queued, one GPU at a time after the route-text run:**
  - temperature 0.6 alone: 13153936-38;
  - temperature 0.6 with chooser v2: 13153939-41.

## 2026-10-04 Route text v3 + chooser v2 on PAI, 120 scenes (13150825-27; all clean, no renderer crash)
- **vs chooser v2: 0.692 to 0.694, +0.002 [-0.040, +0.045]**, W/L 18/16. By batch: -0.052, +0.039, +0.019.
- vs bench: +0.056 [+0.003, +0.107].
- Route text fired in ~9/40 sessions per batch (turns over 30 degrees are rare on PAI routes).
- Categories: pass 70, low progress 21, corridor 18, offroad 7, collision 4.
- **Verdict:** route text adds nothing measurable on top of chooser v2 on PAI either. Left out of the full system. CFG-nav stays parked: it can only act in the ~23% of scenes where text fires, and plain text shows no effect there.

## 2026-10-04 Diffusion temperature 0.6, K=1, 120 PAI scenes (13153936-38; all clean, no renderer crash)
- **vs bench: 0.638 to 0.654, +0.016 [-0.036, +0.068]**, W/L 24/14. By batch: +0.040, +0.013, -0.005.
- Categories: pass 68 (bench 63), low progress 20 (24), corridor 23 (24), collision 4 (5), offroad 5 (4).
- **Reading:** noise-level on its own, but it leans positive. More wins than losses (24/14), with fewer slow scenes and more passes. Corridor exits unchanged, so it doesn't touch lane choice.
- Combined with chooser v2 is running next (13153939-41). That is the configuration that matters for the full system.

## 2026-10-04 Diffusion temperature 0.6 + chooser v2, 120 PAI scenes (13153939-41; all clean, no renderer crash)
| Comparison | Means | Diff [90%] | W/L |
|---|---|---|---|
| vs bench | 0.638 to 0.692 | +0.054 [-0.001, +0.110] | 28/14 |
| vs chooser v2 (temp 1.0) | 0.692 to 0.692 | **-0.000 [-0.054, +0.054]** | 26/17 |
| vs temp 0.6 alone | 0.654 to 0.692 | +0.038 [-0.002, +0.081] | 19/13 |
- Categories: pass 68, low progress 25, corridor 16, offroad 7, collision 4.
- **Verdict:** temperature adds nothing on top of the chooser. Keep the library default (1.0).
- **The chooser is the only part so far that clears the noise floor** (~+0.05-0.06 vs bench). Route text, early replanning, speed scaling, temperature and the graded guard all sit at zero on top of it.
- **Next:** replanning at 2 Hz with chooser v2 (more frequent route corrections and quicker restarts after stops). Then freeze and verify the `v5` image early, to leave buffer.
