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
- Tag register (one sequence for both threads, all in `teams/amanupg`; a push silently overwrites, so list the repo's tags right before any push and confirm the tag is absent):

| Tag | Track | Contents (private) |
|---|---|---|
| v0.1-starter | PAI, nuPlan warmup | organisers' starter driver (straight line, current speed) |
| v0.2-a15int8 | PAI warmup | Alpamayo 1.5 int8, 2 Hz |
| v0.3-a15int8-1hz | PAI, nuPlan | Alpamayo 1.5 int8, 1 Hz (git e7a75d6) |
| v4 | nuPlan | JEPA thread's entry (see `../alpasim-jepa`) |
| v5 | PAI | **next PAI entry** (reserved 2026-10-04, never pushed): chooser v2 or v0.3 base + guarded route snap, chosen on H100 (2026-10-18); `container/Dockerfile.v5` |
| nuplan-wajepa-judge-6672747 | nuPlan warmup | pushed 2026-10-03, before the neutral-tag policy (= v4 contents) |
| v6 | nuPlan | alpasim-jepa candidate Q (`local/q:3`), not pushed |
| v7 | PAI | reserved for a second PAI entry, if any |

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

## 2026-10-04 v5 image prepared (trial build only, not pushed)
- `container/Dockerfile.v5` builds on the submitted v0.3 image: the 21 GB weights and dependency layers are reused, and only the driver code layer is new.
- Settings baked in are chooser v2's: `A15_NUM_SAMPLES=6 A15_CHOOSER=route A15_ROUTE_MARGIN=1.0 A15_ROUTE_MIN_LEN=0.9`. Everything else is at the v0.3 behaviour: no nav, temperature 1.0, no speed scale, no early replan. The replan interval is a build argument, pending the 2 Hz result.
- **Trial build** (local/pai:v5-trial, git 2323ace): builds, the driver imports, and all 5 driver files in the image match the repo byte for byte (sha256).
- The code has gained options since the v2 bench runs (graded guard, temperature). At their defaults they are off, so v5 behaves as the benchmarked v2.
- **Before submission:** push as `v5` only after confirming the tag is unused; the last PAI warmup checks form factor and per-wave time; Aman submits.

## 2026-10-05 Replanning at 2 Hz with chooser v2 (13160657-58; batch 81-120 13160659 cancelled after 1.4 h)
| Scenes | vs chooser v2 at 1 Hz [90%] | W/L/T |
|---|---|---|
| 1-40 | **-0.097 [-0.203, +0.005]** | 4/13/23 |
| 41-80 | +0.007 [-0.097, +0.114] | 6/8/26 |
- **Cost:** Drive RPC mean 0.93 s vs 0.44 s at 1 Hz; 40 plans per session vs 20; batch wall time 2.8 h vs 2.3 h. That doubles model work against the PAI time limit.
- **Verdict: 1 Hz stays.** Batch 3 was cancelled because no plausible result there could pay for 2x compute after -0.045 over 80 scenes (it would need about +0.09).
- **v5 is frozen at chooser v2, 1 Hz.**

## 2026-10-07 Leaderboard snapshot (challenge CLI `leaderboard`)
- **PAI (84 entries): we are rank 3** with v0.3 (PCS 2138, scene 0.611, at-fault spacing 0.795). Rank 2 at submission; PCS values are re-estimated as entries arrive.
  - #1 PCS 2234: scene 0.665, spacing 0.530 (Sep 28).
  - #2 PCS 2195: scene 0.651, spacing 0.511 (Sep 30).
  - October entries so far land at PCS 2098-2119. The highest scene score on the board is 0.706, at PCS 2098 (spacing 0.751).
- **Reading:** the two teams ahead win on scene score; we win on incident spacing. v5's local gain (+0.054 scene score, collisions not up) targets exactly the gap. This is an estimate, not a forecast: local-to-official transfer of a gain this size is unmeasured.
- nuPlan (95 entries, JEPA thread's track): top PCS 1817 (scene 0.938); our v0.3 entry is rank 26 (1295).
- No PAI GPU work ran Oct 5 11:00 to Oct 7 (after the 2 Hz cancel, v5 frozen; waiting on the SimForge decision).

## 2026-10-07 Input contract re-checked; route pull queued
- **Input contract vs NVIDIA's `load_physical_aiavdataset.py`: exact match.** Cameras cross-left, front-wide, cross-right, front-tele (slots 0, 1, 2, 6, sorted); 4 frames 0.1 s apart ending at t0; 16 history poses at 0.1 s ending at t0, in the t0 rig frame; 64 future steps at 0.1 s. No hidden input bug.
- **Route pull (`A15_ROUTE_PULL`, new):** plan points beyond 30 m are blended toward the route waypoints, weight rising to the maximum by 45 m.
  - The route is the recorded path 40-80 m ahead, and corridor exits are measured against that path. The remaining exits end 4.0-4.2 m off, mostly lane changes the car did not make.
  - Points are pulled onto the route itself, never onto the car-to-route chord. Near points are untouched; headings are recomputed. 7 new unit tests pass.
- **Queued:** chooser v2 + pull 0.6 on 120 PAI scenes, one GPU at a time.

## 2026-10-07 Slow-driving scenes are mostly rear-end collisions (analysis of chooser v2 logs, no GPU)
- **The car is not mostly planning to stop.** Low-progress scenes average 1 stopped plan per 20; planned distance per 6.4 s averages 47 m (pass scenes 56 m). No candidate offered to move while the chosen plan stopped, except in 2 scenes.
- **The safety monitor never fired** (0 in all 120 scenes).
- **17 of 24 low-progress scenes (v2) had a not-at-fault collision that ended the rollout early** (mean duration 64% of 20 s). Example: 697226 ended at 16% after 2.5 m.
- **All of them are rear collisions** (v2: 25 scenes, 25 rear / 0 front / 0 lateral; bench v0.3: 17 scenes, all rear). Score lost in those scenes: 8.1 (v2), 7.4 (v0.3). Most end as low progress, some still pass.
- **Mechanism (per-timestep metrics):** at the moment of each rear hit our car is 4-62 m behind where the recorded car was at that time, and the gap grew over the previous 2 s. The replayed follower keeps the recorded pace, so if we fall behind it runs into us.
- **We drive blind to the rear:** PAI sends 6 cameras (adds rear_left_70fov, rear_right_70fov = Alpamayo slots 3 and 5); we use 4. A global speed-up already failed (-0.307, our own front collisions), so the next test is giving the model the rear cameras.
  - Check first: 6 cameras means 50% more image input, against ~1.3 GB of headroom under 16 GiB. 5-scene memory and latency check: 13244708.
- **6-camera memory check (13245747, H100, synthetic frames, K=6 chooser):**
  - 4 cameras: 14,578 MiB, 2.31 s per plan;
  - 6 cameras (adds slots 3 and 5): **15,158 MiB, 2.51 s** (+580 MiB, +9%).
  - Closed-loop peaks run ~470 MiB above this synthetic check (4 cameras: 15,052 vs 14,578), so ~15,600 MiB is expected under the 16,384 MiB cap. The 120-scene run logs the real peak.
- Closed-loop PAI must stay on Blackwell: H100 nodes still run driver 550.54 (checked 2026-10-07), and the NRE renderer needs ≥570.
- **Queued:** chooser v2 + 6 cameras on 120 scenes (as_pai_seq, after the route-pull run, one GPU at a time). The 5-scene closed-loop check 13244708 was cancelled as redundant.

## 2026-10-07 PAI simulation now runs on H100 (not only Blackwell)
- **Why:** both Blackwell nodes were full with 18 R6000 jobs queued (our priority 5051). H100/L40S/A6000 nodes run driver 550.54; the NRE renderer needs CUDA 12.8 (driver ≥570).
- **What it took (Aman approved the download):**
  1. NVIDIA `cuda-compat-12-8_570.211.01` from developer.download.nvidia.com (38 MB .deb, sha256 4363d39d…1493), unpacked to `envs/cuda-compat-12-8` (193 MB).
  2. `compose_apptainer.py --cuda-compat` mounts it into every simulator service and prepends it to LD_LIBRARY_PATH. `sim_local.sbatch` sets this with `CUDA_COMPAT=1`, plus a separate kernel cache per GPU type.
  3. **The real blocker** on L40S/H100 nodes: Slurm sets TMPDIR to `/local/slurm-job-N`, which was not mounted into the containers. The physics service's Warp NVRTC failed with "cannot open temporary file" (surfacing as "CUDA kernel build failed with error code 6"). The launcher now binds TMPDIR.
- **Result (13256805, H100 PCIe, 2 scenes, v0.3 driver):** both rollouts ran, both passed (1.0, 1.0). On Blackwell the same scenes scored 0/0/1 (0ca514, knife-edge at 4.1-4.8 m) and 1/1/1 (52c68f).
- **Memory finding (important: the official run is on H100):**
  - The driver process peaked at **17,340 MiB** (nvidia-smi) on H100 in closed loop, vs 15,052 MiB on Blackwell, and 14,578 MiB in the synthetic H100 check.
  - Torch peak reserved at warm-up is 13.37 GiB on both, so the excess is outside the torch caching allocator.
  - The rule is "each driver instance must use no more than 16 GiB of VRAM", with enforcement unstated. v0.3 completed 987/987 official rollouts on H100.
  - Next: find what grows and cap it.

## 2026-10-07 One GPU thread fixes H100 driver memory; experiments move to H100; storage
- **Fix verified (13257188, H100, chooser v2, 5 scenes):** driver memory is flat at **14,580 MiB** (was climbing 14,578 to 17,340 MiB in 400 MiB steps), torch reserved 13.37 GiB, 5/5 passed.
  - Cause: predict ran on whichever gRPC worker thread handled the request, and each new thread set up its own CUDA library state.
  - Now load, warm-up and every predict run on one dedicated thread (`A15_ONE_GPU_THREAD=1`, default).
  - v0.3 in the official H100 run most likely grew the same way and still completed. Every future image includes the fix.
- **Experiments move to H100** (free, and the official hardware). Two lanes, one batch per lane at a time:
  - lane A: chooser v2 as the H100 reference (13260244-46), then 6 cameras (13260250-52);
  - lane B: route pull 0.6 (13260247-49).
  - The Blackwell copies (13244073-75, 13245807-09) were cancelled.
- **Storage:** the filesystem hit 29 GB free on Oct 7 evening (not from us: our folder stayed at 398 GB). Jobs were held until space returned.
  - Deleted (Aman approved): the raw `.asl` logs of the bench (12991556-58) and chooser v2 (13060758-60), 96 GB. Summaries, videos and per-step metrics are kept.
  - **No duplication between `alpasim` and `alpasim-jepa`:**
    - their 468 navtest scenes share none with our 100;
    - `nuplan_test` is a different format;
    - their simulator runs read our nuPlan shard and sandboxes read-only, so those must stay here.

## 2026-10-08 Run-to-run spread is larger than the 40-scene A/A suggested
- H100 chooser v2 reference, batch 1-40 (13260244): **0.716** (clean; driver flat at 14,580 MiB; no renderer crash).
- **Same 40 scenes, chooser-v2-like configs:** 0.793 (v2, Blackwell), 0.795 (v3), 0.741 (+nav), 0.719 (+temp 0.6), 0.696 (2 Hz), 0.716 (v2, H100). A range of ~0.10 for configs that are equal at 120 scenes.
- **v0.3 on the same scenes:** 0.661, 0.668, 0.693.
- **Reading:**
  - The chooser's gain over v0.3 holds across all these runs (~0.74 vs ~0.67 on 1-40; ~0.69 vs 0.638 over 120).
  - Single-run 40-scene differences under ~0.08 are noise, and v2's original batch-1 0.793 was a high draw.
  - **Decision rule from now on:** a part counts only if its 120-scene paired gain is clearly above the ~0.05 spread of equivalent configs.
- **2026-10-08: lanes moved back to Blackwell.** The H100 queue had 37 jobs pending, with our estimated starts on Oct 9 20:20 and Oct 10 06:15; `sbatch --test-only` showed R6000 could start sooner.
  - Route pull (13266129-31, lane as_bw_a) and 6 cameras (13266132-34, lane as_bw_b) are compared against the Blackwell chooser v2 reference (13060758-60). No CUDA_COMPAT on Blackwell (driver 575).
  - H100 reference batches 41-80 and 81-120 (13260245-46) are on hold; H100 batch 1-40 is kept.

## 2026-10-08 Blackwell lanes, batch 1-40 (vs Blackwell chooser v2 13060758 = 0.793, a high draw; H100 v2 = 0.716)
| Variant | Mean | Diff vs v2 [90%] | W/L | Rear-ended | Categories |
|---|---|---|---|---|---|
| v2 + route pull 0.6 (13266129) | 0.773 | -0.020 [-0.095, +0.054] | 9/4 | 8 | pass 26, low progress 9, corridor 3, offroad 1, collision 1 |
| v2 + 6 cameras (13266132) | 0.681 | -0.112 [-0.244, +0.018] | 6/11 | **3** | pass 25, low progress 5, corridor 4, offroad 3, collision 3 |
| v2 (reference) | 0.793 | | | 9 | pass 27, low progress 9, corridor 1, offroad 2, collision 1 |
- **The rear cameras work as intended on rear-ends** (9 to 3, low progress 9 to 5), but at-fault collisions, offroad and corridor exits go up. Plausibly the 6-camera input is off the model's usual 4-camera setup.
- 6 cameras: driver peak 14,972 MiB (Blackwell, one-thread driver). Route pull: 14,392 MiB.
- Batch 1 holds only 1 v2 corridor exit, so route pull's test is batches 41-80 and 81-120.
- **Idea if 6 cameras confirms this pattern over 120 scenes:** use the rear cameras only to detect a closing vehicle behind (keep the plan, and speed up or avoid braking when tailgated), while planning from the 4 front cameras.

## 2026-10-08 Blackwell lanes, full 120 scenes (vs Blackwell chooser v2 13060758-60 = 0.692)
Insomnia was unreachable from ~07:45 to 15:53 EDT; all six batches had already completed cleanly (40/40 sessions each, 0 fallbacks).

| Variant | Mean | Diff vs v2 [90%] | W/L | Rear-ended | Categories | Wall per batch |
|---|---|---|---|---|---|---|
| v2 (reference) | 0.692 | | | 25 | pass 69, low progress 24, corridor 14, offroad 8, collision 5 | 2:17-2:20 |
| v2 + route pull 0.6 (13266129-31) | 0.690 | -0.002 [-0.051, +0.047] | 19/20 | 22 | pass 68, low progress 24, corridor 15, offroad 9, collision 4 | 2:18-2:21 |
| v2 + 6 cameras (13266132-34) | 0.670 | -0.023 [-0.095, +0.048] | 27/25 | **11** | pass 71, low progress **16**, corridor 16, offroad 10, collision 7 | 2:25-2:28 |

- **By batch:** route pull -0.020 / +0.059 / -0.046; 6 cameras -0.112 / **+0.188** / -0.145.
- **Route pull: no effect. Dropped.**
- **6 cameras: the batch-1 pattern holds over all 120 scenes.**
  - Rear-ended 25 → 11 and low progress 24 → 16.
  - Our own failures rise: at-fault collision 5 → 7, offroad 8 → 10, corridor 14 → 16.
  - v2 → 6-camera category changes: 11 low progress → pass, and 13 pass → own failure (4 offroad, 5 corridor, 4 collision).
  - Net zero; not adopted as is.
- **Next:** the hybrid (`A15_SPEED_CAMS=6`).
  - Path from the 4 front cameras with chooser v2.
  - Speed profile from a 6-camera K=1 sample, applied by re-timing along that path.
  - Jobs 13279194 (1-40) and 13279196 (81-120) on lane as_bw_a; 13279195 (41-80) on lane as_bw_b.
- **Watch the hybrid's wall time.** It runs one extra 6-camera inference per step. The official PAI limit is 14,400 s, and v0.3 used 9,474 s.

## 2026-10-09 Hybrid (path from 4 front cameras, speed from 6), 120 scenes, vs Blackwell chooser v2 (0.692)
| Variant | Mean | Diff vs v2 [90%] | W/L | Rear-ended | Categories |
|---|---|---|---|---|---|
| v2 (reference) | 0.692 | | | 25 | pass 69, low progress 24, corridor 14, offroad 8, collision 5 |
| v2 + 6 cameras (13266132-34) | 0.670 | -0.023 [-0.095, +0.048] | 27/25 | 11 | pass 71, low progress 16, corridor 16, offroad 10, collision 7 |
| **Hybrid (13279194-96)** | 0.640 | **-0.052 [-0.115, +0.013]** | 21/25 | 13 | pass 66, low progress 17, corridor 18, offroad 12, collision 7 |

- **By batch:** -0.110 / +0.041 / -0.087. All 120 sessions were clean (0 fallbacks).
  - Driver peak 14,972 MiB.
  - Mean inference 7.0 s vs 4.5 s for v2 (4 concurrent sessions).
  - Wall 2:29-2:33 per batch vs 2:17-2:20.
- **Verdict: rejected.** Using the path from the usual 4 cameras did not remove the extra own failures; offroad and corridor exits are even higher.
- **Reading:** the rear-cameras effect is a speed effect.
  - Keeping pace avoids not-at-fault rear-ends (25 → 11-13).
  - But driving that profile costs as many own failures (offroad, corridor, collisions), as the 15%-faster run showed in its extreme form (-0.307).
  - Speed alone is not a free lever on this bench.
- **Next:**
  - Second v2 reference (13289312-14, lane as_bw_b), for a less noisy baseline.
  - More flow samples for the route chooser (K = 10, 16): memory/latency check 13289753 first, then a bench if K fits under 16 GiB.

## 2026-10-09 More route-chooser samples: memory and time (check 13289753, L40S, single session)
| K | nvidia-smi | Latency |
|---|---|---|
| 6 | 14,236 MiB | 2.45 s |
| 10 | 14,236 MiB | 3.60 s |
| 16 | 14,236 MiB | 5.40 s |

- Memory is flat in K (samples don't add resident memory), so the limit is time, not VRAM.
- **Time budget.** Local 40-scene wall vs mean inference: v0.3 2:02 at 1.6 s, v2 2:20 at 4.4 s, hybrid 2:33 at 7.0 s.
  - Official v0.3 used 9,474 s of 14,400 s.
  - If the official time scales like local wall, K=12 is about 11-12k s. That's a guess, so check the official timing on any warmup before relying on it.
- **Queued:** K=12 chooser v2 bench, 13289883-85 (lane as_bw_a).

## 2026-10-09 Second chooser v2 reference on Blackwell (A/A), first 80 scenes
| Batch | v2 (13060758-59) | v2 again (13289312-13) | Diff [90%] | W/L |
|---|---|---|---|---|
| 1-40 | 0.793 | 0.797 | +0.004 [-0.050, +0.058] | 8/3 |
| 41-80 | 0.561 | 0.563 | +0.002 [-0.046, +0.052] | 3/7 |

- **On the same hardware the batch mean repeats within 0.005.** About a quarter of scenes flip, but the flips cancel.
- So the earlier "~0.10 spread" came from **different configs and different hardware**, not from rerunning one config:
  - v2 scored 0.716 on H100 vs 0.793 on Blackwell for batch 1-40.
  - Large per-batch differences between Blackwell configs (6 cameras: -0.112 / +0.188 / -0.145) are therefore mostly real config effects, not noise.
- **Open question this raises:** the official evaluation runs on H100, and v2 was 0.077 lower there on batch 1. Before choosing an image, confirm the chooser's gain also holds on H100.

## 2026-10-09 v2 rerun complete: 120-scene A/A on Blackwell
- v2 (13060758-60) 0.692 vs v2 again (13289312-14) **0.692**: diff -0.000 [-0.032, +0.032], W/L 16/15, 89 ties.
- By batch: +0.004 / +0.002 / -0.006.
- **Decision rule, revised:** on the same hardware, a 120-scene paired gain above ~0.03 is a real effect. This replaces the ~0.05 rule, which mixed hardware and configs.
- Earlier verdicts stand:
  - real: chooser v2 +0.054; hybrid -0.052;
  - zero: chooser v3, nav text, temperature, route pull;
  - 6 cameras (-0.023) is borderline negative.
- The last K=12 batch was moved to lane as_bw_b (13289885 cancelled, now 13298461) so it runs alongside 41-80.

## 2026-10-09 Why corridor exits survive the chooser, and route snap
**Analysis: v2 rerun choice logs, 120 scenes; plans vs route waypoints.**
- In 11 of 14 corridor exits, every one of the 6 samples is 3-12 m off the route for most of the scene. The model's samples all keep the lane, so no choice among them can fix this.
  - The recorded driver changed lane: 31f39b +3 to +6 m, d50172 +4 m.
  - Or it bore off: 524a02 -6 to -12 m, likely an exit.
  - Or it took a curve tighter than our plan: 451fc1 ~3 m.
- This also explains why route pull failed (-0.002): it only bent plan points 30 m+ ahead, and since the car replans every second the bend never got closer.
- K=12 on 41-80 confirms it: corridor 9 → 10. More samples don't add on-route options.

**Route snap (`A15_ROUTE_SNAP=<m>`, `route_choice.route_deviation` / `route_path`):**
- Measure the deviation from the route side: mean distance from the waypoints the plan reaches to the plan. This is not inflated by plan length, unlike the chooser cost at highway speed.
- If the deviation exceeds the threshold (half the threshold while a snap is under way, as hysteresis), drive a cubic Hermite path from the car (current heading) onto the first waypoint along the route's direction, then the route, re-timed with the model's chosen plan (`retime_path`). The model keeps control of speed.
- Offline calibration on the logged 8-point plans at 2 m:
  - 11 of 14 corridor-exit scenes trigger at least 3 times;
  - so do 18 of 68 passing scenes (partly from the coarse logged plans).
  - The open risk is what snapping does in those passing scenes.
- Tests: route_deviation/route_path/snap in `tests/test_route_nav.py`; the driver stores and counts snapped plans in `tests/test_driver_plan_storage.py`.

**Queued:**
- Route snap 2.0 m on 120 scenes: 13299355-57, lane as_bw_a, after test job 13299057 passes.
- Margin 0.5 (13299358-60) moved to lane as_bw_b, after K=12 batch 81-120 (13298461).

## 2026-10-10 Route snap 2.0 m, batch 41-80 (13299647), vs v2 0.561 (rerun 0.563)
- **0.632, +0.071 [-0.046, +0.197]**, W/L 7/10.
- Corridor 9 → 5. Now pass: 31f39b, 3b1027, 451fc1, 622908, d50172.
- New losses:
  - pass → offroad: b8fbf5;
  - pass → corridor: fb501e;
  - pass → low progress: b59db2, fa45af;
  - low progress → collision / offroad: 0a7b92 / 2cb4ca.
- Snap fired in 21 of 40 sessions, 75 replans in total.
- **K=12 batch 81-120 (13298461) was lost to a renderer crash** ("CUDA error: an illegal memory access", 4 of 40 sessions). Rerun as 13311170.
  - `sim_local.sbatch` now writes "INCOMPLETE" to driver_health.txt when fewer scenes drove than requested.

## 2026-10-10 Route snap 2.0 m, batch 81-120 (13299648): -0.042; guard added
- **81-120:** 0.681 vs v2 0.724; diff -0.042 [-0.129, +0.044], W/L 5/10.
  - Corridor 4 → 1: 524a02 now passes.
  - Low progress 8 → 11, offroad 2 → 4.
- **Two batches together:** +0.071 and -0.042.
- **Geometry at snap triggers** (from the choice logs):
  - Snaps that FIXED corridor exits (31f39b, 3b1027, 451fc1, d50172, 524a02, c7ed0d) were lane changes and exits on near-straight roads: the route heading at the first waypoint was within ±7°.
  - Snaps that BROKE passing scenes were mostly turns and U-turns, with a route heading of 60-180° (1845f6, a268c2, e7c20a, 8e056b, 36ea68). There the Hermite from the car to a waypoint ~40 m ahead cuts the corner, giving offroad or low progress.
- Some losses are scene-to-scene noise: 331b7f went pass → 0.01 with **0** snaps.
- **Guard (`A15_ROUTE_SNAP_MAX_DEG=15`, `route_choice.route_parallel`):** snap only when both the direction to the first waypoint and the route's direction there are within 15° of the car's heading.
  - On the logged triggers, this keeps 6 of the 8 fixes and removes every turn-related break.
  - Breaks still exposed: b8fbf5 (partly), 2cb4ca, fb501e, 281e1e, 0a7b92.
- Snap v1 batch 1-40 (13299649) was cancelled.
- **Queued:** guarded snap on 120 scenes, 13311489-91 (lane as_bw_a), after test 13311488 (the first test job, 13311461, died on node ins022 before starting).
- **Margin 0.5, batch 41-80 (13299358):** 0.620, +0.059 [-0.047, +0.170], W/L 6/9. Corridor 9 → 8, offroad 4 → 4. Batches 81-120 and 1-40 are running.
- **Margin 0.5, batch 81-120 (13299359):** 0.727 vs 0.724; diff +0.003 [-0.002, +0.008], W/L 4/4. Corridor 4 → 3, offroad 2 → 3. Two batches so far: +0.059, +0.003. Batch 1-40 is next.

## 2026-10-10 Guarded route snap (2.0 m, 15°), batch 81-120 (13311489)
| Variant | Mean | Diff vs v2 [90%] | W/L | Categories |
|---|---|---|---|---|
| v2 (reference) | 0.724 | | | pass 24, low progress 8, corridor 4, offroad 2, collision 2 |
| v2 rerun | 0.717 | -0.006 | 5/5 | |
| Snap, no guard | 0.681 | -0.042 | 5/10 | pass 21, low progress 11, corridor 1, offroad 4, collision 3 |
| **Snap + 15° guard** | 0.726 | **+0.003 [-0.055, +0.060]** | 5/7 | pass 23, low progress 10, corridor 2, offroad 3, collision 2 |

- The guard removed the turn breakage: snaps went from 82 to 41 replans, in 17 of 40 sessions.
- Kept fixes: 524a02 (corridor → pass) and c7ed0d (0.75 → 1.0).
- **331b7f is a coin-flip scene, not a snap effect.**
  - It scored 0.01 (car never moves) in the plain v2 rerun and in both snap runs, with 0 snaps.
  - It scored 1.0 in v2, margin 0.5, 6 cameras, hybrid and route pull.
  - Excluding it, guarded snap is about +0.03 on this batch.
- Batches 41-80 (13311490) and 1-40 (13311491) are next.

## 2026-10-10 Chooser margin 0.5 m, 120 scenes (13299358-60): no gain
- **0.706 vs v2 0.692: +0.013 [-0.031, +0.058]**, W/L 17/19. This is below the ~0.03 same-hardware threshold.
- By batch: -0.023 / +0.059 / +0.003.
- Categories: pass 69, low progress 25, corridor 12, offroad 10, collision 4. Corridor 14 → 12, offroad 8 → 10.
- **Kept margin 1.0.**

## 2026-10-11 Guarded route snap, batch 41-80 (13311490): +0.107
| Variant | Mean | Diff vs v2 [90%] | W/L | Categories |
|---|---|---|---|---|
| v2 | 0.561 | | | pass 18, low progress 7, corridor 9, offroad 4, collision 2 |
| v2 rerun | 0.563 | +0.002 | 3/7 | corridor 9 |
| Snap, no guard | 0.632 | +0.071 | 7/10 | corridor 5 |
| **Snap + 15° guard** | **0.668** | **+0.107 [+0.003, +0.220]** | 8/8 | pass 20, low progress 9, corridor 6, offroad 3, collision 2 |

- Snaps: 48 replans in 16 of 40 sessions.
- **Fixed:**
  - corridor → pass: 31f39b, 3b1027, 451fc1, d50172;
  - offroad → 0.94: 592fa2;
  - 624628: 0.65 → 0.92.
- **Broken by snapping:**
  - fb501e: pass → corridor, in both snap runs. The route is straight ahead (y≈0); worth a look.
  - 2cb4ca: low progress → offroad, 2 snaps.
- **Noise (0 snaps, same as the v2 rerun):** 481dfc offroad → pass; b59db2 and fa45af 1.0 → 0.96.
- **Two batches so far: +0.107, +0.003.** Batch 1-40 (13311491) is next.
- H100 v2 batch 41-80 (13260245) is running on ins049.

## 2026-10-11 fb501e, and a variant: snap only when the route leaves our lane
- **fb501e (pass → corridor in both snap runs):**
  - The route stays in our lane: first waypoint at y = 0-0.3 m.
  - For 4 replans the model planned a ~3 m swerve left, plausibly around something in the road. Snap overrode it.
- **The fixes are different:** in 31f39b, 3b1027, d50172, 524a02 and c7ed0d the route's first waypoint is 0.7-3.4 m to the side, i.e. the recorded driver changed lane or took an exit.
- **Variant `A15_ROUTE_SNAP_MIN_OFFSET=0.5`** (`route_choice.route_lateral_offset`):
  - A new snap needs the route's first waypoint ≥0.5 m to the side. A snap already under way continues, so lane changes are finished.
  - On the logs, this keeps 5 of the 6 fixes; it loses 451fc1, a curve where the route is at y≈0.
- **Caveat:** this rule is chosen on bench scenes, so it is at risk of overfitting. It is kept only if it is clearly better over 120 scenes, and the physical reason (respect the model's swerves when the route stays in lane) has to carry it.
- Tests pass (13328313).
- **Queued:**
  - Lane as_bw_b: 13328318-20, ahead of the K=12 rerun (re-queued as 13328321).
  - Lane as_bw_a: guarded snap batch 1-40 (13311491) is still pending.

## 2026-10-11 Chooser v2 on H100 vs Blackwell (same config, CUDA_COMPAT on H100)
| Batch | Blackwell v2 (rerun) | H100 v2 | Diff [90%] | W/L |
|---|---|---|---|---|
| 1-40 (13260244) | 0.793 (0.797) | 0.716 | -0.077 [-0.175, +0.001] | 6/9 |
| 41-80 (13260245) | 0.561 (0.563) | 0.546 | -0.015 [-0.041, +0.000] | 3/6 |

- H100 runs lower on both batches: about -0.046 over 80 scenes. Batch 81-120 (13260246) is running.
- This is consistent with v0.3: 0.638 on the local Blackwell bench vs 0.611 official (H100), a gap of -0.027.
- **Reading:** the Blackwell bench reads ~0.03-0.05 high in absolute terms. What matters is whether gains carry over to H100. Before a submission, run the chosen candidate on H100 as well, paired against this H100 v2.
- **H100 v2 batch 81-120 (13260246):** 0.723 vs 0.724 on Blackwell (W/L 4/4).
- **Full 120 scenes: H100 v2 0.662 vs Blackwell v2 0.692, -0.031 [-0.064, -0.001].** This matches the v0.3 bench-vs-official gap (-0.027).
- **Queued, transfer check:** guarded snap (2.0 m, 15°) on H100, 13329881-83 (lane as_h100_a, CUDA_COMPAT=1), paired against this H100 v2 (0.662).

## 2026-10-11 Guarded route snap on H100 (transfer check)
- **Batch 81-120 (13329882): 0.770 vs H100 v2 0.723, +0.048 [-0.027, +0.129]**, W/L 10/3.
  - The same fixes as on Blackwell: 524a02 corridor → pass, c7ed0d 0.74 → 1.0.
  - Also on H100: 36ea68 corridor → pass, e15688 0.82 → 1.0.
  - Losses: 281e1e 1.0 → 0.75 (on Blackwell too), and the coin-flip 331b7f.
- **Batch 41-80 (13329881) is void:** a GPU fault on ins049 at 08:22 ("CUDA error: unspecified launch failure" inside the action expert); every inference failed after that. 8 of 40 scenes drove.
  - The new INCOMPLETE flag caught it.
  - The next batch on the same GPU was clean, so this is not the snap code.
  - Rerun as 13332088.
- **H100 batch 1-40 (13329883):** 0.716 vs H100 v2 0.716, +0.000 [-0.076, +0.076], W/L 8/7.
  - Corridor 1 → 2: adfc75 fixed, b3968d broken. Collisions 3 → 2.
  - Snaps: 53 replans in 15 of 40 sessions.
  - This batch has little corridor exposure, so a neutral result means no harm here.
- **Lane moves:**
  - Both Blackwell nodes have been held by other users' long jobs for ~1 day.
  - The min-offset variant moved to H100 (13328318-20 cancelled; now 13333300-02 on a second H100 lane, as_h100_b).
  - It is paired against H100 guarded snap and H100 v2.
- **Blackwell jobs cancelled to keep a two-GPU footprint while H100 carries both lanes:**
  - guarded snap 1-40 on Blackwell (13311491): H100 covers 1-40;
  - the K=12 rerun (13328321): low priority. K=12 so far: +0.011 / +0.037 on 1-40 / 41-80, at +15% wall time.

## 2026-10-11 **Guarded route snap, 120 scenes on H100: +0.053 [+0.004, +0.101]**
| H100, 120 scenes | Mean | Diff [90%] | W/L | Rear-ended | Categories |
|---|---|---|---|---|---|
| v2 (13260244-46) | 0.662 | | | 22 | pass 63, low progress 26, corridor 16, offroad 9, collision 6 |
| **v2 + snap 2.0 m / 15° (13329882-83, 13332088)** | **0.715** | **+0.053 [+0.004, +0.101]** | **27/14** | 18 | pass 69, low progress 27, **corridor 9**, offroad 8, collision 7 |

- **By batch:** +0.000 / **+0.112 [+0.027, +0.210]** / +0.048. Batch 41-80 holds most corridor exits.
- **Corridor exits 16 → 9.** Fixed on H100: 31f39b, 3b1027, 451fc1, b8f636, d50172 (41-80), 524a02, 36ea68 (81-120), adfc75 (1-40). Most of these were also fixed on Blackwell.
- **Costs:**
  - two corridor exits became at-fault collisions (0a7b92, 5688dd);
  - 2cb4ca went from low progress to offroad;
  - 1-40 is neutral.
  - At-fault collisions 6 → 7 matter for the leaderboard's at-fault incident spacing, where we currently lead (0.795).
- **Blackwell, the two batches run:** +0.107 (41-80), +0.003 (81-120).
- **Second gain above noise this month, the first since chooser v2 (+0.054).** No extra inference, so no time-budget cost.
- **Candidate image contents:** chooser v2 + `A15_ROUTE_SNAP=2.0 A15_ROUTE_SNAP_MAX_DEG=15`. Not built or pushed; submitting is Aman's call, no earlier than mid-October.
- **Still running:** the min-offset variant (13333300-02), paired against this run on H100.

## 2026-10-11 Min-offset variant on H100, batch 41-80 (13333300)
- **vs H100 v2: 0.707 vs 0.546, +0.160 [+0.062, +0.269]**, W/L 9/6.
- **vs H100 guarded snap: +0.048 [-0.028, +0.126]**, W/L 7/5.
- Changes vs guarded snap:
  - 0a7b92: at-fault collision → 0.96 (the rule kept the model's own move);
  - 592fa2, 62e0a7: corridor → pass;
  - 624628: 0.93 → 1.0;
  - b8f636: pass → corridor (lost; the route there is in lane);
  - 3b1027: 1.0 → 0.89.
- Snaps: 41 replans in 15 sessions.
- The rule was chosen on these scenes, so the other two batches decide. 81-120 (13333301) and 1-40 (13333302) are next.
- **Min-offset variant, H100 batch 81-120 (13333301):**
  - 0.738 vs H100 v2 0.723: +0.016.
  - **vs guarded snap 0.770: -0.032 [-0.081, -0.003]**, W/L 4/8.
  - Lost 36ea68 (pass → corridor, a fix that needed a snap with the route in lane), 57016f and e15688.
  - **Two batches vs guarded snap: +0.048, -0.032. No better so far, as expected for a rule picked on the same scenes.** 1-40 (13333302) is running.

## 2026-10-11 Snap threshold 1.5 m (15° guard) on H100, batch 41-80 (13335147)
- **vs guarded snap 2.0 m: 0.698 vs 0.659, +0.039 [+0.001, +0.090]**, W/L 6/3.
- vs H100 v2: +0.152.
- Changes vs 2.0 m:
  - 0a7b92: at-fault collision → 0.53;
  - 88b571: offroad → pass;
  - b59db2 and fa45af: low progress → pass.
- No losses.
- 81-120 (13335148) and 1-40 (13335149) are next.
- **Min-offset variant, H100 batch 1-40 (13333302) is void.** The same driver fault: "CUDA error: unspecified launch failure" at 22:41 on ins050; 8 of 40 scenes drove. Rerun queued (lane as_h100_b).
- **H100 driver fault rate: 2 of 10 H100 batches** (ins049 on 10-11 08:22, ins050 on 10-11 22:41). It has not happened in ~30 Blackwell batches.
  - Our H100 runs use the CUDA 12.8 forward-compat libraries on driver 550, so this may be a compat-layer problem rather than something the official H100s (proper driver) would see.
  - Official v0.3 ran cleanly there.
  - It strikes inside the action expert's kernels, not in the numpy-only snap code.
  - **Before a submission, a CUDA_LAUNCH_BLOCKING run might pin down the kernel.**
- **Snap 1.5 m, H100 batch 81-120 (13335148):** 0.728 vs 2.0 m 0.770: **-0.042 [-0.111, +0.012]**, W/L 4/8.
  - Corridor exits back in 36ea68 and 447a3b.
  - 36ea68 flips between snap variants, so it is unstable.
  - **Two batches vs 2.0 m: +0.039, -0.042. No better; 2.0 m stays.** 1-40 (13335149) is running, for completeness.

## 2026-10-12 Min-offset variant, full 120 on H100: not adopted
| H100, 120 scenes | Mean | vs H100 v2 [90%] | W/L | vs guarded snap [90%] | W/L |
|---|---|---|---|---|---|
| Guarded snap (2.0 m, 15°) | 0.715 | +0.053 [+0.004, +0.101] | 27/14 | | |
| + min offset 0.5 m | 0.724 | +0.062 [+0.012, +0.113] | 24/20 | **+0.009 [-0.027, +0.046]** | 17/21 |

- vs guarded snap by batch: +0.048 / -0.032 / +0.011.
- Below the ~0.03 bar. The rule was chosen on these scenes and wins fewer scenes than it loses. **Kept the simpler guarded snap** (Dockerfile.v6 has `A15_ROUTE_SNAP_MIN_OFFSET=0`).
- **Queued:** a second guarded-snap run on H100, 120 scenes (lane as_h100_b). It checks that the candidate's +0.053 repeats, and measures H100 run-to-run spread.

## 2026-10-12 Snap 1.5 m, full 120 on H100: no better than 2.0 m
- 0.706 vs 2.0 m 0.715: **-0.009 [-0.050, +0.030]**, W/L 18/19. By batch: +0.039 / -0.042 / -0.024.
- vs H100 v2: +0.044.
- **2.0 m stays.**
- **Queued:** v0.3 (submitted config: K=1, 1 Hz, no chooser) on H100, 120 scenes (lane as_h100_a). Paired with v0.3's official 0.611, this calibrates H100-local → official for the candidate.
- **H100 guarded-snap repeat, batch 41-80 (13340993):**
  - 0.628 vs the first run's 0.659: -0.031 [-0.103, +0.034], W/L 6/5.
  - vs H100 v2: **+0.082** (first run +0.112).
  - H100 run-to-run spread is ~0.03 per 40 scenes, larger than Blackwell's ~0.005. The gain holds.

## 2026-10-12 v0.3 on H100, batch 41-80 (13341063): **0.603**
| H100, batch 41-80 | Mean | vs H100 v0.3 [90%] | W/L |
|---|---|---|---|
| v0.3 (submitted config) | 0.603 | | |
| v2 | 0.546 | -0.057 [-0.133, +0.015] | 4/8 |
| v2 + guarded snap, run 1 | 0.659 | +0.056 [-0.065, +0.183] | 8/9 |
| v2 + guarded snap, run 2 | 0.628 | +0.025 [-0.100, +0.153] | 9/9 |

- **On this batch, v2 is below v0.3 on H100**, as it was on Blackwell (0.561 vs 0.579). v2's +0.054 over v0.3 on Blackwell came from batches 1-40 and 81-120.
- **Whether chooser v2 (and so v6) beats v0.3 on H100 is open until v0.3 batches 81-120 (13341064) and 1-40 (13341065) land.** This decides the submission candidate.
- v0.3 on H100 vs v0.3 on Blackwell, this batch: 0.603 vs 0.579 (+0.024).
- **Queued hedge:** v0.3 + guarded snap (K=1, no chooser), 13347658-60 on lane as_h100_a after the v0.3 batches. Snap uses the session route whatever K is, so it works without the chooser.
- **Third H100 driver fault:** guarded-snap repeat batch 81-120 (13340994), "unspecified launch failure" at 07:04 on ins049; 8 of 40 drove. Rerun queued.
- **Fault tally:**
  - H100: 3 of 14 batches, all three in snap configs (3 of 10 snap batches vs 0 of 4 non-snap; Fisher p ≈ 0.33).
  - Blackwell: 0 in ~5 snap batches and ~30 others.
- **Physics-service errors are not the cause.** Healthy runs have them too (H100 v2 1-40: 175 tracebacks; H100 v0.3 41-80: 57), and in the faulted runs most of them come after the driver fault.
- **Reading: an H100-on-this-cluster problem (CUDA 12.8 forward-compat on driver 550), not snap.**
  - Two of the three faults came ~36 min into the job.
  - A CUDA_LAUNCH_BLOCKING diagnostic would change plan timing (the driver serves cached plans while inferring), so it belongs in a separate run, not in a measurement.

## 2026-10-12 Qwen-Drive-1.0 (pointer from the alpasim-jepa thread): not prioritised for PAI
- **What it is (verified):**
  - HF Qwen/Qwen-Drive-1.0-4B, Apache-2.0, ungated, created 2026-08-27.
  - Weights: 9.1 GB VLM + 2.1 GB planner-sft + 2.1 GB planner-rl + 0.5 GB perception.
  - GitHub QwenLM/Qwen-Drive-1.0 and arXiv 2609.00111 both exist.
- **Its own card reports AlpaSim closed loop (at-fault score, higher is better):** Alpamayo-1.5 0.45, Qwen-Drive SFT 0.27, RL 0.37.
- **PAI-AV open-loop ADE 3 s / 5 s:** Alpamayo 0.35 / 1.05; Qwen-Drive SFT 0.37 / 1.07; RL 0.42 / 1.11.
- **Verdict:** on PAI, Alpamayo's home data, it is behind by its authors' own numbers. Not worth an adapter plus a 14 GB download ahead of finishing the snap candidate. It may matter more for nuPlan; told the jepa thread.
- **Storage:** the dept fileset is at 99% (64 GB free of 5 TB). Our running jobs write up to ~16 GB of .asl each until they finish.

## 2026-10-12 Space cleanup in alpasim/ (Aman approved both items)
- **Inventory (job 13357506, alpasim only):** 312 GB in total.
  - `data/fs/nre-artifacts` 204 GB: the 120 bench-scene USDZs, all needed.
  - `sandbox` 46 GB: needed, and shared read-only with the jepa thread.
  - `data/nuplan_track` 43 GB: read by the jepa thread.
  - `runs` 21 GB.
- **Deleted:**
  - `runs/local_pai_a15_hz2rcv2_81_120_13160659` (8.2 GB): incomplete 2 Hz batch, no summary, experiment rejected.
  - 100 raw `.asl` logs from `runs/local_nuplan_a15_bench_12991555` (2.7 GB); its summary and metrics are kept.
- About 11 GB freed. Nothing else in alpasim/ is safely removable.

## 2026-10-18 H100: v0.3 and v0.3 + guarded snap
| H100 | v0.3 | v2 | v2 + snap | **v0.3 + snap (K=1)** |
|---|---|---|---|---|
| 1-40 | (running: 13376797) | 0.716 | 0.716 | 0.692 |
| 41-80 | 0.603 | 0.546 | 0.659 / 0.628 | (void; rerun 13532770) |
| 81-120 (13341064, 13347659) | 0.670 | 0.723 (+0.053) | 0.770 (+0.101) | **0.797 (+0.127 [+0.050, +0.217], W/L 11/3)** |

- **Snap helps on top of either base.** v2 vs v0.3 is +0.053 on 81-120 and -0.057 on 41-80; batch 1-40 is still pending on H100.
- **v0.3 + snap vs v2 + snap so far:** +0.027 (81-120), -0.024 (1-40). A tie so far, and v0.3 + snap uses K=1: inference 1.6 s vs 4.4 s, so a much safer time budget.
- **Lost batch:** v0.3 + snap 41-80 (13347658, ins048). The fourth H100 "unspecified launch failure", at the first inference, 0 of 40 scenes. **H100 faults: 4 of 18 batches.** This one is in a non-chooser config. Blackwell: still 0.
- **Lane b was idle 10-13 → 10-18 (my error).** `--dependency=singleton` also waits for EARLIER pending jobs of the same name, including held ones. Holding 13340995/13348532 therefore blocked 13376797 forever once the release script died with the session.
  - Fixed by clearing its dependency.
  - Use `afterany:<id>` chains, not singleton + hold.
- **Dept fileset at 23 GB free.** One job at a time, since each writes up to ~16 GB of .asl before cleanup.

## 2026-10-18 Raw rollout logs on node-local scratch (Aman approved)
- `sim_local.sbatch` now points `$RUN/rollouts` at `$TMPDIR/alpasim_rollouts` (/local/slurm-job-N, 200-700 GB). `compose_apptainer.py` binds TMPDIR into every service at the same path.
- At exit (normal end or trap), everything except `*.asl` is rsynced back. `KEEP_ASL=1` copies those too, and `LOCAL_ROLLOUTS=0` disables the redirect.
- Falls back to the shared dir if TMPDIR is under /tmp (not bound) or has < 40 GB free.
- Jobs queued before this change (13376797, 13532770) still use the old script.
- 2-scene smoke test: 13532879.

## 2026-10-18 Storage dedupe with alpasim-jepa (Aman's terms)
- Each side reads the other's folder. A duplicate is deleted in one place only, each item is approved by Aman, and nobody edits the other's folder.
- Sent alpasim-jepa our inventory.
- Agreed (pending Aman's approval and their sha256 check): they replace their copies of `data/nuplan_track/nuplan_test` (~5 GB) and `navtest/configs` with read-only symlinks into ours.
- **Do not delete `data/nuplan_track` or `sandbox/` from alpasim/: alpasim-jepa reads them.**
- **Dedupe done (Aman approved, 2026-10-18).**
  - alpasim-jepa verified 85,035 + 1,491 files byte-identical (sha256), deleted its copies (~4.8 GB) and now **symlinks** to our `data/nuplan_track/nuplan_test` and `data/nuplan_track/navtest/configs`.
- **Paths in alpasim/ that alpasim-jepa reads. Ask alpasim-jepa and Aman before deleting, moving or rewriting any of them:**
  - `data/nuplan_track/nuplan_test`, `data/nuplan_track/navtest/configs` (symlink targets);
  - `data/nuplan_track/navtest/assets`;
  - `sandbox/alpasim-base`, `sandbox/nre-ga_26.04`;
  - `external/alpasim`;
  - `envs/overlay_driver`, `envs/overlay_torchao`;
  - `src/summarize_run.py`, `src/compare_runs.py`;
  - `.cache/hf/token`;
  - for comparison only: `runs/local_nuplan_a15_bench_*`, `runs/local_pai_a15_bench1_12991558`, `data/fs/nre-artifacts/all-usdzs`.

## 2026-10-18 Fileset hit its quota; v0.3 1-40 on H100 void
- **13376797 (v0.3, H100, batch 1-40) is void.**
  - At 15:37 the dept fileset hit its quota ("OSError: [Errno 122] Disk quota exceeded") while the runtime was writing rollouts. 20 of 40 rollouts were saved, and no summary was written.
  - This job used the pre-node-local script, so its raw .asl went to the shared fileset (8.2 GB left there).
  - The fileset is now at 14 GB free.
- **Cancelled 13532770** (v0.3 + snap 41-80 rerun, old script), so it doesn't do the same.
- From now on every sim uses the node-local script (smoke test 13533635 running).
- Asked Aman about deleting the void run's 8.2 GB of raw logs.
- Told alpasim-jepa about the quota.
- **Node-local smoke test passed (13533635, H100, 2 scenes).**
  - Rollouts were written to /local/slurm-job-N (247 GB free) and copied back without .asl.
  - Summary, metrics and videos are all there, and the run dir on the fileset is 6.7 MB.
- **Requeued with the node-local script:** v0.3 1-40 (13534919, lane as_h100_d) and v0.3 + snap 41-80 (13534920, lane as_h100_e). They run in parallel; each puts only ~100-200 MB on the fileset.
- Cancelled the stale held old-script snap repeats (13340995, 13348532).
- Deleted the void run's raw logs (13376797, 8.2 GB), Aman approved.

## 2026-10-18 v0.3 on H100 complete (13534919 = rerun of 1-40): chooser v2 adds nothing on H100
| H100, 120 scenes | Mean | vs H100 v0.3 [90%] | W/L |
|---|---|---|---|
| v0.3 (submitted config) | 0.657 | | |
| v2 (chooser) | 0.662 | +0.005 [-0.044, +0.054] | 15/24 |
| v2 + guarded snap | 0.715 | **+0.058 [-0.003, +0.122]** | 22/26 |

- **By batch, v2 + snap vs v0.3:** +0.019 / +0.056 / +0.101. v0.3 batch 1-40 scored 0.697.
- **On H100 the chooser's Blackwell gain (+0.054) does not appear; snap carries the gain.**
- **Calibration:** H100 local v0.3 is 0.657 vs official 0.611, so local reads ~0.046 high.
- v0.3 + snap batch 1-40: 0.692 vs v0.3 0.697 (-0.005). Batch 41-80 rerun (13534920) is running; it decides v0.3 + snap vs v2 + snap.
- Fileset back to 22 GB free. Node-local runs leave ~100-130 MB each on the fileset.

## 2026-10-18 Waypoint-memory snap (Aman: hold v5, build this)
- **Idea:** the route covers ~40-80 m ahead, so the stretch where a turn or lane change happens is missing from any single route, and route snap's Hermite bridge cuts corners. That is why the 15° guard exists, and why missed turns are left.
  - That stretch was in the routes received a few seconds earlier.
  - Route data only: the GroundTruth message is never used.
- **How:** `route_choice.RouteMemory` keeps every route's waypoints in the world frame (pose interpolated at `route.timestamp_us`) as one ordered path. A new route replaces the remembered tail from its first waypoint on.
  - `path_in_frame` gives the path from the car's nearest point onwards, in the plan's anchor frame.
  - A car beside the path joins it with a smoothstep over `clip(2.5 s × speed, 15, 60)` m.
  - It returns None if the path does not reach the car (nearest point > 8 m, e.g. at session start); the guarded Hermite snap is then the fallback.
- **Driver:** with `A15_ROUTE_MEMORY=1`, snap measures deviation against the memory path and drives along it.
  - The angle guard applies only to the fallback, so turns can be snapped.
  - Logs `memsnap` and counts `snapped (memory N)`.
- **Tests (13536531):**
  - the memory path lies on a recorded 90° turn (0.00 m), which no single route shows;
  - the offset join starts at the car and is back on the path by 25 m;
  - the driver gets 4 memory snaps via `submit_route`.
- **Queued:** v0.3 base + guarded snap + memory, H100 batch 41-80 (13536545). It is compared against v0.3 + snap (13534920) and v0.3 on the same batch.

## 2026-10-18 **v5 = v0.3 base + guarded route snap** (H100, 120 scenes complete)
| H100, 120 scenes | Mean | vs v0.3 [90%] | W/L |
|---|---|---|---|
| v0.3 (submitted config) | 0.657 | | |
| v2 + guarded snap | 0.715 | +0.058 [-0.003, +0.122] | 22/26 |
| **v0.3 + guarded snap** (13347659-60, 13534920) | **0.728** | **+0.071 [+0.011, +0.131]** | **24/18** |

- **v0.3 + snap by batch:** -0.005 / +0.093 / +0.127.
- **vs v2 + snap:** +0.013 [-0.029, +0.056], W/L 20/16. A tie, but v0.3 + snap uses one sample: the same compute as the officially proven v0.3 (9,474 s of 14,400 s).
- **`container/Dockerfile.v5`:** now `A15_NUM_SAMPLES=1`, no chooser, `A15_ROUTE_SNAP=2.0 A15_ROUTE_SNAP_MAX_DEG=15`, `A15_ROUTE_MEMORY=0`.
- **Expected official result:** local H100 v0.3 reads 0.046 above official, so a scene score of roughly 0.68.
- Not built or pushed. Aman is holding the submission (2026-10-18) while waypoint memory is tested.

## 2026-10-19 Memory snap, two of three batches; `nureasoning/ext/a15env` deleted; fifth H100 fault
- **Memory snap (v0.3 base + guarded snap + `A15_ROUTE_MEMORY=1`) vs v5 (v0.3 + snap), H100:**
  - **41-80 (13536545):** 0.673 vs 0.696, -0.023 [-0.127, +0.081], W/L 5/7. 69 of the 83 snaps used memory.
    - Fixed: 622908 (the turn plain snap can't do), 62e0a7, 7ca7da, 0a7b92.
    - Broke: 31f39b, fb501e (corridor), 481dfc, 88b571 (offroad), 90153b (collision).
  - **1-40 (13536874):** 0.727 vs 0.692, +0.035 [-0.070, +0.140], W/L 13/4.
  - **81-120 (13536873) void:** fifth H100 "unspecified launch failure" (16 of 40 drove). **H100 faults: 5 of 22 batches.**
  - **Two batches: about +0.006 vs v5, a tie so far.**
- **Fault cause, correction:** CUDA_COMPAT applies only to the simulator services (compose_apptainer.py). The **driver runs natively on driver 550** with torch 2.8 cu128, so the earlier "compat layer" reading does not explain driver faults.
  - Newer hypothesis: cu128 torch on driver 550 through CUDA minor-version compatibility.
  - Still cluster-specific: Blackwell (driver 575) has 0 faults in ~35 batches.
- **`nureasoning/ext/a15env` was deleted at 2026-10-19 02:19** (ext/ mtime).
  - Not by the alpasim-jepa or jepa-nureasoning threads, by their own account. GPFS snapshot dirs are empty for us.
  - Every local sim runs the driver with that env, so **local sims are blocked.** The 81-120 rerun (13541844) was cancelled before it started.
  - Our dependency was in alpasim/BORROWED.md but not in nureasoning/SHARED_DEPENDENCIES.md.
  - Untouched: `ext/alpamayo1.5`, the HF hub weights, `.cache/uv` (7.3 GB), `.cache/uvpy`.
  - The submission images are unaffected; they carry their own env.
- **Fileset at 6.5 GB free** (02:20), with nothing of ours running.
- **Recovery (Aman approved):**
  - The nureasoning session deleted a15env (and a15ftenv) in an approved cleanup, not knowing about our use.
  - Every a15 job now runs `cluster/a15env_local.sh`, which builds the exact 64-package set from the old env's install log (`envs/a15env-pins.txt`, Python 3.12.14, `--no-deps`) under $TMPDIR with a node-local uv cache.
  - `ext/alpamayo1.5/src` goes on sys.path via a .pth file. Nothing is written to nureasoning/ or the fileset.
  - Equivalence check (`cluster/check_env_equiv.sbatch`, L40S): max trajectory difference vs references saved with the old env (tmp/chk_t10.npz K=1, tmp/chk_k6.npz K=6), plus the driver tests.
- **v5 image built and verified on the Mac (`local/pai:v5`):**
  - from `container/Dockerfile.v5` on base v0.3 (digest 29a4f9c2, the registered v0.3), git bad9300;
  - code in /app/a15_driver byte-identical to the repo (5 files, sha256);
  - settings `A15_NUM_SAMPLES=1 A15_ROUTE_SNAP=2.0 A15_ROUTE_SNAP_MAX_DEG=15 A15_ROUTE_MEMORY=0`;
  - all driver tests pass inside the image (amd64);
  - config digest sha256:e4d56dc9…, manifest list sha256:6ef89f7b….
- **Env equivalence passed (13547676, L40S).**
  - Rebuilt env: torch 2.8.0+cu128, transformers 4.57.1, built in 756 s on node-local disk.
  - It reproduces the old env's trajectories exactly: **max |diff| 0.000000 m** over 12 clips, at K=1 (vs chk_t10) and K=6 (vs chk_k6). Same 14,236 MiB.
  - Driver tests pass. Local sims are unblocked; the rebuild adds ~12.5 min per job.
- Memory-snap batch 81-120 rerun queued on H100.
- **Memory-snap bug found in batch 41-80's `memsnap` logs: the remembered path is corrupted.**
  - It doubles back: 31f39b at 21:04:46 runs forward to 54 m, then back to 30 m.
  - It swings sideways between replans: +2.8 m to -3.2 m, and once -22 m, where the recorded driver only changed lane ~4 m left. fb501e zigzags the same way.
  - Likely cause: consecutive routes don't register in the world frame (route timestamp vs pose time base, or heading). The "no merge point within 3 m → append" branch then builds backward jumps.
  - **So the 41-80 result (-0.023 vs v5) does not measure the idea.**
  - Added `A15_ROUTE_MEM_DEBUG=1` logging of route time, pose window, pose, first waypoint and merge distance. Debug run on 31f39b, fb501e, 622908: 13547834.

## 2026-10-19 Memory snap v1: full 120 on H100, the bugs, and memory snap v2
- **v1 (buggy) vs v5, H100, 120 scenes: +0.003 [-0.051, +0.057]**, W/L 22/20. By batch: +0.035 / -0.023 / -0.003 (81-120 rerun 13547828).
- **Debug run (13547834, 31f39b / fb501e / 622908, 597 route updates):**
  - Registration is fine: route timestamps equal the latest pose time, and merge distance has median 1.13 m and p95 1.5 m (about one waypoint spacing).
  - The 4 misses of 3-12 m all came after the car had already left the route (31f39b at y=-37 m, heading -43°). The "append" branch then made the path double back.
  - The main problem was behavioural. With the memory path, the deviation included points 0-40 m ahead, so a 1.5-2 m in-lane offset from the recorded line triggered snaps. The car then weaved between the model's line and the recorded one, and consecutive memory paths flipped ±3 m at 40 m as the heading swung.
- **v2 fixes:**
  - A route that does not continue the memory restarts it, no append.
  - Snap deviation uses only memory points ≥ 20 m ahead (`A15_ROUTE_MEM_DEV_MIN=20`). The snap path still uses the whole memory, so turns are followed.
- **Queued:** v2 on 120 scenes, H100 (13554873-75). Test 13554747 hit its 10-min limit (the env build takes ~13 min); test_driver.sbatch is now 45 min. The sims run the same driver tests first anyway, so they are queued without that dependency.

## 2026-10-20 Memory snap v2 vs v5 (H100): not better
| H100 | v5 (v0.3 + snap) | memory v1 | memory v2 |
|---|---|---|---|
| 1-40 | 0.692 | 0.727 (+0.035) | 0.701 (+0.010, W/L 13/3) |
| 41-80 | 0.696 | 0.673 (-0.023) | **void**: 6th H100 launch failure (13554873, 0 of 40 drove) |
| 81-120 | 0.797 | 0.794 (-0.003) | **0.719 (-0.078 [-0.184, +0.027], W/L 3/12)** |

- **v2 on 81-120, fixed:** 3defec (turn; v1 also fixed it), e15688, and the coin-flip 331b7f.
- **v2 on 81-120, broken:**
  - 1845f6 (pass → offroad; a turn scene that unguarded snap also broke);
  - ec6f99 (pass → offroad);
  - 36ea68 and 57016f (→ at-fault collision);
  - df6f61 (→ corridor).
- **Reading:** following the remembered path through turns fixes some missed turns (622908, 3defec) but sends others offroad or into collisions, likely because the path is driven at the model's speed. Net effect: a tie (v1) or worse (v2). **No memory-snap variant beats v5.**
- **H100 faults: 6 of 25 batches.**
- **Per-job env build:** 756-1486 s (uv downloads at variable speed).

## 2026-10-20 Aman: "do all three plus the cache"
1. **Submit v5:** built and verified (`local/pai:v5`). The push and submit need Aman's ECR login and challenge CLI (`ecr-login`, push, submit), as in September; handed to him.
2. **Memory snap v3 = v2 + curvature speed cap** (`route_choice.cap_speed_on_path`, `A15_SNAP_ALAT=3.0`), only on memory snaps.
   - The model's speed is capped at sqrt(a_lat / curvature).
   - The cap is propagated back at 3 m/s² braking, so the car slows before a curve instead of braking hard at plan start. On a test turn (R = 15 m): first step at the model's 15 m/s, deceleration ≤ 3.07 m/s², lateral ≤ 3.74 m/s².
   - Tests added. Queued on 120 H100 scenes: 13569422-24 (test 13569421).
3. **Report:** draft started (`report/`).
4. **Cache:** `cluster/a15env_local.sh` keeps uv's wheel cache and Python in `alpasim/cache/{uv,uvpy}` (shared, ~3 GB). Jobs copy from it instead of downloading; the env stays node-local.
- The fileset is at 86 GB free.

## 2026-10-20 Memory snap v3 (+ curvature speed cap), 120 on H100: worse than v5 — memory snap dropped
| H100, 120 scenes | mean | vs v5 [90%] | W/L | categories |
|---|---|---|---|---|
| v0.3 | 0.657 | | | pass 67, low progress 22, corridor 23, offroad 4, collision 4 |
| **v5 (v0.3 + guarded snap)** | **0.728** | | | pass 75, low progress 22, corridor 13, offroad 4, collision 6 |
| memory v3 (13569422-24) | 0.702 | -0.026 [-0.087, +0.034] | 23/26 | pass 66, low progress 29, corridor 10, offroad 8, collision 7 |
- By batch -0.044 / +0.022 / -0.057. All 3 batches clean (no H100 fault this time).
- Memory fixes more corridor exits (13 -> 10: 622908, 62e0a7, 733560, e64716, 392963 pass) but the speed cap turns passes into low progress (22 -> 29) and offroad/collisions rise (adfc75, d50172, 1845f6 offroad; 3ac7da, 892966, 90153b collisions).
- **Verdict across v1/v2/v3: waypoint-memory snap does not beat v5 (+0.003, ~-0.03, -0.026). Dropped; v5 stays the PAI entry.**
- Shared uv cache works: env build 2402 s (first job, filling the cache), then 59 s, 4 s, 2 s.

## 2026-10-21 **Official PAI submission: v5** (Aman asked me to submit)
- CLI session re-authenticated by Aman (`configure-token`). Before submitting: `limits` 3/3 left, terms accepted (official-2026-08, captain accepted), `ecr-login` OK, tag `v5` absent from the registry.
- Pushed `teams/amanupg:v5`, digest **sha256:6ef89f7b858f5d9c11bb9508c84b43161d9a05fb569a165b4406d85e4af4bdfc**. This is the verified build: git bad9300, `A15_NUM_SAMPLES=1 A15_ROUTE_SNAP=2.0 A15_ROUTE_SNAP_MAX_DEG=15 A15_ROUTE_MEMORY=0`. Only the code layer uploaded.
- **Submitted: `submission_id f594cf5f-77bc-4ace-8785-315315a502e2`, track pai, default controller gains, QUEUED (position 1 of 1).**
- October slots after this: 2 left (shared with nuPlan).
- Expected: local H100 0.728 vs v0.3 0.657 (+0.071); official v0.3 was 0.611, so roughly 0.68 if the local bias (~+0.046) holds.
- **v5 official run SUCCEEDED** (f594cf5f): evaluation 04:12-07:27 UTC; coverage 987/987; wall time **9,522 / 14,400 s** (v0.3: 9,474); image 21.5 GB.
  - The status `score` field is distance to the GT trajectory (v0.3's 2.177 equals its leaderboard `dist_to_gt_trajectory`). **v5: 2.058 vs v0.3 2.177**, so the car stays closer to the recorded path, as route snap intends.
  - The PCS and scene score were not yet on the leaderboard at 07:32 UTC (PCS is recomputed jointly across all submissions).
- **Leaderboard 07:32 UTC (PCS has shifted with the newer entries):**
  - ACM 2259 (0.665 / 0.53);
  - VAIL 2231 (0.722 / 1.301);
  - ACM's 3rd entry 2219 (0.738 / 0.885);
  - team-rvs 2195 (0.722 / 0.749);
  - **our v0.3 2146, rank 8** (0.611 / 0.795).

## 2026-10-21 **v5 official result: PCS 2184, 4th team on PAI**
| PAI official | PCS [95%] | rank interval | avg scene score | at-fault distance | dist to GT |
|---|---|---|---|---|---|
| v0.3 (2026-09-29) | 2150 [2104-2196] | 5-18 | 0.611 | 0.795 | 2.177 |
| **v5 (f594cf5f)** | **2184 [2137-2230]** | **2-11** | **0.652** | 0.771 | 2.058 |

- **Gain over v0.3:** scene score +0.041, PCS +34.
  - Local H100 predicted +0.071, so the official gain is ~58% of the local one.
  - Our "local reads ~0.046 high" estimate predicted ~0.68; the actual is 0.652.
  - At-fault distance is slightly lower (0.795 → 0.771).
- **Team order by best entry:**
  1. ACM 2256 (0.665 / 0.53; their newer 0.738 / 0.885 entry is 2216);
  2. VAIL 2230 (0.722 / 1.301);
  3. team-rvs (host) 2192 (0.722 / 0.749);
  4. **amanupg 2184**;
  5. BASICLAB 2163.
- The top-5 rank intervals overlap.
- **Lesson for the bench:** a local H100 gain converts at ~0.6× officially. Future candidates need more than ~+0.05 locally to show clearly on the board.

## 2026-10-21 Memory geometry-only (snap decisions as v5, path from memory): worse — waypoint memory dropped entirely
- 13604249-51, H100.
  - vs v5 by batch: 1-40 -0.023 [-0.073, +0.003]; 41-80 **-0.096** [-0.172, -0.026]; 81-120 -0.058 (37 of 40 drove, the 7th H100 launch failure).
  - On the 80 complete scenes: **-0.060 [-0.107, -0.022]**.
- With identical snap decisions, the remembered path is a worse geometry than the Hermite bridge, so the memory path itself, not just its triggers, hurts.
- **All four waypoint-memory variants fail to beat v5. Dropped.**
- H100 launch-failure tally: 7 of 31 batches.

## 2026-10-21 Fine-tune Alpamayo (Aman: "try the fine-tune, keep going", /loop)
- **Data access (challenge HF account, token read in place):**
  - `nvidia/PhysicalAI-Autonomous-Vehicles` (PAI-AV) is visible: 70,775 files.
    - Per camera, ~3,146 chunk zips of ~1.9 GB (front-wide alone ~6 TB in total).
    - `labels/egomotion` is ~38 GB; there is also `clip_index.parquet` and calibration.
  - The NuRec scenes (`-NuRec`, 3,219 files) are also visible.
  - Alpamayo's own env ships NVIDIA's loader (`physical-ai-av==0.2.0`).
- **Plan:**
  - Train the action expert (flow-matching head) with the VLM frozen.
  - Data is staged per job on node-local /local, never on the shared fileset.
  - **Exclude every clip that is one of our 120 bench scenes.**
  - Open question: what to train toward.
    - Alpamayo was trained on PAI-AV, so plain fine-tuning reproduces it.
    - Closed-loop slowness and missed lane changes may come from the gap between real and NuRec-rendered images.
    - So check the open-loop speed bias on real vs rendered frames first.
- Reading nureasoning's existing Alpamayo fine-tune pipeline (read-only) for reuse.
- **Fine-tune pipeline (built 2026-10-21):**
  - **Data:** NuRec `nurec_curated_train` (1,761 scenes; no overlap with the 441-scene curated val that our bench draws from).
    - `CAPTURE=1` in `sim_local.sbatch` forces the ego onto the recording for the whole scene (`force_gt_duration_us=60e6`, `skip_driver_during_force_gt=false`) and turns on `send_recording_ground_truth=true`. The recording is never used at inference, and the official evaluator doesn't send it.
    - The driver runs without a model (`A15_NO_MODEL=1`) and saves a sample every 0.5 s (`a15_driver/capture.py`): the 4 cameras × 4 frames at 576×320 as JPEG, the driver's 16-pose history, and the recorded 64 future poses, all in the t0 frame.
    - Scenes download to node-local disk; the samples are tarred to `data/ft_capture/`.
  - **Trainer:** `a15_ft/train.py` with `a15_ft/flow_loss.py` (adapted from NVIDIA alpamayo-recipes `alpamayo1_5_quant`, Apache-2.0).
    - The VLM is frozen; the expert and action_in/out_proj train with flow-matching MSE (AdamW).
    - Inputs are built exactly as the driver builds them.
    - Open-loop check on held-out scenes before and after: ADE, and predicted / recorded 6.4 s path length.
  - **Driver:** `A15_EXPERT_CKPT` loads the fine-tuned expert before int8 quantization.
  - Capture smoke test (2 train scenes): 13608659.

## 2026-10-21 Ported from alpasim-jepa (Aman: "yes port both")
- **Time warp:** the organisers' rule from their GTRS sample (Apache-2.0), via alpasim-jepa's `timewarp.py`. Ported as `route_choice.time_warp_plan` with `A15_TIME_SCALE`.
  - The plan is read ahead in the near term (t ≤ 1 s: 1.1t; 1-3 s: t + 0.1; tapering to 0 by 4 s); times beyond 4 s are unchanged.
  - Applied only if the heading changes < 10° over 4 s (`A15_TS_MAX_TURN_DEG`), and not on snapped plans (`A15_TS_SKIP_SNAPPED`, their judge+warp lesson).
  - On nuPlan: 1.10 straight-only was part of their +0.036 add-on.
- **Stall rescue:** theirs picks from GTRS's vocabulary using GTRS's collision-free / drivable probabilities (a nuPlan-trained model), so it was not ported as-is.
  - Model-sample analogue (`A15_CHOOSER=stall`, `route_choice.stall_rescue_choice`): when sample 0 moves < 3 m in 4 s, draw 5 more.
  - If at least half of all samples move ≥ 3 m, drive the medoid of the moving ones; otherwise stay.
  - Normal replans cost one sample.
- **Queued on H100 vs v5 (v0.3 + snap), 120 scenes each:** warp 13608708-10; stall 13608711-13.
- **Pending on H100:** fast chooser 13608641-43; capture smoke 13608659.
- **GPU choice (Aman, 2026-10-21: use H100 only where genuinely needed):**
  - **Benches stay on H100.** The v5 baseline is H100, results differ by GPU type (the chooser was +0.054 on Blackwell and +0.005 on H100), and the official evaluation runs on H100.
  - **Capture and training go off H100:** `--gres=gpu:1 --constraint="A6000|l40s|l40|R6000"`.
  - `CUDA_COMPAT=auto` uses compat only when the host driver is < 570.
  - Capture smoke resubmitted as 13608985 (13608659 cancelled).
- **Capture smoke passed (13608985, ins057, driver 550, so compat auto-on):**
  - 55 samples from 2 train scenes in 18 min; 34 MB tar.
  - Checked: frames 4 cams × 4 × 576×320 (rendered night scene, correct slots 0/1/2/6); history speed vs the recorded future's first 0.5 s match (11.19 vs 11.24 m/s; 5.72 vs 6.01); valid rotations.
- **Capture at scale:** 400 train scenes (10 × 40, `scenes/pai_train_c0..9.txt` from a seeded shuffle of nurec_curated_train), jobs 13609282-91, 3 lanes, non-H100. Expect ~11k samples.
- **Trainer smoke test:** 13609292 (3 steps on the smoke samples).
- **Screening moved to L40S (Aman, 2026-10-21): "screen them on l40s; if it gains, test it on H100".** The goal is to shorten the dev cycle on GPUs that are easier to land (~34 L40S vs ~8 H100).
  - Cancelled the H100 benches (fast 13608641-43, warp 13608708-10, stall 13608711-13).
  - **L40S screen:** v5 baseline + fast chooser + time warp + stall rescue, 120 scenes each, 12 jobs in 6 lanes (at most 6 L40S at once):
    - v5: 13609305-07;
    - fast: 13609308-10;
    - warp: 13609311-13;
    - stall: 13609314-16.
  - **Rule:** a variant that clearly beats v5 on L40S (local > +0.05) gets an H100 confirmation run before any slot decision.
- **A6000 check (13609319, ins081):** the full v5 stack runs.
  - Node driver 575, so no compat needed (`CUDA_COMPAT=auto` -> 0).
  - 2 scenes, both full drives, 11 snaps; driver peak 14,008 MiB.
  - Mean inference 3.43-3.51 s with 2 sessions (H100 ~1.5 s with 4), so ~2-3x slower per replan.
  - Other A6000 nodes may run driver 550 (compat on A6000 untested); `auto` handles the switch.
  - Use: capture and training (already in the constraint list); screening stays on L40S (closer to H100).
  - Capture jobs 13609282-84 and trainer smoke 13609292 are now running.
- **Trainer smoke passed (13609292, L40S):**
  - 2.281B trainable of 11.08B; 1.44 s per step (2 samples); peak 37.6 GiB; checkpoint saved (409 tensors).
  - Open loop on 10 rendered samples: ADE 2.55, **predicted / recorded 6.4 s path length, median 1.02**, so the model is not slow in open loop on these few samples. The full capture will tell.
- **New lever: missing `<traj_future_start>`.**
  - In closed loop the 4-token generation misses it in **30-42% of plans** (H100 driver logs: 336/800 in v5 batch 41-80, 237/799 in v0.3 batch 81-120). The library then attaches the action block after whatever was generated.
  - `A15_FORCE_TRAJ_START=1` (`policy._forced_rollout`) appends the token instead of generating it, as NVIDIA's teacher-forced training does. It uses the same prefill, position ids and mask, and the model's own sampler (the K-choosers still wrap it).
  - This also aligns inference with our fine-tune's training input.
  - Check on the validation clips: 13609783.
- **Capture c0 (13609282) failed at scene download.** HF xet transfer "I/O error: error decoding response body" in the wizard; it produced an empty tar, which I removed.
  - `sim_local.sbatch` now retries the wizard up to 3 times, keeping already-downloaded files; the last try uses `HF_HUB_DISABLE_XET=1`.
  - c0 requeued as 13610789. Already-queued capture jobs keep the old script, so rerun them if they hit the same error.
- **Throughput on non-H100 GPUs (2026-10-21 15:40):**
  - L40S v5 bench: ~22 of 40 scenes in 2.1 h, so ~3.8 h per batch (H100 ~2.2 h). Fits the 5 h limit.
  - Capture: 10-18 of 40 scenes in ~2.2 h (scene downloads ~68 GB per job first), so 40-scene jobs overrun the 4 h limit.
  - **Fixes:**
    - `sim_local.sbatch` traps TERM/USR1 and saves partial capture and rollouts. New jobs use `--signal=B:USR1@300`.
    - The remaining capture is split into 20-scene jobs: c4a..c9b = 13612127-38, chained after the running ones (cancelled 13609286-91).
    - The running c1/c2/c3/c0r (old script) may lose their tar at timeout; check, and rerun if needed.
- **Forced-start check (13609783, A6000 driver 575, 12 validation clips):**
  - Runs; latency 1.43 s (normal path 1.50 s).
  - Memory 14,786 vs 14,048 MiB (+0.7 GB); on H100 expect ~15.3 GB of the 16 GiB, to verify if it wins.
  - Trajectories differ (max 5.3 m) where the start token had been missing (2 of 12 clips).
  - Screen on L40S, 120 scenes, queued on 3 extra lanes.

## 2026-10-21 evening: action-expert fine-tune, first runs
- **Capture data:** the first four capture jobs saved full tars (data/ft_capture/): c2 1,103 samples, c1 868, c0 922, c0r 1,106, plus the smoke capture's 55. Together that is 4,054 samples from the nurec_curated_train split, symlinked into data/ft_runA/.
  - Before the 4 h limit I snapshotted the partial captures into data/ft_capture_partial/ (2 GB). All four jobs then saved their tars anyway, so the snapshots are superseded.
- **Trainer fixes:**
  - **bf16 AdamW lost updates.** With bf16 weights and lr 1e-5, torch.optim.AdamW rounds most updates away (unit test: 50 steps of -1e-5 left 0.02 at 0.02002). SRAdamW now rounds the updated weight stochastically and tracks fp32 (0.019519 vs 0.019500). tests/test_sr_adamw.py runs before every training job.
  - **Other changes:**
    - 4 (x0, t) draws per VLM prefill;
    - the open-loop check uses the forced-start path that training uses;
    - truncated samples and tars are skipped;
    - an atomic checkpoint every 100 steps;
    - line-buffered logs;
    - training runs on 48 GB cards (A6000/L40S/L40/R6000), measured peak ~44 GB on A6000.
- **Runs:**
  - Run A, 13613201: lr 1e-5, 2 epochs.
  - Run B, 13614344: lr 3e-5, 2 epochs.
- **Closed-loop screens:** L40S, forced start + snap 2.0/15°, A15_EXPERT_CKPT, start afterok:
  - run A: 13614346-48;
  - run B: 13614349-51.
  - Control: the plain forced-start screen, 13612149-51.
- **L40S v5 baseline (13609305-07):** mean 0.689.
  - 1-40: 0.622;
  - 41-80: 0.648;
  - 81-120: 0.796.
- **Fast chooser:** 41-80 scored 0.625 (-0.023 vs v5). It drove further (progress 0.82 vs 0.78) but left the corridor more (0.28 vs 0.23); dist-to-GT 3.33 vs 2.56.
- **Fast chooser rejected on L40S:**
  - 41-80: -0.023;
  - 81-120: 0.726 vs 0.796 (-0.070), more progress (0.87 vs 0.81) but at-fault 0.05 vs 0.03 and dist-to-GT 3.07 vs 2.29.
  - Same pattern as on H100, so I cancelled its 1-40 batch (13609310) to free the GPU.
- **Forced start (L40S, 41-80):** 0.675 vs v5 0.648 (+0.027).
  - left corridor 0.17 vs 0.23; progress 0.73 vs 0.78; dist-to-GT 2.38 vs 2.56.
  - 0 missed start tokens; driver peak 14,914 MiB (< 16 GiB).
- **Open-loop baseline (run B's [before], 150 held-out rendered samples, forced start):** ADE 1.62 m; path-length ratio median 1.005, mean 0.993.
  - The model is NOT slow open-loop on rendered on-route frames.
  - So the closed-loop slowness is a closed-loop effect (drift from the recording), not a speed bias. A fine-tune on force-GT frames can only help through image-domain adaptation (lower ADE).
- **Training speed:** 12.3 s/step (accum 8, 4 noise draws) on A6000, peak 42 GiB; 917 steps ≈ 3.1 h. Run B loss 0.22 at step 1, 0.23 by step 150.
- **Forced start, full L40S screen (13612149-51), paired per scene vs the L40S v5 baseline:** mean 0.707 vs 0.689.
  - diff +0.018, 90% interval [-0.051, +0.089]; wins 25, losses 21, ties 74.
  - By batch: 1-40 0.703 vs 0.622; 41-80 0.675 vs 0.648; 81-120 0.743 vs 0.796.
  - Neutral within noise, below the +0.05 bar for an H100 confirmation. It stays as the fine-tune's base, since training appends the token too, and as the control for the fine-tune screens.
- **Fast chooser, 80 scenes paired:** -0.046 [-0.128, +0.036]. Rejected.
- Merged per-run summaries are in .bench_cache/all_l40sv5, all_l40sforced and all_l40sfast80.
- **Time warp (1.10, straight plans only, not on snapped plans) rejected on PAI (L40S, 13609311-12, 80 scenes paired):**
  - 0.601 vs 0.722, diff -0.121, 90% interval [-0.211, -0.031]; wins 12, losses 17.
  - By batch: 41-80 0.624 vs 0.648; 81-120 0.578 vs 0.796.
  - At-fault collisions 0.07-0.10 vs 0.03; dist-to-GT 3.0-3.5 m vs 2.3-2.6.
  - This is the rear-end / gap-closing failure alpasim-jepa saw on nuPlan at 1.25, here already at 1.10.
  - I cancelled the 1-40 batch (13609313).
- **DAgger capture (CAPTURE=1 CAPTURE_POLICY=1):** the policy drives, and the runtime still sends the recording every step, in the true rig frame at step start (alpasim `events/policy.py:115`).
  - Each sample is labelled with the recorded future from the ego's actual pose: a path back to and along the recording, catching up in time if the ego lags.
  - Samples are skipped beyond A15_CAPTURE_MAX_DEV_M (3 m) so labels stay drivable. Output goes to data/ft_dagger/. A unit test covers the labels from an off-recording pose.
  - **Why:** open-loop the model is not slow (length ratio 1.005), so closed-loop slowness and drift come from states the force-GT data never shows.
  - **Jobs:** v5 + forced-start policy on 120 new train scenes (scenes/pai_train_d0..d5, none used before, none in val), 13615493-98 in 3 afterany lanes.
  - They replace the 6 pending force-GT captures (13612133-38, cancelled before starting). Force-GT data: ~5.7k samples done or in flight.
- **Fine-tune run A (13613201, lr 1e-5, 2 epochs, 917 steps, 3.1 h on A6000):**
  - Held-out open loop (150 samples from 14 unseen train scenes, forced start): ADE 1.624 → 1.285 m (-21%); length ratio median 1.005 → 0.988.
  - Weights moved: median relative change 2.4e-2 over 12 projection matrices.
  - Loss 0.23 → 0.22 (noisy).
  - Before training, unforced generation gave ADE 1.82 vs 1.62 forced, so forced start is also better open loop.
  - Checkpoint data/ft_ckpt/ftA.pt; closed-loop screen 13614346-48 queued. Deleted my smoke checkpoint (smoke.pt, 4.5 GB).
- **Node faults tonight (not our code):**
  - ins058 GPU "uncorrectable ECC error" killed DAgger d1 (13615495); resubmitted as 13615576 excluding ins058.
  - ins085 had /local read-only/missing, which killed d4 (13615496); resubmitted as 13615701 excluding ins058 and ins085.
- **Fine-tune run B (13614344, lr 3e-5, 2 epochs):**
  - Held-out open loop ADE 1.624 → 1.240 m (-24%); length ratio median 1.005 → 0.999.
  - Weights moved 5.0e-2 (vs 2.4e-2 for run A).
  - Better than run A on both measures. Closed-loop screen 13614349-51 queued.
- **Stall rescue, full L40S screen (13609314-16), paired vs v5:** 0.704 vs 0.689.
  - diff +0.015, 90% interval [-0.029, +0.062]; wins 19, losses 13, ties 88.
  - By batch: 1-40 0.669 vs 0.622; 41-80 0.646 vs 0.648; 81-120 0.796 vs 0.796.
  - Neutral: it rarely triggers.
- **Caveat on the v5 baseline:** every variant beat v5 most on 1-40 (forced +0.081, stall +0.047), so the v5 1-40 run may be a low draw.
- **L40S screen summary:**
  - fast -0.046 and warp -0.121, both rejected;
  - forced +0.018 and stall +0.015, both neutral;
  - nothing clears +0.05 for an H100 confirmation.
  - The fine-tune (runs A/B, screened against forced start) is the remaining bet this round.

## 2026-10-22 00:40 fine-tune run C (force-GT + DAgger)
- The Insomnia socket was closed 00:10-00:33; no cluster access in that window. Jobs ran on.
- **Data (data/ft_runC):** symlinks only.
  - 11 force-GT tars (~7.8k samples: c0r, c1-c3, c4a/b, c5a/b, c6a/b, smoke).
  - 4 DAgger tars (1,433 samples): d0 334, d1 394, d2 367, d4 338.
  - d3 and d5 were still running and are left for a later round.
  - DAgger keep rate: ~60-70% of the policy's 0.5 s states lie within 3 m of the recording. The policy scored 0.68-0.86 on those train scenes.
- **Run C (13616891):** lr 3e-5 (B beat A open loop), 2 epochs, -t 10:00:00 (short allows 12 h), checkpoint every 100 steps.
- **Closed-loop screen:** 13616892-94, afterok, same settings as the A/B screens.
- **Run A closed loop (L40S, forced + snap + ftA.pt), 41-120 (80 scenes):** 0.678.
  - vs forced start alone 0.709: -0.031, 90% interval [-0.091, +0.028]; wins 12, losses 17.
  - vs v5 0.722: -0.044.
  - dist-to-GT much lower (1.71/1.74 vs forced 2.38/2.05), but progress lower (0.69/0.74 vs 0.73/0.77) and offroad up on 41-80 (0.15 vs 0.07).
  - The fine-tune makes it track the recording more closely but drive slower, as its open-loop length ratio (0.988) foreshadowed. Not a gain on scene score. The 1-40 batch is pending.
- **Run A closed loop, all 120 scenes:** 0.687.
  - vs forced start 0.707: -0.019, 90% interval [-0.069, +0.030]; wins 16, losses 25.
  - vs v5 0.689: -0.001.
  - Neutral to slightly negative. dist-to-GT is consistently lower (1.59-1.74 vs ~2.2 m).
- **PCS uses only the per-scene scores.** Per alpasim e2e_challenge/local_evaluation/README.md, PCS is a zoib (zero-one-inflated beta) fit over the per-scene canonical scores of all subjects, mapped to an anchor scale. dist-to-GT is a separate column and does not enter it, so the local paired scene-score comparison is the right target. Run A's closer tracking earns nothing by itself.
- **DAgger capture complete:** d3 378 (policy 0.780), d5 301 (0.758). Total 2,112 DAgger samples; d3 and d5 are not in run C.
- **v5 repeat on L40S (A/A), 13618596-98, same settings as 13609305-07:**
  - checks the suspected low 1-40 draw;
  - measures L40S run-to-run noise;
  - doubles the baseline for all paired comparisons.
- **Run B closed loop, 41-120 (80 scenes):** 0.675.
  - vs forced start 0.709: -0.034, 90% interval [-0.098, +0.032]; wins 10, losses 22.
  - Same pattern as run A: progress 0.69 vs 0.73 (41-80), offroad 0.12 vs 0.07, dist-to-GT 1.7 m.
- **Lesson:** fine-tuning on force-GT (on-recording) frames lowers open-loop ADE 21-24% but is neutral-to-negative closed loop.
  - Better imitation of on-recording behaviour does not fix closed-loop driving.
  - Run C, with DAgger recovery data, is the last fine-tune candidate in this line.
- **Run B closed loop, all 120 scenes:** 0.709.
  - vs forced start 0.707: +0.002, 90% interval [-0.053, +0.059]; wins 17, losses 31.
  - vs v5 0.689: +0.020.
  - Neutral, and it loses more scenes than it wins. 1-40 scored 0.777 (high), 41-120 scored 0.675.
- **ins058's faulty GPU struck again:** both v5 repeat jobs 13618597-98 failed with ECC errors. Resubmitted as 13618768-69 with --exclude=ins058,ins085. Run C's screen jobs (13617289-91) were updated to exclude the same nodes.
- **L40S A/A, v5 repeat on 1-40 (13618596):** 0.645 vs 0.622 first time.
  - +0.022, 90% interval [-0.047, +0.091]; 12 of 40 scenes changed outcome (8 up, 4 down).
  - L40S reruns are NOT deterministic (Blackwell agreed within 0.006 per batch). Expect about ±0.02 per 40-scene batch from run-to-run noise alone.
  - The first v5 1-40 run was a lowish draw, which inflated the 1-40 gains of forced start (+0.081) and stall rescue (+0.047). Both were already neutral on 120 scenes.
  - All comparisons will be recomputed against the 2-run v5 average once 41-80 and 81-120 land.
- **L40S A/A, v5 repeat on 41-80 (13618768):** 0.699 vs 0.648 first time.
  - +0.051, 90% interval [-0.025, +0.127]; 8 scenes changed outcome.
  - With 1-40 (+0.022), single 40-scene L40S runs move 0.02-0.05 from rerun noise alone, and the first v5 baseline (0.689) ran low.
  - Against the v5 average, forced start and stall rescue are about 0, and fine-tunes A and B are slightly negative.
  - For the October slot, a candidate needs a gain that survives replicate runs, not one 120-scene run.
- **L40S A/A, full 120 scenes (v5 run 2, 13618596/768/769):** 0.698 vs 0.689; diff +0.009 [-0.029, +0.048]; wins 16, losses 13.
  - By batch: 1-40 0.645 vs 0.622; 41-80 0.699 vs 0.648; 81-120 0.751 vs 0.796.
  - Batches move ±0.05, but 120-scene means agree within ~0.01.
- **Vs the 2-run v5 average (0.693; .bench_cache/all_l40sv5x2):**
  - forced start +0.013 [-0.054, +0.082];
  - stall rescue +0.010 [-0.027, +0.049];
  - fine-tune A -0.006 [-0.075, +0.063];
  - fine-tune B +0.016 [-0.047, +0.080].
  - Nothing beats v5. Run C (DAgger) is pending.
- **Fine-tune run C (13617288, force-GT + DAgger with smooth recovery labels, lr 3e-5, 2 epochs, 1,997 steps, 5.5 h on A6000):**
  - Held-out open loop (150 samples, val includes DAgger sessions): ADE 2.324 → 1.933 m (-17%); length ratio median 0.995 → 0.993.
  - Weights moved 7.6e-2.
  - Loss 0.42 → ~0.25, with isolated spikes at steps 1275/1350 (single outliers; neighbours normal).
  - Closed-loop screen 13617289-91 started 06:50.
- **DAgger round 2 (speculative, 13621306-08):** 60 new train scenes (pai_train_d6-d8; none used before, none in val), same v5 + forced policy.
  - Runs on A6000/R6000, so the L40S screen of run C is not slowed; -t 6 h.
  - Feeds a run D if run C's closed loop shows promise.

## 2026-10-22 10:30 fine-tune run C: best L40S result so far
- **Run C closed loop (L40S, forced + snap + ftC.pt, 13617289-91), 120 scenes:** 0.733.
  - By batch: 1-40 0.770; 41-80 0.703; 81-120 0.726. Offroad 0.05 in all three (runs A/B had 0.12-0.15 on 41-80).
  - vs 2-run v5 average 0.693: **+0.040 [-0.025, +0.107]**; wins 29, losses 26.
  - vs forced start 0.707: +0.026 [-0.025, +0.081].
  - Best L40S result of the campaign. The interval still includes 0, and A/A noise on 120 scenes is ~0.01.
  - DAgger recovery data appears to fix the offroad drift that sank the force-GT-only fine-tunes. Progress is still lower (0.70-0.79).
- **Confirmation launched:**
  - H100 run C 13623483-85;
  - H100 v5 rerun 13623486-88 (pairs with the old H100 v5 = all_h100v03snap 0.728);
  - L40S run C repeat 13623489-91.
- **DAgger iteration with run C's own policy (A15_EXPERT_CKPT=ftC.pt):** 60 new train scenes (pai_train_e0-e2; none used, none in val), 13623492-94 on A6000/R6000. Feeds run D.
- **H100 is scarce:** short/burst show 3 H100 nodes (2 GPUs each); ins001 and ins011 are unavailable.
  - Slurm estimates run C's H100 batches start ~20:50 Oct 22 and the H100 v5 rerun ~Oct 24.
  - Both stay queued. If the v5 rerun lags, run C is compared with the existing H100 v5 (all_h100v03snap 0.728).
- **DAgger round 2:** d7 384 samples; d6 done; d8 running.
- **DAgger round 2 complete:** d6 355, d7 384, d8 318. All DAgger d0-d8 = 3,169 samples, 180 train scenes.
- **Run D (13624503):** run C's recipe (base init, lr 3e-5, 2 epochs) on force-GT (~7.8k) + all DAgger d0-d8 (3,169; run C had 1,433).
  - -t 11 h. Isolates the effect of more DAgger data.
  - L40S screen 13624505-07, afterok, excluding ins058/ins085.

## 2026-10-23 00:40 run C replicated on L40S; run D no gain
- The socket was closed 12:16 Oct 22 to 00:35 Oct 23 (Aman first reopened it on a different Mac; Insomnia port 22 also timed out from there). Jobs ran on.
- **Run C repeat (13623489-91):** 0.740.
  - By batch: 1-40 0.791; 41-80 0.701; 81-120 0.729.
  - vs the first run C (0.733): +0.007 [-0.025, +0.039]; A/A agreement is good.
- **Run C, 2 runs (all_l40sftCx2: 0.737) vs v5, 2 runs (0.693): +0.043 [-0.019, +0.106]**; wins 29, losses 28.
  - vs forced start: +0.030 [-0.016, +0.079].
  - The mean gain replicates across runs (+0.040, +0.047), but wins ≈ losses: a few scenes gain a lot, and the scene-level interval still spans 0.
- **Run D (more DAgger from the base policy, d0-d8; 13624503, screen 13624505-07):** 0.707.
  - By batch: 1-40 0.754; 41-80 0.686; 81-120 0.680.
  - vs v5 x2: +0.013; vs forced: -0.000. Offroad is back up to 0.10-0.12.
  - Open loop barely moved: ADE 2.217 → 2.192 on its own val set. Weights moved 8.7e-2.
  - **More base-policy DAgger data did not help.** Run C's gain does not scale with DAgger volume, so seed luck or overfitting are possible.
- **DAgger with run C's own policy (13623492-94):** e0 344, e1 362, e2 388 (1,094 samples).
- **H100:**
  - run C 13623483-85, estimated start 00:41 Oct 23;
  - v5 rerun 13623486-88, estimated start Oct 25 14:25 (still before the Oct 27-28 slot decision);
  - meanwhile compare with the existing H100 v5 (all_h100v03snap, 0.728).
- **Prediction track record on PAI (asked by Aman, 2026-10-23):** no PAI entry has gone the opposite way to the local prediction.
  - v0.3: predicted ~1900-1990, got 2143-2150 (better).
  - v5: predicted +0.071 local → ~0.68; got 0.652 (+0.041, ~58% of local).
  - **nuPlan v6 fell for a different reason:** a NAVSIM-trained critic validated on public navtest, a benchmark known to flatter that model family.
  - **Risks for run C:** (1) L40S-only so far (GPU-type effects seen before); (2) winner's curse over ~10 variants on the same 120 scenes; (3) wins ≈ losses (few big scene gains); (4) run D's null result suggests training-run luck.
- **Robustness check (risk 4): run C retrained with a different seed** (--seed 1, same data and recipe), 13636811; L40S screen 13636812-14, afterok.
  - If seed 1 lands near C (~0.735), the gain is real; if near D/forced (~0.707), C was luck and the slot should not go to it.

## 2026-10-23 01:30 Full PAI push (Aman: "full push now"; nuPlan dead; Alpamayo 2-Super download approved)
- **Research sweep (4 parallel agents).** Findings:
  - post-fix board: ACM 2316, VAIL 2258, UJS IV Squad 2251, us 2219, host 2220;
  - PAI finals run with the Harmonizer on; there is an innovation prize;
  - RoaD (arXiv 2512.01993: closed-loop SFT, AlpaSim 0.444 → 0.630);
  - AlpaGym;
  - FlashDrive (W4A8 A1.5);
  - Alpamayo 2-Super (34B, PAI-trained, AlpaSim 1.50 vs 1.37);
  - exact scoring code.
- **Our local bench has always run with the Harmonizer ON.** The developer base config passes `--enable-harmonizer`; only `e2e_challenge/ec2.yaml` (official initial runs) removes it. So local results match the finals' rendering; the current board is unharmonized. This may explain part of the ~0.6x local → official ratio.
- **Bets:**
  1. **RoaD-style closed-loop SFT (main).** New `A15_CHOOSER=record` (K samples, drive sample 0, keep all K) and `A15_CAPTURE_AT_INFER=1` (capture at each inference with the K samples).
     - Trainer `road_target`: the sample nearest the recording (mean 4-corner distance over 2 s); if > 3 m, the smooth recovery label.
     - Test: tests/test_road_label.py.
     - Smoke test 13637564; captures 13637565-76 (120 train scenes, pai_train_r0-r11; K=16; base v5+forced policy; A6000/R6000/L40), afterok on the smoke test.
  2. **Weight averaging:** ftC_w50 (base 0.5 + C 0.5) and ftC_w75; job 13637518; L40S screens 13637519-24.
  3. **Controller gains on run C** (lat/heading were already +0.002 in Sept):
     - idx_start_penalty=5: 13637532-34;
     - long_position_weight=4: 13637535-37.
  4. **Alpamayo 2-Super teacher:** gating/size check 13637517 (cluster/fetch_a2.sbatch), then download + NVlabs/alpamayo2 clone.
  5. **FlashDrive:** delegated to the alpasim-jepa session (Aman confirmed); it works in alpasim-jepa/pai_flashdrive/.
  - **Also running:** run C seed-1 retrain 13636811 (screen 13636812-14); H100 run C 13623483-85 and v5 rerun 13623486-88 (queued).
- **6-camera capture for the teacher.**
  - Alpamayo 2-Super's driving profile uses cameras (0,1,2,3,5,6), i.e. with the rears, 4 frames each, 16-step history and 64-step future at 0.1 s. Its pixel budget is 163,840-196,608, so 576x320 fits.
  - New `A15_POLICY_SLOTS`: the session takes all 6 PAI cameras, the policy still sees 0,1,2,6, and capture saves all 6. The trainer subsets to (0,1,2,6).
  - The RoaD capture was resubmitted with 6 cameras: smoke test 13637608, captures 13637609-20. The same captures serve RoaD labels and A2 labels.
- **Alpamayo 2-Super:** not gated, 71.6 GB, 15 shards; downloading in 13637597. NVlabs/alpamayo2 cloned to alpasim/external/alpamayo2 (commit 5977692).
  - Teacher pipeline: src/a15_ft/a2_label.py + cluster/a2_label.sbatch on R6000 (96 GB), with the official prebuilt flash-attn 2.8.3 wheel.
  - Labels go to data/ft_a2labels/<tar>/<session>/<t0>.npz.

## 2026-10-23 10:30 Results from the night (status checks paused ~02:30-10:20)
- **Weight average ftC_w50 = 0.5 base + 0.5 ftC (L40S, 13637519-21):** **0.742**, our best.
  - By batch: 1-40 0.775; 41-80 0.712; 81-120 0.739.
  - vs v5 x2 (0.693): **+0.049 [-0.010, +0.110], wins 28 / losses 22** (more balanced than C's 29/28).
  - vs C x2: +0.006 [-0.032, +0.041], wins 26 / losses 16.
  - Progress is back up (0.73-0.81): averaging undoes the fine-tune's slowdown.
  - **H100 confirmation: 13640435-37.** It replaces the H100 run-C jobs (cancelled).
- **ftC_w75 (41-80, 13637522):** 0.700, no better than C. Remaining batches cancelled.
- **Controller idx_start_penalty=5 on C (41-80, 13637532):** 0.664, worse (offroad 0.12). Remaining batches cancelled. long_position_weight=4 still queued.
- **Run C seed 1 (13636811):** trained (ftC2.pt). Screen 13636812-14 waiting on priority.
- **RoaD capture (K=16, 6 cameras, base v5+forced):** about 1,066 samples from 110 sessions, plus the smoke test's 25.
  - The road_r2 rollouts produced no results (0 samples).
  - Policy scene scores on the train scenes: 0.54-0.87.
- **Run R (RoaD):** force-GT ~7.8k + RoaD tars x3 (oversampled), base init, lr 3e-5, 2,000 steps (13640427); L40S screen 13640428-30.
- **Storage:**
  - The shared dept fileset fell to 95-156 GB free (other users' writes; alpasim is 421 GB).
  - Aman approved deleting ftA/ftB/ftD.pt and data/ft_capture_partial (15.7 GB); done. He did not approve data/nuplan_track (43 GB).
- **sim_local.sbatch:** A15_ENV_SCRIPT / A15_DRIVER_SRC hooks for alternative driver builds (alpasim-jepa's FlashDrive); A15_MODEL_DIR can now be overridden.
- **FlashDrive handed back to this thread** (alpasim-jepa returns to nuPlan, extended to Nov 16).
  - **Measured (job 13639534, A6000, 40 samples):** 884 ms/plan (v5 int8 on A6000 is ~3.4 s), nvidia-smi 11.6 GB, ADE 1.09 m; load 17.5 min.
  - **The faster PARO-only load (13639629) fails its strict check:**
    - missing `*.rotate_linear.qlinear.bias`;
    - unexpected non-persistent buffers (action_space stats, sinus freqs).
  - z-lab's expert weights differ from NVIDIA's release.
  - **Parked.** Its value is speed, and our levers that would spend speed (2 Hz -0.045; K-sample choosers +0.005 on H100) showed nothing. The remaining check 13639630 was cancelled. Code stays in alpasim-jepa/pai_flashdrive.

## 2026-10-23 11:00 October's last slot is PAI's (Aman: "you [have] october's last slot, make it count")
- **The catch:** finals go to the board's top 3, but the board's initial runs have the Harmonizer OFF (ec2.yaml). Every local run so far had it ON (dev preset). The entry must climb the Harmonizer-off board first.
- **New `OFFICIAL_MIMIC=1`** in sim_local.sbatch: `services.renderer.command.7=--no-enable-harmonizer` plus `skip_driver_during_force_gt=true` (as the official initial runs). 2-scene test: 13640533.
- **Selection plan:**
  1. L40S 120-scene screens (Harmonizer on, the finals' condition) for each candidate:
     - w50 (0.742, done);
     - C2 seed check;
     - R (RoaD);
     - R averaged with base;
     - soup base + C + C2;
     - A2-teacher run (if the labels land).
  2. The best 1-2 go to H100, 120 scenes, both Harmonizer ON (finals) and OFF via OFFICIAL_MIMIC (board), paired against v5 on the same H100 setting.
  3. Pick the candidate best on H100 Harmonizer OFF (board ranking gates the finals), provided it is also not worse with Harmonizer ON.
  4. Build v7 (container/Dockerfile.v7 = v5 + forced start + expert checkpoint), use the last PAI warmup to check memory and time, then Aman submits about Oct 28-29. Final container chosen Oct 31.
- **Trial build of v7 with ftC_w50:** copying the checkpoint to container/assets/expert.pt (git-ignored).
- **v7 trial build OK** (`local/pai:v7-trial`, git 33b9b25, expert = ftC_w50 sha256 f9da7cf3…):
  - the driver imports and the expert has 409 tensors;
  - env A15_FORCE_TRAJ_START=1, A15_EXPERT_CKPT=/app/expert/expert.pt, K=1, 1 Hz, snap 2.0/15;
  - driver files match the repo byte for byte.
  - container/Dockerfile.v7.dockerignore sends only src/a15_driver and the expert, so the build takes 93 s.
  - **Not pushed.** The final expert is chosen after the H100 checks.
- **OFFICIAL_MIMIC verified** (13640533, 2 scenes): renderer config shows `--no-enable-harmonizer` and `skip_driver_during_force_gt: true`; 2/2 sessions clean; driver peak 14,196 MiB.
- **Queued:**
  - L40S Harmonizer-off screens, v5 vs ftC_w50 on 120 scenes each (board condition);
  - soup ftCC2_soup (0.5 base + 0.25 C + 0.25 C2), avg 13641310, screen 13641312-14;
  - ftR_w50 (0.5 base + 0.5 R), avg 13641311 afterok on run R, screen 13641315-17.
- **Failure breakdown, L40S 120 scenes** (scene counts averaged over runs; score lost in brackets):

  | | at-fault | offroad | corridor | partial progress | full pass |
  |---|---|---|---|---|---|
  | v5 x2 | 6.0 | 4.5 | 17.5 | 20.5 (-8.8) | 71.5 |
  | forced | 4.0 | 8.0 | 16.0 | 19.0 (-7.2) | 73.0 |
  | C x2 | 2.5 | 6.0 | 12.5 | 27.5 (-10.6) | 71.5 |
  | **w50** | **3.0** | 6.0 | **14.0** | 22.0 (-7.9) | **75.0** |

  - w50 halves at-fault collisions and cuts corridor exits by 3.5 while keeping progress. Zero scenes become passes, which the IRT/PCS model rewards.
  - **Corridor exits are still the largest loss: 14 scenes, about 0.12 of the mean.**
- **Corridor exits are mostly wrong branches at junctions** (per-timestep metrics, 1-40 batch):
  - `0d10d2e3`: w50 drives faster than v5 and about 8 s in leaves the recording entirely (lateral 3 → 84 m: another road). v5, slower, follows the recorded branch.
  - `0b54c8d4`: both v5 and w50 miss a turn at about 17-19 s.
  - **Cause:** the model never sees the route intent (no nav text; route snap is off on turns), so at forks it guesses.
  - **Options:**
    1. nav text in the prompt (Sept test +0.002, untested with the fine-tune);
    2. fine-tune WITH nav text conditioning. RoaD labels pick the recorded branch, but without route intent as input the model cannot learn which branch. Needs a re-capture that stores the route (about 2 days).
- **H100 A/A (v5 rerun, 1-40, 13623486):** 0.695 vs the original H100 v5 run 0.692; +0.003 [-0.000, +0.008]; 9 of 40 scenes changed.
  - **H100 runs are nearly deterministic**, unlike L40S (±0.05 per batch). One H100 run per candidate, against the existing H100 v5 (all_h100v03snap, 0.728), is a reliable test.
  - The remaining v5 rerun batches (13623487-88) were cancelled to free H100s for the w50 confirmation.

## 2026-10-23 16:45 (socket closed 14:39-16:33)
- **Run C seed 1 (C2, L40S, 13636812-14):** 0.686.
  - By batch: 1-40 0.729; 41-80 0.655; 81-120 0.674.
  - vs v5 x2: -0.008 [-0.076, +0.060]; wins 27, losses 38.
  - **The recipe is not reliably good: C was a good draw, C2 is neutral.** w50 is built from C's checkpoint, so its 0.742 stands as a model, but selecting C carries a mild winner's curse on these 120 scenes.
- **Harmonizer OFF (OFFICIAL_MIMIC, L40S), 41-80:** v5 0.734 vs w50 0.780; +0.046 [-0.031, +0.126]; wins 6, losses 9.
  - w50 has 0 at-fault collisions and corridor 0.05 vs 0.10.
  - Both score higher without the Harmonizer (v5 41-80 with Harmonizer: 0.648 / 0.699).
  - v5 81-120 with Harmonizer off: 0.774. The rest are running.
- **Pending:** soup (base + C + C2), its screen running; run R at step 1550/2000; w50 on H100 (queued); w50 + nav text (queued); the A2 teacher labeller started on an R6000 (13637632).
- **A2 Super teacher labels work** (test 13644545: 30 samples, 4.5 s each, 66.7 GiB reserved, 8.5 min load on an R6000). NVIDIA's own smoke test failed only because our HF account lacks PhysicalAI-AV dataset access; our labeller reads our captures.
  - **Sanity check on road_r0 (49 labels):**
    - A2 path lengths match the recording and our policy's sample 0 (e.g. 183 / 183 / 180 m at 28 m/s);
    - the first step matches the speed; turns are followed;
    - large distances from the recording occur where our policy disagrees with it too (the recording stops at ~74-79 m, both models go to ~85 m).
    - **The labels are valid, but A2's ADE to the recording is similar to our sample 0's.** Expect a modest effect, mainly in hard cases.
  - Full labelling: 13644595-96 (2 shards).
  - Then run T: force-GT + RoaD states with A2 targets (--teacher_dir), base init, then blended with base.
- **Harmonizer OFF (board condition, OFFICIAL_MIMIC, L40S, 120 scenes):** v5 0.728 vs w50 0.757.
  - +0.029 [-0.028, +0.086]; wins 24, losses 25.
  - By batch: 1-40 0.675 → 0.735; 41-80 0.734 → 0.780; 81-120 0.774 → 0.756.
  - The gain is smaller than with the Harmonizer on (+0.049) and not significant.
  - At ~0.6x local → official that is ~+0.017 scene, ~+14 PCS: short of the +32 needed to pass UJS (2251).
  - Everyone scores higher without the Harmonizer (v5 0.728 vs 0.693), so the finals will score lower for all teams.
- **Soup (base + C + C2), L40S:** 41-80 0.681, 1-40 0.742. Below w50 on both batches; C2's neutral weights dilute C.
- **Soup (0.5 base + 0.25 C + 0.25 C2), L40S 120 scenes:** 0.727.
  - By batch: 1-40 0.742; 41-80 0.681; 81-120 0.757.
  - vs v5 x2 +0.034; below w50 (0.742). Dropped.
- **Nav-text batch 41-80 (13642667) failed:** runtime DEADLINE_EXCEEDED on ins060, 0/40 scenes drove (infrastructure). Resubmitted as 13644725.
- **A2 labelling shard 0 done** (528 labels, 3.8 s each); shard 1 running.
- **Run T (A2 Super teacher) queued:** ft_runR data (force-GT + RoaD x3), with RoaD states targeting A2 labels via `--teacher_dir data/ft_a2labels_x3`.
  - That dir holds symlinks road_*_{a,b,c} → ft_a2labels/road_*, matching the x3 tar names.
  - Base init, lr 3e-5, 2,000 steps. Job 13644832 (afterok on labelling 13644596).
  - Then ftT_w50 (13644833) and its L40S screen (13644834-36).
- **Run R (RoaD, 13640427) trained:** held-out ADE 1.748 → 1.496 m (-14%); length ratio 1.007 → 1.007 (no slowdown, unlike the force-GT fine-tunes); weights moved 7.7e-2.
  - Screens: R alone 13640428-30 (afterok, now eligible); R_w50 averaging 13641311 → screen 13641315-17.
- **w50 + nav text (A15_NAV=1, A15_NAV_DIST=intersect), L40S, Harmonizer on:**
  - 81-120: 0.805 vs w50 0.739 (corridor 0.05 vs 0.10; at-fault 0.00);
  - 1-40: **0.834** vs w50 0.775 (v5 0.622/0.645);
  - 41-80 still running (rerun 13644725).
  - **+0.06 on both finished batches, via fewer corridor exits (junction branch errors).** Now the lead candidate.
  - **Queued:** H100 Harmonizer-off (board condition) 13646122-24 and L40S Harmonizer-off 13646125-27.
  - The pending w50 H100 batches (13640436-37) were cancelled to free H100s; the running w50 H100 batch (13640435) stays as a reference.
- **H100 w50 1-40 (13640435) VOID:** "CUDA error: unspecified launch failure" in the driver (4,572 failed inferences); only 15/40 scenes drove (the summary shows n=65, mean 0.305).
  - This is the 8th Insomnia H100 batch lost to this cluster-specific fault (driver 550 + cuda-compat); the official evaluator ran v5 987/987.
  - Not rerun: w50+nav is now the lead.
  - If the H100 board-condition screens of w50+nav (13646122-24) hit the same fault, rent an H100 for the final check (Aman approved renting if needed).

## 2026-10-23 22:00 **w50 + nav text: first significant gain (L40S, Harmonizer on, 120 scenes)**
- **w50 + A15_NAV=1 (intersect distance), 13642668/13642669/13644725:** **0.778**.
  - By batch: 1-40 0.834; 41-80 0.694; 81-120 0.805.
  - **vs v5 x2 (0.693): +0.084 [+0.027, +0.142]; wins 31, losses 19.** The first interval excluding 0.
  - vs w50: +0.035 [+0.000, +0.075]; wins 17, losses 14.
- **Failures** (scenes of 120): at-fault 6.0 → 3.0 (w50) → **2.0**; corridor 17.5 → 14.0 → **9.0**; full pass 71.5 → 75.0 → **81.0**.
  - Corridor exits nearly halve: these are the junction branch errors; nav text supplies the route intent.
- **Official estimate at 0.6x:** about +0.05 scene, ~+40 PCS, about 2260 (VAIL 2258, UJS 2251).
- **Attribution test:** v5 + forced + nav, no fine-tune. In Sept, nav on v0.3 + chooser gave +0.002. If the gain is mostly nav, that is a cleaner candidate with no seed luck.
- Board-condition screens of w50+nav queued: H100 13646122-24, L40S 13646125-27.
- **2026-10-23 23:20: socket closed again.**
- **Storage and nuPlan notes from alpasim-jepa:**
  - The fileset was at 61 GB free; alpasim-jepa deleted the FlashDrive weights (32 GB, Aman approved). Code and kernel stay in alpasim-jepa/pai_flashdrive.
  - nuPlan: py123d v0.2.0 joint64 seed 0 scores 0.908 local; it goes in on Nov 1.
- **Aman: keep the A2 Super weights (72 GB) for now**, in case of another labelling round.

## 2026-10-24 10:30 Overnight results (all jobs COMPLETED; batch means, paired compares pending fetch)
- **Board condition (Harmonizer off, skip_driver_during_force_gt), H100, w50+nav (13646122-24):** 1-40 0.787; 41-80 0.756; 81-120 0.802 → **~0.782**. No launch failures.
- **Same, L40S:** w50+nav 0.797 / 0.780 / 0.775 (**~0.784**) vs mimic v5 ~0.728 and mimic w50 ~0.757. The nav gain holds with the Harmonizer off (+0.056 over v5), and L40S agrees with H100 within 0.002.
- **Harmonizer on, L40S:**
  - R (RoaD) alone: 0.725 / 0.636 / 0.697, ~0.686 (below v5 x2 0.693);
  - R_w50: 0.750 / 0.685 / 0.756, ~0.730;
  - T_w50 (A2 Super teacher): 0.769 / 0.705 / 0.720, ~0.731. Run T open loop: ADE 1.639 → 1.535; 3258/10619 teacher labels; weights moved 6.78e-2.
  - All three are below w50 (0.742) and w50+nav (0.778). RoaD and the A2 teacher are not adopted for the slot.
- **forced+nav without fine-tune:** 0.718 / 0.704 / 0.699, ~0.707 (v5 x2 0.693). Nav alone ≈ +0.014, so the gain needs the fine-tuned blended expert plus nav together.
- **Slot candidate: v7 = ftC_w50 expert + A15_NAV=1 intersect + forced traj start** (Dockerfile.v7 already set).
- **Queued 10:25 (excluding ins058/ins085):**
  - H100 board-condition v5 baseline 13648567-69 (pairs with the H100 w50+nav ~0.782);
  - H100 Harmonizer-on (finals condition) w50+nav 13648570-72.
- 10:30: the socket closed again; summaries for mimic w50nav/R/R_w50/T_w50/fnav are not yet in .bench_cache.
- **v7 image bug caught before any push (10:45):** /app/expert/expert.pt was root:root 0640 (from the host file), so the driver, which runs as `alpasim`, would raise PermissionError at model load. That would make it a zero-score submission. The trial build's checks ran as root, so they missed it.
  - Fix in Dockerfile.v7: `mkdir -m 0755 /app/expert`, `COPY --chmod=0644`, and the build checks now run as `alpasim`.
  - A bare `COPY --chmod=0644` into a missing dir also gives the dir 0644, so alpasim could not enter it; hence the mkdir.
  - Rebuilt `teams/amanupg:v7` (local only, not pushed), git 4d809ec. As `alpasim`: driver imports ok, expert 409 tensors, sha256 f9da7cf3… (= ftC_w50); env A15_NAV=1, intersect, FORCE_TRAJ_START=1, NUM_SAMPLES=1.
  - Rule: image checks run as the runtime user.

## 2026-10-24 11:30 Aman: "deterministic gains only". New protocol; fresh held-out confirmation of v7
- **Rule** (memory pai-deterministic-gains-only): a change enters the image only if it shows a paired gain with an interval above 0 on scenes never used for selection, under board conditions (OFFICIAL_MIMIC) on H100, and its scene flips have a mechanism. Expected official gain is quoted at ~0.6x local.
- **Scene splits** (seed 20261024, `shuf --random-source`): the 321 curated-val scenes outside our 120 bench split into:
  - `scenes/pai_fresh_{1_40,41_80,81_120}.txt` (120), used once to confirm v7;
  - `scenes/pai_lockbox_201.txt` (201, read-only), only for the final container choice.
  - The old 120 bench is now a screening set (contaminated by ~40 selections).
- **Fresh confirmation queued, v5 vs v7, OFFICIAL_MIMIC:**
  - H100 13648575/76, 79/80, 83/84;
  - L40S backup 13648577/78, 81/82, 85/86.
  - The old-bench H100 jobs 13648567-72 were cancelled before they started.
- **Paired, old bench, board condition (vs mimic v5 0.728):**
  - w50: +0.029 [-0.028, +0.086], W/L 24/25;
  - **w50+nav L40S: +0.056 [+0.001, +0.111]**, 28/23;
  - w50+nav H100: +0.054 [-0.006, +0.111], 30/22.
  - H100 vs L40S w50+nav A/A: -0.002 [-0.020, +0.016].
  - With the Harmonizer on (vs v5 x2 0.693): R +(-0.007), R_w50 +0.037, T_w50 +0.038, forced+nav +0.013; all intervals include 0.
- **Where v7 still loses (H100 board, loss per 120 scenes):**
  - rear-ended partial-progress 0.069 (19 scenes);
  - corridor 0.067 (8);
  - offroad 0.033 (4);
  - at-fault 0.025 (3);
  - slow partial 0.024 (13).
  - 15 scenes fail in both v5 and v7; 10 flip each way (noise or mechanism, to check).
- **Rear-end mechanism (per-step metrics):** after force-GT hands over (~3 s), the recorded car pulls away (gap 10-20 m) while ours holds at a standstill or brakes to a full stop, and the replayed follower hits us.
  - 2f227f2c stays at 1.9 m; 268949d2 at 2.4 m; 1d6e30bc and 2c84ac16 start stopped and creep <2 m in 15-20 s; 213dfdac stops dead at 59 m.
  - All start near a junction with "Turn left/right in 15-36 m".
- **Diagnosis queued** on these 10 scenes (`scenes/pai_diag_stall10.txt`), with the new A15_LOG_PLANS=1 (plan distance, speed, nav, CoT):
  - v7 with video (13648623);
  - v7 + A15_REASONING=1 (13648624): the CoT says why it holds.
- **Early failures are structural, not caused by the board setting.** 0e05cb0c, 15fd88d6, 2466266e, 0f4b8753 and 32be03e4 fail in every config and condition (mimic or Harmonizer on, v5/v2/v7).
  - Each fails ~1 s after handover (handover ~2 s; fail at 3.0-3.8 s).
  - Mostly highway at 24-36 m/s, with the route bending within 42 m (0e05cb0c: first waypoint [42.1, -3.7]).
  - 0e05cb0c: lateral error 0.04 → 4.1 m in 1.1 s (peak lateral speed ~5.5 m/s), then corrects back, too late.
  - The frames, history (32 m/s) and sim timing (synchronous) look right.
  - Open question: plan or controller? A15_LOG_PLANS now also logs plan xy at 1/2/4/6.4 s and the route's first point.
  - These 5 scenes were added to the diagnostic list, now 15 scenes (both diag jobs still pending).
- **Correction (alpasim-jepa, verified in external/alpasim/src/wizard/configs/controller/default.yaml):** the default gains are:
  - long_position 2.0, lat_position 1.0, heading 1.0, acceleration 0.1;
  - rel_front_steering 5.0, rel_acceleration 1.0, **idx_start_penalty 10**.
  - The linear MPC uses horizon 20 × 0.1 s.
  - So the Oct 22 idx_start_penalty=5 run moved it **down**, which hurt (-0.040 [-0.090, -0.001], offroad). Upward (15) is untested.
  - long_position_weight 4 never ran (cancelled). Shared with alpasim-jepa for its nuPlan gain sweep.
- **New lever: classifier-free guidance on the nav text (A15_NAV_CFG=w).**
  - Alpamayo 1.5 ships `sample_trajectories_from_data_with_vlm_rollout_cfg_nav`: v = (1-w)·v(no nav) + w·v(nav), from Guided Flows eq. 6. The unguided prompt is the same one with `<|route_start|>…<|route_end|>` removed, prefilled separately.
  - We ported it into our forced-start path (policy._forced_rollout; refactored into prefill/make_step). With w=1 or no nav text it is off.
  - The model config sets no default weight (so 1.0). NVIDIA's nav examples use "Turn left onto <street> in 40m".
  - Cost: one more VLM prefill and 2× expert passes, on plans with nav text only. Memory to be read from gpu_mem.log.
  - Target: corridor exits (junction branch errors, 0.067 of the score lost on 120 scenes).
- **Candidate driver code now lives in `src_cand/`** (sim_local.sbatch: A15_DRIVER_SRC without A15_ENV_SCRIPT uses the standard env).
  - `$A/src/a15_driver/policy.py` stays byte-identical to the v7 image (md5 f45182e3…), so the fresh v7 confirmation jobs run the exact image code.
- **Screen:** CFG w=2 on the old bench, OFFICIAL_MIMIC, L40S, 13648847-49, against mimw50nav L40S (0.784). If it passes, confirm on the fresh 120 on H100.
- **12:20: both fresh H100 batches VOID** with "CUDA error: unspecified launch failure" (the 9th and 10th Insomnia H100 losses):
  - v5 1-40 (13648575, ins050): 4 sessions completed, then 6,505 failure lines; summary n=76, mean 0.145;
  - v7 1-40 (13648576, ins048): 3,179 failures; cancelled.
  - The fault starts 5-15 min after "policy ready". Insomnia has only 3 H100 nodes (ins048-050) and every one has faulted (ins049 and ins050 since Oct 13). This is the cluster's driver 550 + cuda-compat setup; the official H100 run of v5 completed 987/987.
  - **Fresh confirmation moved to L40S, two runs per config:**
    - L40S agrees with H100 on w50+nav: A/A -0.002 [-0.020, +0.016], 85 ties of 120.
    - Run 1: 13648577/78, 81/82, 85/86. Run 2: 13648949-54.
    - Cancelled H100 jobs: 13648576, 79, 80, 83, 84.
  - If an H100 number is still needed for the final decision, rent one (Aman approved renting in that case).
- **Fresh 1-40, L40S run 1 (clean: 40/40 sessions, 0 errors; v7 loaded the expert):** v5 0.787, v7 **0.860**.
  - Diff +0.072 [-0.012, +0.166], W/L 6/5, 29 ties.
  - The 82 vs 42 min wall difference is the node (ins057 vs ins061).
  - 41-80 and 81-120 run 1, and all of run 2, are pending.
- **13:20:** only 2 L40S jobs run at a time, so the 13 queued jobs would take ~6-7 h in series.
  - The stall/handover diagnostics moved to R6000 (Blackwell, driver 575, so no cuda-compat): v7+video 13649393, v7+CoT 13649394; old 13648623/24 cancelled.
  - The diagnosis does not need L40S parity.
  - The CFG screen and the fresh confirmation stay on L40S.

## 2026-10-24 13:30 Dept fileset full (5 TB quota, 640 MB free); my fault, fixed
- **Cause:** the fresh-scene runs fetched 88 NuRec usdz (149.9 GB) into the shared `data/fs` today. That set now holds 207 usdz, 370 GB.
  - alpasim-jepa reported it: two of its nuPlan sims died on write.
  - The full disk failed 11 of my jobs (exit 4/1); the 4 running ones were cancelled.
- **Freed (Aman approved these two; he did not approve the fresh usdz):**
  - 7 rejected checkpoints (ftC2, ftCC2_soup, ftC_w75, ftR, ftR_w50, ftT, ftT_w50; 30 GB);
  - the Alpamayo 2-Super weights (`.cache/hf`, 67 GB).
  - Result: 133 GB free. Kept: ftC.pt, ftC_w50.pt.
- **Fix:** `LOCAL_FS=1` in sim_local.sbatch stages scenes on node-local /local (262-291 GB free on ins057/061) and drops them at job end, as capture runs always did. Every fresh and lockbox run uses it.
- **Inventory** (13649482): alpasim 596 G = data 449 G (fs 360, nuplan_track 43, ft_ckpt 39 → 9) + .cache 67 + sandbox 46 + runs 21 + cache/uv 15.
- **Fresh 41-80 run 1:** v7 (13648582) is clean, 40/40 sessions, mean below. v5 (13648581) failed on the full disk; rerun.
- **Resubmitted:**
  - run-1 fresh: v5 41-80 13650548; v5 81-120 13650549; v7 81-120 13650550 (LOCAL_FS);
  - CFG w=2 screen: 13650551-53;
  - diagnostics on R6000: 13650554 (video), 13650555 (CoT).
  - Run 2 of the fresh check goes in after these.

## 2026-10-24 18:30 Diagnosis: route snap at highway speed causes the early post-handover failures
- **Diagnostic run** (13650554, R6000, v7 + A15_LOG_PLANS) gives the first plans after handover.
  - 0e05cb0c (session 072ee540, 32.5 m/s): plan 1 is **snapped**. Route first waypoint [42.1, -3.9]; plan lateral -3.29 m at 1 s.
    Plan 2 sees the route 3.8 m to the LEFT ([41.9, +3.8]): the car overshot right and left the corridor at 3.0 s.
  - 2466266e (6b4e8f0c): plan 2 snapped toward a route 6.5 m left at 41 m → offroad at 3.8 s.
  - 15fd88d6 (a5e457ec, 24.6 m/s): plan 1 snapped → offroad at 3.2 s.
  - 32be03e4 (a2eca934, 37.9 m/s): no snap; the model drifts left of a route bending right (a model error, not the snap).
- **Cross-check:**
  - v0.3 without snap **passes** 0e05cb0c (1.00) and 15fd88d6 (1.00);
  - v0.3 + snap fails 15fd88d6 (offroad at 0.11);
  - every snap config fails 0e05cb0c except v03snap.
- **Mechanism:** route_path bridges from the car to the route's first waypoint (~40-42 m) at any speed. At 32 m/s that is a 3.9 m lateral move in 1.3 s, ~14 m/s² peak lateral acceleration. Humans take 3-5 s.
- **Fix (candidate):** `A15_SNAP_MIN_T=T`: join the route at the first waypoint at least T × speed away.
  - Identical below 40/T m/s (13.3 m/s for T=3), so it only touches fast snaps.
  - Unit tests in tests/test_route_nav.py pass (in the v7 image); the tests are kept local until adoption, because cluster jobs test the main src.
- **Screen on R6000 (free GPUs), paired on the same hardware, old bench, OFFICIAL_MIMIC:**
  - v7 baseline (src_cand, all candidate switches off): 13651323/25/27;
  - v7 + SNAP_MIN_T=3: 13651324/26/28.
- **Fresh 41-80, L40S run 1:** v5 (13650548, LOCAL_FS) 0.774 vs v7 0.816, **+0.041 [-0.022, +0.110], W/L 9/4**. v5 81-120 is done (13650549); v7 81-120 is running (13650550).
- Diagnostics: the v7 video run 13650554 completed; the CoT run 13650555 FAILED (exit 1, to inspect).

## 2026-10-24 19:25 **Fresh-scene confirmation, run 1 complete: v7 +0.046 [+0.000, +0.094]**
- **120 never-used val scenes, OFFICIAL_MIMIC, L40S, all batches clean** (40/40 sessions, 0 errors):
  - v5 (13648577, 13650548, 13650549) **0.761**;
  - v7 (13648578, 13648582, 13650550) **0.807**;
  - diff +0.046 [+0.000, +0.094], W/L 22/16, 82 ties.
  - By batch: +0.072 / +0.041 / +0.026.
- **Failures v5 → v7:** corridor 11 → 6, offroad 7 → 3, at-fault 2 → 4, partial-progress passes 20 → 25.
  - The at-fault rise is mostly not new. 7d16a9a9 and 8a812020 fail in both. 9f393c8a was offroad in v5 (score 0 either way).
  - **dc401449** is new: v5 0.57 → v7 lateral at-fault collision at 46%. Check it in run 2.
- **Reading:** the old-bench gain under the same conditions was +0.054/+0.056. The fresh result (+0.046) is close, so the winner's curse is small here. The interval only just touches 0.
- **Run 2 queued** (LOCAL_FS): v5 13652022/24/26, v7 13652023/25/27.
- R6000 screen: v7 baseline (src_cand) 1-40 13651323 done; snap-min-T=3 batches are running.
- **Screens, batch 1-40:**
  - **CFG w=2** (13650551, L40S) vs mimw50nav L40S (0.797): 0.808, +0.010 [-0.027, +0.058], W/L 2/3.
    - Fixed: 0a228e44 corridor → pass. Lost: 0810968d 0.68 → 0.18; 00fabf60 1.0 → 0.94.
    - Driver peak **15,214 MiB** vs 14,914 without CFG (+300 MiB on L40S; H100 runs ~0.4 GB higher, so ~15.6 GB of the 16,384 MiB cap). Mean inference 1.79 s.
  - **SNAP_MIN_T=3** (13651324, R6000) vs R6000 v7 (13651323, 0.811): 0.786, -0.025 [-0.075, +0.000], W/L 0/1. Lost: 000525f6 pass → corridor.
    - The bridge changed as designed in 0e05cb0c: lateral -1.53 m at 1 s, was -3.29.
    - **But 0e05cb0c still left the corridor**: lateral distance to GT 3.93 m peak (was 4.55), corridor at about the same time.
  - **Diagnosis corrected:** in 0e05cb0c the route's first waypoint ([42.1, -3.9]) is NOT where the recorded car went. Moving toward it moves the car away from GT. A gentler bridge only slows the departure.
    - v0.3 without snap passes this scene; any snap there fails.
    - The issue is "route offset from GT at handover", not bridge steepness.
    - SNAP_MIN_T is unlikely to help. 41-80 and 81-120 still run (15fd88d6 and 2466266e are there) before dropping it.
- **CFG w=2, batch 41-80 (13650552):** 0.710 vs mimw50nav 0.780, **-0.070 [-0.133, -0.015], W/L 2/9**.
  - 22fa5c92 1.0 → 0.01 (stall); 1d6e30bc 0.45 → 0.02; 1d879259 → at-fault; 26004ca8 → offroad; 268949d2 → corridor.
  - Over 80 scenes: about -0.03. **CFG on nav text rejected**: amplifying the instruction makes the car hesitate or swerve more. The 81-120 batch (13650553) was cancelled to free the L40S for fresh run 2.
- **SNAP_MIN_T=3, full 120 on R6000 (13651324/26/28) vs R6000 v7 (13651323/25/27, 0.767):** 0.771, **+0.004 [-0.018, +0.026], W/L 2/3, 115 ties. Neutral; not adopted.**
  - Changes: 000525f6 pass → corridor; 1fcacc17 corridor → pass; 2837a7e8 corridor → 0.63; 26004ca8 0.85 → 0.70.
  - The highway scenes it targeted (0e05cb0c, 15fd88d6, 2466266e) still fail: the route there disagrees with the recording, and a gentler bridge does not change that.
  - R6000 is nearly deterministic (115/120 ties, as Blackwell was before). R6000 v7 vs L40S v7 (same config): -0.017 [-0.043, +0.007].
- **Stall diagnosis from the v7 videos (13650554, front-wide camera plus BEV, frames at 1.5/2.5/4/7 s):**
  - **2f227f2c** (rain, day): our lane's signal is **red**. The car stops at the line; the recorded driver turned right (likely a separate right-turn signal); the replayed follower rear-ends us.
  - **1d6e30bc** (night, rain, fog): red signals ahead. The recorded driver turned right on red; ours waits.
  - **2c84ac16** (night intersection, route turns left): the recorded driver went after a pause; ours waits all 20 s.
  - **Reading:** these stalls are the model obeying (or being unsure about) signals where the human went legally or opportunistically. A "go" rule would trade them for at-fault risk elsewhere, and we cannot confirm it deterministically. **Not pursued.**
  - The CoT diagnostic (13650555) did not crash: driver readiness timed out after 20 min (checkpoint shards took 4 min on slow storage, plus the expert load and quantization). Not rerun, since the videos answer the question.
- **State of PAI levers on 24 Oct evening:**
  - CFG nav rejected (-0.03 on 80);
  - SNAP_MIN_T neutral (+0.004);
  - the stall fix is not viable;
  - **v7 (ftC_w50 + nav + forced token + snap) is the candidate**, pending fresh run 2.
  - The built `teams/amanupg:v7` image (git 4d809ec code, expert sha f9da7cf3, permissions fixed) has the same policy code the fresh runs used ($A/src/policy.py md5 f45182e3). driver.py differs only by default-off logging.

## 2026-10-24 22:25 **With the Harmonizer off, L40S closed loop is deterministic**; lockbox launched
- **Fresh run 2 = run 1 exactly:** 1-40 v5 0.787/0.787 and v7 0.860/0.860; all 40 scenes tie for both (13652022-25 vs run 1). 41-80 is also identical (v5 0.774, v7 0.816).
  - The L40S run-to-run noise measured before (±0.05 per batch, 13618596 etc.) was in **Harmonizer-ON** runs: the diffusion Harmonizer is the source of the non-determinism.
  - Board-condition (OFFICIAL_MIMIC) runs repeat bit-for-bit.
  - **So v7's fresh +0.046 [+0.000, +0.094] is exact on those 120 scenes.** The interval reflects scene sampling only, and more scenes, not reruns, tighten it.
  - Run-2 81-120 (13652026/27) cancelled as redundant.
- **dc401449** repeats: v5 0.57, v7 lateral at-fault collision (a real, repeatable v7 loss).
- **Lockbox (201 never-used scenes, `scenes/pai_lockbox_part_0..5`), v5 vs v7, OFFICIAL_MIMIC, L40S, LOCAL_FS:**
  - parts 0-4 (40 each) + part 5 (1 scene): v5 13652824/26/28/30/32/34, v7 13652825/27/29/31/33/35.
  - With the fresh 120 this gives 321 held-out scenes for the final v5 vs v7 decision.

## 2026-10-24 23:50 Aman: "keep pushing, we have a week, not submitting soon". New improvement round (all screened deterministically, Harmonizer off)
- **Lockbox (201 scenes) running:** v5 part 0 done (13652824); the rest are running or pending.
- **Screens on the old 120, OFFICIAL_MIMIC, L40S, vs mimw50nav (L40S, deterministic):**
  - **Weight-mix sweep:** ftC_w40 (0.6 base + 0.4 C) and ftC_w60 (0.4 + 0.6), from averaging job 13653048. Screens: w40 13653050/54/58, w60 13653051/55/59.
  - **Controller idx_start_penalty=15** (default 10; 5 hurt): 13653052/56/60.
  - **Run E** = run C recipe (`--epochs 2 --lr 3e-5`) on ft_runE = ft_runC + the unused on-policy DAgger from ftC (dagC_e0-e2, 1,094 samples).
    - Train 13653047; average ftE_w50 13653049; screen with nav 13653053/57/61.
- **Nav-conditioned fine-tune (train/inference mismatch fix).** v7 uses route text at inference, but every fine-tune trained with `nav_text=None`. Now:
  - the driver stores each capture's route instruction (`nav`, the same `route_nav_text` rule as inference; "" when none). Code: `_capture_nav` in src_cand driver.py;
  - `load_sample` returns it; `train.py --nav` puts it in the prompt. Old samples without it train as before, which acts as nav dropout.
  - Test: tests/test_capture_nav.py (round trip, turn/straight), passed in the v7 image. Main `src/a15_driver/policy.py` is unchanged (md5 f45182e3 = v7); driver.py with the capture change lives only in src_cand.
- **Captures with nav, 280 new train scenes** (none used before, none in val), A6000/R6000/L40, 4 afterany lanes:
  - force-GT `capnav_n0..n7` (160 scenes): 13653067/69/71/73/75/77/79/80;
  - DAgger with **the v7 policy itself** (ftC_w50 + nav + forced + snap) `dagv7_v0..v5` (120 scenes): 13653068/70/72/74/76/78.
  - Next: run N = base init on ft_runE + the new nav captures with `--nav`, then w50, then screen with nav.
- **Lockbox parts 0-1 (80 scenes, clean):** part 0 v5 0.674 → v7 0.745, +0.071 [-0.042, +0.193], W/L 9/6; part 1 0.768 → 0.818, +0.050 [-0.030, +0.130], W/L 5/3. Same direction and size as the fresh set. Parts 2-5 are running.
- The ftC_w40/w60 averages are done (13653048). Run E training and the nav captures are pending (A6000/R6000 queue). Fileset 112 GB free.

## 2026-10-25 01:10 **v7 confirmed on held-out scenes: +0.044 [+0.005, +0.081] over 281**
- Lockbox part 2 (40) +0.032 [-0.074, +0.140], W/L 7/5; part 3 (40) +0.014 [-0.131, +0.165], 10/9; part 5 (1 scene) tie. All clean.
- **Held-out so far (fresh 120 + lockbox parts 0-3, 5 = 281 scenes, OFFICIAL_MIMIC, L40S, deterministic):**
  - v5 0.746 → v7 0.790, **+0.044 [+0.005, +0.081], W/L 53/39.**
  - The interval excludes 0: v7 passes the "deterministic gains only" rule against v5. Lockbox part 4 (40 scenes, v7 13652833) is still running.
  - Expected official effect at ~0.6×: about +0.026 scene score.

## 2026-10-25 01:55 **Held-out verdict, 321 never-used scenes: v7 +0.057 [+0.023, +0.093]**
- Lockbox part 4: v5 0.687 → v7 0.841, +0.153 [+0.063, +0.252], W/L 12/9.
- **Lockbox (201):** 0.726 → 0.790, **+0.064 [+0.016, +0.113]**, W/L 43/32.
- **All held-out (fresh 120 + lockbox 201):** 0.739 → 0.796, **+0.057 [+0.023, +0.093]**, W/L 65/48, 208 ties. Board condition (Harmonizer off), deterministic L40S.
  - Failures v5 → v7: corridor 37 → 22, offroad 15 → 12, **at-fault 9 → 11** (the tie-break metric; worth a look), partial passes 50 → 60.
  - The old-bench estimate (+0.054/+0.056) is confirmed on held-out data, so no winner's curse to speak of.
  - Expected official effect at ~0.6× local: about +0.034 scene score.
- **v7 is the PAI final-entry candidate under the deterministic-gains rule.** Anything else must beat v7 on the same held-out sets.
- Screens: ftC_w40 + nav, 1-40 (13653050): 0.785 vs w50 0.797, -0.013 [-0.038, +0.004], W/L 4/5.
- **At-fault collisions on held-out (v5 9 → v7 11), per scene:**
  - shared: 7d16a9a9, 7dc59d0d, 8a812020, c92137b4, d32206ef;
  - fixed by v7: 4f779a92, 58814d76, dd0f84df (v7 passes), and 4a863a41 (→ offroad);
  - **new in v7:** 403d558a, 6e6cbe67, a784414a, b661da67 (FRONT collisions late in the scene, at 72-83% of 20 s), dc401449 (lateral); 9f393c8a was offroad in v5.
  - **Reading:** v7 closes on slowing lead vehicles late in some scenes; it drives faster than v5 (the w50 average restored speed). This costs at-fault incident spacing (the rank tie-break) while scene score rises.
  - There is no detection signal to brake on outside the model, so this is a known cost, not something fixable this week.
- **Screens vs v7 (mimw50nav, old 120, L40S, deterministic):**
  - ftC_w40 + nav: -0.013 / -0.030 / +0.029 (≈ -0.005, neutral);
  - ftC_w60 + nav: -0.018 / -0.026 / -0.017 (≈ -0.020, worse);
  - controller idx_start_penalty=15: -0.056 / -0.080 (81-120 cancelled; clearly worse; both directions from the default 10 hurt).
  - **w50 and the default gains stay.**
- **Finals-condition check (Harmonizer ON), fresh 120, v5 vs v7, L40S, LOCAL_FS**, queued. Harmonizer-on runs are noisy, so this is one run for direction, not proof.
- 03:30: the nav captures (13653067-80) were stuck on ReqNodeNotAvail (A6000/R6000/L40 busy). Their constraint was widened to include l40s (scontrol update Features).
- **Run E trained (13653047, 2,227 steps, 4.9 h):** open loop on its val set ADE 2.109 → 1.899 m (-10%); length ratio median 1.001 → 0.999; weights moved 8.2e-2. ftE_w50 built (13653049). Screen 13653053/57/61 pending.
- **Nav captures work:** capnav_n0 (13653067): 552 samples from 20 sessions, 352 MB. In a 28-sample check, 6 carry a turn instruction ("Turn right in 21-37m", "Turn left in 27m") and 22 are "" (no turn). That is the ~20% rate seen at inference. n1-n3 are running.
- Fileset 102 GB free (99%).
- **Finals condition (Harmonizer ON), fresh 1-40:** v5 0.806 (13653600) vs v7 0.840 (13653601), +0.033 [-0.030, +0.102], W/L 4/5. The direction matches the board condition. v7 Harmonizer on vs off on these scenes: 0.840 vs 0.860. Batches 41-80/81-120 are running. Run E screen (13653053/57/61) has started.
- **Run E (ftE_w50 + nav) vs v7, old 120 (13653053/57/61):** +0.014 / -0.065 / -0.036, total **-0.029 [-0.065, +0.005], W/L 16/24. Rejected.** More on-policy DAgger did not help, the same as run D.
- **Finals condition (Harmonizer ON), fresh 41-80:** v5 0.816 vs v7 0.755, **-0.060 [-0.154, +0.034], W/L 4/12.** 1-40 was +0.033.
  - This is noisy: Harmonizer-on single batches move ±0.05 run to run. Here v5 scored higher with the Harmonizer on than off (0.816 vs 0.774) and v7 lower (0.755 vs 0.816), which also points to noise.
  - It matters because the finals use this condition, so a **second Harmonizer-on run** of fresh v5/v7 (all three batches) is queued to average. 81-120 run 1 is still running.
- **Node ins082 (A6000) has a GPU fault** ("CUDA unknown error" at torch init). Captures dagv7_v3 (13653074), capnav_n4 (13653075, empty 10 KB tar removed) and dagv7_v5 (13653078) failed on it; n6 (13653079) was cancelled there.
  - Resubmitted v3/n4/v5/n6 as 13654978-81, excluding ins082.
  - Pending n5/v4/n7 got ExcNodeList ins082.
- Done so far: capnav n0-n3, dagv7 v0-v1; v2/n3 running.
- **Run N (nav-conditioned fine-tune), first version**, started early on the captures finished so far rather than waiting for all 14:
  - data `ft_runN` = ft_runE (~11k, no nav) + capnav n0-n2 + dagv7 v0-v2 (~3k new samples, ~20% of them with a turn instruction);
  - `--epochs 2 --lr 3e-5 --nav`, base init. Train 13655207 (-t 11 h), then average ftN_w50 (13655208), then screen with nav on the old 120 (13655209-11).
  - Caveat: only ~600 samples carry turn text, so the nav-conditioning signal is thin. A run N2 on all captures follows if N shows a direction.

## 2026-10-25 10:05 **Finals condition (Harmonizer ON), fresh 120, run 1: v7 ≈ v5 (-0.012 [-0.063, +0.039], W/L 12/25)**
- By batch: +0.033 (4/5), -0.060 (4/12), -0.009 (4/8, 81-120: v5 0.680, v7 0.671). Means: v5 0.767, v7 0.756.
- Contrast with Harmonizer OFF (leaderboard): +0.057 [+0.023, +0.093] over 321 held-out scenes.
- **Open question with consequences:** the board (Harmonizer off), which decides who reaches the finals, favours v7. The finals reruns (Harmonizer on) may not.
  - On the screening 120 with the Harmonizer on, v7 was +0.084 vs v5 x2; the fresh scenes do not repeat that in run 1.
  - Harmonizer-on runs are non-deterministic (±0.05 per batch), so run 2 (13654801-06, running) is needed before reading anything.
  - Note: our fine-tune captures ran with the dev preset, i.e. Harmonizer ON, so the expert was trained on harmonized renders.

## 2026-10-25 18:35 **Finals condition (Harmonizer ON), fresh 120, two runs: v7 ≈ v5 (+0.006 [-0.042, +0.053])**
- Run 2 (13654801-06, all clean): v5 0.758, v7 0.781, +0.023 [-0.030, +0.078], W/L 15/25. By batch: +0.042 / -0.013 / +0.040.
- Run 1: -0.012. Per-scene average of both runs: v5 0.763, v7 0.768, **+0.006 [-0.042, +0.053], W/L 17/29**.
- A/A with the Harmonizer on: v5 run 1 vs 2 -0.009 (13/10); v7 +0.026 (19/13). Per-scene noise is large in this mode.
- **Reading:**
  - With the Harmonizer OFF (leaderboard initial runs) v7 is clearly better: +0.057 [+0.023, +0.093] over 321 held-out scenes.
  - With it ON (PAI final reruns) v7 is level with v5: no measurable gain, and no measurable loss either.
  - So v7 helps where it is needed to reach the finals (board rank), and is neutral in the finals themselves.
  - Rough board effect: +0.034 scene score at ~0.6× local → about +28 PCS (v5's +0.041 gave +34). That puts v7 near 2247, against UJS 2251 and VAIL 2258.
- **Run N (nav-conditioned) first attempt (13655207) died silently** at ~step 75 on ins039 (a node in the "Kill task failed" drain list earlier).
  - Resubmitted with all 14 nav captures (capnav n0-n7 + dagv7 v0-v5 + ft_runE = 32 tars), excluding ins039: train 13660566, average 13660567, screen 13660568-70.
- Fileset 75 GB free.
- **New screens on v7 (old 120, OFFICIAL_MIMIC, L40S, deterministic, vs mimw50nav):**
  - **2 Hz replanning** (A15_INFERENCE_INTERVAL_US=500000): 13660574/76/78. Faster reaction to braking leads (v7's new frontal at-fault crashes). Model work is ~16 s/wave at 1 Hz, so 2 Hz adds roughly +500 s to the official 9,522/14,400.
  - **20 flow-sampler Euler steps** (A15_DIFF_STEPS=20, new knob in src_cand policy.py; model default 10): 13660575/77/79.
  - **src_cand A/A** (all candidate switches off) on 1-40: must equal mimw50nav exactly if the refactored forced-rollout path is equivalent.

## 2026-10-25 21:15 Re-test earlier components inside the current recipe (Aman's question: are tried methods integrated?)
- Most experiments ran on top of the best recipe of their time. Fine-tune screens all used forced start + route snap, and since Oct 23 also nav.
- **Gap:** components tested before v7's parts existed were never re-tested on v7. Nav only helped with the fine-tuned expert, so such interactions are real.
- **Re-tests on v7, old 120, OFFICIAL_MIMIC, deterministic:**
  - sampler temperature 0.6: 13661431/34/37;
  - stall rescue (A15_CHOOSER=stall): 13661432/35/38;
  - route chooser v2 (K=6, margin 1, speed guard 0.9; +~2.5x model work): 13661433/36/39.
- **Fine-tunes rejected before nav, retrained and screened WITH nav** (their old checkpoints were deleted for space):
  - T (A2-Super teacher labels, ft_runR + ft_a2labels_x3): train 13661440 → ftT_w50 → screen 13661444/46/48;
  - R (RoaD): train 13661441 → ftR_w50 → screen 13661445/47/49.
  - Training runs on A6000/R6000/L40.
- **Cancelled (13661431-49), re-judged on merit:**
  - temperature 0.6 (zero twice before);
  - stall rescue (rarely triggers; stalls are signal compliance);
  - chooser v2 (gain gone on H100; 2.5× model work);
  - T/R retrains with nav (pre-nav 0.730-0.731 < w50 0.742; run E with nav was worse than v7).
  - Low priors; they would delay the better-motivated run N, 2 Hz and 20-step screens on the L40S bottleneck.
  - The answer to Aman's question: tried methods were run on the recipe of their time. Only these pre-v7 components were never re-run on v7.
- Report author line set (Aman, 2026-10-25): Aman Upganlawar, Columbia University, au2327@columbia.edu. Final-week roles: I push + run the PAI warmup, Aman submits.
- **v7 + 2 Hz replanning, 1-40 (13660574):** 0.749 vs 0.797, **-0.049 [-0.134, +0.041], W/L 4/14**. Rejected, as on v2 earlier. The other batches (13660576/78) were cancelled. Mean inference 1.72 s, peak 14,914 MiB.
- **v7 + 20 flow steps, 1-40 (13660575):** 0.784 vs 0.797, -0.014 [-0.100, +0.072], W/L 4/7 (inference 1.96 s, 14,794 MiB). 41-80/81-120 continue.
- Run N at step 1,600/3,701 (~4.6 h left). The challenge CLI session is still expired (needs Aman's configure-token before the warmup).
- **Tag register (2026-10-26):** PAI's final entry uses `v7` (built locally, not pushed). **`v8` is reserved for nuPlan** (alpasim-jepa, py123d v0.2.0 + a wait-then-go start fix, Nov 1+ submission), at their request.
- **To check (from alpasim-jepa):** on nuPlan the policy planned "wait ~2 s, then go" at every replan, so the car never started; a rule dropping the waiting prefix after 3 identical plans gave +0.015.
  - Check v7's plan logs for the stall scenes (diag 13650554 has A15_LOG_PLANS): a near-zero 2 s plan distance with a large 6.4 s distance, repeated, would be the same "always later" pattern, separate from genuine red-light stops.
  - Needs the socket.

## 2026-10-26 17:05 Run N, 20 steps, candidate A/A (results from while the socket was down)
- **src_cand A/A (13660580):** identical to mimw50nav on 1-40 (40/40 ties). The refactored forced-rollout path is exact, so all src_cand screens are clean comparisons.
- **Run N** (nav-conditioned fine-tune: ft_runE + all 14 nav captures, 14,821 samples, `--nav`, 3,701 steps, 8.5 h):
  - open loop ADE 1.944 → 1.619 (-17%); weights moved 0.107.
  - ftN_w50 + nav vs v7, old 120: +0.018 / -0.030 / +0.001, total **-0.004 [-0.044, +0.036], W/L 17/21. No gain; not adopted.** Matching train and test prompts on nav did not change closed-loop driving.
- **20 flow-sampler steps (v7):** -0.014 / -0.072 / -0.066, total **-0.051 [-0.091, -0.013], W/L 11/28. Rejected.** The model was trained for 10 Euler steps.
- **v7 stays the final candidate.**
- Process note: one remote check fell back to direct ssh when the socket file was missing; it timed out with no connection made. Every remote command is now gated with `ssh -O check &&` and uses `-o ControlMaster=no`.
- Fileset 56 GB free.
- **Wait-then-go check (alpasim-jepa's nuPlan finding) on the v7 stall scenes (diag 13650554 plan logs):**
  - present in **1d6e30bc**: ~10 consecutive plans with 2 s distance < 0.6 m but 6.4 s distance 3-14 m while stopped; the start keeps slipping;
  - weaker in 2c84ac16 (6.4 s distance 2-5 m);
  - 268949d2 is a genuine full stop (0.0 m at both horizons for 13 plans, likely a red light); 2f227f2c stops 3 plans then plans and starts moving.
- **Candidate `A15_WAITGO_K=3`** (src_cand; `route_choice.start_now`): when stopped (< 0.5 m/s) and the plan moves < 1 m in 2 s but > `A15_WAITGO_FAR` (5 m) by 6.4 s, for 3 plans in a row, the plan's waiting prefix (points within 0.3 m) is dropped so the move starts now.
  - Tests in tests/test_route_nav.py pass in the v7 image.
  - Screen on the old 120 (deterministic, vs mimw50nav) queued. If positive, held-out 321 follows.
- Wait-go screen moved to R6000 (L40S saturated); compared against R6000 v7 (all_bwv7, src_cand, all switches off, 0.767). L40S jobs 13671507-09 cancelled.
- **Process slip (20:10):** a socket check piped through head did not gate. With the socket missing, one read-only squeue/sacct query ran over a direct ssh login. Gate fixed (no pipe on the check). R6000 wait-go jobs 13671759-61 still pending.
- alpasim-jepa (20:20): since ~17:00 the dept fileset's small-file/metadata access is ~10x slower (big reads fine). PAI risk: the per-job env build from the shared uv cache (many files). If env builds time out, stage cache/uv as a tar node-local.
- **2026-10-28: v7 image re-verified** (local `teams/amanupg:v7`, id 540284f12d4a): runs as alpasim; expert readable, sha256 f9da7cf3; policy.py md5 f45182e3 (= held-out-evaluated code); env A15_NAV=1 intersect, FORCE_TRAJ_START=1, NUM_SAMPLES=1, 1 Hz, snap 2.0/15. **Ready to push.** Blocked on Aman: challenge CLI session expired (configure-token); Insomnia socket closed since 10-26 20:10.

## 2026-10-28 13:20 Aman: keep training to the end; final fine-tune round (queued for when the socket returns)
- Aman (17:11 UTC): wants more training, accepts submitting at the very end. Deadline gate is unchanged: only a held-out-confirmed gain replaces v7.
- Tally so far: 9 expert fine-tunes (A, B, C, C2, D, E, R, T, N). Only run C helped (via w50 + nav = v7), and C's seed-1 retrain scored 0.686 vs 0.737, so training-seed variance is about as large as any data/label change we tried.
- **Not yet tried:** every run started from the released expert. None continued from v7's own expert, which is the second DAgger iteration in the textbook sense.
- **New trainer flag `--init <ckpt>`** (src/a15_ft/train.py) loads an expert checkpoint before training; asserts every key is an expert key.
- **Round (launch on socket return):**
  - V1 "iterate v7": `--init ftC_w50.pt --nav --lr 1e-5 --epochs 1` on the on-policy v7 DAgger + nav captures (dagv7_v0-v5 + capnav_n0-n7), then 0.5 ftC_w50 + 0.5 V1.
  - V2 "seed lottery": run C recipe exactly (ft_runC, base init, lr 3e-5, 2 epochs), --seed 2, then w50.
  - Screen each vs v7 on the old 120 (deterministic), then held-out 321 only if positive.
- **Hard cutoff:** anything not held-out-confirmed by Oct 30 evening is dropped, and v7 ships. The v7 push + warmup go first regardless (needs configure-token).
- **Correction (14:00):** Slurm was never down. My commands skipped `source /etc/profile` (configless DNS lookup fails without it).
- **Alpamayo 2-Super takes route text** (prompt component `nav_instruction`, as in NVIDIA's examples/two_gpu_nav_cfg_demo.py), and its pixel budget (163,840-196,608) already fits our 576x320 frames, so low resolution was not a cause of run T's result.

## 2026-10-28 14:20 Run T2: Alpamayo 2-Super teacher + RoaD, done properly (Aman: "go ahead with the proper A2 teacher retry")
- **States:** v7 itself drives (ftC_w50 + nav + forced + snap; sample 0 = the v7 plan) on **240 unused train scenes** (`scenes/pai_train_t0..t11`).
  - Captured at each 1 Hz plan with all 6 cameras, its 16 own samples (`A15_CHOOSER=record A15_NUM_SAMPLES=16 A15_CAPTURE_AT_INFER=1`), the recording, and the route text; ego within 4 m of the recording.
  - Jobs 13690848-59 (A6000/L40/L40S, src_cand).
- **Teacher:** `a2_label.py --k 6 --nav`: 6 Alpamayo 2-Super samples per state, given the same route text. Weights download to node-local disk (`A2_LOCAL=1`; shared fileset 29 GB free). 2 jobs × 2 R6000: 13690860-61, after smoke 13690824 (6 samples, forced route text).
- **Target (train.py):** RoaD rule over the union of the policy's 16 and the teacher's 6 samples. The policy's own sample wins when it tracks the recording as well (in-distribution), the teacher's where the policy has no good option, and a recovery blend when none is within 3 m. The trainer prints the split by source.
- **Training:**
  - T2a: base init, ft_runC + T2 states ×3, `--steps 2000 --lr 3e-5 --nav`, then 0.5 base + 0.5 (13690862 → 13690864).
  - T2b: `--init ftC_w50`, T2 states only, `--steps 600 --lr 1e-5 --nav`, then 0.5 ftC_w50 + 0.5 (13690863 → 13690865).
- **Screens:** old 120, OFFICIAL_MIMIC, L40S, vs mimw50nav (v7, 0.784): T2a 13690866-68, T2b 13690869-71. Held-out 321 only for a clear winner; cutoff Oct 30 evening, else v7 ships.
- Superseded and cancelled before starting: 13690812-19, 13690832-43 (6-camera DAgger capture without policy samples).
- **14:35 Deleted with Aman's approval:** ftE, ftE_w50, ftN, ftN_w50, ftC_w40, ftC_w60 (6 × 4.56 GB, all rejected). ft_ckpt now holds ftC.pt and ftC_w50.pt (v7's expert) only. Fileset 41 GB free afterwards: other users wrote about 15 GB in the same minutes.
- Aman: stay on Insomnia for now (renting held in reserve for the teacher labelling); he is pausing the other agents so this competition gets the account's full priority.
- **Wait-go (`A15_WAITGO_K=3`) vs v7, R6000, old 41-120 (80 scenes, deterministic):** +0.012 [+0.000, +0.037]. Exactly 1 scene changed, 79 identical: **1d6e30bc** 0.021 → 1.000. v7 stalls at a red light whose start keeps slipping, moves 1 m and is rear-ended; with the fix it drives 57/64 m. This is the stall diagnosed from the plan logs. No losses. Batch 1-40 (13671759) produced no results (sim side); not rerun.
- **Held-out confirmation, L40S, LOCAL_FS, src_cand + A15_WAITGO_K=3, paired with ho_v7 (frv7l + lbv7):** fresh 13692004-06, lockbox parts 0-5 13692007-12.
- **18:30 T2 progress:**
  - Smoke 13690824 passed: 6/6 labelled with route text, 5.3 s per state, peak 77.5 GiB; A2 download to node-local 938 s.
  - Union selection tested offline on a real capture + label: pool (22, 64, 3), picked a policy sample at 0.05 m (teacher best 0.35 m under a forced fake instruction).
  - Captures: 10/12 done, ~197 states per 20 scenes (4 m filter).
  - **Labelling split** to start sooner: early job 13698073 (2 GPUs) on the 10 finished tars; late job 13698074 (1 GPU) after the last two captures and the early job, labels only what is missing. L0/L1 13690860-61 cancelled; T2a/T2b now depend on 13698074.
- **Wait-go held-out fresh 1-40:** 40/40 identical to v7 (0.860). The other 8 parts are queued or running.
- **19:00 R6000 bottleneck:** both R6000 nodes (ins046/047) are filled with other users' multi-day CPU-only jobs. ins046 has 0 free cores; ins047 has 1 GPU + 2 cores but is PLANNED for queued work. The partition's MaxMemPerCPU 6400 MB turns a big --mem request into many cores. A2 labelling needs only ~6.6 GB RSS (smoke MaxRSS), so it is resubmitted as 1-GPU, 2-core, 12 GB jobs: A 13698202 / B 13698203 (shards of the 10 finished tars), late 13698204 (rest; after captures + A). T2a/T2b now depend on 13698204. Slurm's start estimate for A is 21:48.
- **19:15 all 12 T2 captures done: 2,466 states** (240 scenes). Labelling re-split into 3 equal 1-GPU shards over all 12 tars: 13698436-38 (2 cores, 12 GB each). The A/B/late jobs 13698202-04 were cancelled. T2a/T2b depend on all three shards.
- **Wait-go held-out fresh 41-80:** +0.014 [+0.000, +0.042], 1 win, 0 losses, 39 identical. **7b5c9fd6** 0.442 → 1.000 (v7 rear-ended after 24/69 m; wait-go drives 64/69 m). Held-out so far (fresh 1-80): 1 win, 0 losses, 79 identical.
- 19:20 Aman: stay on Insomnia (no rental for labelling).
- **19:30 Labelling plan (Aman: go with a rental):**
  - `a2_label.py` gains `--hard_first` (order by the policy's best-own-sample distance to the recording, the trainer's RoaD measure; a check on 2 real samples matched train.road_target to 4 decimals), `--min_policy_dist` (skip states where the policy already has a sample that close), `--reverse`, and atomic label writes.
  - Insomnia backup X 13698482 (1 R6000 GPU, `--k 6 --nav --hard_first --min_policy_dist 0.5`); T2a/T2b depend on it. If the rental's labels arrive first: copy them into data/ft_a2labels_t2, then scancel X (afterany releases training).
  - Rental bundle staged on the Mac (scratchpad/rental: 12 tars 2.7 GB, alpamayo2 code, labeller, run_label.sh). Waiting on Aman for the host.
- **19:55-20:20 Rental (Aman approved; RunPod via his Chrome session): pod 1mmp3tg546mej7, 4× RTX PRO 6000 96 GB, $10.00/hr, 300 GB container disk, Mac public key via PUBLIC_KEY env.**
  - Env: torch 2.8.0+cu128, flash-attn 2.8.3. A2 Super snapshot 00554695 (same as Insomnia), downloaded in 63 s.
  - Labelling all 2,466 states (Aman: no compromises), `--k 8 --nav --hard_first`, 4 GPU shards. Peak 81.1 GiB, ~8.5 s/state per GPU; pass 1 on the 7 uploaded tars, pass 2 (rest, skip-existing) after the upload.
  - Insomnia jobs re-sized to measured RAM: training MaxRSS ~6.8 GB → T2a/T2b 32 GB (were 120 GB = 20 cores); sim MaxRSS ~32 GB → pending sims 56 GB.
- **Wait-go held-out fresh 81-120:** +0.000 (98694f91 0.455 → 0.465, a slightly earlier start; 39 identical). Fresh 120 total: 2 wins, 0 losses.
- 20:15 Aman added $10 to RunPod (total $30).
- 20:25 Insomnia backup labeller 13698482 cancelled (the rental is labelling); T2a/T2b put on hold until the rental labels are copied in.
- **Wait-go held-out lockbox part 0:** 40/40 identical (0.745). Held-out so far 160 scenes: 2 wins, 0 losses.
- **21:00 Rental labelling done:** 2,466/2,466 labels (493 with route text), `--k 8`, about 55 min on 4 GPUs. Copied pod → Mac → Insomnia data/ft_a2labels_t2 (2,466; x3 view 7,398). Pod 1mmp3tg546mej7 **stopped** ($0/hr); RunPod balance $17.66 of $30 (run cost ~$12).
- **T2a 13690862 and T2b 13690863 released, running on ins093 (21:06).**
- **What the teacher adds (offline, trainer's RoaD rule over the union of the policy's 16 + the teacher's 8):**
  - Chosen target: policy 1,519 (62%), teacher 636 (26%), recovery 311 (13%).
  - Best-sample distance to the recording, median / p90: policy (16) 0.61 / 3.31 m; teacher (8) 0.70 / 3.49 m.
  - At an equal budget (best of 8), the teacher is closer in 68% of states (0.70 vs 0.76 m). Per typical sample they are equal (0.83 vs 0.82 m).
  - In states where the policy's best is > 1 m off (975), the teacher is closer in only 25%.
  - **So A2 Super is not a markedly better driver than v7 on v7's own states. Its value is mostly extra candidate diversity.** T2 is in effect RoaD on v7 with a 26% teacher share; expect a modest effect.
- **Wait-go held-out lockbox parts 1, 2, 4, 5:** parts 1, 2 and 5 identical (81 scenes); part 4 +0.001 (f8ee31cb 0.466 → 0.489, slightly earlier start). Held-out so far 281 scenes: 3 wins (2 large), 0 losses. Part 3 running.
- **Wait-go (A15_WAITGO_K=3) held-out, all 321 scenes (fresh 120 + lockbox 201), OFFICIAL_MIMIC, L40S, deterministic: 0.7964 → 0.8002, +0.0037 [90% +0.0001, +0.0089]; 4 wins, 0 losses, 317 identical.** Lockbox part 3 adds one win (+0.015 on that batch). Every changed scene is a stalled start that now goes. **It meets the deterministic-gains bar → adopt in the final image** (an env setting on top of v7; the code path is the src_cand driver, A/A-identical to v7 when the switch is off).
- **22:00 v7 rebuilt with wait-go** (local only, not pushed): image 05e650bdbecb, git 6ed1db2. Checked as alpasim: driver code md5s = src_cand (driver cfa0141e, policy b880af2b, route_choice 00527e05, trajectory ade286bc, capture a1da5264), the exact code of the held-out runs. Env: v7's settings + A15_WAITGO_K=3, A15_WAITGO_FAR=5.0. Expert sha f9da7cf3 (ftC_w50). test_route_nav and test_driver_plan_storage PASS in the image. If T2 clears held-out, only expert.pt changes.
- **23:25 T2b done** (2 h 15 min): open loop ADE 2.79 → 2.70, weights moved 2.1e-2; ftT2b_w50 built (0.5 ftC_w50 + 0.5 T2b). Screen 13690869-71 queued. **Fileset at 8.3 GB free** (others wrote ~33 GB). T2a saves a 4.5 GB checkpoint every 100 steps. Asked Aman to approve deleting ftT2b.pt and ftC.pt (9 GB).
- 23:30 T2b screens on L40S (13690869-71) are queued behind busy GPUs. Duplicate R6000 screens 13700794-96 (src_cand, paired with all_bwv7 0.767) were added; the first set to finish decides.
- **T2b screen, old 1-40 (13690869, L40S, OFFICIAL_MIMIC) vs mimw50nav (v7): 0.797 → 0.851, +0.054 [90% -0.003, +0.120], W/L 8/1, 31 identical.** Corridor zeros 0a228e44 and 0d79a6a0 now pass; rear-end scenes 0cc7fa67 0.68 → 0.96 and 0b72d117 0.79 → 0.99; one loss, 0810968d 0.68 → 0.19 (low progress). Screening set only (winner's curse).
- **T2b held-out queued early** (to save time): ftT2b_w50 + v7 settings + A15_WAITGO_K=3 (the final-image candidate), src_cand, LOCAL_FS, L40S, paired with the v7+wait-go held-out runs (howg). Jobs 13702153-61.
- **01:15 Deleted with Aman's approval (junk only):** 7 usdz in data/fs/nre-artifacts/all-usdzs that no sceneset references (2b8896a3, 2c1e6689, 320e4bb8, 370b52d9, cae6e567, d7add9f5, e2741464; 12 GB), and 2 abandoned partial-download dirs scenesets/tmp6ubz5ab5 and tmpqtdzgdz7 (6.4 GB). all-usdzs now 200. **Kept** (Aman): the 80 usdz used only by the Oct 24 scenesets. Fileset 57 GB free.
- **Aman (01:15): no image push and no warmup without his explicit supervision and approval.** PAI has one official slot left. Final push/warmup happens with him present.
- **01:40 Extra experiments queued (behind the T2b held-out), all L40S:**
  - **Lateral chooser** = NVIDIA's CLOSEST_LATERAL from alpasim trajectory_selection.py (their A1.5 comparison preset: 6 samples, all 64 points, no skip). Pick the sample whose rig-frame y is nearest the driven (cached, post-snap) plan over the overlap; sample 0 (the K=1 plan) without a plan or under 4 points overlap. Separate tree `src_lat` (src untouched); `trajectory.plan_in_rig_frame` + tests/test_lateral_choice.py PASS in the v7 image. v7 expert, K=6, OFFICIAL_MIMIC, vs mimw50nav: 13702363-65.
  - **T2b unblended** (ftT2b.pt, init ftC_w50, no averaging) vs mimw50nav: 13702366-68.
  - **T2b Harmonizer ON** (finals condition), fresh 120, 2 runs (Harmonizer is not deterministic), vs frh_v7/frh_v7b: 13702369-74.
- **01:21 Insomnia socket dropped** (mux 'broken pipe'; the control file is gone). The fallback ssh to the placeholder host 'x' failed to resolve, so no direct login happened. All jobs are queued or running on the cluster and chained; results are read when Aman reopens the socket.
- **T2b screen 41-80 (13690870): 0.780 → 0.702, -0.078 [-0.158, -0.004], W/L 7/7.** Four new corridor exits (1fcacc17, 22fa5c92, 2837a7e8, 1a69f757), one new offroad (26004ca8); wins include 1d879259 0.51 → 1.00 and 2a0efada 0.81 → 1.00. Old 1-80 combined ≈ -0.012: T2b moves lane/branch choices in both directions. Held-out (321) decides.
- **T2b screen 81-120 (13690871): 0.775 → 0.820, +0.045 [-0.013, +0.113], W/L 12/2.**
- **T2b screen, old 120 total: 0.784 → 0.791, +0.007 [-0.033, +0.047], W/L 27/10.**
  - Failure counts v7 → T2b: rear-ended 26 → 16, at-fault 3 → 1, but corridor 8 → 11 and offroad 4 → 6.
  - So T2b drives longitudinally better (pace, fewer rear-ends) but makes more lateral/branch errors.
  - Not a screening winner on its own. The held-out run (13702153-61) decides. If the lateral chooser (consistency) cuts corridor exits on v7, T2b + lateral is the natural combination to test next.
- T2b duplicate R6000 screens 13700794-96 cancelled (the L40S screen finished).
- **T2b held-out fresh 1-40 (with wait-go) vs v7+wait-go: 0.860 → 0.853, -0.007 [-0.070, +0.051], W/L 5/2.**
- **T2b held-out fresh 41-80: 0.830 → 0.876, +0.046, W/L 11/0. Held-out running total (80): +0.020 [-0.016, +0.052], W/L 16/2.** (Running totals: scratchpad ho_total.py.)
- **T2b held-out fresh 81-120: 0.746 → 0.711, -0.035, W/L 5/4; lockbox part 5 (1 scene): 1.000 → 0.865. Running total (121): +0.000 [-0.030, +0.028], W/L 21/7.**
- **T2b held-out lockbox part 0: 0.745 → 0.698, -0.047, W/L 6/4. Running total (161): -0.012 [-0.042, +0.018], W/L 27/11.** Pattern: many small wins, fewer but larger losses.
- **T2b held-out lockbox part 3: +0.011, W/L 5/5. Running total (201): -0.007 [-0.035, +0.021], W/L 32/16.**
- **T2b held-out lockbox parts 2 and 4 done. Running total (281): +0.005 [-0.018, +0.029], W/L 49/21.** Part 1 still running.
- **T2b held-out FINAL (321 scenes, OFFICIAL_MIMIC, L40S, deterministic; T2b+wait-go vs v7+wait-go): 0.8002 → 0.8051, +0.005 [90% -0.017, +0.027], W/L 53/23. The interval includes 0 → NOT adopted (deterministic-gains rule).**
  - **Longitudinal gains are real:** full passes 200 → 220, rear-ended 52 → 38 (points lost 0.038 → 0.029), low progress 24 → 15.
  - **Zero-score events churn:** at-fault 11 → 13, offroad 12 → 12, corridor 22 → 23. T2b fixes many corridor scenes (4cb05d65, a6941bb6, afcb2149, d08a71c4, e904e9c0, e9384415: 0 → 1) and breaks others (8457182d, 9f8acd00, ba62e10d: 1 → 0; 3 new at-fault).
  - **Reading:** lane/branch decisions in these scenes are fragile coin flips that any model change flips both ways. The zero-score events dominate the score, so T2b's real pace gains net out. A selection rule that stabilises lateral choices (the lateral chooser, queued) is the natural complement; T2b + lateral is the next test if lateral helps v7.
- 03:20 T2b + lateral (K=6, src_lat) screen queued on old 120, behind the lateral-on-v7 screen.
- **Lateral on v7, old 1-40 (13702363): 0.797 → 0.710, -0.087 [-0.196, +0.026], W/L 6/10.** Over 714 plans the chooser kept sample 0 only 82 times (11%); choice counts are uneven across indices (4: 195, 1: 178, 5: 94, 3: 92, 0: 82, 2: 73), likely because of the per-call fixed seed. Mean inference 4.33 s (4 sessions per GPU, K=6). Batches 41-120 still running; T2b+lateral will be cancelled if the total stays negative.
- **Rejected (old 120, L40S, deterministic, vs mimw50nav):**
  - **T2b unblended** (ftT2b.pt): -0.036 [-0.090, +0.016], W/L 27/16 (-0.048 / -0.016 / -0.044 by batch). Averaging with ftC_w50 is better.
  - **Lateral chooser on v7** (K=6, NVIDIA's CLOSEST_LATERAL): -0.073 [-0.145, -0.002] over 80 scenes, W/L 12/22. It drives non-default samples 89% of the time. Batch 81-120 (13702365) and the T2b+lateral screen (13703443-45) were cancelled.
- **T2a done (13690862, 7 h 07 min):** open loop ADE 2.69 → 2.57, weights moved 7.9e-2 (base init, like run C). Average ftT2a_w50 (13690864) running; screen 13690866-68 follows.
- **04:22 ftT2a_w50 built (13690864). T2a held-out queued early** (+wait-go, src_cand, LOCAL_FS): 13703895-903; screen 13690866-68 also queued. The held-out will be cancelled if the screen is clearly negative.
- 06:00 Harmonizer-on T2b replicate b (13702372-74, ~45% done) cancelled: T2b was not adopted, and the GPUs go to T2a's screen and held-out. Replicate a (13702369-71) continues for one finals-condition reading.
- **T2a screen old 1-40 (13690866): 0.797 → 0.781, -0.017, W/L 7/3** (same pattern: more small wins, a few big losses).
- **T2a screen 81-120: -0.049, W/L 10/8. Old 80 total: -0.033 [-0.095, +0.025], W/L 17/11.** If 41-80 leaves the 120 total ≤ -0.02, the T2a held-out (13703895-903) is cancelled (low prior; GPU etiquette).
- **T2a screen, old 120: -0.041 [-0.086, +0.004], W/L 26/15 (by batch -0.017 / -0.056 / -0.049). Rejected; T2a held-out 13703895-903 cancelled.** Starting from the released expert with T2 data is worse than starting from v7's expert (T2b +0.007).
- **Overnight summary:** T2b (held-out +0.005, n.s.), T2a, unblended T2b and lateral are all rejected; wait-go is adopted. **The final candidate stays v7 + wait-go** (local image 05e650bdbecb).
- **T2b Harmonizer ON (finals condition), fresh 41-120, 1 run vs v7's 2 runs averaged: -0.003 [-0.041, +0.033].** 41-80: 0.799 vs 0.755; 81-120: 0.655 vs 0.705 (v7's own two runs differ by 0.068 on that batch). No finals-condition gain; consistent with not adopting T2b. Batch 1-40 still running.
- **T2b Harmonizer ON, all fresh 120 (1 run vs v7's 2 averaged): 0.768 → 0.774, +0.006 [-0.021, +0.033].** Level in the finals condition too. The queue is empty; v7 + wait-go stays final.

## 2026-10-29 09:50 Audit after Aman's challenge ("are the runs being sabotaged?")
- **Checked:**
  - Driver logs: every run loaded its intended expert (hoT2bwg/mimT2bw50nav → ftT2b_w50; howg/mimw50nav → ftC_w50; wait-go set only where intended).
  - Checkpoints are distinct: ftC_w50 f9da7cf3, ftT2b_w50 0dbdb528, ftT2b 996ccbb0.
  - T2b trained on teacher targets: 572 teacher, 1,399 policy, 280 recovery.
- **Found: T2b was under-trained.** Mean loss by sixth of training: 0.405 / 0.461 / 0.422 / 0.408 / 0.408 / 0.428 (flat). Open-loop ADE 2.79 → 2.70 (-3%); weights moved 2.1e-2. For comparison:
  - run C: loss 0.375 → 0.265, ADE -17%, ~5e-2;
  - T2a: loss 0.418 → 0.343, ADE -4.5%, 7.9e-2.
  - Cause: my conservative choice (lr 1e-5, 600 steps from ftC_w50), then halved by averaging. **So T2b's held-out +0.005 tests a weak dose, not the method.** It should have been flagged when the training log came in.
- **T2c (fair dose):** init ftC_w50, data ft_runT2a (run C data + T2 states ×3 with teacher + RoaD targets), `--steps 1200 --lr 3e-5 --nav`, 32 GB. Train 13706660 → average ftT2c_w50 (0.5 ftC_w50 + 0.5 T2c) 13706661.
  - **Gate before any screen:** loss must fall and open-loop ADE must drop ≥ 10%. Otherwise report "under-trained", not a result.
- **10:05 Audit of all fine-tunes (loss first→last sixth, open-loop ADE, weights moved).**
  - Trained properly: A (-21% ADE, weak lr), B (-24%), C (-17%), C2 (-10%), E (-10%), N (-17%), T2a (-4.5%).
  - D: train loss fell but held-out ADE only -1% (uninformative).
  - R: -14%, but judged with the Harmonizer on (±0.05 noise; -0.012 vs C is inside it).
  - T: -6%; only 30% teacher labels, no nav (weak by design).
  - **T2b: did not train** (flat loss, -3%, moved 2.1e-2).
  - **Reporting rule from now on:** every result carries a validity line (trained or activated; loss and ADE) and a power note, and a verdict that says "no gain shown (underpowered)" or "tested properly, no gain".
- **Relaunched (Aman: "retest R and T2c both", stronger fine-tunes):** each is trained, averaged 0.5 base + 0.5, then screened on old 120 (OFFICIAL_MIMIC, nav, L40S) vs mimw50nav.
  - **R2** = run R's exact recipe (ft_runR, `--steps 2000 --lr 3e-5`): 13706765 → 13706766 → 13706767-69.
  - **Cs2 / Cs3** = run C's recipe with `--seed 2` / `--seed 3` (the seed effect C vs C2 was 0.05): 13706770 → 71 → 72-74 and 13706775 → 76 → 77-79.
  - After both seeds, a soup (0.5 base + C, Cs2, Cs3 at 1/6 each) gets a fair deterministic test; the old C+C2 soup was also judged under Harmonizer noise.
  - **T2c** (13706660) running; its screen starts only after the validity gate.
