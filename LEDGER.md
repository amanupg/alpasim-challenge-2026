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
