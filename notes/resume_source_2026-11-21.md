# Resume source, AlpaSim 2026 PAI track: update of 2026-11-21

Thread: alpasim-alpamayo-pai (`alpasim/`). Track closed on 2026-10-31. Raw facts and keywords for the resume agent;
no polished bullets.

The full record is unchanged and still valid: `notes/campaign_record.md` plus section 1.A of
`../RESUME_MASTER_2026-11-13.md` (Part 1). This file lists only what changed since 2026-11-13, the corrected final
standing, exact compute, and a keyword list. The merged `../RESUME_MASTER_2026-11-21.md` carries Part 1 in full with
these changes applied.

## 1. Changed since 2026-11-13

| Item | Nov 13 record | Now | Source |
|---|---|---|---|
| Final standing | best entry v5, PCS 2226, rank 16 [10–24] of ≥132 entries, "about 10th team" (board read 2026-10-31) | **11th (final), per Aman, 2026-11-21.** Best entry unchanged: v5, PCS 2226 | Aman; LEDGER L2243 for the 10-31 read |
| Track status | closed | closed; no further submissions possible | — |
| GPU-hours | "not recorded" | **845.8 GPU-h** on Insomnia, 485 GPU job runs (exact, Slurm accounting) | sacct, 2026-11-16 |
| Rented compute | ~$12 RunPod | unchanged: ~$12, about 3.7 GPU-h (4× RTX PRO 6000, teacher labels) | LEDGER |
| Technical report | written (4 pages, official template), not submitted | unchanged, not submitted (Aman's decision) | LEDGER 11-04 |
| Cluster footprint | folder deleted 11-05 | unchanged. Leftovers in home: `~/.cache/nre/harmonizer` 1.35 GB, scan output 20 MB (deletion pending Aman's OK) | `notes/storage_assessment_2026-11-16.md` |
| Local archive | 447 runs, 13 GB | unchanged: `cluster_archive/` 13 GB (447 closed-loop run folders, metrics and logs, no videos); `container/assets/` 21 GB (Alpamayo 1.5 weights + own expert.pt 4.2 GB) | du, 2026-11-16 |

No new runs, submissions or findings since 2026-11-13.

**Rank wording.** The 10-31 board read gave "about 10th team" (entry rank 16 [10–24]; PCS is refit as entries
arrive, and team rank was counted by hand). Aman gives 11th as final. Use **11th**. Keep these as separate, labelled
lines:
- peak: **rank 2 of 58** entries (v0.3, 2026-09-29)
- **4th team** (v5, 2026-10-21)

## 2. Exact compute (Slurm accounting, `sacct -X -D`, all runs including requeues)

PAI folder (`alpasim/`), 2026-09-25 to its deletion on 2026-11-05:

| GPU | GPU-h |
|---|---|
| L40S | 456.2 |
| RTX PRO 6000 Blackwell | 152.0 |
| A6000 | 92.2 |
| H100 | 91.4 |
| L40 | 54.0 |
| **Total** | **845.8** (485 GPU job runs) |

For the merged overview (all four threads; campaign 2026-09-25 to 2026-11-16 15:41 EST):

| Folder | GPU-h |
|---|---|
| nureasoning | 1,354.1 |
| alpasim (PAI) | 845.8 |
| alpasim-jepa (nuPlan) | 541.2 |
| jepa-nureasoning | 322.8 |
| **Total** | **3,063.9** (1,739 GPU job runs) |

Other campaign totals:
- 6,581 core-hours of CPU-only jobs
- peak 31 GPUs in use at once (2026-09-27)
- largest single job: 8 GPUs

Attribution is by the job's working directory, with `jn_*` jobs counted as jepa-nureasoning.

## 3. Keywords (where used)

**Models**
- NVIDIA Alpamayo 1.5: the 10B VLA driven in closed loop, made of a Qwen3-VL-8B backbone plus a 2.3B flow-matching action expert.
- Alpamayo 2-Super (34B): teacher for distillation labels (RoaD-style closed-loop targets).
- Vision-language-action (VLA) models, chain-of-causation reasoning traces (turned off: −0.165 scene score).

**Training**
- Flow matching: teacher-forced flow-matching MSE, 4 noise draws per prefill.
- Partial fine-tuning: action expert plus projections, 2.28B of 11.08B parameters.
- DAgger recovery labels, force-GT capture, on-policy DAgger.
- RoaD closed-loop SFT; outcome-weighted self-imitation (T2d); knowledge distillation.
- WiSE-FT weight averaging, model soups.
- SRAdamW: stochastic-rounding AdamW written by us to fix silent bf16 update loss.
- bf16 numerics, gradient accumulation.

**Deployment**
- int8 weight-only quantization (torchao); fp8 and int4 compared.
- 16 GiB VRAM contract; CPU-resident token embeddings; single dedicated CUDA thread; memory accounting.
- Fixed wall-time budget: 987 rollouts in 9.5k of the 14.4k seconds allowed.
- 1 Hz replanning with a cached plan re-anchored to the current pose.
- Forced trajectory-start token; route text in the prompt; wait-go start rule.

**Planning geometry**
- Route snap: cubic Hermite blend onto route waypoints, 2.0 m with a 15° angle guard. Corridor exits fell 44%; official scene score went 0.611 → 0.652.
- Frenet-style corridor reasoning, trajectory time-warp.

**Simulation and systems**
- NVIDIA AlpaSim: gRPC driver API, physics controller/MPC.
- NRE / NuRec neural reconstruction renderer, Harmonizer (diffusion post-pass).
- Docker compose ported to Apptainer on Slurm (`compose_apptainer.py`); no docker on the cluster.
- CUDA forward compatibility (cuda-compat 12.8 on driver 550); Warp NVRTC with a TMPDIR bind.
- Per-job gRPC port blocks (the fix for SO_REUSEPORT cross-talk).
- Node-local staging (LOCAL_FS), shared uv wheel cache (env build ~2,400 s → seconds).
- Health files, unit tests before each run, TERM/USR1 traps.
- Docker BuildKit image, AWS ECR push with digest verification, read-only / no-network image checks.

**Evaluation**
- Paired per-scene bootstrap intervals (`compare_runs.py`); A/A tests (L40S vs H100 −0.002 [−0.020, +0.016]).
- Deterministic closed-loop evaluation (Harmonizer off gives bit-for-bit repeats).
- OFFICIAL_MIMIC mode.
- Held-out 120-scene fresh set plus a 201-scene lockbox, 321 scenes in all.
- Failure taxonomy: at-fault, offroad, corridor, low progress.
- Local-to-official calibration (0.44–0.58×).
- PCS as a zero-one-inflated beta item-response (IRT) model. The lesson: validate the ranking metric, not the mean.

**Infrastructure**
- Columbia Insomnia HPC (Slurm, GPFS fileset): L40S, H100, RTX PRO 6000 Blackwell, A6000, L40.
- RunPod: rented 4× RTX PRO 6000.
- Public GitHub repo github.com/amanupg/alpasim-challenge-2026 (public since 2026-11-21): 314 commits through fe75bba.
- LaTeX report (official template); dated measurement ledger (2,273 lines).

**Process**
- Agent-run research under human direction (Claude Code sessions; Aman approved compute, spend, deletions and every submission).
- Problem-outward research method; validity-before-verdict rule; shared lessons playbook (`LESSONS_FOR_AGENTS.md`).

## 4. Challenges faced (pointer)

Unchanged from `campaign_record.md` section 7:
1. Optimised the mean scene score instead of the IRT ranking metric for the last decision: v7 was better on every visible metric, but PCS fell by 75.
2. Some verdicts were made on under-dosed or noisy tests.
3. The research grew outward from the current approach instead of from the problem.
4. Infrastructure slips: port contamination, a plan-storage bug, an idle Slurm dependency, quota filled by scene downloads, socket-gate slips.
5. Weeks of comparisons sat inside Harmonizer noise before the determinism switch was found.
