# Brief for the nuReasoning agent: what the AlpaSim session did (2026-09-27)

From the Claude session working on the **AlpaSim E2E Closed-Loop Challenge**, a separate challenge Aman is also entering. Aman asked for this so you don't mistake any of this work for yours, and don't undo it.

## 1. The rule Aman set
- nuReasoning and AlpaSim are separate projects, kept clearly apart.
- Ideas, models and code may be borrowed in either direction, but must stay recorded and separated.
- AlpaSim **reads** nuReasoning assets and **never writes** to the nuReasoning tree or its environments.
- Every borrowed item is listed in `alpasim/BORROWED.md`.

## 2. Where AlpaSim lives (not yours; please don't edit, move or commit it)
| | nuReasoning (yours) | AlpaSim |
|---|---|---|
| Local | `/Users/amanupg/Desktop/Nuscenes/nureasoning/` (moved 2026-09-28 with Aman's approval) | `/Users/amanupg/Desktop/Nuscenes/alpasim/`: its own git repo |
| GitHub | `amanupg/nureasoning-challenge-2026` | `amanupg/alpasim-challenge-2026` (new, private) |
| Cluster | `/insomnia001/depts/edu/users/au2327/nureasoning` | `/insomnia001/depts/edu/users/au2327/alpasim` |

- **Nothing in nuReasoning was moved.** An attempt to move the repo into `Nuscenes/nureasoning/` was blocked by the permission classifier and not retried. That layout change is Aman's decision.
- `alpasim/` sits inside the nuReasoning working folder but is excluded by `.gitignore`. Don't `git add` it, and don't treat its files as stray or untracked junk.

## 3. Changes made to the nuReasoning repo (commit `fa159c0`, pushed)
- **`.gitignore`:** added `alpasim/`.
- **`README.md`:** a short "Sibling project" note under the intro, pointing to `alpasim/`.
- **`LEDGER.md`:** new section at the end, "2026-09-27 Alpamayo 2 Super exists".
  - `nvidia/Alpamayo2-Super`: public since 2026-08-04, not gated. 32B backbone plus a 2.3B action expert; 71.6 GB of weights.
  - About 72 GB peak on one H100 at bf16, so it fits only our 80 GB H100s at bf16.
  - Reported minADE_6 at 6.4 s of 0.911 m.
  - **A candidate new path source for the nuReasoning chooser menu, same role as `a15`. Not started.** It needs H100 time and Aman decides GPU use. We missed it when we picked Alpamayo 1.5, despite our check-newer-versions rule.
- No nuReasoning code, artifacts, submissions, envs or jobs were modified.

## 4. Shared memory files (same memory directory as yours)
- Added `challenges-demarcated.md` (the separation rule) and `alpasim-campaign.md` (AlpaSim facts).
- Added two index lines to `MEMORY.md`.
- Nothing else in memory was changed.

## 5. nuReasoning assets AlpaSim reads (please don't rename, move or delete them without a note)
- `ext/a15env` (python), `ext/alpamayo1.5`, and `env/bin/python`. The last was used only to `pip install --target` into the AlpaSim tree; nothing was installed into your envs.
- `.cache/hf`: Alpamayo-1.5-10B, the Qwen3-VL-8B config, and the HF token (mode 600).
  - Used with `HF_HUB_OFFLINE=1`; at most harmless hub lock files may appear there.
- `src/a15_infer.py::load_model` and `src/a15ft_data.py::build_sample`, imported read-only.
  - If you change their signatures or behaviour, AlpaSim's `src/a15_fit.py` breaks. A heads-up in the LEDGER is enough.
- `cache/validation/*` and `img_a15/validation/*`: validation inputs, read only.
- **Future candidates, not used yet:** `runs_a15ft` (fine-tuned Alpamayo) and `runs_vla/q3vl2b_full_h0|h1` (nuVLA half-models).

## 6. Cluster footprint, so you don't mistake it for yours
- **Every AlpaSim Slurm job name starts with `as_`:**
  - `as_sif`, `as_nup`, `as_base`, `as_sbox`, `as_smoke` (done);
  - `as_a15fit`, job **12947478**, **running on 1 L40S** (ins038) for about 1-3 h.
  - Please don't cancel `as_*` jobs. Aman approved a one-day AlpaSim feasibility test.
- **CPU quota:** AlpaSim jobs count toward the same per-user CPU cap (about 320 CPUs), so they can briefly delay your jobs, and yours delayed ours today. I only ever ran `scontrol update` on my own `as_*` jobs; no nuReasoning job was touched.
- **Shared storage added under `.../au2327/alpasim`,** about 140 GB total. All of it counts toward the department's shared `/insomnia001` space.
  - `data/nuplan_track` 43 GB
  - `sif/` 22 GB
  - `sandbox/` 47 GB
  - `cache/apptainer` 30 GB (cleanup proposed to Aman, awaiting his OK)
- **Home directory:** `~/.cache/torch_extensions/py312_cu128/gsplat_cuda` is AlpaSim's (gsplat JIT build); `~/.apptainer` too.
- The HF dataset `OpenDriveLab/AlpasimChallenge2026_nuplan_track` and the NVIDIA `nre-ga` renderer image are AlpaSim-only downloads.

## 7. What AlpaSim has found (FYI; context only, not your task)
- **The official closed-loop simulator runs on Insomnia via Apptainer** (no docker or pyxis there).
  - A nuPlan one-scene smoke test passed on an L40S.
  - The nuPlan renderer needs L40S or H100 (sm89/sm90), not A6000.
- **Alpamayo 1.5 at bf16 in closed loop** (L40S, 4 cameras × 4 frames, 1 sample, reasoning on):
  - 22.1 GiB peak;
  - 1.15 s per plan (0.90 s VLM with a ~3,200-token prompt and ~17 generated tokens, 0.24 s diffusion);
  - ADE 3.14 m on 30 nuReasoning validation key frames.
  - This may interest you: the zero-shot reasoning is short, ~17 tokens.
  - Quantized (int8/fp8/int4) and no-reasoning variants are being measured now.

## 8. If something looks off
Don't change anything under `alpasim/` or `.../au2327/alpasim`. Note it in your LEDGER or tell Aman, and the AlpaSim session will handle it.
