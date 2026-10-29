# What AlpaSim borrows from nuReasoning

The two challenges live in separate folders and separate repos:

| | nuReasoning | AlpaSim |
|---|---|---|
| Local | `Nuscenes/nureasoning/` | `Nuscenes/alpasim/` |
| Cluster | `/insomnia001/depts/edu/users/au2327/nureasoning` | `/insomnia001/depts/edu/users/au2327/alpasim` |
| GitHub | `amanupg/nureasoning-challenge-2026` | `amanupg/alpasim-challenge-2026` |

Rule: AlpaSim may **read** nuReasoning assets, never write into the nuReasoning tree.
Anything borrowed is listed here with its source, so either project can be rebuilt on its own.

| Borrowed | Source (read-only) | Used by | Why |
|---|---|---|---|
| Alpamayo 1.5 weights | `nureasoning/.cache/hf` (`nvidia/Alpamayo-1.5-10B`, `Qwen/Qwen3-VL-8B-Instruct` config) | `src/a15_fit.py` | avoid a second 22 GB copy |
| Alpamayo 1.5 code + env | `nureasoning/ext/alpamayo1.5`, `nureasoning/ext/a15env` (torch 2.8 cu128, transformers 4.57.1) | `src/a15_fit.py` | same verified install; quantization libs go in `alpasim/envs/overlay_torchao` instead of touching it |
| Model loader | `nureasoning/src/a15_infer.py::load_model` | `src/a15_fit.py` | Cosmos-Reason2 config swap, verified vocab 155697 / traj offset 151669, zero missing keys |
| Sample builder | `nureasoning/src/a15ft_data.py::build_sample` | `src/a15_fit.py` | same inputs as our zero-shot and fine-tune runs; logged 6.4 s futures for a quality proxy |
| nuReasoning validation clips | `nureasoning/cache/validation`, `nureasoning/img_a15/validation` | `src/a15_fit.py` | real multi-camera inputs until PAI scenes are available |
| ~~HF token~~ | not borrowed: AlpaSim uses HF account **`amanupg-columbia`** (the challenge account). Token in `alpasim/.cache/hf/token` (mode 600, never on a command line) | NuRec / nuPlan downloads | nuReasoning uses the personal `amanupg` account |

Considered but never used (2026-10-29: not needed by AlpaSim; nureasoning may clean them up): the fine-tuned
Alpamayo checkpoints (`nureasoning/runs_a15ft`, Motional 3-camera domain) and the nuVLA full-recipe half-models
(`nureasoning/runs_vla/q3vl2b_full_h0`, `_h1`). AlpaSim's own fine-tunes are in `alpasim/data/ft_ckpt`.
Anything that moves the other way (an AlpaSim idea or model reused for nuReasoning) is
recorded in the nuReasoning LEDGER, not here.

## Update 2026-10-19: nureasoning/ext/a15env deleted
`nureasoning/ext/a15env` was deleted on 2026-10-19 during an approved nureasoning cleanup; our use of it had not been listed in nureasoning's SHARED_DEPENDENCIES.md. AlpaSim no longer borrows that env. Every a15 job now builds the same 64 pinned packages (`envs/a15env-pins.txt`, taken from its 2026-09-26 install log) on node-local disk via `cluster/a15env_local.sh`.

Still borrowed from nureasoning/, read-only (sent to the nureasoning session for its do-not-delete list):
- `ext/alpamayo1.5`: model code, via a `.pth` in the per-job env.
- `ext/setup_a15.sh` and `ext/setup_a15.log`: provenance of the pins.
- `.cache/hf/hub`: `models--nvidia--Alpamayo-1.5-10B`, `models--Qwen--Qwen3-VL-8B-Instruct`, `models--Qwen--Qwen3-VL-2B-Instruct`.
- For the policy check scripts only: `src/` (`a15ft_data.build_sample`, `a15_infer.py`), `recon/devkit`, `cache/validation`, `img_a15/validation`.
