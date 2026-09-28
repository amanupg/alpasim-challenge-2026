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

Candidates to borrow later (not used yet): the fine-tuned Alpamayo checkpoints
(`nureasoning/runs_a15ft`, Motional 3-camera domain) and the nuVLA full-recipe half-models
(`nureasoning/runs_vla/q3vl2b_full_h0`, `_h1`; 8 Motional cameras, the nuPlan-track rig).
Anything that moves the other way (an AlpaSim idea or model reused for nuReasoning) is
recorded in the nuReasoning LEDGER, not here.
