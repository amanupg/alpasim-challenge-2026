"""Weight-space averaging of action-expert checkpoints (WiSE-FT / model soup).

    out = w_base * base + sum_i w_i * ckpt_i      (weights sum to 1)

Only the trained tensors (expert., action_in_proj., action_out_proj.) are averaged; base values come from the
released Alpamayo 1.5 safetensors. The result is a drop-in A15_EXPERT_CKPT.
Usage: python -m a15_ft.average --out soup.pt --base_weight 0.5 ftC.pt:0.5
       python -m a15_ft.average --out soup.pt --base_weight 0 ftC.pt:0.5 ftC2.pt:0.5
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open

from a15_ft.train import TRAIN_PREFIXES


def base_expert(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in idx if k.startswith(TRAIN_PREFIXES)]
    out = {}
    for shard in sorted({idx[k] for k in keys}):
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for k in keys:
                if idx[k] == shard:
                    out[k] = f.get_tensor(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_weight", type=float, required=True)
    ap.add_argument("ckpts", nargs="+", help="path:weight")
    a = ap.parse_args()
    parts = [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) for p in a.ckpts]
    total = a.base_weight + sum(w for _, w in parts)
    assert abs(total - 1.0) < 1e-6, f"weights sum to {total}, not 1"
    hub = os.environ["HUB"]
    model_dir = sorted(glob.glob(f"{hub}/models--nvidia--Alpamayo-1.5-10B/snapshots/*/"))[0]
    base = base_expert(model_dir)
    acc = {k: v.float() * a.base_weight for k, v in base.items()}
    for path, w in parts:
        sd = {k: v for k, v in torch.load(path, map_location="cpu").items() if not k.startswith("_")}
        assert set(sd) == set(base), f"{path}: key mismatch ({len(set(sd) ^ set(base))} keys differ)"
        for k, v in sd.items():
            acc[k] += v.float() * w
    out = {k: v.to(torch.bfloat16) for k, v in acc.items()}
    torch.save(out, a.out)
    print(f"saved {len(out)} tensors to {a.out}: base*{a.base_weight} + " + " + ".join(f"{os.path.basename(p)}*{w}" for p, w in parts))


if __name__ == "__main__":
    main()
