"""Fine-tune Alpamayo 1.5's action expert on simulator-rendered samples (capture mode, a15_driver/capture.py).

The VLM stays frozen; the action expert and its input/output projections train with the flow-matching loss
(a15_ft/flow_loss.py). Inputs are built exactly as the driver builds them at inference (same message, same
"<|cot_start|><|cot_end|>" prompt, same 576x320 frames), so the fine-tuned expert is a drop-in for the driver.

Usage: python -m a15_ft.train --data <dir with capture tars or npz> --out <ckpt.pt> [--steps N] [--lr 1e-5]
Also reports an open-loop check on held-out scenes before and after training: ADE to the recorded future and the
ratio of predicted to recorded 6.4 s path length (< 1 means the model drives slower than the recording).
"""
import argparse
import glob
import json
import os
import random
import tarfile
import time
import types
import zipfile

import numpy as np
import torch

from a15_driver.capture import load_sample
from a15_driver.policy import Alpamayo15Policy
from a15_ft.flow_loss import teacher_forced_flow

TRAIN_PREFIXES = ("expert.", "action_in_proj.", "action_out_proj.")


def load_model(model_dir, vlm_dir, processor_dir):
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    helper.BASE_PROCESSOR_NAME = processor_dir
    d = json.load(open(os.path.join(model_dir, "config.json")))
    d["vlm_name_or_path"] = vlm_dir
    d["attn_implementation"] = "sdpa"
    cfg = Alpamayo1_5.config_class(**d)
    model, info = Alpamayo1_5.from_pretrained(model_dir, config=cfg, dtype=torch.bfloat16,
                                              output_loading_info=True, low_cpu_mem_usage=True)
    assert not any(v for v in info.values()), info
    model.to("cuda")
    return model, helper, helper.get_processor(model.tokenizer)


def build_inputs(helper, proc, s):
    frames, slots = s["frames"], s["slots"]
    msgs = helper.create_message(frames.flatten(0, 1), camera_indices=torch.tensor(slots), nav_text=None)
    msgs[-1]["content"][0]["text"] = "<|cot_start|><|cot_end|>"
    tok = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                   return_dict=True, return_tensors="pt")
    t = lambda a: torch.from_numpy(np.asarray(a)).float()[None, None]
    return helper.to_device({"tokenized_data": tok, "ego_history_xyz": t(s["hist_xyz"]), "ego_history_rot": t(s["hist_rot"]),
                             "ego_future_xyz": t(s["fut_xyz"]), "ego_future_rot": t(s["fut_rot"])}, "cuda")


def collect(data_dir, scratch):
    """npz paths grouped by session; capture tars are extracted to scratch first."""
    for tf in sorted(glob.glob(os.path.join(data_dir, "*.tar"))):
        dst = os.path.join(scratch, os.path.basename(tf)[:-4])
        if not os.path.isdir(dst):
            try:
                with tarfile.open(tf) as z:
                    z.extractall(dst)
            except (tarfile.TarError, EOFError, OSError) as e:          # a capture killed while tarring
                print(f"{os.path.basename(tf)}: {type(e).__name__} ({e}); keeping the members that extracted", flush=True)
    groups, broken = {}, 0
    for p in glob.glob(os.path.join(scratch, "**", "*.npz"), recursive=True) + glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True):
        if not zipfile.is_zipfile(p):                                # a snapshot of a running capture can cut one file
            broken += 1
            continue
        groups.setdefault(os.path.basename(os.path.dirname(p)), []).append(p)
    if broken:
        print(f"skipped {broken} truncated sample files", flush=True)
    return groups


class SRAdamW(torch.optim.Optimizer):
    """AdamW for bf16 weights with stochastic rounding of the updated weight.

    Plain AdamW on bf16 weights loses most updates at lr 1e-5: |lr * step| is ~5e-4 of a typical weight, below bf16's
    2^-8 resolution, so round-to-nearest leaves the weight unchanged. Rounding the fp32 result stochastically keeps
    the update in expectation without fp32 master weights (which would not fit a 48 GB card next to the 11B model).
    Moments are kept in bf16 like the weights (their EMA steps of 5-10% are well above bf16 resolution)."""

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @staticmethod
    def _sr_bf16(x):
        i = x.view(torch.int32)
        i = (i + torch.randint_like(i, 0, 1 << 16)) & -65536          # add noise below bf16 precision, truncate
        return i.view(torch.float32).to(torch.bfloat16)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["t"] += 1
                gr = p.grad.float()
                m = st["m"].float().lerp_(gr, 1 - b1)
                v = st["v"].float().mul_(b2).addcmul_(gr, gr, value=1 - b2)
                st["m"].copy_(m); st["v"].copy_(v)
                upd = (m / (1 - b1 ** st["t"])) / ((v / (1 - b2 ** st["t"])).sqrt_().add_(g["eps"]))
                w = p.float()
                if g["weight_decay"]:
                    w.mul_(1 - g["lr"] * g["weight_decay"])
                w.add_(upd, alpha=-g["lr"])
                p.copy_(self._sr_bf16(w) if p.dtype == torch.bfloat16 else w)


@torch.no_grad()
def open_loop(model, helper, proc, paths, seed=42, forced=True):
    """ADE and predicted / recorded path-length ratio over the 6.4 s horizon (one sample per input, as the driver).
    forced: append <traj_future_start> as training does (the driver's A15_FORCE_TRAJ_START=1 path)."""
    model.eval()
    shim = types.SimpleNamespace(_model=model, _diff_kwargs={})
    ade, ratio = [], []
    for p in paths:
        s = load_sample(p)
        inp = build_inputs(helper, proc, s)
        inp = {k: v for k, v in inp.items() if not k.startswith("ego_future")}
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if forced:
                xyz, _ = Alpamayo15Policy._forced_rollout(shim, inp)
            else:
                xyz, _, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=inp, top_p=0.98, temperature=0.6, num_traj_samples=1, max_generation_length=4, return_extra=True)
        pred = xyz.reshape(-1, 3)[:, :2].float().cpu().numpy()
        gt = s["fut_xyz"][:, :2]
        ade.append(float(np.linalg.norm(pred - gt, axis=1).mean()))
        Lp = np.linalg.norm(np.diff(np.vstack([[0, 0], pred]), axis=0), axis=1).sum()
        Lg = np.linalg.norm(np.diff(np.vstack([[0, 0], gt]), axis=0), axis=1).sum()
        if Lg > 5:                                                  # ignore near-standstill samples for the ratio
            ratio.append(Lp / Lg)
    return {"n": len(paths), "ade": float(np.mean(ade)), "len_ratio_median": float(np.median(ratio)) if ratio else None,
            "len_ratio_mean": float(np.mean(ratio)) if ratio else None, "n_ratio": len(ratio)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp") + "/ft_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=0, help="0 = one epoch")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--val_max", type=int, default=150)
    ap.add_argument("--n_noise", type=int, default=4, help="(x_0, t) draws per VLM prefill")
    ap.add_argument("--unforced_eval", action="store_true", help="also report the open-loop check without forced start")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    hub = os.environ["HUB"]
    snap = lambda r: sorted(glob.glob(f"{hub}/models--{r}/snapshots/*/"))[0]
    model, helper, proc = load_model(snap("nvidia--Alpamayo-1.5-10B"), snap("Qwen--Qwen3-VL-8B-Instruct"),
                                     snap("Qwen--Qwen3-VL-2B-Instruct"))
    groups = collect(a.data, a.scratch)
    sids = sorted(groups); random.Random(a.seed).shuffle(sids)
    n_val = max(1, int(len(sids) * a.val_frac))
    val = [p for s in sids[:n_val] for p in sorted(groups[s])]
    val = random.Random(1).sample(val, min(a.val_max, len(val)))
    train = [p for s in sids[n_val:] for p in groups[s]]
    print(f"sessions {len(sids)} (val {n_val}); samples train {len(train)}, val {len(val)}", flush=True)

    t0 = time.time()
    base = open_loop(model, helper, proc, val)
    print(f"[before] open loop {json.dumps(base)} ({time.time() - t0:.0f}s)", flush=True)
    if a.unforced_eval:
        print(f"[before, unforced] open loop {json.dumps(open_loop(model, helper, proc, val, forced=False))}", flush=True)
    if a.eval_only:
        return

    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(TRAIN_PREFIXES)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params) / 1e9:.3f}B of {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B",
          flush=True)
    opt = SRAdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    ref = dict([(n, p.detach().clone()) for n, p in model.named_parameters()
                if p.requires_grad and n.endswith("proj.weight")][:12])
    steps = a.steps or int(a.epochs * len(train) / a.accum)
    rng = random.Random(a.seed + 1)
    order = []
    model.train(); model.vlm.eval()
    t0, run = time.time(), []
    for step in range(steps):
        for _ in range(a.accum):
            if not order:
                order = train[:]; rng.shuffle(order)
            s = load_sample(order.pop())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = teacher_forced_flow(model, build_inputs(helper, proc, s), n_noise=a.n_noise)
            loss = torch.nn.functional.mse_loss(out["v_pred"], out["v_target"]) / a.accum
            loss.backward()
            run.append(float(loss) * a.accum)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        lr = a.lr * min(1.0, (step + 1) / 20)                        # short warm-up
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 25 == 0 or step == 0:
            print(f"step {step + 1}/{steps} loss {np.mean(run[-25 * a.accum:]):.4f} lr {lr:.2e} "
                  f"{(time.time() - t0) / (step + 1):.2f}s/step peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                  flush=True)
    pn = dict(model.named_parameters())
    rel = [float((pn[n].float() - r.float()).norm() / max(r.float().norm(), 1e-12)) for n, r in ref.items()]
    print(f"weights moved: median relative change {np.median(rel):.2e} over {len(rel)} projection matrices "
          f"(0 would mean the updates were lost)", flush=True)
    sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in model.state_dict().items() if k.startswith(TRAIN_PREFIXES)}
    torch.save(sd, a.out)
    print(f"saved {len(sd)} tensors ({sum(v.numel() for v in sd.values()) / 1e9:.3f}B) to {a.out}", flush=True)
    after = open_loop(model, helper, proc, val)
    print(f"[after] open loop {json.dumps(after)}", flush=True)


if __name__ == "__main__":
    main()
