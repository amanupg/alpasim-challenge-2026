"""AlpaSim feasibility: can Alpamayo 1.5 meet the challenge driver contract?

Contract (challenge README): <= 16 GiB VRAM per driver replica; 16 replicas share 4 H100s
(4 per GPU, 2 rollouts each); target <= 0.1 s model work per Drive call at 10 Hz, enforced
by a per-track throughput budget. Plans may be cached between calls (VAVAM replans at 2 Hz),
so one plan every 0.5 s leaves roughly <= 0.5 s per plan before GPU sharing.

Per configuration (weight quantization x reasoning mode x camera count) this measures
  * GPU memory: CUDA context + peak reserved during inference (what the replica would hold)
  * latency per plan, split into VLM generate (prefill + reasoning decode) and diffusion
  * generated tokens, and whether <traj_future_start> was produced
  * trajectory error (ADE/FDE, xy, 6.4 s) vs the logged future on nuReasoning validation
    key frames, as a quality proxy for damage from quantization / skipped reasoning.
Inputs come from nureasoning/src/a15ft_data.build_sample (borrowed, read-only; see
alpasim/BORROWED.md). Output: one JSON summary + per-clip npz per configuration.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

NR = "/insomnia001/depts/edu/users/au2327/nureasoning"
sys.path[:0] = [os.path.join(NR, "src"), NR, os.path.join(NR, "recon", "devkit")]
from a15_infer import load_model          # noqa: E402  (borrowed: Qwen3-VL config swap + clean-load checks)
from a15ft_data import build_sample       # noqa: E402  (borrowed: same inputs as zero-shot/fine-tune)
from alpamayo1_5 import helper            # noqa: E402

TELE = 6    # Alpamayo slot "Front telephoto camera"; nuReasoning has none, the front image stands in


def quantize(model, q):
    if q == "bf16":
        return 0
    from torchao.quantization import (quantize_, Int8WeightOnlyConfig, Float8WeightOnlyConfig,
                                      Int4WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig)
    cfg = {"int8wo": Int8WeightOnlyConfig(), "fp8wo": Float8WeightOnlyConfig(),
           "fp8dq": Float8DynamicActivationFloat8WeightConfig(),
           "int4wo": Int4WeightOnlyConfig(group_size=128)}[q]
    hit = []

    def keep(m, fqn):
        # language model + action expert linears; vision tower, embeddings, lm_head and the
        # small action in/out projections stay bf16
        ok = (isinstance(m, torch.nn.Linear) and m.in_features % 128 == 0 and m.out_features % 128 == 0
              and not any(k in fqn for k in ("visual", "action_in", "action_out", "diffusion", "lm_head", "embed")))
        if ok:
            hit.append(m.in_features * m.out_features)
        return ok
    quantize_(model, cfg, filter_fn=keep)
    return sum(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="bf16", choices=["bf16", "int8wo", "fp8wo", "fp8dq", "int4wo"])
    ap.add_argument("--mode", default="cot", choices=["cot", "nocot"])
    ap.add_argument("--cams", type=int, default=4, choices=[3, 4])
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--graph", action="store_true", help="CUDA graphs for the diffusion expert")
    ap.add_argument("--out", default="/insomnia001/depts/edu/users/au2327/alpasim/runs/a15_fit")
    a = ap.parse_args()
    tag = f"{a.quant}_{a.mode}_c{a.cams}" + ("_graph" if a.graph else "")
    os.makedirs(a.out, exist_ok=True)

    torch.cuda.init()
    free0, total = torch.cuda.mem_get_info()
    torch.zeros(1, device="cuda")
    ctx = free0 - torch.cuda.mem_get_info()[0]            # CUDA context of this process
    model = load_model("nvidia/Alpamayo-1.5-10B", "Qwen/Qwen3-VL-8B-Instruct")
    t = time.time(); qparams = quantize(model, a.quant); tq = time.time() - t
    if a.graph:
        model.enable_diffusion_expert_cuda_graph(max_batch_size=1)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_reserved()
    proc = helper.get_processor(model.tokenizer)

    # phase timers: wrap VLM generate and diffusion sampling
    T = {}
    g0, d0 = model.vlm.generate, model.diffusion.sample

    def gen(*x, **k):
        torch.cuda.synchronize(); s = time.time(); o = g0(*x, **k); torch.cuda.synchronize()
        T["gen"] = time.time() - s; T["new"] = o.sequences.shape[1] - k["input_ids"].shape[1]
        T["seq"] = o.sequences
        return o

    def dif(*x, **k):
        torch.cuda.synchronize(); s = time.time(); o = d0(*x, **k); torch.cuda.synchronize()
        T["dif"] = time.time() - s
        return o
    model.vlm.generate, model.diffusion.sample = gen, dif
    eos = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")

    clips = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{NR}/img_a15/validation/*/_done.json"))
    rows, preds = [], {}
    for c in clips:
        if len(rows) >= a.clips + 1:
            break
        s = build_sample(os.path.join(NR, "cache", "validation", c), [os.path.join(NR, "img_a15", "validation")], 100)
        if s is None:
            continue
        fr, ci = s["image_frames"], s["camera_indices"]
        if a.cams == 4:
            fr = torch.cat([fr, fr[1:2]]); ci = torch.cat([ci, torch.tensor([TELE])])
        msgs = helper.create_message(fr.flatten(0, 1), camera_indices=ci)
        if a.mode == "nocot":       # close the reasoning block at once; the model emits <traj_future_start>
            msgs[-1]["content"][0]["text"] = "<|cot_start|><|cot_end|>"
        tok = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                       continue_final_message=True, return_dict=True, return_tensors="pt")
        inp = helper.to_device({"tokenized_data": tok, "ego_history_xyz": s["ego_history_xyz"],
                                "ego_history_rot": s["ego_history_rot"]}, "cuda")
        torch.cuda.manual_seed_all(42)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=inp, top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=256 if a.mode == "cot" else 4, return_extra=True)
        torch.cuda.synchronize(); dt = time.time() - t0
        p = xyz[0, 0, 0, :, :2].float().cpu().numpy(); g = s["ego_future_xyz"][0, 0, :, :2].numpy()
        d = np.linalg.norm(p - g, axis=1)
        new = T["seq"][0, -T["new"]:] if T["new"] else T["seq"][0, :0]
        rows.append({"clip": c, "lat": dt, "gen": T["gen"], "dif": T["dif"], "new_tokens": int(T["new"]),
                     "eos": bool((new == eos).any()), "prompt_tokens": int(tok["input_ids"].shape[1]),
                     "ade": float(d.mean()), "fde": float(d[-1]), "cot": str(np.array(extra["cot"]).ravel()[0])[:160]})
        preds[c] = p
        if len(rows) <= 2:
            print(f"[{tag}] {c} lat {dt:.2f}s gen {T['gen']:.2f}s dif {T['dif']:.3f}s new {T['new']} "
                  f"eos {rows[-1]['eos']} ade {d.mean():.2f} cot {rows[-1]['cot']!r}", flush=True)
    R = rows[1:]                                           # first call = warm-up
    peak = torch.cuda.max_memory_reserved()
    summ = {"tag": tag, "gpu": torch.cuda.get_device_name(), "n": len(R), "quantized_params": qparams,
            "quant_s": round(tq, 1), "ctx_GiB": ctx / 2**30, "resident_GiB": resident / 2**30,
            "peak_reserved_GiB": peak / 2**30, "est_process_GiB": (ctx + peak) / 2**30,
            "lat_mean": float(np.mean([r["lat"] for r in R])), "lat_p90": float(np.percentile([r["lat"] for r in R], 90)),
            "gen_mean": float(np.mean([r["gen"] for r in R])), "dif_mean": float(np.mean([r["dif"] for r in R])),
            "new_tokens_mean": float(np.mean([r["new_tokens"] for r in R])),
            "prompt_tokens": int(np.mean([r["prompt_tokens"] for r in R])),
            "eos_rate": float(np.mean([r["eos"] for r in R])),
            "ade": float(np.mean([r["ade"] for r in R])), "fde": float(np.mean([r["fde"] for r in R]))}
    print(json.dumps(summ, indent=1), flush=True)
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, f"{tag}.npz"), **preds)


if __name__ == "__main__":
    main()
