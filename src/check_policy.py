"""Equivalence + memory check for the container policy (src/a15_driver/policy.py).

Runs the SAME nuReasoning validation key frames (seed 42) through the policy with the token
embedding on GPU and on CPU, and reports max trajectory difference, latency, and the
process's GPU memory as nvidia-smi sees it (what a 16 GiB cap would count).
Inputs are built by nureasoning/src/a15ft_data.build_sample (borrowed read-only).
"""
import glob, os, subprocess, sys, time
import numpy as np
import torch

NR = "/insomnia001/depts/edu/users/au2327/nureasoning"
sys.path[:0] = [os.path.join(NR, "src"), NR, os.path.join(NR, "recon", "devkit")]
from a15ft_data import build_sample                      # noqa: E402
from a15_driver.policy import Alpamayo15Policy            # noqa: E402

HUB = f"{NR}/.cache/hf/hub"
snap = lambda r: glob.glob(f"{HUB}/models--{r}/snapshots/*/")[0]


def smi_mib():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return max([int(l.split(",")[1]) for l in out.strip().splitlines() if l.split(",")[0].strip() == str(os.getpid())] or [0])


def run(opts, n=12):
    p = Alpamayo15Policy(model_dir=snap("nvidia--Alpamayo-1.5-10B"), vlm_dir=snap("Qwen--Qwen3-VL-8B-Instruct"),
                         processor_dir=snap("Qwen--Qwen3-VL-2B-Instruct"), **opts)
    clips = sorted(os.path.basename(os.path.dirname(x)) for x in glob.glob(f"{NR}/img_a15/validation/*/_done.json"))
    outs, lat = [], []
    for c in clips:
        if len(outs) >= n:
            break
        s = build_sample(f"{NR}/cache/validation/{c}", [f"{NR}/img_a15/validation/{c}"], 100)
        if s is None:
            continue
        fr = torch.cat([s["image_frames"], s["image_frames"][1:2]])          # 4 slots: front stands in for tele
        t = time.time()
        xy, yaw, _ = p.predict(fr, [0, 1, 2, 6], s["ego_history_xyz"], s["ego_history_rot"])
        torch.cuda.synchronize(); lat.append(time.time() - t); outs.append(xy)
    mem = smi_mib()
    print(f"{opts}: nvidia-smi {mem} MiB, peak reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB, "
          f"latency {np.mean(lat[1:]):.3f}s", flush=True)
    del p; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return np.stack(outs), mem


if __name__ == "__main__":
    # one configuration per process: a second model in the same process inherits the first
    # one's allocations and makes the memory numbers meaningless
    # argv[1] e.g. "embed_cpu=1,cuda_graph=0"
    def _val(v):
        if v in ("0", "1"):
            return v == "1"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    opts = {k: _val(v) for k, v in (kv.split("=") for kv in sys.argv[1].split(","))}
    out = sys.argv[2]
    xy, mem = run(opts)
    np.savez(out, xy=xy, mem=mem)
    if len(sys.argv) > 3:
        other = np.load(sys.argv[3])
        print(f"max |traj diff| vs {sys.argv[3]}: {np.abs(other['xy'] - xy).max():.6f} m over {len(xy)} clips; "
              f"nvidia-smi {int(other['mem'])} -> {mem} MiB")
