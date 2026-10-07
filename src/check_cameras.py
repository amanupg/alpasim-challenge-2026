"""GPU memory and latency of the policy for 4 vs 6 PAI cameras (synthetic frames, no simulator).

Usage: python check_cameras.py
Loads Alpamayo15Policy exactly as the driver does (int8, CPU embedding, K=6 route chooser), then runs
predict() on random frames at the driver's 576x320 size for the 4-camera slots [0,1,2,6] and the
6-camera slots [0,1,2,3,5,6] (adds rear-left / rear-right), and reports the process's nvidia-smi
memory, torch peak reserved, and latency. The 16 GiB contract is per driver process.
"""
import os
import subprocess
import time

import numpy as np
import torch

from a15_driver.policy import Alpamayo15Policy


def smi_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return max([int(l.split(",")[1]) for l in out.strip().splitlines() if l.split(",")[0].strip() == str(os.getpid())] or [0])


def main():
    hub = os.environ["HUB"]
    snap = lambda r: sorted(os.listdir(f"{hub}/models--{r}/snapshots"))[0]
    pol = Alpamayo15Policy(
        model_dir=f"{hub}/models--nvidia--Alpamayo-1.5-10B/snapshots/{snap('nvidia--Alpamayo-1.5-10B')}",
        vlm_dir=f"{hub}/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/{snap('Qwen--Qwen3-VL-8B-Instruct')}",
        processor_dir=f"{hub}/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/{snap('Qwen--Qwen3-VL-2B-Instruct')}",
        num_samples=6, chooser="route", route_margin=1.0, route_min_len=0.9)
    hist_xyz = torch.zeros(1, 1, 16, 3); hist_xyz[..., 0] = torch.arange(-15, 1).float() * 0.8
    hist_rot = torch.eye(3).expand(1, 1, 16, 3, 3).clone()
    route = np.array([[40.0 + 5 * k, 0.0] for k in range(9)])
    for name, slots in [("4 cameras", [0, 1, 2, 6]), ("6 cameras", [0, 1, 2, 3, 5, 6])]:
        torch.cuda.reset_peak_memory_stats()
        lat = []
        for i in range(6):
            frames = torch.randint(0, 255, (len(slots), 4, 3, 320, 576), dtype=torch.uint8)
            t = time.time()
            pol.predict(frames, slots, hist_xyz, hist_rot, route_xy=route)
            torch.cuda.synchronize(); lat.append(time.time() - t)
        print(f"{name} slots {slots}: nvidia-smi {smi_mib()} MiB, peak reserved "
              f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB, latency median {np.median(lat[1:]):.2f} s "
              f"(16 GiB = 16384 MiB)", flush=True)


if __name__ == "__main__":
    main()
