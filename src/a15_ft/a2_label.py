"""Label captured driving states with Alpamayo 2 Super, an offline teacher for the Alpamayo 1.5 expert.

Runs in the Alpamayo 2 Super environment (cluster/a2_label.sbatch), not the driver's. For every captured sample
(capture tars from a15_driver/capture.py with all six PAI cameras), A2 Super plans from the same frames and ego
history the driver saw; its trajectory (64 poses at 0.1 s in the t0 rig frame, same convention as the capture's
recorded future) is written as a sidecar npz: <out>/<tar name>/<session>/<t0>.npz with a2_xyz, a2_rot, cot.
The trainer (a15_ft/train.py --teacher_dir) then uses it as the target. A2 Super does not fit the 16 GiB driver
cap, so it is used only here, offline (Aman approved, 2026-10-23).
"""
import argparse
import glob
import io
import os
import tarfile
import time

import numpy as np
import torch
from PIL import Image

DRIVING_SLOTS = (0, 1, 2, 3, 5, 6)                  # A2 Super's driving profile (input_profiles.py)


def load_capture(z):
    shape = tuple(int(x) for x in z["frames_shape"])
    buf, lens = z["jpeg"].tobytes(), z["jpeg_len"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    imgs = [np.asarray(Image.open(io.BytesIO(buf[offs[i]:offs[i + 1]])).convert("RGB")) for i in range(len(lens))]
    frames = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).reshape(shape)       # (cams, 4, 3, H, W)
    return frames, [int(x) for x in z["slots"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir of capture tars")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", "nvidia/Alpamayo2-Super"))
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp") + "/a2lab")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    a = ap.parse_args()

    from alpamayo2_super import helper
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    jobs = []
    for tf in sorted(glob.glob(os.path.join(a.data, "*.tar"))):
        dst = os.path.join(a.scratch, os.path.basename(tf)[:-4])
        if not os.path.isdir(dst):
            with tarfile.open(tf) as z:
                z.extractall(dst)
        for p in sorted(glob.glob(os.path.join(dst, "**", "*.npz"), recursive=True)):
            jobs.append((os.path.basename(tf)[:-4], p))
    jobs = jobs[a.shard::a.nshards]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} samples in shard {a.shard}/{a.nshards}", flush=True)

    model = Alpamayo2Super.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    print(f"model loaded; reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)
    t0, n, skipped = time.time(), 0, 0
    for tar, p in jobs:
        rel = os.path.relpath(p, os.path.join(a.scratch, tar))
        out = os.path.join(a.out, tar, rel)
        if os.path.exists(out):
            continue
        z = np.load(p)
        frames, slots = load_capture(z)
        if not set(DRIVING_SLOTS) <= set(slots):
            skipped += 1                                 # 4-camera capture: A2 Super needs the rear cameras
            continue
        ip = [slots.index(s) for s in DRIVING_SLOTS]
        data = {"image_frames": frames[ip], "camera_indices": torch.tensor(DRIVING_SLOTS),
                "ego_history_xyz": torch.from_numpy(z["hist_xyz"]).float()[None, None],
                "ego_history_rot": torch.from_numpy(z["hist_rot"]).float()[None, None]}
        inp = helper.to_device(helper.prepare_model_inputs(data, model.config, model.tokenizer), "cuda")
        torch.cuda.manual_seed_all(42)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot, _, extra = model.sample_trajectories_from_data(
                data=inp, top_p=0.98, temperature=0.6, num_traj_samples=1,
                diffusion_kwargs={"inference_step": a.diffusion_steps}, return_extra=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, a2_xyz=xyz[0, 0, 0].float().cpu().numpy(), a2_rot=rot[0, 0, 0].float().cpu().numpy(),
                 cot=np.array(str(np.array(extra.get("cot", [""]), dtype=object).ravel()[0])))
        n += 1
        if n % 25 == 0 or n == 1:
            g = np.linalg.norm(xyz[0, 0, 0, :, :2].float().cpu().numpy() - z["fut_xyz"][:, :2], axis=1).mean()
            print(f"{n} labelled ({(time.time() - t0) / n:.1f} s each); last ADE vs recording {g:.2f} m; "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print(f"done: {n} labelled, {skipped} skipped (no rear cameras)", flush=True)


if __name__ == "__main__":
    main()
