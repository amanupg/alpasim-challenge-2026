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


def _corners(xyz, rot, half_len=2.4, half_wid=1.0):
    """Footprint corners (T, 4, 2); same as a15_ft.train._corners (the trainer's RoaD distance)."""
    yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    c, s_ = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    dx = np.array([half_len, half_len, -half_len, -half_len])[None]
    dy = np.array([half_wid, -half_wid, half_wid, -half_wid])[None]
    return np.stack([xyz[:, :1] + c * dx - s_ * dy, xyz[:, 1:2] + s_ * dx + c * dy], -1)


def policy_best_dist(z, window=20):
    """Distance of the policy's best own sample to the recording (train.road_target's measure); inf without samples."""
    if "samp_xyz" not in z.files:
        return float("inf")
    gt = _corners(z["fut_xyz"][:window], z["fut_rot"][:window])
    return min(float(np.linalg.norm(_corners(x[:window], r[:window]) - gt, axis=-1).mean())
               for x, r in zip(z["samp_xyz"], z["samp_rot"]))


def prepare_inputs(helper, data, model, with_nav):
    """helper.prepare_model_inputs, with the route instruction as an extra prompt component when given. The component
    order is the one NVIDIA's navigation demo uses (examples/two_gpu_nav_cfg_demo.py): image, history, nav, prompt."""
    if not with_nav:
        return helper.prepare_model_inputs(data, model.config, model.tokenizer)
    from alpamayo2_super.chat_template.conversation import build_conversation
    cfg = model.config
    messages = build_conversation(
        data=data, num_tokens_per_history_traj=cfg.tokens_per_history_traj,
        num_tokens_per_future_traj=cfg.tokens_per_future_traj,
        components_order=["image", "traj_history", "nav_instruction", "prompt"], components_prompt=["cot", "traj_future"],
        generation_mode=True, include_camera_ids=cfg.include_camera_ids, camera_ids=data["camera_indices"],
        include_frame_nums=cfg.frame_label == "frame_num")
    if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]
    processor = helper.get_processor(model.tokenizer, cfg)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=False)
    images = data["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    tok = dict(processor(text=text, images=images, videos=None, padding=False, return_tensors="pt", do_rescale=False))
    assert tok["input_ids"].shape[0] == 1
    return {"tokenized_data": tok, "ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]}


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
    ap.add_argument("--k", type=int, default=1, help="teacher samples per state; all are saved (a2_samp_*) and the "
                    "trainer keeps the one nearest the recording over 2 s (RoaD rule, train.road_target)")
    ap.add_argument("--nav", action="store_true", help="give the teacher the capture's route instruction (sample 'nav')")
    ap.add_argument("--hard_first", action="store_true", help="label states whose own policy samples (samp_xyz) are "
                    "farthest from the recording first; the teacher can only change the RoaD target there")
    ap.add_argument("--min_policy_dist", type=float, default=0.0, help="with --hard_first: skip states whose best "
                    "policy sample is already this close to the recording (m, footprint corners over 2 s)")
    ap.add_argument("--reverse", action="store_true", help="label in reverse order (a second job then meets the "
                    "first in the middle; both skip labels that already exist)")
    ap.add_argument("--nav_override", default="", help="smoke test only: this route instruction for every sample")
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
    if a.hard_first:
        d = np.array([policy_best_dist(np.load(p)) for _, p in jobs])
        order = np.argsort(-d)
        n_all = len(jobs)
        jobs = [jobs[i] for i in order if d[i] >= a.min_policy_dist]
        print(f"hard-first: policy best distance p50/p90/max {np.percentile(d, 50):.2f}/{np.percentile(d, 90):.2f}/"
              f"{d.max():.2f} m; {len(jobs)} of {n_all} states >= {a.min_policy_dist} m", flush=True)
    jobs = jobs[a.shard::a.nshards]
    if a.reverse:
        jobs = jobs[::-1]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} samples in shard {a.shard}/{a.nshards}", flush=True)

    model = Alpamayo2Super.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    print(f"model loaded; reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)
    t0, n, skipped, n_nav = time.time(), 0, 0, 0
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
        nav = a.nav_override or (str(z["nav"]) if (a.nav and "nav" in z.files) else "")
        if nav:
            data["nav_text"] = [nav]
            n_nav += 1
        inp = helper.to_device(prepare_inputs(helper, data, model, with_nav=bool(nav)), "cuda")
        torch.cuda.manual_seed_all(42)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot, _, extra = model.sample_trajectories_from_data(
                data=inp, top_p=0.98, temperature=0.6, num_traj_samples=a.k,
                diffusion_kwargs={"inference_step": a.diffusion_steps}, return_extra=True)
        sx, sr = xyz[0, 0].float().cpu().numpy(), rot[0, 0].float().cpu().numpy()      # (K, 64, 3), (K, 64, 3, 3)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        extra_k = {"a2_samp_xyz": sx, "a2_samp_rot": sr} if a.k > 1 else {}
        tmp = out[:-4] + f".tmp{os.getpid()}.npz"        # atomic: a concurrent labeller or the trainer never sees a partial file
        np.savez(tmp, a2_xyz=sx[0], a2_rot=sr[0], nav=np.array(nav), **extra_k,
                 cot=np.array(str(np.array(extra.get("cot", [""]), dtype=object).ravel()[0])))
        os.replace(tmp, out)
        n += 1
        if n % 25 == 0 or n == 1:
            g = np.linalg.norm(sx[:, :, :2] - z["fut_xyz"][None, :, :2], axis=2).mean(1)
            print(f"{n} labelled ({(time.time() - t0) / n:.1f} s each, {n_nav} with route text); last ADE vs "
                  f"recording per sample {np.round(g, 2).tolist()} m; "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print(f"done: {n} labelled ({n_nav} with route text), {skipped} skipped (no rear cameras)", flush=True)


if __name__ == "__main__":
    main()
