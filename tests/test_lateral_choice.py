"""NVIDIA's CLOSEST_LATERAL selection (A15_CHOOSER=lateral): the driven plan re-expressed in the next rig frame, and
the choice it makes. Run with the candidate tree on PYTHONPATH (src_lat) and the driver's protos importable."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("A15_SRC", os.path.join(os.path.dirname(__file__), "..", "src_lat")))
from a15_driver.trajectory import make_cached_plan, plan_in_rig_frame  # noqa: E402
from alpasim_grpc.v0 import common_pb2  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{name}: {'ok' if ok else 'FAIL'} {detail}")


def pose(x, y, yaw, t_us):
    return common_pb2.PoseAtTime(timestamp_us=t_us, pose=common_pb2.Pose(
        vec=common_pb2.Vec3(x=x, y=y, z=0.0),
        quat=common_pb2.Quat(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))))


yaw0, t0 = math.radians(30), 1_000_000
offsets = np.stack([np.arange(1, 65) * 1.0, 0.02 * np.arange(1, 65) ** 1.5], axis=1)   # 10 m/s, drifting left
plan = make_cached_plan(created_time_us=t0, anchor_pose=pose(10.0, 5.0, yaw0, t0), trajectory_xy=offsets,
                        headings=np.zeros(64), source_frequency_hz=10.0)

same = plan_in_rig_frame(plan, pose(10.0, 5.0, yaw0, t0), t0, 64, 0.1)
check("same anchor and time returns the plan", same.shape == (64, 2) and np.allclose(same, offsets, atol=1e-9),
      f"max err {np.abs(same - offsets).max():.2e}")

# one second later the car is at the plan's 1 s point, heading unchanged: the remaining 54 points, shifted back 10 m
p10 = plan.positions_xy[10]
later = plan_in_rig_frame(plan, pose(p10[0], p10[1], yaw0, t0 + 1_000_000), t0 + 1_000_000, 64, 0.1)
want = offsets[10:] - offsets[9]
check("1 s later: 54 points overlap", later.shape == (54, 2), str(later.shape))
check("1 s later: shifted plan", np.allclose(later, want, atol=1e-6), f"max err {np.abs(later - want).max():.2e}")
check("no plan gives None", plan_in_rig_frame(None, pose(0, 0, 0, t0), t0, 64, 0.1) is None)

# the choice rule (as in policy.py): sample laterally nearest the driven plan over the overlap
cand = np.stack([want + [0, dy] for dy in (1.5, -0.2, 0.1, -3.0)])          # (K, 54, 2)
cand = np.concatenate([cand, np.repeat(cand[:, -1:], 10, axis=1)], axis=1)   # 64 points per candidate
n = min(len(later), cand.shape[1])
i = int(np.abs(cand[:, :n, 1] - later[None, :n, 1]).mean(axis=1).argmin())
check("picks the candidate 0.1 m off, not sample 0 at 1.5 m", i == 2, f"chose {i}")

print("FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
