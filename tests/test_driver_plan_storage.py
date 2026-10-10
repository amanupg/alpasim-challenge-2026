"""Regression test: every inference must store its plan, whatever the logging / chooser flags.

2026-09-30: a logging edit nested `s.cached_plan = new` inside `if log_choices and costs`, so runs
without chooser costs silently drove on the straight fallback. This drives the real
A15ChallengeDriver._maybe_infer with a fake policy (no model, CPU only) under every flag combination.
Run: python tests/test_driver_plan_storage.py   (needs torch, grpc, alpasim_grpc on the path)
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from a15_driver import driver as D  # noqa: E402


class FakePolicy:
    output_frequency_hz = 10.0

    def __init__(self, with_costs):
        self.with_costs, self.calls = with_costs, 0

    def predict(self, frames, cam_slots, hx, hr, nav_text=None, route_xy=None):
        self.calls += 1
        t = np.arange(1, 65) * 0.1
        info = {"choice": 0, "costs": [1.0, 2.0] if self.with_costs else None,
                "ends": [[6.4, 0.0], [6.0, 1.0]] if self.with_costs else None}
        return np.stack([5.0 * t, 0 * t], 1), np.zeros(64), info


class FakeHandle:
    def __init__(self, p):
        self.p = p

    def get(self):
        return self.p


def run(log_choices, with_costs, early, nav, route_snap=0.0, route=None, snap_deg=0.0):
    pol = FakePolicy(with_costs)
    drv = D.A15ChallengeDriver(handle=FakeHandle(pol), pai_cameras=[], interval_us=1_000_000,
                               frame_dt_s=0.1, nav=nav, log_choices=log_choices, early_replan=early,
                               route_snap=route_snap, route_snap_max_deg=snap_deg)
    s = D.SessionState(cams=["CAM_F0"], slots=[1], sid="test-session")
    s.frames = {"CAM_F0": D.deque(maxlen=24)}
    s.route_xy = np.array([[40.0, 0.0], [80.0, 0.0]]) if route is None else route
    q = np.array([1.0, 0.0, 0.0, 0.0])
    stored = []
    for k in range(10):                                   # nuPlan-like: 2 Hz steps, 5 s
        now = 17_000 + k * 500_000
        s.frames["CAM_F0"].append((now, torch.zeros(3, 32, 32, dtype=torch.uint8)))
        s.poses.append((now - 20_000, np.array([5.0 * (now - 20_000) / 1e6, 0, 0]), q))
        s.poses.append((now, np.array([5.0 * now / 1e6, 0, 0]), q))
        drv._maybe_infer(s, now)
        stored.append(s.cached_plan is not None)
    return pol.calls, stored, s


def main():
    bad = 0
    for log_choices, with_costs, early, nav in itertools.product([False, True], repeat=4):
        calls, stored, s = run(log_choices, with_costs, early, nav)
        ok = calls > 0 and all(stored) and s.cached_plan is not None
        want = 5 if not early else 7                      # 1 Hz; early adds replans at frames 2-4
        ok_calls = calls == want
        print(f"log={log_choices:d} costs={with_costs:d} early={early:d} nav={nav:d}: "
              f"inferences {calls} (want {want}), plan stored every step {all(stored)}"
              + ("" if ok and ok_calls else "   <-- FAIL"))
        bad += not (ok and ok_calls)
    # route snap: route one lane left (3.5 m) within the 32 m plan's reach -> every plan is snapped and stored
    # (parallel route: guard 15 deg still snaps; a route turning 90 deg ahead is not snapped with the guard)
    turn = np.array([[20.0, 3.5], [20.0, 23.5], [20.0, 43.5]])
    lane = np.array([[20.0, 3.5], [40.0, 3.5], [80.0, 3.5]])
    for snap, deg, route, want_snaps in [(2.0, 0.0, lane, 5), (0.0, 0.0, lane, 0), (2.0, 15.0, lane, 5), (2.0, 15.0, turn, 0)]:
        calls, stored, s = run(True, True, False, False, route_snap=snap, route=route, snap_deg=deg)
        ok = calls == 5 and all(stored) and s.stats["snapped"] == want_snaps
        print(f"route_snap={snap} max_deg={deg}: inferences {calls}, snapped {s.stats['snapped']} (want {want_snaps}), "
              f"plan stored every step {all(stored)}" + ("" if ok else "   <-- FAIL"))
        bad += not ok
    print("PASS" if bad == 0 else f"FAIL ({bad} combinations)")
    sys.exit(bad > 0)


if __name__ == "__main__":
    main()
