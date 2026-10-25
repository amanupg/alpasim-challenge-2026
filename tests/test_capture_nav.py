"""Captured samples carry the route instruction the driver would give the model (for --nav fine-tuning)."""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from a15_driver.capture import Capturer, load_sample  # noqa: E402
from a15_driver.driver import route_nav_text  # noqa: E402

fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    fails += not ok
    print(f"{name}: {got!r} (want {want!r})")


left_turn = np.array([[15.0, 2.0 + 8 * k] for k in range(10)])          # road turning left ~15 m ahead
straight = np.array([[40.0 + 8 * k, 0.0] for k in range(10)])
nav_turn = route_nav_text(left_turn, dist_mode="intersect")
check("turn route gives a left-turn instruction", (nav_turn or "").startswith("Turn left in"), True)
check("straight route gives none", route_nav_text(straight, dist_mode="intersect"), None)

with tempfile.TemporaryDirectory() as d:
    cap = Capturer(d, every_us=500_000)
    fr = torch.zeros((4, 4, 3, 32, 48), dtype=torch.uint8)
    hx, hr = torch.zeros((1, 1, 16, 3)), torch.eye(3).expand(1, 1, 16, 3, 3).clone()
    fut = (np.zeros((64, 3), np.float32), np.tile(np.eye(3, dtype=np.float32), (64, 1, 1)))
    cap.save("s1", 1000, fr, [0, 1, 2, 6], hx, hr, *fut, nav=np.array(nav_turn))
    cap.save("s1", 2000, fr, [0, 1, 2, 6], hx, hr, *fut, nav=np.array(""))
    cap.save("s1", 3000, fr, [0, 1, 2, 6], hx, hr, *fut)                    # old captures: no nav field
    got = [load_sample(os.path.join(d, "s1", f"{t}.npz")).get("nav") for t in (1000, 2000, 3000)]
    check("round trip: turn / empty / absent", got, [nav_turn, None, None])

print("FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
