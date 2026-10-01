"""route_nav_text distance modes and route_choice margins on synthetic routes (numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from a15_driver.driver import route_nav_text  # noqa: E402
from a15_driver.route_choice import choose  # noqa: E402

bad = 0


def check(name, got, want):
    global bad
    ok = got == want
    bad += not ok
    print(f"{name}: {got!r} (want {want!r})" + ("" if ok else "   <-- FAIL"))


# 90-degree left turn 15 m ahead, route trimmed to start ~40 m along the path (so already turned)
left = np.array([[15.0, 10.0 + 4 * k] for k in range(10)])
check("left turn at 15 m, euclid", route_nav_text(left), "Turn left in 18m")
check("left turn at 15 m, intersect", route_nav_text(left, dist_mode="intersect"), "Turn left in 15m")
# diagonal left road (31 deg) whose back-extension meets the car axis at ~14.6 m (logged nuPlan route)
diag = np.array([[38.8, 14.8], [42.4, 17.0], [45.9, 19.2], [49.5, 21.5]])
check("diagonal left, euclid", route_nav_text(diag), "Turn left in 42m")
check("diagonal left, intersect", route_nav_text(diag, dist_mode="intersect"), "Turn left in 15m")
check("straight, intersect", route_nav_text(np.array([[40.0, 0.2], [80.0, 0.3]]), dist_mode="intersect"), None)
nan_route = np.vstack([diag, np.full((5, 2), np.nan)])
check("NaN padding ignored", route_nav_text(nan_route, dist_mode="intersect"), "Turn left in 15m")
# chooser margin: sample 1 is 1 m closer to a straight route than sample 0
t = np.linspace(0, 40, 64)[:, None]
s0, s1 = np.hstack([t, 0 * t + 1.0]), np.hstack([t, 0 * t])
route = np.array([[40.0, 0.0], [80.0, 0.0]])
check("margin 2.0 keeps sample 0", choose(np.stack([s0, s1]), route, margin_m=2.0), 0)
check("margin 0.5 switches", choose(np.stack([s0, s1]), route, margin_m=0.5), 1)
# speed guard: sample 1 follows the route better but is much shorter (slower)
slow = np.hstack([t * 0.5, 0 * t])
check("no guard picks the slower sample", choose(np.stack([s0, slow]), route, margin_m=0.5), 1)
check("guard 0.9 keeps sample 0", choose(np.stack([s0, slow]), route, margin_m=0.5, min_len_ratio=0.9), 0)
check("guard 0.9 allows an equally fast sample", choose(np.stack([s0, s1]), route, margin_m=0.5, min_len_ratio=0.9), 1)
print("PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(bad > 0)
