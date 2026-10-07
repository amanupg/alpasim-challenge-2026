"""route_nav_text distance modes and route_choice margins on synthetic routes (numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from a15_driver.driver import route_nav_text, speed_scale_plan  # noqa: E402
from a15_driver.route_choice import choose, route_pull  # noqa: E402

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
# graded guard: a slower sample (0.85x) that is a lane (3.5 m) closer to the route
lane = np.hstack([t * 0.85, 0 * t])  # 34 m vs 43.5 m (s0far starts 3.5 m to the side): ratio 0.78
s0far = np.hstack([t, 0 * t + 3.5])
check("guard 0.9 blocks slower lane-change sample", choose(np.stack([s0far, lane]), route, 1.0, 0.9), 0)
check("graded guard (gap 3 m, 0.75) allows it", choose(np.stack([s0far, lane]), route, 1.0, 0.9, 3.0, 0.75), 1)
check("graded guard still blocks a much slower one", choose(np.stack([s0far, np.hstack([t * 0.5, 0 * t])]), route, 1.0, 0.9, 3.0, 0.75), 0)
# speed scale: 5 m/s straight plan at 10 Hz; scale 1.2 -> 6 m/s, horizon 6.4/1.2 s
tt = np.arange(64) * 0.1
pxy, pyaw = np.stack([5 * tt, 0 * tt], 1), np.zeros(64)
q, qy = speed_scale_plan(pxy, pyaw, 1.2)
check("scale 1.2 speed m/s", round(float(q[1, 0] - q[0, 0]) / 0.1, 2), 6.0)
check("scale 1.2 points", len(q), 53)
check("scale 1.0 unchanged", speed_scale_plan(pxy, pyaw, 1.0)[0] is pxy, True)
# route pull: straight 64-point plan at 10 m/s (0..63 m) in our lane; route one lane left (y=3.5) 40-80 m
pl = np.stack([np.arange(64) * 1.0, np.zeros(64)], 1)
ry = np.zeros(64)
lane_route = np.array([[40.0 + 5 * k, 3.5] for k in range(9)])
q, qy = route_pull(pl, ry, lane_route, 0.6)
check("pull: near points untouched", float(np.abs(q[:31] - pl[:31]).max()), 0.0)
check("pull: point at 55 m moves 0.6 of 3.5 m", round(float(q[55, 1]), 2), 2.1)
check("pull: point at 37.5 m moves half of 0.6", round(float(q[37, 1] + q[38, 1]) / 2, 2), round(0.6 * 3.5 * (37.5 - 30) / 15, 2))
check("pull: headings turn left where moved", bool(qy[35] > 0), True)
check("pull: no route = unchanged", route_pull(pl, ry, None, 0.6)[0] is pl, True)
short = pl[:25]
check("pull: short plan (<30 m) unchanged", float(np.abs(route_pull(short, ry[:25], lane_route, 0.6)[0] - short).max()), 0.0)
# turn route: points that would project onto the car-to-route chord are pulled to the route itself, not the chord
turn_route = np.array([[15.0, 10.0 + 4 * k] for k in range(10)])
q2, _ = route_pull(pl, ry, turn_route, 1.0)
check("pull: far point lands on the turned road (x=15)", round(float(q2[60, 0]), 1), 15.0)
print("PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(bad > 0)
