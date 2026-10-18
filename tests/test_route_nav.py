"""route_nav_text distance modes and route_choice margins on synthetic routes (numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from a15_driver.driver import route_nav_text, speed_scale_plan  # noqa: E402
from a15_driver.route_choice import (RouteMemory, choose, retime_path, route_deviation,  # noqa: E402
                                     route_lateral_offset, route_parallel, route_path, route_pull)

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
# retime: a left-curving path at 10 m/s driven with a 5 m/s profile covers half the distance, same shape
tt64 = np.arange(1, 65) * 0.1
ang = tt64 * 0.1
curve = np.stack([np.sin(ang) * 100, (1 - np.cos(ang)) * 100], 1)          # radius 100 m, 10 m/s
slow_line = np.stack([5.0 * tt64, 0 * tt64], 1)
rt, ryaw = retime_path(curve, slow_line)
check("retime: covers 5 m/s * 6.4 s = 32 m", round(float(np.linalg.norm(np.diff(np.vstack([[0, 0], rt]), axis=0), axis=1).sum()), 1), 32.0)
check("retime: stays on the curve (radius 100)", round(float(np.abs(np.linalg.norm(rt - [0, 100], axis=1) - 100).max()), 2), 0.0)
fast_line = np.stack([20.0 * tt64, 0 * tt64], 1)
rt2, _ = retime_path(curve, fast_line)
check("retime: beyond the path end it extends straight", round(float(np.linalg.norm(rt2[-1] - curve[-1])), 1), 64.0)
stop = np.zeros((64, 2))
rt3, ry3 = retime_path(curve, stop)
check("retime: stopped profile stays at the car", float(np.abs(rt3).max()), 0.0)
# route snap: deviation is measured from the route points the plan reaches
straight64 = np.stack([10.0 * tt64, 0 * tt64], 1)                         # 64 m straight plan
check("deviation: plan on the route", route_deviation(straight64, np.array([[40.0, 0.0], [80.0, 0.0]])), 0.0)
check("deviation: route one lane left", round(route_deviation(straight64, lane_route), 2), 3.5)
check("deviation: plan too short to reach the route", route_deviation(straight64[:30], lane_route), None)
check("deviation: no route", route_deviation(straight64, None), None)
check("deviation: NaN-padded route ok", round(route_deviation(straight64, np.vstack([lane_route, np.full((5, 2), np.nan)])), 2), 3.5)
rp = route_path(lane_route)
check("route_path: starts at the car heading +x (first point ~1 m ahead, ~0 lateral)",
      (round(float(rp[0, 0]), 1), bool(abs(rp[0, 1]) < 0.01)), (1.0, True))
check("route_path: reaches the first waypoint exactly", bool(np.isclose(rp, lane_route[0]).all(1).any()), True)
check("route_path: then follows the route", np.allclose(rp[-len(lane_route) + 1:], lane_route[1:]), True)
check("route_path: monotone lateral move (no overshoot)", bool((np.diff(rp[:40, 1]) >= -1e-9).all() and rp[:40, 1].max() <= 3.5 + 1e-6), True)
snap_xy, snap_yaw = retime_path(rp, straight64)
check("snap: model speed kept (64 m along the new path)", round(float(np.linalg.norm(np.diff(np.vstack([[0, 0], snap_xy]), axis=0), axis=1).sum()), 1), 64.0)
check("snap: ends in the route's lane", round(float(snap_xy[-1, 1]), 1), 3.5)
check("snap: deviation after snapping ~0", round(route_deviation(snap_xy, lane_route), 2), 0.0)
check("route_path: no route", route_path(None), None)
check("parallel: lane-change route (3.5 m left at 40 m, straight) is parallel", route_parallel(lane_route, 15), True)
check("parallel: turned road (route at x=15 heading +y) is not", route_parallel(turn_route, 15), False)
check("parallel: route bending 30 deg at its start is not", route_parallel(np.array([[40.0, 0.0], [48.7, 5.0]]), 15), False)
check("parallel: no route", route_parallel(None, 15), False)
check("lateral offset: lane-change route", route_lateral_offset(lane_route), 3.5)
check("lateral offset: no route", route_lateral_offset(None), 0.0)
# route memory: routes arrive 40-80 m ahead in the rig frame while the car drives a straight then a 90 deg left
# turn (radius 15 m); 10 m before the turn the memory must give the turn itself, which no single route shows
def _rot(yaw):
    c, s_ = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
tp = [np.array([x, 0.0]) for x in np.arange(0, 60, 1.0)]
tp += [np.array([60 + 15 * np.sin(a), 15 - 15 * np.cos(a)]) for a in np.linspace(0, np.pi / 2, 30)[1:]]
tp += [np.array([75.0, y]) for y in np.arange(16, 120, 1.0)]
tp = np.array(tp); ts_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tp, axis=0), axis=1))]
mem = RouteMemory()
for tt in np.arange(0, 5.0, 0.1):
    ic = int(np.searchsorted(ts_, 10 * tt)); pc = tp[ic]; dd = tp[ic + 1] - tp[ic]; R = _rot(np.arctan2(dd[1], dd[0]))
    sel = tp[(ts_ >= 10 * tt + 40) & (ts_ <= 10 * tt + 80)][::2]
    mem.add((np.c_[sel - pc, np.zeros(len(sel))] @ R)[:, :2], np.r_[pc, 0], R)
ic = int(np.searchsorted(ts_, 50)); pc = tp[ic]
mp = mem.path_in_frame(np.r_[pc, 0], _rot(0.0))
on_path = max(float(np.min(np.linalg.norm(tp - pc - q, axis=1))) for q in mp)
apex = np.array([60 + 15 * np.sin(np.pi / 4), 15 - 15 * np.cos(np.pi / 4)]) - pc
check("memory: path ahead lies on the recorded path (incl. the turn)", round(on_path, 2), 0.0)
check("memory: it goes through the turn (passes the apex within 1 m)", bool(np.min(np.linalg.norm(mp - apex, axis=1)) < 1.0), True)
check("memory: a single route at this point starts past the turn", bool(np.min(np.linalg.norm(tp[(ts_ >= 90) & (ts_ <= 130)] - pc - apex, axis=1)) > 5), True)
mo = mem.path_in_frame(np.r_[pc + [0, -3.5], 0], _rot(0.0), join_m=25.0)
check("memory: car one lane right of the path, join starts at the car", bool(np.linalg.norm(mo[0] - [1.0, 0.0]) < 0.5), True)
check("memory: back on the path within the join distance", round(min(float(np.min(np.linalg.norm(tp - pc - [0, -3.5] - q, axis=1))) for q in mo[np.linalg.norm(mo, axis=1) > 27][:3]), 2), 0.0)
check("memory: empty memory gives no path", RouteMemory().path_in_frame(np.zeros(3), np.eye(3)), None)
check("memory: path that does not reach the car (session start) gives None", mem.path_in_frame(np.r_[tp[0] - [30, 0], 0], _rot(0.0)), None)
print("PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(bad > 0)
