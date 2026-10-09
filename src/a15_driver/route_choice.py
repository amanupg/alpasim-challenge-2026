"""Route-consistent choice among K trajectory samples (numpy only, so it unit-tests without torch).

nuPlan bench (2026-09-29): 15 of the 24 lost scene points were the car taking the wrong branch at a
junction (straight where the route turns, or the reverse), ending as corridor exits or offroad.
The runtime sends route waypoints about 40-80 m ahead in the rig frame; the plan is 6.4 s long.
Sample 0 is exactly the K=1 plan (same seed), so it is kept unless another sample follows the route
by a clear margin: normal scenes drive as before, only a branch choice can change.
"""
import numpy as np


def _dist_to_polyline(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point p (N, 2) to the polyline poly (M, 2)."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.einsum("nmk,mk->nm", p[:, None] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-9)
    q = a[None] + np.clip(t, 0, 1)[..., None] * ab[None]
    return np.linalg.norm(p[:, None] - q, axis=2).min(1)


def route_costs(xy: np.ndarray, route: np.ndarray, min_dist_m: float = 8.0) -> np.ndarray:
    """Mean distance of each sample's points beyond min_dist_m from the car to the polyline
    [car, route...]. xy: (K, T, 2) samples in the rig frame. Zero for samples that stay within
    min_dist_m (too short to tell a branch)."""
    route = np.asarray(route, dtype=np.float64).reshape(-1, 2)
    route = route[route[:, 0] > -2.0]
    if len(route) == 0:
        return np.zeros(len(xy))
    poly = np.vstack([np.zeros((1, 2)), route])
    out = np.zeros(len(xy))
    for k, s in enumerate(np.asarray(xy, dtype=np.float64)):
        far = s[np.linalg.norm(s, axis=1) > min_dist_m]
        if len(far):
            out[k] = _dist_to_polyline(far, poly).mean()
    return out


def path_lengths(xy: np.ndarray) -> np.ndarray:
    """Length of each sample's path from the car (origin) through its points, in metres."""
    xy = np.asarray(xy, dtype=np.float64)
    pts = np.concatenate([np.zeros((len(xy), 1, 2)), xy], 1)
    return np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)


def choose(xy: np.ndarray, route: np.ndarray | None, margin_m: float = 2.0, min_len_ratio: float = 0.0,
           big_gap_m: float = 0.0, big_gap_min_len: float = 0.0) -> int:
    """Index of the sample to drive: 0 unless another is closer to the route by more than margin_m.
    min_len_ratio > 0 is a speed guard: only samples whose path is at least that fraction of sample 0's
    are eligible (PAI 2026-10-01: frequent switching to slower samples turned 4 passes into low progress).
    big_gap_m > 0 grades the guard: a sample closer to the route than sample 0 by more than big_gap_m (a lane
    change) only needs big_gap_min_len of sample 0's length (PAI v2: the guard blocked such samples in
    corridor-exit scenes that ended 4.0-4.2 m off)."""
    if route is None or len(xy) < 2:
        return 0
    c = route_costs(xy, route)
    if min_len_ratio > 0:
        L = path_lengths(xy)
        ok = L >= min_len_ratio * L[0]
        if big_gap_m > 0:
            ok |= (c[0] - c > big_gap_m) & (L >= big_gap_min_len * L[0])
        c = np.where(ok, c, np.inf)
        c[0] = route_costs(xy[:1], route)[0]
    j = int(c.argmin())
    return j if c[0] - c[j] > margin_m else 0


def route_pull(xy: np.ndarray, yaw: np.ndarray, route: np.ndarray | None, w_max: float,
               d0_m: float = 30.0, d1_m: float = 45.0) -> tuple[np.ndarray, np.ndarray]:
    """Blend the far part of a plan toward the route (the recorded path 40-80 m ahead).

    PAI bench (2026-10-03): the remaining corridor exits end 4.0-4.2 m from the recorded path, mostly
    where the recorded driver changed lane and ours did not. A point at distance d from the car moves a
    fraction w(d) of the way to its nearest point ON the route waypoints (never onto the car-to-route
    chord, which cuts corners), with w rising linearly from 0 at d0_m to w_max at d1_m. Near points are
    untouched, so the next few seconds still follow the model; headings are recomputed where points moved.
    xy: (T, 2) plan in the rig frame; yaw: (T,). Returns new (xy, yaw)."""
    if route is None or w_max <= 0 or len(xy) < 2:
        return xy, yaw
    r = np.asarray(route, dtype=np.float64).reshape(-1, 2)
    r = r[np.isfinite(r).all(1) & (r[:, 0] > -2.0)]
    if len(r) < 2:
        return xy, yaw
    p = np.asarray(xy, dtype=np.float64)
    d = np.linalg.norm(p, axis=1)
    w = np.clip((d - d0_m) / max(d1_m - d0_m, 1e-6), 0.0, 1.0) * w_max
    if not (w > 0).any():
        return xy, yaw
    a, b = r[:-1], r[1:]
    ab = b - a
    t = np.einsum("nmk,mk->nm", p[:, None] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-9)
    q = a[None] + np.clip(t, 0, 1)[..., None] * ab[None]
    near = q[np.arange(len(p)), np.linalg.norm(p[:, None] - q, axis=2).argmin(1)]
    out = p + w[:, None] * (near - p)
    new_yaw = np.asarray(yaw, dtype=np.float64).copy()
    moved = np.nonzero(w > 0)[0]
    if len(moved):
        seg = np.diff(out, axis=0)
        hd = np.arctan2(seg[:, 1], seg[:, 0])
        for i in moved:
            if np.linalg.norm(seg[min(i, len(seg) - 1)]) > 0.05:
                new_yaw[i] = hd[min(i, len(seg) - 1)]
    return out.astype(np.asarray(xy).dtype), new_yaw.astype(np.asarray(yaw).dtype)


def retime_path(path_xy: np.ndarray, speed_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drive path_xy's geometry with speed_xy's progress over time.

    Point k of the result lies on path_xy at the arc length speed_xy has covered by step k (both plans
    share the 0.1 s step and start at the car). Beyond the end of path_xy the last segment's direction is
    extended. Headings follow the path tangent. Used to take the route from the 4-camera plan and the
    speed from the 6-camera plan, whose rear cameras see a vehicle closing from behind (PAI 2026-10-08:
    6 cameras cut rear-end hits 9 to 3 on scenes 1-40 but added own collisions and corridor exits)."""
    p = np.vstack([np.zeros((1, 2)), np.asarray(path_xy, dtype=np.float64)])
    q = np.vstack([np.zeros((1, 2)), np.asarray(speed_xy, dtype=np.float64)])
    seg = np.diff(p, axis=0); L = np.linalg.norm(seg, axis=1); S = np.concatenate([[0.0], np.cumsum(L)])
    s_target = np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))          # (T,) progress per step
    out = np.empty((len(s_target), 2))
    last_dir = seg[-1] / max(L[-1], 1e-9) if len(L) else np.array([1.0, 0.0])
    for k, s in enumerate(s_target):
        if s >= S[-1]:
            out[k] = p[-1] + (s - S[-1]) * last_dir
        else:
            i = int(np.searchsorted(S, s, side="right") - 1)
            a = (s - S[i]) / max(L[i], 1e-9)
            out[k] = p[i] + a * seg[i]
    d = np.diff(np.vstack([np.zeros((1, 2)), out]), axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    still = np.linalg.norm(d, axis=1) < 1e-3                                  # stopped: hold the path heading
    if still.any():
        path_yaw = np.arctan2(seg[0, 1], seg[0, 0]) if len(seg) else 0.0
        yaw[still] = path_yaw
    return out, yaw


def _route_points(route: np.ndarray | None) -> np.ndarray | None:
    if route is None:
        return None
    r = np.asarray(route, dtype=np.float64).reshape(-1, 2)
    r = r[np.isfinite(r).all(1) & (r[:, 0] > 0.0)]
    return r if len(r) >= 2 else None


def route_deviation(xy: np.ndarray, route: np.ndarray | None) -> float | None:
    """Mean distance from the route waypoints the plan reaches (within its path length) to the plan.

    Measured from the route side, so a plan that runs past the route's end is not penalised (the
    chooser's route_costs grows with plan length at highway speed). None if no waypoint is in reach."""
    r = _route_points(route)
    if r is None or len(xy) < 2:
        return None
    poly = np.vstack([np.zeros((1, 2)), np.asarray(xy, dtype=np.float64)])
    reach = np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()
    r = r[np.linalg.norm(r, axis=1) <= reach]
    if len(r) == 0:
        return None
    return float(_dist_to_polyline(r, poly).mean())


def route_path(route: np.ndarray | None, step_m: float = 1.0) -> np.ndarray | None:
    """A drivable path from the car onto the route: a cubic Hermite curve from the car (heading +x) to the
    first waypoint, arriving along the route's direction, then the route waypoints. Excludes the origin.

    PAI v2 rerun (2026-10-09): in 11 of 14 corridor exits every one of the 6 samples stayed 3-12 m off the
    route (the recorded driver changed lane or took an exit; ours kept its lane), so no choice among samples
    can fix them. route_pull bent only points 30 m+ ahead and the car replans every second, so the bend
    never came closer; this path starts bending at the car."""
    r = _route_points(route)
    if r is None:
        return None
    p1 = r[0]
    d = np.linalg.norm(p1)
    t1 = (r[1] - r[0]) / max(np.linalg.norm(r[1] - r[0]), 1e-9)
    n = max(int(np.ceil(d / step_m)), 2)
    s = np.linspace(0.0, 1.0, n + 1)[1:, None]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    curve = h10 * np.array([d, 0.0]) + h01 * p1 + h11 * (t1 * d)              # h00 * origin = 0
    return np.vstack([curve, r[1:]])
