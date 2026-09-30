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


def choose(xy: np.ndarray, route: np.ndarray | None, margin_m: float = 2.0) -> int:
    """Index of the sample to drive: 0 unless another is closer to the route by more than margin_m."""
    if route is None or len(xy) < 2:
        return 0
    c = route_costs(xy, route)
    j = int(c.argmin())
    return j if c[0] - c[j] > margin_m else 0
