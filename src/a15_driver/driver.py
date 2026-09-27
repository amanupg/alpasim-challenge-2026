"""AlpaSim E2E challenge driver backed by Alpamayo 1.5 (int8, no-reasoning by default).

Structure follows NVIDIA's VAVAM sample driver (see NOTICE.md): the policy loads in a
background thread after the gRPC server starts; get_version reports UNAVAILABLE until a
warm-up inference has run; each Drive call runs inference inline when a new plan is due
(sim time, A15_INFERENCE_INTERVAL_US) and otherwise serves the cached plan; before the
first plan it returns a straight line at the current speed.

Inputs per plan: 4 frames per camera ending at the newest common frame time t0, chosen
nearest to t0 - k * A15_FRAME_DT_S (PAI sends 10 Hz, so 0.1 s spacing as in training;
nuPlan sends 2 Hz, so the nearest available frames are used and missing ones repeat the
oldest), and a 16-step, 0.1 s ego history ending at t0 in the t0 rig frame.
"""
from __future__ import annotations

import logging
import math
import os
import signal
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass, field
from io import BytesIO

_RUNTIME_WRITE_DIR_DEFAULTS = {
    "XDG_CACHE_HOME": "/tmp/.cache", "TORCH_HOME": "/tmp/torch", "HF_HOME": "/tmp/huggingface",
    "TRITON_CACHE_DIR": "/tmp/triton", "CUDA_CACHE_PATH": "/tmp/nv", "MPLCONFIGDIR": "/tmp/matplotlib",
    "ALPASIM_DRIVER_LOG_DIR": "/run/alpasim-driver",
}
for _k, _v in _RUNTIME_WRITE_DIR_DEFAULTS.items():
    os.environ.setdefault(_k, _v)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from alpasim_grpc import API_VERSION_MESSAGE                        # noqa: E402
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc  # noqa: E402
from PIL import Image                                               # noqa: E402

import grpc                                                         # noqa: E402

from .policy import CAMERA_NAME_TO_INDEX, NUPLAN_TO_SLOT, Alpamayo15Policy  # noqa: E402
from .trajectory import CachedPlan, build_trajectory_from_plan, make_cached_plan  # noqa: E402

LOGGER = logging.getLogger("a15_challenge_driver")
DEFAULT_PAI_CAMERAS = "camera_cross_left_120fov,camera_front_wide_120fov,camera_cross_right_120fov,camera_front_tele_30fov"
N_FRAMES, N_HIST, HIST_DT_S = 4, 16, 0.1


# ---------------------------------------------------------------- geometry helpers
def _quat_arr(q) -> np.ndarray:
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _interp_poses(ts: np.ndarray, pos: np.ndarray, quat: np.ndarray, t: np.ndarray):
    """Linear positions / normalised-lerp quaternions at times t. Outside the known range the
    trajectory is extended at the velocity of the nearest segment (orientation held)."""
    out_p = np.empty((len(t), 3)); out_q = np.empty((len(t), 4))
    for i, ti in enumerate(t):
        j = int(np.clip(np.searchsorted(ts, ti) - 1, 0, max(len(ts) - 2, 0)))
        if len(ts) == 1:
            out_p[i], out_q[i] = pos[0], quat[0]; continue
        dt = ts[j + 1] - ts[j]
        a = (ti - ts[j]) / dt if dt > 0 else 0.0
        out_p[i] = pos[j] + a * (pos[j + 1] - pos[j])
        q0, q1 = quat[j], quat[j + 1]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        ac = min(max(a, 0.0), 1.0)
        q = (1 - ac) * q0 + ac * q1
        out_q[i] = q / np.linalg.norm(q)
    return out_p, out_q


def _yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------- policy handle
class PolicyHandle:
    """Loads the policy after gRPC startup; ready only after a warm-up inference."""

    def __init__(self, kwargs: dict) -> None:
        self._kwargs, self._lock = kwargs, threading.Lock()
        self._policy = None
        self._error: BaseException | None = None

    def start(self) -> None:
        threading.Thread(target=self._load, name="a15-loader", daemon=True).start()

    def get(self):
        with self._lock:
            return self._policy

    def error(self):
        with self._lock:
            return self._error

    def _load(self) -> None:
        try:
            t0 = time.time()
            p = Alpamayo15Policy(**self._kwargs)
            check_processor(p)
            # warm-up at a representative prompt size (4 cameras x 4 frames)
            h, w = target_hw(1080, 1920)
            fr = torch.randint(0, 255, (4, N_FRAMES, 3, h, w), dtype=torch.uint8)
            hx = torch.zeros(1, 1, N_HIST, 3); hr = torch.eye(3).expand(1, 1, N_HIST, 3, 3).clone()
            for _ in range(2):
                p.predict(fr, [0, 1, 2, 6], hx, hr)
            LOGGER.info("policy ready after %.0fs; peak reserved %.2f GiB", time.time() - t0,
                        torch.cuda.max_memory_reserved() / 2**30)
        except BaseException as exc:                           # noqa: BLE001
            LOGGER.exception("policy load failed")
            with self._lock:
                self._error = exc
            return
        with self._lock:
            self._policy = p


# Alpamayo 1.5 processor limits (alpamayo1_5.helper MIN_PIXELS / MAX_PIXELS) and the Qwen3-VL
# resize factor (patch 16 x merge 2); checked against the live processor at warm-up.
MIN_PIXELS, MAX_PIXELS, RESIZE_FACTOR = 163840, 196608, 32


def target_hw(h: int, w: int) -> tuple[int, int]:
    """Size the Qwen processor would resize an (h, w) image to. Frames are pre-resized to
    exactly this, so the processor does not resample again (and JPEG decode can use draft mode)."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    return tuple(smart_resize(h, w, factor=RESIZE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS))


def check_processor(policy) -> None:
    ip = policy._proc.image_processor
    got = (ip.patch_size * ip.merge_size, getattr(ip, "min_pixels", None) or ip.size.get("shortest_edge"),
           getattr(ip, "max_pixels", None) or ip.size.get("longest_edge"))
    assert got == (RESIZE_FACTOR, MIN_PIXELS, MAX_PIXELS), f"processor resize config changed: {got}"


# ---------------------------------------------------------------- session state
@dataclass
class SessionState:
    cams: list[str] = field(default_factory=list)           # logical ids, ascending Alpamayo slot
    slots: list[int] = field(default_factory=list)
    frames: dict[str, deque] = field(default_factory=dict)  # logical id -> deque[(t_us, CHW uint8 tensor)]
    poses: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)  # (t_us, xyz, wxyz)
    speed: float = 0.0
    latest_pose: common_pb2.PoseAtTime | None = None
    cached_plan: CachedPlan | None = None
    stats: dict = field(default_factory=lambda: {"drive": 0, "infer": 0, "infer_err": 0, "fallback": 0,
                                                 "cached": 0, "infer_s": 0.0})


class A15ChallengeDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self, *, handle: PolicyHandle, pai_cameras: list[str], interval_us: int, frame_dt_s: float):
        self._handle, self._pai_cameras = handle, pai_cameras
        self._interval_us, self._frame_dt_us = interval_us, int(frame_dt_s * 1e6)
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()
        self._infer_lock = threading.Lock()                  # one model, one GPU: serialise inference
        self._target_hw: dict[tuple[int, int], tuple[int, int]] = {}
        self._server: grpc.Server | None = None

    def attach_server(self, server) -> None:
        self._server = server

    # ------------------------------------------------------------ sessions
    def start_session(self, request, context):
        avail = [c.logical_id for c in request.rollout_spec.vehicle.available_cameras if c.logical_id]
        pai = [c for c in self._pai_cameras if c in avail]
        if pai:
            chosen = sorted(((CAMERA_NAME_TO_INDEX[c], c) for c in pai))
        else:
            chosen = sorted(((NUPLAN_TO_SLOT[c], c) for c in avail if c in NUPLAN_TO_SLOT))
        s = SessionState(cams=[c for _, c in chosen], slots=[k for k, _ in chosen])
        s.frames = {c: deque(maxlen=24) for c in s.cams}
        with self._lock:
            self._sessions[request.session_uuid] = s
        LOGGER.info("session %s: cameras %s (slots %s) of %d available", request.session_uuid, s.cams, s.slots, len(avail))
        return common_pb2.SessionRequestStatus()

    def close_session(self, request, context):
        with self._lock:
            s = self._sessions.pop(request.session_uuid, None)
        if s is not None:
            st = s.stats
            LOGGER.info("closed session %s: drive %d, inferences %d (errors %d, mean %.3fs), cached %d, fallback %d",
                        request.session_uuid, st["drive"], st["infer"], st["infer_err"],
                        st["infer_s"] / max(st["infer"], 1), st["cached"], st["fallback"])
        return common_pb2.Empty()

    # ------------------------------------------------------------ observations
    def submit_image_observation(self, request, context):
        img = request.camera_image
        s = self._get_session(request.session_uuid, context)
        if img.logical_id not in s.frames:
            return common_pb2.Empty()
        im = Image.open(BytesIO(img.image_bytes))
        key = (im.height, im.width)
        if key not in self._target_hw:
            self._target_hw[key] = target_hw(*key)
        th, tw = self._target_hw[key]
        im.draft("RGB", (tw, th))                            # JPEG decodes at a reduced scale >= target
        im = im.convert("RGB").resize((tw, th), Image.BICUBIC)
        t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
        with self._lock:
            s.frames[img.logical_id].append((int(img.frame_end_us), t))
        return common_pb2.Empty()

    def submit_egomotion_observation(self, request, context):
        s = self._get_session(request.session_uuid, context)
        with self._lock:
            for p in request.trajectory.poses:
                s.poses.append((int(p.timestamp_us), np.array([p.pose.vec.x, p.pose.vec.y, p.pose.vec.z]),
                                _quat_arr(p.pose.quat)))
                s.latest_pose = p
            s.poses.sort(key=lambda x: x[0])
            s.poses = s.poses[-200:]
            if request.dynamic_states:
                v = request.dynamic_states[-1].linear_velocity
                s.speed = float(math.hypot(v.x, v.y))
            elif len(s.poses) >= 2:
                (ta, pa, _), (tb, pb, _) = s.poses[-2], s.poses[-1]
                if tb > ta:
                    s.speed = float(np.linalg.norm((pb - pa)[:2]) / ((tb - ta) / 1e6))
        return common_pb2.Empty()

    def submit_route(self, request, context):
        return common_pb2.Empty()

    def submit_recording_ground_truth(self, request, context):
        return common_pb2.Empty()

    # ------------------------------------------------------------ drive
    def drive(self, request, context):
        s = self._get_session(request.session_uuid, context)
        now = int(request.time_now_us)
        with self._lock:
            s.stats["drive"] += 1
        self._maybe_infer(s, now)
        with self._lock:
            plan, pose, speed = s.cached_plan, s.latest_pose, s.speed
            s.stats["cached" if plan is not None else "fallback"] += 1
        traj = build_trajectory_from_plan(plan, pose, now, callback_frequency_hz=10.0,
                                          fallback_speed_mps=max(1.0, speed), max_horizon_s=6.0)
        return egodriver_pb2.DriveResponse(trajectory=traj)

    def _maybe_infer(self, s: SessionState, now: int) -> None:
        policy = self._handle.get()
        if policy is None:
            return
        with self._lock:
            plan = s.cached_plan
            if plan is not None and now - plan.created_time_us < self._interval_us:
                return
            batch = self._gather(s)
        if batch is None:
            return
        frames, hx, hr, t0, anchor = batch
        t = time.time()
        try:
            with self._infer_lock:
                xy, yaw, info = policy.predict(frames, s.slots, hx, hr)
        except Exception:                                    # noqa: BLE001
            LOGGER.exception("inference failed")
            with self._lock:
                s.stats["infer_err"] += 1
            return
        new = make_cached_plan(created_time_us=t0, anchor_pose=anchor, trajectory_xy=xy, headings=yaw,
                               source_frequency_hz=policy.output_frequency_hz)
        with self._lock:
            s.stats["infer"] += 1; s.stats["infer_s"] += time.time() - t
            if new is not None:
                s.cached_plan = new

    def _gather(self, s: SessionState):
        """Frames (n_cam, 4, 3, H, W), ego history, t0 and the anchor pose; None if not ready."""
        if not s.cams or any(len(s.frames[c]) == 0 for c in s.cams) or len(s.poses) < 2:
            return None
        t0 = min(s.frames[c][-1][0] for c in s.cams)
        per_cam = []
        for c in s.cams:
            buf = [f for f in s.frames[c] if f[0] <= t0]
            if not buf:
                return None
            picked, used = [], set()
            for k in range(N_FRAMES - 1, -1, -1):
                target = t0 - k * self._frame_dt_us
                cand = [(abs(ts - target), i) for i, (ts, _) in enumerate(buf) if i not in used]
                i = min(cand)[1] if cand else 0
                used.add(i); picked.append(buf[i])
            picked.sort(key=lambda f: f[0])
            shapes = {f[1].shape for f in picked}
            if len(shapes) != 1:
                return None
            per_cam.append(torch.stack([f[1] for f in picked]))
        if len({p.shape for p in per_cam}) != 1:
            return None                                      # cameras of different sizes: wait
        frames = torch.stack(per_cam)
        ts = np.array([p[0] for p in s.poses], dtype=np.float64)
        pos = np.stack([p[1] for p in s.poses]); quat = np.stack([p[2] for p in s.poses])
        th = t0 - np.arange(N_HIST - 1, -1, -1) * HIST_DT_S * 1e6
        hp, hq = _interp_poses(ts, pos, quat, th)
        R0 = _quat_to_rot(hq[-1])
        hx = (hp - hp[-1]) @ R0                                 # rows: R0^T (p - p0)
        hr = np.stack([R0.T @ _quat_to_rot(q) for q in hq])
        anchor = common_pb2.PoseAtTime(
            timestamp_us=int(t0),
            pose=common_pb2.Pose(vec=common_pb2.Vec3(x=float(hp[-1][0]), y=float(hp[-1][1]), z=float(hp[-1][2])),
                                 quat=common_pb2.Quat(w=float(hq[-1][0]), x=float(hq[-1][1]),
                                                      y=float(hq[-1][2]), z=float(hq[-1][3]))))
        return (frames, torch.from_numpy(hx).float()[None, None], torch.from_numpy(hr).float()[None, None],
                int(t0), anchor)

    # ------------------------------------------------------------ misc
    def get_version(self, request, context):
        err = self._handle.error()
        if err is not None:
            context.abort(grpc.StatusCode.UNAVAILABLE, f"policy load failed: {err}")
        if self._handle.get() is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "policy is still loading")
        return common_pb2.VersionId(version_id="a15-int8-e2e-driver", git_hash=os.environ.get("A15_GIT_HASH", "local"),
                                    grpc_api_version=API_VERSION_MESSAGE)

    def shut_down(self, request, context):
        if self._server is not None:
            threading.Thread(target=lambda: (time.sleep(0.05), self._server.stop(grace=0.0)), daemon=True).start()
        return common_pb2.Empty()

    def _get_session(self, uuid: str, context) -> SessionState:
        with self._lock:
            s = self._sessions.get(uuid)
        if s is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {uuid}")
        return s


def main() -> None:
    for k in _RUNTIME_WRITE_DIR_DEFAULTS:
        try:
            os.makedirs(os.environ[k], exist_ok=True)
        except OSError:
            if k == "ALPASIM_DRIVER_LOG_DIR":
                os.environ[k] = "/tmp/alpasim-driver"; os.makedirs(os.environ[k], exist_ok=True)
            else:
                raise
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
    logging.basicConfig(level=os.environ.get("ALPASIM_DRIVER_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    assets = os.environ.get("A15_ASSETS", "/app/assets")
    handle = PolicyHandle(dict(
        model_dir=os.environ.get("A15_MODEL_DIR", f"{assets}/alpamayo15"),
        vlm_dir=os.environ.get("A15_VLM_DIR", f"{assets}/qwen3vl8b"),
        processor_dir=os.environ.get("A15_PROCESSOR_DIR", f"{assets}/qwen3vl2b"),
        quant=os.environ.get("A15_QUANT", "int8wo"),
        reasoning=os.environ.get("A15_REASONING", "0") == "1"))
    service = A15ChallengeDriver(
        handle=handle,
        pai_cameras=[c.strip() for c in os.environ.get("A15_PAI_CAMERAS", DEFAULT_PAI_CAMERAS).split(",") if c.strip()],
        interval_us=int(os.environ.get("A15_INFERENCE_INTERVAL_US", "500000")),
        frame_dt_s=float(os.environ.get("A15_FRAME_DT_S", "0.1")))
    host, port = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0"), int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=int(os.environ.get("ALPASIM_DRIVER_GRPC_WORKERS", "8"))))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(service, server)
    service.attach_server(server)
    if server.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    signal.signal(signal.SIGTERM, lambda *_: server.stop(grace=0.0))
    signal.signal(signal.SIGINT, lambda *_: server.stop(grace=0.0))
    server.start()
    LOGGER.info("Alpamayo 1.5 driver listening on %s:%d (replica %s/%s)", host, port,
                os.environ.get("ALPASIM_CONTESTANT_REPLICA_INDEX", "?"), os.environ.get("ALPASIM_CONTESTANT_REPLICAS", "?"))
    handle.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
