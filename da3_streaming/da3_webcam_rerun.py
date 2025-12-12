"""Live webcam streaming to Rerun with Depth Anything 3.

This script is a lightweight, online variant inspired by `da3_streaming.py`:
- Capture RGB frames from a webcam (OpenCV).
- Run Depth Anything 3 inference per-frame to get depth (+ intrinsics when available).
- Estimate frame-to-frame camera motion with a simple RGB-D VO:
  - Track features with LK optical flow.
  - Lift previous-frame features to 3D using the previous depth.
  - SolvePnP(RANSAC) to estimate the relative pose.
- Unproject depth to a colored point cloud and transform it into a global frame.
- Stream the camera trajectory and point cloud to Rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from depth_anything_3.api import DepthAnything3


def _require_rerun():
    try:
        import rerun as rr  # type: ignore
    except ModuleNotFoundError:
        print(
            "[ERROR] rerun is not installed. Install it with: pip install rerun-sdk",
            file=sys.stderr,
        )
        raise
    return rr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam → Depth Anything 3 → point cloud + camera path in Rerun."
    )

    parser.add_argument(
        "--mode",
        choices=("chunk", "vo"),
        default="chunk",
        help="Mapping mode: 'chunk' (DA3 multi-view + overlap Sim3) or 'vo' (PnP-based).",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="cv2.VideoCapture index.")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror frames horizontally (useful for selfie webcams).",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=960,
        help="Resize webcam frames before inference to keep latency manageable.",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Processing resolution passed to DepthAnything3.inference.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (defaults to cuda if available, otherwise cpu).",
    )

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face repo or local path passed to DepthAnything3.from_pretrained.",
    )
    model_group.add_argument(
        "--weights",
        default="weights/model.safetensors",
        help="Local safetensors weights (default: weights/model.safetensors).",
    )
    parser.add_argument(
        "--weights-config",
        default="weights/config.json",
        help="Local model config json (default: weights/config.json).",
    )

    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Spawn the Rerun viewer automatically (recommended).",
    )

    # Point cloud controls
    parser.add_argument("--stride", type=int, default=4, help="Pixel stride for point sampling.")
    parser.add_argument("--min-depth", type=float, default=0.05, help="Min depth to keep.")
    parser.add_argument("--max-depth", type=float, default=50.0, help="Max depth to keep.")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="Min confidence to keep (set <=0 to disable conf filtering).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Cap accumulated map points to this size.",
    )

    # Chunk-streaming controls (DA3-style streaming)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=12,
        help="Number of frames per DA3 multi-view inference (chunk mode).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Number of overlapping frames between chunks (default: chunk_size//2).",
    )
    parser.add_argument(
        "--ref-view-strategy",
        default="saddle_balanced",
        help="DA3 reference view strategy for multi-view inference.",
    )
    parser.add_argument(
        "--align-stride",
        type=int,
        default=8,
        help="Pixel stride for overlap correspondence sampling (chunk alignment).",
    )
    parser.add_argument(
        "--align-max-corr",
        type=int,
        default=50_000,
        help="Max correspondences used to estimate Sim3 between chunks.",
    )
    parser.add_argument(
        "--align-delta",
        type=float,
        default=0.1,
        help="Huber delta for robust Sim3 estimation (chunk alignment).",
    )
    parser.add_argument(
        "--align-max-iters",
        type=int,
        default=15,
        help="Max IRLS iterations for robust Sim3 estimation (chunk alignment).",
    )
    parser.add_argument(
        "--align-min-corr",
        type=int,
        default=5_000,
        help="Minimum correspondences required to accept a Sim3 update.",
    )
    parser.add_argument(
        "--align-conf-ratio",
        type=float,
        default=0.1,
        help="Alignment conf threshold ratio vs. per-frame median.",
    )

    # VO controls
    parser.add_argument("--no-vo", action="store_true", help="Disable VO; keep pose fixed.")
    parser.add_argument("--max-features", type=int, default=1500, help="Max tracked features.")
    parser.add_argument("--quality-level", type=float, default=0.01, help="GFTT qualityLevel.")
    parser.add_argument("--min-distance", type=int, default=8, help="GFTT minDistance.")
    parser.add_argument(
        "--pnp-reproj-error",
        type=float,
        default=3.0,
        help="solvePnPRansac reprojection error threshold (pixels).",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=50,
        help="Minimum inliers to accept a pose update.",
    )

    return parser.parse_args()


def pick_device(user_choice: str | None) -> torch.device:
    if user_choice:
        return torch.device(user_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resize_long_edge_bgr(frame_bgr: np.ndarray, target_edge: int) -> np.ndarray:
    if target_edge <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = target_edge / float(max(h, w))
    if scale >= 1.0:
        return frame_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def ensure_pinhole_from_size(h: int, w: int) -> np.ndarray:
    fx = 0.9 * float(w)
    fy = 0.9 * float(w)
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def as_w2c_34(w2c: np.ndarray) -> np.ndarray:
    if w2c.shape == (3, 4):
        return w2c.astype(np.float32, copy=False)
    if w2c.shape == (4, 4):
        return w2c[:3, :].astype(np.float32, copy=False)
    raise ValueError(f"w2c must be (3,4) or (4,4), got {w2c.shape}")


def invert_w2c_to_c2w_44(w2c: np.ndarray) -> np.ndarray:
    w2c_34 = as_w2c_34(w2c)
    R = w2c_34[:3, :3]
    t = w2c_34[:3, 3]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = (-R.T @ t).astype(np.float32)
    return c2w


def backproject_world_points_from_w2c(
    depth: np.ndarray, K: np.ndarray, w2c: np.ndarray, us: np.ndarray, vs: np.ndarray
) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {depth.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")
    if us.shape != vs.shape:
        raise ValueError("us and vs must have the same shape.")

    w2c_34 = as_w2c_34(w2c)
    R = w2c_34[:3, :3].astype(np.float32, copy=False)
    t = w2c_34[:3, 3].astype(np.float32, copy=False)

    z = depth[vs, us].astype(np.float32, copy=False)
    pix = np.stack([us.astype(np.float32), vs.astype(np.float32), np.ones_like(z)], axis=0)  # 3,M
    rays = (np.linalg.inv(K.astype(np.float64)) @ pix.astype(np.float64)).astype(np.float32)
    Xc = rays * z[None, :]  # 3,M

    Xw = (R.T @ (Xc - t[:, None])).T
    return Xw.astype(np.float32, copy=False)


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32, copy=False)
    return (float(s) * (points @ R.T) + t[None, :]).astype(np.float32, copy=False)


def compose_sim3(
    a: Tuple[float, np.ndarray, np.ndarray], b: Tuple[float, np.ndarray, np.ndarray]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compose Sim3 transforms (points): x' = s R x + t.
    Returns a∘b: x'' = sa Ra (sb Rb x + tb) + ta.
    """
    sa, Ra, ta = a
    sb, Rb, tb = b
    s = float(sa) * float(sb)
    R = Ra @ Rb
    t = (float(sa) * (Ra @ tb) + ta).astype(np.float32)
    return s, R.astype(np.float32), t


def estimate_sim3_weighted(
    src: np.ndarray, tgt: np.ndarray, weights: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Weighted Umeyama (Sim3): tgt ~= s R src + t
    """
    if src.shape != tgt.shape or src.shape[1] != 3:
        raise ValueError(f"src/tgt must be (N,3) and same shape, got {src.shape} / {tgt.shape}")
    w = weights.reshape(-1).astype(np.float64, copy=False)
    if w.shape[0] != src.shape[0]:
        raise ValueError("weights length must match src/tgt length.")
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 1e-12:
        raise ValueError("Sum of weights too small.")
    w = w / w_sum

    src_f = src.astype(np.float64, copy=False)
    tgt_f = tgt.astype(np.float64, copy=False)
    mu_src = np.sum(src_f * w[:, None], axis=0)
    mu_tgt = np.sum(tgt_f * w[:, None], axis=0)
    src_c = src_f - mu_src[None, :]
    tgt_c = tgt_f - mu_tgt[None, :]

    C = (tgt_c * w[:, None]).T @ src_c  # 3x3
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    R = (U @ S @ Vt).astype(np.float32)

    var_src = np.sum(w * np.sum(src_c * src_c, axis=1))
    var_src = float(max(var_src, 1e-12))
    s = float(np.sum(D * np.diag(S)) / var_src)
    t = (mu_tgt - s * (R.astype(np.float64) @ mu_src)).astype(np.float32)
    return s, R, t


def robust_sim3_irls(
    src: np.ndarray,
    tgt: np.ndarray,
    weights: np.ndarray,
    *,
    delta: float,
    max_iters: int,
    tol: float = 1e-9,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    IRLS with Huber weights on top of weighted Umeyama.
    """
    s, R, t = estimate_sim3_weighted(src, tgt, weights)
    prev_err = float("inf")

    w0 = weights.astype(np.float64, copy=False)
    for _ in range(int(max_iters)):
        pred = float(s) * (src @ R.T) + t[None, :]
        resid = np.linalg.norm((tgt - pred).astype(np.float64), axis=1)
        abs_r = np.abs(resid)
        huber = np.ones_like(abs_r)
        mask = abs_r > float(delta)
        huber[mask] = float(delta) / (abs_r[mask] + 1e-12)

        w = w0 * huber
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 1e-12:
            break
        w = (w / w_sum).astype(np.float64)

        s_new, R_new, t_new = estimate_sim3_weighted(src, tgt, w)

        ds = abs(float(s_new) - float(s))
        dt = float(np.linalg.norm(t_new - t))
        err = float(np.sum((resid * resid) * w0))
        if (ds + dt) < tol or abs(prev_err - err) < tol * max(prev_err, 1.0):
            s, R, t = s_new, R_new, t_new
            break

        s, R, t = s_new, R_new, t_new
        prev_err = err

    return float(s), R.astype(np.float32), t.astype(np.float32)


def unproject_depth_to_world(
    depth: np.ndarray,
    rgb_u8: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    *,
    stride: int,
    min_depth: float,
    max_depth: float,
    conf: Optional[np.ndarray] = None,
    conf_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {depth.shape}")
    if rgb_u8.shape[:2] != depth.shape:
        raise ValueError(f"rgb and depth size mismatch: {rgb_u8.shape} vs {depth.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")
    if c2w.shape != (4, 4):
        raise ValueError(f"c2w must be (4,4), got {c2w.shape}")

    H, W = depth.shape
    s = max(1, int(stride))

    vs, us = np.mgrid[0:H:s, 0:W:s]
    d = depth[vs, us].astype(np.float32)

    valid = np.isfinite(d) & (d > float(min_depth)) & (d < float(max_depth))
    if conf is not None and conf_threshold > 0:
        valid &= conf[vs, us] >= float(conf_threshold)

    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    us_f = us[valid].astype(np.float32)
    vs_f = vs[valid].astype(np.float32)
    d_f = d[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x = (us_f - cx) * d_f / fx
    y = (vs_f - cy) * d_f / fy
    z = d_f

    pts_cam = np.stack([x, y, z], axis=-1)  # (N,3)
    cols = rgb_u8[vs[valid], us[valid]].astype(np.uint8)  # (N,3) RGB

    R = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3].astype(np.float32)
    pts_world = (R @ pts_cam.T).T + t[None, :]

    return pts_world.astype(np.float32), cols


@dataclass
class VoResult:
    updated: bool
    num_tracked: int
    num_inliers: int
    c2w: np.ndarray


def estimate_pose_pnp(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_depth: np.ndarray,
    prev_K: np.ndarray,
    curr_K: np.ndarray,
    prev_c2w: np.ndarray,
    *,
    max_features: int,
    quality_level: float,
    min_distance: int,
    min_depth: float,
    max_depth: float,
    reproj_error: float,
    min_inliers: int,
) -> VoResult:
    if prev_gray.ndim != 2 or curr_gray.ndim != 2:
        raise ValueError("prev_gray and curr_gray must be grayscale (H,W).")

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=int(max_features),
        qualityLevel=float(quality_level),
        minDistance=float(min_distance),
        blockSize=7,
        useHarrisDetector=False,
    )
    if prev_pts is None or len(prev_pts) < max(6, min_inliers):
        return VoResult(
            updated=False, num_tracked=0, num_inliers=0, c2w=prev_c2w.astype(np.float32)
        )

    curr_pts, st, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    if curr_pts is None or st is None:
        return VoResult(
            updated=False, num_tracked=0, num_inliers=0, c2w=prev_c2w.astype(np.float32)
        )

    st = st.reshape(-1).astype(bool)
    prev_uv = prev_pts.reshape(-1, 2)[st]
    curr_uv = curr_pts.reshape(-1, 2)[st]

    if prev_uv.shape[0] < max(6, min_inliers):
        return VoResult(
            updated=False,
            num_tracked=int(prev_uv.shape[0]),
            num_inliers=0,
            c2w=prev_c2w.astype(np.float32),
        )

    H, W = prev_depth.shape
    u = prev_uv[:, 0]
    v = prev_uv[:, 1]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    prev_uv = prev_uv[in_bounds]
    curr_uv = curr_uv[in_bounds]

    if prev_uv.shape[0] < max(6, min_inliers):
        return VoResult(
            updated=False,
            num_tracked=int(prev_uv.shape[0]),
            num_inliers=0,
            c2w=prev_c2w.astype(np.float32),
        )

    z = prev_depth[vi, ui].astype(np.float32)
    depth_ok = np.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))
    z = z[depth_ok]
    prev_uv = prev_uv[depth_ok]
    curr_uv = curr_uv[depth_ok]

    if prev_uv.shape[0] < max(6, min_inliers):
        return VoResult(
            updated=False,
            num_tracked=int(prev_uv.shape[0]),
            num_inliers=0,
            c2w=prev_c2w.astype(np.float32),
        )

    fx0, fy0, cx0, cy0 = (
        float(prev_K[0, 0]),
        float(prev_K[1, 1]),
        float(prev_K[0, 2]),
        float(prev_K[1, 2]),
    )
    X = (prev_uv[:, 0] - cx0) * z / fx0
    Y = (prev_uv[:, 1] - cy0) * z / fy0
    obj_pts = np.stack([X, Y, z], axis=-1).astype(np.float32)
    img_pts = curr_uv.astype(np.float32)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=curr_K.astype(np.float64),
        distCoeffs=None,
        reprojectionError=float(reproj_error),
        iterationsCount=100,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    num_inliers = int(len(inliers)) if inliers is not None else 0
    if not ok or num_inliers < int(min_inliers):
        return VoResult(
            updated=False,
            num_tracked=int(img_pts.shape[0]),
            num_inliers=num_inliers,
            c2w=prev_c2w.astype(np.float32),
        )

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    T_prev_to_curr = np.eye(4, dtype=np.float32)
    T_prev_to_curr[:3, :3] = R.astype(np.float32)
    T_prev_to_curr[:3, 3] = t.astype(np.float32)

    T_curr_to_prev = np.linalg.inv(T_prev_to_curr).astype(np.float32)
    c2w = (prev_c2w @ T_curr_to_prev).astype(np.float32)

    return VoResult(
        updated=True, num_tracked=int(img_pts.shape[0]), num_inliers=num_inliers, c2w=c2w
    )


def _collect_overlap_correspondences(
    prev_pred,
    curr_pred,
    *,
    overlap: int,
    stride: int,
    max_corr: int,
    min_depth: float,
    max_depth: float,
    conf_threshold_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if overlap <= 0:
        raise ValueError("overlap must be > 0.")
    if prev_pred.depth.shape[0] < overlap or curr_pred.depth.shape[0] < overlap:
        raise ValueError("predictions shorter than overlap.")

    src_pts_all = []
    tgt_pts_all = []
    w_all = []

    prev_start = prev_pred.depth.shape[0] - overlap
    for i in range(overlap):
        dp = prev_pred.depth[prev_start + i].astype(np.float32, copy=False)
        dc = curr_pred.depth[i].astype(np.float32, copy=False)
        Kp = prev_pred.intrinsics[prev_start + i].astype(np.float32, copy=False)
        Kc = curr_pred.intrinsics[i].astype(np.float32, copy=False)
        w2cp = prev_pred.extrinsics[prev_start + i]
        w2cc = curr_pred.extrinsics[i]

        Hp, Wp = dp.shape
        if dc.shape != (Hp, Wp):
            continue

        vs, us = np.mgrid[0:Hp:int(stride), 0:Wp:int(stride)]
        us = us.reshape(-1)
        vs = vs.reshape(-1)

        zp = dp[vs, us]
        zc = dc[vs, us]
        valid = (
            np.isfinite(zp)
            & np.isfinite(zc)
            & (zp > float(min_depth))
            & (zc > float(min_depth))
            & (zp < float(max_depth))
            & (zc < float(max_depth))
        )

        cp = None
        cc = None
        if prev_pred.conf is not None and curr_pred.conf is not None:
            cp = prev_pred.conf[prev_start + i].astype(np.float32, copy=False)[vs, us]
            cc = curr_pred.conf[i].astype(np.float32, copy=False)[vs, us]
            thr = float(min(np.median(cp), np.median(cc)) * float(conf_threshold_ratio))
            thr = max(thr, 1e-6)
            valid &= np.isfinite(cp) & np.isfinite(cc) & (cp >= thr) & (cc >= thr)

        if not np.any(valid):
            continue

        us_v = us[valid].astype(np.int32, copy=False)
        vs_v = vs[valid].astype(np.int32, copy=False)

        tgt_pts = backproject_world_points_from_w2c(dp, Kp, w2cp, us_v, vs_v)
        src_pts = backproject_world_points_from_w2c(dc, Kc, w2cc, us_v, vs_v)

        if cp is None or cc is None:
            weights = np.ones((src_pts.shape[0],), dtype=np.float32)
        else:
            weights = np.sqrt(np.clip(cp[valid], 0.0, None) * np.clip(cc[valid], 0.0, None)).astype(
                np.float32
            )
            weights = np.maximum(weights, 1e-6)

        src_pts_all.append(src_pts)
        tgt_pts_all.append(tgt_pts)
        w_all.append(weights)

    if not src_pts_all:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    src = np.concatenate(src_pts_all, axis=0)
    tgt = np.concatenate(tgt_pts_all, axis=0)
    w = np.concatenate(w_all, axis=0)

    if src.shape[0] > int(max_corr):
        sel = np.random.choice(src.shape[0], int(max_corr), replace=False)
        src = src[sel]
        tgt = tgt[sel]
        w = w[sel]

    return src.astype(np.float32, copy=False), tgt.astype(np.float32, copy=False), w.astype(
        np.float32, copy=False
    )


def reservoir_update(
    pts: np.ndarray,
    cols: np.ndarray,
    *,
    reservoir_pts: np.ndarray,
    reservoir_cols: np.ndarray,
    filled: int,
    seen: int,
) -> Tuple[int, int]:
    if pts.size == 0:
        return filled, seen
    if pts.shape[0] != cols.shape[0]:
        raise ValueError("pts and cols must have same length.")

    capacity = reservoir_pts.shape[0]
    n_new = int(pts.shape[0])

    # Fill phase
    if filled < capacity:
        take = min(capacity - filled, n_new)
        reservoir_pts[filled : filled + take] = pts[:take]
        reservoir_cols[filled : filled + take] = cols[:take]
        filled += take
        seen += take
        pts = pts[take:]
        cols = cols[take:]

    # Reservoir phase
    n_rem = int(pts.shape[0])
    if n_rem <= 0:
        return filled, seen

    idxs = np.arange(seen + 1, seen + n_rem + 1, dtype=np.float64)  # 1-based stream indices
    rand = (np.random.random(size=n_rem) * idxs).astype(np.int64)
    replace_mask = rand < capacity
    replace_pos = rand[replace_mask]
    if replace_pos.size > 0:
        reservoir_pts[replace_pos] = pts[replace_mask]
        reservoir_cols[replace_pos] = cols[replace_mask]
    seen += n_rem
    return filled, seen


def load_da3_model(args: argparse.Namespace, device: torch.device) -> DepthAnything3:
    if args.model_id:
        model = DepthAnything3.from_pretrained(args.model_id)
        return model.to(device).eval()

    with open(args.weights_config, "r") as f:
        cfg = json.load(f)
    model_name = cfg.get("model_name", "da3nested-giant-large")
    model = DepthAnything3(model_name=model_name)
    weights = load_file(args.weights)
    model.load_state_dict(weights, strict=False)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    rr = _require_rerun()

    if args.overlap is None:
        args.overlap = max(0, int(args.chunk_size) // 2)
    args.overlap = int(args.overlap)
    if args.overlap >= int(args.chunk_size):
        raise SystemExit("[ERROR] --overlap must be < --chunk-size")

    device = pick_device(args.device)
    print(f"[INFO] Loading model on {device}...", file=sys.stderr)
    model = load_da3_model(args, device)

    rr.init("da3_webcam_stream")
    if args.spawn:
        try:
            rr.spawn()
        except Exception:
            pass

    try:
        rr.log("world", rr.ViewCoordinates.RDF, timeless=True)
    except Exception:
        pass

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera_index}", file=sys.stderr)
        raise SystemExit(1)

    prev_gray = None
    prev_depth = None
    prev_K = None
    c2w = np.eye(4, dtype=np.float32)
    cam_positions: list[np.ndarray] = []

    reservoir_pts = np.zeros((int(args.max_points), 3), dtype=np.float32)
    reservoir_cols = np.zeros((int(args.max_points), 3), dtype=np.uint8)
    reservoir_filled = 0
    reservoir_seen = 0

    frame_buf: list[np.ndarray] = []
    id_buf: list[int] = []
    prev_chunk_pred = None
    prev_chunk_sim3: Tuple[float, np.ndarray, np.ndarray] | None = None

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[WARN] Failed to grab frame.", file=sys.stderr)
                break

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_bgr = resize_long_edge_bgr(frame_bgr, args.max_edge)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if args.mode == "vo":
                t0 = time.time()
                pred = model.inference(
                    [frame_rgb],
                    process_res=args.process_res,
                    process_res_method="upper_bound_resize",
                    export_dir=None,
                    export_format="mini_npz",
                )
                infer_ms = (time.time() - t0) * 1000.0

                depth = pred.depth[0].astype(np.float32)
                rgb_u8 = pred.processed_images[0].astype(np.uint8)
                conf = pred.conf[0].astype(np.float32) if pred.conf is not None else None

                K = pred.intrinsics[0].astype(np.float32) if pred.intrinsics is not None else None
                if K is None:
                    K = ensure_pinhole_from_size(depth.shape[0], depth.shape[1])

                gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)

                vo = None
                if (
                    not args.no_vo
                    and prev_gray is not None
                    and prev_depth is not None
                    and prev_K is not None
                ):
                    vo = estimate_pose_pnp(
                        prev_gray=prev_gray,
                        curr_gray=gray,
                        prev_depth=prev_depth,
                        prev_K=prev_K,
                        curr_K=K,
                        prev_c2w=c2w,
                        max_features=args.max_features,
                        quality_level=args.quality_level,
                        min_distance=args.min_distance,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        reproj_error=args.pnp_reproj_error,
                        min_inliers=args.min_inliers,
                    )
                    c2w = vo.c2w

                pts_world, cols = unproject_depth_to_world(
                    depth=depth,
                    rgb_u8=rgb_u8,
                    K=K,
                    c2w=c2w,
                    stride=args.stride,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    conf=conf,
                    conf_threshold=args.conf_threshold,
                )

                reservoir_filled, reservoir_seen = reservoir_update(
                    pts_world,
                    cols,
                    reservoir_pts=reservoir_pts,
                    reservoir_cols=reservoir_cols,
                    filled=reservoir_filled,
                    seen=reservoir_seen,
                )

                cam_positions.append(c2w[:3, 3].copy())
                cam_path = np.asarray(cam_positions, dtype=np.float32)

                try:
                    rr.set_time_sequence("frame", frame_idx)
                except Exception:
                    pass

                pts_view = (
                    reservoir_pts[:reservoir_filled]
                    if reservoir_filled < reservoir_pts.shape[0]
                    else reservoir_pts
                )
                col_view = (
                    reservoir_cols[:reservoir_filled]
                    if reservoir_filled < reservoir_cols.shape[0]
                    else reservoir_cols
                )
                rr.log("world/points", rr.Points3D(pts_view, colors=col_view))
                rr.log("world/camera_path", rr.LineStrips3D([cam_path]))

                try:
                    rr.log("world/rgb", rr.Image(rgb_u8))
                except Exception:
                    pass
                try:
                    rr.log("world/depth", rr.DepthImage(depth))
                except Exception:
                    pass
                try:
                    if vo is not None:
                        rr.log(
                            "world/stats",
                            rr.TextLog(
                                f"infer={infer_ms:.1f}ms, tracked={vo.num_tracked}, inliers={vo.num_inliers}"
                            ),
                        )
                    else:
                        rr.log("world/stats", rr.TextLog(f"infer={infer_ms:.1f}ms"))
                except Exception:
                    pass

                frame_idx += 1
                prev_gray = gray
                prev_depth = depth
                prev_K = K
                continue

            # --- chunk mode ---
            frame_buf.append(frame_rgb)
            id_buf.append(frame_idx)
            frame_idx += 1

            if len(frame_buf) < int(args.chunk_size):
                continue

            t0 = time.time()
            pred = model.inference(
                frame_buf,
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format="mini_npz",
                ref_view_strategy=args.ref_view_strategy,
            )
            infer_ms = (time.time() - t0) * 1000.0

            if pred.extrinsics is None or pred.intrinsics is None:
                print("[ERROR] Model did not return camera parameters; use --mode vo.", file=sys.stderr)
                break

            num_corr = 0
            if prev_chunk_pred is None:
                chunk_sim3 = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            else:
                src, tgt, w = _collect_overlap_correspondences(
                    prev_chunk_pred,
                    pred,
                    overlap=int(args.overlap),
                    stride=int(args.align_stride),
                    max_corr=int(args.align_max_corr),
                    min_depth=float(args.min_depth),
                    max_depth=float(args.max_depth),
                    conf_threshold_ratio=float(args.align_conf_ratio),
                )
                num_corr = int(src.shape[0])
                if num_corr < int(args.align_min_corr):
                    chunk_sim3 = prev_chunk_sim3 if prev_chunk_sim3 is not None else (
                        1.0,
                        np.eye(3, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                    )
                else:
                    s_rel, R_rel, t_rel = robust_sim3_irls(
                        src=src,
                        tgt=tgt,
                        weights=w,
                        delta=float(args.align_delta),
                        max_iters=int(args.align_max_iters),
                    )
                    prev_sim3 = prev_chunk_sim3 if prev_chunk_sim3 is not None else (
                        1.0,
                        np.eye(3, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                    )
                    chunk_sim3 = compose_sim3(prev_sim3, (s_rel, R_rel, t_rel))

            start_k = 0 if prev_chunk_pred is None else int(args.overlap)
            s_cum, R_cum, t_cum = chunk_sim3
            for k in range(start_k, len(frame_buf)):
                depth = pred.depth[k].astype(np.float32)
                rgb_u8 = pred.processed_images[k].astype(np.uint8)
                conf = pred.conf[k].astype(np.float32) if pred.conf is not None else None
                K = pred.intrinsics[k].astype(np.float32)
                w2c = pred.extrinsics[k].astype(np.float32)

                c2w_chunk = invert_w2c_to_c2w_44(w2c)
                pts_chunk, cols = unproject_depth_to_world(
                    depth=depth,
                    rgb_u8=rgb_u8,
                    K=K,
                    c2w=c2w_chunk,
                    stride=args.stride,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    conf=conf,
                    conf_threshold=args.conf_threshold,
                )
                pts_global = apply_sim3(pts_chunk, s_cum, R_cum, t_cum)

                reservoir_filled, reservoir_seen = reservoir_update(
                    pts_global,
                    cols,
                    reservoir_pts=reservoir_pts,
                    reservoir_cols=reservoir_cols,
                    filled=reservoir_filled,
                    seen=reservoir_seen,
                )

                cam_pos_chunk = c2w_chunk[:3, 3].astype(np.float32)
                cam_pos_global = apply_sim3(cam_pos_chunk[None, :], s_cum, R_cum, t_cum)[0]
                cam_positions.append(cam_pos_global)
                cam_path = np.asarray(cam_positions, dtype=np.float32)

                try:
                    rr.set_time_sequence("frame", id_buf[k])
                except Exception:
                    pass

                pts_view = (
                    reservoir_pts[:reservoir_filled]
                    if reservoir_filled < reservoir_pts.shape[0]
                    else reservoir_pts
                )
                col_view = (
                    reservoir_cols[:reservoir_filled]
                    if reservoir_filled < reservoir_cols.shape[0]
                    else reservoir_cols
                )
                rr.log("world/points", rr.Points3D(pts_view, colors=col_view))
                rr.log("world/camera_path", rr.LineStrips3D([cam_path]))

                try:
                    rr.log("world/rgb", rr.Image(rgb_u8))
                except Exception:
                    pass
                try:
                    rr.log("world/depth", rr.DepthImage(depth))
                except Exception:
                    pass
                try:
                    rr.log(
                        "world/stats",
                        rr.TextLog(
                            f"infer={infer_ms:.1f}ms, corr={num_corr}, s={s_cum:.3f}, chunk={len(frame_buf)}"
                        ),
                    )
                except Exception:
                    pass

            prev_chunk_pred = pred
            prev_chunk_sim3 = chunk_sim3

            if args.overlap > 0:
                frame_buf = frame_buf[-args.overlap :]
                id_buf = id_buf[-args.overlap :]
            else:
                frame_buf = []
                id_buf = []
    finally:
        cap.release()


if __name__ == "__main__":
    main()
