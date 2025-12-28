"""Live webcam streaming to Rerun with Depth Anything 3 (weighted alignment benchmark).

This script mirrors da3_webcam_rerun.py, but aligns overlapping chunks with
`weighted_align_point_maps` from da3_streaming for benchmarking:
- Capture RGB frames from a webcam (OpenCV).
- Run Depth Anything 3 inference per-frame to get depth (+ intrinsics when available).
- Estimate chunk-to-chunk Sim3 via weighted_align_point_maps.
- Unproject depth to a colored point cloud and transform it into a global frame.
- Stream the camera trajectory and point cloud to Rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from loop_utils.config_utils import load_config
from loop_utils.sim3utils import precompute_scale_chunks_with_depth, weighted_align_point_maps

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
        "--align-config",
        default="configs/base_config.yaml",
        help="Alignment config (da3_streaming style).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=12,
        help="Number of frames per DA3 multi-view inference.",
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
        help="Pixel stride for downsampling overlap maps (weighted alignment).",
    )
    parser.add_argument(
        "--align-max-corr",
        type=int,
        default=50_000,
        help="Unused in weighted mode (kept for CLI parity).",
    )
    parser.add_argument(
        "--align-delta",
        type=float,
        default=0.1,
        help="IRLS delta override for weighted alignment.",
    )
    parser.add_argument(
        "--align-max-iters",
        type=int,
        default=15,
        help="IRLS iterations override for weighted alignment.",
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


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = isinstance(depth, np.ndarray)
    if input_is_numpy:
        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)
        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    n, h, w = depth_tensor.shape
    device = depth_tensor.device

    u = torch.arange(w, device=device).float().view(1, 1, w, 1).expand(n, h, w, 1)
    v = torch.arange(h, device=device).float().view(1, h, 1, 1).expand(n, h, w, 1)
    ones = torch.ones((n, h, w, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(n, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        return point_cloud_world.cpu().numpy()
    return point_cloud_world


def downsample_for_alignment(
    depth: np.ndarray,
    conf: Optional[np.ndarray],
    intrinsics: np.ndarray,
    stride: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if stride <= 1:
        return depth, conf, intrinsics
    depth_ds = depth[:, ::stride, ::stride]
    conf_ds = conf[:, ::stride, ::stride] if conf is not None else None
    intr_ds = intrinsics.copy()
    intr_ds[:, 0, 0] /= float(stride)
    intr_ds[:, 1, 1] /= float(stride)
    intr_ds[:, 0, 2] /= float(stride)
    intr_ds[:, 1, 2] /= float(stride)
    return depth_ds, conf_ds, intr_ds


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

    align_config_path = Path(args.align_config)
    if not align_config_path.is_file():
        alt_path = Path(__file__).resolve().parent / args.align_config
        if alt_path.is_file():
            align_config_path = alt_path
    if not align_config_path.is_file():
        raise SystemExit(f"[ERROR] Align config not found: {args.align_config}")
    align_config = load_config(str(align_config_path))
    align_config["Model"]["IRLS"]["delta"] = float(args.align_delta)
    align_config["Model"]["IRLS"]["max_iters"] = int(args.align_max_iters)
    if not isinstance(align_config["Model"]["IRLS"]["tol"], str):
        align_config["Model"]["IRLS"]["tol"] = str(align_config["Model"]["IRLS"]["tol"])

    device = pick_device(args.device)
    print(f"[INFO] Loading model on {device}...", file=sys.stderr)
    model = load_da3_model(args, device)

    rr.init("da3_webcam_stream_weighted")
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
                print("[ERROR] Model did not return camera parameters.", file=sys.stderr)
                break

            pred.depth = np.array(pred.depth, dtype=np.float32)
            pred.intrinsics = np.array(pred.intrinsics, dtype=np.float32)
            pred.extrinsics = np.array(pred.extrinsics, dtype=np.float32)
            if pred.processed_images is not None:
                pred.processed_images = np.array(pred.processed_images, dtype=np.uint8)
            if pred.conf is not None:
                pred.conf = np.array(pred.conf, dtype=np.float32) - 1.0

            num_corr = 0
            if prev_chunk_pred is None:
                chunk_sim3 = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            else:
                overlap = int(args.overlap)
                prev_depth = prev_chunk_pred.depth[-overlap:]
                curr_depth = pred.depth[:overlap]
                prev_intr = prev_chunk_pred.intrinsics[-overlap:]
                curr_intr = pred.intrinsics[:overlap]
                prev_extr = prev_chunk_pred.extrinsics[-overlap:]
                curr_extr = pred.extrinsics[:overlap]

                if prev_chunk_pred.conf is not None:
                    prev_conf = prev_chunk_pred.conf[-overlap:]
                else:
                    prev_conf = np.ones_like(prev_depth, dtype=np.float32)
                if pred.conf is not None:
                    curr_conf = pred.conf[:overlap]
                else:
                    curr_conf = np.ones_like(curr_depth, dtype=np.float32)

                prev_depth, prev_conf, prev_intr = downsample_for_alignment(
                    prev_depth, prev_conf, prev_intr, int(args.align_stride)
                )
                curr_depth, curr_conf, curr_intr = downsample_for_alignment(
                    curr_depth, curr_conf, curr_intr, int(args.align_stride)
                )

                point_map_prev = depth_to_point_cloud_vectorized(
                    prev_depth,
                    prev_intr,
                    prev_extr,
                    device=device if device.type == "cuda" else None,
                )
                point_map_curr = depth_to_point_cloud_vectorized(
                    curr_depth,
                    curr_intr,
                    curr_extr,
                    device=device if device.type == "cuda" else None,
                )

                conf_threshold = float(min(np.median(prev_conf), np.median(curr_conf))) * float(
                    args.align_conf_ratio
                )
                mask = (prev_conf > conf_threshold) & (curr_conf > conf_threshold)
                num_corr = int(mask.sum())
                if num_corr < int(args.align_min_corr):
                    chunk_sim3 = prev_chunk_sim3 if prev_chunk_sim3 is not None else (
                        1.0,
                        np.eye(3, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                    )
                else:
                    precompute_scale = None
                    if align_config["Model"]["align_method"] == "scale+se3":
                        scale_factor_return, _, _ = precompute_scale_chunks_with_depth(
                            prev_depth,
                            prev_conf,
                            curr_depth,
                            curr_conf,
                            conf_threshold=conf_threshold,
                            conf_scale=align_config["Model"]["depth_threshold"],
                        )
                        precompute_scale = scale_factor_return[0]
                    try:
                        s_rel, R_rel, t_rel = weighted_align_point_maps(
                            point_map_prev,
                            prev_conf,
                            point_map_curr,
                            curr_conf,
                            conf_threshold=conf_threshold,
                            config=align_config,
                            precompute_scale=precompute_scale,
                        )
                    except ValueError:
                        chunk_sim3 = prev_chunk_sim3 if prev_chunk_sim3 is not None else (
                            1.0,
                            np.eye(3, dtype=np.float32),
                            np.zeros(3, dtype=np.float32),
                        )
                    else:
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
