"""Webcam-based multi-view mapping demo with Depth Anything 3 and Rerun."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import cv2
import numpy as np
import torch

import rerun as rr

from depth_anything_3.api import DepthAnything3


@dataclass
class FramePrediction:
    """Cache for each frame's RGB, depth, and camera parameters."""

    rgb: np.ndarray  # (H, W, 3) uint8 in RGB order
    depth: np.ndarray  # (H, W) float32
    conf: np.ndarray | None  # (H, W) float32 or None
    intrinsics: np.ndarray  # (3, 3)
    extrinsics: np.ndarray  # (3, 4) or (4, 4)
    timestamp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream webcam frames through Depth Anything 3, estimate camera poses, "
            "and visualize the evolving point cloud in Rerun."
        )
    )
    parser.add_argument(
        "--model-id",
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Hugging Face repo or local path for DepthAnything3.from_pretrained().",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string; defaults to 'cuda' if available else 'cpu'.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index for cv2.VideoCapture (0 = default webcam).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Flip webcam frames horizontally to mimic selfie view.",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=448,
        help="Processing resolution fed to DepthAnything3 (square upper-bound).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help="Number of recent frames sent to the model (controls multi-view context).",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=12,
        help="Maximum number of frame predictions kept in memory and visualized.",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=3,
        help="Start inference only after this many frames have been captured.",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=2,
        help="Run model every N captured frames to control latency.",
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=6,
        help="Subsample stride when converting depth maps to point clouds.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=150_000,
        help="Maximum number of 3D points logged per frame.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="Confidence cutoff (0-1) when filtering depth pixels; set <0 to disable.",
    )
    parser.add_argument(
        "--rerun-app-id",
        default="DepthAnything3-Webcam-Mapping",
        help="Name displayed in the Rerun Viewer title bar.",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="If set, do not auto-launch rerun viewer (attach manually via `rerun --listen`).",
    )
    return parser.parse_args()


def pick_device(user_choice: str | None) -> torch.device:
    if user_choice:
        return torch.device(user_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_matrix44(extr: np.ndarray) -> np.ndarray:
    """Convert a (3x4) or (4x4) extrinsic matrix to homogeneous 4x4."""
    if extr.shape == (4, 4):
        return extr
    mat = np.eye(4, dtype=extr.dtype)
    mat[: extr.shape[0], : extr.shape[1]] = extr
    return mat


def rotation_matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion."""
    m00, m01, m02 = rot[0]
    m10, m11, m12 = rot[1]
    m20, m21, m22 = rot[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat /= norm
    return quat


def depth_to_points(
    frame: FramePrediction,
    stride: int,
    max_points: int,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert depth + color to sampled world-space points."""
    depth = frame.depth
    conf = frame.conf
    rgb = frame.rgb

    h, w = depth.shape
    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth[grid_y, grid_x]

    valid = np.isfinite(z) & (z > 0)
    if conf is not None and conf_threshold >= 0.0:
        conf_sample = conf[grid_y, grid_x]
        valid &= conf_sample >= conf_threshold

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    grid_x = grid_x[valid].astype(np.float32)
    grid_y = grid_y[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    fx = frame.intrinsics[0, 0]
    fy = frame.intrinsics[1, 1]
    cx = frame.intrinsics[0, 2]
    cy = frame.intrinsics[1, 2]

    x_norm = (grid_x - cx) / fx
    y_norm = (grid_y - cy) / fy
    cam_points = np.stack([x_norm * z, y_norm * z, z], axis=1)

    cam_to_world = np.linalg.inv(to_matrix44(frame.extrinsics))
    world_points = (
        (cam_to_world[:3, :3] @ cam_points.T) + cam_to_world[:3, 3:4]
    ).T  # (N, 3)

    colors = rgb[grid_y.astype(int), grid_x.astype(int)]

    if world_points.shape[0] > max_points:
        idx = np.random.choice(world_points.shape[0], size=max_points, replace=False)
        world_points = world_points[idx]
        colors = colors[idx]

    return world_points.astype(np.float32), colors.astype(np.uint8)


def log_world_state(
    store: Dict[int, FramePrediction],
    frame_index: int,
    stride: int,
    max_points: int,
    conf_threshold: float,
) -> None:
    """Send current camera poses and point clouds to Rerun."""
    if not store:
        return
    rr.set_time_sequence("frame", frame_index)
    rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

    for frame_id in sorted(store.keys()):
        data = store[frame_id]
        cam_to_world = np.linalg.inv(to_matrix44(data.extrinsics)).astype(np.float32)
        translation = cam_to_world[:3, 3]
        quat = rotation_matrix_to_quaternion(cam_to_world[:3, :3])
        h, w, _ = data.rgb.shape
        rr.log(
            f"world/cameras/{frame_id}",
            [
                rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=quat)),
                rr.Pinhole(
                    focal_length=(float(data.intrinsics[0, 0]), float(data.intrinsics[1, 1])),
                    principal_point=(float(data.intrinsics[0, 2]), float(data.intrinsics[1, 2])),
                    resolution=(w, h),
                ),
            ],
        )
        points, colors = depth_to_points(data, stride, max_points, conf_threshold)
        if points.size == 0:
            continue
        rr.log(
            f"world/points/{frame_id}",
            rr.Points3D(points, colors=colors),
        )


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    rr.init(args.rerun_app_id, spawn=not args.no_spawn)

    print(f"[INFO] Loading model {args.model_id} on {device}...", file=sys.stderr)
    model = DepthAnything3.from_pretrained(args.model_id).to(device)
    model.eval()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera_index}", file=sys.stderr)
        sys.exit(1)

    frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=args.window)
    predictions: Dict[int, FramePrediction] = {}
    frame_index = 0
    last_inference_time = 0.0

    print("[INFO] Press 'q' in the OpenCV preview window to exit.", file=sys.stderr)
    cv2.namedWindow("Webcam Preview", cv2.WINDOW_NORMAL)

    try:
        while True:
            grabbed, frame_bgr = cap.read()
            if not grabbed:
                print("[WARNING] Failed to grab frame from webcam.", file=sys.stderr)
                break

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_buffer.append((frame_index, frame_rgb))

            should_run = (
                len(frame_buffer) >= max(args.min_views, 2)
                and frame_index % max(args.inference_interval, 1) == 0
            )

            if should_run:
                imgs = [arr for _, arr in frame_buffer]
                start = time.time()
                prediction = model.inference(
                    imgs,
                    process_res=args.process_res,
                    process_res_method="upper_bound_resize",
                    export_dir=None,
                    export_format="mini_npz",
                )
                last_inference_time = time.time() - start

                processed_rgbs = (
                    prediction.processed_images
                    if prediction.processed_images is not None
                    else [None] * len(frame_buffer)
                )
                for (frame_id, raw_rgb), proc_rgb, depth, conf, extr, intr in zip(
                    frame_buffer,
                    processed_rgbs,
                    prediction.depth,
                    prediction.conf if prediction.conf is not None else [None] * len(frame_buffer),
                    prediction.extrinsics if prediction.extrinsics is not None else [None] * len(frame_buffer),
                    prediction.intrinsics if prediction.intrinsics is not None else [None] * len(frame_buffer),
                ):
                    if extr is None or intr is None:
                        continue
                    if proc_rgb is None:
                        h, w = depth.shape
                        aligned_rgb = cv2.resize(raw_rgb, (w, h), interpolation=cv2.INTER_AREA)
                    else:
                        aligned_rgb = proc_rgb
                    predictions[frame_id] = FramePrediction(
                        rgb=aligned_rgb,
                        depth=depth,
                        conf=conf,
                        intrinsics=intr,
                        extrinsics=extr,
                        timestamp=time.time(),
                    )

                while len(predictions) > max(args.max_keyframes, 1):
                    oldest_key = min(predictions.keys())
                    predictions.pop(oldest_key)

                log_world_state(
                    predictions,
                    frame_index=frame_index,
                    stride=args.point_stride,
                    max_points=args.max_points,
                    conf_threshold=args.conf_threshold,
                )

            overlay = frame_bgr.copy()
            if last_inference_time > 0:
                cv2.putText(
                    overlay,
                    f"infer: {last_inference_time*1000:.1f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Webcam Preview", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
