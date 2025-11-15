#!/usr/bin/env python3
"""Run Depth Anything 3 on webcam frames and visualize predicted depth."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate depth from webcam frames with Depth Anything 3."
    )
    parser.add_argument(
        "--model-id",
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Hugging Face repo or local path passed to DepthAnything3.from_pretrained.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index passed to cv2.VideoCapture (use 0 for the default webcam).",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Processing resolution passed to model.inference.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (defaults to cuda if available, otherwise cpu).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror frames horizontally to match selfie-view webcams.",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=960,
        help="Resize webcam frames before inference to keep latency manageable.",
    )
    return parser.parse_args()


def pick_device(user_choice: str | None) -> torch.device:
    if user_choice:
        return torch.device(user_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert a depth map to a colored image using percentile clipping."""
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    lo = np.percentile(finite_depth, 2.0)
    hi = np.percentile(finite_depth, 98.0)
    if hi <= lo:
        hi = lo + 1e-6

    depth_norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    depth_norm = 1.0 - depth_norm  # nearer regions appear warmer (red) in the colormap
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)


def prepare_frame(frame: np.ndarray, target_edge: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resize frame for inference while keeping the original for visualization."""
    original = frame.copy()
    if target_edge <= 0:
        return original, original

    h, w = original.shape[:2]
    scale = target_edge / max(h, w)
    if scale >= 1.0:
        return original, original

    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(original, new_size, interpolation=cv2.INTER_AREA)
    return original, resized


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    print(f"[INFO] Loading model {args.model_id} on {device}...", file=sys.stderr)
    model = DepthAnything3.from_pretrained(args.model_id).to(device)
    model.eval()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera_index}", file=sys.stderr)
        sys.exit(1)

    window_name = "Depth Anything 3 Webcam Depth"
    print("[INFO] Press 'q' to exit.", file=sys.stderr)

    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                print("[WARNING] Failed to grab frame from webcam.", file=sys.stderr)
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            original_frame, inference_frame = prepare_frame(frame, args.max_edge)
            rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)

            start = time.time()
            prediction = model.inference(
                [rgb_frame],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format="mini_npz",
            )
            elapsed = (time.time() - start) * 1000.0  # ms

            depth_map = prediction.depth[0]
            depth_vis = colorize_depth(depth_map)
            depth_vis = cv2.resize(depth_vis, (original_frame.shape[1], original_frame.shape[0]))

            side_by_side = np.hstack((original_frame, depth_vis))
            cv2.putText(
                side_by_side,
                f"{elapsed:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, side_by_side)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
