import gzip
import lzma
import pickle
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display
from tqdm import tqdm


def format_speedup(fps):
    speedup = fps / 10
    if speedup.is_integer():
        return f"{int(speedup)}x"
    else:
        return f"{speedup:.1f}x"


def annotate_frames_with_speed(frames: np.ndarray, fps: int) -> np.ndarray:
    assert (
        len(frames.shape) == 4
    ), "Frames must be a 4D array (batch, height, width, channels)"

    frames = np.array(
        [
            cv2.putText(
                frame,
                format_speedup(fps),
                (
                    frame.shape[1] - 55,
                    frame.shape[0] - 15,
                ),  # Adjusted position of the text
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                1,  # Font scale (doubled)
                (255, 255, 255),  # Text color (white)
                2,  # Text thickness (doubled)
                cv2.LINE_AA,  # Line type for better rendering
            )
            for frame in frames
        ]
    )

    return frames


def create_mp4_jupyter(
    np_images,
    filename,
    fps=10,
    speed_annotation=False,
):
    if speed_annotation:
        np_images = annotate_frames_with_speed(np_images, fps)

    with imageio.get_writer(filename, fps=fps) as writer:
        for img in np_images:
            writer.append_data(img)
    print(f"File saved as {filename}")
    # Display the video in the Jupyter Notebook
    video_tag = f'<video controls src="{filename}" width="640" height="480"></video>'

    return display(HTML(video_tag))


def mp4_from_pickle_jupyter(
    pickle_path: Union[str, Path],
    filename=None,
    fps=10,
    speed_annotation=False,
    cameras=[1, 2],
):
    ims = extract_numpy_frames(pickle_path, cameras)

    if speed_annotation:
        ims = annotate_frames_with_speed(ims, fps)

    return create_mp4_jupyter(ims, filename, fps)


def mp4_from_data_dict_jupyter(data_dict: dict, filename, fps=10):
    ims = data_to_video(data_dict)
    return create_mp4_jupyter(ims, filename, fps)


def create_mp4(
    np_images: np.ndarray, filename: Union[str, Path], fps=10, verbose=False
) -> None:
    # duration = 1000 / fps
    with imageio.get_writer(filename, fps=fps) as writer:
        for img in tqdm(np_images, disable=not verbose):
            writer.append_data(img)

    if verbose:
        print(f"File saved as {filename}")


def mp4_from_pickle(pickle_path, filename=None, fps=10, cameras=[1, 2]):
    ims = extract_numpy_frames(pickle_path, cameras)
    create_mp4(ims, filename, fps)


def extract_numpy_frames(pickle_path: Union[str, Path], cameras=[1, 2]) -> np.ndarray:
    data = unpickle_data(pickle_path)
    ims = data_to_video(data, cameras)

    return ims


def data_to_video(data: dict, cameras=[1, 2]) -> np.ndarray:
    ims = []

    for camera in cameras:
        ims.append(np.array([o[f"color_image{camera}"] for o in data["observations"]]))

    ims = np.concatenate(ims, axis=2)

    return ims


def unpickle_data(pickle_path: Union[Path, str]):
    pickle_path = Path(pickle_path)
    if pickle_path.suffix == ".gz":
        with gzip.open(pickle_path, "rb") as f:
            return pickle.load(f)
    elif pickle_path.suffix == ".pkl":
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    elif pickle_path.suffix == ".xz":
        with lzma.open(pickle_path, "rb") as f:
            return pickle.load(f)

    raise ValueError(f"Invalid file extension: {pickle_path.suffix}")


def pickle_data(data, pickle_path: Union[Path, str]):
    pickle_path = Path(pickle_path)
    if pickle_path.suffix == ".gz":
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(data, f)
    elif pickle_path.suffix == ".pkl":
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
    elif pickle_path.suffix == ".xz":
        with lzma.open(pickle_path, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Invalid file extension: {pickle_path.suffix}")


def create_in_memory_mp4(np_images, fps=10):
    output = BytesIO()

    writer_options = {"fps": fps}
    writer_options["format"] = "mp4"
    writer_options["codec"] = "libx264"
    writer_options["pixelformat"] = "yuv420p"

    with imageio.get_writer(output, **writer_options) as writer:
        for img in np_images:
            writer.append_data(img)

    output.seek(0)
    return output

def _depth_sequence_to_numpy(depth_images) -> np.ndarray:
    if torch.is_tensor(depth_images):
        depth_np = depth_images.detach().cpu().numpy()
    else:
        depth_np = np.asarray(
            [
                img.detach().cpu().numpy() if torch.is_tensor(img) else img
                for img in depth_images
            ]
        )
    if depth_np.ndim != 3:
        raise ValueError(f"Expected depth sequence with shape [T,H,W], got {depth_np.shape}")
    return depth_np.astype(np.float32, copy=False)


def _prepare_depth_sequence(depth_images):
    depth_np = _depth_sequence_to_numpy(depth_images)
    finite = np.isfinite(depth_np)
    finite_values = depth_np[finite]

    if finite_values.size == 0:
        depth_distance = np.zeros_like(depth_np, dtype=np.float32)
        valid = np.zeros_like(depth_np, dtype=bool)
        return {
            "depth_distance": depth_distance,
            "valid": valid,
            "depth_sign_mode": "as_is",
            "valid_pixel_ratio_global": 0.0,
        }

    non_pos_ratio = float((finite_values <= 0).mean())
    if non_pos_ratio > 0.5:
        depth_distance = -depth_np
        depth_sign_mode = "negated"
    else:
        depth_distance = depth_np
        depth_sign_mode = "as_is"

    valid = np.isfinite(depth_distance) & (depth_distance > 0)
    return {
        "depth_distance": depth_distance,
        "valid": valid,
        "depth_sign_mode": depth_sign_mode,
        "valid_pixel_ratio_global": float(valid.mean()),
    }


def depth2heatmap(depth_images):
    prepared = _prepare_depth_sequence(depth_images)
    depth_np = prepared["depth_distance"]
    valid = prepared["valid"]
    heatmap_frames = []

    if valid.any():
        valid_values = depth_np[valid]
        # Use one global window for the whole sequence to prevent frame-wise color flicker.
        depth_min = np.percentile(valid_values, 1.0)
        depth_max = np.percentile(valid_values, 99.0)
        if depth_max - depth_min < 1e-6:
            depth_max = depth_min + 1e-6
    else:
        depth_min, depth_max = 0.0, 1.0

    for i in range(depth_np.shape[0]):
        img = np.nan_to_num(depth_np[i], nan=0.0, posinf=0.0, neginf=0.0)
        img = np.clip(img, depth_min, depth_max)
        img_norm = (img - depth_min) / (depth_max - depth_min)
        # Invert so near (small depth) -> yellow, far (large depth) -> blue.
        img_norm = 1.0 - img_norm
        img_uint8 = (img_norm * 255).astype(np.uint8)
        img_uint8[~valid[i]] = 0

        color_map = cv2.applyColorMap(img_uint8, cv2.COLORMAP_VIRIDIS)
        color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        heatmap_frames.append(color_map_rgb)

    return heatmap_frames


def analyze_depth_smoothness(depth_images, jump_ratio=0.08):
    prepared = _prepare_depth_sequence(depth_images)
    depth_np = prepared["depth_distance"]
    valid = prepared["valid"]
    valid_values = depth_np[valid]

    if not valid.any():
        return {
            "global_min": 0.0,
            "global_max": 0.0,
            "threshold": 0.0,
            "per_frame": [],
            "n_jumps": 0,
            "n_frames": int(depth_np.shape[0]),
            "depth_sign_mode": prepared["depth_sign_mode"],
            "valid_pixel_ratio_global": prepared["valid_pixel_ratio_global"],
        }

    global_min = float(np.percentile(valid_values, 1.0))
    global_max = float(np.percentile(valid_values, 99.0))
    depth_range = max(global_max - global_min, 1e-6)
    threshold = max(1e-3, depth_range * float(jump_ratio))

    per_frame = []
    n_jumps = 0
    for i in range(depth_np.shape[0]):
        frame_valid = valid[i]
        frame_vals = depth_np[i][frame_valid]
        frame_stats = {
            "frame": i,
            "valid_ratio": float(frame_valid.mean()),
            "depth_mean": float(frame_vals.mean()) if frame_vals.size > 0 else 0.0,
            "depth_p95": float(np.percentile(frame_vals, 95.0)) if frame_vals.size > 0 else 0.0,
            "delta_mean": 0.0,
            "delta_p95": 0.0,
            "delta_max": 0.0,
            "is_jump": False,
        }
        if i > 0:
            overlap = valid[i] & valid[i - 1]
            if overlap.any():
                delta = np.abs(depth_np[i][overlap] - depth_np[i - 1][overlap])
                frame_stats["delta_mean"] = float(delta.mean())
                frame_stats["delta_p95"] = float(np.percentile(delta, 95.0))
                frame_stats["delta_max"] = float(delta.max())
                frame_stats["is_jump"] = frame_stats["delta_p95"] > threshold
        if frame_stats["is_jump"]:
            n_jumps += 1
        per_frame.append(frame_stats)

    return {
        "global_min": global_min,
        "global_max": global_max,
        "threshold": float(threshold),
        "per_frame": per_frame,
        "n_jumps": int(n_jumps),
        "n_frames": int(depth_np.shape[0]),
        "depth_sign_mode": prepared["depth_sign_mode"],
        "valid_pixel_ratio_global": prepared["valid_pixel_ratio_global"],
    }

def render_mp4(ims1, ims2, filename=None):
    # Initialize plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Function to update plot
    def update(num):
        ax1.clear()
        ax2.clear()
        ax1.axis("off")
        ax2.axis("off")

        img_array1 = ims1[num]
        if isinstance(img_array1, torch.Tensor):
            img_array1 = img_array1.squeeze(0).cpu().numpy()

        img_array2 = ims2[num]
        if isinstance(img_array2, torch.Tensor):
            img_array2 = img_array2.squeeze(0).cpu().numpy()

        ax1.imshow(img_array1)
        ax2.imshow(img_array2)

    frame_indices = range(0, len(ims1), 1)

    framerate_hz = 10

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=tqdm(frame_indices),
        interval=1000 // framerate_hz,
    )

    if not filename:
        filename = f"render-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.mp4"

    ani.save(filename)
