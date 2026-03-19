#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except TypeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def _to_numpy_image(image: Any) -> np.ndarray:
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()
    else:
        image = np.asarray(image)

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 4D image, got shape {image.shape}")
        image = image[0]

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")

    if image.shape[-1] == 1:
        image = image[..., 0]

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the first image stored in a rollout pickle.")
    parser.add_argument("path", type=str, help="Path to rollout pickle")
    parser.add_argument(
        "--key",
        type=str,
        default="color_image2",
        help="Observation image key to show, e.g. color_image2 or color_image1",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save the image instead of only showing it",
    )
    args = parser.parse_args()

    data = _load_pickle(Path(args.path))
    observations = data.get("observations", [])
    if not observations:
        raise ValueError("No observations found in pickle.")

    first_obs = observations[0]
    if args.key not in first_obs:
        raise KeyError(f"{args.key} not found in first observation. Available keys: {sorted(first_obs.keys())}")

    image = _to_numpy_image(first_obs[args.key])

    plt.figure(figsize=(8, 6))
    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.title(f"first observation / {args.key}")
    plt.axis("off")

    if args.save_path is not None:
        output_path = Path(args.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        print(f"saved image to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
