"""
Generates a tiny SYNTHETIC placeholder dataset so the full pipeline
(dataloaders -> training -> evaluation -> Streamlit app) can be smoke-tested
immediately, without waiting to download the real 140k-image dataset.

These images are simple procedurally-generated patterns — NOT real faces
and NOT real GAN output. They exist purely to prove the code runs end to
end. Replace data/train, data/valid, data/test with the real dataset
before drawing any actual conclusions (see README.md).

Usage:
    python -m src.make_sample_data
"""

import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import config


def _make_real_like(size=128, seed=0):
    """Smooth, organic blob pattern with soft gradients — stands in for
    a 'natural photo' style image (smooth low-frequency content)."""
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)

    base = rng.randint(60, 200, size=3)
    for y in range(size):
        shade = base + (rng.randn(3) * 5)
        shade = np.clip(shade, 0, 255).astype(int)
        draw.line([(0, y), (size, y)], fill=tuple(shade))

    # a few soft blobs (eyes/nose-like shapes) with blur
    for _ in range(4):
        x, y = rng.randint(20, size - 20, size=2)
        r = rng.randint(8, 20)
        color = tuple(rng.randint(50, 220, size=3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    img = img.filter(ImageFilter.GaussianBlur(radius=2.5))
    return img


def _make_fake_like(size=128, seed=0):
    """Adds faint periodic checkerboard / grid artifacts on top of a
    similar base pattern — stands in for GAN-style upsampling artifacts
    that show up as periodic structure in the frequency domain."""
    img = _make_real_like(size=size, seed=seed)
    arr = np.array(img).astype(np.float32)

    # periodic grid artifact (simulates checkerboard upsampling artifact)
    xv, yv = np.meshgrid(np.arange(size), np.arange(size))
    period = 8
    grid = (np.sin(2 * np.pi * xv / period) * np.sin(2 * np.pi * yv / period))
    grid = (grid * 6.0)[:, :, None]  # subtle amplitude
    arr = np.clip(arr + grid, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def generate(n_per_split=40, img_size=128):
    random.seed(config.SEED)
    splits = {
        "train": n_per_split,
        "valid": max(8, n_per_split // 4),
        "test": max(8, n_per_split // 4),
    }

    base_dirs = {"train": config.TRAIN_DIR, "valid": config.VALID_DIR, "test": config.TEST_DIR}

    seed_counter = 0
    for split, n in splits.items():
        real_dir = os.path.join(base_dirs[split], "real")
        fake_dir = os.path.join(base_dirs[split], "fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        for i in range(n):
            seed_counter += 1
            real_img = _make_real_like(size=img_size, seed=seed_counter)
            real_img.save(os.path.join(real_dir, f"real_{i:04d}.jpg"), quality=90)

            seed_counter += 1
            fake_img = _make_fake_like(size=img_size, seed=seed_counter)
            fake_img.save(os.path.join(fake_dir, f"fake_{i:04d}.jpg"), quality=90)

        print(f"[{split}] wrote {n} real + {n} fake synthetic images")

    print("\nSynthetic placeholder dataset generated.")
    print("This is ONLY for testing that the pipeline runs end-to-end.")
    print("Replace data/train, data/valid, data/test with the real")
    print("'140k Real and Fake Faces' dataset before drawing real conclusions.")


if __name__ == "__main__":
    generate()
