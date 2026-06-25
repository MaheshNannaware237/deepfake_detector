"""
Frequency-domain (FFT) analysis utilities.

GAN generators build images using transposed-convolution / upsampling
layers, which leave faint periodic artifacts ("checkerboard" patterns)
in the image. These are usually invisible in RGB but show up clearly
as bright periodic spikes in the 2D Fourier spectrum.

This module provides:
  - compute_fft_magnitude(): turn an image into its log-magnitude spectrum
  - build_rgb_fft_tensor(): build the 4-channel (R,G,B,FFT) tensor used by
    FFTAwareNet in models.py
  - average_spectrum(): average spectra across many images, useful for
    visually comparing "average real spectrum" vs "average fake spectrum"
    in your report/viva (this is a classic, compelling figure)
"""

import numpy as np
from PIL import Image
import torch


def compute_fft_magnitude(pil_img: Image.Image, size: int = 128) -> np.ndarray:
    """
    Returns a (size, size) float32 array: the log-scaled, normalized
    magnitude spectrum of the grayscale version of the image.
    """
    img = pil_img.convert("L").resize((size, size))
    arr = np.array(img, dtype=np.float32)

    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # log scale to compress the huge dynamic range, then normalize to [0, 1]
    log_mag = np.log1p(magnitude)
    log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)

    return log_mag.astype(np.float32)


def build_rgb_fft_tensor(pil_img: Image.Image, img_size: int = 128,
                          mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Builds the 4-channel tensor (R, G, B, FFT-magnitude) expected by
    FFTAwareNet. Returns shape (1, 4, img_size, img_size).
    """
    rgb = pil_img.convert("RGB").resize((img_size, img_size))
    rgb_arr = np.array(rgb, dtype=np.float32) / 255.0
    for c in range(3):
        rgb_arr[:, :, c] = (rgb_arr[:, :, c] - mean[c]) / std[c]

    fft_mag = compute_fft_magnitude(pil_img, size=img_size)
    fft_mag = (fft_mag - 0.5) / 0.5  # roughly center to [-1, 1]

    stacked = np.dstack([rgb_arr, fft_mag])           # (H, W, 4)
    tensor = torch.from_numpy(stacked).permute(2, 0, 1).float()  # (4, H, W)
    return tensor.unsqueeze(0)  # (1, 4, H, W)


def average_spectrum(image_paths, size: int = 128) -> np.ndarray:
    """
    Computes the average log-magnitude FFT spectrum across a list of image
    paths. Use this once on a batch of real images and once on a batch of
    fake images, then plot side by side — a strong, simple figure for your
    report showing that fakes have a different average frequency signature
    (often visible as extra bright spokes/rings from upsampling artifacts).
    """
    acc = np.zeros((size, size), dtype=np.float64)
    count = 0
    for p in image_paths:
        try:
            img = Image.open(p)
            acc += compute_fft_magnitude(img, size=size)
            count += 1
        except Exception:
            continue
    if count == 0:
        raise ValueError("No valid images found to compute average spectrum.")
    return (acc / count).astype(np.float32)
