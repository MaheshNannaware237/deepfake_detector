"""
Inference helpers shared by the Streamlit app.

Loads trained checkpoints (cached) and runs predictions + Grad-CAM +
FFT visualization for a single uploaded image.
"""

import os

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from . import config
from .models import build_model, load_checkpoint
from .dataset import get_eval_transforms
from .gradcam import GradCAM, overlay_heatmap_on_image
from .fft_utils import compute_fft_magnitude, build_rgb_fft_tensor


_MODEL_CACHE = {}


def build_input_tensor(pil_img: Image.Image, model_name: str):
    """
    Builds the correctly-shaped input tensor for the given model:
      - baseline / transfer -> 3-channel RGB tensor
      - fft                 -> 4-channel RGB+FFT-magnitude tensor
    This keeps every caller (predict_image, Grad-CAM) consistent so a
    model never silently receives the wrong number of channels.
    """
    if model_name == "fft":
        return build_rgb_fft_tensor(
            pil_img, img_size=config.IMG_SIZE, mean=config.MEAN, std=config.STD
        ).to(config.DEVICE)
    else:
        tfm = get_eval_transforms()
        return tfm(pil_img.convert("RGB")).unsqueeze(0).to(config.DEVICE)


def available_checkpoints():
    """Returns list of model names that have a saved checkpoint."""
    available = []
    for name in ["baseline", "transfer", "fft"]:
        path = os.path.join(config.CHECKPOINT_DIR, f"{name}_best.pt")
        if os.path.exists(path):
            available.append(name)
    return available


def load_model_cached(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found for '{model_name}'. Train it first with "
            f"`python -m src.train --model {model_name}`."
        )

    model = build_model(model_name).to(config.DEVICE)
    model, meta = load_checkpoint(model, ckpt_path, map_location=config.DEVICE)
    model.eval()

    _MODEL_CACHE[model_name] = (model, meta)
    return model, meta


def predict_image(pil_img: Image.Image, model_name: str = "transfer"):
    """
    Runs a forward pass and returns:
        label (str), confidence (float 0-1), probs (dict {label: prob})
    """
    model, meta = load_model_cached(model_name)
    class_to_idx = meta.get("class_to_idx", {"fake": 0, "real": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    tensor = build_input_tensor(pil_img, model_name)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    label = idx_to_class.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx])
    prob_dict = {idx_to_class.get(i, str(i)): float(p) for i, p in enumerate(probs)}

    return label, confidence, prob_dict, tensor


def generate_gradcam_overlay(pil_img: Image.Image, model_name: str = "transfer"):
    """
    Returns a PIL image with the Grad-CAM heatmap overlaid, plus the
    predicted label/confidence.
    """
    model, meta = load_model_cached(model_name)
    class_to_idx = meta.get("class_to_idx", {"fake": 0, "real": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    tensor = build_input_tensor(pil_img, model_name)

    cam_tool = GradCAM(model, target_layer=model.get_last_conv_layer())
    heatmap, target_class = cam_tool.generate(tensor)
    overlay = overlay_heatmap_on_image(pil_img, heatmap)

    label = idx_to_class.get(target_class, str(target_class))
    return overlay, label


def generate_fft_visual(pil_img: Image.Image, size: int = 128):
    """
    Returns a normalized (size, size) array suitable for st.image display
    (the log-magnitude FFT spectrum).
    """
    mag = compute_fft_magnitude(pil_img, size=size)
    return (mag * 255).astype(np.uint8)