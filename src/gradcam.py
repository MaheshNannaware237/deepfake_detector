"""
Grad-CAM implementation (no extra dependency on pytorch-grad-cam, written
manually so you understand exactly what it does — useful for your viva).

Grad-CAM highlights which regions of the input image most influenced the
model's prediction, by looking at the gradients flowing into the last
convolutional layer.

Usage:
    cam = GradCAM(model, target_layer=model.get_last_conv_layer())
    heatmap = cam.generate(input_tensor, target_class=1)  # 1 = fake
    overlay = overlay_heatmap_on_image(pil_img, heatmap)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        input_tensor: (1, C, H, W), already on the correct device.
        target_class: index to explain. If None, uses the predicted class.
        Returns a (H, W) numpy array in [0, 1] — the same spatial size as
        the input image.
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)  # (1, num_classes)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global-average-pool the gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # resize CAM to match input image spatial size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))

        return cam, target_class


def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    pil_img: original PIL image (any size; will be resized to heatmap size)
    heatmap: (H, W) array in [0, 1]
    Returns a PIL image with the heatmap overlaid in a jet colormap.
    """
    h, w = heatmap.shape
    base = np.array(pil_img.convert("RGB").resize((w, h))).astype(np.float32) / 255.0

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = (1 - alpha) * base + alpha * colored
    overlay = np.clip(overlay, 0, 1)
    overlay_img = Image.fromarray(np.uint8(overlay * 255))
    return overlay_img
