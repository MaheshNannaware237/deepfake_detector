"""
Model architectures.

1. BaselineCNN          - a CNN built from scratch (no pretrained weights).
                          Establishes a from-first-principles baseline.
2. TransferLearningNet  - EfficientNet-B0 backbone, fine-tuned.
3. FFTAwareNet          - takes a 4-channel input (RGB + FFT magnitude map)
                          to test whether frequency-domain artifacts help
                          separate real vs GAN-generated faces.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from . import config


# ---------------------------------------------------------------------------
# 1. Baseline CNN (from scratch)
# ---------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """
    Simple 5-block CNN. Input: (B, 3, IMG_SIZE, IMG_SIZE)
    No pretrained weights — this is your "from first principles" model,
    useful to show you understand convolutional architectures before
    leaning on transfer learning.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, img_size: int = config.IMG_SIZE):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),     # img_size / 2
            conv_block(32, 64),    # img_size / 4
            conv_block(64, 128),   # img_size / 8
            conv_block(128, 256),  # img_size / 16
            conv_block(256, 256),  # img_size / 32
        )

        reduced = img_size // 32
        flat_dim = 256 * reduced * reduced

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self):
        """For Grad-CAM: the last conv block's conv layer."""
        return self.features[-1][0]


# ---------------------------------------------------------------------------
# 2. Transfer Learning model (EfficientNet-B0)
# ---------------------------------------------------------------------------
class TransferLearningNet(nn.Module):
    """
    EfficientNet-B0 backbone pretrained on ImageNet, with a new classifier
    head trained on real/fake faces. Backbone is fine-tuned (not frozen) for
    better accuracy, but you can freeze it by setting freeze_backbone=True
    if training time is too long on CPU.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, freeze_backbone: bool = False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        """For Grad-CAM: last conv layer of EfficientNet-B0 features."""
        return self.backbone.features[-1][0]


# ---------------------------------------------------------------------------
# 3. FFT-aware model (4-channel input: RGB + frequency magnitude)
# ---------------------------------------------------------------------------
class FFTAwareNet(nn.Module):
    """
    Same architecture family as BaselineCNN but accepts a 4th input channel
    containing the log-magnitude FFT spectrum of the grayscale image. The
    idea: GAN upsampling layers leave periodic checkerboard artifacts that
    show up clearly in the frequency domain even when invisible to the eye
    in RGB. This model lets you test whether giving the network direct
    access to that information improves detection.

    Use src/fft_utils.py to build the 4-channel tensor before feeding it
    into this model.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, img_size: int = config.IMG_SIZE):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(4, 32),     # 4 channels in: R, G, B, FFT-magnitude
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 256),
        )

        reduced = img_size // 32
        flat_dim = 256 * reduced * reduced

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self):
        return self.features[-1][0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("baseline", "baselinecnn"):
        return BaselineCNN(**kwargs)
    elif name in ("transfer", "transferlearningnet", "efficientnet"):
        return TransferLearningNet(**kwargs)
    elif name in ("fft", "fftaware", "fftawarenet"):
        return FFTAwareNet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")


def save_checkpoint(model: nn.Module, path: str, extra: dict = None):
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str, map_location=None):
    map_location = map_location or config.DEVICE
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return model, payload
