"""
Dataset loading utilities.

Expects an ImageFolder-style layout:

    data/train/real/*.jpg
    data/train/fake/*.jpg
    data/valid/real/*.jpg
    data/valid/fake/*.jpg
    data/test/real/*.jpg
    data/test/fake/*.jpg

This matches the structure of the Kaggle "140k Real and Fake Faces" dataset
once you rename/merge its folders accordingly (see README.md, Step 2).
"""

import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from . import config


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transforms(img_size: int = config.IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


def get_eval_transforms(img_size: int = config.IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


# ---------------------------------------------------------------------------
# Subset helper — caps how many images per class get used, so training
# stays CPU-friendly even if the full 140k dataset is dropped into data/.
# ---------------------------------------------------------------------------
class CappedImageFolder(Dataset):
    def __init__(self, root, transform, max_per_class=None):
        self.base = ImageFolder(root=root, transform=transform)
        self.class_to_idx = self.base.class_to_idx  # {'fake': 0, 'real': 1} alphabetical

        if max_per_class is None:
            self.indices = list(range(len(self.base)))
        else:
            by_class = {}
            for idx, (_, label) in enumerate(self.base.samples):
                by_class.setdefault(label, []).append(idx)
            self.indices = []
            rng = random.Random(config.SEED)
            for label, idxs in by_class.items():
                rng.shuffle(idxs)
                self.indices.extend(idxs[:max_per_class])
            rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        return self.base[real_idx]


def get_dataloaders(batch_size: int = config.BATCH_SIZE, img_size: int = config.IMG_SIZE):
    """
    Returns train_loader, valid_loader, test_loader, class_to_idx
    """
    train_ds = CappedImageFolder(
        config.TRAIN_DIR, get_train_transforms(img_size), config.MAX_TRAIN_IMAGES_PER_CLASS
    )
    valid_ds = CappedImageFolder(
        config.VALID_DIR, get_eval_transforms(img_size), config.MAX_VALID_IMAGES_PER_CLASS
    )
    test_ds = CappedImageFolder(
        config.TEST_DIR, get_eval_transforms(img_size), config.MAX_TEST_IMAGES_PER_CLASS
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=False
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=False
    )

    return train_loader, valid_loader, test_loader, train_ds.class_to_idx


def load_single_image(path: str, img_size: int = config.IMG_SIZE):
    """Load and transform a single image for inference. Returns a (1, C, H, W) tensor."""
    img = Image.open(path).convert("RGB")
    tfm = get_eval_transforms(img_size)
    tensor = tfm(img).unsqueeze(0)
    return tensor, img
