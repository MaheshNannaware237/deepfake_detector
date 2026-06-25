"""
Central configuration for the Deepfake/AI-Generated Image Detector project.
Edit paths and hyperparameters here rather than scattering magic numbers
through the codebase.
"""

import os

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")  # tiny synthetic set for smoke-testing

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")
FFT_DIR = os.path.join(OUTPUT_DIR, "fft")

for d in [CHECKPOINT_DIR, LOG_DIR, GRADCAM_DIR, FFT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
CLASS_NAMES = ["real", "fake"]  # index 0 = real, index 1 = fake
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Image / preprocessing settings
# ---------------------------------------------------------------------------
IMG_SIZE = 128          # 128x128 keeps CPU training time reasonable on i5/16GB
                        # bump to 224 later if you move to a GPU or have more time
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats — fine for natural face images
STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 2          # raise if you have more CPU cores; keep modest on i5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS_BASELINE = 12
EPOCHS_TRANSFER = 8      # transfer learning converges faster
EARLY_STOP_PATIENCE = 4

# Subset sizes recommended when using the full 140k dataset on a CPU-only
# machine. None = use everything found in the folder.
MAX_TRAIN_IMAGES_PER_CLASS = 6000
MAX_VALID_IMAGES_PER_CLASS = 1200
MAX_TEST_IMAGES_PER_CLASS = 1200

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None  # torch not installed yet — fine for data-prep scripts,
                    # but training/inference scripts require torch installed

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
