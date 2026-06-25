"""
Training script.

Usage examples (run from the project root):

    python -m src.train --model baseline
    python -m src.train --model transfer
    python -m src.train --model fft

Each run saves:
  - best checkpoint to outputs/checkpoints/<model>_best.pt
  - a training history CSV to outputs/logs/<model>_history.csv
  - a final metrics summary printed at the end

NOTE: training the FFT model requires a different data pipeline (4-channel
input). For a from-scratch course project, the simplest path is:
  1. Train & compare `baseline` and `transfer` first (standard RGB pipeline,
     this script handles it directly).
  2. For the FFT comparison, use notebooks/03_fft_analysis.ipynb, which
     shows you how to wrap the dataloader with fft_utils.build_rgb_fft_tensor
     and retrain FFTAwareNet. This is kept separate because mixing two
     input pipelines in one script gets messy — the notebook keeps it
     clear and is also better for screenshots in your report.
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from . import config
from .dataset import get_dataloaders, set_seed
from .models import build_model, save_checkpoint


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs_fake = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs_fake.extend(probs[:, 1].cpu().numpy())  # prob of "fake" class index 1

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs_fake)
    except ValueError:
        auc = float("nan")  # happens if a batch/split has only one class present

    return avg_loss, acc, f1, auc


def train_one_model(model_name: str, epochs: int = None, lr: float = config.LEARNING_RATE):
    set_seed()
    device = config.DEVICE
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, class_to_idx = get_dataloaders()
    print(f"Class mapping (folder -> index): {class_to_idx}")
    print(f"Train samples: {len(train_loader.dataset)} | "
          f"Valid samples: {len(valid_loader.dataset)} | "
          f"Test samples: {len(test_loader.dataset)}")

    model = build_model(model_name).to(device)

    if epochs is None:
        epochs = config.EPOCHS_BASELINE if model_name == "baseline" else config.EPOCHS_TRANSFER

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, valid_loader, device, criterion)
        scheduler.step(val_loss)

        elapsed = time.time() - start
        print(f"[{model_name}] Epoch {epoch}/{epochs} "
              f"- train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} "
              f"- val_acc: {val_acc:.4f} - val_f1: {val_f1:.4f} - val_auc: {val_auc:.4f} "
              f"- {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "val_f1": val_f1, "val_auc": val_auc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, ckpt_path, extra={
                "class_to_idx": class_to_idx, "epoch": epoch, "val_acc": val_acc,
            })
            print(f"  -> saved new best checkpoint to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"  -> early stopping (no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
                break

    # write history CSV
    history_path = os.path.join(config.LOG_DIR, f"{model_name}_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to {history_path}")

    # final test-set evaluation using the BEST checkpoint
    from .models import load_checkpoint
    model, _ = load_checkpoint(model, ckpt_path, map_location=device)
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, device, criterion)
    print(f"\n=== FINAL TEST METRICS [{model_name}] ===")
    print(f"loss: {test_loss:.4f} | accuracy: {test_acc:.4f} | f1: {test_f1:.4f} | auc: {test_auc:.4f}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deepfake detector models")
    parser.add_argument("--model", type=str, default="baseline",
                         choices=["baseline", "transfer"],
                         help="Which model to train")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override default epoch count")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    train_one_model(args.model, epochs=args.epochs, lr=args.lr)
