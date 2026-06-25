"""
Compare trained models side by side on the test set, and run the
cross-dataset generalization check (train on GAN fakes, test on a folder
of diffusion-model fakes, if you've set one up — see README Step 5).

Usage:
    python -m src.evaluate --models baseline transfer
    python -m src.evaluate --models baseline transfer --cross_test data/cross_test_diffusion
"""

import argparse
import os

import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from . import config
from .dataset import get_dataloaders, CappedImageFolder, get_eval_transforms
from .models import build_model, load_checkpoint
from torch.utils.data import DataLoader


def full_metrics(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels, all_probs_fake = [], [], []
    total_loss = 0.0

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
            all_probs_fake.extend(probs[:, 1].cpu().numpy())

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs_fake)
    except ValueError:
        metrics["auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
    return metrics


def compare_models(model_names, cross_test_dir=None):
    device = config.DEVICE
    _, _, test_loader, class_to_idx = get_dataloaders()

    results = {}
    for name in model_names:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] No checkpoint found for '{name}' at {ckpt_path}. Train it first.")
            continue

        model = build_model(name).to(device)
        model, _ = load_checkpoint(model, ckpt_path, map_location=device)

        print(f"\nEvaluating '{name}' on standard test set...")
        metrics = full_metrics(model, test_loader, device)
        results[name] = {"standard_test": metrics}

        print(f"  accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
              f"auc={metrics['auc']:.4f}  precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}")
        print(f"  confusion_matrix (rows=true, cols=pred, order={class_to_idx}): "
              f"{metrics['confusion_matrix']}")

        if cross_test_dir and os.path.isdir(cross_test_dir):
            cross_ds = CappedImageFolder(cross_test_dir, get_eval_transforms(), max_per_class=None)
            cross_loader = DataLoader(cross_ds, batch_size=config.BATCH_SIZE, shuffle=False)
            print(f"Evaluating '{name}' on CROSS-DATASET test set ({cross_test_dir})...")
            cross_metrics = full_metrics(model, cross_loader, device)
            results[name]["cross_test"] = cross_metrics
            print(f"  [cross-dataset] accuracy={cross_metrics['accuracy']:.4f} "
                  f"f1={cross_metrics['f1']:.4f} auc={cross_metrics['auc']:.4f}")
            drop = metrics["accuracy"] - cross_metrics["accuracy"]
            print(f"  >> accuracy drop vs standard test set: {drop:+.4f} "
                  f"(positive = model generalizes worse on unseen generator type)")

    # Save a tidy comparison table
    rows = []
    for name, splits in results.items():
        row = {"model": name}
        for split_name, m in splits.items():
            for k, v in m.items():
                if k == "confusion_matrix":
                    continue
                row[f"{split_name}_{k}"] = v
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        out_path = os.path.join(config.LOG_DIR, "model_comparison.csv")
        df.to_csv(out_path, index=False)
        print(f"\nComparison table saved to {out_path}")
        print(df.to_string(index=False))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trained models on test set")
    parser.add_argument("--models", nargs="+", default=["baseline", "transfer"],
                         help="Model names to evaluate (must have saved checkpoints)")
    parser.add_argument("--cross_test", type=str, default=None,
                         help="Optional path to a cross-dataset test folder "
                              "(e.g. diffusion-generated fakes) for generalization testing")
    args = parser.parse_args()

    compare_models(args.models, cross_test_dir=args.cross_test)
