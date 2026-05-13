import os
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

from dataset import get_dataloaders
from model import UNetWithResNet, CombinedLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(preds, targets):
    preds   = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    return precision, recall, f1, iou


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_masks = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs    = model(images)
            total_loss += criterion(outputs, masks).item()
            preds      = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_masks.append(masks)
    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    precision, recall, f1, iou = compute_metrics(all_preds, all_masks)
    return total_loss / len(loader), precision, recall, f1, iou


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(config)

    model     = UNetWithResNet(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"]
    ).to(device)

    criterion = CombinedLoss(config["loss"]["class_weights"], device)
    optimizer = optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        patience=config["optimizer"]["scheduler_patience"],
        factor=config["optimizer"]["scheduler_factor"]
    )

    best_f1   = 0
    save_path = config["training"]["best_model_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | "
          f"{'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'IoU':>6}")
    print("-" * 70)

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, precision, recall, f1, iou = validate(
            model, val_loader, criterion, device)
        scheduler.step(f1)

        saved = ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            saved = "✓ saved"

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
              f"{precision:>9.4f} | {recall:>6.4f} | "
              f"{f1:>6.4f} | {iou:>6.4f} {saved}")

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    print(f"Best model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train change detection model")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    main(args.config)
