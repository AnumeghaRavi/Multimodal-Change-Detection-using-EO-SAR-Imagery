import os
import argparse
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import yaml

from model import UNetWithResNet


def evaluate_full_images(model, data_path, device):
    """
    Evaluate model on full-resolution images (resized to 512x512).
    More reliable than patch-based evaluation for final metrics.
    """
    model.eval()
    all_preds, all_masks = [], []
    filenames = sorted(os.listdir(os.path.join(data_path, "target")))

    with torch.no_grad():
        for fname in filenames:
            # Load images
            pre  = np.array(Image.open(
                os.path.join(data_path, "pre-event",  fname)))
            post = np.array(Image.open(
                os.path.join(data_path, "post-event", fname)))
            if post.ndim == 2:
                post = post[:, :, np.newaxis]
            mask = np.array(Image.open(
                os.path.join(data_path, "target", fname)))

            # Remap labels
            remapped = np.zeros_like(mask)
            remapped[mask == 2] = 1
            remapped[mask == 3] = 1
            mask = remapped

            # Resize to 512x512
            pre_img  = Image.fromarray(pre).resize((512, 512))
            post_img = Image.fromarray(
                post.squeeze()).resize((512, 512))
            mask_img = Image.fromarray(mask).resize(
                (512, 512), Image.NEAREST)

            # To tensor
            pre_t  = torch.tensor(
                np.array(pre_img),  dtype=torch.float32).permute(2,0,1) / 255.0
            post_t = torch.tensor(
                np.array(post_img), dtype=torch.float32).unsqueeze(0) / 255.0
            image  = torch.cat([pre_t, post_t], dim=0).unsqueeze(0).to(device)

            output = model(image)
            pred   = torch.argmax(output, dim=1).squeeze(0)

            all_preds.append(pred.cpu().numpy().flatten())
            all_masks.append(np.array(mask_img).flatten())

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    tp = ((all_preds == 1) & (all_masks == 1)).sum()
    fp = ((all_preds == 1) & (all_masks == 0)).sum()
    fn = ((all_preds == 0) & (all_masks == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)

    return precision, recall, f1, iou, all_preds, all_masks


def plot_confusion_matrix(all_masks, all_preds, save_path):
    cm = confusion_matrix(all_masks, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No-Change', 'Change'],
                yticklabels=['No-Change', 'Change'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")


def main(data_path, weights_path, config_path, output_dir="results"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = UNetWithResNet(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"]
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Model loaded successfully.")

    # Evaluate
    print(f"\nEvaluating on: {data_path}")
    precision, recall, f1, iou, preds, masks = evaluate_full_images(
        model, data_path, device)

    print(f"\n===== RESULTS =====")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"IoU       : {iou:.4f}")

    # Confusion matrix
    plot_confusion_matrix(
        masks, preds,
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate change detection model")
    parser.add_argument("--data_path",   type=str, required=True,
                        help="Path to data split (e.g. data/test)")
    parser.add_argument("--weights",     type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config",      type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir",  type=str, default="results",
                        help="Directory to save evaluation outputs")
    args = parser.parse_args()
    main(args.data_path, args.weights, args.config, args.output_dir)
