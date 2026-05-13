import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ChangeDetectionDataset(Dataset):
    """
    Dataset for Binary Change Detection on EO-SAR image pairs.

    Loads co-registered pre-event (EO/RGB) and post-event (SAR/grayscale)
    image pairs along with binary change masks.

    Label remapping applied internally:
        Original 0 (Background) -> 0 (No-Change)
        Original 1 (Intact)     -> 0 (No-Change)
        Original 2 (Damaged)    -> 1 (Change)
        Original 3 (Destroyed)  -> 1 (Change)
    """

    def __init__(self, base_path, patch_size=256, augment=False):
        """
        Args:
            base_path (str): Path to split folder containing pre-event,
                             post-event, and target subfolders.
            patch_size (int): Size of random square crop for training.
            augment (bool):   Whether to apply data augmentation.
        """
        self.base_path = base_path
        self.patch_size = patch_size
        self.augment = augment
        self.filenames = sorted(os.listdir(os.path.join(base_path, "target")))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load pre-event EO image (RGB, 3 channels)
        pre = np.array(Image.open(
            os.path.join(self.base_path, "pre-event", fname)))

        # Load post-event SAR image (grayscale, 1 channel)
        post = np.array(Image.open(
            os.path.join(self.base_path, "post-event", fname)))
        if post.ndim == 2:
            post = post[:, :, np.newaxis]  # (H, W) -> (H, W, 1)

        # Load and remap mask
        mask = np.array(Image.open(
            os.path.join(self.base_path, "target", fname)))
        remapped = np.zeros_like(mask)
        remapped[mask == 2] = 1
        remapped[mask == 3] = 1
        mask = remapped

        # Random crop to patch_size
        h, w = mask.shape
        top  = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        pre  = pre [top:top+self.patch_size, left:left+self.patch_size]
        post = post[top:top+self.patch_size, left:left+self.patch_size]
        mask = mask[top:top+self.patch_size, left:left+self.patch_size]

        # Random horizontal flip augmentation
        if self.augment and random.random() > 0.5:
            pre  = np.fliplr(pre ).copy()
            post = np.fliplr(post).copy()
            mask = np.fliplr(mask).copy()

        # Normalise to [0, 1] and convert to tensors
        pre  = torch.tensor(pre,  dtype=torch.float32).permute(2, 0, 1) / 255.0
        post = torch.tensor(post, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)

        # Concatenate EO (3ch) + SAR (1ch) = 4 channel input
        image = torch.cat([pre, post], dim=0)  # (4, H, W)

        return image, mask


def get_dataloaders(config):
    """
    Build and return train, val, and test DataLoaders from config.

    Args:
        config (dict): Parsed config.yaml as a dictionary.

    Returns:
        train_loader, val_loader, test_loader
    """
    patch_size = config["training"]["patch_size"]
    batch_size = config["training"]["batch_size"]

    train_dataset = ChangeDetectionDataset(
        config["data"]["train_path"], patch_size=patch_size, augment=True)
    val_dataset   = ChangeDetectionDataset(
        config["data"]["val_path"],   patch_size=patch_size, augment=False)
    test_dataset  = ChangeDetectionDataset(
        config["data"]["test_path"],  patch_size=patch_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
