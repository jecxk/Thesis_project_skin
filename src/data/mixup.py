"""Mixup and CutMix data augmentation for training regularization.

References:
    - Mixup: https://arxiv.org/abs/1710.09412
    - CutMix: https://arxiv.org/abs/1905.04899
"""
from __future__ import annotations

import numpy as np
import torch


def mixup_data(images: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2):
    """Apply Mixup augmentation.

    Args:
        images: (B, C, H, W) batch of images
        targets: (B,) batch of integer labels
        alpha: Beta distribution parameter

    Returns:
        mixed_images, targets_a, targets_b, lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_images, targets_a, targets_b, lam


def cutmix_data(images: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix augmentation.

    Args:
        images: (B, C, H, W) batch of images
        targets: (B,) batch of integer labels
        alpha: Beta distribution parameter

    Returns:
        mixed_images, targets_a, targets_b, lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    _, _, H, W = images.shape

    # Sample bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda based on actual area
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)

    targets_a = targets
    targets_b = targets[index]
    return mixed_images, targets_a, targets_b, lam


def mixup_criterion(criterion, outputs, targets_a, targets_b, lam):
    """Compute mixed loss for Mixup/CutMix.

    Args:
        criterion: loss function (e.g., CrossEntropyLoss)
        outputs: model logits
        targets_a: original targets
        targets_b: shuffled targets
        lam: mixing coefficient
    """
    return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
