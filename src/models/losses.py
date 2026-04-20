from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha,
            reduction='none', label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def build_loss(loss_cfg, class_weights=None, device='cpu'):
    name = loss_cfg['name']
    label_smoothing = float(loss_cfg.get('label_smoothing', 0.0))
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if name == 'cross_entropy':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == 'weighted_cross_entropy':
        return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    if name == 'focal':
        return FocalLoss(
            alpha=weight_tensor,
            gamma=float(loss_cfg.get('focal_gamma', 2.0)),
            label_smoothing=label_smoothing,
        )
    raise ValueError(f'Unsupported loss: {name}')
