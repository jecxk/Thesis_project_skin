"""Image + metadata late-fusion model for skin lesion classification.

Combines a timm backbone (image stream) with a small MLP encoder on the
HAM10000 patient metadata vector (age + sex + localization). The two
representations are concatenated and fed to a single linear classifier,
following the Pacheco & Krohling 2020 ``MetaBlock`` late-fusion design.

The metadata vector layout must match the one produced by
``scripts/parse_ham10000_metadata.py``:

    [ age_standardised (1) | sex_onehot (3) | localization_onehot (15) ]
    → 19 dimensions in total.

Usage
-----
    from src.models.meta_fusion import MetaFusionModel

    model = MetaFusionModel(
        backbone_name='tf_efficientnet_b0',
        meta_dim=19,
        num_classes=7,
        dropout=0.3,
    )
    logits = model(image_tensor, meta_tensor)
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class MetaEncoder(nn.Module):
    """Small MLP that lifts the 19-d metadata vector into a learned space."""

    def __init__(self, in_dim: int = 19, hidden: int = 64, out_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaFusionModel(nn.Module):
    """Image backbone + metadata MLP, concatenated before the final classifier.

    Args:
        backbone_name: any timm model identifier.
        meta_dim: dimensionality of the metadata vector (default 19).
        num_classes: number of output classes (default 7).
        dropout: dropout probability before the final classifier.
        meta_hidden / meta_out: MetaEncoder MLP sizes.
        freeze_backbone: if True, the backbone is frozen and only the
            meta encoder + classifier are trained.
    """

    def __init__(
        self,
        backbone_name: str,
        meta_dim: int = 19,
        num_classes: int = 7,
        dropout: float = 0.3,
        meta_hidden: int = 64,
        meta_out: int = 32,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        # num_classes=0 tells timm to drop the head and return pooled features.
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = self.backbone.num_features
        self.meta_encoder = MetaEncoder(
            in_dim=meta_dim, hidden=meta_hidden, out_dim=meta_out,
            dropout=min(0.1, dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim + meta_out, num_classes)

        self.feat_dim = feat_dim
        self.meta_out = meta_out

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(image)            # (B, feat_dim)
        meta_feat = self.meta_encoder(meta)        # (B, meta_out)
        combined = torch.cat([img_feat, meta_feat], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined)


def build_meta_fusion_model(model_cfg: dict) -> MetaFusionModel:
    """Same shape as build_model(), but expects a ``meta`` sub-config too."""
    return MetaFusionModel(
        backbone_name=model_cfg['name'],
        meta_dim=int(model_cfg.get('meta_dim', 19)),
        num_classes=int(model_cfg.get('num_classes', 7)),
        dropout=float(model_cfg.get('dropout', 0.3)),
        meta_hidden=int(model_cfg.get('meta_hidden', 64)),
        meta_out=int(model_cfg.get('meta_out', 32)),
        freeze_backbone=bool(model_cfg.get('freeze_backbone', False)),
    )
