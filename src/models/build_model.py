from __future__ import annotations

import timm


def build_model(model_cfg: dict):
    model = timm.create_model(
        model_cfg['name'],
        pretrained=bool(model_cfg.get('pretrained', True)),
        num_classes=int(model_cfg['num_classes']),
        drop_rate=float(model_cfg.get('dropout', 0.0)),
    )
    return model
