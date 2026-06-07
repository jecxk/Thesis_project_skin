"""Regenerate test_predictions.csv for the metadata-fusion models.

The original training run wrote NaN to the prob_* columns because the AMP
FP16 softmax tensor was rounded to 6 decimals in pandas, exceeding float16's
precision and overflowing to NaN. This script reloads each best.pth and runs
test-set inference in float32 to produce clean probability columns, then
overwrites the per-model test_predictions.csv. Run once; the run_meta_fusion
ensemble + aggregate phases can then re-run cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset_meta import SkinLesionDatasetWithMeta, load_split_with_meta
from src.data.transforms import build_transforms
from src.models.meta_fusion import build_meta_fusion_model
from src.utils.config import load_config


MODELS = {
    'efficientnet_b0_meta': 'src/configs/meta_fusion/efficientnet_b0_meta.yaml',
    'resnet50_meta':        'src/configs/meta_fusion/resnet50_meta.yaml',
    'densenet121_meta':     'src/configs/meta_fusion/densenet121_meta.yaml',
    'swin_tiny_meta':       'src/configs/meta_fusion/swin_tiny_meta.yaml',
}


def regenerate(model_name: str, cfg_path: str) -> None:
    out_dir = ROOT / 'outputs' / 'meta_fusion' / model_name
    ckpt_path = out_dir / 'best.pth'
    if not ckpt_path.exists():
        print(f'  [skip] {model_name}: no best.pth')
        return

    cfg = load_config(ROOT / cfg_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_names = cfg['classes']

    _, _, test_df, _, _, te_feats = load_split_with_meta(
        cfg['paths']['meta_csv'], cfg['paths']['meta_features'])

    eval_tfms = build_transforms(cfg['train']['image_size'],
                                 cfg['augmentation']['eval'], is_train=False)
    test_ds = SkinLesionDatasetWithMeta(test_df, cfg['paths']['root_dir'],
                                        te_feats, transform=eval_tfms)
    loader = DataLoader(test_ds, batch_size=int(cfg['train']['batch_size']),
                        shuffle=False, num_workers=0,
                        pin_memory=(device == 'cuda'))

    cfg['model']['meta_dim'] = int(te_feats.shape[1])
    cfg['model']['num_classes'] = len(class_names)
    model = build_meta_fusion_model(cfg['model']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_probs, all_pred, all_true, all_paths = [], [], [], []
    with torch.no_grad():
        for images, meta, targets, p_batch in loader:
            images = images.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            # IMPORTANT: no autocast → float32 softmax → no NaN.
            logits = model(images, meta).float()
            probs = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            all_probs.append(probs)
            all_pred.extend(probs.argmax(axis=1).tolist())
            all_true.extend(targets.cpu().numpy().tolist())
            all_paths.extend(list(p_batch))

    probs = np.concatenate(all_probs, axis=0)
    assert not np.isnan(probs).any(), 'still NaN — something is off'
    acc = float(np.mean(np.array(all_pred) == np.array(all_true)))

    df = pd.DataFrame({
        'image_path': all_paths,
        'y_true': all_true,
        'y_pred': all_pred,
    })
    df['true_name'] = df['y_true'].map({i: n for i, n in enumerate(class_names)})
    df['pred_name'] = df['y_pred'].map({i: n for i, n in enumerate(class_names)})
    df['correct'] = df['y_true'] == df['y_pred']
    for i, n in enumerate(class_names):
        df[f'prob_{n}'] = probs[:, i].round(6)

    df.to_csv(out_dir / 'test_predictions.csv', index=False)
    print(f'  [fix] {model_name:25s}  acc={acc:.4f}  '
          f'probs ok (min={probs.min():.4f}, max={probs.max():.4f})')


def main() -> None:
    print('Regenerating clean test_predictions.csv for metadata-fusion models')
    for name, cfg in MODELS.items():
        regenerate(name, cfg)
    print('\nDone. Re-run ensemble + aggregate:')
    print('  python scripts/run_meta_fusion.py ensemble')
    print('  python scripts/run_meta_fusion.py aggregate')


if __name__ == '__main__':
    main()
