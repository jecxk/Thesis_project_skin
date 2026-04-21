"""Test-Time Augmentation (TTA) inference.

Runs inference with several lightweight augmentations (flips, rotations)
and averages the softmax probabilities. Typically improves accuracy by
~0.5-2% without any retraining.

Can be combined with the multi-model ensemble: each model produces
TTA-averaged probabilities, then probabilities are averaged across models.

Usage
-----
# Single model with TTA
python -m src.tta --config src/configs/efficientnet_b0_optimized.yaml \
    --checkpoint "outputs/efficientnet_b0_v3(main_best)/best.pth" \
    --output-dir outputs/tta_eb0

# Ensemble + TTA (use --ensemble flag; ignores --config/--checkpoint)
python -m src.tta --ensemble --output-dir outputs/tta_ensemble
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SkinLesionDataset, load_split_dataframes
from src.data.transforms import build_transforms
from src.models.build_model import build_model
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.utils.metrics import compute_classification_metrics
from src.utils.plots import (
    plot_confusion_matrices,
    plot_per_class_f1,
    plot_roc_curves,
    plot_sensitivity_specificity,
)


# Ensemble model list mirrors src/ensemble.py
ENSEMBLE_MODELS = [
    {
        'name': 'EfficientNet-B0',
        'config': 'src/configs/efficientnet_b0_optimized.yaml',
        'checkpoint': 'outputs/efficientnet_b0_v3(main_best)/best.pth',
    },
    {
        'name': 'ResNet50',
        'config': 'src/configs/resnet50_v3.yaml',
        'checkpoint': 'outputs/resnet50_v3(main)/best.pth',
    },
    {
        'name': 'DenseNet121',
        'config': 'src/configs/densenet121_v3.yaml',
        'checkpoint': 'outputs/densenet121_v3(main)/best.pth',
    },
    {
        'name': 'Swin-Tiny',
        'config': 'src/configs/swin_tiny_v3.yaml',
        'checkpoint': 'outputs/swin_tiny_v3/best.pth',
    },
]


# --------------------------------------------------------------------------- #
#  TTA transforms (applied on tensor batches)
# --------------------------------------------------------------------------- #

TTATransform = Callable[[torch.Tensor], torch.Tensor]


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def _vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def _hvflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2, -1])


def _rot90(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=1, dims=[-2, -1])


def _rot270(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=3, dims=[-2, -1])


TTA_POLICIES: Dict[str, List[Tuple[str, TTATransform]]] = {
    'none': [('identity', _identity)],
    'flip': [
        ('identity', _identity),
        ('hflip', _hflip),
        ('vflip', _vflip),
        ('hvflip', _hvflip),
    ],
    'full': [
        ('identity', _identity),
        ('hflip', _hflip),
        ('vflip', _vflip),
        ('hvflip', _hvflip),
        ('rot90', _rot90),
        ('rot270', _rot270),
    ],
}


# --------------------------------------------------------------------------- #
#  TTA inference for a single model
# --------------------------------------------------------------------------- #

@torch.no_grad()
def tta_predict_model(
    cfg: dict,
    checkpoint_path: str,
    test_df: pd.DataFrame,
    device: str,
    policy: str = 'flip',
) -> Tuple[np.ndarray, List[str]]:
    """Run TTA inference for one model and return averaged probabilities.

    Returns
    -------
    probs: (N, C) numpy array of averaged softmax probabilities.
    paths: list of image paths (aligned with probs).
    """
    class_names = cfg['classes']
    num_classes = len(class_names)
    image_size = int(cfg['train']['image_size'])
    batch_size = int(cfg['train']['batch_size'])
    num_workers = int(cfg['train'].get('num_workers', 0))

    eval_tf = build_transforms(image_size=image_size,
                               aug_cfg=cfg.get('augmentation', {}).get('eval', {}),
                               is_train=False)

    test_ds = SkinLesionDataset(
        df=test_df,
        root_dir=cfg['paths']['root_dir'],
        transform=eval_tf,
    )
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == 'cuda'))

    model = build_model(cfg['model'])
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    amp = bool(cfg['train'].get('amp', False) and device == 'cuda')

    transforms = TTA_POLICIES[policy]

    all_probs = []
    all_paths: List[str] = []

    for images, _labels, paths in tqdm(loader, desc=f'TTA[{policy}]'):
        images = images.to(device, non_blocking=True)
        if isinstance(paths, torch.Tensor):
            paths = paths.tolist()
        all_paths.extend(list(paths))

        probs_accum = None
        for _, fn in transforms:
            x = fn(images)
            if amp:
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(x)
            else:
                logits = model(x)
            probs = F.softmax(logits.float(), dim=1)
            probs_accum = probs if probs_accum is None else probs_accum + probs
        probs_accum = probs_accum / len(transforms)
        all_probs.append(probs_accum.cpu().numpy())

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return np.concatenate(all_probs, axis=0), all_paths


# --------------------------------------------------------------------------- #
#  Reporting helpers
# --------------------------------------------------------------------------- #

def _report(
    probs: np.ndarray,
    y_true: np.ndarray,
    paths: List[str],
    class_names: List[str],
    output_dir: Path,
    tag: str,
) -> Dict:
    y_pred = probs.argmax(axis=1).tolist()
    metrics = compute_classification_metrics(
        y_true=y_true.tolist(),
        y_pred=y_pred,
        class_names=class_names,
        y_prob=probs,
    )
    save_json(output_dir / f'{tag}_metrics.json', metrics)

    # Predictions CSV
    df = pd.DataFrame({
        'image_path': paths,
        'y_true': y_true.tolist(),
        'y_pred': y_pred,
    })
    df['true_name'] = df['y_true'].map({i: n for i, n in enumerate(class_names)})
    df['pred_name'] = df['y_pred'].map({i: n for i, n in enumerate(class_names)})
    df['correct'] = df['y_true'] == df['y_pred']
    for i, name in enumerate(class_names):
        df[f'prob_{name}'] = probs[:, i].round(6)
    df.to_csv(output_dir / f'{tag}_predictions.csv', index=False)

    # Figures
    plot_confusion_matrices(
        cm=metrics['confusion_matrix'],
        class_names=class_names,
        raw_save_path=output_dir / f'{tag}_confusion_matrix_raw.png',
        normalized_save_path=output_dir / f'{tag}_confusion_matrix_normalized.png',
    )
    plot_per_class_f1(metrics['per_class_metrics'],
                      save_path=output_dir / f'{tag}_per_class_f1.png',
                      title=f'{tag.upper()} — Per-class F1')
    plot_sensitivity_specificity(metrics['per_class_metrics'],
                                 save_path=output_dir / f'{tag}_sens_spec.png',
                                 title=f'{tag.upper()} — Sensitivity vs Specificity')
    if 'roc_curves' in metrics:
        plot_roc_curves(metrics['roc_curves'],
                        save_path=output_dir / f'{tag}_roc.png',
                        title=f'{tag.upper()} — ROC Curves')

    return metrics


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ensemble', action='store_true',
                        help='Run TTA on the full 4-model ensemble')
    parser.add_argument('--policy', type=str, default='flip',
                        choices=list(TTA_POLICIES.keys()))
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = ensure_dir(Path(args.output_dir))

    print('=' * 70)
    print('  Test-Time Augmentation (TTA)')
    print(f'  Policy: {args.policy} ({len(TTA_POLICIES[args.policy])} transforms)')
    print(f'  Device: {device}')
    print(f'  Output: {output_dir}')
    print('=' * 70)

    if args.ensemble:
        cfg0 = load_config(ENSEMBLE_MODELS[0]['config'])
        class_names = cfg0['classes']
        _, _, test_df = load_split_dataframes(cfg0['paths']['csv_file'])

        available = [m for m in ENSEMBLE_MODELS if Path(m['checkpoint']).exists()]
        if not available:
            print('No ensemble checkpoints found.')
            return

        print(f'  Ensembling {len(available)} models...')
        model_probs: List[np.ndarray] = []
        paths_ref: List[str] = []
        per_model_metrics: Dict[str, Dict] = {}

        for m in available:
            print(f'\n  → {m["name"]}')
            cfg = load_config(m['config'])
            t0 = time.time()
            probs, paths = tta_predict_model(cfg, m['checkpoint'], test_df, device, args.policy)
            print(f'    time: {time.time() - t0:.1f}s')
            if not paths_ref:
                paths_ref = paths
            model_probs.append(probs)

            # Also report the individual TTA'd model (optional)
            y_true_m = test_df.set_index('image_path').loc[paths, 'label'].to_numpy()
            tag = f'tta_{m["name"].lower().replace("-", "_")}'
            per_model_metrics[m['name']] = _report(
                probs=probs,
                y_true=y_true_m,
                paths=paths,
                class_names=class_names,
                output_dir=output_dir,
                tag=tag,
            )

        avg_probs = np.mean(np.stack(model_probs, axis=0), axis=0)
        y_true = test_df.set_index('image_path').loc[paths_ref, 'label'].to_numpy()
        ens_metrics = _report(
            probs=avg_probs,
            y_true=y_true,
            paths=paths_ref,
            class_names=class_names,
            output_dir=output_dir,
            tag=f'tta_ensemble_{args.policy}',
        )

        summary = {
            'policy': args.policy,
            'n_models': len(available),
            'models': [m['name'] for m in available],
            'ensemble_tta_metrics': {
                'accuracy': ens_metrics['accuracy'],
                'macro_f1': ens_metrics['macro_f1'],
                'weighted_f1': ens_metrics['weighted_f1'],
                'cohen_kappa': ens_metrics['cohen_kappa'],
                'mcc': ens_metrics['mcc'],
                'auc_macro': ens_metrics.get('auc_macro'),
            },
            'per_model_tta_metrics': {
                name: {
                    'accuracy': met['accuracy'],
                    'macro_f1': met['macro_f1'],
                    'auc_macro': met.get('auc_macro'),
                }
                for name, met in per_model_metrics.items()
            },
        }
        save_json(output_dir / 'tta_summary.json', summary)

        print('\n' + '=' * 70)
        print('  TTA-ENSEMBLE RESULTS')
        print(f'  Accuracy   : {ens_metrics["accuracy"]:.4f}')
        print(f'  Macro F1   : {ens_metrics["macro_f1"]:.4f}')
        print(f'  Cohen Kappa: {ens_metrics["cohen_kappa"]:.4f}')
        if 'auc_macro' in ens_metrics:
            print(f'  AUC (macro): {ens_metrics["auc_macro"]:.4f}')
        print('=' * 70)
        return

    # --- Single model path ---
    if not args.config or not args.checkpoint:
        raise ValueError('For single-model TTA, provide --config and --checkpoint')

    cfg = load_config(args.config)
    class_names = cfg['classes']
    _, _, test_df = load_split_dataframes(cfg['paths']['csv_file'])

    probs, paths = tta_predict_model(cfg, args.checkpoint, test_df, device, args.policy)
    y_true = test_df.set_index('image_path').loc[paths, 'label'].to_numpy()

    metrics = _report(
        probs=probs,
        y_true=y_true,
        paths=paths,
        class_names=class_names,
        output_dir=output_dir,
        tag=f'tta_{args.policy}',
    )

    save_json(output_dir / 'tta_summary.json', {
        'policy': args.policy,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'tta_metrics': {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'cohen_kappa': metrics['cohen_kappa'],
            'mcc': metrics['mcc'],
            'auc_macro': metrics.get('auc_macro'),
        },
    })

    print('\n' + '=' * 70)
    print('  SINGLE MODEL + TTA RESULTS')
    print(f'  Accuracy   : {metrics["accuracy"]:.4f}')
    print(f'  Macro F1   : {metrics["macro_f1"]:.4f}')
    if 'auc_macro' in metrics:
        print(f'  AUC (macro): {metrics["auc_macro"]:.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
