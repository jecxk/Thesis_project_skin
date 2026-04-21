"""Ensemble prediction by averaging softmax probabilities from multiple models.

How it works:
    1. Load each of the 4 trained models (EB0, RN50, DN121, Swin-Tiny)
    2. Run each model on the test set → get softmax probabilities (N x 7)
    3. Average the probabilities across all models
    4. Take argmax → final prediction

This "soft voting" approach outperforms individual models because
different architectures tend to make different mistakes, and averaging
smooths out those individual errors.

Usage:
    python -m src.ensemble
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SkinLesionDataset, load_split_dataframes, compute_class_weights
from src.data.transforms import build_transforms
from src.engine.train_eval import evaluate_one_epoch
from src.models.build_model import build_model
from src.models.losses import build_loss
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.utils.metrics import compute_classification_metrics
from src.utils.plots import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_per_class_f1,
    plot_sensitivity_specificity,
)


# ── Models to ensemble ──────────────────────────────────────────────────────

MODELS = [
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


def collect_probabilities(
    model_info: dict,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    device: str,
) -> np.ndarray:
    """Run a single model on the test set and return softmax probabilities.

    Returns:
        probs: (N, num_classes) numpy array of softmax probabilities
    """
    cfg = load_config(model_info['config'])
    class_names = cfg['classes']
    num_classes = len(class_names)
    amp = bool(cfg['train'].get('amp', False) and device == 'cuda')

    eval_tfms = build_transforms(
        cfg['train']['image_size'],
        cfg['augmentation']['eval'],
        is_train=False,
    )
    ds = SkinLesionDataset(test_df, root_dir=cfg['paths']['root_dir'], transform=eval_tfms)
    loader = DataLoader(
        ds,
        batch_size=int(cfg['train']['batch_size']),
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )

    model = build_model(cfg['model']).to(device)
    checkpoint = torch.load(model_info['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_weights = compute_class_weights(train_df['label'].tolist(), num_classes=num_classes)
    criterion = build_loss(cfg['loss'], class_weights=class_weights, device=device)

    _, y_true, _, paths, probs = evaluate_one_epoch(
        model, loader, criterion, device=device, amp=amp,
    )

    del model
    torch.cuda.empty_cache()

    return probs.numpy(), y_true, paths


def main():
    parser = argparse.ArgumentParser(description='Ensemble predictions from multiple models')
    parser.add_argument('--output-dir', type=str, default='outputs/ensemble')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = ensure_dir(Path(args.output_dir))

    print('=' * 70)
    print('  Ensemble Prediction')
    print(f'  Device: {device}')
    print(f'  Output: {output_dir}')
    print('=' * 70)

    # Check which models are available
    available = []
    for m in MODELS:
        if Path(m['checkpoint']).exists():
            available.append(m)
            print(f'  [OK] {m["name"]}: {m["checkpoint"]}')
        else:
            print(f'  [--] {m["name"]}: {m["checkpoint"]} (NOT FOUND, skipping)')

    if len(available) < 2:
        print('Need at least 2 models for ensemble!')
        return

    # Load test data using first model's config
    cfg0 = load_config(available[0]['config'])
    class_names = cfg0['classes']
    train_df, _, test_df = load_split_dataframes(cfg0['paths']['csv_file'])

    # ── Step 1: Collect probabilities from each model ────────────────────
    print(f'\n  Collecting probabilities from {len(available)} models...')
    all_probs = []
    y_true = None
    paths = None

    for m in available:
        print(f'    Running {m["name"]}...')
        probs_m, y_true_m, paths_m = collect_probabilities(m, test_df, train_df, device)
        all_probs.append(probs_m)
        if y_true is None:
            y_true = y_true_m
            paths = paths_m

    # ── Step 2: Average probabilities (soft voting) ──────────────────────
    # Stack: (num_models, N, num_classes) → mean across models → (N, num_classes)
    stacked = np.stack(all_probs, axis=0)       # (M, N, C)
    avg_probs = stacked.mean(axis=0)            # (N, C)
    y_pred = avg_probs.argmax(axis=1).tolist()  # (N,)

    print(f'\n  Ensemble probability shape: {avg_probs.shape}')
    print(f'  Models used: {len(available)}')

    # ── Step 3: Compute metrics ──────────────────────────────────────────
    metrics = compute_classification_metrics(y_true, y_pred, class_names, y_prob=avg_probs)

    save_json(output_dir / 'ensemble_metrics.json', metrics)

    # ── Step 4: Generate plots ───────────────────────────────────────────
    plot_confusion_matrices(
        metrics['confusion_matrix'], class_names,
        output_dir / 'confusion_matrix_raw.png',
        output_dir / 'confusion_matrix_normalized.png',
    )
    plot_per_class_f1(metrics['per_class_metrics'], output_dir / 'per_class_f1.png')
    plot_sensitivity_specificity(metrics['per_class_metrics'], output_dir / 'sensitivity_specificity.png')

    if 'roc_curves' in metrics and metrics['roc_curves']:
        plot_roc_curves(metrics['roc_curves'], output_dir / 'roc_curves.png')

    # ── Step 5: Save predictions ─────────────────────────────────────────
    pred_df = pd.DataFrame({'image_path': paths, 'y_true': y_true, 'y_pred': y_pred})
    pred_df['true_name'] = pred_df['y_true'].map({i: n for i, n in enumerate(class_names)})
    pred_df['pred_name'] = pred_df['y_pred'].map({i: n for i, n in enumerate(class_names)})
    pred_df['correct'] = pred_df['y_true'] == pred_df['y_pred']
    for i, name in enumerate(class_names):
        pred_df[f'prob_{name}'] = avg_probs[:, i].round(6)
    pred_df.to_csv(output_dir / 'ensemble_predictions.csv', index=False)

    # ── Step 6: Print results ────────────────────────────────────────────
    print(f'\n{"=" * 70}')
    print(f'  ENSEMBLE RESULTS ({len(available)} models)')
    print(f'{"=" * 70}')
    print(f'  Models: {", ".join(m["name"] for m in available)}')
    print(f'  Method: Soft Voting (probability averaging)')
    print()
    print(f'  Accuracy:     {metrics["accuracy"]:.4f}')
    print(f'  Macro F1:     {metrics["macro_f1"]:.4f}')
    print(f'  Weighted F1:  {metrics["weighted_f1"]:.4f}')
    print(f'  Cohen Kappa:  {metrics["cohen_kappa"]:.4f}')
    print(f'  MCC:          {metrics["mcc"]:.4f}')
    if 'auc_macro' in metrics:
        print(f'  AUC (macro):  {metrics["auc_macro"]:.4f}')
    print()
    print('  -- Per-Class F1 --')
    for pcm in metrics['per_class_metrics']:
        auc_str = f'  AUC={pcm.get("auc", 0):.3f}' if 'auc' in pcm else ''
        print(f'    {pcm["class_name"]:>6s}: F1={pcm["f1"]:.3f}  '
              f'P={pcm["precision"]:.3f}  R={pcm["recall"]:.3f}{auc_str}  '
              f'(n={pcm["support"]})')

    # ── Step 7: Compare with individual models ───────────────────────────
    print(f'\n  -- Comparison with individual models --')
    print(f'  {"Model":<20s} {"Acc":>8s} {"Macro F1":>10s} {"AUC":>8s}')
    print(f'  {"-"*48}')
    for i, m in enumerate(available):
        # Individual model predictions
        y_pred_i = all_probs[i].argmax(axis=1).tolist()
        met_i = compute_classification_metrics(y_true, y_pred_i, class_names, y_prob=all_probs[i])
        auc_i = met_i.get('auc_macro', 0)
        print(f'  {m["name"]:<20s} {met_i["accuracy"]:>8.4f} {met_i["macro_f1"]:>10.4f} {auc_i:>8.4f}')
    print(f'  {"ENSEMBLE":<20s} {metrics["accuracy"]:>8.4f} {metrics["macro_f1"]:>10.4f} '
          f'{metrics.get("auc_macro", 0):>8.4f}')
    print(f'{"=" * 70}')
    print(f'  Output saved to: {output_dir}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
