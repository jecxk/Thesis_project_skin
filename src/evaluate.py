from __future__ import annotations

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg['output_dir'])
    class_names = cfg['classes']
    num_classes = len(class_names)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = bool(cfg['train'].get('amp', False) and device == 'cuda')

    train_df, val_df, test_df = load_split_dataframes(cfg['paths']['csv_file'])
    root_dir = cfg['paths']['root_dir']
    target_df = val_df if args.split == 'val' else test_df

    eval_tfms = build_transforms(cfg['train']['image_size'], cfg['augmentation']['eval'], is_train=False)
    ds = SkinLesionDataset(target_df, root_dir=root_dir, transform=eval_tfms)
    loader = DataLoader(
        ds,
        batch_size=int(cfg['train']['batch_size']),
        shuffle=False,
        num_workers=int(cfg['train']['num_workers']),
        pin_memory=(device == 'cuda'),
    )

    class_weights = compute_class_weights(train_df['label'].tolist(), num_classes=num_classes)
    model = build_model(cfg['model']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = build_loss(cfg['loss'], class_weights=class_weights, device=device)
    loss, y_true, y_pred, paths, probs = evaluate_one_epoch(model, loader, criterion, device=device, amp=amp)

    probs_np = probs.numpy()
    metrics = compute_classification_metrics(y_true, y_pred, class_names, y_prob=probs_np)
    metrics['loss'] = float(loss)

    save_json(output_dir / f'{args.split}_metrics_from_eval.json', metrics)

    plot_confusion_matrices(
       metrics['confusion_matrix'],
       class_names,
       output_dir / f'{args.split}_confusion_matrix_raw_from_eval.png',
       output_dir / f'{args.split}_confusion_matrix_normalized_from_eval.png',
    )

    plot_per_class_f1(
        metrics['per_class_metrics'],
        output_dir / f'{args.split}_per_class_f1_from_eval.png',
    )

    plot_sensitivity_specificity(
        metrics['per_class_metrics'],
        output_dir / f'{args.split}_sensitivity_specificity_from_eval.png',
    )

    if 'roc_curves' in metrics and metrics['roc_curves']:
        plot_roc_curves(
            metrics['roc_curves'],
            output_dir / f'{args.split}_roc_curves_from_eval.png',
        )

    # Predictions with probabilities
    pred_df = pd.DataFrame({'image_path': paths, 'y_true': y_true, 'y_pred': y_pred})
    pred_df['true_name'] = pred_df['y_true'].map({i: name for i, name in enumerate(class_names)})
    pred_df['pred_name'] = pred_df['y_pred'].map({i: name for i, name in enumerate(class_names)})
    pred_df['correct'] = pred_df['y_true'] == pred_df['y_pred']
    for i, name in enumerate(class_names):
        pred_df[f'prob_{name}'] = probs_np[:, i].round(6)
    pred_df.to_csv(output_dir / f'{args.split}_predictions_from_eval.csv', index=False)

    print(f"\n{'=' * 60}")
    print(f"  {args.split.upper()} Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"  Cohen Kappa:  {metrics['cohen_kappa']:.4f}")
    print(f"  MCC:          {metrics['mcc']:.4f}")
    if 'auc_macro' in metrics:
        print(f"  AUC (macro):  {metrics['auc_macro']:.4f}")
    print()
    for pcm in metrics['per_class_metrics']:
        auc_str = f"  AUC={pcm.get('auc', 0):.3f}" if 'auc' in pcm else ''
        print(f"    {pcm['class_name']:>6s}: F1={pcm['f1']:.3f}  P={pcm['precision']:.3f}  R={pcm['recall']:.3f}{auc_str}  (n={pcm['support']})")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
