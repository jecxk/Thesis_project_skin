from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SkinLesionDataset, load_split_dataframes, compute_class_weights
from src.data.transforms import build_transforms
from src.data.samplers import build_weighted_sampler
from src.engine.train_eval import train_one_epoch, evaluate_one_epoch
from src.models.build_model import build_model
from src.models.losses import build_loss
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.utils.metrics import compute_classification_metrics
from src.utils.plots import (
    plot_confusion_matrices,
    plot_history,
    plot_training_history_detailed,
    plot_roc_curves,
    plot_per_class_f1,
    plot_class_distribution,
    plot_sensitivity_specificity,
)
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()


def build_optimizer(model, cfg):
    opt_name = cfg['optimizer']['name'].lower()
    lr = float(cfg['optimizer']['lr'])
    wd = float(cfg['optimizer'].get('weight_decay', 0.0))
    if opt_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f'Unsupported optimizer: {opt_name}')


def build_scheduler(optimizer, cfg):
    sch_name = cfg['scheduler']['name'].lower()
    total_epochs = int(cfg['train']['epochs'])
    min_lr = float(cfg['scheduler'].get('min_lr', 1e-6))

    if sch_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr,
        )
    if sch_name == 'cosine_warmup':
        warmup_epochs = int(cfg['scheduler'].get('warmup_epochs', 3))
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    if sch_name == 'none':
        return None
    raise ValueError(f'Unsupported scheduler: {sch_name}')


def _get_class_distribution(df: pd.DataFrame, class_names: list) -> dict:
    """Get class distribution from dataframe."""
    counts = df['label'].value_counts().sort_index()
    return {class_names[int(k)]: int(v) for k, v in counts.items()}


def _format_epoch_line(epoch, total_epochs, train_loss, val_loss, val_acc, val_f1, lr, elapsed):
    """Format a single epoch log line for console output."""
    return (
        f"Epoch {epoch:03d}/{total_epochs} │ "
        f"train_loss={train_loss:.4f} │ val_loss={val_loss:.4f} │ "
        f"val_acc={val_acc:.4f} │ val_f1={val_f1:.4f} │ "
        f"lr={lr:.2e} │ {elapsed:.1f}s"
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get('seed', 42)))

    output_dir = ensure_dir(cfg['output_dir'])
    class_names = cfg['classes']
    num_classes = len(class_names)
    total_epochs = int(cfg['train']['epochs'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = bool(cfg['train'].get('amp', False) and device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ── Data ──────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_split_dataframes(cfg['paths']['csv_file'])
    root_dir = cfg['paths']['root_dir']

    train_tfms = build_transforms(cfg['train']['image_size'], cfg['augmentation']['train'], is_train=True)
    eval_tfms = build_transforms(cfg['train']['image_size'], cfg['augmentation']['eval'], is_train=False)

    train_ds = SkinLesionDataset(train_df, root_dir=root_dir, transform=train_tfms)
    val_ds = SkinLesionDataset(val_df, root_dir=root_dir, transform=eval_tfms)
    test_ds = SkinLesionDataset(test_df, root_dir=root_dir, transform=eval_tfms)

    class_weights = compute_class_weights(train_df['label'].tolist(), num_classes=num_classes)

    train_sampler = None
    shuffle = True
    if bool(cfg['sampling'].get('use_weighted_sampler', False)):
        train_sampler = build_weighted_sampler(train_df['label'].tolist(), num_classes=num_classes)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg['train']['batch_size']),
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=int(cfg['train']['num_workers']),
        pin_memory=(device == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg['train']['batch_size']),
        shuffle=False,
        num_workers=int(cfg['train']['num_workers']),
        pin_memory=(device == 'cuda'),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg['train']['batch_size']),
        shuffle=False,
        num_workers=int(cfg['train']['num_workers']),
        pin_memory=(device == 'cuda'),
    )

    # ── Class distribution plot ───────────────────────────────────────────
    class_dist = {
        'train': _get_class_distribution(train_df, class_names),
        'val': _get_class_distribution(val_df, class_names),
        'test': _get_class_distribution(test_df, class_names),
    }
    plot_class_distribution(class_dist, output_dir / 'class_distribution.png')
    save_json(output_dir / 'class_distribution.json', class_dist)

    # ── Model, Loss, Optimizer, Scheduler ─────────────────────────────────
    model = build_model(cfg['model']).to(device)
    criterion = build_loss(cfg['loss'], class_weights=class_weights, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    monitor_metric = cfg['metric']['monitor']
    mode = cfg['metric']['mode']
    best_metric = float('-inf') if mode == 'max' else float('inf')
    patience = int(cfg['train'].get('early_stopping_patience', 5))
    patience_counter = 0
    best_epoch = 0

    # ── History tracking ──────────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_macro_f1': [], 'val_macro_f1': [],
        'val_weighted_f1': [],
        'val_cohen_kappa': [], 'val_mcc': [],
        'lr': [],
    }
    # Per-class F1 history (for detailed tracking)
    per_class_f1_history = {name: [] for name in class_names}

    # Epoch log rows for CSV export
    epoch_log_rows = []

    # ── Mixup / CutMix config ─────────────────────────────────────────────
    mixup_alpha = float(cfg['train'].get('mixup_alpha', 0.0))
    cutmix_alpha = float(cfg['train'].get('cutmix_alpha', 0.0))
    mixup_cutmix_prob = float(cfg['train'].get('mixup_cutmix_prob', 0.0))

    # ── Print experiment info ─────────────────────────────────────────────
    print('=' * 80)
    print(f"  Experiment: {cfg.get('experiment_name', 'unnamed')}")
    print(f"  Model:      {cfg['model']['name']}")
    print(f"  Loss:       {cfg['loss']['name']} (label_smoothing={cfg['loss'].get('label_smoothing', 0.0)})")
    print(f"  Optimizer:  {cfg['optimizer']['name']} (lr={cfg['optimizer']['lr']}, wd={cfg['optimizer'].get('weight_decay', 0.0)})")
    print(f"  Scheduler:  {cfg['scheduler']['name']}")
    print(f"  Device:     {device} | AMP: {amp}")
    if mixup_alpha > 0 or cutmix_alpha > 0:
        print(f"  Mixup:      alpha={mixup_alpha}, CutMix alpha={cutmix_alpha}, prob={mixup_cutmix_prob}")
    print(f"  Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"  Epochs:     {total_epochs} | Patience: {patience}")
    print(f"  Output:     {output_dir}")
    print('=' * 80)

    training_start_time = time.time()

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # --- Train ---
        train_loss, train_targets, train_preds = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=amp,
            grad_clip=cfg['train'].get('grad_clip', None),
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixup_cutmix_prob=mixup_cutmix_prob,
        )
        train_metrics = compute_classification_metrics(train_targets, train_preds, class_names)

        # --- Validate ---
        val_loss, val_targets, val_preds, _, val_probs = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp=amp,
        )
        val_probs_np = val_probs.numpy()
        val_metrics = compute_classification_metrics(val_targets, val_preds, class_names, y_prob=val_probs_np)
        score = val_metrics[monitor_metric]

        epoch_elapsed = time.time() - epoch_start

        # --- Update history ---
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_macro_f1'].append(train_metrics['macro_f1'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_weighted_f1'].append(val_metrics['weighted_f1'])
        history['val_cohen_kappa'].append(val_metrics['cohen_kappa'])
        history['val_mcc'].append(val_metrics['mcc'])
        history['lr'].append(current_lr)

        for pcm in val_metrics['per_class_metrics']:
            per_class_f1_history[pcm['class_name']].append(pcm['f1'])

        # --- Epoch log row ---
        epoch_row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'train_accuracy': round(train_metrics['accuracy'], 6),
            'val_accuracy': round(val_metrics['accuracy'], 6),
            'train_macro_f1': round(train_metrics['macro_f1'], 6),
            'val_macro_f1': round(val_metrics['macro_f1'], 6),
            'val_weighted_f1': round(val_metrics['weighted_f1'], 6),
            'val_macro_precision': round(val_metrics['macro_precision'], 6),
            'val_macro_recall': round(val_metrics['macro_recall'], 6),
            'val_cohen_kappa': round(val_metrics['cohen_kappa'], 6),
            'val_mcc': round(val_metrics['mcc'], 6),
            'lr': current_lr,
            'epoch_time_s': round(epoch_elapsed, 2),
        }
        # Add per-class F1 to epoch row
        for pcm in val_metrics['per_class_metrics']:
            epoch_row[f"val_f1_{pcm['class_name']}"] = round(pcm['f1'], 6)

        # Add AUC if available
        if 'auc_macro' in val_metrics:
            epoch_row['val_auc_macro'] = round(val_metrics['auc_macro'], 6)

        epoch_log_rows.append(epoch_row)

        # --- Check improvement ---
        improved = score > best_metric if mode == 'max' else score < best_metric
        marker = ''
        if improved:
            best_metric = score
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'best_metric': best_metric,
                'epoch': epoch,
            }, output_dir / 'best.pth')
            marker = ' ★'
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()

        # --- Console output ---
        line = _format_epoch_line(epoch, total_epochs, train_loss, val_loss,
                                  val_metrics['accuracy'], val_metrics['macro_f1'],
                                  current_lr, epoch_elapsed)
        print(f"{line}{marker}")

        # --- Early stopping ---
        if patience_counter >= patience:
            print(f'\n⚠ Early stopping triggered at epoch {epoch} (patience={patience}).')
            print(f'  Best {monitor_metric}: {best_metric:.4f} at epoch {best_epoch}.')
            break

    total_training_time = time.time() - training_start_time

    # ── Save epoch log CSV ────────────────────────────────────────────────
    epoch_log_df = pd.DataFrame(epoch_log_rows)
    epoch_log_df.to_csv(output_dir / 'epoch_log.csv', index=False)

    # ── Save training history ─────────────────────────────────────────────
    # Add per-class F1 history
    full_history = {**history, **{f'val_f1_{k}': v for k, v in per_class_f1_history.items()}}
    save_json(output_dir / 'history.json', full_history)
    plot_history(full_history, output_dir / 'history.png')
    plot_training_history_detailed(full_history, output_dir / 'training_history_detailed.png')

    # ── Test evaluation ───────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('  Evaluating best model on test set...')
    print('=' * 80)

    checkpoint = torch.load(output_dir / 'best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_targets, test_preds, test_paths, test_probs = evaluate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp=amp,
    )
    test_probs_np = test_probs.numpy()
    test_metrics = compute_classification_metrics(test_targets, test_preds, class_names, y_prob=test_probs_np)
    test_metrics['test_loss'] = float(test_loss)

    # Save test metrics
    save_json(output_dir / 'test_metrics.json', test_metrics)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_confusion_matrices(
        test_metrics['confusion_matrix'],
        class_names,
        output_dir / 'confusion_matrix_raw.png',
        output_dir / 'confusion_matrix_normalized.png',
    )

    plot_per_class_f1(
        test_metrics['per_class_metrics'],
        output_dir / 'test_per_class_f1.png',
    )

    plot_sensitivity_specificity(
        test_metrics['per_class_metrics'],
        output_dir / 'test_sensitivity_specificity.png',
    )

    if 'roc_curves' in test_metrics and test_metrics['roc_curves']:
        plot_roc_curves(
            test_metrics['roc_curves'],
            output_dir / 'test_roc_curves.png',
        )

    # ── Predictions CSV (with probabilities) ──────────────────────────────
    pred_df = pd.DataFrame({
        'image_path': test_paths,
        'y_true': test_targets,
        'y_pred': test_preds,
    })
    pred_df['true_name'] = pred_df['y_true'].map({i: name for i, name in enumerate(class_names)})
    pred_df['pred_name'] = pred_df['y_pred'].map({i: name for i, name in enumerate(class_names)})
    pred_df['correct'] = pred_df['y_true'] == pred_df['y_pred']

    # Add per-class probabilities
    for i, name in enumerate(class_names):
        pred_df[f'prob_{name}'] = test_probs_np[:, i].round(6)

    pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)

    # ── Experiment summary ────────────────────────────────────────────────
    experiment_summary = {
        'experiment_name': cfg.get('experiment_name', 'unnamed'),
        'model_name': cfg['model']['name'],
        'num_classes': num_classes,
        'class_names': class_names,
        'loss_function': cfg['loss']['name'],
        'label_smoothing': cfg['loss'].get('label_smoothing', 0.0),
        'optimizer': cfg['optimizer']['name'],
        'learning_rate': cfg['optimizer']['lr'],
        'weight_decay': cfg['optimizer'].get('weight_decay', 0.0),
        'scheduler': cfg['scheduler']['name'],
        'batch_size': cfg['train']['batch_size'],
        'image_size': cfg['train']['image_size'],
        'dropout': cfg['model'].get('dropout', 0.0),
        'use_weighted_sampler': cfg['sampling'].get('use_weighted_sampler', False),
        'use_amp': amp,
        'device': device,
        'seed': cfg.get('seed', 42),
        'total_epochs_run': len(history['train_loss']),
        'max_epochs': total_epochs,
        'best_epoch': best_epoch,
        'early_stopped': patience_counter >= patience,
        'training_time_seconds': round(total_training_time, 2),
        'training_time_minutes': round(total_training_time / 60, 2),
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'test_samples': len(test_ds),
        'class_distribution': class_dist,
        'class_weights': class_weights.tolist() if hasattr(class_weights, 'tolist') else list(class_weights),
        'best_val_metric': {
            'name': monitor_metric,
            'value': round(best_metric, 6),
        },
        'test_results': {
            'accuracy': round(test_metrics['accuracy'], 6),
            'macro_precision': round(test_metrics['macro_precision'], 6),
            'macro_recall': round(test_metrics['macro_recall'], 6),
            'macro_f1': round(test_metrics['macro_f1'], 6),
            'weighted_f1': round(test_metrics['weighted_f1'], 6),
            'cohen_kappa': round(test_metrics['cohen_kappa'], 6),
            'mcc': round(test_metrics['mcc'], 6),
            'test_loss': round(test_loss, 6),
        },
    }

    if 'auc_macro' in test_metrics:
        experiment_summary['test_results']['auc_macro'] = round(test_metrics['auc_macro'], 6)
        experiment_summary['test_results']['auc_weighted'] = round(test_metrics['auc_weighted'], 6)

    save_json(output_dir / 'experiment_summary.json', experiment_summary)

    # ── Final console summary ─────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('  TRAINING COMPLETE')
    print('=' * 80)
    print(f"  Experiment:     {cfg.get('experiment_name', 'unnamed')}")
    print(f"  Best epoch:     {best_epoch}/{len(history['train_loss'])}")
    print(f"  Training time:  {total_training_time / 60:.1f} minutes")
    print(f"  Best val {monitor_metric}: {best_metric:.4f}")
    print()
    print('  ── Test Results ──')
    print(f"  Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:    {test_metrics['weighted_f1']:.4f}")
    print(f"  Cohen Kappa:    {test_metrics['cohen_kappa']:.4f}")
    print(f"  MCC:            {test_metrics['mcc']:.4f}")
    if 'auc_macro' in test_metrics:
        print(f"  AUC (macro):    {test_metrics['auc_macro']:.4f}")
    print()
    print('  ── Per-Class F1 ──')
    for pcm in test_metrics['per_class_metrics']:
        auc_str = f"  AUC={pcm.get('auc', 0):.3f}" if 'auc' in pcm else ''
        print(f"    {pcm['class_name']:>6s}: F1={pcm['f1']:.3f}  P={pcm['precision']:.3f}  R={pcm['recall']:.3f}{auc_str}  (n={pcm['support']})")
    print()
    print(f"  Output saved to: {output_dir}")
    print('=' * 80)


if __name__ == '__main__':
    main()
