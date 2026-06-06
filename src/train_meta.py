"""Training script for metadata-fusion models.

Mirrors ``src/train.py`` but uses ``MetaFusionModel`` (image + patient metadata)
and the enriched dataset / dataloader. Reads the same YAML config format, with
two new optional keys under ``paths``:

    paths:
      meta_csv:      data/metadata/skin_metadata_full.csv      # required for metadata
      meta_features: data/metadata/skin_metadata_features.npy  # required

and an optional ``model.meta_dim`` (default 19), ``model.meta_hidden`` (64),
``model.meta_out`` (32) section. The classifier sees the concatenation of the
backbone's pooled features and the encoded metadata.

Usage
-----
    python -m src.train_meta --config src/configs/efficientnet_b0_meta.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import compute_class_weights
from src.data.dataset_meta import SkinLesionDatasetWithMeta, load_split_with_meta
from src.data.transforms import build_transforms
from src.data.samplers import build_weighted_sampler
from src.data.mixup import mixup_data, cutmix_data, mixup_criterion
from src.models.meta_fusion import build_meta_fusion_model
from src.models.losses import build_loss
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.utils.metrics import compute_classification_metrics
from src.utils.plots import (
    plot_confusion_matrices, plot_history,
    plot_training_history_detailed, plot_roc_curves,
    plot_per_class_f1, plot_sensitivity_specificity,
)
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    return p.parse_args()


def build_optimizer(model, cfg):
    name = cfg['optimizer']['name'].lower()
    lr = float(cfg['optimizer']['lr'])
    wd = float(cfg['optimizer'].get('weight_decay', 0.0))
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f'Unsupported optimizer: {name}')


def build_scheduler(optimizer, cfg):
    name = cfg['scheduler']['name'].lower()
    total = int(cfg['train']['epochs'])
    min_lr = float(cfg['scheduler'].get('min_lr', 1e-6))
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total, eta_min=min_lr)
    if name == 'cosine_warmup':
        warmup = int(cfg['scheduler'].get('warmup_epochs', 3))
        warmup_s = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
        cosine_s = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total - warmup, eta_min=min_lr)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_s, cosine_s], milestones=[warmup])
    if name == 'none':
        return None
    raise ValueError(f'Unsupported scheduler: {name}')


def train_one_epoch_meta(model, loader, criterion, optimizer, device,
                         scaler, amp, grad_clip,
                         mixup_alpha, cutmix_alpha, mixup_cutmix_prob):
    model.train()
    losses, all_t, all_p = [], [], []
    for images, meta, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_mix = (np.random.rand() < mixup_cutmix_prob and
                   (mixup_alpha > 0 or cutmix_alpha > 0))
        if use_mix and cutmix_alpha > 0 and np.random.rand() < 0.5:
            images, t_a, t_b, lam = cutmix_data(images, targets, alpha=cutmix_alpha)
        elif use_mix and mixup_alpha > 0:
            images, t_a, t_b, lam = mixup_data(images, targets, alpha=mixup_alpha)
        else:
            t_a = t_b = targets
            lam = 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images, meta)
            loss = mixup_criterion(criterion, logits, t_a, t_b, lam)

        if amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_t.extend(targets.detach().cpu().numpy().tolist())
        all_p.extend(preds.tolist())

    return float(np.mean(losses)), all_t, all_p


@torch.no_grad()
def evaluate_one_epoch_meta(model, loader, criterion, device, amp):
    model.eval()
    losses, all_t, all_p, paths, all_probs = [], [], [], [], []
    for images, meta, targets, p_batch in loader:
        images = images.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images, meta)
            loss = criterion(logits, targets)
        losses.append(loss.item())
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
        all_p.extend(probs.argmax(dim=1).tolist())
        all_t.extend(targets.cpu().tolist())
        paths.extend(list(p_batch))
    return (float(np.mean(losses)), all_t, all_p, paths,
            torch.cat(all_probs, dim=0))


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

    # ---- Data with metadata
    meta_csv = cfg['paths']['meta_csv']
    meta_npy = cfg['paths']['meta_features']
    train_df, val_df, test_df, tr_feats, va_feats, te_feats = load_split_with_meta(
        meta_csv, meta_npy)

    root_dir = cfg['paths']['root_dir']
    train_tfms = build_transforms(cfg['train']['image_size'],
                                  cfg['augmentation']['train'], is_train=True)
    eval_tfms = build_transforms(cfg['train']['image_size'],
                                 cfg['augmentation']['eval'], is_train=False)
    train_ds = SkinLesionDatasetWithMeta(train_df, root_dir, tr_feats, transform=train_tfms)
    val_ds   = SkinLesionDatasetWithMeta(val_df,   root_dir, va_feats, transform=eval_tfms)
    test_ds  = SkinLesionDatasetWithMeta(test_df,  root_dir, te_feats, transform=eval_tfms)

    class_weights = compute_class_weights(train_df['label'].tolist(),
                                          num_classes=num_classes)

    sampler = None
    shuffle = True
    if bool(cfg['sampling'].get('use_weighted_sampler', False)):
        sampler = build_weighted_sampler(train_df['label'].tolist(),
                                         num_classes=num_classes)
        shuffle = False

    train_loader = DataLoader(train_ds,
                              batch_size=int(cfg['train']['batch_size']),
                              shuffle=shuffle, sampler=sampler,
                              num_workers=int(cfg['train']['num_workers']),
                              pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_ds,
                            batch_size=int(cfg['train']['batch_size']),
                            shuffle=False,
                            num_workers=int(cfg['train']['num_workers']),
                            pin_memory=(device == 'cuda'))
    test_loader = DataLoader(test_ds,
                             batch_size=int(cfg['train']['batch_size']),
                             shuffle=False,
                             num_workers=int(cfg['train']['num_workers']),
                             pin_memory=(device == 'cuda'))

    # ---- Model
    cfg['model']['meta_dim'] = int(tr_feats.shape[1])
    cfg['model']['num_classes'] = num_classes
    model = build_meta_fusion_model(cfg['model']).to(device)
    criterion = build_loss(cfg['loss'], class_weights=class_weights, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    monitor = cfg['metric']['monitor']
    mode = cfg['metric']['mode']
    best = float('-inf') if mode == 'max' else float('inf')
    patience = int(cfg['train'].get('early_stopping_patience', 12))
    counter = 0
    best_epoch = 0
    history = {k: [] for k in ('train_loss', 'val_loss', 'train_accuracy',
                                'val_accuracy', 'train_macro_f1',
                                'val_macro_f1', 'lr')}
    epoch_log = []
    mixup_alpha = float(cfg['train'].get('mixup_alpha', 0.0))
    cutmix_alpha = float(cfg['train'].get('cutmix_alpha', 0.0))
    mixup_cutmix_prob = float(cfg['train'].get('mixup_cutmix_prob', 0.0))

    print('=' * 80)
    print(f"  Meta-fusion experiment: {cfg.get('experiment_name', 'unnamed')}")
    print(f"  Backbone:  {cfg['model']['name']}")
    print(f"  Meta dim:  {cfg['model']['meta_dim']}")
    print(f"  Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"  Epochs:    {total_epochs} | Patience: {patience}")
    print('=' * 80)

    start = time.time()
    for epoch in range(1, total_epochs + 1):
        e_start = time.time()
        lr = optimizer.param_groups[0]['lr']
        tr_loss, tr_t, tr_p = train_one_epoch_meta(
            model, train_loader, criterion, optimizer, device, scaler, amp,
            cfg['train'].get('grad_clip', None),
            mixup_alpha, cutmix_alpha, mixup_cutmix_prob)
        tr_m = compute_classification_metrics(tr_t, tr_p, class_names)

        va_loss, va_t, va_p, _, va_probs = evaluate_one_epoch_meta(
            model, val_loader, criterion, device, amp)
        va_m = compute_classification_metrics(va_t, va_p, class_names,
                                              y_prob=va_probs.numpy())
        score = va_m[monitor]
        e_elapsed = time.time() - e_start

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_accuracy'].append(tr_m['accuracy'])
        history['val_accuracy'].append(va_m['accuracy'])
        history['train_macro_f1'].append(tr_m['macro_f1'])
        history['val_macro_f1'].append(va_m['macro_f1'])
        history['lr'].append(lr)
        epoch_log.append({
            'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss,
            'train_acc': tr_m['accuracy'], 'val_acc': va_m['accuracy'],
            'train_f1': tr_m['macro_f1'], 'val_f1': va_m['macro_f1'],
            'lr': lr, 'time_s': e_elapsed,
        })

        improved = score > best if mode == 'max' else score < best
        marker = ''
        if improved:
            best = score
            best_epoch = epoch
            counter = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'config': cfg, 'best_metric': best, 'epoch': epoch},
                       output_dir / 'best.pth')
            marker = ' *'
        else:
            counter += 1
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch:03d}/{total_epochs} | tr_loss={tr_loss:.4f} | "
              f"va_loss={va_loss:.4f} | va_acc={va_m['accuracy']:.4f} | "
              f"va_f1={va_m['macro_f1']:.4f} | lr={lr:.2e} | "
              f"{e_elapsed:.1f}s{marker}")
        if counter >= patience:
            print(f'Early stopping at epoch {epoch} (best={best:.4f} @ {best_epoch}).')
            break

    total_time = time.time() - start

    pd.DataFrame(epoch_log).to_csv(output_dir / 'epoch_log.csv', index=False)
    save_json(output_dir / 'history.json', history)
    plot_history(history, output_dir / 'history.png')
    plot_training_history_detailed(history, output_dir / 'training_history_detailed.png')

    # ---- Test
    print('\n  Evaluating best model on test set...')
    ckpt = torch.load(output_dir / 'best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    te_loss, te_t, te_p, te_paths, te_probs = evaluate_one_epoch_meta(
        model, test_loader, criterion, device, amp)
    te_probs_np = te_probs.numpy()
    te_m = compute_classification_metrics(te_t, te_p, class_names,
                                          y_prob=te_probs_np)
    te_m['test_loss'] = float(te_loss)
    save_json(output_dir / 'test_metrics.json', te_m)

    plot_confusion_matrices(te_m['confusion_matrix'], class_names,
                            output_dir / 'confusion_matrix_raw.png',
                            output_dir / 'confusion_matrix_normalized.png')
    plot_per_class_f1(te_m['per_class_metrics'], output_dir / 'test_per_class_f1.png')
    plot_sensitivity_specificity(te_m['per_class_metrics'],
                                 output_dir / 'test_sensitivity_specificity.png')
    if 'roc_curves' in te_m and te_m['roc_curves']:
        plot_roc_curves(te_m['roc_curves'], output_dir / 'test_roc_curves.png')

    pred_df = pd.DataFrame({
        'image_path': te_paths,
        'y_true': te_t,
        'y_pred': te_p,
    })
    pred_df['true_name'] = pred_df['y_true'].map({i: n for i, n in enumerate(class_names)})
    pred_df['pred_name'] = pred_df['y_pred'].map({i: n for i, n in enumerate(class_names)})
    pred_df['correct'] = pred_df['y_true'] == pred_df['y_pred']
    for i, n in enumerate(class_names):
        pred_df[f'prob_{n}'] = te_probs_np[:, i].round(6)
    pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)

    print('\n  ── Test (meta-fusion) ──')
    print(f"  Accuracy:    {te_m['accuracy']:.4f}")
    print(f"  Macro F1:    {te_m['macro_f1']:.4f}")
    if 'auc_macro' in te_m:
        print(f"  Macro AUC:   {te_m['auc_macro']:.4f}")
    print(f"  Training time: {total_time / 60:.1f} min")
    print(f"  Output:      {output_dir}")


if __name__ == '__main__':
    main()
