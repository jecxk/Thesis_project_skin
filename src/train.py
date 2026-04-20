from __future__ import annotations

import argparse
from pathlib import Path
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
from src.utils.plots import plot_confusion_matrices, plot_history
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
    if sch_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg['train']['epochs']),
            eta_min=float(cfg['scheduler'].get('min_lr', 1e-6)),
        )
    if sch_name == 'none':
        return None
    raise ValueError(f'Unsupported scheduler: {sch_name}')


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get('seed', 42)))

    output_dir = ensure_dir(cfg['output_dir'])
    class_names = cfg['classes']
    num_classes = len(class_names)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = bool(cfg['train'].get('amp', False) and device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

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

    model = build_model(cfg['model']).to(device)
    criterion = build_loss(cfg['loss'], class_weights=class_weights, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    monitor_metric = cfg['metric']['monitor']
    mode = cfg['metric']['mode']
    best_metric = float('-inf') if mode == 'max' else float('inf')
    patience = int(cfg['train'].get('early_stopping_patience', 5))
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': [], 'val_accuracy': []}

    for epoch in range(1, int(cfg['train']['epochs']) + 1):
        train_loss, train_targets, train_preds = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=amp,
            grad_clip=cfg['train'].get('grad_clip', None),
        )
        val_loss, val_targets, val_preds, _ = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp=amp,
        )

        val_metrics = compute_classification_metrics(val_targets, val_preds, class_names)
        score = val_metrics[monitor_metric]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        improved = score > best_metric if mode == 'max' else score < best_metric
        if improved:
            best_metric = score
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'best_metric': best_metric,
                'epoch': epoch,
            }, output_dir / 'best.pth')
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch}.')
            break

    plot_history(history, output_dir / 'history.png')
    save_json(output_dir / 'history.json', history)

    checkpoint = torch.load(output_dir / 'best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_targets, test_preds, test_paths = evaluate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp=amp,
    )
    test_metrics = compute_classification_metrics(test_targets, test_preds, class_names)
    test_metrics['test_loss'] = float(test_loss)
    save_json(output_dir / 'test_metrics.json', test_metrics)
    plot_confusion_matrices(
        test_metrics['confusion_matrix'],
        class_names,
        output_dir / 'confusion_matrix_raw.png',
        output_dir / 'confusion_matrix_normalized.png',
    ) 

    pred_df = pd.DataFrame({
        'image_path': test_paths,
        'y_true': test_targets,
        'y_pred': test_preds,
    })
    pred_df['true_name'] = pred_df['y_true'].map({i: name for i, name in enumerate(class_names)})
    pred_df['pred_name'] = pred_df['y_pred'].map({i: name for i, name in enumerate(class_names)})
    pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)

    print('Training finished.')
    print(f"Best validation {monitor_metric}: {best_metric:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
