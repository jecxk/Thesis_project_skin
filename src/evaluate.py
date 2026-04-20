from __future__ import annotations

import argparse
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
from src.utils.plots import plot_confusion_matrices


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
    loss, y_true, y_pred, paths = evaluate_one_epoch(model, loader, criterion, device=device, amp=amp)
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics['loss'] = float(loss)
    save_json(output_dir / f'{args.split}_metrics_from_eval.json', metrics)
    plot_confusion_matrices(
       metrics['confusion_matrix'],
       class_names,
       output_dir / f'{args.split}_confusion_matrix_raw_from_eval.png',
       output_dir / f'{args.split}_confusion_matrix_normalized_from_eval.png',
    )

    pred_df = pd.DataFrame({'image_path': paths, 'y_true': y_true, 'y_pred': y_pred})
    pred_df.to_csv(output_dir / f'{args.split}_predictions_from_eval.csv', index=False)

    print(f"{args.split} accuracy: {metrics['accuracy']:.4f}")
    print(f"{args.split} macro F1: {metrics['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
