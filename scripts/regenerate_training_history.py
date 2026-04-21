"""Regenerate training-history figures as large, standalone per-model images.

Motivation
----------
The current thesis groups four training histories into a 2x2 subfigure,
making each model's subplot too small to read (see chapter 4, Fig 4.1).
This script produces:
  * `figures/training_history_<modelslug>.png`  — 2x2 grid (Loss, Acc, F1, LR)
    rendered at a large size for each individual model.
  * A small comparison figure `training_loss_all_models.png` overlaying only
    the validation loss curves of all models on a single axis — a compact,
    legible replacement for the previous 2x2 composite.

Output directory
----------------
Default: thesis/figures/  (so LaTeX can \\includegraphics them directly)

Usage
-----
python -m scripts.regenerate_training_history
python -m scripts.regenerate_training_history --output-dir thesis/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Maps model display name -> (history.json path, short slug for filename)
MODELS = [
    ('EfficientNet-B0', 'outputs/efficientnet_b0_v3(main_best)/history.json', 'eb0'),
    ('ResNet50',        'outputs/resnet50_v3(main)/history.json',             'rn50'),
    ('DenseNet121',     'outputs/densenet121_v3(main)/history.json',          'dn121'),
    ('Swin-Tiny',       'outputs/swin_tiny_v3/history.json',                  'swin'),
]


def load_history(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_single_model_2x2(history: Dict, model_name: str, save_path: Path) -> None:
    """Create a large 2x2 figure for ONE model: loss, acc, F1, LR."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history['train_loss']) + 1)
    has_lr = 'lr' in history and len(history.get('lr', [])) > 0
    has_train_acc = 'train_accuracy' in history
    has_train_f1 = 'train_macro_f1' in history

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ((ax_loss, ax_acc), (ax_f1, ax_lr)) = axes

    # Panel 1 — Loss
    ax_loss.plot(epochs, history['train_loss'], 'o-', color='#45B7D1',
                 markersize=4, lw=1.8, label='Train Loss')
    ax_loss.plot(epochs, history['val_loss'], 's-', color='#FF6B6B',
                 markersize=4, lw=1.8, label='Val Loss')
    ax_loss.set_xlabel('Epoch', fontsize=13)
    ax_loss.set_ylabel('Loss', fontsize=13)
    ax_loss.set_title('(a) Loss', fontsize=14, fontweight='bold')
    ax_loss.legend(fontsize=11, loc='best')
    ax_loss.grid(True, alpha=0.3)

    # Panel 2 — Accuracy
    if has_train_acc:
        ax_acc.plot(epochs, history['train_accuracy'], 'o-', color='#45B7D1',
                    markersize=4, lw=1.8, label='Train Acc')
    ax_acc.plot(epochs, history['val_accuracy'], 's-', color='#FF6B6B',
                markersize=4, lw=1.8, label='Val Acc')
    ax_acc.set_xlabel('Epoch', fontsize=13)
    ax_acc.set_ylabel('Accuracy', fontsize=13)
    ax_acc.set_title('(b) Accuracy', fontsize=14, fontweight='bold')
    ax_acc.legend(fontsize=11, loc='best')
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim(0, 1.05)

    # Panel 3 — Macro F1
    if has_train_f1:
        ax_f1.plot(epochs, history['train_macro_f1'], 'o-', color='#45B7D1',
                   markersize=4, lw=1.8, label='Train Macro-F1')
    ax_f1.plot(epochs, history['val_macro_f1'], 's-', color='#FF6B6B',
               markersize=4, lw=1.8, label='Val Macro-F1')
    ax_f1.set_xlabel('Epoch', fontsize=13)
    ax_f1.set_ylabel('Macro F1', fontsize=13)
    ax_f1.set_title('(c) Macro F1-Score', fontsize=14, fontweight='bold')
    ax_f1.legend(fontsize=11, loc='best')
    ax_f1.grid(True, alpha=0.3)
    ax_f1.set_ylim(0, 1.05)

    # Mark best epoch (max val_macro_f1)
    best_ep = int(np.argmax(history['val_macro_f1'])) + 1
    best_val = float(np.max(history['val_macro_f1']))
    ax_f1.axvline(best_ep, color='green', linestyle='--', alpha=0.5, lw=1.5)
    ax_f1.annotate(f'Best: ep {best_ep}\n(F1 = {best_val:.3f})',
                   xy=(best_ep, best_val),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=10, color='green',
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.6))

    # Panel 4 — Learning rate
    if has_lr:
        ax_lr.plot(epochs, history['lr'], '-', color='#4ECDC4', lw=2.2)
        ax_lr.set_xlabel('Epoch', fontsize=13)
        ax_lr.set_ylabel('Learning Rate', fontsize=13)
        ax_lr.set_title('(d) Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax_lr.grid(True, alpha=0.3)
        ax_lr.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
    else:
        ax_lr.axis('off')
        ax_lr.text(0.5, 0.5, 'Learning-rate log not available',
                   ha='center', va='center', fontsize=12, color='#888')

    fig.suptitle(f'Training History — {model_name}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_cross_model_comparison(
    histories: List[Dict[str, object]],
    save_path: Path,
) -> None:
    """Overlay validation loss / F1 of all models on two side-by-side panels."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['#45B7D1', '#FF6B6B', '#4ECDC4', '#FFA94D', '#9775FA']

    # Validation loss
    for i, item in enumerate(histories):
        h = item['history']
        epochs = np.arange(1, len(h['val_loss']) + 1)
        axes[0].plot(epochs, h['val_loss'], '-', lw=2, color=colors[i % len(colors)],
                     label=item['name'])
    axes[0].set_xlabel('Epoch', fontsize=13)
    axes[0].set_ylabel('Validation Loss', fontsize=13)
    axes[0].set_title('Validation Loss — All Models',
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Validation macro F1
    for i, item in enumerate(histories):
        h = item['history']
        epochs = np.arange(1, len(h['val_macro_f1']) + 1)
        axes[1].plot(epochs, h['val_macro_f1'], '-', lw=2, color=colors[i % len(colors)],
                     label=item['name'])
    axes[1].set_xlabel('Epoch', fontsize=13)
    axes[1].set_ylabel('Validation Macro F1', fontsize=13)
    axes[1].set_title('Validation Macro F1 — All Models',
                      fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='thesis/figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('  Regenerate Training History Figures (per-model, large)')
    print(f'  Output: {output_dir}')
    print('=' * 70)

    histories = []
    for name, hist_path, slug in MODELS:
        p = Path(hist_path)
        if not p.exists():
            print(f'  [SKIP] {name}: {p} not found.')
            continue
        h = load_history(p)
        save_path = output_dir / f'training_history_{slug}.png'
        plot_single_model_2x2(h, name, save_path)
        histories.append({'name': name, 'history': h, 'slug': slug})
        print(f'  [OK] {name:<18s} → {save_path.name}')

    if len(histories) >= 2:
        save_path = output_dir / 'training_loss_all_models.png'
        plot_cross_model_comparison(histories, save_path)
        print(f'  [OK] Cross-model comparison → {save_path.name}')

    print('=' * 70)
    print(f'  ✓ Generated {len(histories)} per-model figures '
          f'{"+ 1 comparison figure" if len(histories) >= 2 else ""}.')
    print('=' * 70)


if __name__ == '__main__':
    main()
