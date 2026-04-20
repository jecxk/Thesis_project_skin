from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
#  Training history
# --------------------------------------------------------------------------- #

def plot_history(history: dict, save_path: str | Path) -> None:
    """Simple loss-only history plot (backward compat)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_history_detailed(history: dict, save_path: str | Path) -> None:
    """Detailed training history with subplots: loss, accuracy, macro F1, LR."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)
    has_lr = 'lr' in history and len(history['lr']) > 0

    n_plots = 4 if has_lr else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    # 1. Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', markersize=3, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 's-', markersize=3, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Accuracy
    if 'train_accuracy' in history:
        axes[1].plot(epochs, history['train_accuracy'], 'o-', markersize=3, label='Train Acc')
    axes[1].plot(epochs, history['val_accuracy'], 's-', markersize=3, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    # 3. Macro F1
    if 'train_macro_f1' in history:
        axes[2].plot(epochs, history['train_macro_f1'], 'o-', markersize=3, label='Train F1')
    axes[2].plot(epochs, history['val_macro_f1'], 's-', markersize=3, label='Val F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title('Macro F1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    # 4. Learning Rate
    if has_lr:
        axes[3].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Learning Rate')
        axes[3].set_title('Learning Rate Schedule')
        axes[3].grid(True, alpha=0.3)
        axes[3].ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

    fig.suptitle('Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------- #
#  Confusion matrices
# --------------------------------------------------------------------------- #

def _plot_single_confusion_matrix(
    cm_np: np.ndarray,
    class_names: List[str],
    save_path: str | Path,
    title: str,
    fmt: str = 'd',
    normalize: bool = False,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm_np, interpolation='nearest', cmap='Blues', vmin=0, vmax=1 if normalize else None)
    plt.title(title, fontsize=14)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm_np.max() / 2.0 if cm_np.size > 0 else 0.0
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            value = format(cm_np[i, j], fmt)
            plt.text(
                j,
                i,
                value,
                ha='center',
                va='center',
                color='white' if cm_np[i, j] > thresh else 'black',
                fontsize=10
            )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(
    cm: List[List[int]],
    class_names: List[str],
    raw_save_path: str | Path,
    normalized_save_path: str | Path,
) -> None:
    cm_np = np.array(cm, dtype=np.int64)

    # Raw counts
    _plot_single_confusion_matrix(
        cm_np=cm_np,
        class_names=class_names,
        save_path=raw_save_path,
        title='Confusion Matrix (Raw Counts)',
        fmt='d',
        normalize=False,
    )

    # Normalized by true class (row-wise)
    row_sums = cm_np.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_np, row_sums, where=row_sums != 0)
    cm_norm = np.nan_to_num(cm_norm)

    _plot_single_confusion_matrix(
        cm_np=cm_norm,
        class_names=class_names,
        save_path=normalized_save_path,
        title='Confusion Matrix (Normalized by True Class)',
        fmt='.2f',
        normalize=True,
    )


# --------------------------------------------------------------------------- #
#  ROC Curves
# --------------------------------------------------------------------------- #

def plot_roc_curves(
    roc_curves: Dict,
    save_path: str | Path,
    title: str = 'ROC Curves (One-vs-Rest)',
) -> None:
    """Plot per-class ROC curves with AUC values.

    Args:
        roc_curves: dict mapping class_name -> {'fpr': [...], 'tpr': [...], 'auc': float}
        save_path: path to save the plot
        title: plot title
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(roc_curves)))

    plt.figure(figsize=(9, 8))
    for i, (name, data) in enumerate(roc_curves.items()):
        fpr = data['fpr']
        tpr = data['tpr']
        auc_val = data['auc']
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC = {auc_val:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------- #
#  Per-class F1 bar chart
# --------------------------------------------------------------------------- #

def plot_per_class_f1(
    per_class_metrics: List[Dict],
    save_path: str | Path,
    title: str = 'Per-Class F1-Score',
) -> None:
    """Bar chart of per-class F1 scores with precision & recall annotations."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    names = [m['class_name'] for m in per_class_metrics]
    f1s = [m['f1'] for m in per_class_metrics]
    precisions = [m['precision'] for m in per_class_metrics]
    recalls = [m['recall'] for m in per_class_metrics]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#4ECDC4', alpha=0.85)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#FF6B6B', alpha=0.85)
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#45B7D1', alpha=0.85)

    # Annotate F1 values on top
    for bar, val in zip(bars3, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------- #
#  Class distribution
# --------------------------------------------------------------------------- #

def plot_class_distribution(
    class_counts: Dict[str, Dict[str, int]],
    save_path: str | Path,
    title: str = 'Class Distribution by Split',
) -> None:
    """Stacked bar chart of class distribution across train/val/test splits.

    Args:
        class_counts: dict like {'train': {'akiec': 230, ...}, 'val': {...}, 'test': {...}}
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    splits = list(class_counts.keys())
    class_names = list(class_counts[splits[0]].keys())
    x = np.arange(len(class_names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']

    for i, split in enumerate(splits):
        counts = [class_counts[split].get(name, 0) for name in class_names]
        bars = ax.bar(x + i * width, counts, width, label=f'{split} (n={sum(counts)})', color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(val), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------- #
#  Per-class metrics comparison (sensitivity / specificity / AUC)
# --------------------------------------------------------------------------- #

def plot_sensitivity_specificity(
    per_class_metrics: List[Dict],
    save_path: str | Path,
    title: str = 'Sensitivity vs Specificity per Class',
) -> None:
    """Bar chart comparing sensitivity and specificity per class."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    names = [m['class_name'] for m in per_class_metrics]
    sens = [m.get('sensitivity', m.get('recall', 0)) for m in per_class_metrics]
    spec = [m.get('specificity', 0) for m in per_class_metrics]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, sens, width, label='Sensitivity', color='#FF6B6B', alpha=0.85)
    ax.bar(x + width / 2, spec, width, label='Specificity', color='#4ECDC4', alpha=0.85)

    for i in range(len(names)):
        ax.text(x[i] - width / 2, sens[i] + 0.01, f'{sens[i]:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(x[i] + width / 2, spec[i] + 0.01, f'{spec[i]:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()