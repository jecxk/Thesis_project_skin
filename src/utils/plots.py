from __future__ import annotations

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: dict, save_path: str | Path) -> None:
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