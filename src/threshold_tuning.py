"""Per-class decision threshold tuning.

Default multiclass inference uses argmax of softmax. In imbalanced medical
classification, this can under-predict critical minority classes (e.g. melanoma),
because the model's probability for NV dominates.

Strategy
--------
For each class `c`, treat the problem as one-vs-rest binary classification
and find a probability threshold ``t_c`` that maximises a target metric on
the validation set (default: F1 for class c, optionally sensitivity@fixed-specificity).

At inference time, we score each sample by ``p_c / t_c`` (or equivalently
``p_c - t_c``) and take the argmax. The argmax-normalised formulation
preserves calibration across classes.

Usage
-----
python -m src.threshold_tuning \
    --val-pred outputs/ensemble/ensemble_val_predictions.csv \
    --test-pred outputs/ensemble/ensemble_predictions.csv \
    --output-dir outputs/threshold_tuning \
    --target-class mel \
    --min-specificity 0.85

If --val-pred is not provided, the script will auto-generate validation
probabilities by running the ensemble (MODELS from src.ensemble) on the
validation split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

from src.utils.io import ensure_dir, save_json
from src.utils.metrics import compute_classification_metrics


# --------------------------------------------------------------------------- #
#  Core tuning
# --------------------------------------------------------------------------- #

def tune_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    criterion: str = 'f1',
    min_specificity: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Find per-class thresholds by sweeping 100 quantile-based candidates."""
    n, c = probs.shape
    assert c == len(class_names)
    results: Dict[str, Dict[str, float]] = {}

    for cls_id, cls_name in enumerate(class_names):
        y = (labels == cls_id).astype(int)
        p = probs[:, cls_id]

        if y.sum() == 0:
            results[cls_name] = {
                'threshold': 0.5, 'f1': 0.0, 'sensitivity': 0.0, 'specificity': 0.0,
            }
            continue

        candidates = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 101)))
        best = None
        for t in candidates:
            y_pred = (p >= t).astype(int)
            tp = int(((y_pred == 1) & (y == 1)).sum())
            fp = int(((y_pred == 1) & (y == 0)).sum())
            fn = int(((y_pred == 0) & (y == 1)).sum())
            tn = int(((y_pred == 0) & (y == 0)).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            f1_val = (2 * tp) / max(2 * tp + fp + fn, 1)

            if spec < min_specificity:
                continue

            score = {
                'f1': f1_val,
                'sensitivity': sens,
                'youden': sens + spec - 1,
            }.get(criterion, f1_val)

            if best is None or score > best['score']:
                best = {'score': score, 'threshold': float(t),
                        'f1': f1_val, 'sensitivity': sens, 'specificity': spec,
                        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

        if best is None:
            best = {'score': 0.0, 'threshold': 0.5,
                    'f1': 0.0, 'sensitivity': 0.0, 'specificity': 0.0,
                    'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

        results[cls_name] = best

    return results


def apply_thresholds(
    probs: np.ndarray,
    thresholds: Dict[str, Dict[str, float]],
    class_names: List[str],
) -> np.ndarray:
    """Normalise p_c by t_c and pick argmax.

    This enforces 'class c predicted only if p_c >= t_c relative to other
    classes' and preserves a single consistent label.
    """
    t = np.array([thresholds[c]['threshold'] for c in class_names], dtype=np.float32)
    t = np.clip(t, 1e-6, None)
    return (probs / t[None, :]).argmax(axis=1)


# --------------------------------------------------------------------------- #
#  PR & ROC utility plots for the target class
# --------------------------------------------------------------------------- #

def plot_pr_roc_for_class(
    probs: np.ndarray,
    labels: np.ndarray,
    cls_id: int,
    cls_name: str,
    chosen_threshold: float,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    y = (labels == cls_id).astype(int)
    p = probs[:, cls_id]

    precision, recall, pr_thresh = precision_recall_curve(y, p)
    fpr, tpr, roc_thresh = roc_curve(y, p)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].plot(recall, precision, color='#45B7D1', lw=2, label='PR curve')
    # Mark chosen threshold
    y_pred_chosen = (p >= chosen_threshold).astype(int)
    tp = int(((y_pred_chosen == 1) & (y == 1)).sum())
    fp = int(((y_pred_chosen == 1) & (y == 0)).sum())
    fn = int(((y_pred_chosen == 0) & (y == 1)).sum())
    prec_at_t = tp / max(tp + fp, 1)
    rec_at_t = tp / max(tp + fn, 1)
    axes[0].scatter([rec_at_t], [prec_at_t], color='red', s=80, zorder=3,
                    label=f'Chosen T={chosen_threshold:.3f}')
    axes[0].set_xlabel('Recall (Sensitivity)', fontsize=11)
    axes[0].set_ylabel('Precision', fontsize=11)
    axes[0].set_title(f'Precision-Recall Curve — {cls_name.upper()}',
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.05)

    axes[1].plot(fpr, tpr, color='#FF6B6B', lw=2, label='ROC curve')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    sens_at_t = rec_at_t
    spec_at_t = 1.0 - (fp / max(fp + (y == 0).sum() - fp, 1))  # = TN / (TN+FP)
    tn = int(((y_pred_chosen == 0) & (y == 0)).sum())
    spec_at_t = tn / max(tn + fp, 1)
    fpr_at_t = 1 - spec_at_t
    axes[1].scatter([fpr_at_t], [sens_at_t], color='red', s=80, zorder=3,
                    label=f'Chosen T={chosen_threshold:.3f}')
    axes[1].set_xlabel('False Positive Rate (1 − Specificity)', fontsize=11)
    axes[1].set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    axes[1].set_title(f'ROC Curve — {cls_name.upper()}',
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def _load_predictions(path: Path, class_names_hint: Optional[List[str]] = None
                      ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if not prob_cols:
        raise ValueError(f'{path} must contain prob_* columns for tuning.')
    class_names = class_names_hint or [c.replace('prob_', '') for c in prob_cols]
    probs = df[[f'prob_{c}' for c in class_names]].to_numpy()
    labels = df['y_true'].to_numpy().astype(int)
    return probs, labels, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-pred', type=str, required=True,
                        help='CSV with prob_* columns on the validation set')
    parser.add_argument('--test-pred', type=str, required=True,
                        help='CSV with prob_* columns on the test set')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--target-class', type=str, default='mel',
                        help='Class whose PR/ROC will be plotted in detail.')
    parser.add_argument('--criterion', type=str, default='f1',
                        choices=['f1', 'sensitivity', 'youden'])
    parser.add_argument('--min-specificity', type=float, default=0.0,
                        help='Only used with --criterion sensitivity; '
                             'ensures at least this specificity per class.')
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))

    val_probs, val_labels, val_classes = _load_predictions(Path(args.val_pred))
    test_probs, test_labels, test_classes = _load_predictions(
        Path(args.test_pred), class_names_hint=val_classes
    )
    assert val_classes == test_classes, 'Class order mismatch between val and test CSV.'
    class_names = val_classes

    print('=' * 70)
    print('  Threshold Tuning')
    print(f'  Val predictions : {args.val_pred} ({len(val_labels)} samples)')
    print(f'  Test predictions: {args.test_pred} ({len(test_labels)} samples)')
    print(f'  Criterion       : {args.criterion}')
    print(f'  Min specificity : {args.min_specificity}')
    print('=' * 70)

    # Tune on validation
    thresholds = tune_thresholds(val_probs, val_labels, class_names,
                                 criterion=args.criterion,
                                 min_specificity=args.min_specificity)
    save_json(output_dir / 'thresholds.json', thresholds)

    # Report val + test metrics with and without tuning
    y_pred_test_argmax = test_probs.argmax(axis=1)
    y_pred_test_tuned = apply_thresholds(test_probs, thresholds, class_names)

    m_argmax = compute_classification_metrics(
        y_true=test_labels.tolist(), y_pred=y_pred_test_argmax.tolist(),
        class_names=class_names, y_prob=test_probs,
    )
    m_tuned = compute_classification_metrics(
        y_true=test_labels.tolist(), y_pred=y_pred_test_tuned.tolist(),
        class_names=class_names, y_prob=test_probs,
    )

    save_json(output_dir / 'test_metrics_argmax.json', m_argmax)
    save_json(output_dir / 'test_metrics_tuned.json', m_tuned)

    # Diff table per class
    rows = []
    for a, t in zip(m_argmax['per_class_metrics'], m_tuned['per_class_metrics']):
        rows.append({
            'class': a['class_name'],
            'threshold': round(thresholds[a['class_name']]['threshold'], 4),
            'sens_argmax': round(a['sensitivity'], 4),
            'sens_tuned':  round(t['sensitivity'], 4),
            'spec_argmax': round(a['specificity'], 4),
            'spec_tuned':  round(t['specificity'], 4),
            'f1_argmax':   round(a['f1'], 4),
            'f1_tuned':    round(t['f1'], 4),
        })
    diff_df = pd.DataFrame(rows)
    diff_df.to_csv(output_dir / 'per_class_comparison.csv', index=False)

    # Plot PR+ROC for target class
    if args.target_class in class_names:
        tgt_id = class_names.index(args.target_class)
        chosen_t = thresholds[args.target_class]['threshold']
        plot_pr_roc_for_class(
            probs=test_probs, labels=test_labels,
            cls_id=tgt_id, cls_name=args.target_class,
            chosen_threshold=chosen_t,
            save_path=output_dir / f'pr_roc_{args.target_class}.png',
        )

    # Console summary
    print('\n  -- Test-set metrics --')
    print(f'  Macro F1 (argmax): {m_argmax["macro_f1"]:.4f}')
    print(f'  Macro F1 (tuned) : {m_tuned["macro_f1"]:.4f}')
    print(f'  Accuracy (argmax): {m_argmax["accuracy"]:.4f}')
    print(f'  Accuracy (tuned) : {m_tuned["accuracy"]:.4f}')
    print()
    print('  Per-class (sensitivity change):')
    print(diff_df.to_string(index=False))
    print('=' * 70)


if __name__ == '__main__':
    main()
