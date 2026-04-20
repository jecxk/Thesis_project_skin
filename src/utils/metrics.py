from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    cohen_kappa_score,
    matthews_corrcoef,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: ground-truth labels (int)
        y_pred: predicted labels (int)
        class_names: list of class name strings
        y_prob: optional (N, C) array of predicted probabilities for ROC/AUC
    """
    num_classes = len(class_names)

    # --- Basic metrics ---
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # --- Cohen's Kappa ---
    kappa = cohen_kappa_score(y_true, y_pred)

    # --- Matthews Correlation Coefficient ---
    mcc = matthews_corrcoef(y_true, y_pred)

    # --- Sensitivity (recall) and Specificity per class ---
    per_class_metrics = []
    for idx, name in enumerate(class_names):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        per_class_metrics.append({
            'class_id': idx,
            'class_name': name,
            'precision': float(per_class[0][idx]),
            'recall': float(per_class[1][idx]),
            'f1': float(per_class[2][idx]),
            'support': int(per_class[3][idx]),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        })

    # --- AUC (if probabilities available) ---
    auc_macro = None
    auc_weighted = None
    per_class_auc = [None] * num_classes
    roc_curves = {}

    if y_prob is not None:
        y_true_np = np.array(y_true)
        try:
            # One-vs-rest AUC
            auc_macro = float(roc_auc_score(y_true_np, y_prob, multi_class='ovr', average='macro'))
            auc_weighted = float(roc_auc_score(y_true_np, y_prob, multi_class='ovr', average='weighted'))
        except ValueError:
            pass  # may fail if some class has no samples

        # Per-class AUC and ROC curves
        for idx, name in enumerate(class_names):
            y_true_binary = (y_true_np == idx).astype(int)
            if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
                continue  # skip if class not present or all same
            try:
                auc_val = float(roc_auc_score(y_true_binary, y_prob[:, idx]))
                per_class_auc[idx] = auc_val
                per_class_metrics[idx]['auc'] = auc_val

                fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, idx])
                roc_curves[name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': auc_val,
                }
            except ValueError:
                pass

    result = {
        'accuracy': float(acc),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'weighted_f1': float(f1_weighted),
        'cohen_kappa': float(kappa),
        'mcc': float(mcc),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics,
        'classification_report': report,
    }

    if auc_macro is not None:
        result['auc_macro'] = auc_macro
        result['auc_weighted'] = auc_weighted
        result['per_class_auc'] = per_class_auc
        result['roc_curves'] = roc_curves

    return result
