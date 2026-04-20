from __future__ import annotations

from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def compute_classification_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    per_class_metrics = []
    for idx, name in enumerate(class_names):
        per_class_metrics.append({
            'class_id': idx,
            'class_name': name,
            'precision': float(per_class[0][idx]),
            'recall': float(per_class[1][idx]),
            'f1': float(per_class[2][idx]),
            'support': int(per_class[3][idx]),
        })

    return {
        'accuracy': float(acc),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'weighted_f1': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics,
        'classification_report': report,
    }
