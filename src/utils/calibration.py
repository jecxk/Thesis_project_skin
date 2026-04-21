"""Model calibration analysis.

Metrics
-------
* ECE (Expected Calibration Error)
* MCE (Maximum Calibration Error)
* Brier Score (multiclass)
* Negative Log Likelihood
* Reliability diagram plotting

Temperature scaling
-------------------
Post-hoc calibration that learns a single scalar T to rescale logits via:
    p = softmax(logits / T)

The optimal T is found on the validation set by minimising NLL.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

@dataclass
class CalibrationMetrics:
    ece: float
    mce: float
    brier_score: float
    nll: float
    accuracy: float
    mean_confidence: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _confidence_and_pred(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    return confidences, preds


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, float, np.ndarray]:
    """Compute ECE and MCE.

    Returns:
        ece, mce, bin_stats (array [n_bins, 4]: [lo, hi, conf, acc, count])
    """
    confidences, preds = _confidence_and_pred(probs)
    accuracies = (preds == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    n = len(confidences)
    stats = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            mask |= confidences == 0.0
        count = int(mask.sum())
        if count == 0:
            stats.append((lo, hi, 0.0, 0.0, 0))
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float(accuracies[mask].mean())
        gap = abs(avg_conf - avg_acc)
        ece += (count / n) * gap
        mce = max(mce, gap)
        stats.append((lo, hi, avg_conf, avg_acc, count))
    return float(ece), float(mce), np.array(stats, dtype=object)


def brier_score_multiclass(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(p + eps)))


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_bins: int = 15,
) -> Tuple[CalibrationMetrics, np.ndarray]:
    ece, mce, bin_stats = expected_calibration_error(probs, labels, n_bins=n_bins)
    brier = brier_score_multiclass(probs, labels, num_classes)
    nll = negative_log_likelihood(probs, labels)
    confidences, preds = _confidence_and_pred(probs)
    accuracy = float((preds == labels).mean())
    mean_conf = float(confidences.mean())

    metrics = CalibrationMetrics(
        ece=ece,
        mce=mce,
        brier_score=brier,
        nll=nll,
        accuracy=accuracy,
        mean_confidence=mean_conf,
    )
    return metrics, bin_stats


# --------------------------------------------------------------------------- #
#  Reliability diagram
# --------------------------------------------------------------------------- #

def plot_reliability_diagram(
    bin_stats: np.ndarray,
    save_path: str | Path,
    title: str = 'Reliability Diagram',
    ece_value: float | None = None,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    lo = np.array([s[0] for s in bin_stats], dtype=float)
    hi = np.array([s[1] for s in bin_stats], dtype=float)
    conf = np.array([s[2] for s in bin_stats], dtype=float)
    acc = np.array([s[3] for s in bin_stats], dtype=float)
    count = np.array([s[4] for s in bin_stats], dtype=float)
    centres = (lo + hi) / 2
    width = hi - lo

    # Ideal line
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.2, label='Perfect calibration')

    # Accuracy bars (blue)
    ax.bar(centres, acc, width=width * 0.95, alpha=0.75, edgecolor='black',
           color='#45B7D1', label='Accuracy')

    # Gap (red hatched) — conf above acc means over-confident
    gap = conf - acc
    ax.bar(centres, gap, width=width * 0.95, bottom=acc,
           color='#FF6B6B', alpha=0.45, hatch='//', edgecolor='#8B0000',
           label='Gap (|conf − acc|)')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    subtitle = f'ECE = {ece_value:.4f}' if ece_value is not None else ''
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate with sample counts on twin axis
    if count.sum() > 0:
        ax2 = ax.twinx()
        ax2.plot(centres, count / count.sum(), 'o-', color='#555',
                 markersize=4, alpha=0.6, label='Bin fraction')
        ax2.set_ylabel('Sample fraction', fontsize=10, color='#555')
        ax2.tick_params(axis='y', labelcolor='#555')
        ax2.set_ylim(0, max(0.05, float((count / count.sum()).max()) * 1.15))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------- #
#  Temperature Scaling
# --------------------------------------------------------------------------- #

class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling for post-hoc calibration."""

    def __init__(self, initial_T: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> float:
        """Optimise T on a validation set by minimising NLL."""
        self.to(logits.device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def _closure():
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return float(self.temperature.detach().item())


def apply_temperature(probs_or_logits: np.ndarray, T: float, input_is_logits: bool = False) -> np.ndarray:
    """Apply temperature to probabilities or logits."""
    if input_is_logits:
        logits = probs_or_logits
    else:
        # Convert softmax probs back to logits via log (stable enough for visualisation)
        logits = np.log(np.clip(probs_or_logits, 1e-12, 1.0))
    scaled = logits / T
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
