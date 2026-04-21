"""Quantitative evaluation of Grad-CAM heatmaps.

Measures how "focused" a model's attention is on the lesion region.
Since ISIC 2018 does not provide segmentation masks by default, we use:
  1. Otsu thresholding on the grayscale image to approximate a lesion mask
     (dermoscopy images usually have a darker lesion on lighter skin).
  2. Compute metrics comparing the heatmap with this pseudo-mask.

Metrics
-------
* mean_activation_in / mean_activation_out: average heatmap intensity
    inside vs outside the pseudo-lesion region.
* focus_ratio = mean_in / (mean_in + mean_out): higher = more focused.
* entropy: Shannon entropy of the normalized heatmap. Lower = more concentrated.
* peak_distance: Euclidean distance (in pixels) between the heatmap's
    argmax and the centroid of the pseudo-lesion, normalised by image size.
* coverage_75: fraction of image area covered by the top-25% activation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import cv2
import numpy as np


@dataclass
class GradCamQuantMetrics:
    mean_activation_in: float
    mean_activation_out: float
    focus_ratio: float
    entropy: float
    peak_distance: float
    coverage_75: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_pseudo_lesion_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Approximate a lesion binary mask using Otsu thresholding.

    Args:
        image_rgb: (H, W, 3) RGB image, uint8.

    Returns:
        mask: (H, W) uint8 in {0, 1}. 1 = lesion region.
    """
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb

    # Otsu automatically picks an intensity threshold.
    # Lesions tend to be darker → invert so lesion = 1.
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Simple morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask.astype(np.uint8)


def _shannon_entropy(heatmap: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy of the heatmap's intensity histogram (nats)."""
    hist, _ = np.histogram(heatmap, bins=bins, range=(0.0, 1.0), density=True)
    hist = hist / (hist.sum() + 1e-12)
    # Only count non-zero bins
    nz = hist[hist > 0]
    return float(-np.sum(nz * np.log(nz)))


def compute_gradcam_metrics(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    lesion_mask: np.ndarray | None = None,
) -> GradCamQuantMetrics:
    """Compute quantitative metrics for a single Grad-CAM heatmap.

    Args:
        heatmap: 2D array in [0, 1] or scaleable. If not already (H, W) matching
            ``original_image``, it will be bilinearly resized.
        original_image: (H, W, 3) RGB image in [0, 255].
        lesion_mask: (H, W) uint8 binary mask. If None, computed with Otsu.

    Returns:
        GradCamQuantMetrics
    """
    h, w = original_image.shape[:2]

    # Resize + normalise heatmap
    hm = heatmap.astype(np.float32)
    if hm.shape != (h, w):
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = hm - hm.min()
    if hm.max() > 0:
        hm = hm / hm.max()

    # Mask
    if lesion_mask is None:
        lesion_mask = compute_pseudo_lesion_mask(original_image)

    mask_bool = lesion_mask.astype(bool)
    if mask_bool.sum() == 0 or (~mask_bool).sum() == 0:
        # Degenerate mask → fallback: use centre 60%
        cy, cx = h // 2, w // 2
        ry, rx = int(h * 0.3), int(w * 0.3)
        mask_bool = np.zeros_like(mask_bool)
        mask_bool[cy - ry:cy + ry, cx - rx:cx + rx] = True

    mean_in = float(hm[mask_bool].mean())
    mean_out = float(hm[~mask_bool].mean())
    denom = mean_in + mean_out + 1e-12
    focus_ratio = mean_in / denom

    entropy = _shannon_entropy(hm)

    # Peak distance (normalised by image diagonal)
    peak_idx = np.unravel_index(np.argmax(hm), hm.shape)
    ys, xs = np.where(mask_bool)
    cy = float(ys.mean())
    cx = float(xs.mean())
    diag = float(np.sqrt(h * h + w * w))
    peak_distance = float(np.sqrt((peak_idx[0] - cy) ** 2 + (peak_idx[1] - cx) ** 2) / diag)

    # Coverage: fraction of pixels above 75th percentile
    top_thresh = float(np.quantile(hm, 0.75))
    coverage_75 = float((hm >= top_thresh).mean())

    return GradCamQuantMetrics(
        mean_activation_in=mean_in,
        mean_activation_out=mean_out,
        focus_ratio=focus_ratio,
        entropy=entropy,
        peak_distance=peak_distance,
        coverage_75=coverage_75,
    )


def aggregate_metrics(records: list[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-image Grad-CAM metrics by model or class.

    Args:
        records: list of dicts, each containing the metric fields and
            optionally ``model`` and ``class_name`` keys.

    Returns:
        dict with keys 'overall', 'by_model', 'by_class' containing
        mean/std summaries.
    """
    if not records:
        return {'overall': {}, 'by_model': {}, 'by_class': {}}

    metric_keys = [
        'mean_activation_in',
        'mean_activation_out',
        'focus_ratio',
        'entropy',
        'peak_distance',
        'coverage_75',
    ]

    def _summary(subset: list[Dict[str, float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in metric_keys:
            vals = [r[k] for r in subset if k in r]
            if vals:
                out[f'{k}_mean'] = float(np.mean(vals))
                out[f'{k}_std'] = float(np.std(vals))
        out['n'] = len(subset)
        return out

    overall = _summary(records)

    by_model: Dict[str, Dict[str, float]] = {}
    if any('model' in r for r in records):
        models = sorted({r['model'] for r in records if 'model' in r})
        for m in models:
            by_model[m] = _summary([r for r in records if r.get('model') == m])

    by_class: Dict[str, Dict[str, float]] = {}
    if any('class_name' in r for r in records):
        cls = sorted({r['class_name'] for r in records if 'class_name' in r})
        for c in cls:
            by_class[c] = _summary([r for r in records if r.get('class_name') == c])

    return {'overall': overall, 'by_model': by_model, 'by_class': by_class}
