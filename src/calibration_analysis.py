"""Run calibration analysis for all trained models + ensemble.

For each model:
  1. Load softmax probabilities from its stored test_predictions.csv
     (so we don't re-run inference unnecessarily).
  2. Compute ECE, MCE, Brier, NLL.
  3. Plot reliability diagram.

Temperature scaling:
  - Fit T on validation set if val predictions are available, OR
  - Fit T on test predictions (only for analysis; report both raw & calibrated).

Usage
-----
python -m src.calibration_analysis --output-dir outputs/calibration
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.utils.calibration import (
    TemperatureScaler,
    apply_temperature,
    compute_calibration_metrics,
    plot_reliability_diagram,
)
from src.utils.io import ensure_dir, save_json


# Models and where their test-set prediction CSVs live.
MODEL_PRED_SOURCES = [
    {
        'name': 'EfficientNet-B0',
        'test_pred': 'outputs/efficientnet_b0_v3(main_best)/test_predictions.csv',
    },
    {
        'name': 'ResNet50',
        'test_pred': 'outputs/resnet50_v3(main)/test_predictions.csv',
    },
    {
        'name': 'DenseNet121',
        'test_pred': 'outputs/densenet121_v3(main)/test_predictions.csv',
    },
    {
        'name': 'Swin-Tiny',
        'test_pred': 'outputs/swin_tiny_v3/test_predictions.csv',
    },
    {
        'name': 'Ensemble',
        'test_pred': 'outputs/ensemble/ensemble_predictions.csv',
    },
]


def load_probs_labels(csv_path: Path) -> tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if not prob_cols:
        raise ValueError(f'{csv_path} has no prob_* columns.')
    class_names = [c.replace('prob_', '') for c in prob_cols]
    probs = df[prob_cols].to_numpy().astype(np.float32)
    labels = df['y_true'].to_numpy().astype(int)
    return probs, labels, class_names


def run_one_model(name: str, csv_path: Path, output_dir: Path,
                  n_bins: int = 15, fit_temperature: bool = True) -> Dict:
    probs, labels, class_names = load_probs_labels(csv_path)
    num_classes = len(class_names)

    # Raw calibration
    raw_metrics, raw_bins = compute_calibration_metrics(probs, labels, num_classes, n_bins=n_bins)
    plot_reliability_diagram(
        bin_stats=raw_bins,
        save_path=output_dir / f'{_slug(name)}_reliability_raw.png',
        title=f'Reliability — {name} (Raw)',
        ece_value=raw_metrics.ece,
    )

    result: Dict[str, object] = {
        'model': name,
        'source_csv': str(csv_path),
        'n_samples': int(len(labels)),
        'raw': raw_metrics.to_dict(),
    }

    if fit_temperature:
        # Fit T on half, evaluate on the other half (simple 50/50 split for analysis).
        # For a proper use-case you'd fit on VAL; here we simulate when no VAL preds exist.
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(labels))
        half = len(idx) // 2
        fit_idx, eval_idx = idx[:half], idx[half:]

        # Recover logits from probabilities via log (stable enough)
        logits_fit = torch.log(torch.tensor(np.clip(probs[fit_idx], 1e-12, 1.0)))
        labels_fit = torch.tensor(labels[fit_idx], dtype=torch.long)
        scaler = TemperatureScaler(initial_T=1.0)
        T = scaler.fit(logits_fit, labels_fit)
        scaler.eval()

        # Apply T to eval half for unbiased reporting
        calibrated_eval = apply_temperature(probs[eval_idx], T=T, input_is_logits=False)
        cal_metrics_eval, cal_bins_eval = compute_calibration_metrics(
            calibrated_eval, labels[eval_idx], num_classes, n_bins=n_bins
        )

        # Full-set calibrated probs (for downstream use)
        calibrated_full = apply_temperature(probs, T=T, input_is_logits=False)
        cal_metrics_full, cal_bins_full = compute_calibration_metrics(
            calibrated_full, labels, num_classes, n_bins=n_bins
        )

        plot_reliability_diagram(
            bin_stats=cal_bins_full,
            save_path=output_dir / f'{_slug(name)}_reliability_calibrated.png',
            title=f'Reliability — {name} (T-scaled, T={T:.3f})',
            ece_value=cal_metrics_full.ece,
        )

        # Save calibrated predictions CSV
        df_cal = pd.read_csv(csv_path)
        for i, cls in enumerate(class_names):
            df_cal[f'prob_{cls}'] = calibrated_full[:, i].round(6)
        df_cal.to_csv(output_dir / f'{_slug(name)}_calibrated_predictions.csv', index=False)

        result['temperature'] = float(T)
        result['calibrated_eval_half'] = cal_metrics_eval.to_dict()
        result['calibrated_full'] = cal_metrics_full.to_dict()

    save_json(output_dir / f'{_slug(name)}_calibration.json', result)
    return result


def _slug(name: str) -> str:
    return name.lower().replace(' ', '_').replace('-', '_')


# --------------------------------------------------------------------------- #
#  Cross-model summary plot
# --------------------------------------------------------------------------- #

def plot_calibration_summary(results: List[Dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r['model'] for r in results]
    raw_ece = [r['raw']['ece'] for r in results]
    raw_brier = [r['raw']['brier_score'] for r in results]
    cal_ece = [r['calibrated_full']['ece'] if 'calibrated_full' in r else None for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].bar(x - width / 2, raw_ece, width, color='#FF6B6B', label='Raw ECE', alpha=0.85)
    if any(v is not None for v in cal_ece):
        cal_vals = [v if v is not None else 0 for v in cal_ece]
        axes[0].bar(x + width / 2, cal_vals, width, color='#4ECDC4',
                    label='After T-scaling', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha='right')
    axes[0].set_ylabel('ECE')
    axes[0].set_title('Expected Calibration Error (lower = better)',
                      fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()
    for i, v in enumerate(raw_ece):
        axes[0].text(i - width / 2, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)

    axes[1].bar(x, raw_brier, color='#45B7D1', alpha=0.85, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha='right')
    axes[1].set_ylabel('Brier Score')
    axes[1].set_title('Multiclass Brier Score (lower = better)',
                      fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(raw_brier):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='outputs/calibration')
    parser.add_argument('--no-temperature', action='store_true',
                        help='Skip temperature scaling (only report raw metrics).')
    parser.add_argument('--n-bins', type=int, default=15)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    print('=' * 70)
    print('  Calibration Analysis')
    print(f'  Output: {output_dir}')
    print('=' * 70)

    results: List[Dict] = []
    for entry in MODEL_PRED_SOURCES:
        csv_path = Path(entry['test_pred'])
        if not csv_path.exists():
            print(f'  [SKIP] {entry["name"]}: {csv_path} not found.')
            continue
        print(f'  → {entry["name"]}  ({csv_path.name})')
        res = run_one_model(
            name=entry['name'],
            csv_path=csv_path,
            output_dir=output_dir,
            n_bins=args.n_bins,
            fit_temperature=not args.no_temperature,
        )
        results.append(res)
        print(f'    Raw ECE: {res["raw"]["ece"]:.4f}   Brier: {res["raw"]["brier_score"]:.4f}   '
              f'NLL: {res["raw"]["nll"]:.4f}')
        if 'calibrated_full' in res:
            print(f'    T = {res["temperature"]:.3f}   Calibrated ECE: {res["calibrated_full"]["ece"]:.4f}')

    if not results:
        print('No models available for calibration analysis.')
        return

    # Cross-model summary
    plot_calibration_summary(results, output_dir / 'calibration_summary.png')
    save_json(output_dir / 'calibration_summary.json',
              {'results': results})
    print(f'\n  ✓ Summary saved to {output_dir / "calibration_summary.png"}')
    print('=' * 70)


if __name__ == '__main__':
    main()
