"""Error & Confusion-Pair Analysis.

Input: ensemble_predictions.csv (produced by src.ensemble) or any
per-image prediction CSV with columns:
    image_path, y_true, y_pred, true_name, pred_name,
    prob_<class_0>, ..., prob_<class_K-1>

Outputs (written under --output-dir):
  * confusion_pairs.csv         — most common (true, pred) pairs with counts
  * confusion_pairs.png         — bar chart of top confusion pairs
  * hard_examples.csv           — N lowest-confidence misclassifications
  * per_class_error_rate.png    — error rate per class bar chart
  * ensemble_vs_single.csv      — (optional) examples fixed/broken by ensemble

Usage
-----
# Basic error analysis on ensemble predictions
python -m src.error_analysis \
    --predictions outputs/ensemble/ensemble_predictions.csv \
    --output-dir outputs/error_analysis

# Compare ensemble vs single models
python -m src.error_analysis \
    --predictions outputs/ensemble/ensemble_predictions.csv \
    --single-pred outputs/efficientnet_b0_v3(main_best)/test_predictions.csv \
    --single-name EfficientNet-B0 \
    --output-dir outputs/error_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, save_json


# --------------------------------------------------------------------------- #
#  Confusion pair analysis
# --------------------------------------------------------------------------- #

def compute_confusion_pairs(df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Count (true, pred) occurrences where true != pred, sorted desc."""
    err = df[df['y_true'] != df['y_pred']]
    pairs = (err.groupby(['true_name', 'pred_name'])
               .size()
               .reset_index(name='count')
               .sort_values('count', ascending=False)
               .head(top_k))
    pairs['pair'] = pairs['true_name'] + ' → ' + pairs['pred_name']
    return pairs


def plot_confusion_pairs(pairs: pd.DataFrame, save_path: Path, title: str = 'Top Confusion Pairs') -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(pairs))))
    y = np.arange(len(pairs))
    bars = ax.barh(y, pairs['count'], color='#FF6B6B', alpha=0.85, edgecolor='black')
    ax.set_yticks(y)
    ax.set_yticklabels(pairs['pair'])
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    for bar, val in zip(bars, pairs['count']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{int(val)}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------- #
#  Per-class error rate
# --------------------------------------------------------------------------- #

def plot_per_class_error_rate(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    grp = df.groupby('true_name').agg(
        total=('y_true', 'size'),
        correct=('correct', 'sum'),
    ).reset_index()
    grp['error_rate'] = 1.0 - grp['correct'] / grp['total']
    grp = grp.sort_values('error_rate', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(grp['true_name'], grp['error_rate'], color='#45B7D1',
                  alpha=0.85, edgecolor='black')
    for bar, val, n in zip(bars, grp['error_rate'], grp['total']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.2%}\n(n={int(n)})', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('Per-Class Error Rate', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(0.05, grp['error_rate'].max() * 1.25))
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    return grp


# --------------------------------------------------------------------------- #
#  Hard examples
# --------------------------------------------------------------------------- #

def extract_hard_examples(df: pd.DataFrame, class_names: List[str], top_n: int = 40) -> pd.DataFrame:
    """Return misclassified rows ordered by lowest confidence on predicted class."""
    prob_cols = [f'prob_{c}' for c in class_names]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Prediction CSV is missing probability columns: {missing}')

    err = df[df['y_true'] != df['y_pred']].copy()
    if err.empty:
        return err

    probs = err[prob_cols].to_numpy()
    pred_conf = probs[np.arange(len(err)), err['y_pred'].to_numpy()]
    true_prob = probs[np.arange(len(err)), err['y_true'].to_numpy()]
    err['pred_confidence'] = pred_conf
    err['true_class_prob'] = true_prob
    err['margin'] = pred_conf - true_prob

    # Hard = low pred_confidence AND small margin → model was uncertain yet wrong
    err = err.sort_values(['pred_confidence', 'margin'], ascending=[True, True])

    cols = ['image_path', 'true_name', 'pred_name',
            'pred_confidence', 'true_class_prob', 'margin'] + prob_cols
    return err[cols].head(top_n)


# --------------------------------------------------------------------------- #
#  Ensemble vs single comparison
# --------------------------------------------------------------------------- #

def compare_ensemble_vs_single(
    ensemble_df: pd.DataFrame,
    single_df: pd.DataFrame,
    single_name: str = 'Single',
) -> dict:
    merged = ensemble_df.merge(
        single_df[['image_path', 'y_pred', 'pred_name']].rename(columns={
            'y_pred': f'{single_name}_y_pred',
            'pred_name': f'{single_name}_pred_name',
        }),
        on='image_path', how='inner'
    )

    merged[f'{single_name}_correct'] = (
        merged[f'{single_name}_y_pred'] == merged['y_true']
    )

    fixed = merged[(~merged[f'{single_name}_correct']) & (merged['correct'])]
    broken = merged[(merged[f'{single_name}_correct']) & (~merged['correct'])]
    agree_correct = merged[(merged[f'{single_name}_correct']) & (merged['correct'])]
    agree_wrong = merged[(~merged[f'{single_name}_correct']) & (~merged['correct'])]

    summary = {
        'single_model': single_name,
        'n_total': int(len(merged)),
        'single_correct': int(merged[f'{single_name}_correct'].sum()),
        'ensemble_correct': int(merged['correct'].sum()),
        'ensemble_fixes_single': int(len(fixed)),
        'ensemble_breaks_single': int(len(broken)),
        'both_correct': int(len(agree_correct)),
        'both_wrong': int(len(agree_wrong)),
        'single_accuracy': float(merged[f'{single_name}_correct'].mean()),
        'ensemble_accuracy': float(merged['correct'].mean()),
    }
    return summary, fixed, broken


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True,
                        help='CSV with columns including y_true, y_pred, prob_*')
    parser.add_argument('--single-pred', type=str, default=None,
                        help='(optional) single-model prediction CSV for comparison')
    parser.add_argument('--single-name', type=str, default='Single')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--top-k-pairs', type=int, default=15)
    parser.add_argument('--top-n-hard', type=int, default=40)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f'Predictions not found: {pred_path}')

    output_dir = ensure_dir(Path(args.output_dir))

    df = pd.read_csv(pred_path)
    # Backfill 'correct' if missing
    if 'correct' not in df.columns:
        df['correct'] = df['y_true'] == df['y_pred']

    class_names: List[str] = []
    if 'true_name' in df.columns:
        class_names = sorted(df['true_name'].dropna().unique().tolist())
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if prob_cols and not class_names:
        class_names = [c.replace('prob_', '') for c in prob_cols]

    print('=' * 70)
    print('  Error Analysis')
    print(f'  Predictions: {pred_path}')
    print(f'  Classes: {class_names}')
    print(f'  Output: {output_dir}')
    print('=' * 70)

    # 1. Confusion pairs
    pairs = compute_confusion_pairs(df, top_k=args.top_k_pairs)
    pairs.to_csv(output_dir / 'confusion_pairs.csv', index=False)
    plot_confusion_pairs(pairs, output_dir / 'confusion_pairs.png',
                         title=f'Top-{args.top_k_pairs} Confusion Pairs')
    print(f'  [1/4] Confusion pairs → {output_dir / "confusion_pairs.png"}')

    # 2. Per-class error rate
    per_class_err = plot_per_class_error_rate(df, output_dir / 'per_class_error_rate.png')
    per_class_err.to_csv(output_dir / 'per_class_error_rate.csv', index=False)
    print(f'  [2/4] Per-class error rate → {output_dir / "per_class_error_rate.png"}')

    # 3. Hard examples
    if prob_cols:
        hard = extract_hard_examples(df, class_names=[c.replace('prob_', '') for c in prob_cols],
                                     top_n=args.top_n_hard)
        hard.to_csv(output_dir / 'hard_examples.csv', index=False)
        print(f'  [3/4] Hard examples (top {len(hard)}) → {output_dir / "hard_examples.csv"}')
    else:
        print('  [3/4] Skipping hard examples (no prob_* columns in predictions)')

    # 4. Ensemble vs single comparison
    if args.single_pred:
        single_path = Path(args.single_pred)
        if not single_path.exists():
            print(f'  [4/4] Skipping single comparison: {single_path} not found')
        else:
            single_df = pd.read_csv(single_path)
            summary, fixed, broken = compare_ensemble_vs_single(
                ensemble_df=df, single_df=single_df, single_name=args.single_name,
            )
            save_json(output_dir / f'ensemble_vs_{args.single_name.lower()}.json', summary)
            fixed.to_csv(output_dir / f'ensemble_fixes_{args.single_name.lower()}.csv', index=False)
            broken.to_csv(output_dir / f'ensemble_breaks_{args.single_name.lower()}.csv', index=False)

            print('\n  [4/4] Ensemble vs {}:'.format(args.single_name))
            print(f'    Single accuracy:   {summary["single_accuracy"]:.4f}')
            print(f'    Ensemble accuracy: {summary["ensemble_accuracy"]:.4f}')
            print(f'    Fixed by ensemble: {summary["ensemble_fixes_single"]}')
            print(f'    Broken by ensemble: {summary["ensemble_breaks_single"]}')
    else:
        print('  [4/4] No single-model prediction provided, skipping comparison')

    print('=' * 70)
    print(f'  ✓ Error analysis complete — outputs in {output_dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
