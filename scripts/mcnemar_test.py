"""McNemar's test: is the ensemble statistically better than the best single model?

For each seed, compare ensemble predictions vs. each single model's predictions
on the same 1503 test images. McNemar tests whether the disagreement table is
asymmetric, i.e. whether one classifier is reliably right where the other is
wrong.

Contingency table for two classifiers A and B on the same test set:

                    B correct   B wrong
    A correct       n_aa        n_ab
    A wrong         n_ba        n_bb

McNemar focuses on the off-diagonal (discordant pairs):
    statistic = (|n_ab - n_ba| - 1)^2 / (n_ab + n_ba)     (with continuity correction)
    p-value   = chi^2_1 survival function on `statistic`

Small p (< 0.05) → the two classifiers disagree asymmetrically → one is reliably
better. The sign of (n_ba - n_ab) tells which one (positive = B better).

Usage
-----
    python scripts/mcnemar_test.py
    python scripts/mcnemar_test.py --baseline swin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass


ROOT = Path(__file__).resolve().parent.parent
MULTISEED_DIR = ROOT / 'outputs' / 'multiseed'

SEEDS = [42, 1337, 2024, 7, 11]
BACKBONES = ['eb0', 'rn50', 'dn121', 'swin']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline', type=str, default='all',
                   help='Which single model to compare against ensemble: '
                        f'{BACKBONES} or "all" (default: all).')
    p.add_argument('--seeds', type=str, default=','.join(str(s) for s in SEEDS))
    return p.parse_args()


def _load_predictions(model: str, seed: int) -> pd.DataFrame:
    if model == 'ensemble':
        path = MULTISEED_DIR / f'ensemble_seed{seed}' / 'ensemble_predictions.csv'
    else:
        path = MULTISEED_DIR / f'{model}_seed{seed}' / 'test_predictions.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)[['image_path', 'y_true', 'y_pred']]


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> Dict[str, float]:
    """Run McNemar's test with continuity correction.

    Args:
        a_correct: bool array — A's correctness per sample.
        b_correct: bool array — B's correctness per sample.

    Returns:
        dict with n_ab, n_ba, statistic, p-value, and the sign of (n_ba - n_ab).
    """
    a = a_correct.astype(bool)
    b = b_correct.astype(bool)
    n_ab = int(((a) & (~b)).sum())  # A correct, B wrong
    n_ba = int(((~a) & (b)).sum())  # A wrong, B correct
    n_disc = n_ab + n_ba

    if n_disc == 0:
        return {
            'n_ab': n_ab, 'n_ba': n_ba, 'discordant': 0,
            'statistic': float('nan'), 'p_value': 1.0,
            'sign': 0,
        }

    statistic = (abs(n_ab - n_ba) - 1) ** 2 / n_disc
    p_value = float(stats.chi2.sf(statistic, df=1))
    sign = int(np.sign(n_ba - n_ab))  # +1: B better, -1: A better
    return {
        'n_ab': n_ab, 'n_ba': n_ba, 'discordant': n_disc,
        'statistic': float(statistic), 'p_value': p_value, 'sign': sign,
    }


def compare_for_seed(baseline: str, seed: int) -> Dict:
    """McNemar test: baseline (single model) vs ensemble, for one seed."""
    df_base = _load_predictions(baseline, seed)
    df_ens = _load_predictions('ensemble', seed)

    # Same row order assumption — verify by image_path.
    if not (df_base['image_path'].values == df_ens['image_path'].values).all():
        df_base = df_base.set_index('image_path').loc[df_ens['image_path']].reset_index()

    base_correct = (df_base['y_true'] == df_base['y_pred']).to_numpy()
    ens_correct  = (df_ens['y_true']  == df_ens['y_pred']).to_numpy()

    result = mcnemar(base_correct, ens_correct)
    result['seed'] = seed
    result['baseline'] = baseline
    result['n_samples'] = int(len(df_base))
    result['baseline_acc'] = float(base_correct.mean())
    result['ensemble_acc'] = float(ens_correct.mean())
    return result


def fishers_method_combined_p(p_values: List[float]) -> float:
    """Combine p-values from independent tests via Fisher's method.

    chi^2_combined = -2 * sum(log(p_i)), df = 2k.
    """
    p_arr = np.asarray([max(p, 1e-300) for p in p_values])
    chi2 = -2 * np.log(p_arr).sum()
    df = 2 * len(p_arr)
    return float(stats.chi2.sf(chi2, df=df))


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    baselines = BACKBONES if args.baseline == 'all' else [args.baseline]

    output_dir = MULTISEED_DIR / 'mcnemar'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for baseline in baselines:
        print(f'\n=== McNemar: {baseline} vs ensemble ===\n')
        results_per_seed = []
        for seed in seeds:
            try:
                r = compare_for_seed(baseline, seed)
            except FileNotFoundError as e:
                print(f'  seed={seed}: missing {e}')
                continue
            results_per_seed.append(r)
            sig = '***' if r['p_value'] < 0.001 else \
                  '**' if r['p_value'] < 0.01 else \
                  '*'  if r['p_value'] < 0.05 else 'n.s.'
            direction = 'ens > base' if r['sign'] > 0 else \
                        'base > ens' if r['sign'] < 0 else 'tie'
            print(f'  seed={r["seed"]:>4d}  '
                  f'base_acc={r["baseline_acc"]:.4f}  '
                  f'ens_acc={r["ensemble_acc"]:.4f}  '
                  f'B={r["n_ab"]:>3d} C={r["n_ba"]:>3d}  '
                  f'chi2={r["statistic"]:>6.2f}  '
                  f'p={r["p_value"]:.2e}  {sig}  ({direction})')

        # Combine across seeds with Fisher's method
        p_values = [r['p_value'] for r in results_per_seed]
        combined_p = fishers_method_combined_p(p_values)
        n_significant = sum(p < 0.05 for p in p_values)
        ens_wins = sum(1 for r in results_per_seed if r['sign'] > 0)
        print(f'\n  Combined (Fisher): p = {combined_p:.2e}  '
              f'({n_significant}/{len(p_values)} individual seeds significant at α=0.05; '
              f'ensemble wins {ens_wins}/{len(results_per_seed)} seeds)')

        all_results[baseline] = {
            'per_seed': results_per_seed,
            'combined_p_fisher': combined_p,
            'n_significant': n_significant,
            'ensemble_wins': ens_wins,
            'total_seeds': len(results_per_seed),
        }

    # Write JSON + markdown
    json_path = output_dir / 'mcnemar_results.json'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nWrote {json_path.relative_to(ROOT)}')

    md_lines = ['# McNemar test — ensemble vs single models', '']
    md_lines.append(f'**Seeds:** {seeds}    **n samples per seed:** 1503')
    md_lines.append('')
    md_lines.append(
        'Significance: \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05, n.s. = not significant.')
    md_lines.append('')
    md_lines.append('| Baseline | n seeds | Ensemble wins | Significant (α=0.05) | Combined Fisher p |')
    md_lines.append('|---|---:|---:|---:|---:|')
    for baseline, d in all_results.items():
        md_lines.append(f'| {baseline} vs ensemble | {d["total_seeds"]} | '
                        f'{d["ensemble_wins"]}/{d["total_seeds"]} | '
                        f'{d["n_significant"]}/{d["total_seeds"]} | '
                        f'{d["combined_p_fisher"]:.2e} |')
    md_lines.append('')
    md_lines.append('## Per-seed detail')
    md_lines.append('')
    for baseline, d in all_results.items():
        md_lines.append(f'### {baseline} vs ensemble')
        md_lines.append('')
        md_lines.append('| Seed | base acc | ens acc | n_ab (base✓,ens✗) | n_ba (base✗,ens✓) | χ² | p |')
        md_lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for r in d['per_seed']:
            md_lines.append(f'| {r["seed"]} | {r["baseline_acc"]:.4f} | '
                            f'{r["ensemble_acc"]:.4f} | {r["n_ab"]} | '
                            f'{r["n_ba"]} | {r["statistic"]:.2f} | '
                            f'{r["p_value"]:.2e} |')
        md_lines.append('')

    md_path = output_dir / 'mcnemar_results.md'
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'Wrote {md_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
