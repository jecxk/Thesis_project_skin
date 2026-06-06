"""Multi-seed training runner for thesis-defense statistical robustness.

Runs every backbone with several random seeds, then reports the mean and
standard deviation of the headline metrics so the thesis can report
"89.49 ± σ %" instead of a single point estimate.

Usage
-----
    # End-to-end (train 20 models, ensemble per seed, aggregate)
    python scripts/run_multiseed.py all

    # Or run phases individually
    python scripts/run_multiseed.py train
    python scripts/run_multiseed.py ensemble
    python scripts/run_multiseed.py aggregate

    # Restrict to fewer seeds / specific models for a quick smoke test
    python scripts/run_multiseed.py train --seeds 42,1337 --models eb0,swin

    # Only show what would run, do not execute
    python scripts/run_multiseed.py train --dry-run

Idempotency
-----------
The runner skips any (model, seed) whose ``test_metrics.json`` already exists.
This means you can safely Ctrl-C and resume — the next invocation continues
from where the previous one left off.

Outputs
-------
    outputs/multiseed/
        configs/<model>_seed<N>.yaml          # generated temporary configs
        <model>_seed<N>/                       # one directory per training run
            best.pth, test_metrics.json, ...
        ensemble_seed<N>/
            ensemble_metrics.json
        summary.json                           # aggregated mean ± std
        summary.md                             # human-readable report
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# Force UTF-8 console output on Windows (default cp1252 chokes on → ± ←).
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python < 3.7


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent
MULTISEED_DIR = ROOT / 'outputs' / 'multiseed'
CONFIG_OUT_DIR = MULTISEED_DIR / 'configs'

# (short name, source yaml). The short name is used in output directories.
BACKBONES: Dict[str, str] = {
    'eb0':  'src/configs/efficientnet_b0_optimized.yaml',
    'rn50': 'src/configs/resnet50_v3.yaml',
    'dn121': 'src/configs/densenet121_v3.yaml',
    'swin': 'src/configs/swin_tiny_v3.yaml',
}

DEFAULT_SEEDS: List[int] = [42, 1337, 2024, 7, 11]


# --------------------------------------------------------------------------- #
#  Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Multi-seed training runner for thesis statistical robustness.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('phase', choices=['train', 'ensemble', 'aggregate', 'all'],
                   help='Phase to execute.')
    p.add_argument('--seeds', type=str, default=','.join(str(s) for s in DEFAULT_SEEDS),
                   help='Comma-separated list of seeds (default: 42,1337,2024,7,11).')
    p.add_argument('--models', type=str, default=','.join(BACKBONES.keys()),
                   help=f'Comma-separated subset of backbones to run '
                        f'(choices: {",".join(BACKBONES.keys())}).')
    p.add_argument('--dry-run', action='store_true',
                   help='Print what would happen, do not execute.')
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _run_output_dir(model: str, seed: int) -> Path:
    return MULTISEED_DIR / f'{model}_seed{seed}'


def _config_path(model: str, seed: int) -> Path:
    return CONFIG_OUT_DIR / f'{model}_seed{seed}.yaml'


def _ensemble_output_dir(seed: int) -> Path:
    return MULTISEED_DIR / f'ensemble_seed{seed}'


# --------------------------------------------------------------------------- #
#  Phase 1 — Training
# --------------------------------------------------------------------------- #

def generate_config(model: str, seed: int) -> Path:
    """Create a per-(model, seed) YAML by patching the base config."""
    base_path = ROOT / BACKBONES[model]
    cfg = _load_yaml(base_path)

    out_dir = _run_output_dir(model, seed)
    cfg['seed'] = int(seed)
    cfg['experiment_name'] = f'{model}_seed{seed}'
    cfg['output_dir'] = str(out_dir.relative_to(ROOT)).replace('\\', '/')

    cfg_path = _config_path(model, seed)
    _save_yaml(cfg_path, cfg)
    return cfg_path


def run_training(model: str, seed: int, dry_run: bool) -> str:
    """Return 'skipped' / 'completed' / 'failed' / 'dry-run'."""
    out_dir = _run_output_dir(model, seed)
    metrics_file = out_dir / 'test_metrics.json'

    if metrics_file.exists():
        return 'skipped'

    cfg_path = generate_config(model, seed)
    cmd = [sys.executable, '-m', 'src.train', '--config', str(cfg_path)]

    if dry_run:
        print(f'  [dry-run] {" ".join(cmd)}')
        return 'dry-run'

    print(f'  [run] {model} seed={seed} -> {out_dir.relative_to(ROOT)}')
    # Force UTF-8 in the child Python so train.py can print box-drawing chars
    # (Windows default cp1252 chokes on │ in the epoch log line).
    import os
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if result.returncode != 0:
        print(f'  [FAIL] {model} seed={seed} exited with code {result.returncode}')
        return 'failed'
    return 'completed'


def phase_train(seeds: List[int], models: List[str], dry_run: bool) -> None:
    print(f'\n=== PHASE 1: Training ({len(models)} models × {len(seeds)} seeds = '
          f'{len(models) * len(seeds)} runs) ===\n')
    summary = {'completed': 0, 'skipped': 0, 'failed': 0, 'dry-run': 0}
    for model in models:
        for seed in seeds:
            status = run_training(model, seed, dry_run)
            summary[status] += 1
    print('\n  Training phase summary:', summary)


# --------------------------------------------------------------------------- #
#  Phase 2 — Per-seed ensemble
# --------------------------------------------------------------------------- #

def run_ensemble_for_seed(seed: int, models: List[str], dry_run: bool) -> str:
    """Soft-vote the four backbones for a single seed.

    Implemented inline (rather than calling src.ensemble) so we can point at
    the per-seed checkpoint directories without mutating the canonical script.
    """
    out_dir = _ensemble_output_dir(seed)
    metrics_file = out_dir / 'ensemble_metrics.json'
    if metrics_file.exists():
        return 'skipped'

    # Collect each model's test_predictions.csv (has all probability columns).
    import pandas as pd

    available: List[Path] = []
    for model in models:
        run_dir = _run_output_dir(model, seed)
        pred_csv = run_dir / 'test_predictions.csv'
        if not pred_csv.exists():
            print(f'  [skip] seed={seed}: missing {pred_csv.relative_to(ROOT)}')
            return 'skipped'
        available.append(pred_csv)

    if dry_run:
        print(f'  [dry-run] ensemble seed={seed} from {len(available)} models')
        return 'dry-run'

    print(f'  [run] ensemble seed={seed} ← {len(available)} models')

    # Load each model's probability matrix.
    dfs = [pd.read_csv(p) for p in available]
    class_names = [c[len('prob_'):] for c in dfs[0].columns
                   if c.startswith('prob_')]
    if not class_names:
        raise RuntimeError(f'No prob_* columns found in {available[0]}')

    # Sanity check: every model agrees on row order (same image_path order).
    base_paths = dfs[0]['image_path'].tolist()
    for df, src in zip(dfs[1:], available[1:]):
        if df['image_path'].tolist() != base_paths:
            raise RuntimeError(f'Row order mismatch between models for seed {seed}; '
                               f'first divergence in {src}')

    prob_cols = [f'prob_{c}' for c in class_names]
    stacked = np.stack([df[prob_cols].to_numpy() for df in dfs], axis=0)
    avg_probs = stacked.mean(axis=0)  # (N, C)

    y_true = dfs[0]['y_true'].to_numpy()
    y_pred = avg_probs.argmax(axis=1)

    # Inline metric computation (depends on src.utils.metrics).
    sys.path.insert(0, str(ROOT))
    from src.utils.metrics import compute_classification_metrics
    metrics = compute_classification_metrics(
        y_true.tolist(), y_pred.tolist(), class_names, y_prob=avg_probs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(metrics_file, {
        'seed': int(seed),
        'models': models,
        'class_names': class_names,
        'accuracy': float(metrics['accuracy']),
        'macro_f1': float(metrics['macro_f1']),
        'macro_precision': float(metrics['macro_precision']),
        'macro_recall': float(metrics['macro_recall']),
        'weighted_f1': float(metrics['weighted_f1']),
        'cohen_kappa': float(metrics['cohen_kappa']),
        'mcc': float(metrics['mcc']),
        'auc_macro': float(metrics.get('auc_macro', float('nan'))),
        'per_class_metrics': metrics['per_class_metrics'],
        'confusion_matrix': metrics['confusion_matrix'],
    })

    # Save the averaged predictions for downstream analysis.
    ens_df = dfs[0][['image_path', 'y_true', 'true_name']].copy()
    ens_df['y_pred'] = y_pred
    ens_df['pred_name'] = [class_names[i] for i in y_pred]
    ens_df['correct'] = ens_df['y_true'] == ens_df['y_pred']
    for i, c in enumerate(class_names):
        ens_df[f'prob_{c}'] = avg_probs[:, i].round(6)
    ens_df.to_csv(out_dir / 'ensemble_predictions.csv', index=False)

    return 'completed'


def phase_ensemble(seeds: List[int], models: List[str], dry_run: bool) -> None:
    print(f'\n=== PHASE 2: Per-seed ensemble ({len(seeds)} seeds) ===\n')
    summary = {'completed': 0, 'skipped': 0, 'dry-run': 0}
    for seed in seeds:
        status = run_ensemble_for_seed(seed, models, dry_run)
        summary[status] = summary.get(status, 0) + 1
    print('\n  Ensemble phase summary:', summary)


# --------------------------------------------------------------------------- #
#  Phase 3 — Aggregation
# --------------------------------------------------------------------------- #

METRIC_KEYS = ['accuracy', 'macro_f1', 'auc_macro',
               'weighted_f1', 'cohen_kappa', 'mcc']


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
    if len(values) == 1:
        return {'mean': float(values[0]), 'std': 0.0, 'n': 1}
    return {
        'mean': float(statistics.mean(values)),
        'std':  float(statistics.stdev(values)),
        'n':    len(values),
    }


def _collect_metric_series(paths: List[Path], key: str) -> List[float]:
    out = []
    for p in paths:
        data = _read_json(p)
        if data is None:
            continue
        if key in data and data[key] is not None and not (
                isinstance(data[key], float) and np.isnan(data[key])):
            out.append(float(data[key]))
        elif 'test_results' in data and key in data['test_results']:
            v = data['test_results'][key]
            if v is not None:
                out.append(float(v))
    return out


def _summarise(metrics_paths: List[Path]) -> Dict[str, dict]:
    """Return {metric: {mean, std, n, values}} aggregated across the paths."""
    result: Dict[str, dict] = {}
    for key in METRIC_KEYS:
        series = _collect_metric_series(metrics_paths, key)
        stats = _mean_std(series)
        stats['values'] = series
        result[key] = stats
    return result


def phase_aggregate(seeds: List[int], models: List[str]) -> None:
    print(f'\n=== PHASE 3: Aggregation ===\n')

    summary: Dict[str, dict] = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'seeds': seeds,
        'models': {},
        'ensemble': {},
    }

    # Per-model aggregation
    for model in models:
        paths = [_run_output_dir(model, s) / 'test_metrics.json' for s in seeds]
        summary['models'][model] = _summarise(paths)

    # Ensemble aggregation
    ensemble_paths = [_ensemble_output_dir(s) / 'ensemble_metrics.json'
                      for s in seeds]
    summary['ensemble'] = _summarise(ensemble_paths)

    _save_json(MULTISEED_DIR / 'summary.json', summary)
    _write_markdown_report(MULTISEED_DIR / 'summary.md', summary)
    _print_console_report(summary)


def _fmt(stats: dict, pct: bool = False) -> str:
    n = stats.get('n', 0)
    if n == 0:
        return '—'
    m = stats['mean']
    s = stats['std']
    if pct:
        return f'{m*100:.2f} ± {s*100:.2f} %'
    return f'{m:.4f} ± {s:.4f}'


def _write_markdown_report(path: Path, summary: dict) -> None:
    lines: List[str] = []
    lines.append('# Multi-seed evaluation summary')
    lines.append('')
    lines.append(f'**Generated:** {summary["generated_at"]}')
    lines.append(f'**Seeds:** {summary["seeds"]}')
    lines.append('')
    lines.append('## Per-model results (mean ± std across seeds)')
    lines.append('')
    lines.append('| Model | n | Accuracy | Macro F1 | Macro AUC | Weighted F1 | Kappa | MCC |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for model, stats in summary['models'].items():
        n = stats['accuracy']['n']
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            model, n,
            _fmt(stats['accuracy'], pct=True),
            _fmt(stats['macro_f1']),
            _fmt(stats['auc_macro']),
            _fmt(stats['weighted_f1']),
            _fmt(stats['cohen_kappa']),
            _fmt(stats['mcc']),
        ))
    lines.append('')
    lines.append('## Ensemble (soft voting across the four backbones, per seed)')
    lines.append('')
    es = summary['ensemble']
    lines.append('| Metric | mean ± std (n={n}) |'.format(n=es['accuracy']['n']))
    lines.append('|---|---|')
    lines.append(f'| Accuracy | {_fmt(es["accuracy"], pct=True)} |')
    lines.append(f'| Macro F1 | {_fmt(es["macro_f1"])} |')
    lines.append(f'| Macro AUC | {_fmt(es["auc_macro"])} |')
    lines.append(f'| Weighted F1 | {_fmt(es["weighted_f1"])} |')
    lines.append(f'| Cohen Kappa | {_fmt(es["cohen_kappa"])} |')
    lines.append(f'| MCC | {_fmt(es["mcc"])} |')
    lines.append('')
    lines.append('## Raw values per seed (for the writing)')
    lines.append('')
    for model, stats in summary['models'].items():
        vals = stats['accuracy']['values']
        if vals:
            lines.append(f'- **{model}** accuracy: ' + ', '.join(f'{v:.4f}' for v in vals))
    ens_vals = summary['ensemble']['accuracy']['values']
    if ens_vals:
        lines.append(f'- **ensemble** accuracy: ' + ', '.join(f'{v:.4f}' for v in ens_vals))

    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  Wrote {path.relative_to(ROOT)}')


def _print_console_report(summary: dict) -> None:
    print('\n  ── Per-model (mean ± std) ──')
    for model, stats in summary['models'].items():
        print(f'    {model:>6s}  '
              f'Acc {_fmt(stats["accuracy"], pct=True):>18s}  '
              f'F1 {_fmt(stats["macro_f1"]):>20s}  '
              f'AUC {_fmt(stats["auc_macro"]):>20s}  '
              f'(n={stats["accuracy"]["n"]})')
    es = summary['ensemble']
    print(f'\n  ── Ensemble ──')
    print(f'    Acc {_fmt(es["accuracy"], pct=True)}')
    print(f'    F1  {_fmt(es["macro_f1"])}')
    print(f'    AUC {_fmt(es["auc_macro"])}')
    print(f'    (n={es["accuracy"]["n"]})')
    print(f'\n  Wrote {MULTISEED_DIR / "summary.json"}')


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    for m in models:
        if m not in BACKBONES:
            raise SystemExit(f'Unknown model "{m}"; choices: {list(BACKBONES)}')

    print('Multi-seed runner')
    print(f'  Root:   {ROOT}')
    print(f'  Seeds:  {seeds}')
    print(f'  Models: {models}')
    print(f'  Output: {MULTISEED_DIR.relative_to(ROOT)}')

    MULTISEED_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ('train', 'all'):
        phase_train(seeds, models, args.dry_run)
    if args.phase in ('ensemble', 'all'):
        phase_ensemble(seeds, models, args.dry_run)
    if args.phase in ('aggregate', 'all'):
        phase_aggregate(seeds, models)


if __name__ == '__main__':
    main()
