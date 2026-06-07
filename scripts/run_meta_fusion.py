"""Driver: train the four metadata-fusion models, then ensemble + aggregate.

Mirrors scripts/run_multiseed.py but for the metadata-fusion variant. Idempotent:
each (model) is skipped if its test_metrics.json already exists. After all four
trainings finish, runs soft-voting across them and aggregates the headline
results into outputs/meta_fusion/summary.{json,md}, including a side-by-side
comparison against the image-only ensemble.

Usage
-----
    python scripts/run_meta_fusion.py all
    python scripts/run_meta_fusion.py train --models eb0,rn50
    python scripts/run_meta_fusion.py ensemble
    python scripts/run_meta_fusion.py aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass


ROOT = Path(__file__).resolve().parent.parent
META_DIR = ROOT / 'outputs' / 'meta_fusion'

BACKBONES: Dict[str, str] = {
    'eb0':   'src/configs/meta_fusion/efficientnet_b0_meta.yaml',
    'rn50':  'src/configs/meta_fusion/resnet50_meta.yaml',
    'dn121': 'src/configs/meta_fusion/densenet121_meta.yaml',
    'swin':  'src/configs/meta_fusion/swin_tiny_meta.yaml',
}

OUT_NAMES: Dict[str, str] = {
    'eb0':   'efficientnet_b0_meta',
    'rn50':  'resnet50_meta',
    'dn121': 'densenet121_meta',
    'swin':  'swin_tiny_meta',
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('phase', choices=['train', 'ensemble', 'aggregate', 'all'])
    p.add_argument('--models', type=str, default=','.join(BACKBONES.keys()))
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def _run_dir(model: str) -> Path:
    return META_DIR / OUT_NAMES[model]


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_training(model: str, dry_run: bool) -> str:
    out_dir = _run_dir(model)
    metrics_file = out_dir / 'test_metrics.json'
    if metrics_file.exists():
        return 'skipped'

    cfg_path = ROOT / BACKBONES[model]
    cmd = [sys.executable, '-m', 'src.train_meta', '--config', str(cfg_path)]

    if dry_run:
        print(f'  [dry-run] {" ".join(cmd)}')
        return 'dry-run'

    print(f'  [run] {model} -> {out_dir.relative_to(ROOT)}')
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if result.returncode != 0:
        print(f'  [FAIL] {model} exited with code {result.returncode}')
        return 'failed'
    return 'completed'


def phase_train(models: List[str], dry_run: bool) -> None:
    print(f'\n=== PHASE 1: Metadata-fusion training ({len(models)} models) ===\n')
    summary = {'completed': 0, 'skipped': 0, 'failed': 0, 'dry-run': 0}
    for m in models:
        summary[run_training(m, dry_run)] += 1
    print('\n  Training phase summary:', summary)


# --------------------------------------------------------------------------- #
#  Phase 2 — Ensemble across the four metadata-fusion models
# --------------------------------------------------------------------------- #

def phase_ensemble(models: List[str], dry_run: bool) -> str:
    out_dir = META_DIR / 'ensemble'
    metrics_file = out_dir / 'ensemble_metrics.json'
    if metrics_file.exists():
        print('  [skip] meta-fusion ensemble already computed')
        return 'skipped'

    available = []
    for m in models:
        pred_csv = _run_dir(m) / 'test_predictions.csv'
        if not pred_csv.exists():
            print(f'  [skip] {m} missing test_predictions.csv')
            return 'skipped'
        available.append(pred_csv)

    if dry_run:
        print(f'  [dry-run] ensemble from {len(available)} models')
        return 'dry-run'

    print(f'  [run] ensemble from {len(available)} meta-fusion models')

    dfs = [pd.read_csv(p) for p in available]
    class_names = [c[len('prob_'):] for c in dfs[0].columns
                   if c.startswith('prob_')]

    base_paths = dfs[0]['image_path'].tolist()
    for df, src in zip(dfs[1:], available[1:]):
        if df['image_path'].tolist() != base_paths:
            df_aligned = df.set_index('image_path').loc[base_paths].reset_index()
            df.update(df_aligned)

    prob_cols = [f'prob_{c}' for c in class_names]
    stacked = np.stack([df[prob_cols].to_numpy() for df in dfs], axis=0)
    avg_probs = stacked.mean(axis=0)

    y_true = dfs[0]['y_true'].to_numpy()
    y_pred = avg_probs.argmax(axis=1)

    sys.path.insert(0, str(ROOT))
    from src.utils.metrics import compute_classification_metrics
    metrics = compute_classification_metrics(
        y_true.tolist(), y_pred.tolist(), class_names, y_prob=avg_probs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(metrics_file, {
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

    ens_df = dfs[0][['image_path', 'y_true', 'true_name']].copy()
    ens_df['y_pred'] = y_pred
    ens_df['pred_name'] = [class_names[i] for i in y_pred]
    ens_df['correct'] = ens_df['y_true'] == ens_df['y_pred']
    for i, c in enumerate(class_names):
        ens_df[f'prob_{c}'] = avg_probs[:, i].round(6)
    ens_df.to_csv(out_dir / 'ensemble_predictions.csv', index=False)

    print(f'  Wrote {metrics_file.relative_to(ROOT)}')
    print(f'  Ensemble: Acc={metrics["accuracy"]:.4f}  '
          f'F1={metrics["macro_f1"]:.4f}  '
          f'AUC={metrics.get("auc_macro", float("nan")):.4f}')
    return 'completed'


# --------------------------------------------------------------------------- #
#  Phase 3 — Aggregation + comparison
# --------------------------------------------------------------------------- #

def phase_aggregate(models: List[str]) -> None:
    print('\n=== PHASE 3: Aggregation ===\n')

    summary: Dict[str, dict] = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'models': {},
        'ensemble': None,
        'baseline_seed42': None,
    }

    for m in models:
        path = _run_dir(m) / 'test_metrics.json'
        data = _read_json(path)
        if data is None:
            continue
        summary['models'][m] = {
            'accuracy': data.get('accuracy'),
            'macro_f1': data.get('macro_f1'),
            'auc_macro': data.get('auc_macro'),
            'cohen_kappa': data.get('cohen_kappa'),
            'mcc': data.get('mcc'),
        }

    ens_path = META_DIR / 'ensemble' / 'ensemble_metrics.json'
    if ens_path.exists():
        ed = _read_json(ens_path)
        summary['ensemble'] = {
            'accuracy': ed['accuracy'], 'macro_f1': ed['macro_f1'],
            'auc_macro': ed['auc_macro'], 'cohen_kappa': ed['cohen_kappa'],
            'mcc': ed['mcc'],
        }

    base_path = ROOT / 'outputs' / 'multiseed' / 'ensemble_seed42' / 'ensemble_metrics.json'
    if base_path.exists():
        bd = _read_json(base_path)
        summary['baseline_seed42'] = {
            'accuracy': bd['accuracy'], 'macro_f1': bd['macro_f1'],
            'auc_macro': bd['auc_macro'],
        }

    _save_json(META_DIR / 'summary.json', summary)
    _write_md(META_DIR / 'summary.md', summary)
    _print(summary)


def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v*100:.2f} %' if pct else f'{v:.4f}'


def _write_md(path: Path, s: dict) -> None:
    lines = ['# Metadata-fusion evaluation summary', '',
             f'**Generated:** {s["generated_at"]}', '',
             '## Per-model results (metadata-fusion)', '',
             '| Model | Accuracy | Macro F1 | Macro AUC | Cohen Kappa | MCC |',
             '|---|---:|---:|---:|---:|---:|']
    for name, d in s['models'].items():
        lines.append(f'| {name} | {_fmt(d["accuracy"], pct=True)} | '
                     f'{_fmt(d["macro_f1"])} | {_fmt(d["auc_macro"])} | '
                     f'{_fmt(d["cohen_kappa"])} | {_fmt(d["mcc"])} |')
    lines += ['', '## Ensemble', '']
    if s['ensemble']:
        e = s['ensemble']
        lines += ['| Metric | Value |', '|---|---|',
                  f'| Accuracy | **{_fmt(e["accuracy"], pct=True)}** |',
                  f'| Macro F1 | **{_fmt(e["macro_f1"])}** |',
                  f'| Macro AUC | **{_fmt(e["auc_macro"])}** |',
                  f'| Cohen Kappa | {_fmt(e["cohen_kappa"])} |',
                  f'| MCC | {_fmt(e["mcc"])} |', '']
    lines += ['## Comparison vs.\\ image-only baseline (seed 42 ensemble)', '']
    if s['baseline_seed42'] and s['ensemble']:
        b = s['baseline_seed42']
        e = s['ensemble']
        d_acc = (e['accuracy'] - b['accuracy']) * 100
        d_f1 = e['macro_f1'] - b['macro_f1']
        d_auc = e['auc_macro'] - b['auc_macro']
        lines += ['| Metric | Image-only baseline | + Metadata fusion | Δ |',
                  '|---|---:|---:|---:|',
                  f'| Accuracy | {_fmt(b["accuracy"], pct=True)} | '
                  f'{_fmt(e["accuracy"], pct=True)} | {d_acc:+.2f} pp |',
                  f'| Macro F1 | {_fmt(b["macro_f1"])} | '
                  f'{_fmt(e["macro_f1"])} | {d_f1:+.4f} |',
                  f'| Macro AUC | {_fmt(b["auc_macro"])} | '
                  f'{_fmt(e["auc_macro"])} | {d_auc:+.4f} |']
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  Wrote {path.relative_to(ROOT)}')


def _print(s: dict) -> None:
    print('\n  ── Per-model ──')
    for name, d in s['models'].items():
        print(f'    {name:>5s}  Acc={_fmt(d["accuracy"], pct=True):>10s}  '
              f'F1={_fmt(d["macro_f1"]):>8s}  '
              f'AUC={_fmt(d["auc_macro"]):>8s}')
    if s['ensemble']:
        e = s['ensemble']
        print(f'\n  ── Meta-fusion ensemble ──')
        print(f'    Acc {_fmt(e["accuracy"], pct=True)}')
        print(f'    F1  {_fmt(e["macro_f1"])}')
        print(f'    AUC {_fmt(e["auc_macro"])}')
    if s['baseline_seed42'] and s['ensemble']:
        b = s['baseline_seed42']
        e = s['ensemble']
        d_acc = (e['accuracy'] - b['accuracy']) * 100
        d_f1 = e['macro_f1'] - b['macro_f1']
        print(f'\n  ── vs.\\ image-only seed=42 baseline ──')
        print(f'    ΔAcc = {d_acc:+.2f} pp,  ΔF1 = {d_f1:+.4f}')


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    for m in models:
        if m not in BACKBONES:
            raise SystemExit(f'Unknown model "{m}"; choices: {list(BACKBONES)}')

    print('Metadata-fusion runner')
    print(f'  Models: {models}')
    print(f'  Output: {META_DIR.relative_to(ROOT)}')
    META_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ('train', 'all'):
        phase_train(models, args.dry_run)
    if args.phase in ('ensemble', 'all'):
        phase_ensemble(models, args.dry_run)
    if args.phase in ('aggregate', 'all'):
        phase_aggregate(models)


if __name__ == '__main__':
    main()
