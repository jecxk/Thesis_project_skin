"""Compute **quantitative** Grad-CAM focus metrics across models & classes.

Motivation
----------
Qualitative Grad-CAM overlays are nice pictures but do not scientifically
support claims like "Swin looks at lesion center while ResNet50 gets
distracted by corners". We therefore compute four quantitative
metrics on a held-out test set:
  * focus_ratio       — fraction of activation inside a pseudo-lesion mask
  * mean_act_inside   — average activation value inside the mask
  * entropy           — spatial entropy of the heatmap (lower = sharper)
  * peak_distance     — distance of the heatmap peak to the mask centroid
  * coverage_75       — area where heatmap ≥ 0.75 of its max (fraction)

Pseudo-lesion masks are built with Otsu thresholding + morphological
closing, which is a standard proxy when pixel-level ground-truth masks
are unavailable (ISIC 2018 Task 1 masks exist but cover only part of the
test set).

Usage
-----
# All models, N=500 random test images, class-stratified
python -m src.gradcam_quant_runner \
    --n-per-class 80 \
    --output-dir outputs/gradcam_quant

# Single model
python -m src.gradcam_quant_runner \
    --models EfficientNet-B0 \
    --n-per-class 80 \
    --output-dir outputs/gradcam_quant_eb0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.data.dataset import load_split_dataframes
from src.data.transforms import build_transforms
from src.models.build_model import build_model
from src.utils.config import load_config
from src.utils.grad_cam import GradCAM, get_target_layer
from src.utils.gradcam_quant import (
    compute_gradcam_metrics,
    compute_pseudo_lesion_mask,
)
from src.utils.io import ensure_dir, save_json


MODELS = [
    {
        'name': 'EfficientNet-B0',
        'config': 'src/configs/efficientnet_b0_optimized.yaml',
        'checkpoint': 'outputs/efficientnet_b0_v3(main_best)/best.pth',
    },
    {
        'name': 'ResNet50',
        'config': 'src/configs/resnet50_v3.yaml',
        'checkpoint': 'outputs/resnet50_v3(main)/best.pth',
    },
    {
        'name': 'DenseNet121',
        'config': 'src/configs/densenet121_v3.yaml',
        'checkpoint': 'outputs/densenet121_v3(main)/best.pth',
    },
    {
        'name': 'Swin-Tiny',
        'config': 'src/configs/swin_tiny_v3.yaml',
        'checkpoint': 'outputs/swin_tiny_v3/best.pth',
    },
]


def sample_per_class(test_df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    """Stratified sample of at most `n_per_class` images per class."""
    rng = np.random.default_rng(seed)
    chunks = []
    for lbl, grp in test_df.groupby('label'):
        if len(grp) <= n_per_class:
            chunks.append(grp)
        else:
            idx = rng.choice(len(grp), size=n_per_class, replace=False)
            chunks.append(grp.iloc[idx])
    return pd.concat(chunks, ignore_index=True)


def run_model(
    name: str, cfg_path: str, ckpt_path: str,
    test_df: pd.DataFrame, output_dir: Path,
) -> pd.DataFrame:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = load_config(cfg_path)
    class_names = cfg['classes']
    image_size = int(cfg['train']['image_size'])
    root_dir = Path(cfg['paths']['root_dir'])

    model = build_model(cfg['model'])
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    target_layer = get_target_layer(model, cfg['model']['name'])
    cam = GradCAM(model, target_layer)

    tfm = build_transforms(
        image_size=image_size,
        aug_cfg=cfg.get('augmentation', {}).get('eval', {}),
        is_train=False,
    )

    rows: List[Dict] = []
    try:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f'Grad-CAM[{name}]'):
            img_path = row['image_path']
            full_path = Path(img_path)
            if not full_path.is_absolute():
                full_path = root_dir / full_path
            if not full_path.exists():
                continue

            try:
                img = Image.open(full_path).convert('RGB').resize((image_size, image_size))
            except Exception:
                continue
            img_np = np.array(img)

            x = tfm(Image.open(full_path).convert('RGB')).unsqueeze(0).to(device)
            try:
                heatmap, pred_class, conf = cam(x, target_class=None)
            except Exception:
                continue

            mask = compute_pseudo_lesion_mask(img_np)
            m = compute_gradcam_metrics(
                heatmap=heatmap, original_image=img_np, lesion_mask=mask,
            )

            rows.append({
                'image_path': img_path,
                'true_class': class_names[int(row['label'])],
                'pred_class': class_names[int(pred_class)],
                'correct': class_names[int(pred_class)] == class_names[int(row['label'])],
                'confidence': float(conf),
                'focus_ratio': m.focus_ratio,
                'mean_act_inside': m.mean_activation_in,
                'mean_act_outside': m.mean_activation_out,
                'entropy': m.entropy,
                'peak_distance': m.peak_distance,
                'coverage_75': m.coverage_75,
            })
    finally:
        cam.release()
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df['model'] = name
    csv_path = output_dir / f'gradcam_quant_{_slug(name)}.csv'
    df.to_csv(csv_path, index=False)
    print(f'  [{name}] saved {len(df)} rows → {csv_path.name}')
    return df


def _slug(name: str) -> str:
    return name.lower().replace(' ', '_').replace('-', '_')


# --------------------------------------------------------------------------- #
#  Cross-model plots
# --------------------------------------------------------------------------- #

def plot_focus_ratio_comparison(agg: pd.DataFrame, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    models = agg['model'].unique().tolist()
    classes = agg['class'].unique().tolist()

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(classes))
    width = 0.8 / max(len(models), 1)
    colors = ['#45B7D1', '#FF6B6B', '#4ECDC4', '#FFA94D', '#9775FA']

    for i, m in enumerate(models):
        sub = agg[agg['model'] == m].set_index('class').reindex(classes)
        means = sub['focus_ratio_mean'].to_numpy()
        stds = sub['focus_ratio_std'].to_numpy()
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               yerr=stds, capsize=3, label=m,
               color=colors[i % len(colors)], alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha='right')
    ax.set_ylabel('Focus Ratio (inside lesion mask)', fontsize=12)
    ax.set_title('Grad-CAM Focus Ratio per Class & Model\n(higher = more localised on lesion)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_entropy_vs_peakdist(agg: pd.DataFrame, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    models = agg['model'].unique().tolist()
    markers = ['o', 's', '^', 'D', 'P']
    colors = ['#45B7D1', '#FF6B6B', '#4ECDC4', '#FFA94D', '#9775FA']

    for i, m in enumerate(models):
        sub = agg[agg['model'] == m]
        ax.scatter(sub['entropy_mean'], sub['peak_distance_mean'],
                   s=80, alpha=0.85, label=m,
                   marker=markers[i % len(markers)],
                   color=colors[i % len(colors)])
        for _, row in sub.iterrows():
            ax.annotate(row['class'], (row['entropy_mean'], row['peak_distance_mean']),
                        xytext=(4, 4), textcoords='offset points', fontsize=8, alpha=0.7)

    ax.set_xlabel('Entropy (lower = sharper focus)', fontsize=12)
    ax.set_ylabel('Peak distance to mask centroid', fontsize=12)
    ax.set_title('Heatmap Sharpness vs Spatial Alignment',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of model names to run (default: all available).')
    parser.add_argument('--n-per-class', type=int, default=80,
                        help='Number of test samples per class for Grad-CAM analysis.')
    parser.add_argument('--output-dir', type=str, default='outputs/gradcam_quant')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))

    # Load test split from first available config
    cfg0 = None
    for m in MODELS:
        if Path(m['config']).exists():
            cfg0 = load_config(m['config'])
            break
    if cfg0 is None:
        raise RuntimeError('No model configs found.')
    _, _, test_df = load_split_dataframes(cfg0['paths']['csv_file'])
    sampled = sample_per_class(test_df, args.n_per_class, seed=args.seed)

    print('=' * 70)
    print('  Quantitative Grad-CAM Analysis')
    print(f'  Samples: {len(sampled)}  ({args.n_per_class}/class max)')
    print(f'  Output : {output_dir}')
    print('=' * 70)

    selected = [m for m in MODELS if (args.models is None or m['name'] in args.models)
                and Path(m['config']).exists() and Path(m['checkpoint']).exists()]
    if not selected:
        raise RuntimeError('No runnable models selected.')

    all_dfs = []
    for m in selected:
        df = run_model(
            name=m['name'], cfg_path=m['config'], ckpt_path=m['checkpoint'],
            test_df=sampled, output_dir=output_dir,
        )
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_dir / 'gradcam_quant_all.csv', index=False)

    # Aggregate per (model, class) with pandas for downstream plotting
    metric_cols = ['focus_ratio', 'mean_act_inside', 'mean_act_outside',
                   'entropy', 'peak_distance', 'coverage_75']
    grouped = combined.groupby(['model', 'true_class'])[metric_cols]
    agg = grouped.agg(['mean', 'std']).reset_index()
    # Flatten columns: focus_ratio_mean, focus_ratio_std, ...
    agg.columns = ['model', 'class'] + [f'{m}_{stat}' for m, stat in agg.columns[2:]]
    agg.to_csv(output_dir / 'gradcam_quant_aggregated.csv', index=False)

    plot_focus_ratio_comparison(agg, output_dir / 'focus_ratio_comparison.png')
    plot_entropy_vs_peakdist(agg, output_dir / 'entropy_vs_peakdist.png')

    # Per-model summary (averaged over classes)
    summary = (combined.groupby('model')[[
        'focus_ratio', 'mean_act_inside', 'entropy',
        'peak_distance', 'coverage_75', 'confidence',
    ]].mean().round(4).reset_index())
    summary.to_csv(output_dir / 'gradcam_quant_summary.csv', index=False)
    save_json(output_dir / 'gradcam_quant_summary.json',
              {'per_model': summary.to_dict(orient='records')})

    print('\n  -- Summary (mean over all samples) --')
    print(summary.to_string(index=False))
    print('=' * 70)


if __name__ == '__main__':
    main()
