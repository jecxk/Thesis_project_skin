"""Generate a cleaner, simplified pipeline overview figure for the
Methodology slide. Replaces the older, denser training_pipeline.png.

Kept deliberately high-level: preprocessing/augmentation detail and
training-strategy detail now live on their own slides (and appendix A28),
so this figure only needs to convey the end-to-end shape of the system.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath

OUT_PATH = Path('thesis/figures/pipeline_overview.png')

NAVY = '#1F3864'
BOX_FILL = '#D9E2F3'
BOX_EDGE = '#1F3864'
ACCENT_FILL = '#FFF2CC'
ACCENT_EDGE = '#BF8F00'


def box(ax, xy, w, h, title, subtitle=None, fill=BOX_FILL, edge=BOX_EDGE, fontsize=13):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.08',
        linewidth=1.8, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha='center', va='center',
                 fontsize=fontsize, fontweight='bold', color=NAVY)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha='center', va='center',
                 fontsize=fontsize - 2.5, color='#333333')
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                 fontsize=fontsize, fontweight='bold', color=NAVY)


def arrow(ax, p1, p2, color=NAVY, style='-|>', lw=2.0, connectionstyle='arc3,rad=0.0', ls='solid'):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                         linewidth=lw, color=color, connectionstyle=connectionstyle, linestyle=ls)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    y0, h = 2.6, 1.7
    w = 2.35
    gap = 0.55
    xs = [0.3 + i * (w + gap) for i in range(4)]

    box(ax, (xs[0], y0), w, h, 'Input Image', '224 × 224, RGB')
    box(ax, (xs[1], y0), w, h, 'Preprocessing +\nAugmentation', 'resize · normalize\nflip · rotate · jitter', fontsize=12.5)
    box(ax, (xs[2], y0), w, h, 'Pretrained\nBackbone', 'EfficientNet-B0 / ResNet50 /\nDenseNet121 / Swin-Tiny', fontsize=12.5)
    box(ax, (xs[3], y0), w, h, 'Classifier\nHead', '7-class\nprobabilities', fontsize=12.5)

    for i in range(3):
        arrow(ax, (xs[i] + w, y0 + h / 2), (xs[i + 1], y0 + h / 2))

    # Output arrow to the right
    out_x = xs[3] + w
    arrow(ax, (out_x, y0 + h / 2), (out_x + 0.55, y0 + h / 2))
    ax.text(out_x + 1.05, y0 + h / 2, 'akiec\nbcc\nbkl\n...', ha='center', va='center',
             fontsize=10, color='#333333', linespacing=1.3)

    # Training-only loop (dashed, accent colour) underneath backbone+head
    loop_y = 0.55
    loop_h = 0.95
    loop_x = xs[2]
    loop_w = (xs[3] + w) - xs[2]
    box(ax, (loop_x, loop_y), loop_w, loop_h,
        'Loss (Cross-Entropy)  →  Backpropagation', fill=ACCENT_FILL, edge=ACCENT_EDGE, fontsize=12)

    # dashed connectors: classifier head down to loss box, loss box up to backbone (feedback)
    arrow(ax, (xs[3] + w * 0.75, y0), (xs[3] + w * 0.75, loop_y + loop_h),
          color=ACCENT_EDGE, ls='dashed', lw=1.8)
    arrow(ax, (loop_x + 0.15, loop_y + loop_h), (xs[2] + w * 0.25, y0),
          color=ACCENT_EDGE, ls='dashed', lw=1.8, connectionstyle='arc3,rad=-0.25')

    ax.text((loop_x + loop_x + loop_w) / 2, loop_y - 0.32,
             'training only — updates backbone + classifier weights each step',
             ha='center', va='center', fontsize=10, color='#666666', style='italic')

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
