"""Generate Grad-CAM visualizations for all models.

Creates a grid of Grad-CAM heatmaps for sample images from each class,
comparing how different models "look" at the same skin lesion.

Usage:
    python -m src.grad_cam_analysis
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image

from src.data.dataset import load_split_dataframes
from src.data.transforms import build_transforms
from src.models.build_model import build_model
from src.utils.config import load_config
from src.utils.grad_cam import GradCAM, get_target_layer
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


# ── Model configurations to analyze ──────────────────────────────────────────

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


def load_original_image(image_path: str, size: int = 224) -> np.ndarray:
    """Load and resize image for display (no normalization)."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def preprocess_image(image_path: str, transform) -> torch.Tensor:
    """Load and preprocess image for model input."""
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img)
    return tensor.unsqueeze(0)  # (1, C, H, W)


def get_sample_images(csv_path: str, class_names: List[str], n_per_class: int = 3,
                      split: str = 'test') -> Dict[str, List[str]]:
    """Get sample image paths for each class from the test set."""
    train_df, val_df, test_df = load_split_dataframes(csv_path)
    df = test_df if split == 'test' else val_df

    samples = {}
    for class_id, class_name in enumerate(class_names):
        class_df = df[df['label'] == class_id]
        paths = class_df['image_path'].tolist()
        n = min(n_per_class, len(paths))
        samples[class_name] = random.sample(paths, n)

    return samples


def generate_single_image_comparison(
    image_path: str,
    models_info: List[Dict],
    class_names: List[str],
    true_label: str,
    output_path: Path,
    device: str = 'cuda',
) -> None:
    """Generate Grad-CAM comparison for a single image across all models."""
    n_models = len(models_info)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))

    # Original image
    orig_img = load_original_image(image_path)
    axes[0].imshow(orig_img)
    axes[0].set_title(f'Original\n(True: {true_label})', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    for idx, minfo in enumerate(models_info):
        cfg = load_config(minfo['config'])
        model = build_model(cfg['model']).to(device)
        checkpoint = torch.load(minfo['checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        target_layer = get_target_layer(model, cfg['model']['name'])
        grad_cam = GradCAM(model, target_layer)

        eval_tfms = build_transforms(cfg['train']['image_size'], cfg['augmentation']['eval'], is_train=False)
        input_tensor = preprocess_image(image_path, eval_tfms).to(device)

        heatmap, pred_class, confidence = grad_cam(input_tensor)
        overlay = GradCAM.overlay(orig_img, heatmap, alpha=0.45)

        pred_name = class_names[pred_class]
        correct = '✓' if pred_name == true_label else '✗'
        color = '#2ecc71' if pred_name == true_label else '#e74c3c'

        axes[idx + 1].imshow(overlay)
        axes[idx + 1].set_title(
            f"{minfo['name']}\nPred: {pred_name} ({confidence:.1%}) {correct}",
            fontsize=10, color=color, fontweight='bold',
        )
        axes[idx + 1].axis('off')

        grad_cam.release()
        del model
        torch.cuda.empty_cache()

    plt.suptitle(f'Grad-CAM Comparison — True Label: {true_label}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_class_grid(
    samples: Dict[str, List[str]],
    models_info: List[Dict],
    class_names: List[str],
    output_dir: Path,
    device: str = 'cuda',
) -> None:
    """Generate a large grid: rows=classes, columns=models, showing Grad-CAM for 1 sample per class."""
    n_classes = len(class_names)
    n_models = len(models_info)

    fig = plt.figure(figsize=(4 * (n_models + 1), 3.5 * n_classes))
    gs = gridspec.GridSpec(n_classes, n_models + 1, hspace=0.35, wspace=0.1)

    for row, class_name in enumerate(class_names):
        if not samples.get(class_name):
            continue

        image_path = samples[class_name][0]
        orig_img = load_original_image(image_path)

        # Original image
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(orig_img)
        ax.set_title(f'{class_name.upper()}', fontsize=11, fontweight='bold')
        ax.axis('off')
        if row == 0:
            ax.set_title(f'Original\n{class_name.upper()}', fontsize=11, fontweight='bold')

        # Each model
        for col, minfo in enumerate(models_info):
            cfg = load_config(minfo['config'])
            model = build_model(cfg['model']).to(device)
            checkpoint = torch.load(minfo['checkpoint'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            target_layer = get_target_layer(model, cfg['model']['name'])
            grad_cam = GradCAM(model, target_layer)

            eval_tfms = build_transforms(cfg['train']['image_size'], cfg['augmentation']['eval'], is_train=False)
            input_tensor = preprocess_image(image_path, eval_tfms).to(device)

            heatmap, pred_class, confidence = grad_cam(input_tensor)
            overlay = GradCAM.overlay(orig_img, heatmap, alpha=0.45)

            pred_name = class_names[pred_class]
            correct = pred_name == class_name
            color = '#2ecc71' if correct else '#e74c3c'

            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(overlay)
            title = f'{pred_name} ({confidence:.0%})'
            if row == 0:
                title = f"{minfo['name']}\n{title}"
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')

            grad_cam.release()
            del model
            torch.cuda.empty_cache()

    plt.suptitle('Grad-CAM Analysis — All Classes × All Models',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(output_dir / 'gradcam_class_grid.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: gradcam_class_grid.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=3, help='Number of samples per class')
    parser.add_argument('--output-dir', type=str, default='outputs/grad_cam_analysis')
    args = parser.parse_args()

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = ensure_dir(Path(args.output_dir))

    # Load class names and sample images
    cfg = load_config(MODELS[0]['config'])
    class_names = cfg['classes']
    csv_path = cfg['paths']['csv_file']

    print('=' * 70)
    print('  Grad-CAM Analysis')
    print(f'  Device: {device}')
    print(f'  Models: {", ".join(m["name"] for m in MODELS)}')
    print(f'  Output: {output_dir}')
    print('=' * 70)

    # Check all checkpoints exist
    available_models = []
    for m in MODELS:
        if Path(m['checkpoint']).exists():
            available_models.append(m)
            print(f'  [OK] {m["name"]}: {m["checkpoint"]}')
        else:
            print(f'  [FAIL] {m["name"]}: {m["checkpoint"]} (NOT FOUND, skipping)')

    if not available_models:
        print('No model checkpoints found! Train models first.')
        return

    print(f'\n  Sampling {args.n_samples} images per class from test set...')
    samples = get_sample_images(csv_path, class_names, n_per_class=args.n_samples)
    for cls, paths in samples.items():
        print(f'    {cls}: {len(paths)} images')

    # ── 1. Class Grid (1 sample per class × all models) ──────────────────
    print('\n  [1/2] Generating class comparison grid...')
    generate_class_grid(samples, available_models, class_names, output_dir, device)

    # ── 2. Per-image comparisons (all samples) ───────────────────────────
    print('\n  [2/2] Generating per-image Grad-CAM comparisons...')
    per_image_dir = ensure_dir(output_dir / 'per_image')
    count = 0
    for class_name, paths in samples.items():
        for i, path in enumerate(paths):
            fname = f'gradcam_{class_name}_{i + 1}.png'
            generate_single_image_comparison(
                image_path=path,
                models_info=available_models,
                class_names=class_names,
                true_label=class_name,
                output_path=per_image_dir / fname,
                device=device,
            )
            count += 1
            print(f'    [{count}] {fname}')

    print(f'\n  [OK] Done! {count} Grad-CAM images saved to {output_dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
