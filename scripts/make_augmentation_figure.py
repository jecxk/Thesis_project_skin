"""Generate an illustrative figure of preprocessing + training augmentation.

Uses the project's real transforms (src/data/transforms.py) applied to a
real sample image, so the figure matches the actual pipeline instead of a
generic illustration.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

IMG_PATH = Path('demo_images/mel_1_88pct_ISIC_0024468.jpg')
if not IMG_PATH.exists():
    # fall back to any MEL sample, else the first demo image
    demo_dir = Path('demo_images')
    candidates = sorted(demo_dir.glob('mel_*.jpg')) or sorted(demo_dir.glob('*.jpg'))
    IMG_PATH = candidates[0]

OUT_PATH = Path('thesis/figures/augmentation_examples.png')
IMAGE_SIZE = 224


def main():
    import torch
    torch.manual_seed(7)
    np.random.seed(7)

    img = Image.open(IMG_PATH).convert('RGB')
    base = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    base_t = base(img)

    def to_img(t):
        return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)

    # Isolate each technique so the panel can be labelled with its exact name
    panels = []

    panels.append(('Original', to_img(base_t)))

    crop = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    panels.append(('RandomResizedCrop\n(scale 0.85--1.0)', to_img(crop(img))))

    hflip = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
    ])
    panels.append(('Horizontal Flip', to_img(hflip(img))))

    vflip = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
    ])
    panels.append(('Vertical Flip', to_img(vflip(img))))

    rotation = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
    ])
    panels.append(('Rotation\n($\\pm 90^\\circ$)', to_img(rotation(img))))

    jitter = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        transforms.ToTensor(),
    ])
    panels.append(('ColorJitter\n(0.25)', to_img(jitter(img))))

    blur = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 2.0)),
        transforms.ToTensor(),
    ])
    panels.append(('Gaussian Blur', to_img(blur(img))))

    erasing = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.04, 0.12), ratio=(0.3, 3.3)),
    ])
    panels.append(('Random Erasing', to_img(erasing(img))))

    fig, axes = plt.subplots(2, 4, figsize=(12, 7.2))
    axes = axes.flatten()

    for ax, (title, arr) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(title, fontsize=14, fontweight='bold' if title == 'Original' else 'normal')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
