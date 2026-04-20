from __future__ import annotations

from typing import Dict
from torchvision import transforms


def build_transforms(image_size: int, aug_cfg: Dict, is_train: bool):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train:
        # Use RandomResizedCrop for scale/translation invariance, or plain Resize
        if aug_cfg.get('random_resized_crop', False):
            scale_min = float(aug_cfg.get('random_resized_crop_scale_min', 0.8))
            scale_max = float(aug_cfg.get('random_resized_crop_scale_max', 1.0))
            train_tfms = [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(scale_min, scale_max),
                    ratio=(0.9, 1.1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            ]
        else:
            train_tfms = [
                transforms.Resize((image_size, image_size)),
            ]

        train_tfms.append(transforms.RandomHorizontalFlip(p=aug_cfg.get('hflip', 0.5)))

        if aug_cfg.get('vflip', 0.0) > 0:
            train_tfms.append(transforms.RandomVerticalFlip(p=aug_cfg['vflip']))
        if aug_cfg.get('rotation', 0) > 0:
            train_tfms.append(transforms.RandomRotation(degrees=aug_cfg['rotation']))
        cj = aug_cfg.get('color_jitter', 0.0)
        if cj > 0:
            train_tfms.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(0.1, cj)))

        # GaussianBlur for robustness
        gb_prob = aug_cfg.get('gaussian_blur', 0.0)
        if gb_prob > 0:
            train_tfms.append(transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            ], p=gb_prob))

        train_tfms.extend([
            transforms.ToTensor(),
            normalize,
        ])
        re_prob = aug_cfg.get('random_erasing', 0.0)
        if re_prob > 0:
            train_tfms.append(transforms.RandomErasing(p=re_prob, scale=(0.02, 0.12), ratio=(0.3, 3.3)))
        return transforms.Compose(train_tfms)

    eval_tfms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(eval_tfms)
