"""Skin lesion dataset variant that also returns per-image metadata.

Pairs each image with its 19-dimensional metadata feature vector produced by
``scripts/parse_ham10000_metadata.py``. The feature matrix lives in a separate
``.npy`` file aligned row-for-row with the enriched CSV.

Returned tuple: (image_tensor, meta_tensor, label_int, image_path_str)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SkinLesionDatasetWithMeta(Dataset):
    """Drop-in replacement for SkinLesionDataset with an extra metadata stream.

    Args:
        df: dataframe with at least ``image_path`` and ``label``. Each row's
            position in the dataframe must match its position in ``features``.
        root_dir: prepended to relative image_paths.
        features: (N, meta_dim) numpy array of per-image metadata features in
            the same row order as ``df``.
        transform: torchvision image transform.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path,
        features: np.ndarray,
        transform=None,
    ):
        self.df = df.reset_index(drop=True).copy()
        if len(features) != len(self.df):
            raise ValueError(
                f'Row count mismatch: df has {len(self.df)} rows but '
                f'features has {len(features)}.'
            )
        self.features = np.asarray(features, dtype=np.float32)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]
        image_path = Path(row['image_path'])
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path
        image = Image.open(image_path).convert('RGB')
        label = int(row['label'])
        meta = torch.from_numpy(self.features[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, meta, label, str(image_path)


def load_split_with_meta(
    csv_file: str | Path,
    features_file: str | Path,
):
    """Load enriched CSV + features and split into (train, val, test) tuples.

    Returns:
        train_df, val_df, test_df, train_feats, val_feats, test_feats
    """
    df = pd.read_csv(csv_file)
    features = np.load(features_file)

    if len(df) != len(features):
        raise ValueError(
            f'CSV has {len(df)} rows but features have {len(features)}.'
        )

    required = {'image_path', 'label', 'split'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in CSV: {missing}')

    masks = {
        split: (df['split'].str.lower() == split).to_numpy()
        for split in ('train', 'val', 'test')
    }
    splits = {}
    for name, mask in masks.items():
        sub_df = df[mask].reset_index(drop=True)
        sub_feats = features[mask]
        if len(sub_df) == 0:
            raise ValueError(f'Empty split: {name}')
        splits[name] = (sub_df, sub_feats)

    return (
        splits['train'][0], splits['val'][0], splits['test'][0],
        splits['train'][1], splits['val'][1], splits['test'][1],
    )
