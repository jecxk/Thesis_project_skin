from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True).copy()
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

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(image_path)


def load_split_dataframes(csv_file: str | Path):
    df = pd.read_csv(csv_file)
    required_columns = {'image_path', 'label', 'split'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in CSV: {missing}')

    train_df = df[df['split'].str.lower() == 'train'].reset_index(drop=True)
    val_df = df[df['split'].str.lower() == 'val'].reset_index(drop=True)
    test_df = df[df['split'].str.lower() == 'test'].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError('CSV must contain non-empty train, val, and test splits.')

    return train_df, val_df, test_df


def compute_class_weights(labels: List[int], num_classes: int):
    import numpy as np

    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return weights
