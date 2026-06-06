"""Parse and merge HAM10000 metadata into the project's split CSV.

The ISIC 2018 release distributed with this project includes labels only
(class + split). The full HAM10000 release on Kaggle / Harvard Dataverse adds
per-image patient metadata: ``lesion_id, image_id, dx, dx_type, age, sex,
localization``. This script merges that file with the existing
``data/metadata/skin_metadata.csv`` to produce an enriched CSV that downstream
metadata-fusion training will consume.

Download
--------
HAM10000_metadata.csv can be obtained from:
    https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T

Place the file at ``data/metadata/HAM10000_metadata.csv`` (or pass --src).

Usage
-----
    python scripts/parse_ham10000_metadata.py
    python scripts/parse_ham10000_metadata.py --src path/to/HAM10000_metadata.csv

Output
------
``data/metadata/skin_metadata_full.csv`` with columns
    image_path, label, class_name, split, age, sex, localization

plus a one-hot encoded numpy companion ``skin_metadata_features.npy`` with
the per-image 19-dimensional metadata feature vector (1 normalised age +
3 sex categories + 15 localization categories), in the same row order as the
CSV. Saves a ``meta_encoder.json`` describing the encoding scheme so the
training side reproduces it exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass


ROOT = Path(__file__).resolve().parent.parent
META_DIR = ROOT / 'data' / 'metadata'

DEFAULT_SRC = META_DIR / 'HAM10000_metadata.csv'
DEFAULT_BASE = META_DIR / 'skin_metadata.csv'
DEFAULT_OUT_CSV = META_DIR / 'skin_metadata_full.csv'
DEFAULT_OUT_NPY = META_DIR / 'skin_metadata_features.npy'
DEFAULT_OUT_ENC = META_DIR / 'meta_encoder.json'

# Fixed category vocabularies. These are frozen here so train and eval encode
# metadata identically even if the underlying CSV order changes.
SEX_CATEGORIES: List[str] = ['male', 'female', 'unknown']

LOCALIZATION_CATEGORIES: List[str] = [
    'abdomen', 'acral', 'back', 'chest', 'ear', 'face',
    'foot', 'genital', 'hand', 'lower extremity', 'neck',
    'scalp', 'trunk', 'upper extremity', 'unknown',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, default=DEFAULT_SRC,
                   help=f'Path to HAM10000_metadata.csv (default {DEFAULT_SRC.relative_to(ROOT)}).')
    p.add_argument('--base', type=Path, default=DEFAULT_BASE,
                   help=f'Existing split CSV (default {DEFAULT_BASE.relative_to(ROOT)}).')
    p.add_argument('--out-csv', type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument('--out-npy', type=Path, default=DEFAULT_OUT_NPY)
    p.add_argument('--out-encoder', type=Path, default=DEFAULT_OUT_ENC)
    return p.parse_args()


def _extract_image_id(path: str) -> str:
    """Get 'ISIC_0027419' from a full path."""
    return Path(path).stem


def _normalise_sex(s) -> str:
    if pd.isna(s):
        return 'unknown'
    s = str(s).strip().lower()
    return s if s in SEX_CATEGORIES else 'unknown'


def _normalise_localization(s) -> str:
    if pd.isna(s):
        return 'unknown'
    s = str(s).strip().lower()
    return s if s in LOCALIZATION_CATEGORIES else 'unknown'


def encode_metadata(df: pd.DataFrame,
                    age_mean: float,
                    age_std: float) -> np.ndarray:
    """Return (N, 19) one-hot + standardised metadata feature matrix."""
    n = len(df)
    feats = np.zeros((n, 1 + len(SEX_CATEGORIES) + len(LOCALIZATION_CATEGORIES)),
                     dtype=np.float32)

    # Age — standardise to mean 0 / std 1, fill NaN with 0 (== mean).
    age = df['age'].to_numpy(dtype=float)
    age_filled = np.where(np.isnan(age), age_mean, age)
    feats[:, 0] = (age_filled - age_mean) / (age_std if age_std > 0 else 1.0)

    # Sex one-hot
    sex_idx = {s: i for i, s in enumerate(SEX_CATEGORIES)}
    for i, s in enumerate(df['sex']):
        feats[i, 1 + sex_idx[s]] = 1.0

    # Localization one-hot
    loc_idx = {l: i for i, l in enumerate(LOCALIZATION_CATEGORIES)}
    offset = 1 + len(SEX_CATEGORIES)
    for i, l in enumerate(df['localization']):
        feats[i, offset + loc_idx[l]] = 1.0

    return feats


def main() -> None:
    args = parse_args()

    if not args.src.exists():
        sys.exit(
            f'\n[error] HAM10000_metadata.csv not found at {args.src}.\n'
            f'Download from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000\n'
            f'and place it at {DEFAULT_SRC.relative_to(ROOT)} (or pass --src).'
        )
    if not args.base.exists():
        sys.exit(f'[error] Base split CSV not found at {args.base}.')

    print(f'Reading HAM10000 metadata:  {args.src.relative_to(ROOT)}')
    ham = pd.read_csv(args.src)
    print(f'Reading project split CSV:  {args.base.relative_to(ROOT)}')
    base = pd.read_csv(args.base)

    # Normalise categorical columns up-front.
    ham['sex'] = ham['sex'].map(_normalise_sex)
    ham['localization'] = ham['localization'].map(_normalise_localization)

    # Compute age stats from the HAM metadata (train + val + test combined is
    # acceptable because metadata is non-target and we are just centring).
    age_mean = float(np.nanmean(ham['age']))
    age_std = float(np.nanstd(ham['age']))
    print(f'\nAge stats: mean={age_mean:.2f}, std={age_std:.2f}')

    # Merge on image_id.
    base['image_id'] = base['image_path'].map(_extract_image_id)
    merged = base.merge(
        ham[['image_id', 'age', 'sex', 'localization']],
        on='image_id', how='left',
    )

    missing = merged['sex'].isna().sum()
    if missing > 0:
        print(f'[warn] {missing} images in the base CSV had no row in '
              'HAM10000_metadata.csv. Falling back to "unknown" categorical '
              'and mean age.')
    merged['sex'] = merged['sex'].map(_normalise_sex)
    merged['localization'] = merged['localization'].map(_normalise_localization)

    # Encode features.
    feats = encode_metadata(merged, age_mean=age_mean, age_std=age_std)

    # Persist.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ['image_path', 'label', 'class_name', 'split',
                'age', 'sex', 'localization']
    merged[out_cols].to_csv(args.out_csv, index=False)
    np.save(args.out_npy, feats)

    encoder = {
        'age_mean': age_mean,
        'age_std': age_std,
        'sex_categories': SEX_CATEGORIES,
        'localization_categories': LOCALIZATION_CATEGORIES,
        'feature_dim': feats.shape[1],
        'feature_layout': [
            {'name': 'age_standardised', 'dim': 1},
            {'name': 'sex_onehot', 'dim': len(SEX_CATEGORIES),
             'categories': SEX_CATEGORIES},
            {'name': 'localization_onehot', 'dim': len(LOCALIZATION_CATEGORIES),
             'categories': LOCALIZATION_CATEGORIES},
        ],
    }
    args.out_encoder.write_text(json.dumps(encoder, indent=2), encoding='utf-8')

    print(f'\nWrote {args.out_csv.relative_to(ROOT)}  ({len(merged)} rows)')
    print(f'Wrote {args.out_npy.relative_to(ROOT)}  ({feats.shape})')
    print(f'Wrote {args.out_encoder.relative_to(ROOT)}')

    # Quick sanity: per-split row counts.
    print('\nPer-split counts:')
    print(merged.groupby('split').size())


if __name__ == '__main__':
    main()
