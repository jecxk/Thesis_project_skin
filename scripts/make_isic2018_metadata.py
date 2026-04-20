from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_ORDER = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
ISIC_TO_OURS = {
    'AKIEC': 'akiec',
    'BCC': 'bcc',
    'BKL': 'bkl',
    'DF': 'df',
    'MEL': 'mel',
    'NV': 'nv',
    'VASC': 'vasc',
}
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_ORDER)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--gt-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, default='data/metadata/skin_metadata.csv')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    gt_csv = Path(args.gt_csv)

    df = pd.read_csv(gt_csv)

    if 'image' not in df.columns:
        raise ValueError("Ground truth CSV must contain column 'image'.")

    class_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    missing = [c for c in class_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing class columns in ISIC ground truth CSV: {missing}")

    def one_hot_to_class(row):
        active = [c for c in class_cols if int(row[c]) == 1]
        if len(active) != 1:
            return None
        return ISIC_TO_OURS[active[0]]

    df['class_name'] = df.apply(one_hot_to_class, axis=1)
    df = df[df['class_name'].notna()].copy()

    df['label'] = df['class_name'].map(CLASS_TO_ID)
    df['image_path'] = df['image'].astype(str).apply(lambda x: str(image_dir / f'{x}.jpg'))

    # stratified split: 70 / 15 / 15
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df['label'],
        random_state=args.seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=args.seed
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    out_df = pd.concat([
        train_df[['image_path', 'label', 'class_name', 'split']],
        val_df[['image_path', 'label', 'class_name', 'split']],
        test_df[['image_path', 'label', 'class_name', 'split']],
    ], ignore_index=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f'Saved metadata to: {output_csv}')
    print(out_df['split'].value_counts())
    print(out_df['class_name'].value_counts())


if __name__ == '__main__':
    main()