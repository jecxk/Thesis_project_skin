from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_ORDER = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_ORDER)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--labels-csv', type=str, required=True, help='CSV containing image and diagnosis columns')
    parser.add_argument('--image-col', type=str, default='image_id')
    parser.add_argument('--label-col', type=str, default='dx')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--output-csv', type=str, default='data/metadata/skin_metadata.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    df = pd.read_csv(args.labels_csv)
    df = df[df[args.label_col].isin(CLASS_TO_ID.keys())].copy()
    df['image_path'] = df[args.image_col].astype(str).apply(lambda x: str(image_dir / f'{x}{args.ext}'))
    df['class_name'] = df[args.label_col].astype(str)
    df['label'] = df['class_name'].map(CLASS_TO_ID)

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    train_df = train_df.copy(); train_df['split'] = 'train'
    val_df = val_df.copy(); val_df['split'] = 'val'
    test_df = test_df.copy(); test_df['split'] = 'test'

    out_df = pd.concat([
        train_df[['image_path', 'label', 'class_name', 'split']],
        val_df[['image_path', 'label', 'class_name', 'split']],
        test_df[['image_path', 'label', 'class_name', 'split']],
    ], ignore_index=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f'Saved metadata to {output_csv}')


if __name__ == '__main__':
    main()
