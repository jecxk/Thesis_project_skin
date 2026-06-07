"""Out-of-distribution evaluation: image-only ensemble on PAD-UFES-20.

PAD-UFES-20 (Pacheco et al. 2020, https://doi.org/10.17632/zr7vgbcyr2.1) is a
clinical (smartphone) dermatology dataset --- a different acquisition modality
from ISIC 2018 (dermoscopy). Running the ISIC-trained ensemble on PAD-UFES
measures domain-shift robustness: how much performance survives the move from
polarised-light dermoscopy to consumer-camera photos.

Class mapping (PAD-UFES 6 classes → ISIC 7 classes)
---------------------------------------------------
    ACK  →  akiec     (actinic keratosis)
    SCC  →  akiec     (squamous cell carcinoma — clustered with AKIEC by
                       Pacheco; the AKIEC category in ISIC 2018 covers
                       actinic keratosis and intraepithelial / in-situ SCC)
    BCC  →  bcc
    SEK  →  bkl       (seborrheic keratosis — a type of benign keratosis)
    NEV  →  nv        (melanocytic nevus)
    MEL  →  mel
    (df  and vasc have no PAD-UFES counterpart and are not evaluated.)

Usage
-----
    # Image-only ensemble (4 multiseed seed-42 checkpoints):
    python scripts/evaluate_ood_pad_ufes.py

    # Override which checkpoints to load:
    python scripts/evaluate_ood_pad_ufes.py --variant multiseed
    python scripts/evaluate_ood_pad_ufes.py --variant main      # original v3 ckpts

Outputs
-------
    outputs/ood_pad_ufes/ensemble_predictions.csv
    outputs/ood_pad_ufes/ensemble_metrics.json
    outputs/ood_pad_ufes/ood_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.transforms import build_transforms
from src.models.build_model import build_model
from src.utils.config import load_config


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

PAD_UFES_DIR = ROOT / 'data' / 'pad_ufes'
OUT_DIR = ROOT / 'outputs' / 'ood_pad_ufes'

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

PAD_TO_ISIC: Dict[str, str] = {
    'ACK': 'akiec',
    'SCC': 'akiec',
    'BCC': 'bcc',
    'SEK': 'bkl',
    'NEV': 'nv',
    'MEL': 'mel',
}

CHECKPOINT_VARIANTS = {
    'multiseed': {
        'eb0':   ('src/configs/efficientnet_b0_optimized.yaml',
                  'outputs/multiseed/eb0_seed42/best.pth'),
        'rn50':  ('src/configs/resnet50_v3.yaml',
                  'outputs/multiseed/rn50_seed42/best.pth'),
        'dn121': ('src/configs/densenet121_v3.yaml',
                  'outputs/multiseed/dn121_seed42/best.pth'),
        'swin':  ('src/configs/swin_tiny_v3.yaml',
                  'outputs/multiseed/swin_seed42/best.pth'),
    },
    'main': {
        'eb0':   ('src/configs/efficientnet_b0_optimized.yaml',
                  'outputs/efficientnet_b0_v3(main_best)/best.pth'),
        'rn50':  ('src/configs/resnet50_v3.yaml',
                  'outputs/resnet50_v3(main)/best.pth'),
        'dn121': ('src/configs/densenet121_v3.yaml',
                  'outputs/densenet121_v3(main)/best.pth'),
        'swin':  ('src/configs/swin_tiny_v3.yaml',
                  'outputs/swin_tiny_v3/best.pth'),
    },
}


# --------------------------------------------------------------------------- #
#  Dataset: serve PAD-UFES images straight out of the three zip archives.
# --------------------------------------------------------------------------- #

class PadUfesDataset(Dataset):
    """Reads PNG images directly from PAD-UFES imgs_part_*.zip files.

    Avoids extracting 3.4 GB of images to disk. Each image is loaded on demand
    from whichever zip contains it. A small index built at construction time
    maps img_id → (zip_path, internal_name).
    """

    def __init__(self, df: pd.DataFrame, zip_paths: List[Path], transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

        # Build img_id → (zip_handle, name) index
        index: Dict[str, tuple] = {}
        self._zip_paths = zip_paths
        for zp in zip_paths:
            with zipfile.ZipFile(zp, 'r') as zf:
                for n in zf.namelist():
                    base = Path(n).name
                    if base.endswith('.png'):
                        index[base] = (zp, n)
        self._index = index

        # Drop rows whose image is missing from the zips.
        present = self.df['img_id'].isin(index.keys())
        if not present.all():
            missing = (~present).sum()
            print(f'  [warn] {missing} images referenced in metadata but not '
                  'in the zip archives — dropping them.')
            self.df = self.df[present].reset_index(drop=True)

        # Cache open zip file handles.
        self._zip_handles: Dict[Path, zipfile.ZipFile] = {}

    def _get_zip(self, zp: Path) -> zipfile.ZipFile:
        h = self._zip_handles.get(zp)
        if h is None:
            h = zipfile.ZipFile(zp, 'r')
            self._zip_handles[zp] = h
        return h

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row['img_id']
        zp, name = self._index[img_id]
        with self._get_zip(zp).open(name) as f:
            image = Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = int(row['isic_label'])
        return image, label, img_id


# --------------------------------------------------------------------------- #
#  Inference + ensemble
# --------------------------------------------------------------------------- #

def _check_zips() -> List[Path]:
    paths = sorted(PAD_UFES_DIR.glob('imgs_part_*.zip'))
    if not paths:
        sys.exit('[error] No imgs_part_*.zip in data/pad_ufes/. '
                 'Run the download step first.')
    return paths


def _load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(PAD_UFES_DIR / 'metadata.csv')
    meta['isic_class'] = meta['diagnostic'].map(PAD_TO_ISIC)
    meta = meta.dropna(subset=['isic_class']).reset_index(drop=True)
    meta['isic_label'] = meta['isic_class'].map(CLASS_TO_IDX)
    return meta


def _run_single_model(cfg_path: str, ckpt_path: str,
                      dataset: PadUfesDataset, device: str) -> np.ndarray:
    cfg = load_config(ROOT / cfg_path)
    model = build_model(cfg['model'])
    ckpt = torch.load(ROOT / ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    # Rebuild dataset transforms with this model's eval config.
    eval_tfms = build_transforms(
        cfg['train']['image_size'],
        cfg.get('augmentation', {}).get('eval', {}),
        is_train=False,
    )
    dataset.transform = eval_tfms

    loader = DataLoader(dataset, batch_size=int(cfg['train']['batch_size']),
                        shuffle=False, num_workers=0,
                        pin_memory=(device == 'cuda'))

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            logits = model(images).float()
            probs = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            all_probs.append(probs)

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_probs, axis=0)


def _evaluate(probs: np.ndarray, y_true: np.ndarray) -> Dict:
    """Compute OOD metrics. Restricted to classes present in PAD-UFES (5/7)."""
    from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                                 confusion_matrix, roc_auc_score)

    y_pred = probs.argmax(axis=1)

    overall = {
        'n_samples': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro',
                                   labels=sorted(set(y_true.tolist())))),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
    }

    # Per-class breakdown (only classes that appear in y_true).
    present = sorted(set(y_true.tolist()))
    present_names = [CLASS_NAMES[i] for i in present]
    rep = classification_report(y_true, y_pred, labels=present,
                                target_names=present_names,
                                output_dict=True, zero_division=0)
    per_class = {}
    for c, name in zip(present, present_names):
        d = rep.get(name, {})
        per_class[name] = {
            'precision': float(d.get('precision', 0.0)),
            'recall':    float(d.get('recall', 0.0)),
            'f1':        float(d.get('f1-score', 0.0)),
            'support':   int(d.get('support', 0)),
        }

    # Per-class AUC (one-vs-rest) for classes that are present.
    for c, name in zip(present, present_names):
        try:
            y_binary = (y_true == c).astype(int)
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                per_class[name]['auc'] = None
            else:
                per_class[name]['auc'] = float(
                    roc_auc_score(y_binary, probs[:, c]))
        except Exception:
            per_class[name]['auc'] = None

    # Macro AUC over present classes.
    aucs = [per_class[n]['auc'] for n in present_names
            if per_class[n].get('auc') is not None]
    overall['macro_auc'] = float(np.mean(aucs)) if aucs else None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    return {
        'overall': overall,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'classes_present_in_test': present_names,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', choices=['multiseed', 'main'], default='multiseed',
                   help='Which set of checkpoints to use.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'OOD evaluation on PAD-UFES-20')
    print(f'  Device: {device}')
    print(f'  Checkpoint variant: {args.variant}')

    zip_paths = _check_zips()
    print(f'  Zip files: {[p.name for p in zip_paths]}')

    meta = _load_metadata()
    print(f'  Samples: {len(meta)} '
          f'(class distribution: {dict(meta["isic_class"].value_counts())})')

    dataset = PadUfesDataset(meta, zip_paths, transform=None)

    models = CHECKPOINT_VARIANTS[args.variant]
    print(f'\n  Running {len(models)} backbones...')

    all_probs = []
    for key, (cfg_path, ckpt_path) in models.items():
        if not (ROOT / ckpt_path).exists():
            print(f'  [skip] {key}: {ckpt_path} not found')
            continue
        print(f'    {key}...')
        probs = _run_single_model(cfg_path, ckpt_path, dataset, device)
        all_probs.append(probs)

    if len(all_probs) < 2:
        sys.exit('Need at least 2 models for the ensemble.')

    ensemble_probs = np.stack(all_probs, axis=0).mean(axis=0)
    y_true = dataset.df['isic_label'].to_numpy()
    y_pred = ensemble_probs.argmax(axis=1)

    metrics = _evaluate(ensemble_probs, y_true)

    # Save predictions CSV.
    pred_df = dataset.df[['img_id', 'patient_id', 'diagnostic',
                          'isic_class', 'isic_label', 'age',
                          'gender', 'region']].copy()
    pred_df['y_pred'] = y_pred
    pred_df['pred_name'] = [CLASS_NAMES[i] for i in y_pred]
    pred_df['correct'] = pred_df['isic_label'] == pred_df['y_pred']
    for i, name in enumerate(CLASS_NAMES):
        pred_df[f'prob_{name}'] = ensemble_probs[:, i].round(6)
    pred_df.to_csv(OUT_DIR / 'ensemble_predictions.csv', index=False)

    # Save metrics JSON.
    with (OUT_DIR / 'ensemble_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Markdown report.
    md_lines = [
        '# Out-of-distribution evaluation: PAD-UFES-20',
        '',
        '**Setup.** ISIC 2018 (dermoscopy)-trained ensemble evaluated on PAD-UFES-20',
        '(clinical smartphone photographs). Acquisition-modality domain shift.',
        '',
        f'**Variant:** `{args.variant}`',
        f'**Samples:** {metrics["overall"]["n_samples"]}',
        f'**Classes present:** `{metrics["classes_present_in_test"]}`',
        '',
        '## Headline (5/7 classes appear in PAD-UFES)',
        '',
        f'- **Accuracy:** {metrics["overall"]["accuracy"]:.4f}',
        f'- **Macro F1 (present classes):** {metrics["overall"]["macro_f1"]:.4f}',
        f'- **Weighted F1:** {metrics["overall"]["weighted_f1"]:.4f}',
        '',
        '## Per-class breakdown',
        '',
        '| Class | Precision | Recall | F1 | AUC | Support |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, d in metrics['per_class'].items():
        auc = d.get('auc')
        auc_s = f'{auc:.3f}' if auc is not None else '—'
        md_lines.append(
            f'| {name} | {d["precision"]:.3f} | {d["recall"]:.3f} | '
            f'{d["f1"]:.3f} | {auc_s} | {d["support"]} |')
    md_lines += ['',
        '## Interpretation',
        '',
        'PAD-UFES-20 is a different imaging modality from ISIC 2018: clinical',
        'smartphone photographs versus polarised-light dermoscopy. A substantial',
        'performance drop relative to the in-distribution ISIC 2018 test set',
        '(89.9 % accuracy, 0.852 macro F1 for the same image-only ensemble) is',
        'expected. The drop quantifies how much of the model\'s skill is tied',
        'to dermoscopic visual cues rather than transferable lesion features,',
        'and gives an honest upper bound on what to expect when the same model',
        'is deployed on smartphone images without further training.',
        '']
    (OUT_DIR / 'ood_report.md').write_text('\n'.join(md_lines), encoding='utf-8')

    print('\n  ── OOD ensemble results ──')
    print(f'    Accuracy:    {metrics["overall"]["accuracy"]:.4f}')
    print(f'    Macro F1:    {metrics["overall"]["macro_f1"]:.4f}')
    print(f'    Weighted F1: {metrics["overall"]["weighted_f1"]:.4f}')
    print(f'  Wrote {OUT_DIR.relative_to(ROOT)}/{{ensemble_predictions.csv, '
          f'ensemble_metrics.json, ood_report.md}}')


if __name__ == '__main__':
    main()
