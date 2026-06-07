# Out-of-distribution evaluation: PAD-UFES-20

**Setup.** ISIC 2018 (dermoscopy)-trained ensemble evaluated on PAD-UFES-20
(clinical smartphone photographs). Acquisition-modality domain shift.

**Variant:** `multiseed`
**Samples:** 2298
**Classes present:** `['akiec', 'bcc', 'bkl', 'mel', 'nv']`

## Headline (5/7 classes appear in PAD-UFES)

- **Accuracy:** 0.3246
- **Macro F1 (present classes):** 0.2833
- **Weighted F1:** 0.3395

## Per-class breakdown

| Class | Precision | Recall | F1 | AUC | Support |
|---|---:|---:|---:|---:|---:|
| akiec | 0.543 | 0.145 | 0.229 | 0.614 | 922 |
| bcc | 0.611 | 0.421 | 0.499 | 0.733 | 845 |
| bkl | 0.175 | 0.268 | 0.212 | 0.633 | 235 |
| mel | 0.057 | 0.308 | 0.097 | 0.684 | 52 |
| nv | 0.258 | 0.725 | 0.380 | 0.811 | 244 |

## Interpretation

PAD-UFES-20 is a different imaging modality from ISIC 2018: clinical
smartphone photographs versus polarised-light dermoscopy. A substantial
performance drop relative to the in-distribution ISIC 2018 test set
(89.9 % accuracy, 0.852 macro F1 for the same image-only ensemble) is
expected. The drop quantifies how much of the model's skill is tied
to dermoscopic visual cues rather than transferable lesion features,
and gives an honest upper bound on what to expect when the same model
is deployed on smartphone images without further training.
