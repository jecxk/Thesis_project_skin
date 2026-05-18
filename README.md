# Skin Lesion Classification on Dermoscopic Images

A Comparative Evaluation of Deep Learning Models for 7-Class Skin Lesion Classification
on the ISIC 2018 / HAM10000 Dataset, with Ensemble Learning and Grad-CAM Interpretability.

**Author:** Nguyen Trong Bach (23BI14057)
**Affiliation:** University of Science and Technology of Hanoi (USTH), Department of Information and Communication Technology
**Supervisors:** Dr. Vu Trong Sinh (external), Dr. Nghiem Thi Phuong (internal)

---

## Overview

This repository contains the source code, experiments, thesis manuscript, presentation
slides and demonstration application for a bachelor-level research project on skin
lesion classification.

The work compares four deep learning backbones — EfficientNet-B0, ResNet50, DenseNet121
and Swin-Tiny — fine-tuned from ImageNet on ISIC 2018 (HAM10000), with an emphasis on
mitigating the severe 58:1 class imbalance through Weighted Random Sampling, Mixup,
CutMix and Label Smoothing. The four models are combined via soft-voting ensemble.
Grad-CAM (Selvaraju et al., 2017) is used for visual explanation, complemented by a
set of quantitative interpretability metrics. A Streamlit prototype provides a clinical
"second opinion" interface.

## Headline Results

| Metric          | EfficientNet-B0 | ResNet50 | DenseNet121 | Swin-Tiny | Ensemble    |
|-----------------|:---------------:|:--------:|:-----------:|:---------:|:-----------:|
| Parameters      | 5.3 M           | 25.6 M   | 8.0 M       | 28.0 M    | —           |
| Accuracy        | 88.36 %         | 86.03 %  | 85.43 %     | 84.90 %   | **89.49 %** |
| Macro F1        | 0.831           | 0.790    | 0.797       | 0.786     | **0.845**   |
| Macro AUC       | 0.952           | 0.957    | 0.959       | 0.960     | **0.980**   |
| Cohen's Kappa   | 0.771           | 0.731    | 0.721       | 0.715     | —           |

The ensemble achieves SOTA-level performance among image-only methods on ISIC 2018 Task 3.

## Repository Structure

```
skin_thesis_project/
├── src/                              # Source code
│   ├── configs/                      # YAML experiment configurations
│   │   ├── efficientnet_b0_optimized.yaml
│   │   ├── resnet50_v3.yaml
│   │   ├── densenet121_v3.yaml
│   │   ├── swin_tiny_v3.yaml
│   │   └── ablations/                # Configurations for ablation runs
│   ├── data/                         # Dataset, transforms, samplers, Mixup/CutMix
│   ├── engine/                       # Training and evaluation loops
│   ├── models/                       # Backbone factory and loss functions
│   ├── utils/                        # Metrics, plots, Grad-CAM, calibration
│   ├── train.py                      # Training entry point
│   ├── evaluate.py                   # Single-model evaluation
│   ├── ensemble.py                   # Soft-voting ensemble
│   ├── tta.py                        # Test-Time Augmentation
│   ├── grad_cam_analysis.py          # Qualitative Grad-CAM visualisations
│   ├── gradcam_quant_runner.py       # Quantitative Grad-CAM metrics
│   ├── calibration_analysis.py       # ECE, Brier score, temperature scaling
│   ├── error_analysis.py             # Confusion clusters and hard cases
│   ├── threshold_tuning.py           # Per-class decision-threshold optimisation
│   ├── summarize_ablation.py         # Ablation results aggregation
│   └── gradcam_worker.py             # Subprocess worker for safe Grad-CAM
├── outputs/                          # Auto-generated experiment artefacts
│   ├── efficientnet_b0_v3(main_best)/
│   ├── resnet50_v3(main)/
│   ├── densenet121_v3(main)/
│   ├── swin_tiny_v3(main)/
│   ├── ensemble/
│   ├── ablation_no_mixup/
│   ├── ablation_no_smooth/
│   ├── ablation_no_sampler/
│   ├── ablation_no_adv_aug/
│   └── grad_cam_analysis/
├── thesis/                           # LaTeX manuscript
│   ├── main.tex                      # English thesis
│   ├── report_vn.tex                 # Vietnamese thesis
│   ├── chapter3.tex, chapter4.tex, chapter5.tex
│   ├── references.bib                # Bibliography
│   └── figures/                      # All figures used in the manuscript
├── presentation/                     # Defense slides
│   ├── slides.tex / slides.pdf       # Vietnamese version
│   ├── slides_en.tex / slides_en.pdf # English version
│   └── speaker_notes.md
├── data/metadata/                    # CSV with image paths, labels and splits
├── demo_images/                      # Sample dermoscopy images for the demo
├── app.py                            # Streamlit demonstration application
├── app_v2.py                         # Alternative Gradio implementation
├── study_guide_vn.tex/.pdf           # Comprehensive Vietnamese study guide
├── PROJECT_OVERVIEW.md               # Detailed code walkthrough
└── requirements.txt                  # Python dependencies
```

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
.venv\Scripts\activate               # Windows
pip install -r requirements.txt
```

### Reproducing the Headline Numbers

```bash
# Train EfficientNet-B0 (best single model)
python -m src.train --config src/configs/efficientnet_b0_optimized.yaml

# Train the remaining backbones
python -m src.train --config src/configs/resnet50_v3.yaml
python -m src.train --config src/configs/densenet121_v3.yaml
python -m src.train --config src/configs/swin_tiny_v3.yaml

# Evaluate a single checkpoint
python -m src.evaluate \
    --config src/configs/efficientnet_b0_optimized.yaml \
    --checkpoint outputs/efficientnet_b0_v3\(main_best\)/best.pth

# Run the soft-voting ensemble
python -m src.ensemble --output-dir outputs/ensemble

# Aggregate ablation results
python -m src.summarize_ablation
```

### Launching the Demo

```bash
streamlit run app.py
# → http://localhost:8501
```

The Streamlit interface accepts a dermoscopy image, runs the four-model ensemble, and
displays:

- The predicted class with a severity-coloured accent.
- A bar chart of all seven class probabilities.
- A Grad-CAM heatmap from a selectable backbone, in a separate section below the
  prediction panel.
- A collapsible reference of the seven lesion classes.

The dark or light theme can be toggled from the Streamlit settings menu.

## Training Pipeline

The optimised training recipe combines:

1. **Stratified 70 / 15 / 15 split** preserving the 7-class distribution across train,
   validation and test.
2. **Weighted Random Sampler** with weights `w_c = N / (C · n_c)` to balance each
   mini-batch.
3. **Data augmentation** — RandomResizedCrop (scale 0.85 – 1.0), horizontal and
   vertical flip, ±90° rotation, ColorJitter, GaussianBlur, RandomErasing.
4. **Mixup (α = 0.2) and CutMix (α = 1.0)** applied with probability 0.5 each.
5. **Label Smoothing (ε = 0.1)** integrated into the cross-entropy loss.
6. **AdamW** optimiser with cosine annealing and a 3 – 5 epoch linear warmup.
7. **Mixed-precision FP16** training with gradient clipping at norm 1.0.
8. **Early stopping** on validation Macro F1 with patience 12.

## Ablation Study

Each component is removed individually from the EfficientNet-B0 baseline to measure its
contribution to Macro F1:

| Configuration              | Accuracy | Macro F1 | AUC    | Δ Macro F1 |
|----------------------------|:--------:|:--------:|:------:|:----------:|
| Full pipeline (baseline)   | 0.8836   | 0.8311   | 0.9522 | —          |
| − Mixup and CutMix         | 0.8589   | 0.7860   | 0.9486 | −0.0451    |
| − Label Smoothing          | 0.8776   | 0.8000   | 0.9674 | −0.0311    |
| − Weighted Random Sampler  | 0.8762   | 0.8041   | 0.9521 | −0.0270    |
| − Advanced Augmentation    | 0.8896   | 0.8204   | 0.9664 | −0.0107    |

The single most important component is Mixup / CutMix. Removing Label Smoothing or
heavy augmentation paradoxically improves either AUC or Accuracy while degrading
Macro F1 — a textbook trade-off on imbalanced data.

## Interpretability

Grad-CAM heatmaps are computed at the final convolutional block for CNNs and at the
final attention normalisation layer for Swin-Tiny. The implementation is in
`src/utils/grad_cam.py`. Beyond qualitative inspection, four quantitative metrics are
reported in `src/utils/gradcam_quant.py`:

- `focus_ratio` — fraction of activation inside the lesion mask.
- `mean_activation_in / out` — average heatmap intensity inside vs. outside the lesion.
- `entropy` — Shannon entropy of the heatmap.
- `peak_distance` — normalised distance between the activation peak and the lesion centroid.

All four models achieve `focus_ratio > 0.6` and `peak_distance < 0.2` on the test set,
indicating that the networks attend to the lesion rather than to background artefacts.

## Additional Analyses

| Module                          | Purpose                                                        |
|---------------------------------|----------------------------------------------------------------|
| `src/calibration_analysis.py`   | Expected Calibration Error, Brier score, temperature scaling.  |
| `src/threshold_tuning.py`       | Per-class decision-threshold optimisation via Youden's J.      |
| `src/tta.py`                    | Test-Time Augmentation with six deterministic transforms.      |
| `src/error_analysis.py`         | Confusion clusters and hard-case reporting.                    |
| `src/summarize_ablation.py`     | Aggregates every ablation run into a single comparison table.  |

## Documentation

| Artefact                                  | Description                                                            |
|-------------------------------------------|------------------------------------------------------------------------|
| `thesis/main.pdf`                         | Full English thesis manuscript.                                        |
| `thesis/report_vn.pdf`                    | Vietnamese version of the thesis.                                      |
| `presentation/slides.pdf`                 | Defense slides (Vietnamese) — 22 main + 22 appendix.                   |
| `presentation/slides_en.pdf`              | Defense slides (English) — same structure as the Vietnamese version.   |
| `study_guide_vn.pdf`                      | Comprehensive Vietnamese study guide covering every concept and code.  |
| `PROJECT_OVERVIEW.md`                     | Code-level walkthrough of the entire repository.                       |

To rebuild the thesis or slides:

```bash
cd thesis
xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex

cd ../presentation
xelatex slides.tex && xelatex slides.tex
xelatex slides_en.tex && xelatex slides_en.tex
```

## Dataset

ISIC 2018 Task 3 (HAM10000) is publicly available from the
[International Skin Imaging Collaboration](https://challenge.isic-archive.com/). The
dataset is not redistributed in this repository; place the raw images at the path
referenced by `data/metadata/skin_metadata.csv`.

Class distribution (7,010 train / 1,502 validation / 1,503 test):

| Class | Description           | Train | Val   | Test  | Total  | Share |
|:-----:|-----------------------|------:|------:|------:|-------:|------:|
| NV    | Melanocytic Nevus     | 4,693 | 1,006 | 1,006 | 6,705  | 66.9 %|
| MEL   | Melanoma              |   779 |   167 |   167 | 1,113  | 11.1 %|
| BKL   | Benign Keratosis-like |   769 |   165 |   165 | 1,099  | 11.0 %|
| BCC   | Basal Cell Carcinoma  |   360 |    77 |    77 |   514  |  5.1 %|
| AKIEC | Actinic Keratosis     |   229 |    49 |    49 |   327  |  3.3 %|
| VASC  | Vascular Lesion       |    99 |    21 |    22 |   142  |  1.4 %|
| DF    | Dermatofibroma        |    81 |    17 |    17 |   115  |  1.1 %|

## Hardware and Runtime

Training was performed on a single NVIDIA GPU with mixed precision. EfficientNet-B0
trains in approximately 45 minutes for 50 epochs on this configuration. The
demonstration application runs on CPU by default and uses subprocess isolation for
Grad-CAM generation to prevent GPU memory leaks during prolonged sessions.

## Citation

If this work is referenced, please cite as:

> Nguyen Trong Bach, *A Comparative Evaluation of Deep Learning Models for Skin Lesion
> Classification on Public Dermoscopic Datasets*, Bachelor's Thesis, University of
> Science and Technology of Hanoi, 2026.

## License and Disclaimer

This codebase is released for academic and research purposes only. The accompanying
demonstration application is a research prototype and is **not** intended for clinical
diagnosis. All predictions must be reviewed by a qualified dermatologist before any
medical decision is made.
