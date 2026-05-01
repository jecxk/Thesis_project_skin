# 🔬 Skin Lesion Classification — Bachelor's Thesis Project

> **A Comparative Evaluation of Deep Models for Skin Lesion Classification on Public Dermoscopy Datasets**  
> Author: Nguyen Trong Bach (23BI14057) — USTH, Department of ICT

---

## 🎯 Project Overview

This project implements and evaluates a comprehensive deep learning pipeline for **7-class skin lesion classification** using the ISIC 2018 (HAM10000) dataset. The pipeline addresses severe class imbalance (58:1 ratio) through an optimized training recipe and compares four architectures: three CNNs and one Vision Transformer.

### Key Results

| Metric | Best Single Model (EB0) | Ensemble (4 Models) |
|--------|:-----------------------:|:-------------------:|
| **Accuracy** | 88.36% | **89.49%** |
| **Macro F1** | 0.831 | **0.845** |
| **AUC** | 0.952 | **0.980** |
| **MCC** | 0.772 | **0.796** |
| **Cohen's Kappa** | 0.771 | **0.795** |

## 🏗️ Architecture

```
skin_thesis_project/
├── src/                          # Core source code
│   ├── configs/                  # YAML experiment configs
│   │   ├── efficientnet_b0_optimized.yaml
│   │   ├── resnet50_v3.yaml
│   │   ├── densenet121_v3.yaml
│   │   ├── swin_tiny_v3.yaml
│   │   └── ablations/           # Ablation study configs
│   ├── data/                    # Dataset + augmentation utilities
│   ├── engine/                  # Training + evaluation loops
│   ├── models/                  # Model building + architecture wrappers
│   ├── utils/                   # Config, metrics, grad-cam helpers
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   ├── ensemble.py              # Soft-voting ensemble
│   ├── tta.py                   # Test-Time Augmentation
│   ├── grad_cam_analysis.py     # Qualitative Grad-CAM
│   ├── gradcam_quant_runner.py  # Quantitative Grad-CAM metrics
│   ├── calibration_analysis.py  # ECE / Brier / Temperature scaling
│   ├── error_analysis.py        # Confusion analysis & hard cases
│   ├── threshold_tuning.py      # Per-class threshold optimization
│   └── summarize_ablation.py    # Ablation results table
├── outputs/                     # All experiment results
│   ├── efficientnet_b0_v3(main_best)/
│   ├── resnet50_v3(main)/
│   ├── densenet121_v3(main)/
│   ├── swin_tiny_v3(main)/
│   ├── ensemble/
│   ├── ablation_no_*/           # 4 ablation experiments
│   └── grad_cam_analysis/
├── thesis/                      # LaTeX thesis (30 references, 38 pages)
├── app.py                       # Streamlit web demo (v1)
├── app_v2.py                    # Gradio web demo (v2)
└── data/                        # Dataset & metadata
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python -m src.train --config src/configs/efficientnet_b0_optimized.yaml

# Evaluate
python -m src.evaluate --config src/configs/efficientnet_b0_optimized.yaml \
    --checkpoint outputs/efficientnet_b0_v3(main_best)/best.pth

# Run ensemble
python -m src.ensemble --output-dir outputs/ensemble

# Run web demo
streamlit run app.py
```

## 📊 Training Pipeline

The optimized pipeline combines:
1. **Weighted Random Sampling** — class-balanced mini-batches (weight ∝ 1/class_frequency)
2. **Mixup (α=0.2) + CutMix (α=1.0)** — applied with p=0.5 each
3. **Label Smoothing (ε=0.1)** — prevents overconfident predictions
4. **AdamW optimizer** — decoupled weight decay
5. **Cosine annealing with linear warmup** — 3–5 epoch warmup
6. **Mixed precision (FP16)** — faster training, lower memory

## 📚 Models Evaluated

| Model | Params | Type | Accuracy | Macro F1 | AUC |
|-------|--------|------|----------|----------|-----|
| EfficientNet-B0 | 5.3M | CNN | **88.36%** | **0.831** | 0.952 |
| ResNet50 | 25.6M | CNN | 86.03% | 0.790 | 0.957 |
| DenseNet121 | 8.0M | CNN | 85.43% | 0.797 | 0.959 |
| Swin-Tiny | 28.0M | ViT | 84.90% | 0.786 | **0.960** |
| **Ensemble** | 66.9M | Mix | **89.49%** | **0.845** | **0.980** |

## 📖 Thesis

The thesis is written in LaTeX with 30 academic references. To compile:

```bash
cd thesis
pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex
```

**Structure:** Introduction → Literature Review → Methodology → Experiments & Results → Conclusion & Future Work

## 📄 License

This project is part of an academic thesis submission at USTH.
