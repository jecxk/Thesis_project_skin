"""Skin Lesion Classification — Advanced Demo (v2).

Features beyond app.py:
  * Model selector (4 individual models + soft-voting Ensemble).
  * Grad-CAM overlay for visual explanation.
  * Top-3 predictions with medical descriptions and severity tags.
  * Low-confidence warnings (decision-threshold logic).
  * Clinical disclaimer and ISIC 2018 context.

Run
---
python app_v2.py
# or specify port: GRADIO_SERVER_PORT=7861 python app_v2.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import build_transforms
from src.models.build_model import build_model
from src.utils.config import load_config
from src.utils.grad_cam import GradCAM, get_target_layer


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

APP_TITLE = "Skin Lesion Classification — Research Demo"
APP_SUBTITLE = "7-class dermoscopic image classification on ISIC 2018 / HAM10000"

MODELS: Dict[str, Dict[str, str]] = {
    'EfficientNet-B0 (Best)': {
        'config': 'src/configs/efficientnet_b0_optimized.yaml',
        'checkpoint': 'outputs/efficientnet_b0_v3(main_best)/best.pth',
        'model_name_hint': 'tf_efficientnet_b0',
    },
    'ResNet50': {
        'config': 'src/configs/resnet50_v3.yaml',
        'checkpoint': 'outputs/resnet50_v3(main)/best.pth',
        'model_name_hint': 'resnet50',
    },
    'DenseNet121': {
        'config': 'src/configs/densenet121_v3.yaml',
        'checkpoint': 'outputs/densenet121_v3(main)/best.pth',
        'model_name_hint': 'densenet121',
    },
    'Swin-Tiny': {
        'config': 'src/configs/swin_tiny_v3.yaml',
        'checkpoint': 'outputs/swin_tiny_v3/best.pth',
        'model_name_hint': 'swin_tiny',
    },
    'Ensemble (soft voting)': {'ensemble': True},
}

CLASS_INFO: Dict[str, Dict[str, str]] = {
    'akiec': {
        'full_name': 'Actinic keratosis / Intraepithelial carcinoma',
        'severity': 'Pre-malignant',
        'description': 'Rough scaly patches from long-term sun exposure. May progress to squamous cell carcinoma.',
    },
    'bcc': {
        'full_name': 'Basal cell carcinoma',
        'severity': 'Malignant (low metastasis)',
        'description': 'Most common skin cancer. Rarely metastasises but can cause local tissue damage.',
    },
    'bkl': {
        'full_name': 'Benign keratosis-like lesions',
        'severity': 'Benign',
        'description': 'Includes seborrhoeic keratoses, solar lentigo and lichen planus-like keratoses.',
    },
    'df': {
        'full_name': 'Dermatofibroma',
        'severity': 'Benign',
        'description': 'Small firm bumps caused by fibrous tissue growth. Harmless.',
    },
    'mel': {
        'full_name': 'Melanoma',
        'severity': '⚠️ Malignant (high risk)',
        'description': 'Most dangerous form of skin cancer. Early detection is critical for survival.',
    },
    'nv': {
        'full_name': 'Melanocytic nevi',
        'severity': 'Benign',
        'description': 'Common moles. Usually harmless but should be monitored for changes.',
    },
    'vasc': {
        'full_name': 'Vascular lesions',
        'severity': 'Benign',
        'description': 'Includes cherry angiomas, haemangiomas and pyogenic granulomas.',
    },
}

LOW_CONFIDENCE_THRESHOLD = 0.50
MALIGNANT_CLASSES = {'mel', 'bcc', 'akiec'}


# --------------------------------------------------------------------------- #
#  Model loading & caching
# --------------------------------------------------------------------------- #

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_MODEL_CACHE: Dict[str, Tuple[torch.nn.Module, dict]] = {}


def _get_model(model_key: str) -> Tuple[torch.nn.Module, dict]:
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    info = MODELS[model_key]
    cfg_path = Path(info['config'])
    ckpt_path = Path(info['checkpoint'])

    if not cfg_path.exists():
        raise FileNotFoundError(f'Config missing: {cfg_path}')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint missing: {ckpt_path}')

    cfg = load_config(cfg_path)
    model = build_model(cfg['model'])
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(_DEVICE).eval()

    _MODEL_CACHE[model_key] = (model, cfg)
    return model, cfg


def _predict_single(
    model_key: str,
    image: Image.Image,
) -> Tuple[np.ndarray, dict, torch.Tensor]:
    """Return (probs[C], cfg, input_tensor) for a single model."""
    model, cfg = _get_model(model_key)
    image_size = int(cfg['train']['image_size'])
    tfm = build_transforms(
        image_size=image_size,
        aug_cfg=cfg.get('augmentation', {}).get('eval', {}),
        is_train=False,
    )
    x = tfm(image.convert('RGB')).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs, cfg, x


def _predict_ensemble(image: Image.Image) -> Tuple[np.ndarray, dict]:
    """Soft-voting across individual models (skips the Ensemble entry itself)."""
    probs_list = []
    class_names: Optional[List[str]] = None
    base_cfg: Optional[dict] = None
    for key, info in MODELS.items():
        if info.get('ensemble'):
            continue
        if not Path(info['config']).exists() or not Path(info['checkpoint']).exists():
            continue
        probs, cfg, _ = _predict_single(key, image)
        probs_list.append(probs)
        if class_names is None:
            class_names = cfg['classes']
            base_cfg = cfg
    if not probs_list:
        raise RuntimeError('No models available for ensembling.')
    avg = np.mean(np.stack(probs_list, axis=0), axis=0)
    return avg, base_cfg


# --------------------------------------------------------------------------- #
#  Grad-CAM helper
# --------------------------------------------------------------------------- #

def _grad_cam_overlay(model_key: str, image: Image.Image) -> Optional[Image.Image]:
    """Compute Grad-CAM overlay for the given model + image."""
    if MODELS[model_key].get('ensemble'):
        return None
    model, cfg = _get_model(model_key)
    image_size = int(cfg['train']['image_size'])
    tfm = build_transforms(
        image_size=image_size,
        aug_cfg=cfg.get('augmentation', {}).get('eval', {}),
        is_train=False,
    )
    rgb = image.convert('RGB').resize((image_size, image_size))
    x = tfm(image.convert('RGB')).unsqueeze(0).to(_DEVICE)

    target_layer = get_target_layer(model, cfg['model']['name'])
    cam = GradCAM(model, target_layer)
    try:
        heatmap, _, _ = cam(x, target_class=None)
    finally:
        cam.release()

    rgb_np = np.array(rgb)
    overlay = GradCAM.overlay(rgb_np, heatmap, alpha=0.5)
    return Image.fromarray(overlay)


# --------------------------------------------------------------------------- #
#  Plotting helpers
# --------------------------------------------------------------------------- #

def _bar_chart(class_names: List[str], probs: np.ndarray):
    order = np.argsort(-probs)
    labels = [class_names[i] for i in order]
    values = probs[order]
    colors = ['#FF6B6B' if labels[i] in MALIGNANT_CLASSES else '#4ECDC4'
              for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(labels, values, color=colors, alpha=0.9, edgecolor='black')
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Probability', fontsize=11)
    ax.set_title('Class Probabilities (red = malignant/pre-malignant)',
                 fontsize=12, fontweight='bold')
    for bar, v in zip(bars, values):
        ax.text(min(v + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))  # normalised [0,1]


# --------------------------------------------------------------------------- #
#  Main prediction callback
# --------------------------------------------------------------------------- #

def predict(image: Image.Image, model_key: str, show_gradcam: bool):
    if image is None:
        raise gr.Error('Please upload a dermoscopy image first.')

    # 1. Inference
    if MODELS[model_key].get('ensemble'):
        probs, cfg = _predict_ensemble(image)
    else:
        probs, cfg, _ = _predict_single(model_key, image)

    class_names: List[str] = cfg['classes']
    top_idx = np.argsort(-probs)[:3]

    pred_class = class_names[int(top_idx[0])]
    pred_info = CLASS_INFO.get(pred_class, {})
    confidence = float(probs[top_idx[0]])

    # 2. Top-3 table
    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        name = class_names[int(idx)]
        info = CLASS_INFO.get(name, {})
        rows.append({
            'Rank': rank,
            'Class': name,
            'Full name': info.get('full_name', ''),
            'Severity': info.get('severity', ''),
            'Probability': round(float(probs[int(idx)]), 4),
        })
    top_df = pd.DataFrame(rows)

    # 3. Summary markdown with warnings
    severity_tag = pred_info.get('severity', '')
    is_low_conf = confidence < LOW_CONFIDENCE_THRESHOLD
    is_malignant = pred_class in MALIGNANT_CLASSES
    ent = _entropy(probs)

    warn_lines = []
    if is_low_conf:
        warn_lines.append(
            f'⚠️ **Low-confidence prediction** (top prob = {confidence:.2%} '
            f'below {LOW_CONFIDENCE_THRESHOLD:.0%}). Consider a specialist review.'
        )
    if ent > 0.6:
        warn_lines.append(
            f'⚠️ **High predictive entropy** ({ent:.2f}). '
            'Model is uncertain between several classes.'
        )
    if is_malignant and confidence >= LOW_CONFIDENCE_THRESHOLD:
        warn_lines.append(
            f'🔴 **Suspected malignant / pre-malignant lesion** — '
            'recommend prompt dermatological examination.'
        )

    warn_md = ('\n'.join(warn_lines)) if warn_lines else \
        '✅ No confidence / uncertainty warnings.'

    summary = (
        f'### Prediction: `{pred_class}`\n'
        f'**Full name:** {pred_info.get("full_name", "-")}\n\n'
        f'**Severity:** {severity_tag or "-"}\n\n'
        f'**Confidence:** {confidence:.2%}   •   '
        f'**Entropy:** {ent:.3f}   •   '
        f'**Model:** {model_key}\n\n'
        f'**Description:** {pred_info.get("description", "-")}\n\n'
        f'---\n{warn_md}'
    )

    # 4. Bar chart
    fig = _bar_chart(class_names, probs)

    # 5. Grad-CAM (skip for ensemble)
    cam_img: Optional[Image.Image] = None
    if show_gradcam and not MODELS[model_key].get('ensemble'):
        try:
            cam_img = _grad_cam_overlay(model_key, image)
        except Exception as e:
            cam_img = None
            summary += f'\n\n_⚠ Grad-CAM failed: {e}_'

    return summary, top_df, fig, cam_img


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #

def build_demo() -> gr.Blocks:
    css = """
    .title {text-align:center; font-size:1.8em; font-weight:700; margin:0;}
    .subtitle {text-align:center; color:#666; margin-bottom:18px;}
    .panel {border-radius:14px; box-shadow:0 6px 18px rgba(0,0,0,0.08); padding:8px;}
    .disclaimer {background:#fff3cd; border:1px solid #ffeaa7; padding:10px;
                 border-radius:8px; font-size:0.92em; color:#856404;}
    """

    available = [k for k, v in MODELS.items()
                 if v.get('ensemble') or
                 (Path(v.get('config', '')).exists() and Path(v.get('checkpoint', '')).exists())]
    default_model = available[0] if available else list(MODELS.keys())[0]

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"<div class='title'>{APP_TITLE}</div>")
        gr.Markdown(f"<div class='subtitle'>{APP_SUBTITLE}</div>")
        gr.Markdown(
            "<div class='disclaimer'>⚠ <b>Research prototype</b> for academic "
            "demonstration only. Not intended for clinical diagnosis. "
            "Always consult a qualified dermatologist.</div>"
        )

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group(elem_classes='panel'):
                    image_in = gr.Image(type='pil', label='Upload dermoscopy image',
                                        height=360)
                    model_dd = gr.Dropdown(
                        choices=available,
                        value=default_model,
                        label='Model',
                        info='Single CNN/Transformer or soft-voting Ensemble',
                    )
                    gradcam_chk = gr.Checkbox(
                        value=True, label='Show Grad-CAM heatmap '
                        '(disabled for Ensemble)',
                    )
                    with gr.Row():
                        predict_btn = gr.Button('Predict', variant='primary', size='lg')
                        clear_btn = gr.ClearButton(value='Clear', components=[image_in])

            with gr.Column(scale=6):
                with gr.Group(elem_classes='panel'):
                    summary_md = gr.Markdown(
                        '### Prediction\nUpload an image and click **Predict**.'
                    )
                    with gr.Row():
                        prob_plot = gr.Plot(label='Probability distribution')
                        cam_out = gr.Image(label='Grad-CAM overlay',
                                           type='pil', height=300)
                    top_table = gr.Dataframe(
                        headers=['Rank', 'Class', 'Full name', 'Severity', 'Probability'],
                        datatype=['number', 'str', 'str', 'str', 'number'],
                        row_count=(3, 'fixed'),
                        col_count=(5, 'fixed'),
                        label='Top-3 predictions',
                        wrap=True,
                    )

        predict_btn.click(
            fn=predict,
            inputs=[image_in, model_dd, gradcam_chk],
            outputs=[summary_md, top_table, prob_plot, cam_out],
        )

        with gr.Accordion('About this demo', open=False):
            gr.Markdown(
                """
                **Training data:** ISIC 2018 Task 3 — 10 015 dermoscopic images
                across 7 classes (heavily imbalanced, 58:1 NV vs DF).

                **Best single model:** EfficientNet-B0 — 88.4 % accuracy,
                0.831 macro-F1 on the held-out test set.

                **Ensemble (4 models, soft voting):** ~89 % accuracy, 0.980 AUC.

                **Interpretability:** Grad-CAM (Selvaraju et al., 2017) highlights
                regions most influential for the prediction. For CNNs we use the
                last convolutional block; for Swin-Tiny we use the final
                transformer layer's normalisation output.

                **Severity tags:**
                - 🔴 Malignant / Pre-malignant: `mel`, `bcc`, `akiec`
                - 🟢 Benign: `bkl`, `df`, `nv`, `vasc`
                """
            )

    return demo


if __name__ == '__main__':
    server_name = os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1')
    server_port = int(os.environ.get('GRADIO_SERVER_PORT', '7860'))
    demo = build_demo()
    demo.launch(server_name=server_name, server_port=server_port, show_error=True)
