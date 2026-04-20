from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from src.data.transforms import build_transforms
from src.models.build_model import build_model
from src.utils.config import load_config

APP_TITLE = "Skin Lesion Classification Demo"
APP_SUBTITLE = "Research prototype for 7-class dermoscopic image classification"
DEFAULT_CONFIG = "src/configs/efficientnet_b0.yaml"
DEFAULT_CHECKPOINT = "outputs/efficientnet_b0_ce_baseline/best.pth"

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

MODEL_CACHE: Dict[Tuple[str, str], Tuple[torch.nn.Module, dict, str]] = {}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def load_model_and_cfg(config_path: str, checkpoint_path: str):
    key = (config_path, checkpoint_path)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    cfg_path = resolve_path(config_path)
    ckpt_path = resolve_path(checkpoint_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = load_config(cfg_path)
    device = get_device()

    model = build_model(cfg["model"])
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    MODEL_CACHE[key] = (model, cfg, device)
    return model, cfg, device


def build_eval_transform(cfg: dict):
    image_size = int(cfg["train"]["image_size"])
    aug_cfg = cfg.get("augmentation", {}).get("eval", {})
    return build_transforms(image_size=image_size, aug_cfg=aug_cfg, is_train=False)


def make_prob_figure(class_names: List[str], probs: List[float]):
    pairs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(labels, values)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Class Probabilities")

    for i, v in enumerate(values):
        ax.text(min(v + 0.01, 0.98), i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def predict_image(image: Image.Image, config_path: str, checkpoint_path: str):
    if image is None:
        raise gr.Error("Please upload an image first.")

    model, cfg, device = load_model_and_cfg(config_path, checkpoint_path)
    tfm = build_eval_transform(cfg)
    class_names = cfg["classes"]

    image_rgb = image.convert("RGB")
    x = tfm(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    top_rows = []
    for rank, idx in enumerate(top_indices, start=1):
        name = class_names[idx]
        top_rows.append({
            "Rank": rank,
            "Class": name,
            "Probability": round(probs[idx], 4),
            "Description": CLASS_DESCRIPTIONS.get(name, ""),
        })
    top_df = pd.DataFrame(top_rows)

    pred_idx = top_indices[0]
    pred_name = class_names[pred_idx]
    confidence = probs[pred_idx]
    summary = (
        f"### Prediction Summary\n"
        f"**Predicted class:** `{pred_name}`  \n"
        f"**Confidence:** `{confidence:.2%}`  \n"
        f"**Description:** {CLASS_DESCRIPTIONS.get(pred_name, '-') }"
    )

    fig = make_prob_figure(class_names, probs)
    note = (
        "This is a research prototype for academic demonstration only. "
        "It is not intended for clinical diagnosis."
    )
    return summary, top_df, fig, note


def build_demo() -> gr.Blocks:
    css = """
    .app-title {text-align:center; margin-bottom: 4px;}
    .app-subtitle {text-align:center; color:#555; margin-bottom: 18px;}
    .panel {border-radius: 18px; box-shadow: 0 8px 22px rgba(0,0,0,0.08);}
    .footer-note {font-size: 13px; color:#555;}
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"## <div class='app-title'>{APP_TITLE}</div>")
        gr.Markdown(f"<div class='app-subtitle'>{APP_SUBTITLE}</div>")

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group(elem_classes="panel"):
                    image_input = gr.Image(type="pil", label="Upload dermoscopy image")
                    with gr.Row():
                        predict_btn = gr.Button("Predict", variant="primary")
                        clear_btn = gr.ClearButton(value="Clear", components=[image_input])

                    config_path = gr.Textbox(
                        value=DEFAULT_CONFIG,
                        label="Config path",
                        info="Relative to project root",
                    )
                    checkpoint_path = gr.Textbox(
                        value=DEFAULT_CHECKPOINT,
                        label="Checkpoint path",
                        info="Path to best.pth",
                    )

            with gr.Column(scale=5):
                with gr.Group(elem_classes="panel"):
                    summary_md = gr.Markdown("### Prediction Summary\nUpload an image and click Predict.")
                    top_table = gr.Dataframe(
                        headers=["Rank", "Class", "Probability", "Description"],
                        datatype=["number", "str", "number", "str"],
                        row_count=3,
                        col_count=(4, "fixed"),
                        label="Top-3 Predictions",
                    )
                    prob_plot = gr.Plot(label="Probability Chart")
                    footer_note = gr.Markdown(
                        "<div class='footer-note'>This is a research prototype and is not intended for clinical diagnosis.</div>"
                    )

        predict_btn.click(
            fn=predict_image,
            inputs=[image_input, config_path, checkpoint_path],
            outputs=[summary_md, top_table, prob_plot, footer_note],
        )

        gr.Examples(
            examples=[],
            inputs=[image_input],
            label="You can upload your own test images or add demo samples later.",
        )

    return demo


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo = build_demo()
    demo.launch(server_name=server_name, server_port=server_port, show_error=True)
