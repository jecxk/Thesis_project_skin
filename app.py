import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os

import sys
from src.models.build_model import build_model
from src.utils.config import load_config

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Skin Lesion Diagnosis AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hardcode device to CPU for web demo to avoid Out-of-Memory issues with concurrent training
DEVICE = 'cpu'

# Class mapping and descriptions
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES_FULL = {
    'akiec': 'Actinic Keratoses (Pre-cancerous)',
    'bcc': 'Basal Cell Carcinoma (Cancerous)',
    'bkl': 'Benign Keratosis-like Lesions (Benign)',
    'df': 'Dermatofibroma (Benign)',
    'mel': 'Melanoma (Cancerous - High Risk)',
    'nv': 'Melanocytic Nevi (Benign Mole)',
    'vasc': 'Vascular Lesions (Benign)'
}
MODELS_INFO = [
    {
        'id': 'eb0',
        'name': 'EfficientNet-B0',
        'config': 'src/configs/efficientnet_b0_optimized.yaml',
        'checkpoint': 'outputs/efficientnet_b0_v3(main_best)/best.pth',
    },
    {
        'id': 'rn50',
        'name': 'ResNet50',
        'config': 'src/configs/resnet50_v3.yaml',
        'checkpoint': 'outputs/resnet50_v3(main)/best.pth',
    },
    {
        'id': 'dn121',
        'name': 'DenseNet121',
        'config': 'src/configs/densenet121_v3.yaml',
        'checkpoint': 'outputs/densenet121_v3(main)/best.pth',
    },
    {
        'id': 'swin',
        'name': 'Swin-Tiny',
        'config': 'src/configs/swin_tiny_v3.yaml',
        'checkpoint': 'outputs/swin_tiny_v3(main)/best.pth',
    }
]

# --- UI STYLING ---
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin: 0; }
    .sub-header  { font-size: 1.0rem; color: #6B7280; margin: 4px 0 1.5rem; }
    .pred-card {
        padding: 1.1rem 1.3rem;
        border-radius: 0.6rem;
        background: rgba(148, 163, 184, 0.08);
        border-left: 4px solid var(--pred-accent, #3b82f6);
        margin-bottom: 1rem;
    }
    .pred-card .pred-class {
        font-size: 1.55rem; font-weight: 600;
        line-height: 1.2; margin: 0;
    }
    .pred-card .pred-meta {
        color: #94a3b8; margin-top: 6px; font-size: 0.95rem;
    }
    .pred-card .pred-meta b { color: var(--pred-accent, #3b82f6); }
</style>
""", unsafe_allow_html=True)


# --- CACHED FUNCTIONS ---
@st.cache_resource(show_spinner="Loading models...")
def load_all_models():
    models_dict = {}
    for info in MODELS_INFO:
        if not os.path.exists(info['checkpoint']):
            st.warning(f"Checkpoint not found for {info['name']}")
            continue
        
        cfg = load_config(info['config'])
        model = build_model(cfg['model'])
        checkpoint = torch.load(info['checkpoint'], map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        # Determine image size from config
        img_size = cfg['train'].get('image_size', 224)
        
        models_dict[info['name']] = {
            'model': model,
            'config': cfg,
            'img_size': img_size
        }
    return models_dict

# Preprocessing transform
def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_ensemble(image_pil, models_dict):
    """Run inference on all models and average probabilities."""
    all_probs = []
    
    with torch.no_grad():
        for name, data in models_dict.items():
            transform = get_transform(data['img_size'])
            img_t = transform(image_pil).unsqueeze(0).to(DEVICE)
            
            outputs = data['model'](img_t)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            
    # Soft voting: average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = np.argmax(avg_probs)
    return avg_probs, pred_idx


import subprocess
import tempfile
import uuid

def generate_gradcam(image_pil, model_name, target_class_idx):
    """Generate Grad-CAM heatmap safely by isolating the PyTorch backward pass in a subprocess."""
    
    # Save the input image to a temporary file
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    in_path = os.path.join(temp_dir, f"in_{unique_id}.jpg")
    out_path = os.path.join(temp_dir, f"out_{unique_id}.png")
    
    image_pil.save(in_path)
    
    try:
        # Run the isolated worker script
        cmd = [
            sys.executable, 
            "src/gradcam_worker.py", 
            in_path, 
            model_name, 
            str(target_class_idx), 
            out_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            
        if not os.path.exists(out_path):
            raise Exception("Subprocess finished but output file was not created.")
            
        # Read the result
        cam_image = Image.open(out_path).convert('RGB')
        return np.array(cam_image)
        
    finally:
        # Cleanup temp files
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)


# --- MAIN APP ---
def main():
    # Sidebar
    with st.sidebar:
        if os.path.exists("thesis/figures/logo.png"):
            st.image("thesis/figures/logo.png", width=150)

        models_dict = load_all_models()
        model_names = list(models_dict.keys())

        st.markdown("**Grad-CAM viewer**")
        selected_cam_model = st.selectbox(
            "Backbone to inspect",
            model_names,
            index=0,
            help="Prediction uses an ensemble of all four models. This dropdown only changes the heatmap shown below.",
        )
        st.caption(
            "Prediction always runs the soft-voting ensemble of all four "
            "backbones. The choice here only affects the Grad-CAM heatmap."
        )

        st.markdown("---")
        with st.expander("Class reference", expanded=False):
            for k, v in CLASS_NAMES_FULL.items():
                st.markdown(f"**{k.upper()}** — {v}")

    # Main content
    st.markdown('<p class="main-header">🩺 Skin Lesion Diagnosis System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ISIC 2018 · 7-class dermoscopic classification</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a dermoscopy image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Create two columns for Layout
            col1, col2 = st.columns([1, 1.2], gap="large")
            
            with col1:
                st.markdown("### Original image")
                st.image(image, width='stretch')

            with col2:
                st.markdown("### Ensemble prediction")
                with st.spinner("Running ensemble..."):
                    probs, pred_idx = predict_ensemble(image, models_dict)

                pred_class = CLASSES[pred_idx]
                confidence = probs[pred_idx] * 100
                full_name = CLASS_NAMES_FULL[pred_class]

                # Severity-based accent — muted, not gradient.
                if "Cancerous" in full_name and "Pre-cancerous" not in full_name:
                    accent = "#dc2626"     # cancer
                elif "Pre-cancerous" in full_name:
                    accent = "#d97706"     # pre-malignant
                else:
                    accent = "#0284c7"     # benign

                st.markdown(f"""
                <div class="pred-card" style="--pred-accent:{accent};">
                    <div class="pred-class">{full_name}</div>
                    <div class="pred-meta">Confidence&nbsp;<b>{confidence:.2f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Native Streamlit Bar Chart (much more stable than Plotly)
                df_probs = pd.DataFrame({
                    'Probability': probs * 100
                }, index=[CLASS_NAMES_FULL[c].split(' (')[0] for c in CLASSES])
                df_probs = df_probs.sort_values('Probability', ascending=True)
                
                st.bar_chart(df_probs, height=300)

            # --- GRAD-CAM SECTION ---
            st.markdown("---")
            st.markdown("### Grad-CAM")
            st.caption(
                f"Heatmap generated from {selected_cam_model}. "
                "Warmer regions contributed more strongly to the prediction."
            )

            cam_col1, cam_col2 = st.columns([1, 1.2], gap="large")

            with cam_col1:
                try:
                    with st.spinner("Generating Grad-CAM..."):
                        cam_img = generate_gradcam(image, selected_cam_model, -1)
                    st.image(cam_img, width='stretch')
                except Exception as cam_err:
                    import traceback
                    st.error(f"Error generating Grad-CAM: {str(cam_err)}")
                    st.code(traceback.format_exc())

            with cam_col2:
                clinical_notes = {
                    'mel': "Melanoma — a high-risk skin cancer. Typical dermoscopic clues include asymmetry, irregular borders and colour variegation. Dermatological review is recommended.",
                    'nv':  "Melanocytic nevus — a benign mole. Tends to show well-defined, symmetric borders and uniform pigmentation.",
                    'bcc': "Basal cell carcinoma — a common skin cancer. Common features include arborising vessels, blue-grey ovoid nests and ulceration.",
                    'akiec': "Actinic keratosis / intraepithelial carcinoma — pre-malignant, caused by long-term sun exposure.",
                    'bkl': "Benign keratosis-like lesion — visually similar to melanoma and frequently confused with it.",
                    'df':  "Dermatofibroma — a benign dermal nodule from fibrous-tissue growth.",
                    'vasc': "Vascular lesion — typically benign vascular proliferations such as angiomas.",
                }
                with st.expander(f"Notes on {pred_class.upper()}", expanded=False):
                    st.write(clinical_notes.get(pred_class, ""))
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == '__main__':
    main()
