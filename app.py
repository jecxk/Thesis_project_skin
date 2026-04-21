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
CLASS_COLORS = {
    'akiec': '#FFA07A',  # Light Salmon
    'bcc': '#DC143C',    # Crimson
    'bkl': '#8FBC8F',    # Dark Sea Green
    'df': '#20B2AA',     # Light Sea Green
    'mel': '#8B0000',    # Dark Red (Danger)
    'nv': '#4682B4',     # Steel Blue
    'vasc': '#DA70D6'    # Orchid
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
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0rem; }
    .sub-header { font-size: 1.2rem; color: #6B7280; margin-bottom: 2rem; }
    .pred-box { padding: 1.5rem; border-radius: 0.5rem; color: white; text-align: center; margin-bottom: 1rem; }
    .pred-cancer { background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%); }
    .pred-benign { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); }
    .pred-pre { background: linear-gradient(135deg, #f59e0b 0%, #b45309 100%); }
</style>
""", unsafe_allow_html=True)


# --- CACHED FUNCTIONS ---
@st.cache_resource(show_spinner="Loading deep learning models into memory...")
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
        st.markdown("### ⚙️ Configuration")
        
        models_dict = load_all_models()
        model_names = list(models_dict.keys())
        
        st.markdown("**Grad-CAM Interpretability Model**")
        selected_cam_model = st.selectbox(
            "Select which model to use for visualization:", 
            model_names,
            index=0,
            help="All models are used for the Ensemble prediction. This selection only affects the Grad-CAM heatmap."
        )
        
        st.markdown("---")
        st.markdown("### 📚 Disease Dictionary")
        for k, v in CLASS_NAMES_FULL.items():
            color = CLASS_COLORS[k]
            st.markdown(f"**<span style='color:{color}'>{k.upper()}</span>**: {v}", unsafe_allow_html=True)

    # Main content
    st.markdown('<p class="main-header">🩺 Skin Lesion Diagnosis System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Ensemble Deep Learning (EfficientNet, ResNet, DenseNet, Swin Transformer)</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a Dermoscopy Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Create two columns for Layout
            col1, col2 = st.columns([1, 1.2], gap="large")
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True, caption="Uploaded Dermoscopy Image")
                
            with col2:
                st.markdown("### Ensemble Prediction Analysis")
                with st.spinner("Running 4 models & ensembling results..."):
                    probs, pred_idx = predict_ensemble(image, models_dict)
                    
                pred_class = CLASSES[pred_idx]
                confidence = probs[pred_idx] * 100
                full_name = CLASS_NAMES_FULL[pred_class]
                
                # Dynamic styling based on severity
                if "Cancerous" in full_name:
                    box_class = "pred-cancer"
                    icon = "⚠️"
                elif "Pre-cancerous" in full_name:
                    box_class = "pred-pre"
                    icon = "🔍"
                else:
                    box_class = "pred-benign"
                    icon = "✅"
                    
                st.markdown(f"""
                <div class="pred-box {box_class}">
                    <h2 style="margin:0;color:white;">{icon} {full_name}</h2>
                    <p style="font-size:1.2rem;margin-top:5px;color:white;">Confidence: <b>{confidence:.2f}%</b></p>
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
            st.markdown("### 🔬 Interpretability: What is the AI looking at?")
            
            cam_col1, cam_col2 = st.columns([1, 1.2], gap="large")
            
            with cam_col1:
                try:
                    with st.spinner(f"Generating Grad-CAM using {selected_cam_model} (loading fresh weights to ensure stability)..."):
                        cam_img = generate_gradcam(image, selected_cam_model, pred_idx)
                    
                    st.image(cam_img, use_container_width=True, caption=f"Grad-CAM Heatmap ({selected_cam_model})")
                except Exception as cam_err:
                    import traceback
                    st.error(f"Error generating Grad-CAM: {str(cam_err)}")
                    st.code(traceback.format_exc())
                
            with cam_col2:
                st.info("""
                **How to read this heatmap:**
                - **Red/Yellow regions** indicate the areas the AI focused on the most to make its decision.
                - **Blue regions** indicate areas the AI ignored.
                - For cancerous lesions (like Melanoma), the AI typically focuses on irregular borders, asymmetric pigmentation, or specific dermoscopic structures.
                """)
                
                if pred_class == 'mel':
                    st.warning("**Clinical Note:** Melanoma is a high-risk skin cancer. The AI is likely attending to asymmetry, border irregularity, or color variegation. Immediate dermatological review is recommended.")
                elif pred_class == 'nv':
                    st.success("**Clinical Note:** Melanocytic Nevi are benign moles. The AI usually focuses on the well-defined, symmetric borders and uniform pigmentation.")
                elif pred_class == 'bcc':
                    st.warning("**Clinical Note:** Basal Cell Carcinoma is a common skin cancer. The AI may be detecting features like arborizing vessels, blue-gray ovoid nests, or ulceration.")
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == '__main__':
    main()
