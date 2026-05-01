import sys
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Important for Windows PyTorch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Add parent directory to sys.path so 'src' can be found when run as a subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.build_model import build_model
from src.utils.config import load_config
from src.utils.grad_cam import GradCAM, get_target_layer
from torchvision import transforms

# Define models info (must match app.py)
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

def main():
    if len(sys.argv) != 5:
        print("Usage: python worker.py <image_path> <model_name> <target_class_idx> <output_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    model_name = sys.argv[2]
    target_class_idx = int(sys.argv[3])
    if target_class_idx == -1:
        target_class_idx = None
    output_path = sys.argv[4]
    
    DEVICE = 'cpu'
    
    try:
        # Load image
        image_pil = Image.open(image_path).convert('RGB')
        
        # Find model info
        info = next(i for i in MODELS_INFO if i['name'] == model_name)
        
        cfg = load_config(info['config'])
        model = build_model(cfg['model'])
        checkpoint = torch.load(info['checkpoint'], map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        img_size = cfg['train'].get('image_size', 224)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_t = transform(image_pil).unsqueeze(0).to(DEVICE)
        
        target_layer = get_target_layer(model, cfg['model']['name'])
        cam = GradCAM(model, target_layer)
        
        # Generate heatmap
        grayscale_cam, _, _ = cam(img_t, target_class_idx)
        cam.release()
        
        # Use existing overlay function which handles resizing and colormaps correctly via OpenCV
        img_arr = np.array(image_pil.resize((img_size, img_size)))
        cam_image = GradCAM.overlay(img_arr, grayscale_cam, alpha=0.5)
        
        # Save output
        out_img = Image.fromarray(cam_image)
        out_img.save(output_path)
        sys.exit(0)
        
    except Exception as e:
        print(f"Error in worker: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
