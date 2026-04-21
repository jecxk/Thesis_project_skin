import torch
from PIL import Image
import numpy as np
from src.models.build_model import build_model
from src.utils.config import load_config
from src.utils.grad_cam import GradCAM, get_target_layer
from torchvision import transforms

def main():
    print("Loading config...")
    cfg = load_config('src/configs/efficientnet_b0_optimized.yaml')
    print("Building model...")
    model = build_model(cfg['model'])
    
    # Load dummy image
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0)
    
    print("Getting target layer...")
    target_layer = get_target_layer(model, 'efficientnet_b0')
    print(f"Target layer: {target_layer}")
    
    print("Initializing GradCAM...")
    cam = GradCAM(model, target_layer)
    
    print("Running GradCAM...")
    heatmap, pred_class, conf = cam(img_t)
    print("Success!")
    print("Heatmap shape:", heatmap.shape)

if __name__ == '__main__':
    main()
