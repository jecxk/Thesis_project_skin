"""Pick demo images based on ENSEMBLE predictions (not single model)."""
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import shutil, os

from src.models.build_model import build_model
from src.utils.config import load_config

DEVICE = 'cpu'
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

MODELS_INFO = [
    ('EfficientNet-B0', 'src/configs/efficientnet_b0_optimized.yaml', 'outputs/efficientnet_b0_v3(main_best)/best.pth'),
    ('ResNet50', 'src/configs/resnet50_v3.yaml', 'outputs/resnet50_v3(main)/best.pth'),
    ('DenseNet121', 'src/configs/densenet121_v3.yaml', 'outputs/densenet121_v3(main)/best.pth'),
    ('Swin-Tiny', 'src/configs/swin_tiny_v3.yaml', 'outputs/swin_tiny_v3(main)/best.pth'),
]

# Load all models
models = []
for name, config_path, ckpt_path in MODELS_INFO:
    cfg = load_config(config_path)
    model = build_model(cfg['model'])
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()
    img_size = cfg['train'].get('image_size', 224)
    models.append((name, model, img_size))
    print(f"  Loaded {name}")

# Load test split
df = pd.read_csv('data/metadata/skin_metadata.csv')
test_df = df[df['split'] == 'test'].reset_index(drop=True)
print(f"\nTest set: {len(test_df)} images")

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Run ensemble on all test images
results = []
with torch.no_grad():
    for idx, row in test_df.iterrows():
        img_path = row['image_path']
        true_label = int(row['label'])
        
        if not os.path.exists(img_path):
            continue
        
        image = Image.open(img_path).convert('RGB')
        all_probs = []
        
        for name, model, img_size in models:
            tfm = get_transform(img_size)
            img_t = tfm(image).unsqueeze(0).to(DEVICE)
            out = model(img_t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
        
        avg_probs = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))
        confidence = avg_probs[pred_idx]
        
        results.append({
            'image_path': img_path,
            'true_label': true_label,
            'pred_label': pred_idx,
            'correct': pred_idx == true_label,
            'confidence': confidence,
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(test_df)}...")

res_df = pd.DataFrame(results)
print(f"\nEnsemble accuracy: {res_df['correct'].mean()*100:.2f}%")

# Pick top 2 per class
out_dir = 'demo_images'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

for cls_idx, cls_name in enumerate(CLASSES):
    correct = res_df[(res_df['true_label'] == cls_idx) & (res_df['correct'] == True)]
    top2 = correct.nlargest(2, 'confidence')
    
    for rank, (_, row) in enumerate(top2.iterrows(), 1):
        src = row['image_path']
        conf = row['confidence'] * 100
        basename = os.path.basename(src)
        dst_name = f"{cls_name}_{rank}_{conf:.0f}pct_{basename}"
        shutil.copy2(src, os.path.join(out_dir, dst_name))
        print(f"  {dst_name}")

print(f"\nDone! {len(os.listdir(out_dir))} demo images ready.")
