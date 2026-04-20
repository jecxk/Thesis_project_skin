# Skin Thesis Project

A config-driven PyTorch project for 7-class skin lesion classification using public dermoscopy datasets such as HAM10000 and ISIC 2018.

## Features
- Comparative experiments with ResNet50, DenseNet121, EfficientNet-B0, and Swin-Tiny
- Cross-entropy, weighted cross-entropy, and focal loss
- Optional weighted sampling for imbalanced classes
- Training, evaluation, confusion matrix, and prediction export
- Designed for local RTX training and Colab fallback

## Expected CSV format
Create a CSV file with at least these columns:
- `image_path`: relative or absolute path to image
- `label`: integer class id from 0 to 6
- `class_name`: optional readable class name
- `split`: one of `train`, `val`, `test`

Example:
```csv
image_path,label,class_name,split
images/ISIC_0000000.jpg,0,akiec,train
images/ISIC_0000001.jpg,1,bcc,val
images/ISIC_0000002.jpg,2,bkl,test
```

## Class order
Default order in configs:
- 0: akiec
- 1: bcc
- 2: bkl
- 3: df
- 4: mel
- 5: nv
- 6: vasc

Adjust if your metadata uses a different mapping.

## Quick start
```bash
pip install -r requirements.txt
python -m src.train --config src/configs/efficientnet_b0.yaml
python -m src.evaluate --config src/configs/efficientnet_b0.yaml --checkpoint outputs/efficientnet_b0/best.pth
```

## Colab
In Colab:
```python
!git clone <your-repo-url>
%cd skin_thesis_project
!pip install -r requirements.txt
!python -m src.train --config src/configs/efficientnet_b0.yaml
```

Update dataset paths inside the YAML config before training.
