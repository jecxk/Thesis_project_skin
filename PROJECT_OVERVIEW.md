# Project Overview — Skin Lesion Classification

> **Mục đích file này:** Tài liệu tổng hợp toàn bộ project — từ dữ liệu, code,
> cách huấn luyện, đến kết quả. Đọc file này là hiểu được toàn bộ hệ thống.

---

## Mục lục

1. [Tổng quan project](#1-tổng-quan-project)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Dữ liệu](#3-dữ-liệu)
4. [Kiến trúc code](#4-kiến-trúc-code)
5. [Pretrain vs Fine-tune — Định nghĩa](#5-pretrain-vs-fine-tune--định-nghĩa)
6. [Pipeline huấn luyện](#6-pipeline-huấn-luyện)
7. [Các model và config](#7-các-model-và-config)
8. [Ensemble](#8-ensemble)
9. [Ablation Study](#9-ablation-study)
10. [Kết quả và chỉ số](#10-kết-quả-và-chỉ-số)

---

## 1. Tổng quan project

**Bài toán:** Phân loại 7 loại tổn thương da từ ảnh dermoscopy (ảnh da soi kính).

**Dataset:** ISIC 2018 Task 3 — 10,015 ảnh, 7 lớp, mất cân bằng nghiêm trọng (tỉ lệ 58:1 giữa lớp đông nhất và ít nhất).

**Mô hình đánh giá:** EfficientNet-B0, ResNet50, DenseNet121, Swin-Tiny — đều được **fine-tune** từ ImageNet pretrained weights.

**Kết quả tốt nhất:**
- Đơn lẻ: EfficientNet-B0 — Accuracy 88.36%, Macro F1 0.831, AUC 0.952
- Ensemble 4 model: Accuracy 89.49%, Macro F1 0.845, AUC **0.980**

---

## 2. Cấu trúc thư mục

```
d:\skin_thesis_project\
├── src/                          # Toàn bộ source code Python
│   ├── configs/                  # File YAML cấu hình từng experiment
│   │   ├── ablations/            # Config cho ablation study
│   │   ├── efficientnet_b0_optimized.yaml   ← config chính EB0
│   │   ├── resnet50_v3.yaml
│   │   ├── densenet121_v3.yaml
│   │   └── swin_tiny_v3.yaml
│   ├── data/                     # Xử lý dữ liệu
│   │   ├── dataset.py            # Dataset class + CSV loader
│   │   ├── transforms.py         # Augmentation pipeline
│   │   ├── samplers.py           # Weighted Random Sampler
│   │   └── mixup.py              # Mixup + CutMix logic
│   ├── models/
│   │   ├── build_model.py        # Tạo model từ timm
│   │   └── losses.py             # Cross-entropy + label smoothing
│   ├── engine/
│   │   └── train_eval.py         # Vòng lặp train/eval từng epoch
│   ├── utils/
│   │   ├── config.py             # Đọc YAML config
│   │   ├── metrics.py            # Tính toàn bộ metrics
│   │   ├── plots.py              # Vẽ confusion matrix, ROC, training curves
│   │   ├── grad_cam.py           # Grad-CAM implementation
│   │   ├── gradcam_quant.py      # Metrics định lượng cho Grad-CAM
│   │   ├── calibration.py        # Temperature scaling calibration
│   │   ├── io.py                 # save_json, ensure_dir
│   │   └── seed.py               # Fix random seed
│   ├── train.py                  # ← ENTRY POINT: chạy training
│   ├── evaluate.py               # Evaluate model trên test set
│   ├── ensemble.py               # Soft-voting ensemble
│   ├── grad_cam_analysis.py      # Generate Grad-CAM visualization grid
│   ├── gradcam_quant_runner.py   # Chạy Grad-CAM quantitative metrics
│   ├── gradcam_worker.py         # Worker subprocess cho Grad-CAM (dùng trong app)
│   ├── error_analysis.py         # Phân tích lỗi, hard cases
│   ├── calibration_analysis.py   # Phân tích calibration
│   ├── threshold_tuning.py       # Tối ưu decision threshold
│   ├── summarize_ablation.py     # Tổng hợp ablation results
│   └── tta.py                    # Test-Time Augmentation
├── data/
│   └── metadata/
│       └── skin_metadata.csv     # ← FILE QUAN TRỌNG: map ảnh → label → split
├── outputs/                      # Kết quả training (auto-generated)
│   ├── efficientnet_b0_v3(main_best)/   # Checkpoint tốt nhất EB0
│   ├── resnet50_v3(main)/
│   ├── densenet121_v3(main)/
│   ├── swin_tiny_v3(main)/
│   ├── ensemble/                 # Kết quả ensemble
│   ├── ablation_no_mixup/
│   ├── ablation_no_sampler/
│   ├── ablation_no_smooth/
│   ├── ablation_no_adv_aug/
│   ├── grad_cam_analysis/        # Ảnh Grad-CAM visualization
│   └── ablations_summary/
├── thesis/                       # LaTeX thesis + figures
│   ├── main.tex                  # File LaTeX chính
│   ├── chapter3.tex              # Methodology
│   ├── chapter4.tex              # Experiments & Results
│   ├── chapter5.tex              # Conclusion
│   ├── references.bib            # Bibliography
│   └── figures/                  # Ảnh dùng trong thesis
├── app.py                        # Streamlit web demo v1
├── app_v2.py                     # Streamlit web demo v2 (đang dùng)
├── requirements.txt              # Python dependencies
└── test_gradcam.py               # Script test nhanh Grad-CAM
```

**Ảnh gốc** nằm ở ổ đĩa riêng:
```
D:\skin_project\data\raw\ISIC2018_Task3_Training_Input\
    ISIC_0024306.jpg
    ISIC_0024307.jpg
    ...  (10,015 ảnh .jpg)
```

---

## 3. Dữ liệu

### 3.1 Dataset gốc

**ISIC 2018 Task 3 — HAM10000**

| Lớp | Tên đầy đủ | Train | Val | Test | Tổng |
|-----|-----------|-------|-----|------|------|
| `nv` | Melanocytic Nevus | 4,693 | 1,006 | 1,006 | 6,705 |
| `mel` | Melanoma | 779 | 167 | 167 | 1,113 |
| `bkl` | Benign Keratosis | 769 | 165 | 165 | 1,099 |
| `bcc` | Basal Cell Carcinoma | 360 | 77 | 77 | 514 |
| `akiec` | Actinic Keratosis | 229 | 49 | 49 | 327 |
| `vasc` | Vascular Lesion | 99 | 21 | 22 | 142 |
| `df` | Dermatofibroma | 81 | 17 | 17 | 115 |
| **Tổng** | | **7,010** | **1,502** | **1,503** | **10,015** |

**Tỉ lệ mất cân bằng:** `nv` (4,693) vs `df` (81) = **58:1**

**Label encoding:**
```
0 = akiec  |  1 = bcc  |  2 = bkl  |  3 = df
4 = mel    |  5 = nv   |  6 = vasc
```

### 3.2 File metadata

**[data/metadata/skin_metadata.csv](data/metadata/skin_metadata.csv)**

```
image_path,label,class_name,split
D:\skin_project\data\raw\...\ISIC_0026323.jpg,5,nv,train
D:\skin_project\data\raw\...\ISIC_0030604.jpg,4,mel,train
...
```

4 cột: `image_path` (đường dẫn tuyệt đối), `label` (int 0-6), `class_name` (string), `split` (`train`/`val`/`test`).

Split được tạo theo **stratified sampling 70/15/15** — mỗi split giữ nguyên tỉ lệ các lớp.

### 3.3 Xử lý dữ liệu — code flow

**Bước 1 — Đọc CSV:** [`src/data/dataset.py`](src/data/dataset.py) → `load_split_dataframes()`

```python
# Đọc CSV, chia thành 3 DataFrame theo cột 'split'
train_df, val_df, test_df = load_split_dataframes("data/metadata/skin_metadata.csv")
```

**Bước 2 — Dataset class:** `SkinLesionDataset(df, root_dir, transform)`

- Mỗi `__getitem__` mở ảnh bằng PIL, convert sang RGB, apply transform
- Trả về `(image_tensor, label_int, image_path_str)`

**Bước 3 — Augmentation:** [`src/data/transforms.py`](src/data/transforms.py) → `build_transforms(image_size, aug_cfg, is_train)`

| Augmentation | Train | Eval |
|---|---|---|
| Resize → `224×224` | ✓ | ✓ |
| RandomResizedCrop (scale 0.85–1.0) | ✓ | ✗ |
| Horizontal flip (p=0.5) | ✓ | ✗ |
| Vertical flip (p=0.5) | ✓ | ✗ |
| Rotation ±90° | ✓ | ✗ |
| Color Jitter (brightness/contrast/sat/hue 0.25) | ✓ | ✗ |
| Gaussian Blur (p=0.1) | ✓ | ✗ |
| Random Erasing (p=0.1) | ✓ | ✗ |
| Normalize (ImageNet mean/std) | ✓ | ✓ |

**Bước 4 — Weighted Sampler:** [`src/data/samplers.py`](src/data/samplers.py) → `build_weighted_sampler()`

```python
# Trọng số mỗi sample = 1 / (số lượng class của sample đó)
# → mỗi mini-batch xấp xỉ cân bằng 7 lớp
weight_per_sample = class_weights[label_of_sample]
sampler = WeightedRandomSampler(weight_per_sample, num_samples=len(train_ds))
```

Công thức tính class weight: `w_c = N / (C × n_c)`
- N = tổng samples, C = số lớp, n_c = số samples lớp c

**Bước 5 — Mixup / CutMix:** [`src/data/mixup.py`](src/data/mixup.py)

Áp dụng **trong vòng lặp training**, mỗi batch có xác suất 0.5 được chọn 1 trong 2:

- **Mixup:** `x̃ = λ·xᵢ + (1-λ)·xⱼ`, label soft `ỹ = λ·yᵢ + (1-λ)·yⱼ` với `λ ~ Beta(0.2, 0.2)`
- **CutMix:** Cắt patch ngẫu nhiên của ảnh j dán vào ảnh i, label theo tỉ lệ diện tích

---

## 4. Kiến trúc code

### 4.1 Entry point — Training

**[src/train.py](src/train.py)** — chạy bằng:
```bash
python -m src.train --config src/configs/efficientnet_b0_optimized.yaml
```

Flow:
```
load_config(yaml) 
  → build dataset + dataloader (với weighted sampler nếu bật)
  → build_model (timm, pretrained=True)
  → build_loss (cross-entropy + label smoothing)
  → build_optimizer (AdamW)
  → build_scheduler (cosine warmup)
  → training loop 50 epochs:
      train_one_epoch() → evaluate_one_epoch()
      → lưu best.pth khi macro_f1 cải thiện
      → early stopping patience=12
  → evaluate best.pth trên test set
  → lưu: test_metrics.json, confusion_matrix, ROC curves, history.json
```

### 4.2 Vòng lặp train/eval

**[src/engine/train_eval.py](src/engine/train_eval.py)** — 2 hàm chính:

**`train_one_epoch()`:**
1. model.train()
2. Với mỗi batch: apply Mixup/CutMix → forward → loss → backward → grad_clip(1.0) → optimizer.step()
3. Mixed precision (FP16) với `torch.cuda.amp.GradScaler`
4. Trả về `(avg_loss, all_targets, all_preds)`

**`evaluate_one_epoch()`:**
1. model.eval() + `torch.no_grad()`
2. Forward pass → softmax probabilities
3. Trả về `(avg_loss, targets, preds, paths, probs_tensor)`

### 4.3 Build model

**[src/models/build_model.py](src/models/build_model.py)** — 14 dòng:

```python
import timm

def build_model(model_cfg: dict):
    model = timm.create_model(
        model_cfg['name'],       # "tf_efficientnet_b0", "resnet50", etc.
        pretrained=True,         # load ImageNet weights
        num_classes=7,           # thay đầu ra thành 7 lớp
        drop_rate=0.3,           # dropout trước classifier
    )
    return model
```

`timm` tự động:
- Load pretrained weights từ ImageNet
- Thay `classifier` head (linear layer) thành `Linear(features, 7)`
- Khởi tạo head mới ngẫu nhiên

### 4.4 Loss function

**[src/models/losses.py](src/models/losses.py)** — `build_loss(cfg, class_weights, device)`

Dùng `nn.CrossEntropyLoss(label_smoothing=0.1)` — PyTorch đã tích hợp sẵn.

Label smoothing biến one-hot target `[0,0,1,0,0,0,0]` thành soft target:
```
y_smooth = (1 - ε) × one_hot + ε/C
         = 0.9 × [0,0,1,...] + 0.1/7 × [1,1,1,...]
         = [0.014, 0.014, 0.914, 0.014, 0.014, 0.014, 0.014]
```

### 4.5 Metrics

**[src/utils/metrics.py](src/utils/metrics.py)** → `compute_classification_metrics(y_true, y_pred, class_names, y_prob)`

Tính đồng thời:
- Accuracy, Macro/Weighted Precision/Recall/F1
- Cohen's Kappa, MCC (Matthews Correlation Coefficient)
- Per-class: TP/FP/FN/TN, Sensitivity, Specificity, AUC
- ROC curves (fpr, tpr per class)
- Confusion matrix

### 4.6 Grad-CAM

**[src/utils/grad_cam.py](src/utils/grad_cam.py)**

Class `GradCAM(model, target_layer)`:
- Dùng PyTorch hooks: `register_forward_hook` (lưu activation) + `register_full_backward_hook` (lưu gradient)
- `__call__(input_tensor, target_class=None)`:
  1. Forward pass → lấy predicted class
  2. Backward pass trên score của class đó
  3. `weights = gradients.mean(dim=(H,W))` — global average pooling của gradient
  4. `cam = ReLU(Σ weight_k × activation_k)` — weighted combination
  5. Normalize về [0,1]
- `overlay(image, heatmap, alpha=0.5)` — overlay colormap lên ảnh gốc

**Target layer theo model:**

| Model | Target layer | Kích thước | Lý do |
|-------|-------------|-----------|-------|
| EfficientNet-B0 | `model.blocks[-1][-1]` | (320, 7, 7) | Block MBConv cuối, có depthwise 3×3 conv thực sự |
| ResNet50 | `model.layer4[-1]` | (2048, 7, 7) | Residual block cuối |
| DenseNet121 | `model.features.norm5` | (1024, 7, 7) | BatchNorm sau dense block cuối |
| Swin-Tiny | `model.layers[-1].blocks[-1].norm2` | (49, 768) | LayerNorm sau attention block cuối |

> **Tại sao không dùng `model.bn2` cho EfficientNet?**
> `bn2` là `BatchNormAct2d + SiLU` — SiLU(x) ≈ 0 với x âm, sau BN (mean~0) thì
> ~50% activation = 0 → chỉ 18% pixel có activation → heatmap xanh đồng đều.
> `blocks[-1][-1]` cho 51% pixel active, kết quả giống ResNet/DenseNet.

**[src/utils/gradcam_quant.py](src/utils/gradcam_quant.py)** — metrics định lượng Grad-CAM:

| Metric | Ý nghĩa |
|--------|---------|
| `focus_ratio` | Tỉ lệ activation nằm trong lesion mask (Otsu threshold) — cao = model focus đúng chỗ |
| `mean_activation_in/out` | Cường độ trung bình trong/ngoài lesion mask |
| `entropy` | Shannon entropy của heatmap — thấp = focus sắc nét |
| `peak_distance` | Khoảng cách peak activation đến centroid lesion (normalize theo diagonal) |
| `coverage_75` | Phần trăm diện tích có activation ≥ top-25% — thấp = compact focus |

---

## 5. Pretrain vs Fine-tune — Định nghĩa

### Pretrain (Pre-training)

Huấn luyện model từ đầu (random initialization) trên tập dữ liệu **lớn và tổng quát** nhằm học các đặc trưng chung của dữ liệu (edge, texture, shape...).

**Ví dụ:** Training EfficientNet-B0 trên ImageNet-1K (1.28 triệu ảnh, 1000 lớp) trong hàng tuần trên nhiều GPU.

Kết quả là bộ weights `pretrained_efficientnet_b0.pth` — model đã "biết" nhìn ảnh, dù chưa biết gì về da liễu.

**Trong project này:** Không tự pretrain. Download weights sẵn từ `timm` library (`pretrained=True`).

### Fine-tune (Fine-tuning)

Lấy model **đã pretrained**, thay đổi output head cho phù hợp task mới, rồi tiếp tục train toàn bộ model (hoặc một phần) trên tập dữ liệu nhỏ hơn và chuyên biệt hơn.

**Ví dụ (project này):**
1. Load EfficientNet-B0 với weights ImageNet pretrained
2. Thay `classifier` head: `Linear(1280, 1000)` → `Linear(1280, 7)`
3. Train toàn bộ model trên ISIC 2018 (7,010 ảnh) với lr=3e-4

**Tại sao fine-tune hiệu quả hơn train từ đầu?**

| | Train từ đầu | Fine-tune |
|--|--|--|
| Dữ liệu cần | Rất nhiều (>100K) | Ít (vài nghìn) |
| Thời gian | Rất dài | Nhanh hơn nhiều |
| Kết quả trên data nhỏ | Kém (overfitting) | Tốt hơn nhiều |
| Lý do | Phải học lại từ scratch | Tận dụng features ImageNet |

**Transfer learning trong project này:**
- Các lớp đầu (conv layers) đã học edge/texture từ ImageNet → giữ lại
- Lớp cuối (classifier) học đặc trưng da liễu từ ISIC → cần train
- Toàn bộ network được fine-tune với learning rate nhỏ (3e-4) và warmup để tránh "quên" các features đã học

---

## 6. Pipeline huấn luyện

```
Dataset ISIC 2018
      │
      ▼
skin_metadata.csv
(image_path, label, split)
      │
      ├──► train_df (7,010 samples)
      │         │
      │         ├── WeightedRandomSampler (w_c = N/C/n_c)
      │         │         → mỗi batch ~cân bằng 7 lớp
      │         │
      │         └── train_transforms:
      │               RandomResizedCrop, Flip, Rotation,
      │               ColorJitter, Blur, Erasing, Normalize
      │
      ├──► val_df (1,502 samples) → eval_transforms (chỉ resize + normalize)
      └──► test_df (1,503 samples) → eval_transforms
      
Training Loop (50 epochs, early stopping patience=12):
┌────────────────────────────────────────────────────────┐
│  for batch in train_loader:                            │
│    apply Mixup hoặc CutMix (prob=0.5 mỗi loại)        │
│    forward pass (mixed precision FP16)                 │
│    cross_entropy loss + label_smoothing=0.1            │
│    backward + grad_clip(max_norm=1.0)                  │
│    AdamW.step()                                        │
│    CosineAnnealingLR với linear warmup 3 epochs        │
└────────────────────────────────────────────────────────┘
      │
      ▼
Mỗi epoch: evaluate trên val → track macro_f1
Lưu best.pth khi macro_f1 cải thiện
Early stop nếu 12 epoch không cải thiện
      │
      ▼
Load best.pth → evaluate trên test set
Lưu: test_metrics.json, confusion_matrix.png, roc_curves.png
```

### Chi tiết hyperparameter

| Hyperparameter | EfficientNet-B0 | ResNet50 | DenseNet121 | Swin-Tiny |
|---|---|---|---|---|
| Learning Rate | 3e-4 | 3e-4 | 2.5e-4 | 1e-4 |
| Weight Decay | 1e-3 | 1e-3 | 1e-3 | 5e-2 |
| Warmup epochs | 3 | 3 | 3 | 5 |
| Max epochs | 50 | 50 | 50 | 50 |
| Early stop patience | 12 | 12 | 12 | 12 |
| Batch size | 16 | 16 | 16 | 16 |
| Dropout | 0.30 | 0.30 | 0.25 | 0.30 |
| Mixup α | 0.2 | 0.2 | 0.2 | 0.2 |
| CutMix α | 1.0 | 1.0 | 1.0 | 1.0 |
| Label smoothing ε | 0.1 | 0.1 | 0.1 | 0.1 |

> **Tại sao Swin-Tiny dùng lr=1e-4 và weight_decay=5e-2?**
> Transformer nhạy cảm hơn với learning rate cao → dễ phá vỡ attention weights đã pretrained.
> Weight decay cao hơn vì Swin có nhiều parameters (28M) và dễ overfit hơn trên dataset nhỏ.

---

## 7. Các model và config

### 7.1 EfficientNet-B0

**Config:** [src/configs/efficientnet_b0_optimized.yaml](src/configs/efficientnet_b0_optimized.yaml)
**Checkpoint:** [outputs/efficientnet_b0_v3(main_best)/best.pth](outputs/efficientnet_b0_v3(main_best)/best.pth)

- timm name: `tf_efficientnet_b0`
- Params: **5.3M** (nhỏ nhất)
- Kiến trúc: MBConv blocks với depthwise separable convolution + Squeeze-and-Excitation
- Compound scaling: đồng thời scale depth/width/resolution
- Feature map cuối: (1280, 7, 7) → GAP → Linear(1280, 7)
- Best epoch: **39/50** (early stop sau epoch 50)
- Training time: ~45 phút (CUDA)

### 7.2 ResNet50

**Config:** [src/configs/resnet50_v3.yaml](src/configs/resnet50_v3.yaml)
**Checkpoint:** [outputs/resnet50_v3(main)/best.pth](outputs/resnet50_v3(main)/best.pth)

- timm name: `resnet50`
- Params: **25.6M**
- Kiến trúc: Bottleneck residual blocks (1×1 → 3×3 → 1×1 conv)
- Feature map cuối: (2048, 7, 7) → GAP → Linear(2048, 7)
- Best epoch: **39/50**

### 7.3 DenseNet121

**Config:** [src/configs/densenet121_v3.yaml](src/configs/densenet121_v3.yaml)
**Checkpoint:** [outputs/densenet121_v3(main)/best.pth](outputs/densenet121_v3(main)/best.pth)

- timm name: `densenet121`
- Params: **8.0M**
- Kiến trúc: Dense blocks — mỗi layer nhận concatenation của tất cả layer trước
- Feature map cuối: (1024, 7, 7) → GAP → Linear(1024, 7)
- Best epoch: **49/50** (gần max epochs)
- Cần training lâu hơn vì dense connectivity tạo optimization landscape phức tạp hơn

### 7.4 Swin-Tiny

**Config:** [src/configs/swin_tiny_v3.yaml](src/configs/swin_tiny_v3.yaml)
**Checkpoint:** [outputs/swin_tiny_v3(main)/best.pth](outputs/swin_tiny_v3(main)/best.pth)

- timm name: `swin_tiny_patch4_window7_224`
- Params: **28.0M** (lớn nhất)
- Kiến trúc: Shifted Window Self-Attention, phân cấp 4 stages
- Input: chia thành 4×4 patches → 56×56 tokens → 28×28 → 14×14 → 7×7
- Feature cuối: 7×7 tokens × 768 dim → global avg → Linear(768, 7)
- Best epoch: **13/50** (early stop tại epoch 25) → overfit nhanh nhất

---

## 8. Ensemble

**Script:** [src/ensemble.py](src/ensemble.py)
**Chạy:** `python -m src.ensemble`
**Output:** [outputs/ensemble/](outputs/ensemble/)

### Cơ chế Soft Voting

```python
# 4 model mỗi model cho 1 probability vector (N × 7)
probs_eb0   = model_eb0(test_set)   # shape: (1503, 7)
probs_rn50  = model_rn50(test_set)  # shape: (1503, 7)
probs_dn121 = model_dn121(test_set) # shape: (1503, 7)
probs_swin  = model_swin(test_set)  # shape: (1503, 7)

# Stack và average
stacked = np.stack([probs_eb0, probs_rn50, probs_dn121, probs_swin])  # (4, 1503, 7)
avg_probs = stacked.mean(axis=0)  # (1503, 7)

# Dự đoán cuối cùng
y_pred = avg_probs.argmax(axis=1)  # (1503,)
```

### Tại sao Soft Voting hiệu quả?

1. **Mỗi model sai ở những chỗ khác nhau** — CNN focus local features, Swin focus global context → lỗi không tương quan cao
2. **Trung bình xác suất giảm variance** — ảnh nào 3/4 model đồng ý thì gần như chắc chắn đúng
3. **Không cần train thêm** — chỉ cần inference 4 lần, không cần validation set cho ensemble weights

### Files output

```
outputs/ensemble/
├── ensemble_metrics.json         # Toàn bộ metrics JSON
├── ensemble_predictions.csv      # Per-image: y_true, y_pred, prob_akiec, ..., prob_vasc
├── confusion_matrix_normalized.png
├── roc_curves.png
├── per_class_f1.png
└── sensitivity_specificity.png
```

---

## 9. Ablation Study

**Mục đích:** Xác định đóng góp của từng thành phần trong pipeline.

**Phương pháp:** Xóa 1 thành phần mỗi lần, giữ nguyên tất cả còn lại. Base model là EfficientNet-B0.

### Các config ablation

| Experiment | Config | Output dir |
|---|---|---|
| Full pipeline (baseline) | [efficientnet_b0_optimized.yaml](src/configs/efficientnet_b0_optimized.yaml) | [outputs/efficientnet_b0_v3(main_best)/](outputs/efficientnet_b0_v3(main_best)/) |
| Xóa Mixup/CutMix | [ablations/eb0_no_mixup.yaml](src/configs/ablations/eb0_no_mixup.yaml) | [outputs/ablation_no_mixup/](outputs/ablation_no_mixup/) |
| Xóa Weighted Sampler | [ablations/eb0_no_sampler.yaml](src/configs/ablations/eb0_no_sampler.yaml) | [outputs/ablation_no_sampler/](outputs/ablation_no_sampler/) |
| Xóa Label Smoothing | [ablations/eb0_no_smooth.yaml](src/configs/ablations/eb0_no_smooth.yaml) | [outputs/ablation_no_smooth/](outputs/ablation_no_smooth/) |
| Xóa Advanced Aug | [ablations/eb0_no_adv_aug.yaml](src/configs/ablations/eb0_no_adv_aug.yaml) | [outputs/ablation_no_adv_aug/](outputs/ablation_no_adv_aug/) |

### Kết quả ablation — EfficientNet-B0

| Configuration | Accuracy | Macro F1 | AUC | Kappa | ΔMacro F1 |
|---|---|---|---|---|---|
| **Full pipeline** | **0.8836** | **0.8311** | 0.9522 | **0.7710** | — |
| − Mixup / CutMix | 0.8589 | 0.7860 | 0.9486 | 0.7383 | **−0.0451** |
| − Label Smoothing | 0.8776 | 0.8000 | **0.9674** | 0.7568 | −0.0311 |
| − Weighted Sampler | 0.8762 | 0.8041 | 0.9521 | 0.7487 | −0.0270 |
| − Advanced Augmentation | **0.8896** | 0.8204 | 0.9664 | 0.7772 | −0.0107 |

### Phân tích từng thành phần

**Mixup / CutMix (−4.51% Macro F1) — quan trọng nhất**

Xóa đi gây drop lớn nhất. Lý do: với chỉ 81 training samples cho DF và 99 cho VASC, model không có Mixup/CutMix sẽ memorize các sample hiếm trong vài epoch đầu. Các ảnh mixed-label ngăn model gán probability=1.0 cho bất kỳ sample cụ thể nào, buộc model phải học features tổng quát hơn.

**Label Smoothing (−3.11% Macro F1) — nhưng tăng AUC!**

Nghịch lý: xóa label smoothing làm **tăng AUC** từ 0.952 → 0.967 nhưng **giảm Macro F1** 3.11%.

Giải thích: Label smoothing làm mờ ranh giới quyết định giữa các lớp gần nhau (MEL/NV/BKL), giúp classification chính xác hơn nhưng làm mờ độ phân tách probability → AUC thấp hơn. Không có label smoothing → model tự tin hơn → AUC cao hơn nhưng ranh giới quyết định cứng hơn → F1 thấp hơn ở các lớp khó.

**Weighted Sampler (−2.70% Macro F1)**

Không có sampler → model học chủ yếu từ NV (4,693 samples). Recall của DF, VASC, AKIEC sụt giảm rõ rệt. Accuracy giảm ít (−0.7%) vì NV vẫn được dự đoán đúng.

**Advanced Augmentation (−1.07% Macro F1) — nhưng tăng Accuracy!**

Nghịch lý khác: xóa aug mạnh làm **tăng accuracy** (88.96% > 88.36%) nhưng **giảm F1**.

Giải thích: Aug mạnh (RandomResizedCrop, Blur, Erasing) làm training hard hơn với NV (majority class) → accuracy NV giảm nhẹ → overall accuracy giảm. Nhưng model được buộc học features robust hơn → phân loại minority classes tốt hơn → F1 tăng. Đây là trade-off điển hình của augmentation mạnh trên data mất cân bằng.

---

## 10. Kết quả và chỉ số

### 10.1 Định nghĩa các chỉ số

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| **Accuracy** | (TP_all) / N | Tỉ lệ dự đoán đúng tổng thể — misleading khi imbalanced |
| **Macro F1** | mean(F1 per class) | F1 trung bình không trọng số — bình đẳng với mọi lớp |
| **Weighted F1** | Σ(F1_c × n_c) / N | F1 có trọng số theo số lượng → ưu tiên lớp lớn |
| **AUC (macro)** | mean(AUC per class, OvR) | Khả năng phân biệt xác suất — không phụ thuộc threshold |
| **Cohen's κ** | (P_o − P_e) / (1 − P_e) | Agreement so với random chance — κ>0.6 = tốt |
| **MCC** | TP×TN−FP×FN / √(...) | Cân bằng nhất, dùng toàn bộ confusion matrix |
| **Sensitivity** | TP / (TP + FN) | Recall — tỉ lệ phát hiện đúng positives |
| **Specificity** | TN / (TN + FP) | Tỉ lệ xác định đúng negatives |

> **Metric chính để so sánh model:** **Macro F1** — vì không bị bias bởi NV (class đông nhất).

### 10.2 Kết quả từng model

| Model | Params | Accuracy | Macro F1 | AUC | MCC | Kappa | Best epoch |
|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | 5.3M | **0.8836** | **0.8311** | 0.9522 | **0.7721** | **0.7710** | 39 |
| ResNet50 | 25.6M | 0.8603 | 0.7902 | 0.9565 | 0.7310 | 0.7309 | 39 |
| DenseNet121 | 8.0M | 0.8543 | 0.7967 | 0.9587 | 0.7209 | 0.7206 | 49 |
| Swin-Tiny | 28.0M | 0.8490 | 0.7855 | **0.9600** | 0.7157 | 0.7150 | 13 |
| **Ensemble** | — | **0.8949** | **0.8446** | **0.9802** | **0.7956** | **0.7949** | — |

### 10.3 Per-class F1 — EfficientNet-B0 vs Ensemble

| Class | EB0 F1 | Ens F1 | ΔF1 | EB0 AUC | Ens AUC | ΔAUC |
|---|---|---|---|---|---|---|
| AKIEC | 0.717 | 0.710 | −0.007 | 0.921 | 0.972 | **+0.051** |
| BCC | 0.865 | 0.825 | −0.040 | 0.984 | 0.996 | +0.012 |
| BKL | 0.758 | 0.796 | **+0.038** | 0.938 | 0.968 | +0.030 |
| DF | 0.938 | 0.938 | 0.000 | 0.963 | 0.996 | +0.033 |
| MEL | 0.695 | 0.744 | **+0.049** | 0.913 | 0.958 | **+0.045** |
| NV | 0.940 | 0.946 | +0.006 | 0.946 | 0.972 | +0.026 |
| VASC | 0.905 | 0.955 | **+0.050** | 1.000 | 1.000 | 0.000 |

**Nhận xét quan trọng:**
- **MEL** là lớp khó nhất — F1 thấp nhất cả đơn lẻ (0.695) lẫn ensemble (0.744). Do MEL overlap morphological với NV và BKL.
- **DF** F1 cao nhất (0.938) dù training samples ít nhất (81) — DF có đặc trưng visual rõ ràng.
- **AUC ensemble tăng ở mọi lớp** — consistent improvement, không có class nào bị drop AUC.
- **BCC, AKIEC F1 giảm nhẹ khi ensemble** — trade-off của soft voting: làm mịn predictions → một số confident đúng bị "pha loãng".

### 10.4 Accuracy vs AUC — nghịch lý quan trọng

```
Hard classification (Accuracy/F1):  EfficientNet-B0 > ResNet50 > DenseNet121 > Swin-Tiny
Probability ranking (AUC):          Swin-Tiny > DenseNet121 > ResNet50 > EfficientNet-B0
```

**Giải thích:** Swin-Tiny với self-attention tạo probability distributions được calibrate tốt hơn dù hard decision kém hơn. Trên dataset 7,010 ảnh, Transformer chưa đủ data để cạnh tranh CNN về F1 nhưng thắng về AUC.

**Implication thực tế:**
- Dùng EfficientNet-B0 nếu cần binary decision (flag/no-flag)
- Dùng Swin-Tiny hoặc Ensemble nếu cần risk ranking (triage)

### 10.5 Output files của mỗi training run

```
outputs/{model_name}/
├── best.pth                    # Model weights tốt nhất (theo val macro_f1)
├── experiment_summary.json     # Toàn bộ config + final test results
├── test_metrics.json           # Chi tiết metrics trên test set
├── test_predictions.csv        # Per-image predictions + probabilities
├── history.json                # Loss/accuracy/F1 theo từng epoch
├── epoch_log.csv               # Detailed epoch-by-epoch log
├── confusion_matrix_normalized.png
├── confusion_matrix_raw.png
├── test_roc_curves.png
├── test_per_class_f1.png
├── test_sensitivity_specificity.png
├── training_history_detailed.png
└── class_distribution.png
```

---

## Quick Reference — Chạy lại từng thứ

```bash
# Train một model
python -m src.train --config src/configs/efficientnet_b0_optimized.yaml

# Train tất cả ablations
python -m src.train --config src/configs/ablations/eb0_no_mixup.yaml
python -m src.train --config src/configs/ablations/eb0_no_sampler.yaml
python -m src.train --config src/configs/ablations/eb0_no_smooth.yaml
python -m src.train --config src/configs/ablations/eb0_no_adv_aug.yaml

# Ensemble
python -m src.ensemble

# Grad-CAM visualization
python -m src.grad_cam_analysis --n-samples 3

# Grad-CAM quantitative metrics
python -m src.gradcam_quant_runner --n-per-class 80

# Streamlit demo
streamlit run app_v2.py
```
