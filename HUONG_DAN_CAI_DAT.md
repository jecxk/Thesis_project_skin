# Hướng dẫn Cài đặt và Sử dụng

**Đề tài:** Đánh giá so sánh các mô hình học sâu cho phân loại tổn thương da trên
ảnh dermoscopy — tập ISIC 2018 / HAM10000, kết hợp ensemble và Grad-CAM.

**Tác giả:** Nguyễn Trọng Bách (23BI14057), Khoa Công nghệ Thông tin và Truyền thông,
Trường Đại học Khoa học và Công nghệ Hà Nội (USTH).

---

## 1. Yêu cầu hệ thống

| Thành phần | Mức tối thiểu | Đề xuất |
|------------|---------------|---------|
| Hệ điều hành | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 22.04 |
| Python | 3.9 | **3.10** hoặc 3.11 |
| RAM | 8 GB | 16 GB |
| GPU (huấn luyện) | Không bắt buộc | NVIDIA GPU ≥ 8 GB, CUDA 11.8+ |
| GPU (chạy demo) | Không bắt buộc — chạy được trên CPU | — |
| Dung lượng ổ đĩa | 5 GB (mã + dữ liệu ISIC) | 10 GB |

Với ảnh dermoscopy 224×224 và ensemble 4 backbone, một lần dự đoán trên CPU mất
khoảng 2–4 giây; trên GPU thấp hơn 0,5 giây.

## 2. Cài đặt môi trường

### 2.1. Giải nén và di chuyển vào thư mục dự án

```powershell
# Windows PowerShell
Expand-Archive skin_thesis_source.zip -DestinationPath .
cd skin_thesis_source
```

```bash
# Linux / macOS
unzip skin_thesis_source.zip -d skin_thesis_source
cd skin_thesis_source
```

### 2.2. Tạo môi trường ảo Python

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3. Cài đặt thư viện phụ thuộc

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Nếu có GPU NVIDIA, cài PyTorch bản CUDA (thay 11.8 bằng phiên bản CUDA driver của bạn):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Bản CPU-only (mặc định trong `requirements.txt`) đủ để chạy ứng dụng demo.

## 3. Chuẩn bị dữ liệu (chỉ cần khi huấn luyện lại)

Tập ISIC 2018 Task 3 (HAM10000) tải công khai tại
<https://challenge.isic-archive.com/data/#2018>. Sau khi tải:

1. Tạo thư mục `data/isic2018/images/` và chép toàn bộ ảnh vào đó.
2. File nhãn `data/metadata/skin_metadata.csv` đã được đóng gói sẵn trong zip
   với cột `image_path` là đường dẫn tương đối tính từ gốc dự án.

Nếu chỉ dùng ứng dụng demo, có thể **bỏ qua bước này** — dùng luôn ảnh mẫu trong
thư mục `demo_images/`.

## 4. Chạy ứng dụng demo (khuyến nghị dùng thử trước)

### 4.1. Tải các checkpoint đã huấn luyện

Các file trọng số `best.pth` (~17 MB mỗi backbone) không được đóng gói trong zip
do giới hạn 30 MB. Tải riêng theo hướng dẫn ở phần **Liên hệ** cuối tài liệu, rồi
đặt vào các thư mục sau:

```
outputs/efficientnet_b0_v3(main_best)/best.pth
outputs/resnet50_v3(main)/best.pth
outputs/densenet121_v3(main)/best.pth
outputs/swin_tiny_v3(main)/best.pth
```

### 4.2. Khởi chạy Streamlit

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại địa chỉ hiển thị trên terminal (mặc định
`http://localhost:8501`). Giao diện cho phép:

- Tải lên một ảnh dermoscopy (JPG/PNG) hoặc chọn ảnh mẫu trong `demo_images/`.
- Xem lớp dự đoán cùng biểu đồ xác suất 7 lớp.
- Xem bản đồ Grad-CAM ứng với từng backbone.

## 5. Huấn luyện lại mô hình

```bash
# Huấn luyện từng backbone
python -m src.train --config src/configs/efficientnet_b0_optimized.yaml
python -m src.train --config src/configs/resnet50_v3.yaml
python -m src.train --config src/configs/densenet121_v3.yaml
python -m src.train --config src/configs/swin_tiny_v3.yaml

# Đánh giá một checkpoint
python -m src.evaluate \
    --config src/configs/efficientnet_b0_optimized.yaml \
    --checkpoint "outputs/efficientnet_b0_v3(main_best)/best.pth"

# Chạy ensemble bốn mô hình
python -m src.ensemble --output-dir outputs/ensemble

# Tổng hợp kết quả ablation
python -m src.summarize_ablation
```

Thời gian huấn luyện tham khảo trên NVIDIA RTX 3060 12 GB:

| Backbone | Thời gian / 50 epoch |
|----------|----------------------|
| EfficientNet-B0 | ~45 phút |
| DenseNet121 | ~55 phút |
| ResNet50 | ~60 phút |
| Swin-Tiny | ~75 phút |

## 6. Phân tích bổ sung

```bash
# Grad-CAM định tính + định lượng
python -m src.grad_cam_analysis
python -m src.gradcam_quant_runner

# Hiệu chuẩn xác suất (ECE, Brier, temperature scaling)
python -m src.calibration_analysis

# Phân tích lỗi (confusion clusters, hard cases)
python -m src.error_analysis

# Tối ưu ngưỡng quyết định theo lớp (Youden's J)
python -m src.threshold_tuning

# Late-fusion metadata bệnh nhân (tuổi, giới, vị trí tổn thương)
python -m scripts.run_meta_fusion

# Multi-seed reproducibility
python -m scripts.run_multiseed

# Đánh giá tổng quát ngoài phân phối trên PAD-UFES-20
python -m scripts.evaluate_ood_pad_ufes
```

## 7. Cấu trúc thư mục

```
skin_thesis_source/
├── src/                     # Toàn bộ mã nguồn Python
│   ├── configs/             # File YAML cấu hình huấn luyện
│   ├── data/                # Dataset, augmentation, sampler
│   ├── engine/              # Vòng huấn luyện và đánh giá
│   ├── models/              # Nhà máy backbone và hàm mất mát
│   ├── utils/               # Metric, Grad-CAM, hiệu chuẩn
│   ├── train.py             # Điểm vào huấn luyện
│   ├── evaluate.py          # Đánh giá một checkpoint
│   ├── ensemble.py          # Ensemble soft-voting
│   └── ...
├── scripts/                 # Script tiện ích (multi-seed, OOD, ...)
├── data/metadata/           # File CSV nhãn và metadata bệnh nhân
├── demo_images/             # 14 ảnh dermoscopy mẫu
├── app.py                   # Ứng dụng demo Streamlit
├── app_v2.py                # Phiên bản Streamlit thay thế (dark theme)
├── requirements.txt         # Danh sách thư viện phụ thuộc
├── README.md                # Mô tả dự án (tiếng Anh)
└── HUONG_DAN_CAI_DAT.md     # Tài liệu bạn đang đọc
```

## 8. Xử lý lỗi thường gặp

**`ModuleNotFoundError: No module named 'torch'`** — chưa kích hoạt venv, hoặc
chưa chạy `pip install -r requirements.txt`.

**`CUDA out of memory`** — giảm `batch_size` trong file YAML tương ứng, hoặc thêm
cờ `--device cpu`.

**Streamlit báo lỗi `best.pth not found`** — chưa tải checkpoint, xem mục 4.1.

**Ảnh Grad-CAM đen hoàn toàn** — kiểm tra lại `target_layer` trong
`src/utils/grad_cam.py`. Với Swin-Tiny phải trỏ vào `norm` cuối cùng, không phải
`stages[-1]`.

## 9. Liên hệ

- **Tác giả:** Nguyễn Trọng Bách — `nguyentrongdu@tnu.edu.vn`
- **Giảng viên hướng dẫn ngoài:** TS. Vũ Trọng Sinh
- **Giảng viên hướng dẫn trong:** TS. Nghiêm Thị Phương

Vì lý do dung lượng 30 MB, các file checkpoint đã huấn luyện (`best.pth`), toàn bộ
thư mục `outputs/` (~1,9 GB) và tập ảnh gốc ISIC 2018 không được đóng gói. Vui
lòng liên hệ tác giả để nhận link tải riêng.
