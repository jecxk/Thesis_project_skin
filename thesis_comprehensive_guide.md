# 📚 TỔNG HỢP KIẾN THỨC VÀ HƯỚNG DẪN BẢO VỆ LUẬN VĂN
## Phân Loại Tổn Thương Da Bằng Deep Learning

Tài liệu này tổng hợp toàn bộ các kết quả, đánh giá, kiến thức cốt lõi và hướng dẫn bảo vệ đồ án tốt nghiệp của bạn.

---

# PHẦN 1: ĐÁNH GIÁ TỔNG THỂ DỰ ÁN

## 1.1 Scorecard — Trước vs Sau sửa đổi (Cập nhật 27/04/2026)

| Tiêu chí | Trước | Sau | Thay đổi |
|---------|:-----:|:---:|:--------:|
| Kỹ thuật ML | 9/10 | 9/10 | — (đã tốt) |
| Chỉ số kết quả | 8.5/10 | 8.5/10 | — (data không đổi) |
| Cấu trúc report | 7.5/10 | **9.5/10** | +2.0 |
| References | 5/10 | **9/10** | +4.0 ⬆️ |
| Consistency | 6/10 | **9.5/10** | +3.5 ⬆️ |
| Độ sâu analysis | 8/10 | **9/10** | +1.0 |
| **TỔNG** | **7.3/10** | **9.1/10** | **+1.8** ⬆️ |

## 1.2 Những điểm chính đã hoàn thiện trong luận văn
- **References:** Đã tăng từ 13 lên 30 tài liệu tham khảo chất lượng cao.
- **Nội dung mới:** Bổ sung các phần về Ensemble Methods, Modern Hybrid Perspectives (ConvNeXt, DINO).
- **Phân tích sâu:** Giải thích rõ hiện tượng Test loss do label smoothing, thêm cảnh báo về sample size của DF/VASC.
- **Tính năng mới:** Đưa Web app vào làm Contribution #6, thêm các mục hạn chế và hướng phát triển tương lai.

---

# PHẦN 2: KẾT QUẢ ĐÀO TẠO MÔ HÌNH (EFFICIENTNET-B0 v3)

Model EfficientNet-B0 v3 (Single Model) đã đạt **Accuracy 88.36%** và **Macro F1 0.831** trên bộ dữ liệu ISIC 2018 mà không bị overfitting. Khi kết hợp Ensemble 4 model, kết quả nâng lên **Accuracy 89.49%**.

## 2.1 Per-Class F1 (Kết quả chi tiết từng lớp)

| Class | F1 | Precision | Recall | AUC | Support | Đánh giá |
|---|---|---|---|---|---|---|
| akiec | 0.717 | 0.767 | 0.673 | 0.921 | 49 | ⭐ Tốt (class nhỏ) |
| **bcc** | **0.865** | 0.859 | 0.870 | **0.984** | 77 | 🏆 Xuất sắc |
| bkl | 0.758 | 0.823 | 0.703 | 0.938 | 165 | ⭐ Tốt |
| **df** | **0.938** | **1.000** | 0.882 | 0.963 | 17 | 🏆 **Precision hoàn hảo!** |
| mel | 0.695 | 0.708 | 0.683 | 0.913 | 167 | ⭐ Khá (lớp khó nhất) |
| **nv** | **0.940** | 0.922 | 0.958 | 0.946 | 1006 | 🏆 Xuất sắc |
| **vasc** | **0.905** | 0.950 | 0.864 | **1.000** | 22 | 🏆 **AUC hoàn hảo!** |

**Điểm nổi bật:**
- **DF**: Precision = 1.000 (0% false positive) — rất hiếm gặp!
- **Vasc**: AUC = 1.000 (phân biệt hoàn hảo).
- **NV (lớp đa số)**: F1 = 0.94 — không hy sinh class lớn để cứu class nhỏ.

---

# PHẦN 3: KIẾN THỨC KỸ THUẬT - DATA & MODELS

## 3.1 Dataset — ISIC 2018 (HAM10000)
- **10.015 ảnh**, 7 loại bệnh, mất cân bằng cực lớn (NV chiếm ~67%, DF chỉ ~1.1%).
- Mất cân bằng dữ liệu là thách thức lớn nhất của bài toán này.

## 3.2 Data Pipeline & Augmentation
- **Augmentation cơ bản:** RandomResizedCrop, Flip, Rotation, ColorJitter, GaussianBlur, RandomErasing.
- **Mixup & CutMix:** (Quan trọng nhất) Trộn ảnh và nhãn theo tỷ lệ, hoặc dán 1 mảng ảnh này sang ảnh kia. Giúp model không overfitting và học biên giới quyết định mượt mà hơn.
- **Weighted Random Sampler:** Xử lý mất cân bằng bằng cách tăng xác suất chọn các ảnh của class thiểu số (như DF, VASC) để model thấy chúng thường xuyên hơn trong mỗi epoch.

## 3.3 Model Architectures (Ensemble 4 Models)
- **EfficientNet-B0 (5.3M params):** Dùng Compound Scaling (cân bằng sâu, rộng, độ phân giải). Rất nhẹ nhưng hiệu quả cao nhất (Best individual). Dùng MBConv và Squeeze-and-Excitation.
- **ResNet50 (25.6M params):** Dùng Residual Connections (skip connection) để gradient "chảy thẳng" qua nhiều lớp, chống vanishing gradient.
- **DenseNet121 (8.0M params):** Dùng Dense Connections, mỗi lớp nhận input từ TẤT CẢ các lớp trước đó. Tái sử dụng feature map cực tốt.
- **Swin-Tiny (28M params):** Vision Transformer, dùng Shifted Window Self-Attention. Nhìn toàn cục bức ảnh (Global context) thay vì cục bộ như CNN. AUC cao nhất nhưng thiếu data để đẩy Acc lên đỉnh.

---

# PHẦN 4: KIẾN THỨC KỸ THUẬT - TRAINING & METRICS

## 4.1 Training Pipeline
- **Loss:** CrossEntropy kết hợp Label Smoothing (ε = 0.1). Label Smoothing giảm nhãn 100% xuống 90% để model bớt quá tự tin, giúp cải thiện calibration.
- **Optimizer:** AdamW kết hợp Cosine Annealing LR + Warmup. Warmup 3 epochs đầu để model không bị "sốc" phá hỏng pre-trained weights, sau đó giảm LR theo đường hình sin (cosine).
- **AMP & Grad Clipping:** Dùng mixed precision (FP16/FP32) để train nhanh hơn, kết hợp Gradient Clipping (1.0) giữ training ổn định, tránh bùng nổ gradient.

## 4.2 Metrics Đánh Giá
- **Macro F1 (0.845 - Ensemble):** Là trung bình F1 của 7 class, không có trọng số theo kích thước class. Rất quan trọng vì ép model phải dự đoán tốt cả các lớp thiểu số.
- **Accuracy (89.49% - Ensemble):** Cải thiện mạnh so với các Single model.
- **AUC-ROC (0.980 - Ensemble):** Rất sát 1.0, cho thấy khả năng xếp hạng xác suất dự đoán (ranking) của mô hình là xuất sắc, không phụ thuộc ngưỡng threshold.

## 4.3 Grad-CAM (Giải Thích Mô Hình)
- Gradient-weighted Class Activation Mapping: Trực quan hóa xem mô hình nhìn vào đâu để đoán.
- Kết quả cho thấy mô hình nhìn đúng vào vùng tổn thương và viền tổn thương (border irregularity, asymmetry), chuẩn theo tiêu chuẩn y khoa ABCDE, chứ không nhìn vào thước đo hay tóc.
- Đã khắc phục lỗi border suppression (gradient giả ở viền do padding bất đối xứng) của EfficientNet.

## 4.4 Ablation Study (Chứng Minh Tính Hiệu Quả)
Khi tắt dần các kỹ thuật:
- Bỏ Mixup/CutMix: F1 giảm mạnh nhất (-4.5%).
- Bỏ Label Smoothing: F1 giảm 3.1%. Thú vị là AUC tăng nhưng F1 giảm (do model bớt chắc chắn).
- Bỏ Weighted Sampler: F1 giảm 2.7%.

---

# PHẦN 5: HƯỚNG DẪN THUYẾT TRÌNH VÀ Q&A

## 5.1 Pitch 30 giây mở đầu
> *"Đồ án của em nghiên cứu và so sánh 4 kiến trúc deep learning hiện đại — 3 CNN và 1 Vision Transformer — cho bài toán phân loại 7 loại tổn thương da từ ảnh dermoscopy, sử dụng bộ dữ liệu ISIC 2018. Thách thức chính là mất cân bằng dữ liệu nghiêm trọng với tỷ lệ 58:1. Em đã xây dựng một training pipeline tối ưu kết hợp Mixup, CutMix, Label Smoothing và Weighted Sampling, đạt 89.49% accuracy và AUC 0.980 với ensemble 4 mô hình. Em cũng thực hiện Grad-CAM để xác nhận model tập trung vào vùng tổn thương có ý nghĩa lâm sàng, và ablation study để chứng minh tính cần thiết của từng component."*

## 5.2 Số liệu quan trọng cần thuộc lòng
- **Số ảnh:** 10.015 ảnh (7 classes).
- **Tỷ lệ mất cân bằng:** 58:1 (giữa lớp nhiều nhất NV và lớp ít nhất DF).
- **Ensemble Accuracy:** 89.49% (vượt qua baseline ISIC challenge 85-88%).
- **Ensemble AUC:** 0.980.
- **Mixup tác động:** Giảm 4.5% F1 nếu bỏ.

## 5.3 Câu hỏi phản biện thường gặp & Cách trả lời

**Q1: Tại sao không dùng Focal Loss mà dùng Cross Entropy + Label Smoothing?**
> *Trọng tâm mất cân bằng đã được xử lý ở tầng dữ liệu (Weighted Sampler + Mixup/CutMix). Label smoothing giải quyết vấn đề model bị quá tự tin (overconfidence), điều mà Focal Loss không làm tốt bằng. Nó cũng giúp cải thiện calibration.*

**Q2: Tại sao Swin-Tiny lại có Accuracy thấp hơn các mô hình CNN? Transformer có thực sự tốt?**
> *Swin-Tiny có Accuracy thấp hơn do số lượng ảnh (10.000) hơi ít so với "cơn khát" dữ liệu của Transformer. Tuy nhiên, AUC của Swin lại cao nhất (0.960), chứng tỏ khả năng xếp hạng xác suất rất tốt nhờ góc nhìn toàn cục (Global Context). Khi kết hợp Swin vào Ensemble, F1 của Melanoma (class cực khó) tăng từ 0.695 lên 0.744.*

**Q3: Tại sao lại chọn Batch Size = 16?**
> *Batch size 16 được chọn để đảm bảo tất cả 4 mô hình, bao gồm cả Swin-Tiny (28M params), đều có thể chạy được trên cùng cấu hình GPU mà không bị Out-Of-Memory (OOM). Giữ chung cấu hình giúp việc so sánh công bằng.*

**Q4: Grad-CAM chỉ là trực quan bằng mắt, có đáng tin không?**
> *Em có cung cấp thêm module định lượng Grad-CAM (quantitative Grad-CAM) đo mức độ hội tụ, entropy. Hơn nữa, việc heatmap của 4 kiến trúc khác nhau đều hội tụ về vùng rìa tổn thương và độ bất đối xứng rất khớp với chuẩn ABCDE của bác sĩ da liễu.*

**Q5: Accuracy cao nhưng Melanoma (MEL) F1 chỉ đạt 0.744. Mô hình này có áp dụng thực tế được không?**
> *(Thừa nhận hạn chế) Mô hình này thiên về hỗ trợ sàng lọc (Triage) hơn là chẩn đoán cuối cùng. AUC cho Melanoma là 0.958, rất cao, chứng tỏ mô hình có khả năng phân biệt, chỉ là điểm ngưỡng cắt (Threshold) cần được điều chỉnh (hạ xuống) trong thực tế y tế để tăng Recall/Sensitivity, tránh bỏ sót bệnh nhân.*

---
> **Chúc bạn bảo vệ luận văn thành công! Mọi thứ đã được chuẩn bị rất kỹ lưỡng và chỉn chu.**
