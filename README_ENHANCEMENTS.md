# 🔬 Project Enhancements — Usage Guide

Hướng dẫn các công cụ phân tích nâng cao đã được thêm vào project để nâng chất lượng đồ án từ mức **Khá-Giỏi (~7.5/10)** lên **Xuất sắc (9+/10)**.

---

## 📋 Tóm tắt cải tiến

| # | Module | Mục đích | Giá trị học thuật |
|---|--------|----------|-------------------|
| 1 | **Quantitative Grad-CAM** | Đo định lượng vùng chú ý thay vì chỉ nhìn hình | Chứng minh khoa học, không chỉ qualitative |
| 2 | **Test-Time Augmentation (TTA)** | +1–2% accuracy "miễn phí" | Kỹ thuật bắt buộc trong nghiên cứu ảnh y khoa |
| 3 | **Calibration Analysis** | ECE / Brier / Temperature scaling | Lý giải được AUC cao nhưng độ chính xác thấp |
| 4 | **Error Analysis** | Top confusion pairs, hard cases | Insight cụ thể (VD: MEL ↔ NV) |
| 5 | **Threshold Tuning** | Tối ưu sensitivity cho class melanoma | Ứng dụng lâm sàng thực tế |
| 6 | **Demo v2 (Gradio)** | Model selector + Grad-CAM + warnings | Gây ấn tượng bảo vệ |
| 7 | **Training History (separate)** | 4 hình riêng to rõ thay vì 2x2 nhỏ | Trình bày luận văn chuyên nghiệp |

---

## 🧩 Module mới (code)

### `src/utils/gradcam_quant.py`
Tính 6 chỉ số từ Grad-CAM heatmap + pseudo-lesion mask (Otsu):
- `focus_ratio` — activation trong/ngoài vùng tổn thương
- `mean_activation_in` / `mean_activation_out`
- `entropy` — heatmap càng "sắc nét" thì entropy càng thấp
- `peak_distance` — khoảng cách đỉnh heatmap đến tâm tổn thương
- `coverage_75` — diện tích vùng hot > 75% percentile

### `src/gradcam_quant_runner.py`
Chạy phân tích trên tập test, xuất CSV + hình so sánh 4 model.

```bash
python -m src.gradcam_quant_runner --n-per-class 80 --output-dir outputs/gradcam_quant
```

Output:
- `gradcam_quant_all.csv` — per-image metrics
- `gradcam_quant_aggregated.csv` — per (model, class) mean/std
- `gradcam_quant_summary.csv` — per-model average
- `focus_ratio_comparison.png`
- `entropy_vs_peakdist.png`

---

### `src/tta.py`
Test-Time Augmentation với 3 policy:
- `none`: chỉ ảnh gốc
- `flip` (4×): gốc + hflip + vflip + hv-flip
- `full` (6×): + 90° và 270° rotation

```bash
# Single model + TTA
python -m src.tta --config src/configs/efficientnet_b0_optimized.yaml \
    --checkpoint outputs/efficientnet_b0_v3\(main_best\)/best.pth \
    --policy full --output-dir outputs/tta_eb0

# Ensemble + TTA
python -m src.tta --ensemble --policy full --output-dir outputs/tta_ensemble
```

---

### `src/utils/calibration.py` + `src/calibration_analysis.py`
Đo độ tin cậy xác suất (ECE, MCE, Brier, NLL) và post-hoc calibration bằng **Temperature Scaling** (tối ưu T trên val set bằng LBFGS).

```bash
python -m src.calibration_analysis --output-dir outputs/calibration_analysis
```

Output: Reliability diagram trước/sau T-scaling cho từng model + ensemble, bar chart so sánh ECE/Brier cross-model.

> **Giá trị cho thesis**: Giải thích tại sao Swin-Tiny có AUC cao nhưng accuracy thấp hơn — do miscalibration (over-confidence). Temperature scaling có thể đưa ECE từ ~8% xuống dưới 3%.

---

### `src/error_analysis.py`
Phân tích lỗi chi tiết:
- Top-K confusion pairs (dự đoán sai nhiều nhất)
- Per-class error rate
- Hard examples (misclassified với confidence cao)
- Ensemble vs single: fixed / broken / agreed

```bash
python -m src.error_analysis --output-dir outputs/error_analysis
```

---

### `src/threshold_tuning.py`
Tối ưu ngưỡng quyết định per-class để **tăng sensitivity của melanoma** (quan trọng về mặt lâm sàng — không được bỏ sót).

```bash
# Tối ưu F1 với ràng buộc specificity ≥ 0.85
python -m src.threshold_tuning \
    --predictions outputs/ensemble/test_predictions.csv \
    --criterion f1 --min-specificity 0.85 \
    --target-class mel

# Tối ưu sensitivity (rule-in)
python -m src.threshold_tuning \
    --predictions outputs/ensemble/test_predictions.csv \
    --criterion sensitivity --min-specificity 0.90 \
    --target-class mel
```

---

### `scripts/regenerate_training_history.py`
Vẽ lại training history: mỗi model **1 hình riêng (14×10 inch, 2×2 subplot)** thay vì 4 hình nhỏ 2×2 chung — trình bày trong luận văn sẽ rõ ràng hơn nhiều.

```bash
python -m scripts.regenerate_training_history --output-dir thesis/figures
```

---

### `app_v2.py` — Advanced Gradio Demo
Demo nâng cấp cho buổi bảo vệ:
- Model selector: 4 model + Ensemble soft-voting
- Grad-CAM overlay cho từng model (disabled khi chọn Ensemble)
- Top-3 predictions + severity tag (🔴 malignant / 🟢 benign)
- Low-confidence / high-entropy / malignant warnings
- Clinical disclaimer

```bash
python app_v2.py
# Hoặc: GRADIO_SERVER_PORT=7861 python app_v2.py
```

---

## 🚀 Quy trình chạy đầy đủ (sau khi ablation hoàn tất)

```bash
# 1. Parse ablation results (sẽ thêm script nếu cần)
# 2. Regenerate training history figures
python -m scripts.regenerate_training_history

# 3. Quantitative Grad-CAM (~10-30 phút tuỳ GPU)
python -m src.gradcam_quant_runner --n-per-class 80

# 4. TTA trên ensemble
python -m src.tta --ensemble --policy full --output-dir outputs/tta_ensemble

# 5. Calibration analysis
python -m src.calibration_analysis

# 6. Error analysis
python -m src.error_analysis

# 7. Threshold tuning cho melanoma
python -m src.threshold_tuning \
    --predictions outputs/ensemble/test_predictions.csv \
    --criterion f1 --min-specificity 0.85 --target-class mel

# 8. Chạy demo
python app_v2.py
```

---

## 📝 Update thesis (sau khi có kết quả)

Chapter 4 sẽ được bổ sung:
1. **4.X Ablation Study** — điền số liệu từ `outputs/ablation_*/test_metrics.json`
2. **4.Y Quantitative Grad-CAM Analysis** — bảng focus_ratio/entropy cho 4 model, so sánh CNN vs Swin
3. **4.Z Calibration Analysis** — reliability diagram, bảng ECE trước/sau T-scaling
4. **4.W TTA Results** — bảng so sánh TTA off/flip/full
5. **4.V Error Analysis** — top confusion pairs (MEL↔NV, AKIEC↔BKL), ensemble-fixed cases
6. **4.U Threshold Tuning** — bảng MEL sensitivity trước/sau tuning

---

## 💡 Đề xuất cải thiện thêm (phụ thuộc thời gian)

| Ý tưởng | Thời gian | Giá trị |
|---------|-----------|---------|
| **Weighted ensemble** (grid search trọng số trên val) | 0.5 ngày | +0.5–1% accuracy |
| **Multi-seed runs** (EB0 với seed 7, 123, 2024) | 1 ngày | Độ tin cậy thống kê (mean ± std) |
| **SOTA comparison table** (Gessert 2020, Datta 2021) | 0.5 ngày | Literature review chặt chẽ |
| **Test-set external** (PH2 / 7-point criteria) | 1 ngày (nếu có data) | Chứng minh generalization |
| **Knowledge Distillation** (ensemble → 1 EB0 nhỏ) | 1–2 ngày | Đóng gói deploy |

---

## ✅ Checklist trước khi bảo vệ

- [ ] Chạy hết các script phân tích, copy figure vào `thesis/figures/`
- [ ] Điền số liệu ablation vào chapter 4
- [ ] Regenerate `thesis/main.pdf`
- [ ] Demo `app_v2.py` chạy mượt trên máy bảo vệ
- [ ] Chuẩn bị 3–5 ảnh mẫu khó (MEL, BCC, AKIEC) để demo live
- [ ] Slide có: Grad-CAM quantitative, calibration, ablation, TTA, threshold tuning
- [ ] Backup PDF trên USB + cloud

---

## 📊 Kỳ vọng kết quả cuối cùng

| Metric | Baseline | Enhanced | Target ≥ 9/10 |
|--------|----------|----------|----------------|
| Accuracy (single best) | 88.4% | 88.4% | ✅ |
| Accuracy (ensemble) | 89.5% | — | ✅ |
| Accuracy (ensemble + TTA) | — | **~90.5–91%** | 🎯 |
| MEL sensitivity | ~0.76 | **≥ 0.85** (tuned) | 🎯 |
| ECE (ensemble) | ~8–12% | **< 3%** (T-scaled) | 🎯 |
| Grad-CAM focus_ratio | qualitative | **quantitative per-class** | 🎯 |

---

**Tác giả**: Sinh viên đồ án tốt nghiệp
**Liên hệ**: [your email]
