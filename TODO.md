# TODO — Project Improvement Plan

> Track tiến độ các cải tiến code + thesis song song với quá trình train ablation.

## 📌 Legend
- [ ] = chưa làm
- [x] = đã hoàn thành
- [~] = đang làm / phụ thuộc ablation results

---

## Phase 1 — Priority 1 (MUST HAVE)

### 1.1 Quantitative Grad-CAM Analysis
- [x] Tạo `src/utils/gradcam_quant.py` (mean activation in/out lesion, entropy, peak location, coverage_75)
- [x] Tạo `src/gradcam_quant_runner.py` — runner cho 4 model, export per-image CSV + aggregated per (model, class) + summary JSON
- [x] Tạo plot `focus_ratio_comparison.png` và `entropy_vs_peakdist.png`
- [ ] Chạy `python -m src.gradcam_quant_runner --n-per-class 80` (sau khi ablation xong)

### 1.2 Test-Time Augmentation (TTA)
- [x] Tạo `src/tta.py` (hflip, vflip, 90°/180°/270°; TTA_POLICIES none/flip/full)
- [x] Hỗ trợ ensemble + TTA kết hợp (`--ensemble`)
- [ ] Chạy trên best EB0 + ensemble, compare metrics

### 1.3 Calibration Analysis
- [x] Tạo `src/utils/calibration.py` (ECE, MCE, Brier, NLL, reliability diagram, TemperatureScaler LBFGS)
- [x] Tạo `src/calibration_analysis.py` chạy trên 4 model + ensemble
- [x] Temperature scaling post-hoc calibration (50/50 val split)
- [x] **SMOKE RUN DONE** — outputs/calibration_analysis/ (22 files)
  - Raw ECE: EB0=5.87% RN50=4.73% DN121=5.60% Swin=4.35% Ensemble=9.55%
  - After T-scaling: EB0=2.52% RN50=3.70% DN121=5.01% Swin=5.80% Ensemble=2.63%
  - Insight: Ensemble over-confident → T-scaling giảm 72% ECE
- [ ] Ghi kết quả vào chapter 4 (sau khi ablation xong)

### 1.4 Error & Confusion Pair Analysis
- [x] Tạo `src/error_analysis.py`
  - [x] Top confusion pairs
  - [x] Per-class error rate
  - [x] Hard examples (lowest confidence misclassifications)
  - [x] Ensemble-vs-single comparison (fixed / broken / agree)
- [x] **SMOKE RUN DONE** — outputs/error_analysis/
  - Ensemble fix 62 cases, break 45 cases (net +17, +1.13% acc)
- [ ] Generate figures cuối cùng cho thesis

### 1.5 Nâng cấp Gradio Demo
- [x] Tạo `app_v2.py` với:
  - [x] Model selector (4 model + Ensemble soft voting)
  - [x] Grad-CAM overlay
  - [x] Top-3 + clinical description + severity tag
  - [x] Low-confidence / high-entropy / malignant warnings
- [ ] Test chạy local (`python app_v2.py`)

### 1.6 Fix Training History Figures
- [x] Tạo `scripts/regenerate_training_history.py` — mỗi model 1 figure 2x2 (14×10), + cross-model val loss/F1
- [x] **SMOKE RUN DONE** — thesis/figures_new/ (6 files: 4 per-model + 2 cross-model)
- [ ] Copy `thesis/figures_new/*` → `thesis/figures/` thay cho hình cũ
- [ ] Cập nhật `thesis/chapter4.tex` dùng 4 figure to riêng

---

## Phase 2 — Priority 2 (SHOULD HAVE)

### 2.1 Fix & Clean Code
- [ ] Verify `src/train.py` không có bug
- [ ] Bổ sung type hints + docstrings các module chính
- [ ] Update `README.md` với full commands

### 2.2 Threshold Tuning for Melanoma
- [x] Tạo `src/threshold_tuning.py` (criteria: f1 / sensitivity / youden, min-specificity constraint, PR+ROC plot)
- [x] **SMOKE RUN DONE** (val=test → optimistic preview)
  - Macro F1: 0.845 → 0.860 (+1.5 pp)
  - AKIEC sensitivity: 0.674 → 0.796 (+12.2 pp!)
  - MEL sensitivity: 0.695 → 0.713 (+1.8 pp)
- [ ] Cần tạo **val_predictions.csv** (chạy ensemble trên split val) → tune trên đó → apply lên test để có kết quả công bằng

### 2.3 Rewrite `thesis/chapter4.tex`
- [ ] Fill ablation results (sau khi train xong)
- [ ] Add Calibration Analysis section
- [ ] Add TTA Results section
- [ ] Add Error Analysis section
- [ ] Replace training history figure với 4 figure to rõ

### 2.4 Enrich `thesis/chapter2.tex` (Literature Review)
- [ ] Add SOTA comparison table (Gessert 2020, Datta 2021, v.v.)
- [ ] Add ~5-8 references mới vào `references.bib`

### 2.5 Update `thesis/chapter5.tex`
- [ ] Remove "web deployment" from future work (đã làm)
- [ ] Add TTA, calibration insights vào contributions
- [ ] Update Future Work cho đúng status

---

## Phase 3 — Priority 3 (NICE TO HAVE)

### 3.1 Weighted Ensemble (Grid Search)
- [ ] Script tìm trọng số tối ưu trên val set
- [ ] Compare với uniform average

### 3.2 Multi-seed Runs (optional nếu còn thời gian)
- [ ] Train EB0 với seed 7, 123, 2024
- [ ] Report mean ± std

### 3.3 Knowledge Distillation Preview
- [ ] (optional) distill ensemble → 1 small EB0

---

## Ablation Integration (when ablation training completes)
- [ ] Parse `outputs/ablation_*/test_metrics.json`
- [ ] Auto-generate LaTeX table
- [ ] Write ablation analysis paragraph

---

## Final Deliverables
- [ ] Updated `thesis/main.pdf` với đầy đủ sections
- [ ] Updated `README.md`
- [ ] Deploy `app_v2.py` local/HuggingFace Spaces (optional)
- [ ] All figures regenerated with better quality
