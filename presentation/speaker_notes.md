# Defense Speaker Notes — Skin Lesion Classification

**Author:** Nguyễn Trọng Bách — 23BI14057
**Updated:** 2026-06-07 (post Week-1/2/3 work)

Use this as a memory aid, **not** a script to read. Aim for ~15 minutes total. The slides
have all the numbers and tables — speaker notes are about narrative and pacing.

---

## Pacing budget (15 minutes total)

| Slides | Content | Time |
|---|---|---|
| 1–2  | Title + outline | 0:30 |
| 3–4  | Motivation + objectives | 2:00 |
| 5–6  | Dataset + imbalance | 1:30 |
| 7–10 | Pipeline + 4 backbones + hyperparams | 2:30 |
| 11–14 | Single-model + ensemble + per-class + confusion | 3:00 |
| 15–16 | Ablation | 1:30 |
| 17–18 | Grad-CAM + prototype | 1:30 |
| 19–20 | Conclusion + future work | 1:30 |
| 21 | Thank you | 0:30 |
| —    | Buffer | 1:00 |

Aim to finish at minute 14, leaving 1 minute buffer. Demo prototype only if asked
during Q&A (saves 2–3 minutes of presentation time).

---

## Slide 1 — Title

> "Kính thưa hội đồng, em là Nguyễn Trọng Bách, sinh viên K23 USTH. Hôm nay em xin
> trình bày khóa luận tốt nghiệp về \"Đánh giá so sánh các mô hình học sâu cho bài
> toán phân loại tổn thương da trên ảnh dermoscopy\". GVHD nội bộ là TS. Nghiêm
> Thị Phương, GVHD bên ngoài là TS. Vũ Trọng Sinh."

**Tip:** đứng thẳng, nhìn hội đồng, không nhìn slide.

## Slide 2 — Outline

Nói nhanh 5 phần chính. Không cần đọc từng dòng.

## Slide 3 — Motivation

> "Ung thư hắc tố melanoma là nguyên nhân chính của tử vong do ung thư da, nhưng
> nếu phát hiện sớm thì sống sót 5 năm trên 99%. AI hỗ trợ chẩn đoán cần vượt qua
> ba rào cản: (1) mất cân bằng dữ liệu nghiêm trọng, (2) hình thái tổn thương
> rất giống nhau giữa các lớp, (3) bản chất hộp đen làm bác sĩ khó tin."

## Slide 4 — Objectives

Highlight: **6 đóng góp**, đặc biệt là 2 phần mới so với khóa luận ban đầu:
- **Metadata fusion** (vượt 90%)
- **OOD evaluation** trên PAD-UFES-20

## Slide 5 — Dataset ISIC 2018

> "Tập ISIC 2018 Task 3 có 10.015 ảnh dermoscopy chia 7 lớp. Chia 70/15/15 stratified.
> Lớp NV chiếm 67%, lớp DF chỉ 1.1% — tỉ lệ 58:1."

## Slide 6 — Imbalance

Show bảng phân bố. Nhấn vào "Macro F1" là metric chính, không phải Accuracy.

## Slide 7 — Pipeline

Đi nhanh: data → augmentation → backbone fine-tune → loss.

## Slide 8 — 4 backbones

Highlight Swin-Tiny là Transformer duy nhất → cung cấp ngữ cảnh toàn cục bù cho CNN.

## Slide 9 — Training strategy

Nhấn vào ba thành phần regularization:
- Weighted Sampler (cân bằng minibatch)
- Mixup + CutMix (làm mềm decision boundary)
- Label Smoothing (chống over-confidence)

## Slide 10 — Hyperparameters

> "Swin-Tiny khác biệt: lr thấp 1e-4 (CNN dùng 3e-4) và weight decay cao 5e-2 (CNN
> 1e-3). Lý do: Transformer không có inductive bias của CNN nên cần regularize mạnh hơn."

## Slide 11 — Single-model results

Đọc bảng:
- EB0 5.3M tham số đạt 88.36% — best single ở seed 42.
- Swin 28M tham số ở seed 42 chỉ 84.90% (overfit sớm epoch 13).

**Nhưng quan trọng:** đây mới chỉ 1 hạt giống. Chuyển sang slide 12 sẽ thấy bức tranh đầy đủ.

## Slide 12 — Ensemble + multi-seed

**Đây là slide có "wow moment" đầu tiên.**

> "Ensemble đạt 89.49% ở hạt giống đại diện. Em nhận ra một con số đơn lẻ chưa đủ
> thuyết phục, nên chạy lại toàn bộ pipeline với 5 hạt giống. Kết quả: **89.93 ± 0.66%**
> qua 5 hạt giống. McNemar test xác nhận ensemble vượt mọi mô hình đơn có ý nghĩa thống kê."

## Slide 13 — Per-class

Highlight cụ thể:
- MEL F1 tăng từ 0.695 (EB0) lên 0.744 (ensemble) — quan trọng lâm sàng nhất.
- AUC tăng đồng đều mọi lớp.

## Slide 14 — Confusion matrix

> "Không có lớp nào bị bỏ rơi. DF và VASC dù hiếm vẫn được phát hiện chính xác.
> Nhầm lẫn còn lại tập trung ở cụm MEL/NV/BKL — đây là cụm khó nhất về mặt lâm sàng."

## Slide 15 — Ablation

> "Em loại bỏ từng thành phần để đo đóng góp. Mixup/CutMix là quan trọng nhất:
> bỏ đi giảm 4.51% F1. Đây là bài học kỹ thuật cốt lõi của khóa luận."

## Slide 16 — Ablation insights

Nhấn vào 2 nghịch lý:
- Bỏ Label Smoothing tăng AUC nhưng giảm F1
- Bỏ Augmentation tăng Accuracy nhưng giảm F1

> "Hai nghịch lý này cho thấy chọn metric nào để tối ưu là quyết định thiết kế, không
> phải mặc nhiên."

## Slide 17 — Grad-CAM

> "Em không dừng ở định tính. Em đo 4 chỉ số định lượng so với mask lesion: focus_ratio,
> peak_distance, entropy, coverage_75. Tất cả 4 mô hình đều có focus_ratio > 0.6 và
> peak_distance < 0.2 — bằng chứng định lượng rằng mô hình thực sự nhìn vào tổn thương."

Có thể nhắc story `model.bn2` debug nếu còn thời gian.

## Slide 18 — Streamlit prototype

Mô tả app ngắn gọn. Không demo trừ khi được hỏi.

## Slide 19 — Conclusion (6 contributions)

**Đây là slide có "wow moment" thứ hai và thứ ba:**

> "Khóa luận có 6 đóng góp chính:
> 1. Ensemble chỉ-ảnh đạt 89.93 ± 0.66% qua 5 hạt giống.
> 2. McNemar test xác nhận ensemble vượt mọi mô hình đơn p < 10⁻⁷.
> 3. **Thêm metadata bệnh nhân đẩy lên 90.49%** — vượt Pacheco & Krohling 2020.
>    Đây là kết quả image+metadata fusion mạnh nhất công bố trên ISIC 2018.
> 4. **Đánh giá OOD trên PAD-UFES-20 (ảnh smartphone) chỉ đạt 32.5%** — em báo cáo
>    trung thực để giới hạn kỳ vọng triển khai thực tế.
> 5. Grad-CAM định lượng (focus_ratio > 0.6) — bằng chứng mô hình tin cậy.
> 6. Streamlit prototype subprocess-isolated."

## Slide 20 — Future Work

3 việc còn lại:
1. Domain adaptation cho ảnh smartphone (đóng khoảng 56pp OOD drop).
2. Open-set detection (phát hiện input không phải tổn thương).
3. Active learning loop.

**Tránh nhắc multimodal hoặc OOD evaluation ở đây — vì đã làm rồi.**

## Slide 21 — Thank you

> "Em xin cảm ơn hội đồng đã lắng nghe. Em sẵn sàng đón nhận câu hỏi."

Đứng thẳng, mỉm cười, chờ câu hỏi.

---

## Q&A — chuẩn bị riêng

Xem file `defense_qa.pdf` để biết 10 câu hỏi khó nhất và cách trả lời.

Slide phụ lục từ A1 đến A23 là **kho tham chiếu** — không trình bày tuần tự nhưng phải
biết slide nào tham chiếu khi hội đồng hỏi:
- A8: Grad-CAM target layer
- A14: Tại sao Swin overfit
- A18: Threshold tuning
- A19: SOTA comparison
- A21: Multi-seed robustness
- A22: Metadata fusion details
- A23: OOD evaluation details

---

## Hai điều bị cấm

1. **Đừng đọc slide** — speaker notes là gợi ý, không phải kịch bản.
2. **Đừng over-claim** — nếu hội đồng hỏi "deploy được chưa", trả lời thẳng "chưa, cần
   domain adaptation cho ảnh smartphone và OOD detection."
