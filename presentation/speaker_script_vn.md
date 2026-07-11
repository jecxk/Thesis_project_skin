# Kịch bản thuyết trình — Bảo vệ khoá luận (13 slide)

**Sinh viên:** Nguyễn Trọng Bách — 23BI14057
**Mục tiêu:** ~10 phút thuyết trình + 15 phút trả lời câu hỏi
**Bám sát:** `slides.pdf` sau khi tái cấu trúc ngày 2026-07-11

Đọc kịch bản một lần, sau đó **nói theo ý, đừng đọc thuộc lòng**. Mỗi slide có
thời lượng gợi ý; nếu chậm, cắt các câu in nghiêng trước.

---

## Bảng phân bổ thời gian

| Slide | Nội dung                                     | Đến phút |
|:-----:|----------------------------------------------|:--------:|
| 1     | Trang bìa                                     | 0:30     |
| 2     | Nội dung trình bày                            | 0:45     |
| 3     | Vấn đề & Mục tiêu                             | 1:45     |
| 4     | Dữ liệu & Thách thức                          | 2:45     |
| 5     | Pipeline tổng quan                            | 3:30     |
| 6     | Bốn kiến trúc so sánh                         | 4:15     |
| 7     | Chiến lược huấn luyện                         | 5:00     |
| 8     | Mô hình đơn + Ensemble                        | 6:15     |
| 9     | Ma trận nhầm lẫn                              | 7:00     |
| 10    | Ablation                                      | 8:00     |
| 11    | Grad-CAM + Streamlit                          | 9:00     |
| 12    | Kết luận + Hướng phát triển                   | 9:45     |
| 13    | Cảm ơn                                        | 10:00    |

Nhắm kết thúc đúng phút 10:00. Nếu chậm, bỏ các câu in nghiêng trước.

---

## Slide 1 — Trang bìa (30 s)

> "Kính thưa hội đồng, em là Nguyễn Trọng Bách, sinh viên K23 khoa Công nghệ
> Thông tin và Truyền thông, Trường Đại học Khoa học và Công nghệ Hà Nội.
> Hôm nay em xin trình bày khoá luận tốt nghiệp *Đánh giá so sánh các mô hình
> học sâu cho bài toán phân loại tổn thương da trên ảnh dermoscopy*.
> Giảng viên hướng dẫn bên ngoài là TS. Vũ Trọng Sinh, giảng viên hướng dẫn
> nội bộ là TS. Nghiêm Thị Phương."

**Cách nói:** đứng thẳng, nhìn hội đồng, không nhìn slide.

---

## Slide 2 — Nội dung trình bày (15 s)

> "Bài trình bày gồm sáu phần chính: giới thiệu, dữ liệu và thách thức,
> phương pháp, kết quả thực nghiệm, ablation, và khả diễn giải kèm ứng dụng.
> Em sẽ giữ phần trình bày trong mười phút để dành thời gian cho câu hỏi."

**Cách nói:** chỉ vào sáu mục, không đọc từng dòng.

---

## Slide 3 — Vấn đề & Mục tiêu (60 s)

> "Ung thư hắc tố melanoma là dạng ung thư da nguy hiểm nhất — nhưng nếu
> phát hiện sớm, hơn chín mươi chín phần trăm bệnh nhân sống khỏe sau năm
> năm. Kính soi da dermoscopy giúp bác sĩ nhìn rõ tổn thương mà không cần
> chích rạch, tuy nhiên kết luận cuối cùng vẫn phụ thuộc rất nhiều vào kinh
> nghiệm người khám.
>
> Bất kỳ hệ thống AI nào muốn hỗ trợ chẩn đoán đều phải vượt qua ba trở
> ngại: mất cân bằng dữ liệu nghiêm trọng, hình thái các lớp rất giống nhau
> đặc biệt giữa melanoma và nốt ruồi lành, và bản chất hộp đen — bác sĩ
> không thể tin một quyết định mà họ không kiểm tra được lý do.
>
> Khoá luận này có bốn đóng góp: pipeline huấn luyện chống lệch dữ liệu,
> so sánh công bằng ba CNN cùng một Transformer, soft-voting ensemble vượt
> mọi mô hình đơn lẻ, và Grad-CAM cùng ứng dụng Streamlit để bác sĩ thấy
> mô hình đang nhìn vào đâu."

**Cách nói:** con số 99% là điểm nhớ — dừng nửa giây sau khi nói. Bốn đóng
góp chính là lộ trình của cả bài.

---

## Slide 4 — Dữ liệu & Thách thức (60 s)

> "Bộ dữ liệu là ISIC 2018 Task 3, còn gọi là HAM10000: mười nghìn không
> trăm mười lăm ảnh dermoscopy chia thành bảy lớp tổn thương, chia
> 70 / 15 / 15 theo phương pháp stratified.
>
> Đặc trưng quan trọng nhất là mất cân bằng cực đoan: nốt ruồi lành NV
> chiếm sáu mươi bảy phần trăm toàn bộ dữ liệu, trong khi dermatofibroma DF
> chỉ chiếm một phẩy một phần trăm. Tỉ lệ giữa lớp lớn nhất và nhỏ nhất là
> năm mươi tám trên một.
>
> Đây là vấn đề nghiêm trọng, vì một mô hình lười chỉ cần dự đoán 'NV cho
> mọi ảnh' đã đạt sáu mươi bảy phần trăm accuracy — nhưng hoàn toàn vô dụng
> lâm sàng. Chính vì thế em tối ưu Macro F1 và Macro AUC, để mọi lớp được
> bình đẳng.
>
> Bốn cơ chế đối phó trực tiếp với mất cân bằng: Weighted Sampler cân bằng
> từng mini-batch, Mixup và CutMix làm mịn ranh giới quyết định, Label
> Smoothing chống over-confidence, và Macro F1 là tiêu chí chọn checkpoint
> tốt nhất."

**Cách nói:** con số 58:1 là điểm nhấn — nói chậm lại khi tới đó.

---

## Slide 5 — Pipeline tổng quan (45 s)

> "Sơ đồ này tóm tắt toàn bộ pipeline trong một hình.
> Ảnh dermoscopy thô đi qua tiền xử lý về kích thước hai trăm hai mươi tư
> nhân hai trăm hai mươi tư, sau đó augmentation, rồi tới một trong bốn
> backbone pretrained ImageNet, cuối cùng là classifier head đưa ra xác
> suất bảy lớp.
>
> Quá trình huấn luyện dùng AdamW với cosine annealing, mixed-precision
> FP16, gradient clipping và early stopping theo Macro F1. Cùng một công
> thức được áp dụng cho cả bốn mô hình để giữ so sánh công bằng."

**Cách nói:** chỉ vào từng khối khi gọi tên. Đừng đi sâu — chi tiết ở hai
slide tiếp theo.

---

## Slide 6 — Bốn kiến trúc so sánh (45 s)

> "Em so sánh bốn backbone. EfficientNet-B0 chỉ có năm phẩy ba triệu tham
> số — CNN gọn nhẹ bắt tốt đặc trưng cục bộ. ResNet50 hai mươi lăm triệu
> tham số — CNN sâu, baseline cổ điển. DenseNet121 tám triệu tham số — CNN
> tận dụng feature reuse. Và Swin-Tiny hai mươi tám triệu — Vision
> Transformer cung cấp ngữ cảnh toàn cục.
>
> Cả bốn đều được fine-tune từ trọng số pretrained ImageNet qua thư viện
> timm. Đầu ra được thay bằng một lớp Linear bảy đầu ra kèm dropout. Việc
> chọn hỗn hợp CNN và Transformer là có chủ đích — nó cho ensemble những
> lỗi không tương quan để khai thác."

**Cách nói:** nhấn từ *không tương quan* — đây là lý do ensemble hoạt động.
Giữ slide này ngắn gọn.

---

## Slide 7 — Chiến lược huấn luyện (45 s)

> "Bên trái là khối chống mất cân bằng: Weighted Sampler với trọng số mỗi
> mẫu bằng N chia cho C nhân n_c, giúp mỗi mini-batch cân bằng gần đúng.
> Mixup tạo ảnh ảo bằng cách kết hợp tuyến tính hai ảnh, CutMix dán một
> patch ngẫu nhiên thay vì trộn, còn Label Smoothing làm mềm nhãn one-hot
> với epsilon bằng không phẩy một.
>
> Bên phải là khối tối ưu: AdamW với cosine annealing và linear warmup,
> mixed-precision FP16 tiết kiệm khoảng bốn mươi phần trăm bộ nhớ GPU,
> gradient clipping tại norm bằng một, và early stopping với patience mười
> hai epoch theo Macro F1. Batch size mười sáu, kích thước ảnh
> hai trăm hai mươi tư.
>
> *Siêu tham số cụ thể từng backbone — learning rate, weight decay, warmup
> — nằm ở phụ lục A24.*"

**Cách nói:** nếu còn thời gian thì nhắc câu in nghiêng, không thì bỏ qua.

---

## Slide 8 — Mô hình đơn và Ensemble (75 s)

> "Đây là những con số chính trên tập test.
>
> Bốn mô hình đơn cho accuracy từ tám mươi lăm đến tám mươi tám phần trăm.
> EfficientNet-B0 dẫn đầu về Macro F1 trong nhóm CNN dù là mô hình nhỏ nhất
> — bằng chứng rằng kiến trúc quan trọng hơn số tham số thô trên dữ liệu
> y tế hạn chế.
>
> Dòng cuối cùng là điểm nhấn: soft-voting ensemble đạt
> **tám mươi chín phẩy bốn chín phần trăm accuracy, Macro F1 không phẩy tám
> bốn năm, và AUC không phẩy chín tám** — vượt rõ ràng mọi mô hình đơn.
>
> Để chắc chắn đây không phải kết quả ngẫu nhiên của một hạt giống, em chạy
> lại toàn bộ pipeline qua năm hạt giống ngẫu nhiên. Ensemble ổn định ở mức
> **tám mươi chín phẩy chín ba phần trăm, cộng trừ không phẩy sáu sáu**.
> Kiểm định McNemar xác nhận ensemble vượt mọi backbone với p nhỏ hơn
> mười mũ trừ bảy so với mô hình mạnh nhất, và nhỏ hơn mười mũ trừ ba mươi
> tám so với mô hình yếu nhất.
>
> Soft voting hiệu quả ở đây vì các CNN bắt kết cấu cục bộ trong khi Swin
> bắt cấu trúc toàn cục, nên lỗi của chúng phần lớn không tương quan. Trung
> bình bốn vector xác suất không tương quan giúp giảm phương sai — không
> cần huấn luyện thêm, chỉ bốn lần inference."

**Cách nói:** đây là **khoảnh khắc "wow" đầu tiên** — dừng một giây tròn
sau khi nói p-value McNemar. Nhìn hội đồng, đừng nhìn slide.

---

## Slide 9 — Ma trận nhầm lẫn (45 s)

> "Ma trận nhầm lẫn xác nhận ba tính chất quan trọng.
>
> Thứ nhất, đường chéo mạnh trên mọi lớp — recall không bị hy sinh ở lớp
> hiếm. Dermatofibroma và Vascular với chưa tới hai trăm ảnh huấn luyện
> tổng cộng vẫn được nhận diện chính xác — kết quả trực tiếp của Mixup kết
> hợp với Weighted Sampler.
>
> Thứ hai, nhầm lẫn còn lại tập trung ở cụm Melanoma, Nevus và Benign
> Keratosis — ba lớp thực sự có hình thái tương đồng. Đây là cụm khó nhất
> của ISIC 2018.
>
> Thứ ba, và quan trọng nhất: không có lớp nào bị bỏ rơi. Lỗi kinh điển
> của phân loại mất cân bằng — quên hẳn các lớp minority — không xảy ra
> ở đây."

**Cách nói:** dùng bút chỉ, dẫn hội đồng đi qua đường chéo ngắn gọn.

---

## Slide 10 — Ablation (60 s)

> "Để đo đóng góp của từng thành phần, em bỏ từng thứ một khỏi baseline
> EfficientNet-B0 rồi huấn luyện lại.
>
> Full pipeline đạt Macro F1 không phẩy tám ba một. Bỏ Mixup và CutMix
> làm F1 giảm bốn phẩy năm phần trăm — đây là đòn bẩy lớn nhất, vì với chỉ
> tám mươi mốt ảnh DF mô hình sẽ ngay lập tức memorize các mẫu hiếm nếu
> không có Mixup.
>
> Bỏ Label Smoothing thì AUC tăng nhưng F1 giảm ba phẩy một phần trăm —
> trade-off kinh điển giữa calibration và quyết định cứng.
>
> Bỏ augmentation mạnh làm Accuracy thực ra tăng nhẹ, nhưng Macro F1 lại
> giảm — các lớp minority đang bị bỏ rơi. Đây chính xác là lý do vì sao em
> không dùng Accuracy làm tiêu chí chính.
>
> Mọi thành phần đều đóng góp dương cho Macro F1 — không có module thừa.
> *Diễn giải chi tiết từng dòng ablation nằm ở phụ lục A25.*"

**Cách nói:** câu *"Accuracy không phải tiêu chí chính"* là phát biểu
phương pháp luận cốt lõi — nói chậm lại khi tới câu này.

---

## Slide 11 — Grad-CAM + Streamlit (60 s)

> "Khả diễn giải khép lại vòng lặp giữa mô hình và bác sĩ. Grad-CAM tạo
> heatmap từ lớp convolution cuối, có trọng số theo gradient của lớp được
> dự đoán — hiển thị đúng những pixel khiến mô hình đưa ra quyết định.
>
> Trên ảnh melanoma mẫu này, cả bốn mô hình đều tập trung vào ổ sắc tố
> trung tâm chứ không phải nền, không phải lông, không phải bọt gel — cơ sở
> để bác sĩ có thể tin cậy hệ thống. Và quan trọng, em không dừng ở đánh
> giá định tính: chỉ số focus-ratio của Grad-CAM đều lớn hơn không phẩy sáu
> trên cả bốn backbone — một minh chứng định lượng.
>
> Prototype Streamlit đưa toàn bộ điều đó thành một công cụ trực tiếp: bác
> sĩ upload ảnh, hệ thống trả về xác suất bảy lớp cùng heatmap Grad-CAM
> trong dưới hai giây. Người dùng có thể chọn mô hình dùng để phân loại
> và mô hình hiển thị trong heatmap. Ứng dụng đóng vai trò **second opinion**
> cho bác sĩ — xác suất kèm bằng chứng trực quan, không bao giờ thay thế
> chẩn đoán."

**Cách nói:** cụm *"second opinion, không bao giờ thay thế"* là điểm neo
đạo đức của toàn khoá luận. Nói chậm lại.

---

## Slide 12 — Kết luận + Hướng phát triển (45 s)

> "Tóm tắt năm đóng góp chính: ensemble đạt tám mươi chín phẩy bốn chín
> phần trăm Accuracy và không phẩy chín tám AUC — ngang tầm SOTA cho các
> phương pháp chỉ-ảnh trên ISIC 2018. Ổn định qua nhiều hạt giống, với p
> nhỏ hơn mười mũ trừ bảy. Mọi thành phần được ablation đều đóng góp dương.
> Grad-CAM focus-ratio lớn hơn không phẩy sáu xác nhận mô hình tập trung
> vào tổn thương, không vào nhiễu. Và có một prototype Streamlit đang hoạt
> động mà bác sĩ có thể dùng thực sự.
>
> Bốn hướng phát triển: domain adaptation cho ảnh smartphone — chủ đề em
> đã khám phá ở phụ lục A23; open-set detection để mô hình biết đánh dấu
> input không phải tổn thương da; active learning loop để bác sĩ sửa nhãn
> sai; và triển khai biên qua ONNX hoặc TensorRT.
>
> Thông điệp em muốn để lại: một pipeline được điều chuẩn cẩn thận có thể
> thu hẹp phần lớn khoảng cách giữa mô hình nhẹ và mô hình lớn trên dữ liệu
> y tế hạn chế."

**Cách nói:** đây là đoạn kết — nói với sự tự tin, nhìn từng thành viên
hội đồng.

---

## Slide 13 — Cảm ơn (30 s)

> "Em xin cảm ơn hội đồng đã lắng nghe. Em sẵn sàng đón nhận các câu hỏi."

**Cách nói:** mỉm cười, đứng thẳng, tay thả tự nhiên. Chờ câu hỏi đầu tiên
— đừng lấp khoảng lặng.

---

## Kế hoạch dự phòng

- **Nếu chậm (đã qua 8:00 mà mới tới slide 8):** bỏ các câu in nghiêng,
  rút slide 6 và 7 còn 30 s mỗi cái, vẫn kết thúc slide 12 đúng 10:00.
- **Nếu hội đồng ngắt hỏi giữa bài:** trả lời ngắn gọn và đánh dấu chỗ
  đang trình bày. Đừng làm lại từ đầu phần đó.
- **Nếu hội đồng yêu cầu demo Streamlit giữa bài:** lịch sự nói *"Em có
  demo đang chạy sẵn và xin phép trình bày sau khi kết thúc phần slide
  chính"* — để bảo vệ tiến độ.

## Chuẩn bị cho phần Q&A

Mười câu hỏi có khả năng nhất, kèm gợi ý trả lời, nằm trong
[defense_qa.pdf](defense_qa.pdf). Tập trung vào:

- Câu 1: Vì sao soft voting thay vì stacking?
- Câu 2: Làm sao biết Grad-CAM đúng?
- Câu 5: Ảnh ngoài 7 lớp thì sao?
- Câu 6: Bao nhiêu melanoma bị bỏ sót?
- Câu 9: Bias về màu da tối — bối cảnh Việt Nam.

## Hai điều bị cấm

1. **Đừng đọc slide.** Hội đồng đọc nhanh hơn tốc độ em nói. Slide chỉ là
   đạo cụ, không phải kịch bản.
2. **Đừng nói quá.** Nếu bị hỏi "hệ thống này triển khai được chưa?" — câu
   trả lời là *chưa*, và đây là những gì cần làm thêm. Trung thực được điểm.
