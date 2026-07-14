# Kịch bản thuyết trình — Bảo vệ khoá luận (13 slide)

**Sinh viên:** Nguyễn Trọng Bách — 23BI14057
**Mục tiêu:** ~10 phút thuyết trình, nói tự nhiên, thuyết phục, đầy đủ số liệu
**Bám sát:** `slides.pdf` bản mới nhất (đã đổi ảnh Pipeline Overview, chuyển
70/15/15 lên slide Dữ liệu, gọn lại slide Augmentation và Four Architectures)

Đọc qua 1–2 lần cho quen mạch, sau đó **kể lại bằng lời của mình**, đừng học
thuộc từng chữ. Mỗi slide có thời lượng gợi ý; nếu chậm, cắt câu in nghiêng
trước tiên. Giữ giọng chắc, tốc độ vừa phải — đừng đọc vội cho kịp giờ, thà
cắt bớt còn hơn nói nhanh không rõ chữ.

---

## Bảng phân bổ thời gian

| Slide | Nội dung                                      | Đến phút |
|:-----:|-----------------------------------------------|:--------:|
| 1     | Trang bìa                                      | 0:20     |
| 2     | Nội dung trình bày                             | 0:35     |
| 3     | Vấn đề & Mục tiêu                              | 1:35     |
| 4     | Dữ liệu & Thách thức                           | 2:40     |
| 5     | Pipeline tổng quan                             | 3:15     |
| 6     | Tiền xử lý & Augmentation                      | 4:05     |
| 7     | Bốn kiến trúc so sánh                          | 4:45     |
| 8     | Mô hình đơn + Ensemble                         | 5:55     |
| 9     | Ma trận nhầm lẫn                               | 6:35     |
| 10    | Ablation                                       | 7:30     |
| 11    | Grad-CAM                                       | 8:25     |
| 12    | Kết luận + Hướng phát triển                    | 9:05     |
| 13    | Câu hỏi và Thảo luận                           | 9:20     |

Nhắm kết thúc trước phút 9:30, còn dư ~30 giây dự phòng.

---

## Slide 1 — Trang bìa (20 s)

> "Kính thưa hội đồng, em là Nguyễn Trọng Bách, sinh viên K23 khoa Công nghệ
> Thông tin và Truyền thông, Trường Đại học Khoa học và Công nghệ Hà Nội.
> Hôm nay em xin trình bày khoá luận tốt nghiệp *Đánh giá so sánh các mô hình
> học sâu cho bài toán phân loại tổn thương da trên ảnh dermoscopy*. Giảng
> viên hướng dẫn ngoài trường là TS. Vũ Trọng Sinh, hướng dẫn trong trường là
> TS. Nghiêm Thị Phương."

**Cách nói:** đứng thẳng, mắt nhìn hội đồng, không nhìn slide, không cầm giấy.

---

## Slide 2 — Nội dung trình bày (15 s)

> "Bài trình bày gồm sáu phần: giới thiệu bài toán, dữ liệu và thách thức,
> phương pháp, kết quả thực nghiệm, nghiên cứu ablation, và khả năng diễn
> giải kèm ứng dụng. Em sẽ giữ phần trình bày trong khoảng mười phút để dành
> thời gian cho phần hỏi đáp."

**Cách nói:** chỉ tay lướt nhanh qua sáu mục, không đọc từng dòng.

---

## Slide 3 — Vấn đề & Mục tiêu (60 s)

> "Melanoma — ung thư hắc tố — là dạng ung thư da nguy hiểm nhất. Nhưng nếu
> phát hiện sớm, hơn **chín mươi chín phần trăm** bệnh nhân sống khỏe sau
> năm năm. Dermoscopy — kính soi da chuyên dụng — giúp bác sĩ nhìn rõ tổn
> thương hơn, nhưng kết luận cuối cùng vẫn phụ thuộc rất nhiều vào kinh
> nghiệm của người khám.
>
> Bất kỳ hệ thống AI nào muốn hỗ trợ chẩn đoán đều phải vượt qua ba rào cản:
> dữ liệu mất cân bằng nghiêm trọng giữa các loại bệnh, hình thái nhiều lớp
> rất giống nhau — đặc biệt giữa melanoma và nốt ruồi lành — và bản chất
> hộp đen của mạng nơ-ron sâu, khiến bác sĩ khó tin một quyết định mà họ
> không kiểm tra được lý do.
>
> Khoá luận này giải quyết cả ba, với bốn đóng góp: một pipeline huấn luyện
> chống lệch dữ liệu, một so sánh công bằng giữa ba CNN và một Vision
> Transformer, một ensemble kết hợp cả bốn cho kết quả vượt trội, và cuối
> cùng là Grad-CAM để bác sĩ nhìn thấy chính xác mô hình đang chú ý vào đâu
> trên ảnh."

**Cách nói:** con số 99% là điểm nhấn cảm xúc đầu tiên — dừng nửa giây sau
khi nói. Bốn đóng góp chính là lộ trình cho toàn bộ phần còn lại.

---

## Slide 4 — Dữ liệu & Thách thức (65 s)

> "Bộ dữ liệu em dùng là ISIC 2018 Task 3, còn gọi là HAM10000: mười nghìn
> không trăm mười lăm ảnh dermoscopy, chia thành bảy lớp tổn thương. Em chia
> dữ liệu theo tỉ lệ bảy mươi, mười lăm, mười lăm phần trăm cho train, val,
> test — theo kiểu stratified, nghĩa là tỉ lệ bảy lớp ở cả ba tập đều giống
> hệt tỉ lệ gốc, không lớp nào bị dồn lệch vào một tập.
>
> Đặc điểm nghiêm trọng nhất của bộ dữ liệu này là mất cân bằng cực đoan:
> lớp Nevus — nốt ruồi lành — chiếm sáu mươi bảy phần trăm toàn bộ dữ liệu,
> trong khi Dermatofibroma chỉ chiếm một phẩy một phần trăm. Tỉ lệ giữa lớp
> lớn nhất và nhỏ nhất là **năm mươi tám trên một**.
>
> Đây là vấn đề thật sự nghiêm trọng: một mô hình lười biếng chỉ cần luôn
> đoán 'Nevus' cho mọi tấm ảnh cũng đạt sáu mươi bảy phần trăm accuracy —
> nhìn có vẻ cao, nhưng hoàn toàn vô dụng về mặt lâm sàng vì nó không phát
> hiện được ca bệnh nào khác. Vì vậy, em không dùng Accuracy làm thước đo
> chính, mà tối ưu Macro F1 và Macro AUC — hai chỉ số buộc mọi lớp bệnh
> phải được đối xử công bằng như nhau, bất kể nhiều hay ít ảnh.
>
> Để đối phó trực tiếp với mất cân bằng, em áp dụng bốn cơ chế: Weighted
> Sampler để cân bằng từng mini-batch khi lấy mẫu huấn luyện, Mixup và
> CutMix để làm mịn ranh giới quyết định giữa các lớp, Label Smoothing để
> chống việc mô hình quá tự tin, và dùng chính Macro F1 làm tiêu chí chọn
> checkpoint tốt nhất."

**Cách nói:** con số "58 trên 1" là điểm nhấn quan trọng nhất slide — nói
chậm, rõ ràng, để hội đồng thấm được mức độ nghiêm trọng.

---

## Slide 5 — Pipeline tổng quan (35 s)

> "Sơ đồ này tóm gọn toàn bộ hệ thống trong một hình. Ảnh dermoscopy đầu vào
> đi qua bước tiền xử lý và augmentation, sau đó tới một trong bốn backbone
> pretrained mà em sẽ giới thiệu ngay sau đây, qua classifier head, và cho
> ra xác suất của bảy lớp bệnh.
>
> Phần màu vàng bên dưới là vòng lặp huấn luyện — tính loss, lan truyền
> ngược để cập nhật trọng số — chỉ diễn ra lúc train, không xuất hiện khi
> mô hình đã huấn luyện xong và dùng để dự đoán thực tế."

**Cách nói:** chỉ tay vào từng khối theo đúng thứ tự trái sang phải, rồi
chỉ vào khối vàng khi nói tới "vòng lặp huấn luyện". Không đi sâu — hai
slide tiếp theo sẽ giải thích chi tiết.

---

## Slide 6 — Tiền xử lý & Augmentation (50 s)

> "Trước khi vào mô hình, mỗi ảnh phải qua hai bước. Tiền xử lý áp dụng cho
> mọi tập dữ liệu — đơn giản là resize ảnh về hai trăm hai mươi tư nhân hai
> trăm hai mươi tư, và chuẩn hóa theo mean, độ lệch chuẩn của ImageNet, vì
> cả bốn backbone đều được pretrained trên ImageNet.
>
> Augmentation thì khác biệt hơn — chỉ áp dụng cho tập train, và được tạo
> động mỗi epoch, không lưu sẵn. Lý do là vì một số lớp bệnh chỉ có vài
> chục ảnh để học — như Dermatofibroma chỉ có tám mươi mốt ảnh train. Nếu
> học đi học lại đúng từng ấy ảnh, mô hình sẽ ghi nhớ y nguyên từng tấm
> thay vì học đặc điểm bệnh lý thật sự. Nên mỗi epoch, ảnh được biến đổi
> ngẫu nhiên: xoay, lật ngang dọc, phóng to thu nhỏ một phần ảnh, đổi nhẹ
> màu sắc, làm mờ nhẹ, và che đi một góc nhỏ.
>
> Ảnh minh họa bên phải là ví dụ thật, chạy trên đúng đoạn code em dùng —
> mỗi ô là một kỹ thuật riêng biệt áp dụng lên cùng một ảnh gốc."

**Cách nói:** dừng 2 giây cho hội đồng nhìn ảnh minh họa trước khi nói câu
cuối. Nếu bị hỏi tại sao không dùng phép biến dạng mạnh hơn như shear hay
warp — trả lời: hình dạng và đường viền của tổn thương chính là dấu hiệu
chẩn đoán, nên không được phép làm méo nó.

---

## Slide 7 — Bốn kiến trúc so sánh (40 s)

> "Em so sánh bốn backbone. EfficientNet-B0 chỉ năm phẩy ba triệu tham số —
> CNN gọn nhẹ, bắt tốt đặc trưng cục bộ. ResNet50 hai mươi lăm triệu tham
> số — CNN sâu, kiến trúc kinh điển. DenseNet121 tám triệu tham số — CNN
> tận dụng feature reuse. Và Swin-Tiny hai mươi tám triệu tham số — Vision
> Transformer, cung cấp ngữ cảnh toàn cục thay vì chỉ nhìn cục bộ.
>
> Cả bốn đều fine-tune từ trọng số pretrained ImageNet. Việc chọn kết hợp cả
> CNN lẫn Transformer là có chủ đích — nó cho ensemble sau này khai thác
> được những lỗi không tương quan giữa hai trường phái kiến trúc. Biểu đồ
> bên dưới là đường cong loss và Macro F1 thật khi train — thấy rõ Swin-Tiny
> dừng sớm nhất, ở epoch mười ba trên năm mươi, một dấu hiệu overfit sớm mà
> em sẽ quay lại phân tích ở phần sau."

**Cách nói:** nhấn mạnh cụm *"lỗi không tương quan"* — đây là chìa khóa lý
giải vì sao ensemble ở slide sau lại hiệu quả. Không cần đọc hết bảng số
liệu tham số.

---

## Slide 8 — Mô hình đơn và Ensemble (75 s)

> "Đây là kết quả chính trên tập test — dữ liệu mô hình chưa từng nhìn thấy.
>
> Bốn mô hình đơn cho Accuracy từ tám mươi lăm đến tám mươi tám phần trăm.
> Đáng chú ý, EfficientNet-B0 dẫn đầu về Macro F1 trong nhóm CNN dù là mô
> hình nhỏ nhất — bằng chứng cho thấy kiến trúc phù hợp quan trọng hơn số
> tham số thô khi dữ liệu y tế còn hạn chế.
>
> Nhưng dòng cuối cùng mới là điểm nhấn: khi kết hợp cả bốn mô hình bằng
> soft-voting — lấy trung bình xác suất dự đoán của cả bốn — kết quả đạt
> **tám mươi chín phẩy bốn chín phần trăm Accuracy, Macro F1 không phẩy tám
> bốn năm, và AUC không phẩy chín tám** — vượt rõ rệt so với bất kỳ mô hình
> đơn lẻ nào.
>
> Để chắc chắn đây không phải may mắn của một lần chạy, em lặp lại toàn bộ
> pipeline với năm hạt giống ngẫu nhiên khác nhau. Ensemble vẫn ổn định ở
> mức tám mươi chín phẩy chín ba phần trăm, cộng trừ chỉ không phẩy sáu
> sáu. Và em còn chạy kiểm định thống kê McNemar, cho kết quả p nhỏ hơn
> mười mũ trừ bảy so với mô hình mạnh nhất — nghĩa là sự vượt trội này gần
> như chắc chắn không phải do ngẫu nhiên.
>
> Vì sao soft voting hiệu quả đến vậy? Vì nhóm CNN bắt tốt kết cấu cục bộ,
> trong khi Swin bắt tốt cấu trúc toàn cục — nên sai số của chúng phần lớn
> không tương quan với nhau. Lấy trung bình bốn dự đoán không tương quan
> giúp giảm phương sai đáng kể, mà không cần huấn luyện thêm gì cả — chỉ
> tốn thêm ba lần inference."

**Cách nói:** đây là **khoảnh khắc thuyết phục nhất của cả bài** — nói chậm
rãi, dừng trọn một giây sau khi nói p-value McNemar, nhìn thẳng vào hội
đồng chứ không nhìn slide.

---

## Slide 9 — Ma trận nhầm lẫn (40 s)

> "Ma trận nhầm lẫn của ensemble xác nhận ba điều quan trọng.
>
> Một, đường chéo mạnh ở mọi lớp — recall không bị hy sinh ở các lớp hiếm.
> Dermatofibroma và Vascular, dù cộng lại chưa tới hai trăm ảnh huấn luyện,
> vẫn được nhận diện chính xác — kết quả trực tiếp của Mixup kết hợp với
> Weighted Sampler.
>
> Hai, phần nhầm lẫn còn sót lại tập trung ở cụm Melanoma, Nevus, và Benign
> Keratosis — ba lớp thực sự có hình thái tương đồng, khó phân biệt kể cả
> với bác sĩ có kinh nghiệm.
>
> Ba, và quan trọng nhất: không có lớp nào bị bỏ rơi hoàn toàn — đây là lỗi
> điển hình của bài toán phân loại mất cân bằng, và nó không xảy ra ở đây."

**Cách nói:** dùng bút chỉ, lướt nhanh qua đường chéo của ma trận, không
cần đọc từng ô số liệu.

---

## Slide 10 — Ablation (55 s)

> "Câu hỏi tiếp theo: mỗi kỹ thuật em thêm vào có thực sự cần thiết không?
> Để trả lời, em bỏ từng thứ một khỏi baseline EfficientNet-B0 rồi huấn
> luyện lại từ đầu.
>
> Full pipeline đạt Macro F1 không phẩy tám ba một. Bỏ Mixup và CutMix làm
> F1 giảm bốn phẩy năm phần trăm — đây là đòn bẩy lớn nhất, vì chỉ với tám
> mươi mốt ảnh Dermatofibroma, mô hình sẽ lập tức ghi nhớ y nguyên các mẫu
> hiếm nếu thiếu kỹ thuật này.
>
> Bỏ Label Smoothing thì AUC tăng, nhưng Macro F1 lại giảm ba phẩy một phần
> trăm — một sự đánh đổi kinh điển giữa độ hiệu chuẩn xác suất và chất
> lượng quyết định cứng.
>
> Và điều thú vị nhất: bỏ augmentation mạnh thực ra làm Accuracy tăng nhẹ,
> nhưng Macro F1 lại giảm — vì các lớp thiểu số đang bị bỏ rơi. Đây chính
> xác là lý do vì sao em không chọn Accuracy làm tiêu chí chính ngay từ
> đầu.
>
> Kết luận: mọi thành phần đều đóng góp dương cho Macro F1 — không có
> module nào là thừa."

**Cách nói:** câu *"đây chính xác là lý do vì sao em không chọn Accuracy làm
tiêu chí chính"* là phát biểu phương pháp luận quan trọng nhất bài — nói
chậm, rõ, như một lời khẳng định.

---

## Slide 11 — Grad-CAM (55 s)

> "Phần cuối cùng giải quyết đúng vấn đề 'hộp đen' em nêu ở đầu bài. Grad-
> CAM tạo ra một bản đồ nhiệt từ lớp convolution cuối cùng của mô hình, có
> trọng số theo gradient của lớp được dự đoán — nói cách khác, nó chỉ ra
> chính xác những pixel nào đã khiến mô hình đưa ra kết luận đó.
>
> Trên ảnh melanoma mẫu này, cả bốn mô hình đều tập trung đúng vào ổ sắc tố
> trung tâm — không lệch ra nền da xung quanh, không bị đánh lừa bởi lông
> hay bọt gel dính trên da lúc chụp. Và quan trọng, em không dừng lại ở
> đánh giá bằng mắt: đo được cụ thể, hơn sáu mươi phần trăm vùng heatmap
> 'nóng nhất' luôn nằm đúng trong vùng tổn thương, trên cả bốn backbone ở
> tập test — một minh chứng định lượng, không chỉ là cảm quan."

**Cách nói:** nếu hội đồng hỏi "công cụ này có thay thế bác sĩ được không",
trả lời dứt khoát: đây là công cụ hỗ trợ tham khảo, không thay thế chẩn
đoán y khoa.

---

## Slide 12 — Kết luận + Hướng phát triển (40 s)

> "Tóm lại, khóa luận đạt bốn kết quả chính: ensemble đạt tám mươi chín
> phẩy bốn chín phần trăm Accuracy và không phẩy chín tám AUC — tương đương
> mức SOTA cho các phương pháp chỉ dùng ảnh trên ISIC 2018; kết quả này ổn
> định qua nhiều hạt giống ngẫu nhiên với p nhỏ hơn mười mũ trừ bảy; mọi
> thành phần trong ablation đều đóng góp dương, không có gì thừa; và
> Grad-CAM xác nhận bằng số liệu cụ thể rằng mô hình tập trung đúng vào
> tổn thương, không phải nhiễu nền.
>
> Về hướng phát triển, em xác định bốn việc tiếp theo: thích nghi mô hình
> với ảnh chụp bằng điện thoại thường thay vì máy soi da chuyên dụng; dạy
> mô hình biết nhận ra khi gặp ảnh không phải tổn thương da thay vì đoán
> bừa; xây dựng cơ chế học chủ động để bác sĩ có thể sửa lỗi và mô hình học
> tiếp từ đó; và tối ưu hóa để triển khai được trên các thiết bị nhỏ gọn
> hơn."

**Cách nói:** đây là đoạn kết — nói chậm rãi, tự tin, nhìn lần lượt từng
thành viên hội đồng, không cúi nhìn slide.

---

## Slide 13 — Câu hỏi và Thảo luận (20 s)

> "Em xin cảm ơn hội đồng đã lắng nghe. Em sẵn sàng đón nhận các câu hỏi và
> thảo luận thêm."

**Cách nói:** mỉm cười, đứng thẳng, tay thả tự nhiên. Chờ câu hỏi đầu tiên
— đừng vội lấp khoảng lặng.

---

## Kế hoạch dự phòng

- **Nếu chậm (đã qua 7:15 mà mới tới slide 8):** cắt câu in nghiêng ở các
  slide trước, rút slide 6 và 7 xuống còn khoảng 25–30 giây mỗi cái, vẫn
  đảm bảo kết thúc slide 12 trước phút 9:30.
- **Nếu hội đồng ngắt hỏi giữa bài:** trả lời ngắn gọn, ghi nhớ chỗ đang
  dừng, đừng lặp lại từ đầu phần đó.
- **Nếu bị hỏi một con số chính xác mà không nhớ:** đừng đoán bừa — nói
  thẳng "con số chính xác có trên slide/phụ lục, em xin phép chỉ lại" rồi
  trỏ đúng chỗ. Thành thật luôn tốt hơn bịa số.

## Chuẩn bị cho phần Q&A

Mười câu hỏi khả năng cao nhất, kèm gợi ý trả lời, nằm trong
[defense_qa.pdf](defense_qa.pdf). Ưu tiên ôn kỹ:

- Câu 1: Vì sao chọn soft voting thay vì stacking?
- Câu 2: Làm sao biết Grad-CAM chỉ đúng chỗ, không phải ngẫu nhiên?
- Câu 5: Nếu đưa vào một ảnh không thuộc 7 lớp thì sao?
- Câu 6: Có bao nhiêu ca melanoma bị bỏ sót?
- Câu 9: Model có thiên lệch với tông da tối không — đặt trong bối cảnh
  Việt Nam?

Nếu hội đồng hỏi sâu về công thức Mixup/CutMix, Label Smoothing, Weighted
Sampler, hay cấu hình tối ưu hóa (AdamW, cosine annealing, mixed-precision,
gradient clipping, early stopping) — toàn bộ nằm ở **phụ lục A28**, không
còn xuất hiện ở slide chính.

## Hai điều bị cấm

1. **Đừng đọc slide.** Hội đồng đọc nhanh hơn tốc độ em nói — slide chỉ là
   đạo cụ hỗ trợ, không phải kịch bản để đọc.
2. **Đừng nói quá.** Nếu bị hỏi "hệ thống này đã dùng thực tế được chưa?"
   — câu trả lời trung thực là *chưa*, và đây là những gì cần làm thêm.
   Trung thực luôn được đánh giá cao hơn là phóng đại.
