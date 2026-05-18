# Thesis Defense Script - Skin Lesion Classification

## Slide 1: Title Slide
**English:** Good morning, honorable committee members and professors. My name is [Student Name]. Today, I am honored to present my graduation thesis titled: "Deep Learning for Skin Lesion Classification: An Ensemble and Interpretability Approach".
**Vietnamese:** Chào buổi sáng hội đồng đánh giá và các thầy cô. Em tên là [Tên sinh viên]. Hôm nay, em rất vinh dự được trình bày đồ án tốt nghiệp với đề tài: "Phân loại bệnh lý da bằng Deep Learning: Phương pháp tiếp cận kết hợp Ensemble và khả năng giải thích".

## Slide 2: Presentation Outline
**English:** My presentation is divided into five main sections: We will start with the Introduction to the clinical problem, followed by the Dataset challenges. Then, I will explain the proposed Methodology, present the Experiments and Results, and finally conclude with Model Interpretability and a prototype demonstration.
**Vietnamese:** Bài thuyết trình của em gồm 5 phần chính: Đầu tiên là phần giới thiệu về vấn đề lâm sàng, tiếp theo là những thách thức về dữ liệu. Sau đó, em sẽ trình bày phương pháp đề xuất, các thí nghiệm và kết quả, cuối cùng là phần khả năng giải thích của mô hình và demo sản phẩm thực tế.

## Slide 3: Motivation & Problem Statement
**English:** To begin with the motivation, skin cancer is one of the most common malignancies globally. While dermoscopy helps with early detection, it relies heavily on the subjective expertise of dermatologists. When applying AI to this field, we face three major limitations: severe data imbalance where most data is healthy, high visual similarity between different skin diseases, and the "black-box" nature of deep learning which limits doctors' trust.
**Vietnamese:** Bắt đầu với động lực nghiên cứu, ung thư da là một trong những bệnh ác tính phổ biến nhất toàn cầu. Việc soi da giúp phát hiện sớm nhưng lại phụ thuộc nhiều vào chuyên môn chủ quan của bác sĩ. Khi áp dụng AI vào lĩnh vực này, chúng ta gặp 3 hạn chế lớn: mất cân bằng dữ liệu nghiêm trọng khi đa số ảnh là da khỏe, sự tương đồng về hình ảnh giữa các bệnh lý khác nhau, và bản chất "hộp đen" của AI làm giảm đi sự tin tưởng của bác sĩ.

## Slide 4: Project Objectives
**English:** To address these challenges, our project has five key objectives. First, we aim to overcome data imbalance using advanced augmentation techniques like Mixup and CutMix. Second, we fine-tune diverse architectures from CNNs to Vision Transformers and combine them. Third, we conduct a rigorous ablation study. Fourth, we integrate Grad-CAM for visual interpretability. And finally, we deploy these models into a real-time web diagnostic prototype.
**Vietnamese:** Để giải quyết những thách thức này, dự án có 5 mục tiêu chính. Đầu tiên là khắc phục sự mất cân bằng dữ liệu bằng các kĩ thuật tăng cường ảnh như Mixup và CutMix. Thứ hai, tinh chỉnh đa dạng các kiến trúc từ CNN đến Vision Transformer và kết hợp chúng. Thứ ba, thực hiện đánh giá thành phần (ablation study). Thứ tư, tích hợp Grad-CAM để tăng khả năng giải thích bằng hình ảnh. Và cuối cùng, triển khai mô hình lên một web nguyên mẫu chẩn đoán thời gian thực.

## Slide 5: Dataset Overview (ISIC 2018)
**English:** Moving to the dataset, we utilized the ISIC 2018 HAM10000 dataset containing over 10,000 dermoscopic images across 7 diagnostic categories. We stratified the split into 70% for training, 15% for validation, and 15% for testing to ensure the label distribution remains identical across all sets.
**Vietnamese:** Chuyển sang phần dữ liệu, chúng em sử dụng bộ dữ liệu ISIC 2018 HAM10000 chứa hơn 10.000 ảnh soi da của 7 loại bệnh lý. Chúng em chia tập dữ liệu theo tỷ lệ 70% để huấn luyện, 15% xác thực và 15% kiểm thử, đảm bảo tỷ lệ phân bố nhãn bệnh được giữ nguyên trên mọi tập.

## Slide 6: The Challenge of Data Imbalance
**English:** The most critical characteristic of this dataset is its extreme skewness. The ratio between the most frequent class, Nevus, and the least frequent class, Dermatofibroma, is 58 to 1. This means a naive model could simply predict "Nevus" for every image and still achieve 67% accuracy. Therefore, our evaluation strictly prioritizes Macro F1 and Macro AUC metrics over simple accuracy to ensure fair performance across all classes, especially the rare cancers.
**Vietnamese:** Đặc điểm quan trọng nhất của bộ dữ liệu này là sự mất cân bằng cực kỳ lớn. Tỷ lệ giữa lớp phổ biến nhất (nốt ruồi - NV) và lớp hiếm nhất (u xơ da - DF) là 58:1. Điều này có nghĩa là một mô hình ngây ngô chỉ cần đoán "NV" cho mọi ảnh là đã đạt độ chính xác 67%. Do đó, phương pháp đánh giá của chúng em ưu tiên dùng chỉ số Macro F1 và Macro AUC thay vì Accuracy thông thường, nhằm đảm bảo mô hình nhận diện tốt mọi lớp, đặc biệt là các ca ung thư hiếm.

## Slide 7: Methodology Pipeline
**English:** Our proposed methodology consists of two main steps. First, during data preprocessing, images are resized and heavily augmented. Crucially, we use a Weighted Random Sampler to feed balanced mini-batches to the model. Second, we fine-tune four pre-trained models: EfficientNet-B0, ResNet50, DenseNet121, and a Swin-Tiny Vision Transformer.
**Vietnamese:** Phương pháp đề xuất của chúng em gồm 2 bước chính. Đầu tiên, trong bước tiền xử lý, ảnh được đổi kích thước và tăng cường dữ liệu mạnh. Quan trọng nhất, chúng em dùng Weighted Random Sampler để cung cấp các mini-batch cân bằng cho mô hình học. Thứ hai, chúng em tinh chỉnh 4 kiến trúc model: EfficientNet-B0, ResNet50, DenseNet121 và Transformer Swin-Tiny.

## Slide 8: Advanced Training Strategies
**English:** To further combat overfitting on the rare classes, we implemented two advanced regularization techniques. First, Mixup and CutMix, which mathematically blend two images and their labels together, preventing the model from memorizing rare samples. Second, we use Label Smoothing with a factor of 0.1, which softens the hard target probabilities, effectively preventing the network from making overconfident but incorrect predictions.
**Vietnamese:** Để chống lại việc học vẹt ở các lớp hiếm, chúng em tích hợp 2 kỹ thuật nâng cao. Một là Mixup và CutMix, giúp trộn lẫn 2 ảnh và nhãn của chúng lại với nhau, ngăn mô hình "học vẹt" các mẫu hiếm. Hai là Label Smoothing với hệ số 0.1, làm mềm các nhãn mục tiêu, giúp ngăn chặn mạng lưới đưa ra các dự đoán quá tự tin nhưng sai lệch.

## Slide 9: Experimental Setup
**English:** This table summarizes our hyperparameter settings. All models use the AdamW optimizer with a Cosine Annealing learning rate schedule and Mixed Precision training. A notable detail here is that the Swin-Tiny Transformer requires a much lower learning rate and a higher weight decay compared to the CNNs. This is necessary to preserve its pre-trained attention weights on our relatively small medical dataset.
**Vietnamese:** Bảng này tóm tắt các tham số hyperparameter. Tất cả model dùng bộ tối ưu AdamW với lịch trình Cosine Annealing và huấn luyện ở chế độ Mixed Precision. Một điểm đáng chú ý là model Swin-Tiny yêu cầu learning rate thấp hơn và weight decay cao hơn hẳn so với CNN. Điều này là cần thiết để bảo toàn các trọng số attention đã được học trước đó khi áp dụng lên bộ dữ liệu y tế khá nhỏ này.

## Slide 10: Single Model Performance
**English:** Looking at the results for single models, EfficientNet-B0 proved to be the most efficient, achieving the highest hard classification metrics with 88.36% Accuracy and a Macro F1 of 0.83. Interestingly, while Swin-Tiny had the lowest F1 score, it achieved the highest AUC of 0.96. This indicates that the Vision Transformer is superior at probability ranking and calibration, even if its hard threshold decisions are slightly worse.
**Vietnamese:** Nhìn vào kết quả của các mô hình đơn lẻ, EfficientNet-B0 hiệu quả nhất khi đạt Accuracy 88.36% và Macro F1 là 0.83 cao nhất. Thú vị là dù Swin-Tiny có điểm F1 thấp nhất, nó lại đạt AUC cao nhất là 0.96. Điều này chứng tỏ Vision Transformer vượt trội ở khả năng xếp hạng xác suất (calibration), dù cho việc chốt phân loại cứng của nó có thể kém hơn chút ít.

## Slide 11: Ensemble Approach
**English:** To leverage the strengths of all architectures, we implemented a Soft Voting Ensemble. By averaging the probability vectors of the four models, we fuse the local feature extraction capabilities of CNNs with the global context understanding of the Transformer. This ensemble approach achieved state-of-the-art level results: 89.49% Accuracy, 0.84 Macro F1, and an outstanding AUC of 0.98.
**Vietnamese:** Để tận dụng điểm mạnh của mọi kiến trúc, chúng em tạo ra một mô hình kết hợp Ensemble Soft Voting. Bằng cách lấy trung bình xác suất của 4 model, chúng em hợp nhất khả năng trích xuất đặc trưng cục bộ của CNN với tầm nhìn bao quát của Transformer. Mô hình Ensemble này đạt được kết quả ở mức SOTA: Accuracy 89.49%, Macro F1 0.84 và AUC xuất sắc lên tới 0.98.

## Slide 12: Per-Class Analysis
**English:** When we break down the performance per class between the baseline EfficientNet and the Ensemble, we see consistent improvements. Melanoma (MEL) is the most difficult class to detect due to its visual overlap with benign lesions, yet the ensemble improved its F1 score by nearly 5%. Furthermore, the AUC improved across all 7 classes uniformly, confirming the ensemble's robustness.
**Vietnamese:** Khi phân tích hiệu năng trên từng lớp bệnh giữa EfficientNet và Ensemble, chúng ta thấy sự cải thiện đồng đều. Melanoma (MEL) là lớp ung thư khó phát hiện nhất vì rất giống các nốt lành tính, nhưng Ensemble đã tăng được điểm F1 của nó lên gần 5%. Hơn nữa, chỉ số AUC tăng ở toàn bộ 7 lớp, khẳng định sự ổn định của phương pháp Ensemble.

## Slide 13: Ensemble Confusion Matrix
**English:** The ensemble confusion matrix visually confirms our success in handling the data imbalance. The high diagonal values show strong recall across the board. Notably, Dermatofibroma and Vascular lesions, despite being the rarest classes with only around 100 samples each, are classified with high precision, largely thanks to our Mixup and weighted sampling strategies.
**Vietnamese:** Confusion matrix của Ensemble minh chứng rõ ràng cho sự thành công trong việc xử lý mất cân bằng dữ liệu. Đường chéo đậm cho thấy độ nhạy (recall) cao ở tất cả các bệnh. Đáng chú ý là Dermatofibroma và Vascular, dù là 2 bệnh cực hiếm với chỉ khoảng 100 mẫu, vẫn được phân loại rất chuẩn xác, phần lớn nhờ vào chiến lược Mixup và Weighted sampling.

## Slide 14: Ablation Study - Component Analysis
**English:** We then conducted an ablation study, systematically removing components from our baseline pipeline to measure their exact impact. The table shows the performance drops when removing Mixup, Label Smoothing, the Weighted Sampler, or Advanced Augmentation.
**Vietnamese:** Tiếp theo, chúng em thực hiện Ablation study, lần lượt loại bỏ từng kĩ thuật khỏi pipeline gốc để đo lường chính xác tác động của chúng. Bảng này hiển thị mức độ sụt giảm hiệu năng khi ta tháo bỏ Mixup, Label Smoothing, Weighted Sampler hay Augmentation.

## Slide 15: Ablation Study - Key Insights
**English:** Three key insights emerged from this study. First, Mixup and CutMix are absolutely supreme; removing them causes the most severe drop in F1. Second, we observed a label smoothing paradox: removing it increases AUC but drops F1, because without smoothing, the model is more confident but makes harder errors on edge cases. Finally, removing heavy augmentation actually increases global accuracy because the model overfits the easy majority class, but it severely hurts the F1 score for minority classes.
**Vietnamese:** Có 3 bài học lớn được rút ra. Thứ nhất, Mixup/CutMix đóng vai trò tối thượng; tháo chúng ra làm F1 tụt thê thảm nhất. Thứ hai là nghịch lý Label Smoothing: tháo nó ra thì AUC tăng nhưng F1 lại giảm, bởi vì mô hình tự tin hơn nhưng lại sai trầm trọng ở các ca khó ranh giới. Cuối cùng, nếu bỏ Augmentation thì Accuracy tổng lại tăng vì mô hình dễ dàng học vẹt lớp đông nhất, nhưng F1 của các lớp hiếm lại bị sụt giảm nghiêm trọng.

## Slide 16: Model Interpretability: Grad-CAM
**English:** Moving to Explainable AI, we utilized Grad-CAM to visualize the regions our models focus on. As seen in the generated heatmap for this Melanoma sample, the model successfully localizes the lesion geometry and ignores dermoscopic artifacts like skin hair, dark corners, or gel bubbles. This is crucial for building trust with dermatologists.
**Vietnamese:** Chuyển sang phần AI có thể giải thích (Explainable AI), chúng em dùng Grad-CAM để trực quan hóa vùng ảnh mà mô hình đang chú ý. Như trong ảnh heatmap của ca Melanoma này, mô hình đã khoanh vùng chính xác tổn thương và phớt lờ hoàn toàn các nhiễu như lông, viền đen hay bọt gel. Điều này cực kỳ quan trọng để xây dựng niềm tin với bác sĩ.

## Slide 17: Web Diagnostic Prototype
**English:** To make our research practical, we deployed the ensemble pipeline into a real-time Streamlit web application. Doctors can upload an image, receive an instant probability distribution from the ensemble, and dynamically generate Grad-CAM heatmaps from any individual model. This acts as a robust "second opinion" triage system.
**Vietnamese:** Để đưa nghiên cứu vào thực tiễn, chúng em đã triển khai pipeline Ensemble thành một ứng dụng web Streamlit chạy thời gian thực. Bác sĩ có thể tải ảnh lên, nhận ngay dự đoán xác suất từ Ensemble, và xem trực tiếp Grad-CAM của từng mô hình. Ứng dụng này đóng vai trò như một hệ thống sàng lọc hỗ trợ quyết định (second opinion) cực kỳ vững chắc.

## Slide 18: Conclusion
**English:** In conclusion, we successfully developed a high-performing ensemble pipeline capable of handling severely imbalanced medical data, achieving an AUC of 0.98. Our ablation analysis proved that data regularization techniques like Mixup are far more critical than simply scaling up model size. Finally, we bridged the gap between black-box AI and clinical utility through Grad-CAM and a functional web prototype.
**Vietnamese:** Tóm lại, chúng em đã phát triển thành công một hệ thống Ensemble hiệu năng cao có khả năng xử lý dữ liệu y tế mất cân bằng, đạt AUC 0.98. Phân tích Ablation chứng minh rằng các kĩ thuật làm sạch và tăng cường dữ liệu (như Mixup) quan trọng hơn rất nhiều so với việc chỉ tăng kích thước mô hình. Cuối cùng, dự án đã thu hẹp khoảng cách giữa AI "hộp đen" và ứng dụng lâm sàng thông qua Grad-CAM và Web prototype.

## Slide 19: Future Work
**English:** For future directions, we plan to integrate multimodal patient metadata such as age and anatomical location to further boost detection rates. We also aim to validate our models on external, out-of-distribution clinical datasets and implement a continuous active learning loop where doctors can correct misclassifications to improve the model over time.
**Vietnamese:** Về hướng phát triển tương lai, chúng em dự định tích hợp thêm thông tin đa phương thức của bệnh nhân như tuổi, vị trí trên cơ thể để tăng độ chính xác. Đồng thời, mô hình cần được kiểm chứng trên các bộ dữ liệu bệnh viện bên ngoài (out-of-distribution), và xây dựng một cơ chế học tập liên tục để bác sĩ có thể sửa sai và dạy lại mô hình theo thời gian.

## Slide 20: Thank You
**English:** That concludes my presentation. Thank you very much for your attention. I am now open to any questions from the committee. (And if the setup allows, I would be happy to demonstrate the live diagnostic prototype).
**Vietnamese:** Bài thuyết trình của em đến đây là kết thúc. Cảm ơn hội đồng đã lắng nghe. Em xin phép được trả lời các câu hỏi từ hội đồng. (Và nếu được phép, em rất sẵn lòng demo trực tiếp ứng dụng phân loại ngay bây giờ).
