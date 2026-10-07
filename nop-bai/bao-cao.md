# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Dương Hà Đức Anh |
| MSSV | 2A202602977 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/duonghaducanh/K4-L3-DAY21-DuongHaDucAnh-2A202602977-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 2 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 3 | 300 | 0.1 | 3 | 0.7130 | 0.876 |
| 4 | 200 | 0.1 | 5 | 0.7149 | 0.874 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ này đạt f1_score cao nhất (0.7149) nên được ghi vào `params.yaml`. Lần chạy có accuracy cao nhất (bộ số 2, accuracy 0.878) không trùng với lần có f1 cao nhất, cho thấy accuracy không phản ánh đúng chất lượng trên lớp thiểu số. Tăng `n_estimators` từ 100 lên 300 gần như không cải thiện f1 vì mô hình đã bão hòa; yếu tố tạo khác biệt rõ nhất là tăng `max_depth` lên 5, giúp bắt được nhiều tương tác phi tuyến hơn giữa các đặc trưng.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có tỷ lệ lớp dương (thu nhập > 50K) chỉ khoảng 24,8%, tức mất cân bằng lớp rõ rệt. Một mô hình luôn trả lời "thu nhập thấp" đã đạt accuracy khoảng 75% mà không học được gì về lớp cần quan tâm, nên accuracy cao rất dễ gây hiểu nhầm. F1 của lớp dương là trung bình điều hòa giữa precision và recall, chỉ cao khi mô hình vừa dự đoán đúng vừa không bỏ sót người thu nhập cao — điều accuracy không đo được. Không dùng `average="weighted"` vì bị chi phối bởi lớp đa số và gần trùng accuracy; không dùng `"macro"` vì coi hai lớp ngang nhau trong khi mục tiêu chỉ tập trung vào lớp dương.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `pip install` tự dừng giữa chừng | Xung đột phiên bản giữa các gói | Tạo môi trường ảo riêng, ghim phiên bản trong `requirements.txt` |
| `dvc push` lỗi xác thực lên S3 | Remote thiếu region và access key | Khai báo remote `labstore` với `region us-east-1`, nạp credentials qua GitHub Secret |
| Pipeline không chạy sau khi push | Fork chưa bật Actions, commit không khớp `paths` filter | Bật Actions cho fork, commit đúng file `data/*.dvc` để kích hoạt |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.874 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.882 |

**Nhận xét:** Khi gấp đôi dữ liệu, f1 tăng nhẹ 0.0205 và accuracy tăng 0.008. Mức cải thiện nhỏ vì hai batch được chia ngẫu nhiên từ cùng một nguồn nên có cùng phân phối; mô hình đã học gần hết tín hiệu từ 22.361 mẫu đầu. Giá trị thực sự của Bước 3 là quy trình tự động chạy đúng từ commit dữ liệu đến mô hình đang phục vụ.

---

## 5. Phần Bonus Đã Thực Hiện

- [ ] Bonus 1 - DagsHub: không thực hiện.
- [x] Bonus 2 - Ngưỡng quyết định: ngưỡng tối ưu 0.30 đưa f1 lên 0.7368.
- [x] Bonus 3 - Báo cáo precision/recall: ghi ra `outputs/detail.txt`.
- [x] Bonus 4 - Hoàn trả phiên bản trước: hủy upload nếu f1 mới thấp hơn f1 trên S3.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: lệch quá 5 điểm phần trăm so với mốc 24,8%.
