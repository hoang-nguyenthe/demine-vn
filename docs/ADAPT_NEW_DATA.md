# Thích ứng với dữ liệu mới

Tài liệu này dành cho hai tình huống: khi đội thi có được dữ liệu thật, và khi Ban Tổ
chức cung cấp dữ liệu thô tại vòng khu vực — vốn là một **hackathon hai ngày**, nên
thời gian để hiểu và ghép dữ liệu là rất ngắn.

Nguyên tắc thiết kế xuyên suốt: **chỉ tầng dữ liệu phụ thuộc vào nguồn cụ thể**. Ba
tầng còn lại làm việc với ba hợp đồng dữ liệu cố định. Tạo đúng ba hợp đồng đó là mọi
thứ phía sau chạy không phải sửa.

---

## Ba hợp đồng dữ liệu

### Hợp đồng 1 — Ảnh và nhãn hố bom, theo quy ước YOLO

```
<root>/images/{train,val,test}/<ten>.png
<root>/labels/{train,val,test}/<ten>.txt
<root>/tiles.jsonl
<root>/data.yaml
```

Mỗi dòng trong tệp nhãn: `0 <x_tâm> <y_tâm> <rộng> <cao>`, tất cả chuẩn hoá về khoảng
`[0, 1]` theo kích thước ảnh. Lớp `0` là hố bom.

Mỗi dòng trong `tiles.jsonl` mô tả một ảnh con và là thứ cho phép quy kết quả phát
hiện về lưới toạ độ:

```json
{"tile_id": 7, "name": "tile_00007", "split": "train",
 "origin_x_m": 4200.0, "origin_y_m": 1560.0,
 "gsd_m": 1.2, "size_px": 640, "n_craters": 31}
```

`origin_x_m` và `origin_y_m` là toạ độ góc dưới bên trái của ảnh trong hệ phẳng của
vùng nghiên cứu; `gsd_m` là kích thước điểm ảnh trên mặt đất.

> **Chú ý về trục toạ độ.** Trục dọc của ảnh hướng xuống, còn trục bắc của mặt đất
> hướng lên. Phép quy đổi trong `data/dataset.py` đã tính đến điều này. Nếu dữ liệu
> mới dùng quy ước khác, phải sửa đúng một chỗ đó.

### Hợp đồng 2 — Hồ sơ không kích

Hệ thống cần bốn mảng cùng độ dài, một phần tử cho mỗi bản ghi:

| Mảng | Ý nghĩa |
|---|---|
| `record_x`, `record_y` | Toạ độ bản ghi trong hệ phẳng, đơn vị mét |
| `record_n_bombs` | Số lượng vũ khí của bản ghi đó |
| `record_mission_id` | Mã phi vụ, dùng để gộp |

Khi dùng bộ THOR thật, các cột tương ứng là toạ độ mục tiêu, số lượng và loại vũ khí.
Cần chuyển toạ độ về hệ phẳng của vùng nghiên cứu và quy đổi số lượng sang tải trọng.

**Việc bắt buộc phải làm:** ước lượng lại **sai số định vị** của nguồn hồ sơ mới và
đặt vào `RiskConfig.record_kernel_radius_m`. Đây không phải tham số tinh chỉnh cho
đẹp chỉ tiêu — nó là một đại lượng vật lý của chính bộ dữ liệu, và đặt sai sẽ làm hỏng
toàn bộ tầng ba.

### Hợp đồng 3 — Dữ liệu rà phá và hồ sơ tai nạn

Rà phá, theo từng ô lưới:

| Mảng | Ý nghĩa |
|---|---|
| `is_cleared` | Ô đã được rà phá hoàn toàn hay chưa |
| `items_found` | Số vật nổ thu hồi được trên ô đó |

Tai nạn, mỗi phần tử một vụ:

| Mảng | Ý nghĩa |
|---|---|
| `col`, `row` | Chỉ số ô lưới nơi xảy ra |
| `year` | Năm xảy ra, dùng cho phép chia theo thời gian |

Dữ liệu thật thường theo chuẩn quản lý thông tin hành động bom mìn, xuất ra được dưới
dạng bảng hoặc tệp không gian địa lý.

---

## Công cụ khảo sát dữ liệu lạ

```bash
python scripts/inspect_dataset.py --path /kaggle/input/<bộ-dữ-liệu>
```

Tệp lệnh này liệt kê cấu trúc thư mục, nhận dạng định dạng, đếm số tệp theo đuôi, đọc
thử vài mẫu và **tự động báo xem dữ liệu đã khớp hợp đồng nào chưa**. Nên chạy đầu
tiên, trước khi viết bất kỳ dòng mã chuyển đổi nào.

---

## Kế hoạch hai ngày tại vòng khu vực

Đây là kế hoạch đã chuẩn bị sẵn để không phải nghĩ lại trong lúc căng thẳng.

### Ngày thứ nhất

| Thời lượng | Việc |
|---|---|
| 30 phút đầu | Chạy `inspect_dataset.py`. Không viết mã chuyển đổi trước khi hiểu dữ liệu. |
| 1 giờ tiếp | Xác định dữ liệu có những nguồn nào trong ba hợp đồng. Thường sẽ thiếu ít nhất một. |
| 2 giờ | Viết mã chuyển đổi cho các nguồn có sẵn, ghi ra đúng ba hợp đồng. |
| 1 giờ | Chạy `run_pipeline.py --quick` để kiểm tra đường ống thông suốt. |
| Còn lại | Chạy đầy đủ, xem kết quả, ghi nhận vấn đề. |

### Ngày thứ hai

| Thời lượng | Việc |
|---|---|
| Buổi sáng | Tinh chỉnh tham số vật lý cho dữ liệu mới: bán kính hạt nhân, tỉ lệ bom không nổ, kích thước ô lưới. |
| Đầu giờ chiều | Chạy đầy đủ khung kiểm chứng bốn tầng. Không bỏ tầng nào. |
| Giữa chiều | Dựng hình, bản đồ, báo cáo. |
| Cuối ngày | Diễn tập trình bày. Chuẩn bị sẵn câu trả lời cho câu hỏi về kiểm chứng. |

---

## Nếu thiếu một nguồn dữ liệu

Hệ thống vẫn chạy được, và đây là điều quan trọng cần biết trước khi vào vòng thi.

**Thiếu ảnh vệ tinh.** Bỏ qua tầng hai, truyền vào một lớp hố bom toàn số không. Tầng
ba tự làm việc với nhóm đặc trưng còn lại. Kết quả sẽ tương đương cấu hình A trong
phân tích đóng góp thành phần — thấp hơn nhưng vẫn tốt hơn quét trải đều rõ rệt.

**Thiếu hồ sơ không kích.** Tương tự, để trống nhóm đặc trưng lịch sử. Tương đương cấu
hình B.

**Thiếu dữ liệu rà phá — nghĩa là không có nhãn nào.** Đây là trường hợp khó nhất.
Khi đó không huấn luyện được mô hình có giám sát. Phương án: dùng công thức vật lý
trực tiếp làm điểm nguy cơ, tức là chính đặc trưng **kỳ vọng bom không nổ**, và trình
bày rõ rằng đây là mô hình vật lý chứ không phải mô hình học máy. Trung thực về điều
này quan trọng hơn là cố tỏ ra có mô hình.

**Thiếu hồ sơ tai nạn.** Mất tầng kiểm chứng số 2 — tầng độc lập và thuyết phục nhất.
Phải nêu rõ hạn chế này trong báo cáo thay vì im lặng.

---

## Những chỗ dễ sai nhất

1. **Chia tập ngẫu nhiên theo điểm.** Sẽ cho chỉ tiêu rất đẹp và hoàn toàn vô nghĩa.
   Luôn dùng `make_spatial_blocks` và `block_kfold`.

2. **So sánh hố bom với hồ sơ không kích chưa lan toả.** Hồ sơ ghi một điểm cho mỗi
   phi vụ, hố bom rải trên nhiều ô. So thẳng sẽ ra tương quan âm — đó là lỗi phương
   pháp, không phải phát hiện.

3. **Quên hiệu chỉnh thiên lệch chọn mẫu.** Đất đã rà phá không phải mẫu ngẫu nhiên.
   Bỏ qua bước này thì mô hình chỉ học lại phán đoán của người đi trước.

4. **Nhầm trục toạ độ ảnh và mặt đất.** Kiểm tra bằng cách chồng hố bom phát hiện được
   lên bản đồ và xem chúng có rơi vào đúng hành lang không kích hay không.

5. **Công bố chỉ tiêu mà không nói rõ tầng hai đã hoạt động đến mức nào.** Nếu mô hình
   phát hiện gần như không tìm được hố bom nào, nhóm đặc trưng ảnh vệ tinh sẽ toàn số
   không và mọi kết luận về nguồn đó đều vô nghĩa. Hệ thống tự ghi cảnh báo trong
   `ket_qua.json`, nhưng người trình bày phải chủ động nêu ra.
