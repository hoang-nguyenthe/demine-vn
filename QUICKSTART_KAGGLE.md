# Hướng dẫn chạy nhanh trên Kaggle

Tài liệu này mô tả đúng các thao tác cần làm để chạy toàn bộ hệ thống DeMine-VN trên
Kaggle với hai card đồ hoạ Tesla T4. Tổng thời gian từ lúc bắt đầu đến khi có đầy đủ
kết quả là khoảng **30 đến 45 phút** ở chế độ đầy đủ, hoặc **6 đến 10 phút** ở chế độ
rút gọn.

---

## Cách nhanh nhất: dùng notebook tự chứa

Notebook `notebooks/DeMine_VN_Kaggle.ipynb` **tự ghi ra toàn bộ mã nguồn** khi chạy,
nên không cần tải kho mã lên Kaggle dưới dạng bộ dữ liệu. Chỉ cần tải lên đúng một
tệp notebook rồi bấm chạy.

### Bước 1 — Tạo notebook mới

Vào <https://www.kaggle.com/code>, chọn **New Notebook**, sau đó chọn
**File → Import Notebook** và tải lên tệp `DeMine_VN_Kaggle.ipynb`.

### Bước 2 — Bật hai GPU T4

Mở bảng điều khiển bên phải, mục **Session options**:

| Thiết lập | Giá trị cần chọn |
|---|---|
| Accelerator | **GPU T4 × 2** |
| Internet | Bật nếu được (không bắt buộc) |
| Persistence | Không cần |
| Environment | Để mặc định |

Sau khi đổi Accelerator, Kaggle sẽ khởi động lại phiên. Đợi đến khi báo sẵn sàng rồi
mới chạy.

### Bước 3 — Chọn Run All

Chọn **Run All** ở thanh trên cùng. Các ô chạy tuần tự từ đầu đến cuối, không cần
can thiệp gì thêm.

Nếu muốn chạy thử nhanh trước, mở ô ở Bước 3 và đổi `QUICK = False` thành
`QUICK = True`. Chế độ rút gọn giảm số phi vụ, số ảnh và số chu kỳ huấn luyện; toàn
bộ đường ống vẫn chạy trọn vẹn nhưng các chỉ tiêu sẽ thấp hơn.

---

## Nếu Internet bị tắt

Hệ thống được thiết kế để chạy trọn vẹn cả khi không có mạng:

- Tầng phát hiện tự chuyển sang **Faster R-CNN của Torchvision**, vốn luôn có sẵn
  trong ảnh Python của Kaggle và không cần tải trọng số từ Internet.
- Nếu có Ultralytics nhưng không tải được trọng số huấn luyện trước, hệ thống khởi
  tạo mô hình từ tệp mô tả kiến trúc. Huấn luyện vẫn diễn ra bình thường, chỉ cần
  thêm một số chu kỳ để hội tụ.
- Tầng nguy cơ tự lùi từ XGBoost về `HistGradientBoosting` của scikit-learn.
- Bản đồ tự lùi từ Folium về một trang web tự vẽ bằng canvas.

Không bước nào bị bỏ qua. Các chỉ tiêu có thể thấp hơn đôi chút, và báo cáo ghi rõ
phương án nào đã được dùng.

---

## Cách thứ hai: tải cả kho mã lên

Nếu muốn giữ mã nguồn dưới dạng bộ dữ liệu để dùng lại nhiều lần:

1. Nén thư mục `demine-vn` thành tệp zip và tải lên Kaggle ở mục **Datasets**.
2. Trong notebook, gắn bộ dữ liệu đó vào phiên.
3. Chạy:

```python
import sys, shutil, pathlib
src = pathlib.Path('/kaggle/input/<ten-bo-du-lieu>/demine-vn')
dst = pathlib.Path('/kaggle/working/demine-vn')
shutil.copytree(src, dst, dirs_exist_ok=True)
sys.path.insert(0, str(dst / 'src'))
%cd /kaggle/working/demine-vn
```

4. Rồi chạy toàn bộ quy trình bằng một dòng:

```python
!python scripts/run_pipeline.py
```

---

## Chạy từ dòng lệnh

```bash
# chạy đầy đủ
python scripts/run_pipeline.py

# chạy thử nhanh
python scripts/run_pipeline.py --quick

# chỉ huấn luyện tầng phát hiện, phân tán trên hai GPU
python scripts/train_detector.py --data data/imagery/data.yaml --epochs 40 --devices 0,1

# kiểm thử nhanh trên CPU, khoảng một phút
python tests/test_smoke.py
```

Các tham số thường dùng của `run_pipeline.py`:

| Tham số | Ý nghĩa |
|---|---|
| `--quick` | Chế độ rút gọn |
| `--epochs N` | Số chu kỳ huấn luyện tầng phát hiện |
| `--missions N` | Số phi vụ mô phỏng |
| `--tiles N` | Số ảnh con |
| `--backend` | `auto`, `ultralytics` hoặc `torchvision` |
| `--seed N` | Hạt giống ngẫu nhiên, để tái lập kết quả |
| `--no-train-detector` | Dùng lại trọng số đã huấn luyện |

---

## Kết quả nằm ở đâu

Sau khi chạy xong, thư mục `outputs/` chứa:

| Tệp | Nội dung |
|---|---|
| `bao_cao_tong_hop.md` | Báo cáo đầy đủ, đọc được ngay trong notebook |
| `ket_qua.json` | Toàn bộ chỉ tiêu ở dạng máy đọc được |
| `ban_do_uu_tien.html` | Bản đồ nguy cơ tương tác |
| `danh_muc_uu_tien_ra_pha.csv` | Danh mục khoảnh đất xếp theo thứ tự rà phá |
| `figures/` | Bảy hình minh hoạ |
| `runs/` | Trọng số mô hình phát hiện |

Hình quan trọng nhất là `figures/01_duong_cong_hieu_qua_ra_pha.png` — đường cong hiệu
quả rà phá. Đây là hình nên đưa lên trước tiên khi trình bày trước hội đồng.

---

## Khắc phục sự cố

**Nhân bị treo khi huấn luyện.** Huấn luyện phân tán được gọi qua một tiến trình con
độc lập chính vì lý do này. Nếu vẫn treo, đổi `--devices 0` để chỉ dùng một GPU.

**Hết bộ nhớ GPU.** Giảm `--batch` xuống 8, hoặc giảm `--imgsz` xuống 512.

**Tầng hai báo phát hiện được rất ít hố bom.** Đây là dấu hiệu huấn luyện chưa đủ,
không phải kết luận khoa học, và hệ thống sẽ ghi cảnh báo rõ ràng trong nhật ký cùng
trong `ket_qua.json`. Tăng số chu kỳ huấn luyện, hoặc kiểm tra xem có thực sự đang
chạy trên GPU hay không.

**Chạy trên CPU rất chậm.** Đúng như vậy — tầng phát hiện cần GPU. Để kiểm tra mã
nguồn trên CPU, hãy dùng `python tests/test_smoke.py`, chạy xong trong khoảng một
phút và kiểm tra toàn bộ phần phương pháp mà không cần huấn luyện mô hình thị giác.

---

> **Nguyên tắc an toàn bắt buộc.** Hệ thống chỉ xếp thứ tự ưu tiên rà phá. Hệ thống
> **không bao giờ** tuyên bố một khu đất là an toàn. Mọi khu đất, kể cả khi được chấm
> mức nguy cơ thấp nhất, vẫn phải được rà phá đầy đủ theo đúng quy trình kỹ thuật
> hiện hành trước khi đưa vào sử dụng.
