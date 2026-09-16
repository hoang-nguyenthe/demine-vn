# DeMine-VN

**Hệ thống lập bản đồ nguy cơ và xếp thứ tự ưu tiên rà phá bom mìn, vật nổ còn sót
lại sau chiến tranh, trên cơ sở hợp nhất hồ sơ không kích giải mật, ảnh vệ tinh lịch
sử và dữ liệu rà phá thực địa.**

Dự án nghiên cứu và bản trình diễn


> **Nguyên tắc an toàn bắt buộc.** Hệ thống chỉ xếp thứ tự ưu tiên rà phá. Hệ thống
> **không bao giờ** tuyên bố một khu đất là an toàn. Mọi khu đất, kể cả khi được mô
> hình chấm mức nguy cơ thấp nhất, vẫn phải được rà phá đầy đủ theo đúng quy trình
> kỹ thuật hiện hành trước khi đưa vào sử dụng.

---

## 1. Vấn đề

Chiến tranh tại Việt Nam kết thúc năm 1975. Năm mươi năm sau, hậu quả vật lý của nó
vẫn nằm nguyên trong lòng đất trên phạm vi toàn quốc.

Theo số liệu của Trung tâm Hành động bom mìn Quốc gia Việt Nam:

- **Gần 6,1 triệu héc ta** bị ô nhiễm hoặc nghi ngờ ô nhiễm bom mìn, vật nổ —
  **chiếm 18,71% tổng diện tích cả nước**.
- **63 trên 63 tỉnh, thành phố** đều được xác định có ô nhiễm.
- Trong 11.134 xã, phường được khảo sát, **9.116 đơn vị vẫn còn bị ô nhiễm**.
- Khối lượng bom đạn còn sót lại ước tính khoảng **800.000 tấn**.

Con số quan trọng hơn cả là thương vong. **Kể từ sau năm 1975 — nghĩa là sau khi
chiến tranh đã chấm dứt — hơn 40.000 người đã thiệt mạng và hơn 60.000 người bị
thương.** Giai đoạn 2013–2018 vẫn còn 1.813 vụ tai nạn với gần một nghìn người tử
vong. Với năng lực rà phá hiện tại, việc làm sạch được ước tính cần thêm **hàng trăm
năm**.

Phần lớn nạn nhân không phải người làm nhiệm vụ quân sự, mà là người dân lao động
trong sinh hoạt thường nhật: người canh tác trên chính thửa ruộng của gia đình, người
khai hoang đất sản xuất, và trẻ em trong độ tuổi đi học thu nhặt phế liệu kim loại.

**Vì sao đây là bài toán phát triển kinh tế – xã hội.** Gần một phần năm diện tích đất
đai của đất nước không thể đưa vào canh tác, xây dựng hạ tầng hay quy hoạch dân cư một
cách an toàn. Bản đồ các địa phương ô nhiễm nặng nhất trùng lặp đáng kể với bản đồ các
địa phương có tỉ lệ hộ nghèo cao nhất — không phải ngẫu nhiên, vì đất không sử dụng
được thì sinh kế không hình thành được. Đẩy nhanh tiến độ rà phá là điều kiện tiên
quyết để giải phóng nguồn lực đất đai phục vụ phát triển.

---

## 2. Vai trò của trí tuệ nhân tạo

Quy trình rà phá gồm hai bước: khảo sát phi kỹ thuật để khoanh vùng, sau đó rà phá kỹ
thuật bằng thiết bị dò trên từng mét vuông. Bước thứ hai tiêu tốn gần như toàn bộ chi
phí và thời gian, và **không thể rút ngắn bằng công nghệ thông tin** — vẫn phải có
người cầm máy đi qua từng mét đất.

Điểm nghẽn nằm ở bước thứ nhất. Khi thông tin khoanh vùng còn thô, lực lượng rà phá
buộc phải quét trải đều, nghĩa là phần lớn công sức được dồn vào những khoảnh đất vốn
không chứa vật nổ.

Từ đó xác định rõ vai trò của hệ thống này:

> Hệ thống **không tìm ra vật nổ** — đó là việc của thiết bị dò và của con người tại
> thực địa. Hệ thống **xếp thứ tự ưu tiên**: chỉ ra trong hàng triệu héc ta nghi ngờ,
> những khoảnh đất nào có xác suất chứa vật nổ cao nhất để rà phá trước.

Nếu mô hình xác định được phần diện tích nhỏ chứa phần lớn nguy cơ, thì với cùng ngân
sách và cùng số đội rà phá, số vật nổ được xử lý sớm — và do đó số thương vong được
ngăn chặn — sẽ tăng lên nhiều lần.

---

## 3. Kiến trúc bốn tầng

| Tầng | Chức năng | Đầu ra | Mã nguồn |
|---|---|---|---|
| **Một** | Chuẩn hoá hồ sơ không kích giải mật về lưới toạ độ | Tải trọng bom, mật độ phi vụ theo từng ô | `data/sorties.py`, `data/geo.py` |
| **Hai** | Phát hiện hố bom trên ảnh vệ tinh trinh sát lịch sử | Toạ độ và đường kính từng hố bom | `detect/` |
| **Ba** | Mô hình nguy cơ hợp nhất, có hiệu chỉnh | Xác suất còn tồn tại vật nổ theo ô | `risk/` |
| **Bốn** | Xếp thứ tự ưu tiên và trình bày | Danh mục rà phá, bản đồ nguy cơ | `viz/`, `reporting.py` |

### Cơ chế vật lý mà hệ thống khai thác

Đây là phần quyết định giá trị khoa học của đề tài, nên được nêu rõ:

1. **Hồ sơ ghi chép lệch so với điểm rơi thật.** Toạ độ trong hồ sơ thời chiến được
   ghi theo lưới bản đồ quân sự và làm tròn; một phần hồ sơ đã thất lạc. Hồ sơ là chỉ
   báo có nhiễu, không phải chân lý. Hệ thống mô hình hoá sai số này tường minh thành
   một hạt nhân lan toả không gian.

2. **Nền đất mềm làm tăng tỉ lệ bom không nổ.** Đầu nổ chạm nổ cần lực cản đủ lớn mới
   kích hoạt; bom cắm sâu vào nền mềm thường không kích nổ.

3. **Hố bom chỉ hình thành ở nơi bom đã nổ, và trên nền mềm thì bị bồi lấp nhanh.**
   Nghĩa là nơi nhiều vật nổ còn sót nhất lại chính là nơi ít hố bom quan sát được
   nhất.

Điểm thứ ba là mấu chốt: nó khiến bài toán **không thể giải bằng một nguồn dữ liệu
duy nhất**. Hệ thống xử lý bằng hai đặc trưng có cơ sở vật lý trực tiếp:

- **kỳ vọng bom không nổ** = tải trọng bom × tỉ lệ không nổ theo độ mềm nền đất;
- **hố bom đã hiệu chỉnh tầm nhìn** = mật độ hố bom ÷ xác suất còn quan sát được.

Cả hai đều được suy ra từ cơ chế vật lý, không phải từ thử nghiệm mò.

---

## 4. Kiểm chứng

Không thể đào toàn bộ diện tích lên để kiểm tra. Đề tài kiểm chứng ở **bốn tầng độc
lập**, mỗi tầng bù cho điểm yếu của tầng khác. Chi tiết đầy đủ ở
**[`docs/VALIDATION.md`](docs/VALIDATION.md)**.

| Tầng | Nguồn đối chứng | Điểm mạnh | Điểm yếu được tầng khác bù |
|---|---|---|---|
| 1 | Đất đã rà phá | Nhãn thật, đầy đủ | Không phải mẫu ngẫu nhiên |
| 2 | Hồ sơ tai nạn | **Độc lập hoàn toàn** | Không đầy đủ |
| 3 | Nhất quán hai nguồn | Không cần nhãn nào | Chỉ gián tiếp |
| 4 | Chuyển vùng, hiệu chỉnh | Đo khả năng tổng quát hoá | — |

**Chi tiết phương pháp quan trọng nhất:** chia tập **theo khối không gian**, tuyệt
đối không chia ngẫu nhiên theo điểm. Phân bố vật nổ có tương quan không gian rất mạnh;
chia ngẫu nhiên sẽ đặt điểm huấn luyện và điểm kiểm tra cạnh nhau, và mô hình chỉ cần
nội suy từ hàng xóm là đạt chỉ tiêu cao mà không học được quy luật nào. Kiểm thử số 5
trong `tests/test_smoke.py` xác nhận rằng hai cách chia cho kết quả khác nhau rõ rệt —
nếu chúng bằng nhau thì việc chia khối đã hỏng.

**Chỉ tiêu chính — đường cong hiệu quả rà phá.** Nếu rà phá theo đúng thứ tự mô hình
đề xuất, sau khi xử lý 20% diện tích thì đã thu hồi được bao nhiêu phần trăm tổng số
vật nổ. Đường chéo là phương án quét trải đều (20% diện tích cho 20% vật nổ). Đây là
đại lượng duy nhất chuyển thẳng được thành ý nghĩa thực tiễn.

---

## 5. Kết quả đã đo được

Toàn bộ hệ thống đã được chạy hoàn chỉnh trên vùng nghiên cứu mô phỏng rộng 22 × 18 km
(39.600 ô lưới, 20.430 điểm rơi, 3.536 vật nổ còn sót, 420 ảnh vệ tinh lịch sử với
7.398 hố bom có nhãn). Huấn luyện trên Apple M1 Pro dùng Metal, 40 chu kỳ, tổng thời
gian chạy toàn tuyến khoảng 14 phút.

| Chỉ tiêu | Mục tiêu | Kết quả |
|---|---|---|
| mAP@0.5 phát hiện hố bom | ≥ 0,70 | **0,987** |
| Recall phát hiện hố bom | ≥ 0,80 | **0,987** |
| Precision phát hiện hố bom | — | 0,981 |
| Thu hồi vật nổ tại 20% diện tích | ≥ 0,35 | **0,474** — gấp 2,4 lần quét trải đều |
| Thu hồi tại 10% / 50% diện tích | — | 0,379 / 0,740 |
| Bao phủ tai nạn tại 20% diện tích | ≥ 0,25 | **0,264**; **0,328** khi chia theo thời gian |
| Spearman giữa hai nguồn độc lập | ≥ 0,20 | **0,326** trên 19.438 ô có ảnh phủ tới |
| Suy giảm khi chuyển vùng địa lý | ≤ 15% | **không suy giảm** |
| Sai số hiệu chỉnh kỳ vọng | ≤ 0,05 | **0,028** |

### Phân tích đóng góp thành phần

| Cấu hình | Thu hồi tại 20% diện tích |
|---|---|
| A — chỉ hồ sơ không kích | 0,439 |
| B — chỉ hố bom từ ảnh vệ tinh | 0,333 |
| C — hợp nhất hai nguồn | 0,444 |
| **D — hệ thống đầy đủ** | **0,474** |

Hợp nhất vượt trội từng nguồn riêng lẻ, xác nhận ba nguồn bổ sung cho nhau chứ không
trùng lặp. Trong bảng đóng góp đặc trưng, hai vị trí cao nhất sau tải trọng bom ghi
nhận chính là hai đặc trưng suy ra từ cơ chế vật lý — *kỳ vọng bom không nổ* và *hố bom
đã hiệu chỉnh tầm nhìn*.

### Một lưu ý về giới hạn

Thí nghiệm cho thấy nếu biết chính xác vị trí mọi điểm rơi thì chỉ tiêu thu hồi đạt
khoảng 0,84; khi chỉ có hồ sơ với sai số định vị thực tế và một phần tư liệu đã thất
lạc, trần tụt xuống khoảng 0,50. Khoảng cách đó là lượng thông tin đã mất vĩnh viễn
cùng tư liệu lịch sử, không phải chỗ để cải tiến mô hình. Con số 0,474 đạt được nằm sát
trần đó.

---

## 6. Về bộ dữ liệu mô phỏng

Hệ thống chạy trên bộ dữ liệu mô phỏng do đội thi tự xây dựng. Lý do **không phải** vì
thiếu dữ liệu thật — hồ sơ không kích giải mật (THOR) và ảnh vệ tinh giải mật (CORONA,
HEXAGON) đều công khai và miễn phí.

Lý do thật sự sâu hơn: **dữ liệu thật không có nhãn đối chứng**. Ngoài thực địa không
ai biết chắc dưới một thửa ruộng chưa rà phá có vật nổ hay không, nên không thể đo
được mô hình đúng bao nhiêu phần trăm. Bộ mô phỏng tái hiện đúng chuỗi nhân quả vật lý
và cung cấp nhãn chính xác, nhờ đó toàn bộ khung kiểm chứng bốn tầng chạy và đo được
ngay.

Khi có dữ liệu thật, chỉ cần thay tầng dữ liệu theo
[`docs/ADAPT_NEW_DATA.md`](docs/ADAPT_NEW_DATA.md); ba tầng còn lại không phải sửa.

---

## 7. Chạy như thế nào

Xem [`QUICKSTART_KAGGLE.md`](QUICKSTART_KAGGLE.md) để có hướng dẫn đầy đủ.

Cách nhanh nhất: tải `notebooks/DeMine_VN_Kaggle.ipynb` lên Kaggle, chọn **GPU T4 ×
2**, bấm **Run All**. Notebook tự ghi ra toàn bộ mã nguồn khi chạy.

```bash
python scripts/run_pipeline.py            # chạy đầy đủ
python scripts/run_pipeline.py --quick    # chạy thử nhanh
python tests/test_smoke.py                # kiểm thử trên CPU, khoảng một phút
```

---

## 8. Cấu trúc kho mã

```
demine-vn/
├── src/demine/
│   ├── config.py           tham số, mỗi hằng số có căn cứ nêu tại chỗ
│   ├── pipeline.py         điều phối mười bước
│   ├── reporting.py        báo cáo và danh mục ưu tiên
│   ├── data/               tầng một — địa hình, phi vụ, ảnh, rà phá, tai nạn
│   ├── detect/             tầng hai — hai phương án phát hiện hố bom
│   ├── risk/               tầng ba — đặc trưng, học từ quan sát dương, mô hình
│   ├── evaluation/         khung kiểm chứng bốn tầng
│   └── viz/                tầng bốn — hình minh hoạ và bản đồ
├── scripts/                chạy quy trình, huấn luyện, dựng notebook, nhật ký câu lệnh
├── docs/                   kiểm chứng, thích ứng dữ liệu mới, nhật ký câu lệnh
├── tests/test_smoke.py     27 kiểm thử, chạy trên CPU
├── notebooks/              notebook Kaggle tự chứa
├── KE_KHAI_CONG_CU_AI.md   bản kê khai công cụ AI đã sử dụng
└── QUICKSTART_KAGGLE.md    hướng dẫn chạy
```

---

## 9. Nguồn dữ liệu thật

| Nguồn | Nội dung | Địa chỉ |
|---|---|---|
| THOR | Hồ sơ không kích giải mật, toạ độ từng phi vụ | <https://data.world/datamil/vietnam-war-thor-data> |
| USGS EarthExplorer | Ảnh vệ tinh trinh sát giải mật CORONA, HEXAGON | <https://earthexplorer.usgs.gov> |
| Copernicus | Sentinel-2 hiện trạng, mô hình số độ cao | <https://dataspace.copernicus.eu> |
| VNMAC | Số liệu ô nhiễm, dữ liệu rà phá, hồ sơ tai nạn | <https://vnmac.gov.vn> |
| IMAS | Chuẩn quốc tế về hành động bom mìn | <https://www.mineactionstandards.org> |

Dữ liệu rà phá và hồ sơ tai nạn thuộc quyền quản lý của cơ quan chuyên môn; mọi hoạt
động sử dụng phải thông qua đề nghị chính thức. Hồ sơ tai nạn được xử lý ở dạng đã ẩn
danh hoàn toàn: chỉ giữ toạ độ, thời điểm và hoàn cảnh, không thu thập hay lưu trữ bất
kỳ thông tin định danh cá nhân nào.

---

## 10. Hồ sơ dự thi

Hai tệp liên quan trực tiếp đến yêu cầu của Ban Tổ chức:

- **`KE_KHAI_CONG_CU_AI.md`** — bản kê khai công cụ AI, bộ dữ liệu, thư viện theo Điều
  5 khoản 5. Phần xác định được từ chính mã nguồn đã điền sẵn; các ô `[CẦN ĐIỀN]` đội
  thi phải hoàn thiện trung thực.
- **`docs/PROMPT_LOG.md`** và `scripts/promptlog.py` — quản lý nhật ký câu lệnh. Chạy
  `python scripts/promptlog.py check` để biết hồ sơ còn thiếu gì trước khi nộp.

Nhật ký câu lệnh phải được ghi **ngay trong khi làm**, không dựng lại về sau. Kê khai
không trung thực là hành vi vi phạm nghiêm trọng quy trình phát triển của dự án.
