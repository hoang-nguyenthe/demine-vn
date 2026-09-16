# Phương pháp kiểm chứng

Tài liệu này trình bày đầy đủ cách đề tài chứng minh mô hình có giá trị, và vì sao
từng lựa chọn phương pháp lại cần thiết. Đây là phần hội đồng sẽ chất vấn kỹ nhất,
nên nó được viết để đọc độc lập với phần còn lại của mã nguồn.

## Vấn đề gốc

Không thể kiểm chứng bằng cách đào toàn bộ diện tích nghiên cứu lên xem có bom hay
không. Nhãn đối chứng chỉ tồn tại ở hai nơi, và cả hai đều không hoàn hảo:

- **Đất đã rà phá.** Có nhãn thật và đầy đủ, nhưng **không phải mẫu ngẫu nhiên** —
  cơ quan chuyên môn chọn nơi để rà dựa trên hồ sơ không kích, mức độ gần khu dân
  cư và nhu cầu sử dụng đất.
- **Hồ sơ tai nạn.** Không đầy đủ, nhưng **độc lập** — vị trí tai nạn không do ai
  chọn cả.

Vì vậy phải kiểm chứng ở bốn tầng, mỗi tầng bù cho điểm yếu của tầng khác.

---

## Tầng 1 — Đối chứng trên đất đã rà phá

**Nguồn nhãn.** Mỗi khoảnh đã rà phá có ranh giới, diện tích và số vật nổ thu hồi.
Trong mã nguồn, đây là `ClearanceData.items_found`.

**Cách chia tập — điểm mấu chốt.** Chia **theo khối không gian**, cài đặt ở
`evaluation/spatial_cv.py`. Tuyệt đối không chia ngẫu nhiên theo từng ô.

Lý do: phân bố vật nổ có tương quan không gian rất mạnh vì các quả bom trong cùng
một loạt rơi cách nhau vài chục mét. Nếu chia ngẫu nhiên theo điểm, mỗi ô trong tập
kiểm tra sẽ có hàng xóm nằm trong tập huấn luyện, và mô hình chỉ cần nội suy từ
hàng xóm là đạt chỉ tiêu rất cao mà không học được quy luật nào. Kết quả thu được
sẽ đẹp và vô nghĩa.

**Chỉ tiêu chính — đường cong hiệu quả rà phá.** Cài đặt ở
`evaluation/prioritisation.py`. Sắp các khoảnh theo thứ tự mô hình đề xuất, rồi đo:
sau khi rà phá $x$ phần trăm diện tích, đã thu hồi được bao nhiêu phần trăm tổng số
vật nổ.

Đường chéo là phương án quét trải đều: rà 20% diện tích thì thu được khoảng 20% vật
nổ. Giá trị của mô hình chính là khoảng cách giữa đường cong và đường chéo.

Chỉ tiêu này được chọn làm chỉ tiêu chính vì nó là đại lượng duy nhất chuyển thẳng
được thành ý nghĩa thực tiễn: cùng ngân sách, cùng nhân lực, xử lý sớm được gấp mấy
lần số vật nổ.

---

## Tầng 2 — Đối chứng độc lập bằng hồ sơ tai nạn

**Vì sao tầng này cần thiết.** Tầng 1 có một điểm yếu cố hữu: đất đã rà phá là kết
quả lựa chọn có chủ đích, nên mô hình có thể chỉ đang tái tạo phán đoán sẵn có của
những người đi trước. Chỉ tiêu vẫn cao, nhưng đóng góp khoa học bằng không.

Hồ sơ tai nạn khắc phục điểm yếu đó. Tai nạn xảy ra ở nơi người dân vô tình chạm
phải vật nổ trong sinh hoạt — phụ thuộc vào vật nổ có thật ở đó và mức độ lui tới
của con người, chứ không phụ thuộc phán đoán chuyên môn.

**Chỉ tiêu.** Tỉ lệ các vụ tai nạn rơi vào phần diện tích được mô hình xếp nguy cơ
cao nhất. Báo cáo ở ba mức 10%, 20% và 30% diện tích, kèm hệ số vượt ngẫu nhiên.

**Phép thử nghiêm hơn — chia theo thời gian.** Mô hình chỉ được học dữ liệu đến một
mốc thời gian, rồi đánh giá bằng các vụ tai nạn xảy ra **sau** mốc đó. Khi ấy kết
quả không còn là mô tả quá khứ mà là một dự báo đã được thực tế kiểm định. Tham số
điều khiển: `AccidentConfig.temporal_split_year`.

---

## Tầng 3 — Nhất quán giữa hai nguồn độc lập

**Ý tưởng.** Hồ sơ không kích và ảnh vệ tinh không liên quan gì về xuất xứ: một bên
là sổ sách tác chiến của phi đội, một bên là dấu vết vật lý trên mặt đất do một hệ
thống hoàn toàn khác ghi lại. Nếu hai nguồn khớp nhau về mặt không gian, độ tin cậy
của cả hai cùng được củng cố **mà không cần viện đến bất kỳ nhãn đối chứng nào**.

**Chỉ tiêu.** Hệ số tương quan hạng Spearman giữa mật độ hố bom phát hiện được và
tải trọng bom ghi nhận, trên cùng ô lưới. Dùng tương quan hạng chứ không phải tương
quan tuyến tính vì quan hệ giữa hai đại lượng là đồng biến nhưng không tuyến tính,
và cả hai phân bố đều lệch mạnh.

**Lưu ý khi so sánh.** Phải so trên tải trọng **đã lan toả qua hạt nhân sai số định
vị**, không phải trên điểm ghi chép thô. Hồ sơ ghi một điểm cho mỗi phi vụ với sai
số hàng trăm mét, còn hố bom thì rải trên nhiều ô. So thẳng hai thứ đó sẽ cho tương
quan âm, và đó là lỗi phương pháp chứ không phải phát hiện khoa học.

**Kiểm tra bậc độ lớn.** Từ tổng lượng bom theo hồ sơ nhân với tỉ lệ bom không nổ
đã biết trong tài liệu kỹ thuật, ước lượng khối lượng vật nổ còn sót, rồi so với
con số mà mô hình đưa ra. Không đòi hỏi trùng khít, chỉ đòi hỏi cùng bậc.

---

## Tầng 4 — Chuyển vùng địa lý và chất lượng hiệu chỉnh

**Chuyển vùng.** Huấn luyện trên nửa tây của vùng nghiên cứu, áp dụng cho nửa đông
như thể đó là một địa bàn hoàn toàn mới. Nếu hiệu năng được duy trì thì mô hình đã
học quy luật chung chứ không ghi nhớ đặc thù một địa bàn. Chỉ tiêu: mức suy giảm
tương đối của hiệu quả ưu tiên.

**Hiệu chỉnh xác suất.** Đầu ra được dùng để phân bổ nguồn lực công, nên xếp hạng
đúng thôi chưa đủ. Nếu mô hình nói một ô có xác suất ba mươi phần trăm thì trong
thực tế, trong số các ô được nói như vậy, phải có khoảng ba mươi phần trăm thực sự
chứa vật nổ. Báo cáo biểu đồ tin cậy, sai số hiệu chỉnh kỳ vọng và điểm Brier. Phép
hiệu chỉnh dùng hồi quy đẳng hướng, khớp trên một tập giữ lại riêng theo khối không
gian, không dùng để học.

---

## Xử lý thiên lệch chọn mẫu

Cài đặt ở `risk/pu_learning.py`. Hai biện pháp chạy song song.

**Trọng số nghịch đảo xác suất được chọn.** Ước lượng xác suất một ô được đưa vào
rà phá bằng hồi quy logistic, rồi gán cho mỗi mẫu huấn luyện trọng số bằng nghịch
đảo xác suất đó. Ô ít có khả năng được chọn mà vẫn lọt vào tập thì đại diện cho
nhiều ô tương tự chưa ai tới, nên đáng được coi trọng hơn. Trọng số được cắt ngọn
theo phân vị để tránh vài mẫu hiếm chi phối toàn bộ.

**Khung học từ dữ liệu chỉ có quan sát dương.** Ngay trong phần đất đã rà phá, việc
rà phá cũng không hoàn hảo — một tỉ lệ nhỏ vật nổ bị bỏ sót nên ô đó bị gán nhãn âm
trong khi thực tế là dương. Theo Elkan và Noto, nếu ước lượng được xác suất `c` mà
một ô thực sự dương được ghi nhận là dương, thì xác suất thật xấp xỉ bằng xác suất
mô hình chia cho `c`.

Cả hai biện pháp đều làm ước lượng thận trọng hơn, tức là nghiêng về phía cho rằng
còn nhiều vật nổ hơn những gì đã quan sát. Đó là chiều nghiêng đúng cho bài toán
này.

---

## Nguyên tắc an toàn

> **Hệ thống chỉ xếp thứ tự ưu tiên rà phá. Hệ thống không bao giờ tuyên bố một khu
> đất là an toàn.** Mọi khu đất, kể cả khi được mô hình chấm mức nguy cơ thấp nhất,
> vẫn phải được rà phá đầy đủ theo đúng quy trình kỹ thuật hiện hành trước khi đưa
> vào sử dụng.

Nguyên tắc này xuất phát từ tính bất đối xứng của sai lầm: một dự báo sót gây hậu
quả là sinh mạng con người, còn một cảnh báo thừa chỉ làm phát sinh thêm công rà
phá. Vì vậy:

- Mô hình được đánh giá theo hướng ưu tiên tuyệt đối cho việc không bỏ sót.
- Giao diện và mọi bản kết xuất không dùng nhãn, màu sắc hay cách diễn đạt nào có
  thể bị hiểu là xác nhận an toàn. Thang màu của bản đồ cố ý không có màu xanh lá ở
  đầu thang, vì xanh lá đọc thành an toàn.
- Mỗi dòng trong danh mục ưu tiên đều mang ghi chú nhắc lại nguyên tắc này.

---

## Chỉ tiêu mục tiêu

| Chỉ tiêu | Mục tiêu | Vị trí trong mã nguồn |
|---|---|---|
| mAP@0.5 phát hiện hố bom | ≥ 0,70 | `evaluation/detection_metrics.py` |
| Recall phát hiện hố bom | ≥ 0,80 | `evaluation/detection_metrics.py` |
| Thu hồi tại 20% diện tích | ≥ 0,70 | `evaluation/prioritisation.py` |
| Bao phủ tai nạn tại 20% diện tích | ≥ 0,75 | `evaluation/prioritisation.py` |
| Spearman hố bom và tải trọng | ≥ 0,60 | `evaluation/consistency.py` |
| Suy giảm khi chuyển vùng | ≤ 0,15 | `evaluation/spatial_cv.py` |
| Sai số hiệu chỉnh kỳ vọng | ≤ 0,05 | `evaluation/calibration.py` |

---

## Phạm vi cam kết

Đề tài **không** cam kết hệ thống phát hiện được vật nổ. Đề tài cam kết bốn điều,
và cả bốn đều kiểm chứng được bằng dữ liệu đã có:

1. Xếp hạng đúng các khoảnh đất đã rà phá, đánh giá trên khối không gian giữ lại.
2. Bao phủ phần lớn các vụ tai nạn đã xảy ra bằng một phần nhỏ diện tích.
3. Duy trì hiệu năng khi chuyển sang địa bàn chưa từng xuất hiện trong huấn luyện.
4. Cung cấp ước lượng xác suất đã hiệu chỉnh.

Đây là phạm vi cam kết hẹp nhưng chứng minh được. Chính tính kỷ luật đó mới là điều
kiện để kết quả được các cơ quan chuyên môn xem xét sử dụng.
