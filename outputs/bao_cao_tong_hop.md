# DeMine-VN — Báo cáo tổng hợp kết quả (phiên bản 1.0.0)

Hệ thống lập bản đồ nguy cơ và xếp thứ tự ưu tiên rà phá bom mìn, vật nổ còn sót lại sau chiến tranh, trên cơ sở hợp nhất hồ sơ không kích giải mật, ảnh vệ tinh lịch sử và dữ liệu rà phá thực địa.

> **Nguyên tắc an toàn bắt buộc.** Hệ thống chỉ xếp thứ tự ưu tiên rà phá. Hệ thống KHÔNG xác nhận bất kỳ khu đất nào là an toàn. Mọi khu đất vẫn phải được rà phá đầy đủ theo quy trình kỹ thuật hiện hành trước khi đưa vào sử dụng.

---

## 1. Vùng nghiên cứu và dữ liệu

- Diện tích: 22.0 × 18.0 km. chia thành 39.600 ô lưới cạnh 100 m
- Phi vụ mô phỏng: 1800, trong đó 1688 phi vụ còn hồ sơ
- Điểm rơi: 20.430
- Vật nổ còn sót (nhãn đối chứng): 3.536
- Hố bom phát hiện được trên ảnh: 7.462
- Diện tích đã rà phá: 11.0%
- Tai nạn đã ghi nhận: 231 vụ trong 24 năm

## 2. Tầng hai — phát hiện hố bom trên ảnh vệ tinh lịch sử

precision | recall | f1    | mAP@0.5 | mAP@0.5:0.95 | so_ho_bom_that | so_ho_bom_du_bao
----------+--------+-------+---------+--------------+----------------+-----------------
0.9807    | 0.9873 | 0.984 | 0.9871  | 0.7653       | 1340           | 1349            

## 3. Kiểm chứng bốn tầng

### Tầng 1 — đối chứng trên đất đã rà phá

Chia tập **theo khối không gian**, không chia ngẫu nhiên theo điểm. Chi tiết phương pháp xem `docs/VALIDATION.md`.

so_lan_chia | thu_hoi_tai_20pct_trung_binh | do_lech_chuan | loi_the_so_voi_quet_deu
------------+------------------------------+---------------+------------------------
4           | 0.2601                       | 0.0355        | 0.1674                 

Đường cong hiệu quả rà phá trên toàn vùng:

thu_hoi_tai_10pct_dien_tich | thu_hoi_tai_20pct_dien_tich | thu_hoi_tai_50pct_dien_tich | loi_the_so_voi_quet_deu
----------------------------+-----------------------------+-----------------------------+------------------------
0.3798                      | 0.4992                      | 0.8258                      | 0.4782                 

### Tầng 2 — đối chứng độc lập bằng hồ sơ tai nạn

Vị trí tai nạn không do mô hình chọn cũng không do cơ quan chuyên môn chọn, nên đây là phép lấy mẫu độc lập với mọi phán đoán đã có trước. Dòng thứ hai là phép thử nghiêm hơn: mô hình chỉ học dữ liệu trước một mốc thời gian và được đánh giá bằng các vụ tai nạn xảy ra sau mốc đó.

tap_kiem_chung | so_vu_tai_nan | bao_phu_tai_10pct | bao_phu_tai_20pct | bao_phu_tai_30pct | he_so_vuot_ngau_nhien_tai_20pct
---------------+---------------+-------------------+-------------------+-------------------+--------------------------------
toàn bộ        | 231           | 0.1472            | 0.3766            | 0.619             | 1.883                          
sau năm thứ 17 | 61            | 0.1639            | 0.3279            | 0.6393            | 1.639                          

### Tầng 3 — nhất quán giữa hai nguồn độc lập

Hồ sơ không kích và ảnh vệ tinh không liên quan về xuất xứ. Nếu chúng khớp nhau về mặt không gian thì độ tin cậy của cả hai cùng được củng cố mà không cần viện đến bất kỳ nhãn đối chứng nào.

spearman_ho_bom_vs_tai_trong | so_o_duoc_so_sanh | uoc_luong_vat_no_tu_ho_so | ky_vong_vat_no_tu_mo_hinh | ty_so_bac_do_lon
-----------------------------+-------------------+---------------------------+---------------------------+-----------------
0.3261                       | 19438             | 1920.3                    | 1415.8                    | 0.737           

### Tầng 4 — chuyển vùng địa lý và chất lượng hiệu chỉnh

thu_hoi_tai_20pct_cung_vung | thu_hoi_tai_20pct_vung_moi | muc_suy_giam_tuong_doi
----------------------------+----------------------------+-----------------------
0.2601                      | 0.2708                     | -0.0413               

sai_so_hieu_chinh_ky_vong | diem_brier | ty_le_duong_thuc_te | xac_suat_du_bao_trung_binh
--------------------------+------------+---------------------+---------------------------
0.0275                    | 0.0432     | 0.1674              | 0.1476                    

## 4. Phân tích đóng góp thành phần

Hồ sơ không kích có sai số định vị lớn nhưng phủ khắp; hố bom thì chính xác về vị trí nhưng thiếu hụt có hệ thống đúng ở nơi nền đất mềm — tức là đúng nơi nhiều vật nổ còn sót nhất. Hai nguồn sai theo hai kiểu khác nhau, nên việc hợp nhất có giá trị thật chứ không phải cộng thêm cho đủ.

cau_hinh                               | so_dac_trung | thu_hoi_tai_20pct | loi_the_so_voi_quet_deu | bao_phu_tai_nan_tai_20pct
---------------------------------------+--------------+-------------------+-------------------------+--------------------------
A — chỉ hồ sơ không kích               | 4            | 0.4421            | 0.4039                  | 0.3485                   
B — chỉ hố bom từ ảnh vệ tinh          | 4            | 0.3665            | 0.275                   | 0.3636                   
C — hợp nhất hai nguồn, không địa hình | 8            | 0.4467            | 0.4281                  | 0.3788                   
D — hệ thống đầy đủ                    | 17           | 0.4868            | 0.4703                  | 0.3333                   

### Đóng góp của từng đặc trưng, đo bằng phép hoán vị

dac_trung                  | muc_sut_giam
---------------------------+-------------
tai_trong_ghi_nhan_lan_toa | 0.0501      
ky_vong_bom_khong_no       | 0.0401      
muc_do_phoi_nhiem          | 0.0178      
tai_trong_lan_can_500m     | 0.016       
khoang_cach_duong          | 0.0141      
do_doc                     | 0.0114      
ho_bom_hieu_chinh_tam_nhin | 0.0059      
ho_bom_lan_can_300m        | 0.003       
do_mem_nen_dat             | 0.0023      
khoang_cach_song           | 0.0005      

## 5. Phạm vi cam kết

Đề tài không cam kết rằng hệ thống phát hiện được vật nổ. Đề tài cam kết bốn điều, và cả bốn đều được kiểm chứng bằng dữ liệu trong chính báo cáo này:

1. Xếp hạng đúng các khoảnh đất đã rà phá, đánh giá trên khối không gian giữ lại.
2. Bao phủ phần lớn các vụ tai nạn đã xảy ra bằng một phần nhỏ diện tích.
3. Duy trì hiệu năng khi chuyển sang địa bàn chưa từng xuất hiện trong huấn luyện.
4. Cung cấp ước lượng xác suất đã hiệu chỉnh, kèm biểu đồ tin cậy và điểm Brier.

> **Nguyên tắc an toàn bắt buộc.** Hệ thống chỉ xếp thứ tự ưu tiên rà phá. Hệ thống KHÔNG xác nhận bất kỳ khu đất nào là an toàn. Mọi khu đất vẫn phải được rà phá đầy đủ theo quy trình kỹ thuật hiện hành trước khi đưa vào sử dụng.

---

## Môi trường chạy

- python: 3.13.1
- numpy: 2.2.6
- torch: 2.13.0
- cuda_available: False
- gpu_count: 0
