"""DeMine-VN — Hệ thống lập bản đồ nguy cơ và xếp thứ tự ưu tiên rà phá bom mìn.

Hợp nhất hồ sơ không kích giải mật, dấu vết hố bom trên ảnh vệ tinh lịch sử và dữ
liệu rà phá thực địa để ước lượng xác suất còn tồn tại vật nổ theo từng ô lưới,
phục vụ công tác xếp thứ tự ưu tiên rà phá.

Hệ thống chỉ xếp thứ tự ưu tiên. Hệ thống không bao giờ tuyên bố một khu đất là an
toàn. Mọi khu đất, kể cả khi được chấm mức nguy cơ thấp nhất, vẫn phải rà phá đầy
đủ theo đúng quy trình kỹ thuật hiện hành trước khi đưa vào sử dụng.
"""

__version__ = "1.0.0"

SAFETY_NOTICE = (
    "Hệ thống chỉ xếp thứ tự ưu tiên rà phá. Hệ thống KHÔNG xác nhận bất kỳ khu "
    "đất nào là an toàn. Mọi khu đất vẫn phải được rà phá đầy đủ theo quy trình "
    "kỹ thuật hiện hành trước khi đưa vào sử dụng."
)
