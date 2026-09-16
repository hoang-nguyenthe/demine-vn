"""Tham số cấu hình toàn hệ thống.

Mọi hằng số có ý nghĩa vật lý đều được nêu rõ căn cứ lựa chọn ngay tại chỗ khai
báo, để người đọc kiểm chứng được thay vì phải tin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GridConfig:
    """Lưới ô vuông dùng chung cho toàn bộ hệ thống."""

    # Cạnh ô lưới, mét. Chọn 100 m vì đây là bậc kích thước của một khoảnh rà phá
    # trong thực tế, đồng thời đủ lớn để thống kê ổn định trên từng ô.
    cell_size_m: float = 100.0

    # Kích thước vùng nghiên cứu theo số ô.
    n_cells_x: int = 220
    n_cells_y: int = 180

    # Toạ độ gốc quy ước của vùng nghiên cứu (kinh độ, vĩ độ).
    origin_lon: float = 106.85
    origin_lat: float = 16.72


@dataclass
class SortieConfig:
    """Mô phỏng hồ sơ không kích theo cấu trúc của bộ dữ liệu THOR."""

    n_missions: int = 1800

    # Số quả bom mỗi phi vụ. Dải giá trị phản ánh tải trọng thực tế của các loại
    # máy bay cường kích sử dụng trong giai đoạn được mô phỏng.
    bombs_per_mission: Tuple[int, int] = (6, 18)

    # Tỉ trọng của thành phần rải đều trong phân bố điểm ngắm. Phần còn lại bám
    # theo tuyến giao thông, sông ngòi và khu dân cư. Giá trị nhỏ vì không kích
    # trong chiến tranh tập trung rất mạnh vào các mục tiêu vận tải, và chính sự
    # tập trung đó tạo ra những hành lang ô nhiễm đậm đặc ngoài thực địa.
    uniform_target_fraction: float = 0.06

    # Độ tản mát của điểm rơi quanh điểm ngắm, mét. Đây là sai số kỹ thuật của
    # phương thức ném bom không dẫn đường.
    impact_dispersion_m: float = 95.0

    # Sai số định vị của chính hồ sơ ghi chép, mét. Toạ độ trong hồ sơ thời chiến
    # được ghi theo lưới bản đồ quân sự và làm tròn, nên lệch đáng kể so với điểm
    # rơi thật. Đây là nguồn bất định lớn nhất của tầng một.
    record_position_error_m: float = 180.0

    # Tỉ lệ phi vụ hoàn toàn không có trong hồ sơ, do mất mát tư liệu.
    missing_record_fraction: float = 0.08


@dataclass
class OrdnanceConfig:
    """Đặc tính vật lý của bom đạn, chi phối việc còn sót lại vật nổ."""

    # Tỉ lệ bom không nổ ở điều kiện nền chuẩn. Các tài liệu kỹ thuật quân sự ghi
    # nhận tỉ lệ này ở mức khoảng một phần mười đối với bom thông thường.
    base_dud_rate: float = 0.10

    # Hệ số nhân tỉ lệ không nổ trên nền đất mềm và ngập nước. Bom cắm sâu vào nền
    # mềm thường không kích nổ, đây là cơ chế vật lý then chốt của bài toán.
    soft_soil_dud_multiplier: float = 2.4

    # Xác suất một quả bom đã nổ để lại hố quan sát được trên ảnh, theo nền cứng.
    crater_visible_rate_hard: float = 0.92

    # Trên nền mềm, hố bom bị bồi lấp nhanh nên khó quan sát hơn nhiều.
    crater_visible_rate_soft: float = 0.45

    # Đường kính hố bom, mét.
    crater_diameter_m: Tuple[float, float] = (9.0, 22.0)


@dataclass
class ImageryConfig:
    """Ảnh vệ tinh trinh sát lịch sử mô phỏng."""

    # Kích thước điểm ảnh trên mặt đất, mét. Tương ứng độ phân giải của máy chụp
    # toàn cảnh trên vệ tinh trinh sát thế hệ được sử dụng, sau khi quét phim.
    ground_sample_distance_m: float = 1.2

    tile_size_px: int = 640

    # Số ảnh con. Con số này quyết định tỉ lệ diện tích vùng nghiên cứu thực sự có
    # ảnh vệ tinh phủ tới. Mỗi ảnh con phủ khoảng 0,59 km2, nên để phủ được phần lớn
    # một vùng rộng vài trăm km2 thì cần vài trăm ảnh. Độ phủ quá thấp sẽ khiến nhóm
    # đặc trưng quan sát khuyết ở hầu hết các ô và mất tác dụng.
    n_tiles: int = 420

    # Cường độ nhiễu hạt phim. Ảnh trinh sát là ảnh phim quét lại nên có hạt rõ.
    film_grain_sigma: float = 0.055

    # Độ lệch chuẩn của trường độ sáng nền, mô phỏng chiếu sáng không đều và
    # chênh lệch mật độ phim giữa các vùng của tấm ảnh.
    illumination_sigma: float = 0.13

    train_fraction: float = 0.70
    val_fraction: float = 0.15


@dataclass
class TerrainConfig:
    """Địa hình, thổ nhưỡng và hiện trạng sử dụng đất."""

    n_villages: int = 14
    n_roads: int = 5
    n_rivers: int = 2

    # Tỉ lệ diện tích là đất canh tác. Đây là nơi con người tiếp xúc nhiều nhất
    # với lòng đất, nên cũng là nơi nguy cơ tai nạn cao nhất.
    farmland_fraction: float = 0.42


@dataclass
class ClearanceConfig:
    """Mô phỏng hoạt động khảo sát và rà phá đã thực hiện."""

    # Tỉ lệ diện tích vùng nghiên cứu đã được rà phá. Con số nhỏ phản ánh đúng
    # thực tế: phần đã làm sạch chỉ chiếm phần rất nhỏ của diện tích ô nhiễm.
    cleared_area_fraction: float = 0.11

    # Trọng số của các yếu tố chi phối việc chọn khoảnh đất đưa vào rà phá. Đây
    # chính là nguồn thiên lệch chọn mẫu mà tầng ba phải hiệu chỉnh.
    selection_weight_tonnage: float = 1.0
    selection_weight_near_village: float = 1.6
    selection_weight_near_road: float = 0.9

    # Hiệu suất phát hiện của công tác rà phá thực địa. Không tuyệt đối, nhưng rất
    # cao, nên vẫn dùng làm nhãn đối chứng được.
    detection_efficiency: float = 0.96


@dataclass
class AccidentConfig:
    """Hồ sơ tai nạn bom mìn, dùng làm tập kiểm chứng độc lập."""

    n_years: int = 24

    # Xác suất một vật nổ còn sót gây tai nạn trong một năm, tính trên mỗi đơn vị
    # mức độ phơi nhiễm. Giá trị nhỏ vì phần lớn vật nổ không bao giờ bị chạm tới.
    annual_incident_rate: float = 0.0115

    # Mốc thời gian chia tập kiểm chứng theo thời gian. Mô hình chỉ học dữ liệu
    # trước mốc này và được đánh giá bằng tai nạn xảy ra sau mốc.
    temporal_split_year: int = 17


@dataclass
class DetectConfig:
    """Tầng hai — phát hiện hố bom trên ảnh."""

    backend: str = "auto"
    model_spec: str = "yolo11n.pt"
    epochs: int = 40
    batch_size: int = 16
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.50
    devices: List[int] = field(default_factory=lambda: [0, 1])
    workers: int = 2
    seed: int = 20260914


@dataclass
class RiskConfig:
    """Tầng ba — mô hình nguy cơ."""

    # Bán kính lan toả của hạt nhân quy chiếu hồ sơ không kích về lưới, mét. Bằng
    # đúng sai số định vị của hồ sơ; đây là cách mô hình hoá tường minh bất định
    # thay vì coi mỗi bản ghi là một điểm chính xác.
    record_kernel_radius_m: float = 180.0

    # Số khối không gian dùng để chia tập. Chia theo khối chứ không theo điểm là
    # bắt buộc, vì phân bố vật nổ có tương quan không gian rất mạnh.
    n_spatial_blocks: int = 24
    n_cv_folds: int = 4

    learning_rate: float = 0.06
    max_iter: int = 400
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 25
    l2_regularization: float = 1.0

    # Bật hiệu chỉnh thiên lệch chọn mẫu bằng trọng số nghịch đảo xác suất được
    # chọn đưa vào rà phá.
    use_propensity_weighting: bool = True

    # Bật hiệu chỉnh theo khung học từ dữ liệu chỉ có quan sát dương.
    use_pu_correction: bool = True

    # Ba hằng số dưới đây phải khớp với OrdnanceConfig. Chúng được lặp lại ở đây vì
    # tầng ba dùng chúng để dựng đặc trưng vật lý, và trong triển khai thật thì
    # chúng đến từ tài liệu kỹ thuật quân sự chứ không đến từ bộ mô phỏng.
    dud_softness_gain: float = 2.4
    crater_visibility_hard: float = 0.92
    crater_visibility_soft: float = 0.45

    # Trọng số của mức phơi nhiễm cộng đồng trong chỉ số ưu tiên, trộn trên thứ
    # hạng. Giá trị 0,35 được chọn qua phép quét có hệ thống (scripts/tune_risk.py):
    # đây là điểm mà chỉ tiêu bao phủ tai nạn gần như đạt trần trong khi chỉ tiêu
    # thu hồi vật nổ hầu như không suy giảm. Đặt bằng 0 thì hệ thống xếp hạng thuần
    # theo xác suất còn vật nổ, bỏ qua việc con người có lui tới hay không.
    priority_exposure_weight: float = 0.35

    n_calibration_bins: int = 12
    seed: int = 20260914


@dataclass
class RunConfig:
    """Cấu hình tổng của một lần chạy."""

    grid: GridConfig = field(default_factory=GridConfig)
    sortie: SortieConfig = field(default_factory=SortieConfig)
    ordnance: OrdnanceConfig = field(default_factory=OrdnanceConfig)
    imagery: ImageryConfig = field(default_factory=ImageryConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    clearance: ClearanceConfig = field(default_factory=ClearanceConfig)
    accident: AccidentConfig = field(default_factory=AccidentConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    output_dir: str = "outputs"
    data_dir: str = "data"
    seed: int = 20260914
    quick: bool = False
    multi_gpu: bool = True

    def apply_quick_mode(self) -> "RunConfig":
        """Rút gọn khối lượng tính toán để chạy thử nhanh."""
        self.sortie.n_missions = 420
        self.imagery.n_tiles = 120
        self.grid.n_cells_x = 120
        self.grid.n_cells_y = 100
        self.detect.epochs = 6
        self.risk.max_iter = 120
        self.risk.n_spatial_blocks = 12
        self.risk.n_cv_folds = 3
        self.quick = True
        return self
