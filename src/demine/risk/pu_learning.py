"""Học từ dữ liệu chỉ có quan sát dương, và hiệu chỉnh thiên lệch chọn mẫu.

Đây là phần phương pháp luận cốt lõi của tầng ba, và là chỗ dễ làm sai nhất.

**Vấn đề.** Nhãn thật chỉ tồn tại ở những khoảnh đất đã được rà phá. Nhưng các
khoảnh đó không được chọn ngẫu nhiên — cơ quan chuyên môn chọn chúng dựa trên hồ
sơ không kích, mức độ gần khu dân cư và nhu cầu sử dụng đất. Nếu huấn luyện thẳng
trên tập này rồi đánh giá cũng trên tập này, mô hình sẽ học lại đúng phán đoán của
những người đi trước, và mọi chỉ tiêu đều bị thổi phồng.

**Hai biện pháp được áp dụng song song.**

1. *Hiệu chỉnh trọng số theo nghịch đảo xác suất được chọn.* Ước lượng xác suất
   một ô lưới được đưa vào rà phá, rồi gán cho mỗi mẫu trong tập huấn luyện một
   trọng số bằng nghịch đảo xác suất đó. Ô nào ít có khả năng được chọn mà vẫn lọt
   vào tập thì đại diện cho nhiều ô tương tự chưa ai tới, nên đáng được coi trọng
   hơn. Trọng số được cắt ngọn để tránh một vài mẫu hiếm chi phối toàn bộ.

2. *Hiệu chỉnh theo khung học từ dữ liệu chỉ có quan sát dương.* Ngay trong phần
   đất đã rà phá, việc rà phá cũng không hoàn hảo: một tỉ lệ nhỏ vật nổ bị bỏ sót
   nên ô đó bị gán nhãn âm trong khi thực tế là dương. Theo cách tiếp cận của
   Elkan và Noto, nếu ước lượng được xác suất ``c`` mà một ô thực sự dương được
   ghi nhận là dương, thì xác suất thật xấp xỉ bằng xác suất mô hình chia cho
   ``c``. Hằng số ``c`` được ước lượng ngay từ dữ liệu, trên tập giữ lại.

Cả hai biện pháp đều làm cho ước lượng xác suất thận trọng hơn — tức là nghiêng
về phía cho rằng còn nhiều vật nổ hơn những gì đã quan sát được. Đó là chiều
nghiêng đúng cho bài toán này, nơi bỏ sót là sai lầm phải tránh bằng mọi giá.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PropensityModel:
    """Mô hình xác suất một ô lưới được đưa vào rà phá."""

    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    fill_values: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = _impute(np.asarray(X, dtype=np.float64), self.fill_values)
        Z = (Xf - self.feature_mean) / self.feature_std
        logit = Z @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))


def _impute(X: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Điền giá trị khuyết bằng trung vị của cột.

    Nhóm đặc trưng quan sát mang giá trị khuyết ở những ô không có ảnh vệ tinh phủ
    tới. Mô hình dự báo chính là cây quyết định nên xử lý được giá trị khuyết một
    cách tự nhiên, nhưng mô hình xác suất được chọn ở đây là hồi quy tuyến tính nên
    không. Vì đây chỉ là mô hình phụ trợ dùng để ước lượng trọng số, việc điền bằng
    trung vị là đủ và không ảnh hưởng tới kết luận.
    """
    if not np.isnan(X).any():
        return X
    out = X.copy()
    idx = np.where(np.isnan(out))
    out[idx] = np.take(fill, idx[1])
    return out


def fit_propensity(X: np.ndarray, was_selected: np.ndarray, seed: int = 0) -> PropensityModel:
    """Ước lượng xác suất được chọn đưa vào rà phá bằng hồi quy logistic.

    Cài đặt bằng phương pháp giảm gradient có phạt chuẩn bậc hai, không phụ thuộc
    thư viện ngoài, để phần phương pháp luận này hoàn toàn tự chứa và kiểm tra
    được.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(was_selected, dtype=np.float64)

    with np.errstate(invalid="ignore"):
        fill = np.nanmedian(X, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    X = _impute(X, fill)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Z = (X - mean) / std

    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=Z.shape[1])
    b = 0.0

    lr = 0.25
    n = Z.shape[0]
    for step in range(400):
        logit = np.clip(Z @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-logit))
        diff = p - y
        grad_w = Z.T @ diff / n + 1e-3 * w
        grad_b = float(diff.mean())
        w -= lr * grad_w
        b -= lr * grad_b
        if step == 250:
            lr *= 0.4

    return PropensityModel(
        coefficients=w,
        intercept=b,
        feature_mean=mean,
        feature_std=std,
        fill_values=fill,
    )


def inverse_propensity_weights(
    propensity: np.ndarray, clip_quantile: float = 0.98
) -> np.ndarray:
    """Trọng số nghịch đảo xác suất được chọn, có cắt ngọn.

    Cắt ngọn là bắt buộc: khi xác suất được chọn tiến gần không, nghịch đảo của nó
    bùng nổ và một vài mẫu hiếm sẽ chi phối toàn bộ quá trình huấn luyện. Ngưỡng
    cắt được lấy theo phân vị chứ không phải một hằng số cố định, để phương pháp
    không phụ thuộc thang đo của bộ dữ liệu cụ thể.
    """
    p = np.clip(np.asarray(propensity, dtype=float), 1e-4, 1.0)
    w = 1.0 / p
    cap = float(np.quantile(w, clip_quantile))
    w = np.minimum(w, cap)
    return w / w.mean()


def estimate_label_frequency(
    scores_positive: np.ndarray, scores_all: np.ndarray
) -> float:
    """Ước lượng hằng số ``c`` của khung Elkan–Noto.

    ``c`` là xác suất một ô thực sự dương được ghi nhận là dương trong dữ liệu. Ước
    lượng bằng điểm trung bình mà mô hình gán cho các mẫu đã được ghi nhận dương;
    đây là ước lượng chuẩn và không thiên lệch khi giả thiết chọn mẫu ngẫu nhiên
    trong nhóm dương được thoả mãn xấp xỉ.
    """
    if scores_positive.size == 0:
        return 1.0
    c = float(np.mean(scores_positive))
    # Chặn dưới để tránh chia cho số rất nhỏ làm xác suất vượt quá một cách vô lý.
    return float(np.clip(c, 0.25, 1.0))


def apply_pu_correction(scores: np.ndarray, c: float) -> np.ndarray:
    """Quy đổi điểm của mô hình thành xác suất thật theo khung Elkan–Noto."""
    if c <= 0:
        return scores
    return np.clip(np.asarray(scores, dtype=float) / c, 0.0, 1.0)
