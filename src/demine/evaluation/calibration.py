"""Hiệu chỉnh xác suất và các chỉ tiêu đo chất lượng hiệu chỉnh.

Đầu ra của hệ thống được dùng để phân bổ nguồn lực công, nên xếp hạng đúng thôi
chưa đủ. Nếu mô hình nói một ô có xác suất ba mươi phần trăm thì trong thực tế,
trong số các ô được nói như vậy, phải có khoảng ba mươi phần trăm thực sự chứa vật
nổ. Một mô hình xếp hạng hoàn hảo nhưng ước lượng lệch mức độ vẫn dẫn tới chia
ngân sách sai giữa các địa bàn.

Ba đại lượng được báo cáo: biểu đồ tin cậy, sai số hiệu chỉnh kỳ vọng và điểm
Brier. Phép hiệu chỉnh dùng hồi quy đẳng hướng, cài đặt bằng thuật toán gộp các
vi phạm liền kề, không phụ thuộc thư viện ngoài.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class IsotonicCalibrator:
    """Hiệu chỉnh xác suất bằng hồi quy đẳng hướng."""

    x_thresholds: np.ndarray
    y_values: np.ndarray

    def predict(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if self.x_thresholds.size == 0:
            return np.clip(s, 0.0, 1.0)
        return np.clip(
            np.interp(s, self.x_thresholds, self.y_values), 0.0, 1.0
        )


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    """Khớp hồi quy đẳng hướng bằng thuật toán gộp các vi phạm liền kề."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if s.size == 0:
        return IsotonicCalibrator(np.array([]), np.array([]))

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    values = list(y_sorted)
    weights = [1.0] * len(values)
    positions = list(range(len(values)))

    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1] + 1e-12:
            i += 1
            continue
        total_w = weights[i] + weights[i + 1]
        merged = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total_w
        values[i] = merged
        weights[i] = total_w
        del values[i + 1]
        del weights[i + 1]
        del positions[i + 1]
        if i > 0:
            i -= 1

    x_out, y_out = [], []
    idx = 0
    for value, weight in zip(values, weights):
        n = int(round(weight))
        x_out.append(s_sorted[min(idx + n - 1, s_sorted.size - 1)])
        y_out.append(value)
        idx += n

    return IsotonicCalibrator(
        x_thresholds=np.asarray(x_out, dtype=float),
        y_values=np.asarray(y_out, dtype=float),
    )


def reliability_curve(
    probability: np.ndarray, labels: np.ndarray, n_bins: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Biểu đồ tin cậy: xác suất dự báo trung bình so với tần suất thực tế."""
    p = np.asarray(probability, dtype=float)
    y = np.asarray(labels, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    mean_pred, mean_true, counts = [], [], []
    for b in range(n_bins):
        sel = bin_idx == b
        counts.append(int(sel.sum()))
        if sel.any():
            mean_pred.append(float(p[sel].mean()))
            mean_true.append(float(y[sel].mean()))
        else:
            mean_pred.append(np.nan)
            mean_true.append(np.nan)

    return (
        np.asarray(mean_pred),
        np.asarray(mean_true),
        np.asarray(counts, dtype=float),
    )


def expected_calibration_error(
    probability: np.ndarray, labels: np.ndarray, n_bins: int = 12
) -> float:
    """Sai số hiệu chỉnh kỳ vọng, trung bình có trọng số theo số mẫu mỗi khoảng."""
    pred, true, counts = reliability_curve(probability, labels, n_bins)
    valid = counts > 0
    if not valid.any():
        return 0.0
    w = counts[valid] / counts[valid].sum()
    return float(np.sum(w * np.abs(pred[valid] - true[valid])))


def brier_score(probability: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def calibration_report(
    probability: np.ndarray, labels: np.ndarray, n_bins: int = 12
) -> Dict[str, float]:
    return {
        "sai_so_hieu_chinh_ky_vong": round(
            expected_calibration_error(probability, labels, n_bins), 4
        ),
        "diem_brier": round(brier_score(probability, labels), 4),
        "ty_le_duong_thuc_te": round(float(np.mean(labels)), 4),
        "xac_suat_du_bao_trung_binh": round(float(np.mean(probability)), 4),
    }
