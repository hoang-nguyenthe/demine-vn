"""Chỉ tiêu đánh giá tầng phát hiện hố bom.

Cài đặt mAP theo đúng quy ước PASCAL VOC từ năm 2010 trở đi: đường cong độ chính
xác theo độ bao phủ được nội suy đơn điệu trước khi lấy tích phân. Việc tự cài đặt
thay vì gọi thư viện là có chủ ý — đội thi phải hiểu và giải thích được từng bước
của chỉ tiêu mà mình công bố, và cách này cũng loại bỏ một phụ thuộc có thể không
cài được khi notebook chạy ngắt mạng.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Ma trận tỉ số giao trên hợp giữa hai tập khung bao."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=float)

    ax1, ay1, ax2, ay2 = [boxes_a[:, i][:, None] for i in range(4)]
    bx1, by1, bx2, by2 = [boxes_b[:, i][None, :] for i in range(4)]

    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    inter = inter_w * inter_h

    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)

    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    ap50: float
    ap50_95: float
    n_true: int
    n_pred: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mAP@0.5": round(self.ap50, 4),
            "mAP@0.5:0.95": round(self.ap50_95, 4),
            "so_ho_bom_that": self.n_true,
            "so_ho_bom_du_bao": self.n_pred,
        }


def _average_precision(
    all_scores: np.ndarray, all_tp: np.ndarray, n_true: int
) -> float:
    """Độ chính xác trung bình theo quy ước nội suy đơn điệu."""
    if n_true == 0:
        return 0.0
    if all_scores.size == 0:
        return 0.0

    order = np.argsort(-all_scores, kind="mergesort")
    tp = all_tp[order].astype(float)
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / n_true
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # Nội suy đơn điệu: độ chính xác tại mỗi mức bao phủ lấy bằng giá trị lớn nhất
    # ở các mức bao phủ từ đó trở đi.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if precision.size else 0.0], precision])
    return float(np.trapezoid(precision, recall))


def _match_at_threshold(
    predictions: Sequence, ground_truth: Dict[str, np.ndarray], iou_threshold: float
):
    scores_all: List[float] = []
    tp_all: List[float] = []
    n_true = 0

    for det in predictions:
        gt = ground_truth.get(det.image_id, np.zeros((0, 4), dtype=float))
        n_true += int(gt.shape[0])

        if det.boxes.shape[0] == 0:
            continue

        order = np.argsort(-det.scores, kind="mergesort")
        boxes = det.boxes[order]
        scores = det.scores[order]

        matched = np.zeros(gt.shape[0], dtype=bool)
        ious = iou_matrix(boxes, gt)

        for i in range(boxes.shape[0]):
            scores_all.append(float(scores[i]))
            if gt.shape[0] == 0:
                tp_all.append(0.0)
                continue
            candidates = ious[i].copy()
            candidates[matched] = -1.0
            best = int(np.argmax(candidates))
            if candidates[best] >= iou_threshold:
                matched[best] = True
                tp_all.append(1.0)
            else:
                tp_all.append(0.0)

    return np.asarray(scores_all), np.asarray(tp_all), n_true


def evaluate_detections(
    predictions: Sequence,
    ground_truth: Dict[str, np.ndarray],
    score_threshold: float = 0.25,
) -> DetectionMetrics:
    """Tính toàn bộ chỉ tiêu của tầng phát hiện."""
    scores, tp, n_true = _match_at_threshold(predictions, ground_truth, 0.50)
    ap50 = _average_precision(scores, tp, n_true)

    aps = []
    for thr in np.arange(0.50, 0.951, 0.05):
        s, t, n = _match_at_threshold(predictions, ground_truth, float(thr))
        aps.append(_average_precision(s, t, n))
    ap50_95 = float(np.mean(aps)) if aps else 0.0

    keep = scores >= score_threshold
    n_pred = int(keep.sum())
    tp_count = float(tp[keep].sum()) if n_pred else 0.0
    precision = tp_count / n_pred if n_pred else 0.0
    recall = tp_count / n_true if n_true else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        ap50=ap50,
        ap50_95=ap50_95,
        n_true=n_true,
        n_pred=n_pred,
    )
