"""Tầng ba — mô hình nguy cơ hợp nhất.

Mô hình dự báo chính là cây quyết định tăng cường gradient. Lựa chọn này có ba lý
do, và cả ba đều phục vụ yêu cầu của bài toán chứ không phải sở thích kỹ thuật.

Thứ nhất, dữ liệu ở đây là bảng số không đồng nhất — khoảng cách tính bằng mét,
độ dốc tính bằng độ, biến nhị phân về lớp phủ — và cây quyết định xử lý loại dữ
liệu này tốt hơn mạng nơ-ron mà không cần chuẩn hoá cầu kỳ.

Thứ hai, kết quả được dùng để phân bổ nguồn lực công, nên phải giải thích được
từng ô lưới vì sao được xếp hạng cao. Cây quyết định cho phép đo mức đóng góp của
từng đặc trưng một cách trực tiếp.

Thứ ba, mô hình chạy trong vài giây trên máy thường, nên toàn bộ quy trình kiểm
chứng bốn tầng — vốn đòi hỏi huấn luyện lại nhiều lần — vẫn hoàn tất trong thời
gian chấp nhận được.

Đầu ra thô của mô hình đi qua hai bước hiệu chỉnh trước khi được công bố: hiệu
chỉnh theo khung học từ dữ liệu chỉ có quan sát dương, rồi hiệu chỉnh xác suất
bằng hồi quy đẳng hướng trên tập giữ lại riêng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import RiskConfig
from ..utils import get_logger
from ..evaluation.calibration import IsotonicCalibrator, fit_isotonic
from .pu_learning import (
    apply_pu_correction,
    estimate_label_frequency,
    fit_propensity,
    inverse_propensity_weights,
)

logger = get_logger(__name__)


def _build_estimator(cfg: RiskConfig):
    """Khởi tạo bộ học, ưu tiên XGBoost và lùi về scikit-learn nếu không có."""
    try:
        from xgboost import XGBClassifier

        return (
            "xgboost",
            XGBClassifier(
                n_estimators=cfg.max_iter,
                learning_rate=cfg.learning_rate,
                max_depth=6,
                min_child_weight=5,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=cfg.l2_regularization,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=cfg.seed,
                n_jobs=4,
            ),
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return (
            "sklearn",
            HistGradientBoostingClassifier(
                max_iter=cfg.max_iter,
                learning_rate=cfg.learning_rate,
                max_leaf_nodes=cfg.max_leaf_nodes,
                min_samples_leaf=cfg.min_samples_leaf,
                l2_regularization=cfg.l2_regularization,
                random_state=cfg.seed,
            ),
        )


@dataclass
class RiskModel:
    """Mô hình nguy cơ đã huấn luyện, kèm toàn bộ thành phần hiệu chỉnh."""

    cfg: RiskConfig
    estimator: object = None
    backend: str = ""
    propensity: object = None
    label_frequency: float = 1.0
    calibrator: Optional[IsotonicCalibrator] = None
    feature_names: List[str] = field(default_factory=list)
    _sample_X: Optional[np.ndarray] = None
    _sample_y: Optional[np.ndarray] = None

    def _permutation_importance_on_sample(self) -> np.ndarray:
        """Ước lượng mức đóng góp bằng phép hoán vị trên mẫu giữ lại từ lúc học."""
        rng = np.random.default_rng(self.cfg.seed)
        X, y = self._sample_X, self._sample_y
        base = self._raw_score(X)
        base_loss = float(
            np.mean((base - y) ** 2)
        )
        out = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            loss = float(np.mean((self._raw_score(Xp) - y) ** 2))
            out[j] = max(0.0, loss - base_loss)
        return out

    # -- Huấn luyện --------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        was_selected: np.ndarray,
        X_all: np.ndarray,
        feature_names: Optional[List[str]] = None,
        calibration_mask: Optional[np.ndarray] = None,
    ) -> "RiskModel":
        """Huấn luyện trên phần đất đã rà phá.

        ``X`` và ``y`` chỉ gồm các ô nằm trong khoảnh đã rà phá — đó là toàn bộ
        phần có nhãn thật. ``X_all`` gồm mọi ô của vùng nghiên cứu và được dùng để
        ước lượng cơ chế chọn mẫu.
        """
        self.feature_names = list(feature_names or [])
        self.backend, self.estimator = _build_estimator(self.cfg)

        sample_weight = None
        if self.cfg.use_propensity_weighting:
            self.propensity = fit_propensity(X_all, was_selected, seed=self.cfg.seed)
            p_train = self.propensity.predict(X)
            sample_weight = inverse_propensity_weights(p_train)
            logger.info(
                "Hiệu chỉnh thiên lệch chọn mẫu: trọng số trong dải [%.2f, %.2f]",
                float(sample_weight.min()),
                float(sample_weight.max()),
            )

        try:
            self.estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            self.estimator.fit(X, y)

        raw = self._raw_score(X)

        # Giữ lại một mẫu nhỏ của tập huấn luyện để ước lượng mức đóng góp đặc trưng
        # khi bộ học không tự cung cấp đại lượng đó.
        n_sample = min(2000, X.shape[0])
        idx = np.random.default_rng(self.cfg.seed).choice(
            X.shape[0], size=n_sample, replace=False
        )
        self._sample_X = np.asarray(X, dtype=float)[idx]
        self._sample_y = np.asarray(y, dtype=float)[idx]

        if self.cfg.use_pu_correction:
            self.label_frequency = estimate_label_frequency(raw[y == 1], raw)
            logger.info(
                "Hệ số tần suất nhãn của khung học chỉ có quan sát dương: c = %.3f",
                self.label_frequency,
            )

        if calibration_mask is not None and calibration_mask.any():
            scores_cal = apply_pu_correction(
                self._raw_score(X[calibration_mask]), self.label_frequency
            )
            self.calibrator = fit_isotonic(scores_cal, y[calibration_mask])

        logger.info(
            "Đã huấn luyện mô hình nguy cơ bằng %s trên %d ô có nhãn (%.1f%% dương).",
            self.backend,
            int(X.shape[0]),
            100.0 * float(np.mean(y)),
        )
        return self

    # -- Suy luận ----------------------------------------------------------
    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        proba = self.estimator.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=float)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Điểm thô của bộ học, trước mọi bước hiệu chỉnh."""
        return self._raw_score(X)

    def predict_ranking_score(self, X: np.ndarray) -> np.ndarray:
        """Điểm dùng để xếp thứ tự ưu tiên.

        Vì sao không dùng thẳng xác suất đã hiệu chỉnh: hồi quy đẳng hướng gán giá
        trị **đúng bằng không** cho toàn bộ khoảng thấp nhất, mà khoảng đó chiếm
        phần lớn diện tích vùng nghiên cứu. Khi ấy hàng chục nghìn ô có cùng một
        giá trị, và thứ tự giữa chúng trở thành thứ tự ngẫu nhiên theo chỉ số ô —
        tức là danh mục ưu tiên mất hoàn toàn ý nghĩa ở phần đuôi.

        Cách xử lý: giữ nguyên xác suất đã hiệu chỉnh để báo cáo và để đánh giá
        chất lượng hiệu chỉnh, nhưng khi xếp hạng thì cộng thêm một lượng rất nhỏ
        tỉ lệ với thứ hạng của điểm thô. Lượng cộng thêm đủ nhỏ để không đảo thứ tự
        giữa các ô có xác suất khác nhau, nhưng đủ để phá thế giữa các ô bằng nhau.
        """
        from ..evaluation.prioritisation import normalised_rank

        prob = self.predict_probability(X)
        raw = self._raw_score(X)
        return prob + 1e-6 * normalised_rank(raw)

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """Xác suất còn tồn tại vật nổ, đã qua toàn bộ các bước hiệu chỉnh."""
        scores = self._raw_score(X)
        if self.cfg.use_pu_correction:
            scores = apply_pu_correction(scores, self.label_frequency)
        if self.calibrator is not None:
            scores = self.calibrator.predict(scores)
        return np.clip(scores, 0.0, 1.0)

    # -- Giải thích --------------------------------------------------------
    def feature_importance(self) -> Dict[str, float]:
        """Mức đóng góp của từng đặc trưng, chuẩn hoá về tổng bằng một.

        XGBoost cung cấp sẵn mức đóng góp nội tại. HistGradientBoosting của
        scikit-learn thì không, nên trong trường hợp đó mức đóng góp được ước lượng
        bằng phép hoán vị trên một mẫu nhỏ của chính tập huấn luyện.
        """
        values = None
        if hasattr(self.estimator, "feature_importances_"):
            values = np.asarray(self.estimator.feature_importances_, dtype=float)

        if (values is None or values.size != len(self.feature_names)) and (
            self._sample_X is not None
        ):
            values = self._permutation_importance_on_sample()

        if values is None or values.size != len(self.feature_names):
            return {}
        total = values.sum()
        if total <= 0:
            return {}
        values = values / total
        pairs = sorted(
            zip(self.feature_names, values), key=lambda kv: -kv[1]
        )
        return {k: round(float(v), 4) for k, v in pairs}


def permutation_importance(
    model: RiskModel,
    X: np.ndarray,
    items: np.ndarray,
    n_repeats: int = 3,
    seed: int = 0,
) -> Dict[str, float]:
    """Mức đóng góp theo phép hoán vị, đo trên chính chỉ tiêu ưu tiên rà phá.

    Mức đóng góp nội tại của cây quyết định đo bằng mức giảm tạp chất, vốn không
    phải là thứ mà người lập kế hoạch quan tâm. Phép hoán vị đo trực tiếp trên chỉ
    tiêu mà đề tài cam kết: xáo trộn một đặc trưng rồi xem hiệu quả xếp thứ tự ưu
    tiên sụt bao nhiêu.
    """
    from ..evaluation.prioritisation import clearance_efficiency_curve

    rng = np.random.default_rng(seed)
    base = clearance_efficiency_curve(
        model.predict_ranking_score(X), items
    ).recovered_at_20pct

    out: Dict[str, float] = {}
    for j, name in enumerate(model.feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, j])
            score = clearance_efficiency_curve(
                model.predict_ranking_score(X_perm), items
            ).recovered_at_20pct
            drops.append(base - score)
        out[name] = round(float(np.mean(drops)), 4)

    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
