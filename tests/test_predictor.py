"""
RhinitisPredictor 유닛 테스트 (LightGBM 우선 / K-Means 폴백 구조 대응)
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path


# ── 공통 mock 헬퍼 ─────────────────────────────────────────────────
def _make_lgbm_bundle(**overrides):
    """LightGBM 번들 mock 반환."""
    lgbm_model = MagicMock()
    lgbm_model.predict.return_value = np.array([1])
    lgbm_model.predict_proba.return_value = np.array([[0.15, 0.70, 0.15]])

    bundle = {
        "model":     lgbm_model,
        "features":  ["has_asthma", "has_atopic_derm", "has_food_allergy",
                      "food_allergy_count", "rhinitis_onset_age",
                      "rhinitis_duration", "atopic_march", "is_female"],
        "label_map": {0: "호흡기 알레르기형", 1: "비염+천식 복합형", 2: "아토픽 마치형"},
        "cv_acc": 1.0,
        "cv_f1":  1.0,
    }
    bundle.update(overrides)
    return bundle


def _make_kmeans_bundle(**overrides):
    """K-Means 번들 mock 반환."""
    km_model = MagicMock()
    km_model.predict.return_value = [0]
    km_model.transform.return_value = [[0.1, 0.5, 0.8]]

    scaler = MagicMock()
    scaler.transform.return_value = [[0.0] * 8]

    bundle = {
        "model":     km_model,
        "scaler":    scaler,
        "features":  ["has_asthma", "has_atopic_derm", "has_food_allergy",
                      "food_allergy_count", "rhinitis_onset_age",
                      "rhinitis_duration", "atopic_march", "is_female"],
        "label_map": {0: "호흡기 알레르기형", 1: "비염+천식 복합형", 2: "아토픽 마치형"},
    }
    bundle.update(overrides)
    return bundle


# ── 테스트: LightGBM 로딩 ─────────────────────────────────────────
class TestLGBMLoading:

    def test_lgbm_loaded_sets_model_type(self):
        from src.api.predictor import RhinitisPredictor
        bundle = _make_lgbm_bundle()
        with patch.object(Path, "exists", return_value=True), \
             patch("src.api.predictor.joblib.load", return_value=bundle):
            p = RhinitisPredictor()
        assert p.model_type == "lightgbm"
        assert p.is_loaded is True

    def test_lgbm_features_stored(self):
        from src.api.predictor import RhinitisPredictor
        bundle = _make_lgbm_bundle()
        with patch.object(Path, "exists", return_value=True), \
             patch("src.api.predictor.joblib.load", return_value=bundle):
            p = RhinitisPredictor()
        assert p._lgbm_features == bundle["features"]

    def test_lgbm_broken_falls_back_to_kmeans(self):
        """LightGBM 로드 실패 시 K-Means로 폴백."""
        from src.api.predictor import RhinitisPredictor
        call_count = {"n": 0}

        def load_side(path):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"model": "broken"}   # 파손 번들
            return _make_kmeans_bundle()

        with patch.object(Path, "exists", return_value=True), \
             patch("src.api.predictor.joblib.load", side_effect=load_side):
            p = RhinitisPredictor()

        assert p.model_type == "kmeans"
        assert p.is_loaded is True


# ── 테스트: K-Means 로딩 ──────────────────────────────────────────
class TestKMeansLoading:

    def _exists_kmeans_only(self):
        """LightGBM 없음, K-Means 있음을 흉내."""
        call = {"n": 0}
        def side(self_path):
            call["n"] += 1
            return call["n"] > 1
        return side

    def test_kmeans_missing_scaler_not_loaded(self):
        from src.api.predictor import RhinitisPredictor
        bundle = _make_kmeans_bundle()
        del bundle["scaler"]
        with patch.object(Path, "exists", self._exists_kmeans_only()), \
             patch("src.api.predictor.joblib.load", return_value=bundle):
            p = RhinitisPredictor()
        assert p.is_loaded is False

    def test_no_model_files_not_loaded(self):
        from src.api.predictor import RhinitisPredictor
        with patch.object(Path, "exists", return_value=False):
            p = RhinitisPredictor()
        assert p.is_loaded is False
        assert p.model_type is None


# ── 테스트: predict() 출력 구조 ───────────────────────────────────
class TestPredictOutput:

    SAMPLE = {
        "has_asthma": 0, "has_atopic_derm": 0,
        "has_food_allergy": 0, "food_allergy_count": 0,
        "rhinitis_onset_age": 8.0, "rhinitis_duration": 2.0,
        "atopic_march": 0, "is_female": 0,
    }

    @pytest.fixture
    def lgbm_pred(self):
        from src.api.predictor import RhinitisPredictor
        with patch.object(Path, "exists", return_value=True), \
             patch("src.api.predictor.joblib.load", return_value=_make_lgbm_bundle()):
            return RhinitisPredictor()

    @pytest.fixture
    def kmeans_pred(self):
        from src.api.predictor import RhinitisPredictor
        call = {"n": 0}
        def exists_side(self_path):
            call["n"] += 1
            return call["n"] > 1
        with patch.object(Path, "exists", exists_side), \
             patch("src.api.predictor.joblib.load", return_value=_make_kmeans_bundle()):
            return RhinitisPredictor()

    def test_lgbm_required_keys(self, lgbm_pred):
        result = lgbm_pred.predict(self.SAMPLE)
        for key in ("cluster_id", "cluster_label", "confidence",
                    "cluster_probs", "description", "guide", "model_type"):
            assert key in result

    def test_lgbm_model_type_value(self, lgbm_pred):
        assert lgbm_pred.predict(self.SAMPLE)["model_type"] == "lightgbm"

    def test_lgbm_confidence_range(self, lgbm_pred):
        conf = lgbm_pred.predict(self.SAMPLE)["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_lgbm_cluster_probs_sum(self, lgbm_pred):
        probs = lgbm_pred.predict(self.SAMPLE)["cluster_probs"]
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_lgbm_guide_non_empty(self, lgbm_pred):
        guide = lgbm_pred.predict(self.SAMPLE)["guide"]
        assert isinstance(guide, list) and len(guide) > 0

    def test_kmeans_model_type_value(self, kmeans_pred):
        assert kmeans_pred.predict(self.SAMPLE)["model_type"] == "kmeans"

    def test_kmeans_confidence_range(self, kmeans_pred):
        conf = kmeans_pred.predict(self.SAMPLE)["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_not_loaded_raises(self):
        from src.api.predictor import RhinitisPredictor
        with patch.object(Path, "exists", return_value=False):
            p = RhinitisPredictor()
        with pytest.raises(RuntimeError, match="로드"):
            p.predict(self.SAMPLE)


# ── 테스트: 실제 모델 파일 (CI skip) ──────────────────────────────
class TestRealModel:

    LGBM_PATH   = Path("outputs/models/lgbm_rhinitis.pkl")
    KMEANS_PATH = Path("outputs/models/kmeans_rhinitis.pkl")

    @pytest.fixture
    def real_pred(self):
        if not (self.LGBM_PATH.exists() or self.KMEANS_PATH.exists()):
            pytest.skip("모델 파일 없음 — CI 환경에서는 건너뜁니다.")
        from src.api.predictor import RhinitisPredictor
        return RhinitisPredictor()

    def test_valid_label(self, real_pred):
        result = real_pred.predict({
            "has_asthma": 1, "has_atopic_derm": 0,
            "has_food_allergy": 0, "food_allergy_count": 0,
            "rhinitis_onset_age": 7.0, "rhinitis_duration": 3.0,
            "atopic_march": 0, "is_female": 0,
        })
        valid = {"호흡기 알레르기형", "비염+천식 복합형", "아토픽 마치형"}
        assert result["cluster_label"] in valid

    def test_model_type_set(self, real_pred):
        assert real_pred.model_type in ("lightgbm", "kmeans")

    def test_lgbm_preferred_when_exists(self, real_pred):
        if self.LGBM_PATH.exists():
            assert real_pred.model_type == "lightgbm"
