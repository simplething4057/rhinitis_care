"""
RhinitisPredictor 유닛 테스트
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestModelValidation:
    """_load_model 유효성 검증 테스트"""

    def _make_bundle(self, **overrides):
        """유효한 번들 기본값 반환."""
        model = MagicMock()
        model.predict.return_value = [1]
        model.transform.return_value = [[0.1, 0.5, 0.8]]
        scaler = MagicMock()
        scaler.transform.return_value = [[0.0] * 7]

        bundle = {
            "model":     model,
            "scaler":    scaler,
            "features":  ["has_asthma", "has_atopic_derm", "has_food_allergy",
                          "food_allergy_count", "rhinitis_onset_age",
                          "rhinitis_duration", "atopic_march"],
            "label_map": {0: "아토픽 마치형", 1: "호흡기 알레르기형", 2: "비염+천식 복합형"},
        }
        bundle.update(overrides)
        return bundle

    def test_load_model_file_not_found(self):
        from src.api.predictor import RhinitisPredictor
        with patch("src.api.predictor.Path.exists", return_value=False):
            p = RhinitisPredictor.__new__(RhinitisPredictor)
            p.model = p.scaler = p.features = p.label_map = None
            p._load_model()
        assert p.model is None

    def test_load_model_missing_key_raises(self):
        from src.api.predictor import RhinitisPredictor
        bundle = self._make_bundle()
        del bundle["scaler"]

        with patch("src.api.predictor.Path.exists", return_value=True), \
             patch("src.api.predictor.joblib.load", return_value=bundle):
            p = RhinitisPredictor.__new__(RhinitisPredictor)
            with pytest.raises(ValueError, match="필수 키 누락"):
                p._load_model()

    def test_load_model_invalid_model_type_raises(self):
        from src.api.predictor import RhinitisPredictor
        bundle = self._make_bundle(model="not_a_model")

        with patch("src.api.predictor.Path.exists", return_value=True), \
             patch("src.api.predictor.joblib.load", return_value=bundle):
            p = RhinitisPredictor.__new__(RhinitisPredictor)
            with pytest.raises(TypeError, match="predict"):
                p._load_model()

    def test_load_model_empty_features_raises(self):
        from src.api.predictor import RhinitisPredictor
        bundle = self._make_bundle(features=[])

        with patch("src.api.predictor.Path.exists", return_value=True), \
             patch("src.api.predictor.joblib.load", return_value=bundle):
            p = RhinitisPredictor.__new__(RhinitisPredictor)
            with pytest.raises(TypeError, match="features"):
                p._load_model()

    def test_load_model_invalid_label_map_raises(self):
        from src.api.predictor import RhinitisPredictor
        bundle = self._make_bundle(label_map=[0, 1, 2])  # list, not dict

        with patch("src.api.predictor.Path.exists", return_value=True), \
             patch("src.api.predictor.joblib.load", return_value=bundle):
            p = RhinitisPredictor.__new__(RhinitisPredictor)
            with pytest.raises(TypeError, match="label_map"):
                p._load_model()

    def test_load_model_valid_sets_is_loaded(self):
        from src.api.predictor import RhinitisPredictor
        bundle = self._make_bundle()

        with patch("src.api.predictor.Path.exists", return_value=True), \
             patch("src.api.predictor.joblib.load", return_value=bundle):
            p = RhinitisPredictor.__new__(RhinitisPredictor)
            p.model = p.scaler = p.features = p.label_map = None
            p._load_model()

        assert p.is_loaded is True
        assert p.features == bundle["features"]


class TestPredictOutput:
    """predict() 출력 구조 테스트 (실제 모델 파일 필요)"""

    MODEL_PATH = Path("outputs/models/kmeans_childhood.pkl")

    @pytest.fixture
    def real_predictor(self):
        if not self.MODEL_PATH.exists():
            pytest.skip("모델 파일 없음 — CI 환경에서는 건너뜁니다.")
        from src.api.predictor import RhinitisPredictor
        return RhinitisPredictor()

    def test_predict_returns_required_keys(self, real_predictor):
        result = real_predictor.predict({
            "has_asthma": 0, "has_atopic_derm": 0,
            "has_food_allergy": 0, "food_allergy_count": 0,
            "rhinitis_onset_age": 8.0, "rhinitis_duration": 2.0,
            "atopic_march": 0,
        })
        for key in ("cluster_id", "cluster_label", "confidence", "description", "guide"):
            assert key in result

    def test_predict_confidence_range(self, real_predictor):
        result = real_predictor.predict({
            "has_asthma": 1, "has_atopic_derm": 1,
            "has_food_allergy": 1, "food_allergy_count": 2,
            "rhinitis_onset_age": 5.0, "rhinitis_duration": 4.0,
            "atopic_march": 1,
        })
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_not_loaded_raises(self):
        from src.api.predictor import RhinitisPredictor
        p = RhinitisPredictor.__new__(RhinitisPredictor)
        p.model = None
        with pytest.raises(RuntimeError, match="로드"):
            p.predict({})
