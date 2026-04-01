"""
공통 픽스처 정의
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── 예측 결과 픽스처 ─────────────────────────────────
MOCK_PREDICT_RESULT = {
    "cluster_id":    1,
    "cluster_label": "호흡기 알레르기형",
    "confidence":    0.85,
    "description":   "동반 질환이 없는 순수 비염 유형입니다.",
    "guide":         ["실내 습도를 50~60%로 유지하세요."],
}

# ── 테스트 입력 케이스 ────────────────────────────────
SAMPLE_INPUTS = {
    "respiratory": {
        "has_asthma": 0, "has_atopic_derm": 0,
        "has_food_allergy": 0, "food_allergy_count": 0,
        "rhinitis_onset_age": 8.0, "rhinitis_duration": 2.0,
        "atopic_march": 0,
    },
    "asthma_complex": {
        "has_asthma": 1, "has_atopic_derm": 0,
        "has_food_allergy": 0, "food_allergy_count": 0,
        "rhinitis_onset_age": 7.0, "rhinitis_duration": 3.0,
        "atopic_march": 0,
    },
    "atopic_march": {
        "has_asthma": 1, "has_atopic_derm": 1,
        "has_food_allergy": 1, "food_allergy_count": 2,
        "rhinitis_onset_age": 5.0, "rhinitis_duration": 4.0,
        "atopic_march": 1,
    },
}


@pytest.fixture
def client():
    """모델을 mock한 FastAPI TestClient."""
    mock_pred = MagicMock()
    mock_pred.is_loaded = True
    mock_pred.predict.return_value = MOCK_PREDICT_RESULT

    with patch("src.api.main.predictor", mock_pred), \
         patch("src.api.main.setup_logging"):          # 테스트 중 로깅 설정 스킵
        from src.api.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_no_model():
    """모델이 로드되지 않은 상태의 TestClient."""
    mock_pred = MagicMock()
    mock_pred.is_loaded = False

    with patch("src.api.main.predictor", mock_pred), \
         patch("src.api.main.setup_logging"):
        from src.api.main import app
        with TestClient(app) as c:
            yield c
