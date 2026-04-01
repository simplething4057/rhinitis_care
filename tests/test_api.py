"""
FastAPI 엔드포인트 통합 테스트 (TestClient, 서버 불필요)
"""
import pytest
from tests.conftest import SAMPLE_INPUTS


class TestHealth:
    def test_health_ok(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body

    def test_health_model_not_loaded(self, client_no_model):
        res = client_no_model.get("/health")
        assert res.status_code == 200
        assert res.json()["model_loaded"] is False


class TestRoot:
    def test_root_returns_env(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "env" in res.json()


class TestClusters:
    def test_list_clusters(self, client):
        res = client.get("/clusters")
        assert res.status_code == 200
        body = res.json()
        assert "clusters" in body
        assert len(body["clusters"]) > 0
        # 각 항목에 label, description 필드 존재 확인
        for item in body["clusters"]:
            assert "label" in item
            assert "description" in item


class TestPredict:
    @pytest.mark.parametrize("case_name", list(SAMPLE_INPUTS.keys()))
    def test_predict_valid_input(self, client, case_name):
        res = client.post("/predict", json=SAMPLE_INPUTS[case_name])
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        result = body["result"]
        assert "cluster_id" in result
        assert "cluster_label" in result
        assert "confidence" in result
        assert "guide" in result
        assert isinstance(result["guide"], list)

    def test_predict_invalid_asthma(self, client):
        data = {**SAMPLE_INPUTS["respiratory"], "has_asthma": 2}
        res = client.post("/predict", json=data)
        assert res.status_code == 422

    def test_predict_onset_age_negative(self, client):
        data = {**SAMPLE_INPUTS["respiratory"], "rhinitis_onset_age": -1}
        res = client.post("/predict", json=data)
        assert res.status_code == 422

    def test_predict_onset_age_too_large(self, client):
        data = {**SAMPLE_INPUTS["respiratory"], "rhinitis_onset_age": 21}
        res = client.post("/predict", json=data)
        assert res.status_code == 422

    def test_predict_missing_required_field(self, client):
        data = {k: v for k, v in SAMPLE_INPUTS["respiratory"].items()
                if k != "rhinitis_onset_age"}
        res = client.post("/predict", json=data)
        assert res.status_code == 422

    def test_predict_model_not_loaded_returns_503(self, client_no_model):
        res = client_no_model.post("/predict", json=SAMPLE_INPUTS["respiratory"])
        assert res.status_code == 503


class TestGuide:
    @pytest.mark.parametrize("label", ["호흡기 알레르기형", "비염+천식 복합형", "아토픽 마치형"])
    def test_guide_known_label(self, client, label):
        res = client.get(f"/guide/{label}")
        assert res.status_code == 200
        body = res.json()
        assert body["cluster_label"] == label
        assert "description" in body
        assert "guide" in body

    def test_guide_unknown_label_returns_404(self, client):
        res = client.get("/guide/없는유형")
        assert res.status_code == 404


class TestEnv:
    def test_env_returns_info(self, client):
        res = client.get("/env")
        assert res.status_code == 200
        body = res.json()
        assert "app_env" in body
        assert "python" in body
        assert "model_loaded" in body
