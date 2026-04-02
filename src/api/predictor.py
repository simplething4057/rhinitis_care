"""
모델 로딩 및 예측 로직
- 증상 기반 K-Means (kmeans_rhinitis.pkl)
- 입력: 증상 점수 4종 (콧물, 코막힘, 재채기, 눈증상) + 동반질환 보조 정보
"""
import numpy as np
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = "outputs/models/kmeans_rhinitis.pkl"

# ── 클러스터별 설명 & 가이드 (증상 중심) ──────────────
CLUSTER_INFO = {
    "콧물·재채기 우세형": {
        "description": (
            "수양성 콧물과 연속적인 재채기가 주된 증상인 전형적인 알레르기 비염 유형입니다. "
            "눈 가려움이나 충혈이 동반되는 경우가 많으며, "
            "특정 항원(꽃가루, 집먼지진드기 등)에 민감하게 반응합니다."
        ),
        "guide": [
            "생리식염수로 코 세척을 하여 원인 항원을 씻어내세요.",
            "외출 시 전용 안경과 KF80 이상 마스크를 착용해 항원 접촉을 최소화하세요.",
            "증상 발현 전 미리 항히스타민제를 복용하는 방안을 의사와 상의하세요.",
            "침구류를 60°C 이상 온수로 세탁하고 집먼지진드기 방지 커버를 사용하세요.",
            "꽃가루 농도가 높은 오전 6~10시 사이에는 환기를 자제하세요.",
        ],
    },
    "코막힘 우세형": {
        "description": (
            "콧물보다 코 점막 부종으로 인한 코막힘이 주된 증상인 유형입니다. "
            "만성적인 경향이 있으며 수면 장애나 집중력 저하를 유발할 수 있습니다. "
            "천식 동반 가능성이 상대적으로 높습니다."
        ),
        "guide": [
            "취침 전 코 세척으로 비강 내 분비물을 제거하고 점막 부종을 완화하세요.",
            "실내 습도를 45~55%로 유지하고 찬 공기 직접 노출을 피하세요.",
            "스테로이드 비강 분무제(나잘스프레이)를 꾸준히 사용하세요 (효과 발현 3~7일).",
            "취침 시 베개를 높게 하면 두부 정맥압이 낮아져 코막힘이 완화됩니다.",
            "축농증(부비동염) 합병증 여부를 이비인후과에서 정기적으로 확인하세요.",
        ],
    },
    "복합 과민형": {
        "description": (
            "눈 가려움·충혈 등 안구 증상이 두드러지면서 콧물, 코막힘, 재채기가 함께 나타나는 유형입니다. "
            "다양한 환경 자극(꽃가루, 동물 털, 화학물질 등)에 복합적으로 반응하며, "
            "봄·가을 환절기에 증상이 급격히 악화되는 경향이 있습니다."
        ),
        "guide": [
            "알레르기 전문의를 통해 원인 항원 검사(MAST/RAST)를 받으세요.",
            "세안 시 눈을 깨끗이 씻고, 콘택트렌즈 사용을 자제하세요.",
            "항히스타민 안약과 비강 스프레이를 병행하면 효과적입니다.",
            "외출 후 즉시 세안·세척하여 피부·점막에 붙은 항원을 제거하세요.",
            "공기청정기(HEPA 필터)를 사용해 실내 항원 농도를 낮추세요.",
        ],
    },
}

# 예상치 못한 클러스터명 fallback
DEFAULT_INFO = {
    "description": "비염 유형이 분류됐습니다. 전문의 상담을 권장합니다.",
    "guide": ["정밀 검사를 통해 정확한 유형을 파악하고 맞춤 치료를 시작하세요."],
}


class RhinitisPredictor:
    def __init__(self):
        self.model     = None
        self.scaler    = None
        self.features  = None
        self.label_map = None
        self._load_model()

    def _load_model(self):
        path = Path(MODEL_PATH)
        if not path.exists():
            logger.warning(f"모델 파일 없음: {MODEL_PATH}")
            return

        bundle = joblib.load(path)

        # 번들 키 검증
        required_keys = {"model", "scaler", "features", "label_map"}
        missing = required_keys - bundle.keys()
        if missing:
            raise ValueError(f"모델 번들에 필수 키 누락: {missing}")

        if not hasattr(bundle["model"], "predict"):
            raise TypeError("bundle['model']이 predict 메서드를 가지지 않습니다.")
        if not hasattr(bundle["scaler"], "transform"):
            raise TypeError("bundle['scaler']이 transform 메서드를 가지지 않습니다.")
        if not isinstance(bundle["features"], list) or not bundle["features"]:
            raise TypeError("bundle['features']는 비어 있지 않은 list여야 합니다.")
        if not isinstance(bundle["label_map"], dict):
            raise TypeError("bundle['label_map']은 dict여야 합니다.")

        self.model     = bundle["model"]
        self.scaler    = bundle["scaler"]
        self.features  = bundle["features"]
        self.label_map = bundle["label_map"]
        logger.info(
            f"모델 로드 완료: {MODEL_PATH} "
            f"(features={self.features}, labels={list(self.label_map.values())})"
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, input_data: dict) -> dict:
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        # 피처 벡터 (증상 점수만 사용)
        X = np.array([[input_data.get(f, 0) for f in self.features]])
        X_scaled = self.scaler.transform(X)

        cluster_id = int(self.model.predict(X_scaled)[0])
        label      = self.label_map.get(cluster_id, f"클러스터 {cluster_id}")

        # 유형별 확률: 거리 역수를 softmax 방식으로 정규화
        distances = self.model.transform(X_scaled)[0]
        inv_dist  = 1.0 / (1.0 + distances)
        probs     = inv_dist / inv_dist.sum()
        confidence = float(round(probs[cluster_id], 4))

        cluster_probs = {
            self.label_map.get(i, f"클러스터 {i}"): float(round(probs[i], 4))
            for i in range(len(self.label_map))
        }

        info = CLUSTER_INFO.get(label, DEFAULT_INFO)

        return {
            "cluster_id":    cluster_id,
            "cluster_label": label,
            "confidence":    confidence,
            "cluster_probs": cluster_probs,
            "description":   info["description"],
            "guide":         info["guide"],
        }


# 싱글톤 인스턴스
predictor = RhinitisPredictor()
