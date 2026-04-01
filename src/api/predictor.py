"""
모델 로딩 및 예측 로직
- K-Means 코호트 모델 (kmeans_rhinitis.pkl = kmeans_childhood.pkl 동일 번들)
- Step 3에서 학습한 k=3 모델 사용
- 클러스터 3종: 호흡기 알레르기형 / 비염+천식 복합형 / 아토픽 마치형
"""
import numpy as np
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# step3_clustering.py 가 두 경로에 동일 번들 저장 (kmeans_childhood.pkl 복사본)
MODEL_PATH = "outputs/models/kmeans_rhinitis.pkl"

# ── 클러스터별 설명 & 가이드 (동반질환 기반 k=3 — step3와 일치) ──────
CLUSTER_INFO = {
    "호흡기 알레르기형": {
        "description": (
            "천식·식품알레르기 등 동반 질환 없이 비염 단독으로 진행되는 가장 흔한 유형(약 47%)입니다. "
            "발병 나이가 상대적으로 늦고(평균 7.9세), 콧물·재채기·눈 가려움이 주 증상입니다. "
            "특정 항원(꽃가루, 집먼지진드기 등)에 계절·환경적으로 반응합니다."
        ),
        "guide": [
            "생리식염수 비강 세척으로 원인 항원을 주기적으로 씻어내세요.",
            "외출 시 KF80 이상 마스크와 보호 안경을 착용해 항원 접촉을 줄이세요.",
            "꽃가루 농도가 높은 오전 6~10시에는 환기를 자제하세요.",
            "침구류를 60°C 이상 온수로 세탁하고 집먼지진드기 방지 커버를 사용하세요.",
            "증상이 계절마다 심해지면 항히스타민제 선제 복용을 의사와 상의하세요.",
        ],
    },
    "비염+천식 복합형": {
        "description": (
            "비염과 천식이 함께 동반되는 유형(약 40%)으로, 코와 기관지가 동시에 영향을 받습니다. "
            "발병 나이는 평균 7.0세이며, 코막힘이 심하고 호흡기 전반의 집중 관리가 필요합니다. "
            "천식 악화 시 비염 증상도 함께 심해지는 연동 패턴이 특징입니다."
        ),
        "guide": [
            "취침 전 코 세척으로 비강 분비물을 제거하고 점막 부종을 완화하세요.",
            "스테로이드 비강 분무제를 꾸준히 사용하세요 (효과 발현 3~7일 소요).",
            "천식 증상(쌕쌕거림, 숨가쁨)이 나타나면 즉시 흡입기를 사용하고 전문의를 찾으세요.",
            "실내 습도를 45~55%로 유지하고 차가운 공기에 직접 노출되지 않도록 하세요.",
            "정기적으로 이비인후과 + 호흡기내과를 함께 방문해 두 질환을 통합 관리하세요.",
        ],
    },
    "아토픽 마치형": {
        "description": (
            "영아기 식품알레르기에서 시작해 아토피 피부염 → 비염 → 천식으로 진행되는 유형(약 13%)입니다. "
            "발병 나이가 가장 이르고(평균 5.9세), 식품알레르기 100% 동반·천식 동반율 약 67%로 "
            "복합적인 알레르기 질환 관리가 필요합니다."
        ),
        "guide": [
            "원인 식품(땅콩·우유·달걀·밀·조개류 등)을 정확히 파악하고 철저히 회피하세요.",
            "소아 알레르기 전문의 정기 추적 관찰로 아토픽 마치 진행을 조기에 차단하세요.",
            "피부 보습을 하루 2회 이상 유지하여 아토피 피부염 악화를 예방하세요.",
            "알레르기 면역치료(설하·피하) 가능 여부를 전문의와 상담하세요.",
            "학교·어린이집에 알레르기 정보를 공유하고 에피네프린 자가주사기를 비치하세요.",
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
