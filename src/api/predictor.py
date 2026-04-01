"""
모델 로딩 및 예측 로직
"""
import numpy as np
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = "outputs/models/kmeans_childhood.pkl"

# ── 클러스터별 설명 & 가이드 ──────────────────────────
CLUSTER_INFO = {
    "호흡기 알레르기형": {
        "description": (
            "동반 질환이 없는 순수 비염 유형입니다. "
            "주로 학령기(7~8세) 이후 발병하며 천식·식품알레르기 없이 "
            "비염 증상만 나타납니다."
        ),
        "guide": [
            "실내 습도를 50~60%로 유지하세요.",
            "외출 시 꽃가루 지수를 확인하고 KF80 이상 마스크를 착용하세요.",
            "항히스타민제 복용 전 증상 기록을 2주 이상 쌓아보세요.",
            "귀가 후 손·얼굴 세척으로 알레르기 유발 물질을 제거하세요.",
        ]
    },
    "비염+천식 복합형": {
        "description": (
            "비염과 천식이 함께 나타나는 복합 호흡기 알레르기 유형입니다. "
            "천식 증상(기침, 호흡 곤란)이 동반되므로 호흡기 전반 관리가 필요합니다."
        ),
        "guide": [
            "호흡기 전문의 정기 방문을 권장합니다 (6개월 1회 이상).",
            "기온이 5도 이상 급변하는 날은 실외 활동을 최소화하세요.",
            "미세먼지 '나쁨' 이상 시 반드시 KF94 마스크를 착용하세요.",
            "흡입형 기관지 확장제를 처방받아 상비하세요.",
            "찬 공기 노출 전 코·목 보호대를 착용하세요.",
        ]
    },
    "아토픽 마치형": {
        "description": (
            "어린 시절 식품알레르기로 시작해 아토피→비염→천식 순서로 "
            "진행되는 아토픽 마치(Atopic March) 유형입니다. "
            "다중 알레르기 관리와 조기 개입이 핵심입니다."
        ),
        "guide": [
            "알레르기 전문의를 통해 원인 항원(식품·흡입) 검사를 받으세요.",
            "알레르기 유발 식품 목록을 작성하고 섭취를 조절하세요.",
            "피부 보습 관리를 꾸준히 하여 아토피 악화를 예방하세요.",
            "실내 공기청정기(HEPA 필터)를 사용하고 주 1회 이상 침구를 세탁하세요.",
            "꽃가루 시즌(봄·가을)에는 항히스타민제를 예방적으로 복용하는 방안을 의사와 상담하세요.",
        ]
    },
}

# fallback (예상치 못한 클러스터명)
DEFAULT_INFO = {
    "description": "비염 유형이 분류됐습니다. 전문의 상담을 권장합니다.",
    "guide": ["증상을 꾸준히 기록하고 전문의와 상담하세요."]
}


class RhinitisPredictor:
    def __init__(self):
        self.model    = None
        self.scaler   = None
        self.features = None
        self.label_map = None
        self._load_model()

    def _load_model(self):
        path = Path(MODEL_PATH)
        if not path.exists():
            logger.warning(f"모델 파일 없음: {MODEL_PATH}")
            return
        bundle = joblib.load(path)
        self.model     = bundle["model"]
        self.scaler    = bundle["scaler"]
        self.features  = bundle["features"]
        self.label_map = bundle["label_map"]
        logger.info(f"모델 로드 완료: {MODEL_PATH}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, input_data: dict) -> dict:
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        # 피처 벡터 생성
        X = np.array([[input_data.get(f, 0) for f in self.features]])
        X_scaled = self.scaler.transform(X)

        # 클러스터 예측
        cluster_id = int(self.model.predict(X_scaled)[0])
        label      = self.label_map.get(cluster_id, f"클러스터 {cluster_id}")

        # 신뢰도: 클러스터 중심까지 거리의 역수 (정규화)
        distances  = self.model.transform(X_scaled)[0]
        closest    = distances[cluster_id]
        confidence = float(round(1 / (1 + closest), 4))

        info = CLUSTER_INFO.get(label, DEFAULT_INFO)

        return {
            "cluster_id":    cluster_id,
            "cluster_label": label,
            "confidence":    confidence,
            "description":   info["description"],
            "guide":         info["guide"],
        }


# 싱글톤 인스턴스
predictor = RhinitisPredictor()
