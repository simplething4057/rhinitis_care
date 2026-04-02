"""
모델 로딩 및 예측 로직
---------------------------------------------------------------------------
우선순위
  1순위 : LightGBM 분류기 (lgbm_rhinitis.pkl)
          - step6_lightgbm.py 실행 후 outputs/models/ 에 생성됨
          - predict_proba() 로 보정된 확률 점수 제공
  2순위 : K-Means (kmeans_rhinitis.pkl)
          - step3_clustering.py 실행 후 생성됨
          - 거리 역수 softmax 로 확률 근사

클러스터 3종: 호흡기 알레르기형 / 비염+천식 복합형 / 아토픽 마치형
"""
import numpy as np
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LGBM_PATH   = "outputs/models/lgbm_rhinitis.pkl"
KMEANS_PATH = "outputs/models/kmeans_rhinitis.pkl"

# ── LightGBM이 사용하는 임상 피처 ─────────────────────────────────
LGBM_FEATURE_COLS = [
    "has_asthma", "has_atopic_derm", "has_food_allergy",
    "food_allergy_count", "rhinitis_onset_age", "rhinitis_duration",
    "atopic_march", "is_female",
]

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

DEFAULT_INFO = {
    "description": "비염 유형이 분류됐습니다. 전문의 상담을 권장합니다.",
    "guide": ["정밀 검사를 통해 정확한 유형을 파악하고 맞춤 치료를 시작하세요."],
}


class RhinitisPredictor:
    """
    LightGBM 우선 / K-Means 폴백 예측기.

    반환 dict 공통 키
      - cluster_id    : int (0~2)
      - cluster_label : str
      - confidence    : float  (예측 클래스 확률)
      - cluster_probs : dict   {label: prob}
      - description   : str
      - guide         : list[str]
      - model_type    : 'lightgbm' | 'kmeans'
    """

    def __init__(self):
        self.model_type = None
        # LightGBM 번들 키
        self._lgbm_model    = None
        self._lgbm_features = None
        self._lgbm_label_map = None
        # K-Means 번들 키
        self._km_model     = None
        self._km_scaler    = None
        self._km_features  = None
        self._km_label_map = None

        self._load_model()

    # ------------------------------------------------------------------
    # 로딩
    # ------------------------------------------------------------------
    def _load_model(self):
        # 1순위: LightGBM
        lgbm_path = Path(LGBM_PATH)
        if lgbm_path.exists():
            try:
                bundle = joblib.load(lgbm_path)
                self._lgbm_model     = bundle["model"]
                self._lgbm_features  = bundle.get("features", LGBM_FEATURE_COLS)
                self._lgbm_label_map = bundle["label_map"]
                self.model_type      = "lightgbm"
                logger.info(
                    f"LightGBM 로드 완료: {LGBM_PATH}  "
                    f"cv_acc={bundle.get('cv_acc', '?'):.4f}"
                )
                return
            except Exception as exc:
                logger.warning(f"LightGBM 로드 실패({exc}), K-Means fallback 시도")

        # 2순위: K-Means
        km_path = Path(KMEANS_PATH)
        if km_path.exists():
            try:
                bundle = joblib.load(km_path)
                required = {"model", "scaler", "features", "label_map"}
                if required - bundle.keys():
                    raise ValueError(f"번들 키 누락: {required - bundle.keys()}")
                self._km_model     = bundle["model"]
                self._km_scaler    = bundle["scaler"]
                self._km_features  = bundle["features"]
                self._km_label_map = bundle["label_map"]
                self.model_type    = "kmeans"
                logger.info(f"K-Means 로드 완료: {KMEANS_PATH}")
            except Exception as exc:
                logger.error(f"K-Means 로드 실패: {exc}")
        else:
            logger.warning(f"모델 파일 없음: {LGBM_PATH}, {KMEANS_PATH}")

    @property
    def is_loaded(self) -> bool:
        return self.model_type is not None

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------
    def predict(self, input_data: dict) -> dict:
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        if self.model_type == "lightgbm":
            return self._predict_lgbm(input_data)
        return self._predict_kmeans(input_data)

    # ── LightGBM 예측 ─────────────────────────────────────────────────
    def _predict_lgbm(self, input_data: dict) -> dict:
        X = np.array([[input_data.get(f, 0) for f in self._lgbm_features]])

        cluster_id = int(self._lgbm_model.predict(X)[0])
        probs_arr  = self._lgbm_model.predict_proba(X)[0]   # shape (3,)

        label = self._lgbm_label_map.get(cluster_id, f"클러스터 {cluster_id}")
        cluster_probs = {
            self._lgbm_label_map.get(i, f"클러스터 {i}"): float(round(probs_arr[i], 4))
            for i in range(len(self._lgbm_label_map))
        }
        confidence = float(round(probs_arr[cluster_id], 4))
        info = CLUSTER_INFO.get(label, DEFAULT_INFO)

        return {
            "cluster_id":    cluster_id,
            "cluster_label": label,
            "confidence":    confidence,
            "cluster_probs": cluster_probs,
            "description":   info["description"],
            "guide":         info["guide"],
            "model_type":    "lightgbm",
        }

    # ── K-Means 예측 (폴백) ──────────────────────────────────────────
    def _predict_kmeans(self, input_data: dict) -> dict:
        X = np.array([[input_data.get(f, 0) for f in self._km_features]])
        X_scaled = self._km_scaler.transform(X)

        cluster_id = int(self._km_model.predict(X_scaled)[0])
        label      = self._km_label_map.get(cluster_id, f"클러스터 {cluster_id}")

        # 거리 역수 softmax로 확률 근사
        distances  = self._km_model.transform(X_scaled)[0]
        inv_dist   = 1.0 / (1.0 + distances)
        probs_arr  = inv_dist / inv_dist.sum()
        confidence = float(round(probs_arr[cluster_id], 4))

        cluster_probs = {
            self._km_label_map.get(i, f"클러스터 {i}"): float(round(probs_arr[i], 4))
            for i in range(len(self._km_label_map))
        }
        info = CLUSTER_INFO.get(label, DEFAULT_INFO)

        return {
            "cluster_id":    cluster_id,
            "cluster_label": label,
            "confidence":    confidence,
            "cluster_probs": cluster_probs,
            "description":   info["description"],
            "guide":         info["guide"],
            "model_type":    "kmeans",
        }


# 싱글톤 인스턴스
predictor = RhinitisPredictor()
