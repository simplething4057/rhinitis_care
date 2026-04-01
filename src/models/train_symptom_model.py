"""
증상 기반 K-Means 클러스터링 학습 스크립트

실제 비염 증상 점수(VAS 0~10) 데이터가 없으므로,
임상적으로 의미 있는 3가지 유형의 대표 프로파일을 기반으로
학습 데이터를 생성하고 K-Means 모델을 학습합니다.

유형 정의:
  - 콧물·재채기 우세형: 콧물·재채기 높음, 코막힘 낮음
  - 코막힘 우세형:      코막힘 높음, 콧물·재채기 낮음
  - 복합 과민형:        눈 증상 높음 + 전반적 중등도 이상

출력: outputs/models/kmeans_rhinitis.pkl
"""

import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────
FEATURES = [
    "symptom_rhinorrhea",   # 콧물
    "symptom_congestion",   # 코막힘
    "symptom_sneezing",     # 재채기·가려움
    "symptom_ocular",       # 눈 증상
]
N_PER_CLUSTER = 600   # 클러스터당 샘플 수
RANDOM_STATE  = 42
OUTPUT_PATH   = "outputs/models/kmeans_rhinitis.pkl"

# ── 임상 유형별 증상 프로파일 (mean, std) ─────────────
# 각 행: [rhinorrhea, congestion, sneezing, ocular]
PROFILES = {
    "콧물·재채기 우세형": {
        "means": [8.0, 3.5, 8.0, 5.0],
        "stds":  [1.0, 1.5, 1.0, 1.5],
    },
    "코막힘 우세형": {
        "means": [3.0, 8.5, 3.0, 2.5],
        "stds":  [1.5, 1.0, 1.5, 1.2],
    },
    "복합 과민형": {
        "means": [5.5, 5.5, 5.5, 8.0],
        "stds":  [1.5, 1.5, 1.5, 1.0],
    },
}


def _generate_data(rng: np.random.Generator) -> np.ndarray:
    """3가지 프로파일 기반 학습 데이터 생성."""
    blocks = []
    for profile in PROFILES.values():
        block = rng.normal(
            loc=profile["means"],
            scale=profile["stds"],
            size=(N_PER_CLUSTER, len(FEATURES)),
        )
        block = np.clip(block, 0, 10)   # 0~10 범위 클리핑
        blocks.append(block)
    return np.vstack(blocks)


def _assign_labels(km: KMeans) -> dict[int, str]:
    """
    클러스터 중심 분석으로 레이블 자동 할당.
    - 코막힘(idx 1) 최고 → "코막힘 우세형"
    - 눈 증상(idx 3) 최고 → "복합 과민형"
    - 나머지 → "콧물·재채기 우세형"
    """
    centers = km.cluster_centers_  # shape: (3, 4)
    congestion_idx = int(np.argmax(centers[:, 1]))  # 코막힘 최고 클러스터
    ocular_idx     = int(np.argmax(centers[:, 3]))  # 눈 증상 최고 클러스터
    if congestion_idx == ocular_idx:
        # 충돌 시: 코막힘 우선, 눈 증상은 두 번째로 높은 클러스터에 배정
        sorted_ocular = np.argsort(centers[:, 3])[::-1]
        ocular_idx = int(sorted_ocular[1])

    label_map = {}
    for k in range(km.n_clusters):
        if k == congestion_idx:
            label_map[k] = "코막힘 우세형"
        elif k == ocular_idx:
            label_map[k] = "복합 과민형"
        else:
            label_map[k] = "콧물·재채기 우세형"
    return label_map


def train_and_save():
    rng = np.random.default_rng(RANDOM_STATE)
    X_raw = _generate_data(rng)
    logger.info(f"학습 데이터 생성 완료: {X_raw.shape}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    km.fit(X)
    logger.info(f"K-Means 학습 완료 (inertia={km.inertia_:.2f})")

    label_map = _assign_labels(km)
    logger.info(f"레이블 할당: {label_map}")

    bundle = {
        "model":     km,
        "scaler":    scaler,
        "features":  FEATURES,
        "label_map": label_map,
    }

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    logger.info(f"모델 저장 완료: {OUTPUT_PATH}")

    # 저장 검증
    b = joblib.load(out)
    assert b["features"] == FEATURES
    assert set(b["label_map"].values()) == {"콧물·재채기 우세형", "코막힘 우세형", "복합 과민형"}
    logger.info("저장 검증 통과")
    return bundle


if __name__ == "__main__":
    train_and_save()
