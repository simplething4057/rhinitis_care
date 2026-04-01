"""
증상 클러스터링 모델 모듈
- K-Means 기반 비염 유형 분류 (콧물형 / 코막힘형 / 자극민감형)
- 최적 클러스터 수 탐색 (Elbow Method, Silhouette Score)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


CLUSTER_LABELS = {
    0: "콧물-재채기 우세형",
    1: "코막힘 우세형",
    2: "자극 민감형",
}

VAS_FEATURES = ["nasal_discharge", "nasal_congestion", "sneezing", "eye_itching"]


# ──────────────────────────────────────────────
# 1. 최적 클러스터 수 탐색
# ──────────────────────────────────────────────
def find_optimal_k(X: np.ndarray, k_range: range = range(2, 8), save_path: str | None = None):
    """Elbow Method + Silhouette Score로 최적 k 탐색."""
    inertias, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        logger.info(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil_scores[-1]:.4f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    k_list = list(k_range)

    axes[0].plot(k_list, inertias, "bo-")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("클러스터 수 (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_list, sil_scores, "rs-")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("클러스터 수 (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Elbow 그래프 저장: {save_path}")
    plt.show()

    best_k = k_list[np.argmax(sil_scores)]
    logger.info(f"최적 k 추천: {best_k} (silhouette={max(sil_scores):.4f})")
    return best_k


# ──────────────────────────────────────────────
# 2. K-Means 학습
# ──────────────────────────────────────────────
def train_kmeans(
    df: pd.DataFrame,
    features: list = VAS_FEATURES,
    n_clusters: int = 3,
    model_save_path: str | None = None,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    K-Means 학습 후 클러스터 레이블 부여.
    Returns: (df with 'cluster' column, model, scaler)
    """
    X_raw = df[features].dropna().values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    # 원본 df에 클러스터 컬럼 추가 (결측치 행은 NaN)
    df = df.copy()
    valid_idx = df[features].dropna().index
    df.loc[valid_idx, "cluster"] = labels
    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS)

    logger.info(f"K-Means 학습 완료 (k={n_clusters})")
    logger.info(f"클러스터 분포:\n{df['cluster_label'].value_counts()}")

    if model_save_path:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": km, "scaler": scaler}, model_save_path)
        logger.info(f"모델 저장: {model_save_path}")

    return df, km, scaler


# ──────────────────────────────────────────────
# 3. 클러스터 프로파일 시각화
# ──────────────────────────────────────────────
def plot_cluster_profiles(df: pd.DataFrame, features: list = VAS_FEATURES, save_path: str | None = None):
    """클러스터별 평균 VAS 점수 레이더/바 차트."""
    cluster_means = df.groupby("cluster_label")[features].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(features))
    width = 0.25

    for i, (label, row) in enumerate(cluster_means.iterrows()):
        ax.bar(x + i * width, row.values, width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["콧물", "코막힘", "재채기", "눈가려움"], fontsize=11)
    ax.set_ylabel("평균 VAS 점수")
    ax.set_title("비염 유형별 증상 프로파일")
    ax.legend()
    ax.set_ylim(0, 10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"클러스터 프로파일 저장: {save_path}")
    plt.show()


# ──────────────────────────────────────────────
# 4. 모델 로드 및 예측
# ──────────────────────────────────────────────
def predict_cluster(model_path: str, symptom_scores: dict) -> str:
    """
    저장된 모델로 새 사용자 증상 분류.
    symptom_scores: {'nasal_discharge': 7, 'nasal_congestion': 3, ...}
    """
    bundle = joblib.load(model_path)
    km: KMeans = bundle["model"]
    scaler: StandardScaler = bundle["scaler"]

    X = np.array([[symptom_scores.get(f, 0) for f in VAS_FEATURES]])
    X_scaled = scaler.transform(X)
    cluster_id = km.predict(X_scaled)[0]
    label = CLUSTER_LABELS.get(int(cluster_id), "알 수 없음")
    logger.info(f"예측 유형: {label} (cluster={cluster_id})")
    return label


if __name__ == "__main__":
    # 간단한 더미 테스트
    np.random.seed(42)
    dummy = pd.DataFrame({
        "nasal_discharge": np.random.randint(0, 10, 100),
        "nasal_congestion": np.random.randint(0, 10, 100),
        "sneezing":         np.random.randint(0, 10, 100),
        "eye_itching":      np.random.randint(0, 10, 100),
    })
    df_result, model, scaler = train_kmeans(dummy, n_clusters=3)
    plot_cluster_profiles(df_result)
