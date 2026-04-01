"""
Step 2. 최적 클러스터 수 탐색
Elbow Method + Silhouette Score로 최적 k 결정
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# ── 한글 폰트 설정 ────────────────────────────────────
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Malgun Gothic' in available_fonts:
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif 'AppleGothic' in available_fonts:
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 50)
print("Step 2. 최적 클러스터 수 탐색")
print("=" * 50)

# ── 데이터 로드 ───────────────────────────────────────
df = pd.read_csv('data/processed/rhinitis_features.csv')
print(f"데이터 로드: {len(df):,}명\n")

# ── 클러스터링 피처 선택 ──────────────────────────────
feature_cols = [
    'has_asthma', 'has_atopic_derm', 'has_food_allergy',
    'food_allergy_count', 'rhinitis_onset_age',
    'rhinitis_duration', 'atopic_march'
]

X_raw = df[feature_cols].fillna(0).values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f"클러스터링 피처: {feature_cols}\n")

# ── Elbow + Silhouette 계산 ───────────────────────────
# 대용량이므로 샘플링하여 속도 확보
SAMPLE_SIZE = 10000
np.random.seed(42)
if len(X) > SAMPLE_SIZE:
    idx = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
    X_sample = X[idx]
    print(f"속도를 위해 {SAMPLE_SIZE:,}명 샘플로 탐색 (전체 {len(X):,}명)")
else:
    X_sample = X

k_range = range(2, 9)
inertias, sil_scores = [], []

print("\nk별 결과:")
print(f"{'k':>4} | {'Inertia':>12} | {'Silhouette':>10}")
print("-" * 32)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sample)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_sample, labels)
    sil_scores.append(sil)
    print(f"{k:>4} | {km.inertia_:>12.1f} | {sil:>10.4f}")

# ── 시각화 ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
k_list = list(k_range)

axes[0].plot(k_list, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Elbow Method (Inertia)', fontsize=13)
axes[0].set_xlabel('클러스터 수 (k)')
axes[0].set_ylabel('Inertia')
axes[0].grid(alpha=0.3)

axes[1].plot(k_list, sil_scores, 'rs-', linewidth=2, markersize=8)
axes[1].set_title('Silhouette Score', fontsize=13)
axes[1].set_xlabel('클러스터 수 (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(alpha=0.3)

# 최적 k 표시
best_k = k_list[np.argmax(sil_scores)]
axes[1].axvline(best_k, color='green', linestyle='--', label=f'최적 k={best_k}')
axes[1].legend()

plt.suptitle('Childhood Allergies - 최적 클러스터 수 탐색', fontsize=14, y=1.02)
plt.tight_layout()

os.makedirs('outputs/figures', exist_ok=True)
plt.savefig('outputs/figures/step2_optimal_k.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: outputs/figures/step2_optimal_k.png")

print(f"\n==> 추천 클러스터 수: k = {best_k} (Silhouette = {max(sil_scores):.4f})")
print("step3_clustering.py 를 실행하세요.")

plt.show()
