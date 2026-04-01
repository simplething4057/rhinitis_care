"""
Step 3. K-Means 클러스터링 실행
최적 k로 클러스터링 → 결과 저장 + 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# ── 한글 폰트 설정 ────────────────────────────────────
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Malgun Gothic' in available_fonts:
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif 'AppleGothic' in available_fonts:
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 50)
print("Step 3. K-Means 클러스터링")
print("=" * 50)

# ── 설정: step2 결과를 보고 k 조정 ───────────────────
N_CLUSTERS = 3   # step2 Silhouette 결과에 맞게 수정하세요

# ── 데이터 로드 ───────────────────────────────────────
df = pd.read_csv('data/processed/rhinitis_features.csv')
print(f"데이터 로드: {len(df):,}명")

feature_cols = [
    'has_asthma', 'has_atopic_derm', 'has_food_allergy',
    'food_allergy_count', 'rhinitis_onset_age',
    'rhinitis_duration', 'atopic_march'
]

X_raw = df[feature_cols].fillna(0).values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── K-Means 학습 ──────────────────────────────────────
print(f"\nK-Means 학습 중 (k={N_CLUSTERS})...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
df['cluster'] = km.fit_predict(X)
print("학습 완료")

# ── 클러스터별 통계 분석 ──────────────────────────────
print("\n=== 클러스터별 규모 ===")
print(df['cluster'].value_counts().sort_index())

summary = df.groupby('cluster')[feature_cols].mean().round(3)
print("\n=== 클러스터별 피처 평균 ===")
print(summary.to_string())

# ── 클러스터 유형 자동 명명 ───────────────────────────
# 천식 동반 비율이 가장 높은 클러스터 → "비염+천식 복합형"
# 아토픽 마치 비율이 가장 높은 클러스터 → "아토픽 마치형"
# 나머지 → "비염 단독형"
asthma_cluster  = summary['has_asthma'].idxmax()
march_cluster   = summary['atopic_march'].idxmax()
single_clusters = [c for c in range(N_CLUSTERS)
                   if c != asthma_cluster and c != march_cluster]
single_cluster  = single_clusters[0] if single_clusters else 0

label_map = {
    asthma_cluster : '비염+천식 복합형',
    march_cluster  : '아토픽 마치형',
    single_cluster : '호흡기 알레르기형',
}
# 클러스터 수가 3이 아닐 때 fallback
for c in range(N_CLUSTERS):
    if c not in label_map:
        label_map[c] = f'클러스터 {c}'

df['cluster_label'] = df['cluster'].map(label_map)

print("\n=== 클러스터 유형 분류 ===")
for c, label in sorted(label_map.items()):
    count = (df['cluster'] == c).sum()
    pct   = count / len(df) * 100
    print(f"  클러스터 {c} → {label:15s}: {count:,}명 ({pct:.1f}%)")

# ── 시각화 1: 클러스터별 피처 프로파일 바 차트 ──────
fig, ax = plt.subplots(figsize=(12, 5))
plot_cols = ['has_asthma', 'has_atopic_derm', 'has_food_allergy', 'atopic_march']
col_labels = ['천식 동반율', '아토피 동반율', '식품알레르기율', '아토픽 마치율']

x = np.arange(len(plot_cols))
width = 0.25
colors = ['#4C72B0', '#DD8452', '#55A868']

for i, (c, label) in enumerate(sorted(label_map.items())):
    vals = [summary.loc[c, col] for col in plot_cols]
    ax.bar(x + i * width, vals, width,
           label=label, color=colors[i % len(colors)], alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_ylabel('비율 (0~1)')
ax.set_ylim(0, 1)
ax.set_title('비염 유형별 동반 질환 프로파일', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/step3_cluster_profiles.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장: outputs/figures/step3_cluster_profiles.png")
plt.show()

# ── 시각화 2: 발병 나이 분포 (클러스터별 비교) ──────
fig, ax = plt.subplots(figsize=(9, 4))
for c, label in sorted(label_map.items()):
    subset = df[df['cluster'] == c]['rhinitis_onset_age']
    ax.hist(subset, bins=30, alpha=0.6, label=label, density=True)
ax.set_xlabel('비염 발병 나이 (세)')
ax.set_ylabel('밀도')
ax.set_title('클러스터별 비염 발병 나이 분포', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/step3_onset_age_dist.png', dpi=150, bbox_inches='tight')
print("그래프 저장: outputs/figures/step3_onset_age_dist.png")
plt.show()

# ── 모델 저장 ─────────────────────────────────────────
os.makedirs('outputs/models', exist_ok=True)
joblib.dump({'model': km, 'scaler': scaler,
             'features': feature_cols, 'label_map': label_map},
            'outputs/models/kmeans_childhood.pkl')
df.to_csv('data/processed/rhinitis_clustered.csv',
          index=False, encoding='utf-8-sig')

print("\n모델 저장: outputs/models/kmeans_childhood.pkl")
print("결과 저장: data/processed/rhinitis_clustered.csv")
print("\n클러스터링 완료!")
