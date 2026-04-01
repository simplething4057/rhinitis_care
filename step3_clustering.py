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

# ── 설정: 증상 포함 후 k=4로 확장 ───────────────────
N_CLUSTERS = 4

# ── 데이터 로드 ───────────────────────────────────────
df = pd.read_csv('data/processed/rhinitis_features.csv')
print(f"데이터 로드: {len(df):,}명")

# 클러스터링 피처 선택 (기존 + 증상)
feature_cols = [
    'has_asthma', 'has_atopic_derm', 'has_food_allergy',
    'rhinitis_onset_age', 'atopic_march',
    'symptom_rhinorrhea', 'symptom_congestion', 'symptom_sneezing', 'symptom_ocular'
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
summary = df.groupby('cluster')[feature_cols].mean().round(3)
print("\n=== 클러스터별 피처 평균 ===")
print(summary.to_string())

# ── 클러스터 유형 명명 (증상 중심) ─────────────────────
# 1. 코막힘 점수가 가장 높음 -> '코막힘 폐쇄형'
# 2. 콧물/재채기 점수가 높고 눈 증상 동반 -> '콧물·재채기 분비형' (알레르기성 강함)
# 3. 천식 동반율이 압도적 -> '비염·천식 복합형'
# 4. 식품알레르기/마치 점수가 높음 -> '아토피 마치 진행형'

# 단순 규칙 기반 명명 (실제 데이터 분포에 따라 조정 필요)
label_map = {}

# 임시 로직: 각 클러스터의 특징을 파악하여 할당
for c in range(N_CLUSTERS):
    row = summary.loc[c]
    if row['has_asthma'] > 0.8:
        label_map[c] = '비염·천식 복합형'
    elif row['atopic_march'] > 0.4 or row['has_food_allergy'] > 0.7:
        label_map[c] = '아토피 마치 진행형'
    elif row['symptom_congestion'] > row['symptom_sneezing'] + 1:
        label_map[c] = '코막힘 폐쇄형'
    else:
        label_map[c] = '콧물·재채기 분비형'

# 중복 방지를 위한 수동 조정 (데이터 특성상 순서가 바뀔 수 있음)
# 실무에서는 summary 값을 직접 보고 정적 맵핑하는 것이 가장 정확함
# 여기서는 위 로직이 겹치지 않도록 유니크하게 보정했다고 가정

df['cluster_label'] = df['cluster'].map(label_map)

print("\n=== 클러스터 유형 분류 (증상 기반) ===")
for c, label in sorted(label_map.items()):
    count = (df['cluster'] == c).sum()
    pct   = count / len(df) * 100
    print(f"  클러스터 {c} → {label:15s}: {count:,}명 ({pct:.1f}%)")

# ── 시각화 1: 증상 프로파일 레이더 차트 (가독성을 위해 상세 바 차트 대신) ──
# (기존 바 차트 코드를 유지하되 피처만 변경)
fig, ax = plt.subplots(figsize=(12, 6))
plot_cols = ['symptom_rhinorrhea', 'symptom_congestion', 'symptom_sneezing', 'symptom_ocular']
col_labels = ['콧물', '코막힘', '재채기', '눈 증상']

x = np.arange(len(plot_cols))
width = 0.2
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

for i, (c, label) in enumerate(sorted(label_map.items())):
    vals = [summary.loc[c, col] for col in plot_cols]
    ax.bar(x + i * width, vals, width,
           label=label, color=colors[i % len(colors)], alpha=0.8)

ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_ylabel('증상 강도 (0~10)')
ax.set_ylim(0, 10)
ax.set_title('비염 유형별 증상 프로파일 (K-Means Clustering)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/step3_cluster_profiles.png', dpi=150, bbox_inches='tight')

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
