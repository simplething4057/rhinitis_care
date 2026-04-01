"""
Step 3. K-Means 클러스터링 실행
최적 k=3으로 클러스터링 → 결과 저장 + 시각화

클러스터 3종 (Step 2 Silhouette 분석 + 임상 해석 기반 최종 채택):
  - 호흡기 알레르기형  : 동반 질환 없는 순수 비염 (47%)
  - 비염+천식 복합형   : 천식 100% 동반 (40%)
  - 아토픽 마치형      : 식품알레르기 → 아토피 → 비염 진행 (13%)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

from src.data.preprocess import handle_missing_values, remove_outliers_iqr

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

# ── 설정: Step 2 탐색 결과 기반 k=3 확정 ─────────────
N_CLUSTERS = 3

# ── 데이터 로드 ───────────────────────────────────────
df = pd.read_csv('data/processed/rhinitis_features.csv')
print(f"데이터 로드: {len(df):,}명")

# 클러스터링 피처 선택 (동반질환 + 발병 특성 + 증상 프로파일)
feature_cols = [
    'has_asthma', 'has_atopic_derm', 'has_food_allergy',
    'rhinitis_onset_age', 'atopic_march',
    'symptom_rhinorrhea', 'symptom_congestion', 'symptom_sneezing', 'symptom_ocular'
]

# ── preprocess 모듈 연결: 연속형 피처 결측·이상치 처리 ──
cont_cols = ['rhinitis_onset_age']
df_feat = handle_missing_values(df[feature_cols].copy(), strategy='interpolate')
df_feat = remove_outliers_iqr(df_feat, cont_cols)
df[feature_cols] = df_feat.values

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

# ── 클러스터 유형 명명 (동반질환 특성 기반 — k=3 확정) ──
# 우선순위: 천식 동반 여부 → 식품알레르기/아토픽마치 → 나머지(순수 비염)
# 세 유형은 README / PROGRESS.md 문서와 일치:
#   비염+천식 복합형  : has_asthma ≈ 1.0
#   아토픽 마치형     : has_food_allergy 높음 또는 atopic_march 높음
#   호흡기 알레르기형 : 두 조건 모두 해당 없음
label_map = {}
used_labels = set()

for c in range(N_CLUSTERS):
    row = summary.loc[c]
    if row['has_asthma'] > 0.8 and '비염+천식 복합형' not in used_labels:
        lbl = '비염+천식 복합형'
    elif (row['has_food_allergy'] > 0.7 or row['atopic_march'] > 0.3) \
            and '아토픽 마치형' not in used_labels:
        lbl = '아토픽 마치형'
    else:
        lbl = '호흡기 알레르기형'
    label_map[c] = lbl
    used_labels.add(lbl)

df['cluster_label'] = df['cluster'].map(label_map)

print("\n=== 클러스터 유형 분류 (동반질환 기반, k=3) ===")
for c, label in sorted(label_map.items()):
    count = (df['cluster'] == c).sum()
    pct   = count / len(df) * 100
    print(f"  클러스터 {c} → {label:20s}: {count:,}명 ({pct:.1f}%)")

# ── 시각화 1: 증상 프로파일 레이더 차트 (가독성을 위해 상세 바 차트 대신) ──
# (기존 바 차트 코드를 유지하되 피처만 변경)
fig, ax = plt.subplots(figsize=(12, 6))
plot_cols = ['symptom_rhinorrhea', 'symptom_congestion', 'symptom_sneezing', 'symptom_ocular']
col_labels = ['콧물', '코막힘', '재채기', '눈 증상']

x = np.arange(len(plot_cols))
width = 0.25                                       # k=3이므로 0.25 (4일 땐 0.2)
colors = ['#4C72B0', '#DD8452', '#55A868']         # 3색

for i, (c, label) in enumerate(sorted(label_map.items())):
    vals = [summary.loc[c, col] for col in plot_cols]
    ax.bar(x + i * width, vals, width,
           label=label, color=colors[i % len(colors)], alpha=0.8)

ax.set_xticks(x + width)                           # 3개 클러스터 기준 중앙 정렬
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_ylabel('증상 강도 (0~10)')
ax.set_ylim(0, 10)
ax.set_title('비염 유형별 증상 프로파일 (K-Means Clustering, k=3)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/step3_cluster_profiles.png', dpi=150, bbox_inches='tight')

# ── 모델 저장 ─────────────────────────────────────────
# kmeans_childhood.pkl : 코호트 학습 원본
# kmeans_rhinitis.pkl  : API predictor 참조 파일 (동일 번들 복사)
os.makedirs('outputs/models', exist_ok=True)
bundle = {'model': km, 'scaler': scaler,
          'features': feature_cols, 'label_map': label_map}
joblib.dump(bundle, 'outputs/models/kmeans_childhood.pkl')
joblib.dump(bundle, 'outputs/models/kmeans_rhinitis.pkl')   # predictor.py 참조용
df.to_csv('data/processed/rhinitis_clustered.csv',
          index=False, encoding='utf-8-sig')

print("\n모델 저장: outputs/models/kmeans_childhood.pkl")
print("모델 저장: outputs/models/kmeans_rhinitis.pkl  (predictor.py 참조용)")
print("결과 저장: data/processed/rhinitis_clustered.csv")
print("\n클러스터링 완료!")
