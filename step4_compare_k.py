"""
Step 4. k=2 vs k=3 vs k=4 클러스터 비교
각 k별 클러스터 특성을 나란히 출력하여 최적 유형 수 결정
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

print("=" * 60)
print("Step 4. k=2 / k=3 / k=4 클러스터 비교")
print("=" * 60)

# ── 데이터 로드 & 표준화 ──────────────────────────────
df = pd.read_csv('data/processed/rhinitis_features.csv')
feature_cols = [
    'has_asthma', 'has_atopic_derm', 'has_food_allergy',
    'food_allergy_count', 'rhinitis_onset_age',
    'rhinitis_duration', 'atopic_march'
]
X_raw = df[feature_cols].fillna(0).values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# 실루엣용 샘플
np.random.seed(42)
idx = np.random.choice(len(X), 10000, replace=False)
X_sample = X[idx]

# ── k별 클러스터링 및 요약 출력 ───────────────────────
plot_cols   = ['has_asthma', 'has_atopic_derm', 'has_food_allergy', 'atopic_march']
col_labels  = ['천식 동반율', '아토피 동반율', '식품알레르기율', '아토픽 마치율']
colors_list = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax_idx, k in enumerate([2, 3, 4]):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_full   = km.fit_predict(X)
    labels_sample = km.predict(X_sample)
    sil = silhouette_score(X_sample, labels_sample)

    df_temp = df.copy()
    df_temp['cluster'] = labels_full
    summary = df_temp.groupby('cluster')[feature_cols].mean().round(3)

    # ── 콘솔 출력 ──
    print(f"\n{'='*55}")
    print(f"k = {k}  (Silhouette = {sil:.4f})")
    print(f"{'='*55}")
    print(f"{'클러스터':>5} | {'규모':>7} | {'비율':>6} | "
          f"{'천식':>5} | {'아토피':>6} | {'식품알레르기':>10} | "
          f"{'발병나이':>8} | {'아토픽마치':>9}")
    print("-" * 75)
    for c in range(k):
        cnt = (df_temp['cluster'] == c).sum()
        pct = cnt / len(df_temp) * 100
        row = summary.loc[c]
        print(f"  클러스터{c}  | {cnt:>7,} | {pct:>5.1f}% | "
              f"{row['has_asthma']:>5.3f} | {row['has_atopic_derm']:>6.3f} | "
              f"{row['has_food_allergy']:>10.3f} | "
              f"{row['rhinitis_onset_age']:>8.2f} | "
              f"{row['atopic_march']:>9.3f}")

    # ── 시각화 ──
    ax = axes[ax_idx]
    x = np.arange(len(plot_cols))
    width = 0.8 / k

    for i in range(k):
        vals = [summary.loc[i, col] for col in plot_cols]
        ax.bar(x + i * width - (k - 1) * width / 2, vals, width,
               label=f'클러스터 {i}',
               color=colors_list[i % len(colors_list)], alpha=0.85)

    ax.set_title(f'k = {k}  (Silhouette={sil:.3f})', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(col_labels, fontsize=9, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('비율 (0~1)' if ax_idx == 0 else '')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('k별 클러스터 동반질환 프로파일 비교', fontsize=15, y=1.02)
plt.tight_layout()

os.makedirs('outputs/figures', exist_ok=True)
plt.savefig('outputs/figures/step4_k_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n\n그래프 저장: outputs/figures/step4_k_comparison.png")

# ── k별 Silhouette 최종 비교 ──────────────────────────
print("\n=== 최종 Silhouette 비교 ===")
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    sil = silhouette_score(X_sample, km.fit_predict(X_sample))
    bar = '█' * int(sil * 40)
    print(f"  k={k}: {sil:.4f}  {bar}")

print("\n위 결과를 보고 step3_clustering.py 의 N_CLUSTERS 값을 최종 결정하세요.")
plt.show()
