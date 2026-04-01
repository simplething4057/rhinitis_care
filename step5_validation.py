"""
Step 5. 클러스터링 일반화 성능 검증
========================================
비지도 학습(K-Means)은 train/test 정확도 개념이 없으므로
아래 3가지 방법으로 모델 신뢰성을 정량 검증합니다.

  [A] 홀드아웃 세트 클러스터 분포 안정성
      ─ 80% 학습 / 20% 홀드아웃 분리 후
        홀드아웃에서의 클러스터 비율이 학습 세트와 유사한지 확인

  [B] 다중 랜덤 시드 Silhouette 검증
      ─ 시드 10개로 반복 학습 → Silhouette 평균·표준편차 계산
        표준편차가 작을수록 클러스터 구조가 안정적

  [C] Bootstrap 클러스터 안정성 (Jaccard 유사도)
      ─ 부트스트랩 샘플(n=50)로 반복 학습 후
        기준 모델과의 클러스터 멤버십 Jaccard 평균 산출
        0.75 이상이면 안정적인 클러스터로 판단
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
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment

# ── 한글 폰트 ─────────────────────────────────────────
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Malgun Gothic' in available_fonts:
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif 'AppleGothic' in available_fonts:
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Step 5. 클러스터링 일반화 성능 검증")
print("=" * 60)

# ── 데이터 로드 & 표준화 ──────────────────────────────
df = pd.read_csv('data/processed/rhinitis_features.csv')

feature_cols = [
    'has_asthma', 'has_atopic_derm', 'has_food_allergy',
    'rhinitis_onset_age', 'atopic_march',
    'symptom_rhinorrhea', 'symptom_congestion', 'symptom_sneezing', 'symptom_ocular'
]

X_raw = df[feature_cols].fillna(0).values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

N_CLUSTERS = 3
RANDOM_SEED = 42
print(f"\n데이터: {len(df):,}명 / 피처: {len(feature_cols)}개 / k={N_CLUSTERS}\n")


# ══════════════════════════════════════════════════════
# [A] 홀드아웃 세트 클러스터 분포 안정성 검증
# ══════════════════════════════════════════════════════
print("─" * 60)
print("[A] 홀드아웃 세트 클러스터 분포 안정성 검증 (80/20 분리)")
print("─" * 60)

X_train, X_holdout = train_test_split(X, test_size=0.2, random_state=RANDOM_SEED)

km_ref = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10, max_iter=300)
km_ref.fit(X_train)

train_labels   = km_ref.predict(X_train)
holdout_labels = km_ref.predict(X_holdout)

train_dist   = np.bincount(train_labels,   minlength=N_CLUSTERS) / len(X_train)
holdout_dist = np.bincount(holdout_labels, minlength=N_CLUSTERS) / len(X_holdout)

print(f"\n{'클러스터':>6} | {'학습 비율':>10} | {'홀드아웃 비율':>12} | {'차이':>8}")
print("-" * 44)
max_drift = 0.0
for c in range(N_CLUSTERS):
    drift = abs(train_dist[c] - holdout_dist[c])
    max_drift = max(max_drift, drift)
    print(f"  클러스터 {c} | {train_dist[c]:>9.3f}  | {holdout_dist[c]:>11.3f}  | {drift:>7.4f}")

print(f"\n최대 클러스터 비율 드리프트: {max_drift:.4f}")
if max_drift < 0.05:
    print("  → ✅ 클러스터 분포 매우 안정  (드리프트 < 5%)")
elif max_drift < 0.10:
    print("  → ✅ 클러스터 분포 안정       (드리프트 < 10%)")
else:
    print("  → ⚠️  클러스터 분포 불안정    (드리프트 ≥ 10%, 재검토 권장)")

sil_train   = silhouette_score(X_train,   train_labels)
sil_holdout = silhouette_score(X_holdout, holdout_labels)
print(f"\nSilhouette — 학습: {sil_train:.4f} / 홀드아웃: {sil_holdout:.4f} "
      f"(차이: {abs(sil_train - sil_holdout):.4f})")


# ══════════════════════════════════════════════════════
# [B] 다중 랜덤 시드 Silhouette 검증
# ══════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[B] 다중 랜덤 시드 Silhouette 검증 (시드 10개, 샘플 10,000명)")
print("─" * 60)

# 대용량이므로 샘플링
np.random.seed(RANDOM_SEED)
sample_idx = np.random.choice(len(X), min(10_000, len(X)), replace=False)
X_sample   = X[sample_idx]

SEEDS = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]
sil_results = []

print(f"\n{'시드':>6} | {'Silhouette':>12} | {'클러스터 크기 분포'}")
print("-" * 55)

for seed in SEEDS:
    km_s = KMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=10)
    labels_s = km_s.fit_predict(X_sample)
    sil_s    = silhouette_score(X_sample, labels_s)
    sizes    = np.bincount(labels_s, minlength=N_CLUSTERS)
    sil_results.append(sil_s)
    print(f"  {seed:>4} | {sil_s:>12.4f} | {sizes}")

sil_mean = np.mean(sil_results)
sil_std  = np.std(sil_results)
sil_min  = np.min(sil_results)
sil_max  = np.max(sil_results)

print(f"\n  평균: {sil_mean:.4f}  표준편차: {sil_std:.4f}  "
      f"범위: [{sil_min:.4f}, {sil_max:.4f}]")

if sil_std < 0.01:
    print("  → ✅ 시드 변화에 매우 강건  (std < 0.01)")
elif sil_std < 0.02:
    print("  → ✅ 시드 변화에 안정적    (std < 0.02)")
else:
    print("  → ⚠️  시드 민감성 존재     (std ≥ 0.02, 클러스터 구조 재검토)")


# ══════════════════════════════════════════════════════
# [C] Bootstrap 클러스터 안정성 (Jaccard 유사도)
# ══════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("[C] Bootstrap 클러스터 안정성 검증 (n=50회, Jaccard 유사도)")
print("─" * 60)
print("  (최솟값 0 ~ 최댓값 1, 0.75 이상이면 안정적 클러스터로 판단)\n")

# 작은 샘플에서 부트스트랩 수행 (속도)
np.random.seed(RANDOM_SEED)
BS_SIZE   = min(5_000, len(X))
BS_ROUNDS = 50
bs_idx    = np.random.choice(len(X), BS_SIZE, replace=False)
X_bs_pool = X[bs_idx]

# 기준 모델
km_base   = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
base_labels = km_base.fit_predict(X_bs_pool)


def _jaccard(labels_a: np.ndarray, labels_b: np.ndarray, k: int) -> float:
    """
    헝가리안 알고리즘으로 클러스터 레이블을 최적 매칭한 뒤
    전체 클러스터에 걸친 평균 Jaccard 유사도 반환.
    """
    # 혼동 행렬 (k × k)
    cost = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            cost[i, j] = np.sum((labels_a == i) & (labels_b == j))

    # 최대화를 위해 음수 변환 후 헝가리안 알고리즘
    row_ind, col_ind = linear_sum_assignment(-cost)

    jaccards = []
    for r, c in zip(row_ind, col_ind):
        intersection = cost[r, c]
        union = (np.sum(labels_a == r) + np.sum(labels_b == c) - intersection)
        jaccards.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(jaccards))


jaccard_scores = []
for i in range(BS_ROUNDS):
    # 복원 추출 부트스트랩
    boot_idx    = np.random.choice(BS_SIZE, BS_SIZE, replace=True)
    X_boot      = X_bs_pool[boot_idx]
    km_boot     = KMeans(n_clusters=N_CLUSTERS, random_state=i, n_init=5)
    boot_labels = km_boot.fit_predict(X_boot)

    # 기준 모델로 동일 샘플 예측 후 비교
    ref_labels  = km_base.predict(X_boot)
    jac         = _jaccard(ref_labels, boot_labels, N_CLUSTERS)
    jaccard_scores.append(jac)

jac_mean = np.mean(jaccard_scores)
jac_std  = np.std(jaccard_scores)
jac_min  = np.min(jaccard_scores)
jac_max  = np.max(jaccard_scores)

print(f"  Bootstrap Jaccard — 평균: {jac_mean:.4f}  표준편차: {jac_std:.4f}  "
      f"범위: [{jac_min:.4f}, {jac_max:.4f}]")

if jac_mean >= 0.75:
    print(f"  → ✅ 클러스터 안정성 우수  (Jaccard ≥ 0.75)")
elif jac_mean >= 0.60:
    print(f"  → 🔶 클러스터 안정성 보통  (0.60 ≤ Jaccard < 0.75)")
else:
    print(f"  → ⚠️  클러스터 안정성 낮음 (Jaccard < 0.60, 재검토 권장)")


# ══════════════════════════════════════════════════════
# 시각화 — 3가지 검증 결과 요약 대시보드
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (1) 홀드아웃 분포 비교
ax = axes[0]
x_pos = np.arange(N_CLUSTERS)
width = 0.35
ax.bar(x_pos - width/2, train_dist,   width, label='학습 세트',    color='#4C72B0', alpha=0.85)
ax.bar(x_pos + width/2, holdout_dist, width, label='홀드아웃 세트', color='#DD8452', alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'클러스터 {c}' for c in range(N_CLUSTERS)], fontsize=10)
ax.set_ylabel('클러스터 비율')
ax.set_ylim(0, 0.7)
ax.set_title('[A] 홀드아웃 클러스터 분포 안정성', fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (2) 다중 시드 Silhouette
ax = axes[1]
ax.plot(range(len(SEEDS)), sil_results, 'o-', color='#55A868', linewidth=2, markersize=7)
ax.axhline(sil_mean, color='red', linestyle='--', linewidth=1.5,
           label=f'평균 {sil_mean:.4f}')
ax.fill_between(range(len(SEEDS)),
                [sil_mean - sil_std] * len(SEEDS),
                [sil_mean + sil_std] * len(SEEDS),
                color='red', alpha=0.12, label=f'±1σ ({sil_std:.4f})')
ax.set_xticks(range(len(SEEDS)))
ax.set_xticklabels([str(s) for s in SEEDS], fontsize=8, rotation=30)
ax.set_xlabel('랜덤 시드')
ax.set_ylabel('Silhouette Score')
ax.set_title('[B] 다중 시드 Silhouette 안정성', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (3) Bootstrap Jaccard 분포
ax = axes[2]
ax.hist(jaccard_scores, bins=15, color='#C44E52', alpha=0.75, edgecolor='white')
ax.axvline(jac_mean, color='navy', linestyle='--', linewidth=2,
           label=f'평균 {jac_mean:.4f}')
ax.axvline(0.75, color='green', linestyle=':', linewidth=1.5,
           label='기준선 0.75')
ax.set_xlabel('Jaccard 유사도')
ax.set_ylabel('빈도')
ax.set_title('[C] Bootstrap 클러스터 안정성', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle(f'K-Means 클러스터링 일반화 성능 검증 (k={N_CLUSTERS})',
             fontsize=14, y=1.02)
plt.tight_layout()

os.makedirs('outputs/figures', exist_ok=True)
save_path = 'outputs/figures/step5_validation.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n검증 차트 저장: {save_path}")

# ══════════════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  검증 결과 요약")
print("=" * 60)
print(f"  [A] 홀드아웃 분포 드리프트 (최대): {max_drift:.4f}")
print(f"      학습/홀드아웃 Silhouette:      {sil_train:.4f} / {sil_holdout:.4f}")
print(f"  [B] 다중 시드 Silhouette:          {sil_mean:.4f} ± {sil_std:.4f}")
print(f"  [C] Bootstrap Jaccard 평균:        {jac_mean:.4f} ± {jac_std:.4f}")
print()
