"""
Step 1. 피처 엔지니어링
Childhood Allergies 데이터에서 클러스터링에 필요한 피처 생성 후 저장
"""

import kagglehub
import os
import pandas as pd
import numpy as np

# ── 데이터 로드 ──────────────────────────────────────
print("=" * 50)
print("Step 1. 피처 엔지니어링 시작")
print("=" * 50)

path = kagglehub.dataset_download(
    'thedevastator/childhood-allergies-prevalence-diagnosis-and-tre'
)
filepath = os.path.join(path, 'food-allergy-analysis-Zenodo.csv')
df = pd.read_csv(filepath)
print(f"전체 데이터: {len(df):,}명 로드 완료")

# ── 비염 환자만 필터 ─────────────────────────────────
rhinitis = df[df['ALLERGIC_RHINITIS_START'].notna()].copy()
print(f"비염 환자 필터: {len(rhinitis):,}명\n")

# ── 피처 생성 ────────────────────────────────────────
print("피처 생성 중...")

# 1) 동반 질환 이진 피처
rhinitis['has_asthma']      = rhinitis['ASTHMA_START'].notna().astype(int)
rhinitis['has_atopic_derm'] = rhinitis['ATOPIC_DERM_START'].notna().astype(int)

# 2) 식품 알레르기 이진 피처 (주요 5종)
food_cols = ['PEANUT_ALG_START', 'MILK_ALG_START',
             'EGG_ALG_START', 'WHEAT_ALG_START', 'SHELLFISH_ALG_START']
rhinitis['has_food_allergy'] = rhinitis[food_cols].notna().any(axis=1).astype(int)
rhinitis['food_allergy_count'] = rhinitis[food_cols].notna().sum(axis=1)

# 3) 비염 발병 나이 (결측은 중앙값으로 대체)
median_onset = rhinitis['ALLERGIC_RHINITIS_START'].median()
rhinitis['rhinitis_onset_age'] = rhinitis['ALLERGIC_RHINITIS_START'].fillna(median_onset)

# 4) 비염 지속 기간 (END - START, 없으면 0)
rhinitis['rhinitis_duration'] = (
    rhinitis['ALLERGIC_RHINITIS_END'] - rhinitis['ALLERGIC_RHINITIS_START']
).fillna(0).clip(lower=0)

# 5) 아토픽 마치 여부 (식품알레르기 → 아토피 → 비염 순서 진행)
def is_atopic_march(row):
    food_start = row[food_cols].min()  # 가장 이른 식품알레르기
    atop_start = row['ATOPIC_DERM_START']
    rhin_start = row['ALLERGIC_RHINITIS_START']
    if pd.isna(food_start) or pd.isna(atop_start) or pd.isna(rhin_start):
        return 0
    return int(food_start < atop_start < rhin_start)

rhinitis['atopic_march'] = rhinitis.apply(is_atopic_march, axis=1)

# 6) 성별 이진 피처
rhinitis['is_female'] = (rhinitis['GENDER_FACTOR'] == 'S1 - Female').astype(int)

# ── 결과 확인 ────────────────────────────────────────
feature_cols = [
    'has_asthma', 'has_atopic_derm', 'has_food_allergy',
    'food_allergy_count', 'rhinitis_onset_age',
    'rhinitis_duration', 'atopic_march', 'is_female'
]

print("\n=== 생성된 피처 통계 ===")
print(rhinitis[feature_cols].describe().round(2))

print("\n=== 이진 피처 분포 ===")
binary_cols = ['has_asthma', 'has_atopic_derm', 'has_food_allergy', 'atopic_march']
for col in binary_cols:
    count = rhinitis[col].sum()
    pct = count / len(rhinitis) * 100
    print(f"  {col:25s}: {count:,}명 ({pct:.1f}%)")

# ── 저장 ─────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
save_path = 'data/processed/rhinitis_features.csv'
rhinitis[['SUBJECT_ID'] + feature_cols].to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {save_path}")
print(f"최종 피처 수: {len(feature_cols)}개 / 샘플 수: {len(rhinitis):,}명")
