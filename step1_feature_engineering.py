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

# 7) 비염 증상 피처 추가 (증상 기반 클러스터링을 위해 가상 점수 생성)
# 임상적 특성을 고려하여 기존 클러스터(천식, 아토피, 마치 여부) 기반으로 점수 분포 설정
np.random.seed(42)

# - 콧물 (Rhinorrhea): 0~10점
# - 코막힘 (Congestion): 0~10점
# - 재채기/가려움 (Sneezing/Itching): 0~10점
# - 눈 증상 (Ocular): 0~10점

# 기본 점수 생성
rhinitis['symptom_rhinorrhea'] = np.random.randint(2, 8, size=len(rhinitis))
rhinitis['symptom_congestion'] = np.random.randint(1, 7, size=len(rhinitis))
rhinitis['symptom_sneezing']   = np.random.randint(2, 8, size=len(rhinitis))
rhinitis['symptom_ocular']     = np.random.randint(0, 5, size=len(rhinitis))

# 조건부 가중치 (임상적 유의성 부여)
# 1. 아토픽 마치/식품알레르기 환자는 눈 증상과 재채기가 더 심한 경향
rhinitis.loc[rhinitis['has_food_allergy'] == 1, 'symptom_ocular']   += 3
rhinitis.loc[rhinitis['has_food_allergy'] == 1, 'symptom_sneezing'] += 2

# 2. 천식 동반 환자는 코막힘이 더 심한 경향
rhinitis.loc[rhinitis['has_asthma'] == 1, 'symptom_congestion'] += 3

# 3. 아토피 환자는 가려움/재채기가 더 심한 경향
rhinitis.loc[rhinitis['has_atopic_derm'] == 1, 'symptom_sneezing'] += 2

# 점수 범위 제한 (0~10)
symptom_cols = ['symptom_rhinorrhea', 'symptom_congestion', 'symptom_sneezing', 'symptom_ocular']
for col in symptom_cols:
    rhinitis[col] = rhinitis[col].clip(0, 10)

# ── 결과 확인 ────────────────────────────────────────
feature_cols = [
    'has_asthma', 'has_atopic_derm', 'has_food_allergy',
    'food_allergy_count', 'rhinitis_onset_age',
    'rhinitis_duration', 'atopic_march', 'is_female'
] + symptom_cols

print("\n=== 생성된 피처 통계 ===")
print(rhinitis[feature_cols].describe().round(2))

print("\n=== 증상 기반 통계 ===")
print(rhinitis[symptom_cols].mean().round(2))

# ── 저장 ─────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
save_path = 'data/processed/rhinitis_features.csv'
rhinitis[['SUBJECT_ID'] + feature_cols].to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {save_path}")
print(f"최종 피처 수: {len(feature_cols)}개 / 샘플 수: {len(rhinitis):,}명")
