import kagglehub, os
import pandas as pd

path = kagglehub.dataset_download('thedevastator/childhood-allergies-prevalence-diagnosis-and-tre')
filepath = os.path.join(path, 'food-allergy-analysis-Zenodo.csv')

df = pd.read_csv(filepath)
print(f'전체 행: {len(df):,}명')
print()

# 비염 환자만 필터
rhinitis = df[df['ALLERGIC_RHINITIS_START'].notna()]
print(f'비염 환자: {len(rhinitis):,}명 ({len(rhinitis)/len(df)*100:.1f}%)')
print()

# 동반 질환 현황
print('=== 동반 질환 보유율 ===')
print(f'  천식(Asthma):        {df["ASTHMA_START"].notna().sum():,}명')
print(f'  아토피(Atopic Derm): {df["ATOPIC_DERM_START"].notna().sum():,}명')
print()

# 비염 발병 나이 분포
print('=== 비염 발병 나이 통계 ===')
print(rhinitis['AGE_START_YEARS'].describe().round(1))
