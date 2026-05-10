import os
import pandas as pd

# Kaggle 데이터셋 다운로드 및 로드
from kagglehub import dataset_download
path = dataset_download('thedevastator/childhood-allergies-prevalence-diagnosis-and-tre')
filepath = os.path.join(path, 'food-allergy-analysis-Zenodo.csv')
df = pd.read_csv(filepath)

# 비염 환자만 필터
rhinitis = df[df['ALLERGIC_RHINITIS_START'].notna()].copy()

# 동반질환 클러스터링
rhinitis['asthma'] = rhinitis['ASTHMA_START'].notna()
rhinitis['atopic'] = rhinitis['ATOPIC_DERM_START'].notna()
def cluster_type(row):
    if row['asthma']:
        return '비염+천식형'
    elif row['atopic']:
        return '비염+아토피형'
    else:
        return '비염 단독형'
rhinitis['cluster'] = rhinitis.apply(cluster_type, axis=1)

# 2. Healthcare.csv 증상 텍스트 분석
healthcare_path = os.path.join('data', 'raw', 'Healthcare.csv')
health = pd.read_csv(healthcare_path)

# 증상 키워드 매핑
symptom_keywords = {
    '콧물': ['runny nose'],
    '재채기': ['sneezing'],
    '코막힘': ['nasal congestion', 'congestion'],
    '눈가려움': ['eye itching', 'itchy eyes'],
}
def extract_symptoms(text):
    found = []
    for k, vlist in symptom_keywords.items():
        for v in vlist:
            if v in text.lower():
                found.append(k)
    return ','.join(found) if found else '기타'
health['증상_프로파일'] = health['Symptoms'].astype(str).apply(extract_symptoms)

# 3. 두 데이터 통합 (나이+성별 근사 매칭)
# Kaggle 컬럼 표준화: AGE_START_YEARS → AGE (정수 반올림), GENDER_FACTOR → GENDER (Male/Female)
rhinitis = rhinitis.copy()
rhinitis['AGE'] = rhinitis['AGE_START_YEARS'].round().astype('Int64')

_gender_map = {0: 'Male', 1: 'Female', '0': 'Male', '1': 'Female',
               'M': 'Male', 'F': 'Female', 'male': 'Male', 'female': 'Female'}
rhinitis['GENDER'] = rhinitis['GENDER_FACTOR'].map(_gender_map).fillna(rhinitis['GENDER_FACTOR'].astype(str))

health['Age'] = health['Age'].round().astype('Int64')

merged = pd.merge(
    rhinitis,
    health,
    left_on=['AGE', 'GENDER'],
    right_on=['Age', 'Gender'],
    how='inner',
)
print(f"병합 결과: {len(merged)}건 (Kaggle {len(rhinitis)}건 × Healthcare {len(health)}건)")

# 4. 유형 정의서 생성
os.makedirs('outputs/reports', exist_ok=True)

if len(merged) > 0:
    # 병합 성공: 클러스터 × 증상 프로파일 교차 분석
    profile = merged.groupby(['cluster', '증상_프로파일']).size().reset_index(name='환자수')
    profile = profile.sort_values(['cluster', '환자수'], ascending=[True, False])
    profile.to_csv('outputs/reports/rhinitis_patient_profiles.csv', index=False, encoding='utf-8-sig')
    print('비염 환자 유형 정의서 저장 완료: outputs/reports/rhinitis_patient_profiles.csv')
else:
    # 병합 0건: 두 데이터셋의 나이 범위가 달라 직접 조인 불가
    # (Kaggle: 소아 발병 나이 평균 7세 / Healthcare: 성인 25~80세)
    # → 각 데이터셋을 독립 분석 후 결합
    print("⚠️  병합 0건 — 나이 범위 불일치 (Kaggle: 소아, Healthcare: 성인)")
    print("    대안: Kaggle 클러스터 분포 × Healthcare 증상 분포를 독립 집계")

    cluster_profile = rhinitis.groupby('cluster').size().reset_index(name='Kaggle_환자수')
    symptom_profile = health.groupby('증상_프로파일').size().reset_index(name='Healthcare_환자수')

    cluster_profile.to_csv('outputs/reports/kaggle_cluster_profile.csv',   index=False, encoding='utf-8-sig')
    symptom_profile.to_csv('outputs/reports/healthcare_symptom_profile.csv', index=False, encoding='utf-8-sig')
    print('독립 분석 저장 완료: outputs/reports/kaggle_cluster_profile.csv / healthcare_symptom_profile.csv')
