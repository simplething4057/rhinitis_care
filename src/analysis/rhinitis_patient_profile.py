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

# 3. 두 데이터 통합 (예시: 나이, 성별 등 공통 컬럼 기준)
# 실제로는 환자 ID 등 공통키 필요. 여기선 예시로 나이+성별 매칭
merged = pd.merge(
    rhinitis,
    health,
    left_on=['AGE', 'GENDER'],
    right_on=['Age', 'Gender'],
    how='inner',
)

# 4. 유형 정의서 생성 (클러스터+증상 프로파일)
profile = merged.groupby(['cluster', '증상_프로파일']).size().reset_index(name='환자수')
profile = profile.sort_values(['cluster', '환자수'], ascending=[True, False])

# 결과 저장
profile.to_csv('outputs/reports/rhinitis_patient_profiles.csv', index=False, encoding='utf-8-sig')
print('비염 환자 유형 정의서 저장 완료: outputs/reports/rhinitis_patient_profiles.csv')
