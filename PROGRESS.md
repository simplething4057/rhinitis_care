# 비염 케어 AI — 진행 현황 기록

> 마지막 업데이트: 2026-04-01

---

## 완료 단계

### Phase 1. 프로젝트 기본 세팅 ✅
- 디렉토리 구조 생성 (`data/`, `src/`, `notebooks/`, `outputs/`)
- 설정 파일: `config/config.yaml`, `.env.example`, `requirements.txt`, `.gitignore`
- 소스 모듈 초안: `preprocess.py`, `api_collector.py`, `clustering.py`, `correlation.py`
- 분석 노트북 초안: `01_EDA.ipynb`, `02_Clustering.ipynb`

### Phase 2. 데이터 수집 ✅
| 데이터셋 | 출처 | 규모 | 성격 | 용도 |
|---|---|---|---|---|
| Childhood Allergies | Kaggle / Zenodo | 333,200명 | 실제 임상 코호트 | 클러스터링 메인 |
| Healthcare.csv | Kaggle | 25,000건 | 합성 데이터 | 파이프라인 테스트용만 |

### Phase 3. 피처 엔지니어링 ✅ (`step1_feature_engineering.py`)
- 비염 환자 필터: 55,567명 추출
- 생성 피처 7개:

| 피처 | 설명 |
|---|---|
| `has_asthma` | 천식 동반 여부 (0/1) |
| `has_atopic_derm` | 아토피 동반 여부 (0/1) |
| `has_food_allergy` | 식품알레르기 동반 여부 (0/1) |
| `food_allergy_count` | 식품알레르기 종류 수 |
| `rhinitis_onset_age` | 비염 발병 나이 (세) |
| `rhinitis_duration` | 비염 지속 기간 (년) |
| `atopic_march` | 아토픽 마치 여부 (0/1) |

- 저장: `data/processed/rhinitis_features.csv`

### Phase 4. 클러스터 수 탐색 ✅ (`step2_find_optimal_k.py`, `step4_compare_k.py`)

| k | Silhouette | 판단 |
|---|---|---|
| 2 | 0.4894 | 식품알레르기 유무만으로 분리 — 단순 |
| **3** | **0.2876** | **임상적 의미 가장 명확 — 채택** |
| 4 | 0.3058 | 347명(0.6%) 극소수 클러스터 발생 — 실용성 낮음 |

### Phase 5. K-Means 클러스터링 확정 ✅ (`step3_clustering.py`, k=3)

| 클러스터 | 명칭 | 규모 | 천식 | 아토피 | 식품알레르기 | 발병나이 |
|---|---|---|---|---|---|---|
| 0 | 아토픽 마치형 | 7,103명 (12.8%) | 66.9% | 44.0% | 100% | 5.9세 |
| 1 | 호흡기 알레르기형 | 26,122명 (47.0%) | 0% | 18.6% | 0% | 7.9세 |
| 2 | 비염+천식 복합형 | 22,342명 (40.2%) | 100% | 26.6% | 0% | 7.0세 |

**클러스터 해석:**
- **호흡기 알레르기형 (47%)**: 동반 질환 없는 순수 비염. 발병 나이 가장 늦음.
- **비염+천식 복합형 (40%)**: 천식 100% 동반. 호흡기 집중 관리 필요.
- **아토픽 마치형 (13%)**: 식품알레르기에서 시작해 아토피·비염·천식으로 진행. 가장 이른 발병(5.9세).

- 저장: `outputs/models/kmeans_childhood.pkl`
- 결과: `data/processed/rhinitis_clustered.csv`

---

## 다음 단계

### Phase 6. FastAPI 백엔드 구축
- `src/api/main.py` 생성
- 엔드포인트: `POST /predict`, `GET /guide`, `GET /env`
- 클러스터 모델 로딩 및 추론 파이프라인 연결

### Phase 7. Streamlit 대시보드
- `app.py` 생성
- 증상/동반질환 입력 → 비염 유형 분류 → 맞춤 가이드 출력

### Phase 8. 배포
- Streamlit Cloud 무료 배포
- GitHub 연동

---

## 주요 파일 경로

```
rhinitis_care/
├── data/processed/
│   ├── rhinitis_features.csv     # 피처 엔지니어링 결과
│   └── rhinitis_clustered.csv    # 클러스터 레이블 포함 최종 데이터
├── outputs/
│   ├── models/kmeans_childhood.pkl  # 학습된 K-Means 모델
│   └── figures/
│       ├── step2_optimal_k.png
│       ├── step3_cluster_profiles.png
│       ├── step3_onset_age_dist.png
│       └── step4_k_comparison.png
└── step1~4_*.py                  # 분석 스크립트
```
