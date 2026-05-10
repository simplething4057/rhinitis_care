# 🌿 비염 케어 AI — 증상 클러스터링 & 맞춤형 가이드

> **"병원에 가기엔 애매하지만 괴로운 오늘, 데이터로 해결책을 제시합니다."**

---

## 프로젝트 개요

55,567명의 소아 알레르기 코호트 데이터를 K-Means 클러스터링으로 분석해 비염 환자를 3가지 유형으로 분류하고,
**LightGBM 분류기**로 실시간 확률 기반 예측 및 맞춤형 관리 가이드를 제공하는 AI 서비스입니다.

- 동반 질환(천식·아토피·식품알레르기) + 증상 점수 기반 비염 유형 자동 분류
- LightGBM 1순위 / K-Means 폴백 — 모델 파일 부재 시에도 동작
- Streamlit 대시보드로 누구나 쉽게 사용
- FastAPI 백엔드 + 직접 추론 모드 지원 (API 서버 없이도 동작)
- Supabase PostgreSQL 연동으로 예측 이력 저장 및 7일 추이 시각화 (선택)

---

## 비염 3대 유형

| 유형 | 규모 | 천식 | 아토피 | 식품알레르기 | 평균 발병 나이 |
|---|---|---|---|---|---|
| 🌿 **호흡기 알레르기형** | 47% (26,122명) | 0% | 18.6% | 0% | 7.9세 |
| 💨 **비염+천식 복합형** | 40% (22,342명) | 100% | 26.6% | 0% | 7.0세 |
| 🔶 **아토픽 마치형** | 13% (7,103명) | 66.9% | 44.0% | 100% | 5.9세 |

---

## 주요 기능

### 1. 비염 유형 분류 — 🔍 유형 분석 탭

![비염 케어 AI 대시보드](assets/screenshot.png)

증상 점수(콧물·코막힘·재채기·눈 증상, 0~10점)와 동반 질환 정보를 입력하면 **LightGBM이 실시간으로 비염 유형을 분류**합니다.

| 제공 정보 | 내용 |
|---|---|
| 유형 분류 결과 | 3가지 유형 중 해당 유형과 설명 |
| 확률 바 차트 | 각 유형별 소속 확률 (예: 비염+천식 복합형 97.0%) |
| 7일 유형 변화 추이 | 내 유형 변화 이력 + 동일 유형 평균 비교 |
| 증상 레이더 차트 | 내 증상 패턴 vs 유형 평균 비교 |
| 맞춤 관리 가이드 | 분류된 유형에 맞는 생활 관리법 |

---

### 2. 유형별 동반 질환 현황 — 📊 유형 현황 탭

![비염 유형별 동반 질환 프로파일](assets/cluster_comorbidity.png)

55,567명 코호트 데이터에서 도출한 **각 유형의 동반 질환 패턴**을 시각화합니다.

| 제공 정보 | 내용 |
|---|---|
| 유형별 비율 파이 차트 | 전체 환자 중 각 유형 분포 (47% / 40% / 13%) |
| 증상 강도 비교 차트 | 유형별 콧물·코막힘·재채기·눈 증상 평균 점수 비교 |
| 동반 질환 프로파일 | 천식·아토피·식품알레르기 동반율 비교 |

---

### 3. 유형별 맞춤 관리 가이드 — ℹ️ 유형 안내 탭

각 유형 항목을 펼치면 **해당 유형의 특징과 맞춤 관리 가이드**를 제공합니다.

| 유형 | 핵심 관리 포인트 |
|---|---|
| 🌿 호흡기 알레르기형 | 꽃가루·집먼지진드기 회피, 항히스타민제 활용 |
| 💨 비염+천식 복합형 | 천식 악화 연동 모니터링, 흡입기 상시 휴대 |
| 🔶 아토픽 마치형 | 피부·식이 알레르겐 복합 관리, 의사 협진 권장 |

각 유형 패널에서 증상별 평균 점수(콧물/코막힘/재채기/눈 증상 각 /10)도 함께 확인할 수 있습니다.

---

### 4. 실시간 대기·날씨 정보 — 🌍 환경 정보 탭

에어코리아·기상청 공공 API를 연동해 **비염 증상에 영향을 주는 환경 정보**를 실시간으로 제공합니다.

| 제공 정보 | 내용 |
|---|---|
| PM10 / PM2.5 / O3 | 현재 수치 + 등급 (좋음/보통/나쁨/매우나쁨) |
| PM10 24시간 추이 | 기준선(30/80/150 ㎍/㎥) 포함 라인 차트 |
| 기상청 날씨 예보 | 기온·습도·강수확률 (3시간 단위) |
| 위험 알림 | PM10 > 80 또는 PM2.5 > 35 시 경보 표시 |

> 측정 지역은 사이드바에서 변경할 수 있습니다. API 키 없이도 나머지 기능은 정상 동작합니다.

---

### 모델 성능

![LightGBM 피처 중요도 및 혼동행렬](assets/model_performance.png)

K-Means로 도출한 클러스터 레이블을 타깃으로 LightGBM을 학습시킨 결과, **5-Fold 교차검증 기준 Accuracy 1.000 / F1 1.000**을 달성했습니다.

- 분류에 가장 중요한 피처: `rhinitis_duration` (비염 지속 기간) > `rhinitis_onset_age` (발병 나이) > `has_food_allergy`
- 55,567건 전체에 오분류 0건

---

## 기술 스택

| 범주 | 라이브러리 |
|---|---|
| 언어 | Python 3.9+ |
| ML | `lightgbm>=4.0`, `scikit-learn>=1.3`, `numpy>=1.24`, `pandas>=2.0` |
| API | `fastapi>=0.110`, `uvicorn>=0.29`, `pydantic>=2.0` |
| 대시보드 | `streamlit>=1.35`, `plotly>=5.15` |
| DB | `sqlalchemy>=2.0`, `psycopg2-binary>=2.9` (Supabase, 선택) |
| 유틸 | `python-dotenv>=1.0`, `pyyaml>=6.0`, `requests>=2.31` |

---

## 빠른 시작

### 1. 패키지 설치

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Python 3.9 이상이 필요합니다.

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 API 키 입력
```

| 환경변수 | 필수 | 발급처 | 용도 |
|---|---|---|---|
| `AIRKOREA_API_KEY` | 필수 | 공공데이터포털 → 에어코리아 | 미세먼지 실시간 수집 |
| `KMA_API_KEY` | 필수 | 공공데이터포털 → 기상청 단기예보 | 기온·습도 수집 |
| `DATABASE_URL` | 선택 | Supabase 대시보드 → Project Settings → Database | 예측 이력 저장 |

`DATABASE_URL`을 설정하지 않으면 이력 저장 기능이 비활성화되며 나머지 기능은 정상 동작합니다.

### 3. Streamlit 대시보드 실행 (API 서버 없이도 동작)

```bash
streamlit run app.py
```

### 4. FastAPI 서버 + 대시보드 함께 실행

```bash
# 터미널 1 — API 서버
uvicorn src.api.main:app --reload

# 터미널 2 — 대시보드
streamlit run app.py
```

API 문서: http://localhost:8000/docs

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/` | 서버 실행 확인 |
| `GET` | `/health` | 서버 및 모델 로드 상태 |
| `POST` | `/predict` | 비염 유형 분류 + 관리 가이드 반환 |
| `GET` | `/guide/{cluster_label}` | 특정 유형 관리 가이드 조회 |
| `GET` | `/clusters` | 전체 유형 목록 조회 |
| `GET` | `/env` | 서버 환경 및 패키지 버전 정보 |

### POST /predict 예시

**요청**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "has_asthma": 1,
    "has_atopic_derm": 0,
    "has_food_allergy": 0,
    "rhinitis_onset_age": 7.0,
    "symptom_rhinorrhea": 4,
    "symptom_congestion": 6,
    "symptom_sneezing": 3,
    "symptom_ocular": 2
  }'
```

**응답**
```json
{
  "status": "success",
  "result": {
    "cluster_id": 1,
    "cluster_label": "비염+천식 복합형",
    "confidence": 0.97,
    "cluster_probs": {
      "호흡기 알레르기형": 0.01,
      "비염+천식 복합형": 0.97,
      "아토픽 마치형": 0.02
    },
    "model_type": "lightgbm",
    "description": "...",
    "guide": ["..."]
  },
  "summary": "..."
}
```

**PatientInput 전체 필드**

| 필드 | 타입 | 필수 | 범위 | 설명 |
|---|---|---|---|---|
| `has_asthma` | int | ✅ | 0/1 | 천식 보유 여부 |
| `has_atopic_derm` | int | ✅ | 0/1 | 아토피 보유 여부 |
| `has_food_allergy` | int | ✅ | 0/1 | 식품알레르기 보유 여부 |
| `rhinitis_onset_age` | float | ✅ | ≥0 | 비염 발병 나이 (세) |
| `food_allergy_count` | int | — | 0~10 | 식품알레르기 종류 수 |
| `rhinitis_duration` | float | — | ≥0 | 비염 지속 기간 (년) |
| `atopic_march` | int | — | 0/1 | 아토픽 마치 여부 |
| `symptom_rhinorrhea` | int | — | 0~10 | 콧물 정도 |
| `symptom_congestion` | int | — | 0~10 | 코막힘 정도 |
| `symptom_sneezing` | int | — | 0~10 | 재채기·가려움 정도 |
| `symptom_ocular` | int | — | 0~10 | 눈 가려움·충혈 정도 |

---

## 디렉토리 구조

```
rhinitis_care/
├── app.py                       # Streamlit 대시보드
├── requirements.txt
├── .env.example                 # 환경변수 예시
├── .streamlit/
│   ├── config.toml              # Streamlit 테마/서버 설정
│   └── secrets.toml             # 로컬 시크릿 (git 제외)
├── assets/
│   └── screenshot.png           # 대시보드 스크린샷
├── config/
│   ├── config.yaml              # 공통 설정
│   ├── config.dev.yaml          # 개발 환경 오버라이드
│   └── config.prod.yaml         # 운영 환경 오버라이드
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI 앱 (엔드포인트)
│   │   ├── predictor.py         # 모델 로드 + 추론
│   │   └── schemas.py           # 입출력 스키마
│   ├── data/
│   │   ├── preprocess.py        # 전처리 파이프라인
│   │   └── api_collector.py     # 에어코리아·기상청 API 수집 (TTL 캐싱)
│   ├── models/
│   │   ├── clustering.py        # K-Means 학습·저장
│   │   └── history.py           # 예측 이력 ORM 모델
│   └── utils/
│       ├── config.py            # 환경별 설정 로더
│       ├── history.py           # 이력 저장·조회·가상 데이터 생성
│       └── logging_config.py    # 중앙화된 로깅 설정
├── database.py → src/database.py  # SQLAlchemy 엔진·세션 초기화
├── data/
│   ├── raw/                     # 원본 데이터
│   └── processed/               # 전처리 완료 데이터
├── outputs/
│   ├── models/
│   │   ├── lgbm_rhinitis.pkl    # LightGBM 분류기 (주 모델)
│   │   └── kmeans_rhinitis.pkl  # K-Means 클러스터러 (폴백)
│   └── figures/                 # 시각화 결과
├── tests/
│   ├── test_api.py              # FastAPI 엔드포인트 테스트
│   ├── test_predictor.py        # 모델 유효성 검증 테스트
│   └── test_schemas.py          # Pydantic 스키마 테스트
└── notebooks/
    ├── 01_EDA.ipynb
    └── 02_Clustering.ipynb
```

---

## 분석 파이프라인

```
데이터 수집 (Kaggle 코호트 / 공공 API)
    ↓
피처 엔지니어링 + 전처리 (step1_feature_engineering.py)
  └─ src/data/preprocess.py: handle_missing_values, remove_outliers_iqr
    ↓
최적 클러스터 수 탐색 (step2_find_optimal_k.py)
    ↓
K-Means 클러스터링 확정 (step3_clustering.py, k=3)
  └─ outputs/models/kmeans_rhinitis.pkl 생성
    ↓
LightGBM 분류기 학습 (클러스터 레이블 → 지도 학습)
  └─ outputs/models/lgbm_rhinitis.pkl 생성 (주 모델)
    ↓
클러스터링 일반화 성능 검증 (step5_validation.py)
  ├─ [A] 홀드아웃 분포 안정성 (80/20 분리)
  ├─ [B] 다중 시드 Silhouette 검증
  └─ [C] Bootstrap Jaccard 안정성
    ↓
FastAPI 추론 서버 (src/api/)
    ↓
Streamlit 대시보드 (app.py)
```

---

## 테스트 실행

```bash
pip install -r requirements-dev.txt
pytest
```

테스트 범위: API 엔드포인트 (16개), 모델 추론 유효성 (24개), 스키마 경계 검증 (7개)

---

## Streamlit Cloud 배포

1. GitHub에 push
2. [Streamlit Cloud](https://streamlit.io/cloud) → New app → 이 저장소 선택
3. Main file: `app.py`
4. **Secrets 메뉴**에서 아래 내용을 붙여넣고 실제 값으로 교체

```toml
APP_ENV = "prod"

AIRKOREA_API_KEY = "your_airkorea_api_key_here"
KMA_API_KEY = "your_kma_api_key_here"

# 예측 이력 저장용 (선택 — 없으면 이력 기능 비활성화)
DATABASE_URL = "postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres"
```

> Supabase 연결 문자열: 대시보드 → Project Settings → Database → Connection String (URI)

---

## 성공 지표

| 지표 | 목표 |
|---|---|
| AI 유형 분류 일치율 | 사용자 체감과 80% 이상 일치 |
| 증상 개선 만족도 | 가이드 준수 후 설문 4점/5점 이상 |
