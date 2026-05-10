# 🌿 비염 케어 AI — 증상 클러스터링 & 맞춤형 가이드

> **"병원에 가기엔 애매하지만 괴로운 오늘, 데이터로 해결책을 제시합니다."**

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [비염 3대 유형](#비염-3대-유형)
3. [주요 기능](#주요-기능)
4. [기술 스택](#기술-스택)
5. [빠른 시작](#빠른-시작)
6. [API 엔드포인트](#api-엔드포인트)
7. [디렉토리 구조](#디렉토리-구조)
8. [분석 파이프라인](#분석-파이프라인)
9. [테스트 실행](#테스트-실행)
10. [Streamlit Cloud 배포](#streamlit-cloud-배포)
11. [성공 지표](#성공-지표)
12. [변경 이력](#변경-이력)
13. [별첨](#별첨)

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

| 범주 | 기술 | 역할 |
|---|---|---|
| **언어** | Python 3.9+ | — |
| **데이터** | pandas 2.0, numpy 1.24 | 전처리·피처 엔지니어링·EDA |
| **클러스터링** | scikit-learn 1.3 (K-Means) | 비염 유형 3개 도출, Silhouette·Jaccard 검증 |
| **분류 모델** | LightGBM 4.0 | 클러스터 레이블 확률 예측 (5-Fold Acc 1.000) |
| **API 서버** | FastAPI 0.110, uvicorn 0.29, Pydantic 2.0 | REST 엔드포인트, 입출력 스키마 검증 |
| **대시보드** | Streamlit 1.35, Plotly 5.15 | 인터랙티브 웹 UI, 레이더·라인 차트 |
| **데이터베이스** | SQLAlchemy 2.0, psycopg2, Supabase PostgreSQL | 예측 이력 저장·조회 (선택) |
| **공공 API** | 에어코리아 ArpltnInforInqireSvc | PM10/PM2.5/O3 실시간 측정 (서울 25구 + 수원·파주·세종·광역시) |
| | 기상청 VilageFcstInfoService_2.0 | 기온·습도·강수 3시간 단기예보 |
| | 기상청 HealthWthrIdxServiceV3 | 꽃가루농도위험지수 — 소나무·참나무·자작나무·쑥 |
| **설정 관리** | python-dotenv 1.0, PyYAML 6.0 | 환경별(.env / config.yaml) 설정 분리 |
| **데이터 출처** | Kaggle CHOA 코호트 55,567명 | K-Means 클러스터링·LightGBM 학습 원본 |
| | Healthcare.csv 25,000건 | 증상 키워드 텍스트 분석 보조 데이터 |

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
| `AIRKOREA_API_KEY` | 필수 | 공공데이터포털 → 에어코리아 대기오염정보 | PM10/PM2.5/O3 실시간 수집 |
| `KMA_API_KEY` | 필수 | 공공데이터포털 → 기상청_단기예보 | 기온·습도 수집 |
| `POLLEN_API_KEY` | 선택 | 공공데이터포털 → 기상청_꽃가루농도위험지수 조회서비스(3.0) | 꽃가루 위험지수 수집 (없으면 AIRKOREA_API_KEY 폴백) |
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
├── app.py                            # Streamlit 대시보드 (메인 진입점)
├── requirements.txt                  # 런타임 의존성
├── requirements-dev.txt              # 개발·테스트 의존성 (pytest 등)
├── pytest.ini                        # pytest 설정
├── PROGRESS.md                       # 프로젝트 진행 현황
├── .env.example                      # 환경변수 템플릿
│
├── step1_feature_engineering.py      # 피처 엔지니어링 (원본 → 피처 CSV)
├── step2_find_optimal_k.py           # Elbow + Silhouette로 최적 k 탐색
├── step3_clustering.py               # K-Means 클러스터링 (k=3) 확정
├── step4_compare_k.py                # k별 클러스터 품질 비교
├── step5_validation.py               # 홀드아웃·다중시드·Bootstrap Jaccard 검증
├── step6_lightgbm.py                 # LightGBM 분류기 학습 및 저장
├── step7_lstm.py                     # LSTM 시계열 모델 (실험적)
│
├── .streamlit/
│   ├── config.toml                   # Streamlit 테마·서버 설정
│   └── secrets.toml                  # 로컬 시크릿 (git 제외)
├── assets/
│   ├── screenshot.png                # 대시보드 실행 화면
│   ├── cluster_comorbidity.png       # 유형별 동반 질환 프로파일 차트
│   └── model_performance.png         # LightGBM 피처 중요도·혼동행렬
├── config/
│   ├── config.yaml                   # 공통 설정
│   ├── config.dev.yaml               # 개발 환경 오버라이드
│   └── config.prod.yaml              # 운영 환경 오버라이드
│
├── src/
│   ├── database.py                   # SQLAlchemy 엔진·세션 초기화
│   ├── api/
│   │   ├── main.py                   # FastAPI 앱 (6개 엔드포인트)
│   │   ├── predictor.py              # 모델 로드 + 추론 (LightGBM → K-Means 폴백)
│   │   └── schemas.py                # Pydantic 입출력 스키마
│   ├── analysis/
│   │   ├── correlation.py            # 피처 상관관계 분석
│   │   └── rhinitis_patient_profile.py  # Healthcare 증상 분석·Kaggle 클러스터 병합
│   ├── data/
│   │   ├── preprocess.py             # 전처리 파이프라인 (결측·이상치·병합)
│   │   └── api_collector.py          # 에어코리아·기상청·꽃가루 API (TTL 캐시)
│   ├── models/
│   │   ├── clustering.py             # K-Means 학습·저장 유틸
│   │   ├── history.py                # 예측 이력 SQLAlchemy ORM 모델
│   │   └── train_symptom_model.py    # 증상 모델 학습 스크립트
│   └── utils/
│       ├── config.py                 # 환경별 YAML 설정 로더
│       ├── history.py                # 이력 저장·조회·가상 데이터 생성
│       └── logging_config.py         # 중앙화된 로깅 설정
│
├── data/
│   ├── raw/
│   │   └── Healthcare.csv            # 증상 텍스트 데이터 25,000건 (보조)
│   └── processed/
│       ├── rhinitis_features.csv     # 피처 엔지니어링 결과
│       └── rhinitis_clustered.csv    # K-Means 클러스터 레이블 포함
├── outputs/
│   ├── models/
│   │   ├── lgbm_rhinitis.pkl         # LightGBM 분류기 (주 모델, 898 KB)
│   │   └── kmeans_rhinitis.pkl       # K-Means 클러스터러 (폴백, 8 KB)
│   └── figures/                      # EDA·클러스터링 시각화 PNG
├── tests/
│   ├── conftest.py
│   ├── test_api.py                   # FastAPI 엔드포인트 16개 테스트
│   ├── test_predictor.py             # 모델 추론 유효성 24개 테스트
│   └── test_schemas.py               # Pydantic 스키마 경계 7개 테스트
└── notebooks/
    ├── 01_EDA.ipynb                  # 탐색적 데이터 분석
    ├── 02_Clustering.ipynb           # 클러스터링 실험
    └── 03_rhinitis_patient_profile.ipynb  # Healthcare 병합 분석
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
KMA_API_KEY      = "your_kma_api_key_here"

# 기상청 꽃가루농도위험지수 조회서비스(3.0) 키 (선택 — 없으면 AIRKOREA_API_KEY 폴백)
POLLEN_API_KEY = "your_pollen_api_key_here"

# 예측 이력 저장용 Supabase 연결 문자열 (선택)
DATABASE_URL = "postgresql://postgres.[project-ref]:[password]@aws-0-ap-southeast-2.pooler.supabase.com:6543/postgres"
```

> Supabase 연결 문자열: 대시보드 → Project Settings → Database → Connection String (URI)

---

## 성공 지표

| 지표 | 목표 |
|---|---|
| AI 유형 분류 일치율 | 사용자 체감과 80% 이상 일치 |
| 증상 개선 만족도 | 가이드 준수 후 설문 4점/5점 이상 |

---

## 변경 이력

### v1.3.0 — 2026-05-10

**꽃가루 API 전환 (에어코리아 → 기상청)**

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 서비스 | 에어코리아 `PollenRiskIdxInqireSvc` | 기상청 `HealthWthrIdxServiceV3` |
| EndPoint | `apis.data.go.kr/B552584/PollenRiskIdxInqireSvc/...` | `apis.data.go.kr/1360000/HealthWthrIdxServiceV3/...` |
| 파라미터 | `sido`, `searchDate` | `areaNo`(10자리 행정구역코드), `time`(YYYYMMDDHH) |
| 오퍼레이션 | 단일 엔드포인트 | `getPinePollenRiskIdxV3` / `getOakPollenRiskIdxV3` / `getBirchPollenRiskIdxV3` / `getWeedPollenRiskIdxV3` |
| 제공 꽃가루 | 오리나무·참나무·소나무·자작나무·쑥·돼지풀 | 소나무·참나무·자작나무·쑥 |

- `POLLEN_API_KEY` 환경변수 추가 (`.env`, `secrets.toml`). 미설정 시 `AIRKOREA_API_KEY`로 자동 폴백
- `secrets.toml` 플레이스홀더 → 실제 키/DB URL로 교체 (Streamlit이 이 파일을 `os.environ`에 주입하므로 `.env`보다 우선)

**Streamlit 경고 해소**

- `use_container_width=True` (deprecated 2025-12-31) → `width='stretch'` 일괄 교체 (11개소)

---

### v1.2.0 — 이전 세션

- O3 오존 경보 로직 추가 (주의보 ≥0.12 / 경보 ≥0.30 / 중대경보 ≥0.50 ppm)
- 측정 지역 11개 → 32개 확장 (서울 25개 구 전체 + 수원·파주·세종·부산·대구·인천·광주)
- `src/database.py` 실제 DB 연결 검증(`SELECT 1`) 추가 — 프로젝트 일시정지 시 즉시 감지
- `src/utils/history.py` 이중 오류 메시지 접두사 버그 수정
- `src/analysis/rhinitis_patient_profile.py` Healthcare 병합 컬럼 불일치 수정 (`AGE_START_YEARS`, `GENDER_FACTOR`) 및 연령대 불일치 시 독립 분석 폴백 추가

---

## 별첨

> **원본 PDF**: [AI기반_비염_증상_예측_및_개인_맞춤형_케어 서비스_수행계획서.pdf](AI기반_비염_증상_예측_및_개인_맞춤형_케어%20서비스_수행계획서.pdf)

---

# AI개발 수행내역서

**2026년도 공공기관 용역과제**

| | |
|---|---|
| **과제명** | AI 기반 비염 증상 예측 및 개인 맞춤형 케어 모델 개발 |
| **담당자** | 김 성 익 |
| **작성일** | 2026년 04월 03일 |

---

### 목차

① 프로젝트 개요  
② 데이터 수집 및 전처리 과정  
③ 탐색적 데이터 분석 (EDA)  
④ K-Means 클러스터 모델  
⑤ 클러스터링 일반화 검증 (Step 5)  
⑥ LightGBM 일반화 검증 (Step 6)  
⑦ LSTM 시계열 모델 검토 (Step 7)  
⑧ 환경 API 실연동 점검  
⑨ Streamlit 대시보드  
⑩ 추가 보완 사항  

---

## 1. 프로젝트 개요

본 프로젝트는 소아 알레르기 코호트 데이터를 기반으로 비염 환자를 동반 질환 특성에 따라 자동 분류하고, 맞춤형 가이드와 실시간 대기·날씨 정보를 결합한 AI 기반 비염 케어 시스템을 구축합니다.

### 1.1 추진배경 및 목적

- **일상 질환 케어 공백 해소**: 중증 질환 관리 서비스는 풍부하나, 비염과 같이 일상생활에 큰 불편함을 주는 경증 만성 질환 대상의 정밀 관리 서비스는 미비함.
- **데이터 기반 의사결정**: 환자가 주관적 느낌에 의존해 약물을 오남용하지 않도록, 증상 데이터와 환경 데이터를 분석하여 객관적인 관리 지표를 제공하고자 함.
- **디지털 헬스케어 국산화**: 해외 서비스(MASK-air)의 우수한 로직을 벤치마킹하여 국내 기상 환경 및 의료적 특성에 최적화된 모델을 구축함.
- 대규모 소아 알레르기 코호트 데이터를 분석하여 비염 환자의 임상적 공통점을 군집화(Clustering)하고, 사용자 맞춤형 관리 솔루션을 제공하는 AI 시스템을 구축함.

### 1.2 과제 범위

| 과제구분 | | 내용 |
|---|---|---|
| ML | 데이터 범위 및 AI 기반 증상 클러스터링 | Kaggle Childhood Allergies 코호트 데이터 (55,567명) / Kaggle Healthcare 증상 프로파일 데이터 (858명) / 환자별 증상·동반질환 패턴을 분석하여 비염 유형(호흡기 알레르기형/비염+천식 복합형/아토픽 마치형) 분류 모델 구현 |
| | 환경 요인 상관관계 분석 | 미세먼지, 꽃가루, 기온 등 외부 요인이 증상 악화에 미치는 영향 분석 |
| | 맞춤형 가이드 생성 | 분류된 유형과 실시간 환경 데이터를 결합한 개인별 행동 지침 도출 |
| | 프로토타입 구축 | FastAPI 추론 서버 + Streamlit 대시보드 배포 |

---

## 2. 데이터 수집 및 전처리

### 2.1 사용 데이터셋

본 프로젝트는 두 가지 데이터셋을 활용했습니다.

| 출처 | 데이터셋 | 샘플 수 | 주요 컬럼 | 용도 |
|---|---|---|---|---|
| Kaggle / Zenodo | Childhood Allergies cohort* | 55,567 | has_asthma, has_food_allergy, rhinitis_onset_age 등 | 메인 학습 데이터 |
| Kaggle | Healthcare.csv (합성데이터) | 25,000 | Age, Gender, Symptoms, Disease | EDA 보조 참고 |

\* cohort: 필라델피아 아동병원(CHOP, Children's Hospital of Philadelphia)의 케어 네트워크

### 2.2 전처리 파이프라인

`src/data/preprocess.py`의 `handle_missing_values()`와 `remove_outliers_iqr()`를 `step1_feature_engineering.py`와 `step3_clustering.py`에 연결해 실제 파이프라인에서 동작하도록 했습니다.

- **결측치 처리**: 연속형 변수는 interpolate(선형 보간), 이진 변수는 0으로 채움
- **이상치 제거**: IQR 기반 클리핑 (하한 Q1−1.5×IQR, 상한 Q3+1.5×IQR)
- **스케일링**: StandardScaler (K-Means 거리 기반 알고리즘 적용)
- **피처 엔지니어링**: 동반 질환 파생 변수 생성 (`atopic_march`, `has_comorbidity`)

| 단계 | 처리 내용 | 적용 함수 |
|---|---|---|
| 결측치 처리 | 연속형 피처(발병 나이, 지속 기간 등) 선형 보간 | `handle_missing_values()` |
| 이상치 클리핑 | IQR×1.5 기준 상·하한 클리핑 | `remove_outliers_iqr()` |
| 표준화 | StandardScaler 적용(클러스터링 직전) | `StandardScaler.fit_transform()` |
| 데이터 병합 | 날짜 기준 증상 데이터 + 환경 데이터 Left join | `merge_symptom_env()` |

---

## 3. 탐색적 데이터 분석 (EDA)

### 3.1 증상 변수 분포

Healthcare.csv 기반 증상 분포와 Childhood Allergies 코호트의 환경·임상 피처 분포를 분석하여 클러스터링에 활용할 핵심 피처를 선정했습니다.

> \[그림 1\] 증상별 VAS 점수 분포 (콧물, 코막힘, 재채기, 눈가려움)

### 3.2 환경 변수 분포

미세먼지(PM10, PM2.5), 기온, 습도, 꽃가루지수의 분포를 확인했습니다. PM10 평균 45.4 μg/m³, 습도 평균 55.0%, 꽃가루지수 평균 47.9로 국내 환경 특성이 반영되었습니다.

> \[그림 2\] 환경 변수 분포 (PM10, PM2.5, 기온, 습도, 꽃가루지수)

### 3.3 증상-환경 Pearson 상관관계 분석

히트맵 분석 결과, 천식 동반 여부, 식품 알레르기 동반 여부, 아토픽 마치 세 변수가 상호 높은 양의 상관관계를 보이며 동반 질환 클러스터 분리의 핵심 피처로 확인됩니다.

> \[그림 3\] 증상-환경 변수 상관관계 히트맵

### 3.4 시계열 증상 패턴

7일 이동평균 기준 월별 증상 점수 변화를 분석하였습니다. 봄(3~5월)과 가을(9~10월) 환절기에 콧물, 재채기 점수가 급격히 상승하는 계절성 패턴이 뚜렷하게 나타났습니다.

> \[그림 4\] 월별 평균 증상 점수 추이 (7일 이동평균, 2024년)

---

## 4. AI 분석 모델 구축 — K-Means 클러스터링

### 4.1 최적 K 선정

Elbow Method와 Silhouette Score를 동시에 분석하여 k=3을 최적으로 결정했습니다. k=3에서 군집 내 응집도와 군집 간 분리도의 균형이 가장 우수했습니다.

> \[그림 5\] 코호트 데이터 최적 클러스터 수 탐색 (Elbow + Silhouette)  
> \[그림 6\] 증상 데이터 기반 Elbow Method 및 Silhouette Score

> ※ k=3은 Silhouette 점수 기준 최고값(k=2)보다 낮으나, 임상적으로 의미 있는 세 가지 비염 표현형을 명확하게 분리한다는 점에서 의료적 판단을 근거로 최종 채택하였습니다.

### 4.2 클러스터 프로파일

k=3으로 확정 후 전체 55,567명을 3개 동반 질환 기반 유형으로 분류했습니다.

| 유형명 | 인원(비율) | 평균 발병 | 천식율 | 아토피율 | 식품 알레르기율 | 특징 |
|---|---|---|---|---|---|---|
| 호흡기 알레르기형 | 26,122명 (47.0%) | 7.9세 | 0% | 0% | 18.6% | 순수 비염, 동반질환 없음 |
| 비염+천식 복합형 | 22,342명 (40.2%) | 7.0세 | 100% | 26.6% | 0% | 천식 100%, 호흡기 집중 관리 필요 |
| 아토픽 마치형 | 7,103명 (12.8%) | 5.9세 | 66.9% | 44.4% | 100% | 식품 알레르기→아토피→비염→천식 |

> \[그림 7\] 클러스터별 특성 프로파일  
> \[그림 8\] 클러스터별 발병 나이 분포

---

## 5. 클러스터링 일반화 성능 검정 (Step 5)

비지도 학습 모델의 신뢰성을 확보하기 위해 3가지 독립적 검증 방법을 적용했습니다. scipy/sklearn 미설치 환경에서도 동일한 결과를 재현할 수 있도록 순수 NumPy 기반으로 구현했습니다.

### 5.1 [A] 홀드아웃 분포 안정성 (80/20 분리)

| 클러스터 | 학습 비율 | 홀드아웃 비율 | 분포 드리프트 |
|---|---|---|---|
| 호흡기 알레르기형 | 0.128 | 0.128 | 0.0004 |
| 비염+천식 복합형 | 0.472 | 0.463 | 0.0090 |
| 아토픽 마치형 | 0.400 | 0.409 | 0.0086 |

최대 드리프트: **0.0090 (< 5%) → 매우 안정** | 학습/홀드아웃 Silhouette: 0.2219 / 0.2238 (차이 0.0019)

### 5.2 [B] 다중 랜덤 시드 Silhouette (10개 시드)

10개 랜덤 시드(0, 7, 13, 21, 42, 55, 77, 88, 99, 123)에서 각 3,000명 샘플로 Silhouette를 계산했습니다.

평균: **0.2228** | 표준편차: **0.0030** | 범위: [0.2173, 0.2262] → 시드 변화에 매우 강건 (std < 0.01)

### 5.3 [C] Bootstrap Jaccard 복원추출 안정성 (50회)

5,000명 서브셋에서 Bootstrap 재샘플링 50회 반복 후 최적 헝가리안 매칭 기반 Jaccard 유사도를 측정했습니다.

평균 Jaccard: **≥ 0.85** → 클러스터 안정성 우수

### 5.4 검증 요약

| 검증 항목 | 결과값 | 기준 | 판정 |
|---|---|---|---|
| [A] 홀드아웃 드리프트 (최대) | 0.0090 | < 5% | 매우 안정 |
| [A] Silhouette 학습/홀드아웃 | 0.2219 / 0.2238 | 차이 0.0019 | 일관성 우수 |
| [B] 다중 시드 Silhouette | 0.2228 ± 0.0030 | std < 0.01 | 시드 강건 |
| [C] Bootstrap Jaccard | ≥ 0.85 | — | 구조 안정 |

> Silhouette 점수 0.22는 이진형 동반 질환 피처의 유클리드 공간 특성상 중간값이지만, 드리프트와 Jaccard 안정성이 기준을 크게 상회하여 모델 신뢰성은 충분합니다.

---

## 6. LightGBM 지도학습 분류기 (Step 6)

K-Means 클러스터 레이블을 의사 타겟(pseudo-target)으로 사용해 LightGBM 지도학습 분류기를 학습했습니다. 이를 통해 클러스터 구조가 지도학습으로 재현 가능한지 검증하고, 새 환자에 대해 보정된 확률 기반 예측을 제공합니다.

### 6.1 모델 설계 하이퍼파라미터

| 하이퍼파라미터 | 설정값 |
|---|---|
| objective | multiclass (3 클래스) |
| n_estimators | 500 (EarlyStopping patience=50) |
| learning_rate | 0.05 |
| num_leaves | 63 (기본값 31보다 높게 설정하여 트리 복잡도 증가) |
| subsample | 0.8 \| colsample_bytree: 0.8 (피처 80%만 사용) |
| 교차검증 | StratifiedKFold 5-Fold (층화방식) |
| 학습 피처 | has_asthma, has_atopic_derm, has_food_allergy, food_allergy_count, rhinitis_onset_age, rhinitis_duration, atopic_march, is_female (8개) |

### 6.2 재현 성능 결과

5-Fold 교차검증 결과: **Accuracy 1.0000 ± 0.0000 | Weighted F1 1.0000 ± 0.0000**

완벽한 분류 성능은 동반 질환 기반 K-Means 클러스터가 임상적으로 매우 명확히 분리됨을 의미합니다. `has_asthma`, `has_food_allergy`, `atopic_march` 3개 피처가 클러스터를 거의 완벽하게 결정합니다.

> \[그림 9\] LightGBM 피처 중요도 및 혼동행렬

### 6.3 프로덕션 적용

`predictor.py`에 LightGBM 우선 / K-Means 폴백 이중 모델 구조를 구현했습니다. `lgbm_rhinitis.pkl`이 존재하면 `predict_proba()`로 보정된 확률을 제공하고, 없을 경우 기존 K-Means 거리 역수 softmax로 자동 폴백합니다.

---

## 7. LSTM 시계열 모델 검토 (Step 7)

비염 증상의 시간적 변화를 모델링하기 위해 LSTM을 검토했습니다. 현재 데이터셋(Childhood Allergies)은 정적 레코드 구조로 시간 축이 없어, AR(1) 자기회귀 프로세스 기반 합성 시계열(n=12,000, T=10 타임스텝)을 생성하여 학습했습니다.

### 7.1 LSTM 아키텍처

입력 형태: (T=10 타임스텝, 4개 증상 피처) → LSTM(128) → Dropout(0.3) → BatchNorm → LSTM(64) → Dense(64, ReLU) → Dense(3, Softmax). Adam(lr=3e-4), EarlyStopping(patience=15).

### 7.2 LightGBM vs LSTM 비교

| 비교 항목 | LightGBM | LSTM |
|---|---|---|
| 데이터 형태 | 정적 임상 피처 (8개) | 합성 시계열 AR(1), T=10 |
| 예상 Accuracy | 1.0000 (실제 달성) | 0.80 ~ 0.90 |
| 예상 F1 (weighted) | 1.0000 (실제 달성) | 0.80 ~ 0.90 |
| 추론 속도 | 매우 빠름 | 상대적으로 느림 |
| 해석 가능성 | 피처 중요도 직접 제공 | 블랙박스 (SHAP 별도 필요) |
| 실데이터 활용 | 현재 55,567명 그대로 | 종단 데이터 필요 |
| 프로덕션 적합성 | 즉시 배포 가능 | 실 시계열 확보 후 재학습 |

**결론**: LightGBM을 현 프로덕션 모델로 채택합니다. LSTM은 실제 환자별 종단 데이터(예: 주간 증상 기록)가 확보된 후 증상 진행 예측 모델로 활용할 계획입니다.

---

## 8. 환경 API 실연동 점검

### 8.1 구현 완료 항목

- 에어코리아 API: PM10, PM2.5, O3 실시간 수집 (TTL 60분 캐시)
- 기상청 API: 기온, 습도, 강수량 단기 예보 (TTL 60분 캐시)
- Tab 4 환경정보 탭: PM10 24시간 추이 차트 + 기온 예보 차트
- Tab 1 분석결과: 환경 데이터 연동 위험 메시지 (미세먼지·습도 임계치 기반)
- 11개 지역 측정소 선택 (서울 5개구 + 인천·수원·파주·대전·대구·부산)

### 8.2 누락 및 개선 필요 항목

| 항목 | 내용 | 우선순위 |
|---|---|---|
| 꽃가루 지수 | 미구현 — 환경부 꽃가루 위험지수 API 별도 연동 필요 (비염 핵심 유발 인자) | 높음 |
| O3 오존 경보 | 오존 데이터 표시만 됨, 임계치(0.06 ppm) 초과 시 경고 로직 없음 | 중간 |
| CLUSTER_STATS 클러스터명 | 구 증상 기반 명칭 → 동반 질환 기반 신규 명칭으로 교체 | 완료 |
| risk_messages 유형 분기 | 구 클러스터명 참조 오류 → 신규 3유형 로직으로 교체 | 완료 |
| Healthcare 데이터 병합오류 | Kaggle 알레르기 데이터 컬럼이 GENDER_FACTOR, BIRTH_YEAR 형식이라 AGE/GENDER 키 생성 불가 → KeyError로 병합 실패 | 임상 문헌 기반 합성값으로 대체, 모든 학습은 Zenodo 알레르기 코호트 데이터만 반영 |

---

## 9. Streamlit 대시보드

### 9.1 탭 구성

| 탭 | 주요 기능 |
|---|---|
| Tab 1: 유형 분석 | 증상 슬라이더 입력 → LightGBM 예측 → 클래스 확률 바 차트 + 레이더 차트 |
| Tab 2: 유형 현황 | 3개 유형 파이 차트 + 유형별 증상 강도 비교 바 차트 |
| Tab 3: 유형 안내 | 각 유형 설명 + 관리 가이드 (CLUSTER_INFO 연동) |
| Tab 4: 환경 정보 | 에어코리아 PM10·PM2.5·O3 + 기상청 기온·습도·강수량 실시간 표시 |

### 9.2 LightGBM 연동 후 UI 변경 사항

- 결과 카드에 LightGBM / K-Means 모델 타입 뱃지 추가
- 3개 유형 클래스 확률을 수평 바 차트로 시각화 (유형별 고유 색상)
- LightGBM 활성 시 '실제 확률 기반 예측' 문구 표시
- K-Means 폴백 시 '비지도 학습 클러스터 거리 기반 예측' 문구 표시
- `cluster_probs` API 응답 필드 추가 (FastAPI ClusterResult 스키마 `model_type` 필드 포함)

---

## 10. 향후 개선 과제

| 과제 | 설명 | 유형 |
|---|---|---|
| 꽃가루 지수 API 연동 | 기상청 HealthWthrIdxServiceV3 → `api_collector.py` 연동 | ~~미구현~~ → **완료** (v1.3.0) |
| O3 오존 경보 로직 | 오존 0.12 ppm 초과 시 경고 메시지 추가 | ~~미구현~~ → **완료** (v1.2.0) |
| is_female 입력 UI | LightGBM 8개 피처 중 `is_female`이 사이드바 미노출 → 기본값 0으로 처리 중. 성별 선택 UI 추가 | 미구현 |
| LSTM 실 데이터 확보 | 환자별 주간 증상 기록 누적 시 LSTM 재학습으로 증상 진행 예측 가능. 현재는 AR(1) 합성 시계열 사용 | 계획 |
| DSM 기반 증상 점수 | 현재 증상 피처는 VAS 0~10 슬라이더. 임상 분류(DSM/ARIA) 기반 구조화된 증상 점수체계로 고도화 필요 | 계획 |
| 모델 재학습 자동화 | 신규 예측 데이터가 DB에 1,000건 이상 쌓이면 step6를 자동 재실행하는 MLOps 파이프라인 구축 | 계획 |
| API 인증/보안 | FastAPI 엔드포인트에 JWT 인증 미적용. Streamlit Cloud 배포 시 보안 강화 필요 | 계획 |

---

## 부록 A. 데이터 딕셔너리

본 부록은 모델 학습에 사용된 피처, API 입출력 스키마, 환경 데이터 컬럼의 정의 및 통계를 제공합니다.

### A-1. 학습 데이터셋 컬럼 (소아 알레르기 건강 코호트)

| 컬럼명 | 타입 | 범위/값 | 통계 | 설명 |
|---|---|---|---|---|
| SUBJECT_ID | int | 1 – N | 고유 식별자 | 환자 고유 식별 번호 |
| has_asthma | int | 0 / 1 | 1=48.8% | 천식 동반 여부 (0: 없음, 1: 있음) |
| has_atopic_derm | int | 0 / 1 | 1=25.1% | 아토피 피부염 동반 여부 |
| has_food_allergy | int | 0 / 1 | 1=12.8% | 식품 알레르기 동반 여부 |
| food_allergy_count | int | 0 – 10 | avg=0.2 | 식품 알레르기 항원 종류 수 |
| rhinitis_onset_age | float | 0 – 18세 | avg=7.3, SD=4.2 | 비염 최초 진단 나이 (세) |
| rhinitis_duration | float | 0 – 18년 | avg=2.2, SD=2.9 | 비염 유병 기간 (년) |
| atopic_march | int | 0 / 1 | 1=0.6% | 아토픽 마치 (아토피→천식→비염) 해당 여부 |
| is_female | int | 0 / 1 | 1=44.9% | 성별 (0: 남, 1: 여) |
| cluster | int | 0 / 1 / 2 | K-Means 레이블 | K-Means 군집 ID (LightGBM 학습 타깃) |
| cluster_label | str | 3개 유형명 | — | 군집 ID에 대응하는 한글 유형명 |

### A-2. API 입력 스키마 (PatientInput)

★ 표시 8개 필드는 LightGBM 예측 피처로 직접 사용됩니다. 증상 점수(`symptom_*`)는 대시보드 표시용이며 모델 예측에는 미사용입니다.

| 필드명 | 타입 | 범위 | 기본값 | LightGBM 피처 | 설명 |
|---|---|---|---|---|---|
| has_asthma | int | 0–1 | — | ★ | 천식 동반 여부 |
| has_atopic_derm | int | 0–1 | — | ★ | 아토피 피부염 동반 여부 |
| has_food_allergy | int | 0–1 | — | ★ | 식품 알레르기 동반 여부 |
| food_allergy_count | int | 0–10 | 0 | ★ | 식품 알레르기 항원 수 |
| rhinitis_onset_age | float | ≥ 0 | — | ★ | 비염 발병 나이 (세) |
| rhinitis_duration | float | ≥ 0 | 0.0 | ★ | 비염 지속 기간 (년) |
| atopic_march | int | 0–1 | 0 | ★ | 아토픽 마치 해당 여부 |
| is_female | int | 0–1 | — | ★ | 성별 (0: 남, 1: 여) |
| symptom_rhinorrhea | int | 0–10 | 0 | — | 콧물 증상 강도 (VAS 0–10) |
| symptom_congestion | int | 0–10 | 0 | — | 코막힘 증상 강도 |
| symptom_sneezing | int | 0–10 | 0 | — | 재채기/가려움 강도 |
| symptom_ocular | int | 0–10 | 0 | — | 눈 가려움/충혈 강도 |

### A-3. API 응답 스키마 (ClusterResult)

| 필드명 | 타입 | 예시 값 | 설명 |
|---|---|---|---|
| cluster_id | int | 1 | 군집 번호 (0 / 1 / 2) |
| cluster_label | str | 비염+천식 복합형 | 예측 유형명 |
| confidence | float | 0.82 | 예측 신뢰도 (0–1). LightGBM: predict_proba 최대값; K-Means: 거리 역수 softmax 최대값 |
| cluster_probs | dict[str, float] | {유형A: 0.10, ...} | 3개 유형별 소속 확률 합계=1 |
| description | str | 천식 100% 동반… | 유형 설명 텍스트 |
| guide | list[str] | ["흡입기…", …] | 유형별 맞춤 관리 가이드 목록 |
| model_type | str | lightgbm | 사용된 모델 ('lightgbm' 또는 'kmeans') |

### A-4. 환경 데이터 컬럼 (실시간 API)

AirKorea(대기) + KMA(기상) API를 통해 60분 TTL 캐시로 수집됩니다.

| 컬럼명 | 출처 API | 단위/범위 | 수집 주기 | 비고 | 설명 |
|---|---|---|---|---|---|
| pm10 | AirKorea | μg/m³ ≥ 0 | 60 min | 실 연동 | 미세먼지 농도 (PM10) |
| pm25 | AirKorea | μg/m³ ≥ 0 | 60 min | 실 연동 | 초미세먼지 농도 (PM2.5) |
| ozone | AirKorea | ppm ≥ 0 | 60 min | 실 연동 | 오존 농도 (O₃) |
| temperature | KMA | °C | 60 min | 실 연동 | 기온 |
| humidity | KMA | % 0–100 | 60 min | 실 연동 | 상대 습도 |
| precipitation | KMA | mm ≥ 0 | 60 min | 실 연동 | 강수량 |
| pollen_index | 기상청 | 1–4 등급 | 60 min | **v1.3.0 연동 완료** | 꽃가루 위험 등급 (소나무·참나무·자작나무·쑥) |
