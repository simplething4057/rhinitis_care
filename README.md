# 🌿 비염 케어 AI — 증상 클러스터링 & 맞춤형 가이드

> **"병원에 가기엔 애매하지만 괴로운 오늘, 데이터로 해결책을 제시합니다."**

---

## 프로젝트 개요

55,567명의 소아 알레르기 코호트 데이터를 K-Means 클러스터링으로 분석해
비염 환자를 3가지 유형으로 분류하고 맞춤형 관리 가이드를 제공하는 AI 서비스입니다.

- 동반 질환(천식·아토피·식품알레르기) 정보 기반 비염 유형 자동 분류
- Streamlit 대시보드로 누구나 쉽게 사용
- FastAPI 백엔드 + 직접 추론 모드 지원 (API 서버 없이도 동작)

---

## 비염 3대 유형

| 유형 | 규모 | 천식 | 아토피 | 식품알레르기 | 평균 발병 나이 |
|---|---|---|---|---|---|
| 🌿 **호흡기 알레르기형** | 47% (26,122명) | 0% | 18.6% | 0% | 7.9세 |
| 💨 **비염+천식 복합형** | 40% (22,342명) | 100% | 26.6% | 0% | 7.0세 |
| 🔶 **아토픽 마치형** | 13% (7,103명) | 66.9% | 44.0% | 100% | 5.9세 |

---

## 빠른 시작

### 1. 패키지 설치

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 API 키 입력
```

| 환경변수 | 발급처 | 용도 |
|---|---|---|
| `AIRKOREA_API_KEY` | 공공데이터포털 → 에어코리아 | 미세먼지 실시간 수집 |
| `KMA_API_KEY` | 공공데이터포털 → 기상청 단기예보 | 기온·습도 수집 |

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

## 디렉토리 구조

```
rhinitis_care/
├── app.py                       # Streamlit 대시보드
├── requirements.txt
├── .env.example                 # 환경변수 예시
├── .streamlit/
│   └── config.toml              # Streamlit 테마/서버 설정
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
│   │   └── clustering.py        # K-Means 학습·저장
│   └── utils/
│       ├── config.py            # 환경별 설정 로더
│       └── logging_config.py    # 중앙화된 로깅 설정
├── data/
│   ├── raw/                     # 원본 데이터
│   └── processed/               # 전처리 완료 데이터
├── outputs/
│   ├── models/
│   │   └── kmeans_childhood.pkl # 학습된 K-Means 모델
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
피처 엔지니어링 (step1_feature_engineering.py)
    ↓
최적 클러스터 수 탐색 (step2_find_optimal_k.py)
    ↓
K-Means 클러스터링 확정 (step3_clustering.py, k=3)
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

---

## Streamlit Cloud 배포

1. GitHub에 push
2. [Streamlit Cloud](https://streamlit.io/cloud) → New app → 이 저장소 선택
3. Main file: `app.py`
4. Secrets 메뉴에서 `.streamlit/secrets.toml` 내용 입력

---

## 성공 지표

| 지표 | 목표 |
|---|---|
| AI 유형 분류 일치율 | 사용자 체감과 80% 이상 일치 |
| 증상 개선 만족도 | 가이드 준수 후 설문 4점/5점 이상 |
