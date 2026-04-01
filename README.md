# 🌿 비염 케어 AI — 증상 클러스터링 & 맞춤형 가이드

> **"병원에 가기엔 애매하지만 괴로운 오늘, 데이터로 해결책을 제시합니다."**

---

## 프로젝트 개요

비염 등 만성 경증 질환자를 위한 **증상-환경 상관관계 분석** 기반 개인 맞춤형 케어 가이드 시스템입니다.

- 증상(VAS 점수)과 환경 데이터(미세먼지·꽃가루·기온)를 결합하여 분석
- K-Means 클러스터링으로 비염 유형 자동 분류
- 실시간 환경 데이터 기반 행동 가이드 생성

---

## 디렉토리 구조

```
rhinitis_care/
├── config/
│   └── config.yaml          # 전체 설정 (API키, 모델 파라미터 등)
├── data/
│   ├── raw/                 # 원본 데이터 (CSV, 공공API 결과)
│   ├── processed/           # 전처리 완료 데이터
│   └── external/            # 외부 참조 데이터
├── notebooks/
│   ├── 01_EDA.ipynb         # 탐색적 데이터 분석
│   └── 02_Clustering.ipynb  # K-Means 클러스터링
├── src/
│   ├── data/
│   │   ├── preprocess.py    # 전처리 파이프라인
│   │   └── api_collector.py # 에어코리아·기상청 API 수집
│   ├── models/
│   │   └── clustering.py    # K-Means 학습·예측
│   ├── analysis/
│   │   └── correlation.py   # 상관관계 분석·시각화
│   └── utils/               # 공통 유틸리티
├── outputs/
│   ├── figures/             # 시각화 결과 이미지
│   └── reports/             # 분석 리포트
├── .env.example             # 환경변수 예시
├── .gitignore
└── requirements.txt
```

---

## 빠른 시작

### 1. 가상환경 생성 및 패키지 설치

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

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

### 3. Jupyter 노트북 실행

```bash
jupyter notebook notebooks/
```

---

## 분석 흐름

```
데이터 수집 (api_collector.py)
    ↓
전처리·병합 (preprocess.py)
    ↓
EDA·상관관계 (01_EDA.ipynb / correlation.py)
    ↓
K-Means 클러스터링 (02_Clustering.ipynb / clustering.py)
    ↓
유형별 맞춤 가이드 생성
```

---

## 비염 유형 분류 (3대 클러스터)

| 유형 | 주요 증상 | 특징 |
|---|---|---|
| **콧물-재채기 우세형** | 콧물↑ 재채기↑ | 꽃가루·알레르기 유발 물질 민감 |
| **코막힘 우세형** | 코막힘↑ | 기온차·미세먼지 민감, 혈관운동성 비염 의심 |
| **자극 민감형** | 모든 증상 중등도 + 눈가려움↑ | 복합 알레르기 반응 |

---

## 핵심 모듈

### `src/data/preprocess.py`
- CSV 로드, 결측치(보간), 이상치(IQR), 표준화, 날짜 병합

### `src/data/api_collector.py`
- 에어코리아 PM10·PM2.5 실시간 수집
- 기상청 기온·습도 단기예보 수집

### `src/models/clustering.py`
- Elbow Method + Silhouette Score로 최적 k 탐색
- K-Means 학습·저장·예측

### `src/analysis/correlation.py`
- Pearson 상관계수 히트맵
- 증상-환경 쌍별 r, p-value 테이블
- 자동 인사이트 문장 생성

---

## 참고 데이터셋

- [Kaggle: Allergic Rhinitis Dataset](https://www.kaggle.com/) — VAS 증상 점수
- [에어코리아 API](https://www.airkorea.or.kr/) — 미세먼지 실시간 관측
- [기상청 공공데이터](https://www.data.go.kr/) — 꽃가루·기온·습도

---

## 성공 지표

| 지표 | 목표 |
|---|---|
| 7일 연속 기록 Retention | 40% 이상 |
| AI 유형 분류 일치율 | 사용자 체감과 80% 일치 |
| 증상 개선 만족도 | 서비스 가이드 준수 후 설문 4점/5점 이상 |
