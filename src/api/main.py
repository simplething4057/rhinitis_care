"""
FastAPI 메인 앱
비염 유형 분류 & 맞춤 가이드 API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PatientInput, PredictResponse, ClusterResult, HealthResponse
from src.api.predictor import predictor

# ── 앱 초기화 ─────────────────────────────────────────
app = FastAPI(
    title="비염 케어 AI API",
    description="동반 질환 정보를 입력하면 비염 유형을 분류하고 맞춤형 관리 가이드를 제공합니다.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 엔드포인트 ────────────────────────────────────────
@app.get("/", tags=["root"])
def root():
    return {"message": "비염 케어 AI API가 실행 중입니다. /docs 에서 API 문서를 확인하세요."}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check():
    """서버 및 모델 상태 확인"""
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
    )


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(data: PatientInput):
    """
    동반 질환 정보를 입력받아 비염 유형을 분류하고 맞춤 가이드를 반환합니다.

    - **has_asthma**: 천식 보유 여부 (0 또는 1)
    - **has_atopic_derm**: 아토피 보유 여부 (0 또는 1)
    - **has_food_allergy**: 식품알레르기 보유 여부 (0 또는 1)
    - **food_allergy_count**: 식품알레르기 종류 수 (0~10)
    - **rhinitis_onset_age**: 비염 발병 나이 (세)
    - **rhinitis_duration**: 비염 지속 기간 (년)
    - **atopic_march**: 아토픽 마치 여부 (0 또는 1)
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    result = predictor.predict(data.model_dump())

    summary = (
        f"입력하신 정보를 분석한 결과 "
        f"'{result['cluster_label']}' 유형으로 분류됐습니다. "
        f"(신뢰도: {result['confidence']*100:.1f}%)"
    )

    return PredictResponse(
        result=ClusterResult(**result),
        summary=summary,
    )


@app.get("/guide/{cluster_label}", tags=["guide"])
def get_guide(cluster_label: str):
    """
    클러스터 유형명으로 관리 가이드를 직접 조회합니다.
    예: /guide/호흡기 알레르기형
    """
    from src.api.predictor import CLUSTER_INFO
    info = CLUSTER_INFO.get(cluster_label)
    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"유형 '{cluster_label}'을 찾을 수 없습니다. "
                   f"사용 가능한 유형: {list(CLUSTER_INFO.keys())}"
        )
    return {"cluster_label": cluster_label, **info}


@app.get("/clusters", tags=["guide"])
def list_clusters():
    """분류 가능한 모든 비염 유형 목록 조회"""
    from src.api.predictor import CLUSTER_INFO
    return {
        "clusters": [
            {"label": label, "description": info["description"]}
            for label, info in CLUSTER_INFO.items()
        ]
    }
