"""
FastAPI 메인 앱
비염 유형 분류 & 맞춤 가이드 API
"""
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from src.utils.config import load_config
from src.utils.logging_config import setup_logging
from src.api.schemas import PatientInput, PredictResponse, ClusterResult, HealthResponse
from src.api.predictor import predictor

# ── 설정 로드 ─────────────────────────────────────────
cfg = load_config()
server_cfg = cfg.get("server", {})

# ── 로깅 초기화 (앱 시작 시 한 번) ───────────────────
setup_logging(
    log_level=server_cfg.get("log_level", "INFO"),
    enable_file=(cfg.get("_env", "dev") == "prod"),
)

logger = logging.getLogger(__name__)

# ── 앱 초기화 ─────────────────────────────────────────
app = FastAPI(
    title="비염 케어 AI API",
    description="동반 질환 정보를 입력하면 비염 유형을 분류하고 맞춤형 관리 가이드를 제공합니다.",
    version="0.1.0",
)

# ── CORS 설정 (환경별 분리) ──────────────────────────
ALLOWED_ORIGINS = server_cfg.get("cors_origins", ["http://localhost:3000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 전역 예외 핸들러 ─────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):  # noqa: ARG001
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다.", "error": str(exc)},
    )


# ── 엔드포인트 ────────────────────────────────────────
@app.get("/", tags=["root"])
def root():
    return {
        "message": "비염 케어 AI API가 실행 중입니다. /docs 에서 API 문서를 확인하세요.",
        "env": cfg.get("_env", "dev"),
    }


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
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    try:
        result = predictor.predict(data.model_dump())
    except Exception as e:
        logger.error(f"예측 오류: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"예측 실패: {e}")
    summary = (
        f"입력하신 정보를 분석한 결과 '",
        f"{result['cluster_label']}' 유형으로 분류됐습니다. (신뢰도: {result['confidence']*100:.1f}%)"
    )
    return PredictResponse(
        result=ClusterResult(**result),
        summary=" ".join(summary),
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
            detail=f"유형 '{cluster_label}'을 찾을 수 없습니다. 사용 가능한 유형: {list(CLUSTER_INFO.keys())}"
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


@app.get("/env", tags=["env"])
def get_env_info():
    """
    서버 환경 및 주요 패키지/모델 정보 반환
    """
    import sys
    import platform
    import pkg_resources
    from src.api.predictor import MODEL_PATH, predictor

    def get_version(pkg):
        try:
            return pkg_resources.get_distribution(pkg).version
        except Exception:
            return None

    return {
        "app_env": cfg.get("_env", "dev"),
        "python": sys.version,
        "platform": platform.platform(),
        "fastapi": get_version("fastapi"),
        "numpy": get_version("numpy"),
        "joblib": get_version("joblib"),
        "model_path": MODEL_PATH,
        "model_loaded": predictor.is_loaded,
        "cors_origins": ALLOWED_ORIGINS,
    }


# ── 커스텀 OpenAPI 문서화 ────────────────────────────
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=(
            app.description
            + "\n\n- /predict: 비염 유형 예측 및 가이드"
            + "\n- /guide/{cluster_label}: 유형별 가이드"
            + "\n- /clusters: 유형 목록"
            + "\n- /env: 환경 정보"
            + "\n- /health: 서버/모델 상태"
        ),
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


# ── 직접 실행 시 uvicorn 구동 ────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit("uvicorn이 설치되지 않았습니다: pip install uvicorn") from e
    uvicorn.run(
        "src.api.main:app",
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 8000),
        reload=server_cfg.get("reload", True),
        log_level=server_cfg.get("log_level", "info"),
    )
