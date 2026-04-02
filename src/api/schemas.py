"""
요청/응답 데이터 스키마 정의
"""
from pydantic import BaseModel, Field
from typing import Optional


# ── 입력 스키마 ──────────────────────────────────────
class PatientInput(BaseModel):
    """사용자 입력: 동반 질환 및 정보 + 비염 증상"""
    has_asthma:         int = Field(..., ge=0, le=1, description="천식 보유 여부 (0/1)")
    has_atopic_derm:    int = Field(..., ge=0, le=1, description="아토피 보유 여부 (0/1)")
    has_food_allergy:   int = Field(..., ge=0, le=1, description="식품알레르기 보유 여부 (0/1)")
    food_allergy_count: int = Field(0,  ge=0, le=10, description="식품알레르기 종류 수")
    rhinitis_onset_age: float = Field(..., ge=0, description="비염 발병 나이 (세)")
    rhinitis_duration:  float = Field(0.0, ge=0, description="비염 지속 기간 (년)")
    atopic_march:       int = Field(0,  ge=0, le=1, description="아토픽 마치 여부 (0/1)")
    
    # 증상 점수 (0~10)
    symptom_rhinorrhea: int = Field(0, ge=0, le=10, description="콧물 정도")
    symptom_congestion: int = Field(0, ge=0, le=10, description="코막힘 정도")
    symptom_sneezing:   int = Field(0, ge=0, le=10, description="재채기/가려움 정도")
    symptom_ocular:     int = Field(0, ge=0, le=10, description="눈 가려움/충혈 정도")

    class Config:
        json_schema_extra = {
            "example": {
                "has_asthma": 0,
                "has_atopic_derm": 0,
                "has_food_allergy": 0,
                "food_allergy_count": 0,
                "rhinitis_onset_age": 8.0,
                "rhinitis_duration": 2.0,
                "atopic_march": 0,
                "symptom_rhinorrhea": 5,
                "symptom_congestion": 3,
                "symptom_sneezing": 6,
                "symptom_ocular": 2
            }
        }


# ── 응답 스키마 ──────────────────────────────────────
class ClusterResult(BaseModel):
    """클러스터링 예측 결과"""
    cluster_id:    int
    cluster_label: str
    confidence:    float = Field(..., description="예측 확률 (LightGBM: 보정 확률 / K-Means: 거리 역수 softmax)")
    cluster_probs: dict[str, float] = Field(default_factory=dict, description="각 유형별 소속 확률")
    description:   str
    guide:         list[str]
    model_type:    str = Field("kmeans", description="사용된 모델: 'lightgbm' | 'kmeans'")


class PredictResponse(BaseModel):
    """예측 API 전체 응답"""
    status:  str = "success"
    result:  ClusterResult
    summary: str


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status:     str
    model_loaded: bool
    version:    str = "0.1.0"
