"""
데이터 전처리 모듈
- 증상 데이터 (VAS 점수) 로드 및 정제
- 환경 데이터 (미세먼지, 꽃가루, 기온) 로드 및 정제
- 두 데이터셋 날짜 기준 병합
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# 1. 증상 데이터 로드
# ──────────────────────────────────────────────
def load_symptom_data(file_path: str) -> pd.DataFrame:
    """
    Kaggle Allergic Rhinitis 증상 CSV 로드.
    예상 컬럼: date, user_id, nasal_discharge, nasal_congestion,
               sneezing, eye_itching, medication, notes
    """
    df = pd.read_csv(file_path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()
    logger.info(f"증상 데이터 로드 완료: {df.shape}")
    return df


# ──────────────────────────────────────────────
# 2. 환경 데이터 로드
# ──────────────────────────────────────────────
def load_env_data(file_path: str) -> pd.DataFrame:
    """
    공공데이터 환경 CSV 로드.
    예상 컬럼: date, pm10, pm25, temperature, humidity, pollen_index
    """
    df = pd.read_csv(file_path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower()
    logger.info(f"환경 데이터 로드 완료: {df.shape}")
    return df


# ──────────────────────────────────────────────
# 3. 결측치 처리
# ──────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame, strategy: str = "interpolate") -> pd.DataFrame:
    """
    결측치 처리.
    strategy: 'interpolate' | 'ffill' | 'drop'
    """
    missing = df.isnull().sum()
    if missing.any():
        logger.info(f"결측치 현황:\n{missing[missing > 0]}")

    if strategy == "interpolate":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    elif strategy == "ffill":
        df = df.fillna(method="ffill").fillna(method="bfill")
    elif strategy == "drop":
        df = df.dropna()

    logger.info(f"결측치 처리 완료 (strategy={strategy})")
    return df


# ──────────────────────────────────────────────
# 4. 이상치 처리 (IQR 방법)
# ──────────────────────────────────────────────
def remove_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """IQR 기반 이상치를 상/하한값으로 클리핑."""
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        if clipped:
            logger.info(f"{col}: {clipped}개 이상치 클리핑 ({lower:.2f} ~ {upper:.2f})")
    return df


# ──────────────────────────────────────────────
# 5. 표준화
# ──────────────────────────────────────────────
def standardize(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, StandardScaler]:
    """지정 컬럼 StandardScaler 표준화. scaler도 반환(역변환용)."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logger.info(f"표준화 완료: {columns}")
    return df, scaler


# ──────────────────────────────────────────────
# 6. 데이터 병합
# ──────────────────────────────────────────────
def merge_symptom_env(symptom_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜 기준으로 증상 데이터 + 환경 데이터 left join.
    증상 데이터의 date를 기준으로 환경 데이터 매칭.
    """
    merged = pd.merge(symptom_df, env_df, on="date", how="left")
    logger.info(f"데이터 병합 완료: {merged.shape} (환경 결측 {merged['pm10'].isnull().sum()}건)")
    return merged


# ──────────────────────────────────────────────
# 7. 파이프라인 실행 (통합 전처리)
# ──────────────────────────────────────────────
def run_preprocessing_pipeline(
    symptom_path: str,
    env_path: str,
    output_path: str,
    config: dict | None = None
) -> pd.DataFrame:
    """전처리 전 과정을 한 번에 실행."""

    # 1) 로드
    symptom_df = load_symptom_data(symptom_path)
    env_df = load_env_data(env_path)

    # 2) 결측치 처리
    vas_cols = ["nasal_discharge", "nasal_congestion", "sneezing", "eye_itching"]
    env_cols = ["pm10", "pm25", "temperature", "humidity", "pollen_index"]

    symptom_df = handle_missing_values(symptom_df, strategy="interpolate")
    env_df = handle_missing_values(env_df, strategy="interpolate")

    # 3) 이상치 처리
    symptom_df = remove_outliers_iqr(symptom_df, vas_cols)
    env_df = remove_outliers_iqr(env_df, env_cols)

    # 4) 병합
    merged = merge_symptom_env(symptom_df, env_df)

    # 5) 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"전처리 완료 → 저장: {output_path}")

    return merged


if __name__ == "__main__":
    df = run_preprocessing_pipeline(
        symptom_path="data/raw/symptom_data.csv",
        env_path="data/raw/env_data.csv",
        output_path="data/processed/merged_data.csv",
    )
    print(df.head())
