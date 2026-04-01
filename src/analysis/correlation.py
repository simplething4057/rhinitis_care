"""
증상-환경 상관관계 분석 모듈
- Pearson 상관계수 히트맵
- 환경 변수별 증상 영향도 분석
- 통계 리포트 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VAS_COLS = ["nasal_discharge", "nasal_congestion", "sneezing", "eye_itching"]
ENV_COLS = ["pm10", "pm25", "temperature", "humidity", "pollen_index"]
COL_KR = {
    "nasal_discharge": "콧물",
    "nasal_congestion": "코막힘",
    "sneezing": "재채기",
    "eye_itching": "눈가려움",
    "pm10": "미세먼지(PM10)",
    "pm25": "초미세먼지(PM2.5)",
    "temperature": "기온(℃)",
    "humidity": "습도(%)",
    "pollen_index": "꽃가루지수",
}


# ──────────────────────────────────────────────
# 1. 전체 상관관계 히트맵
# ──────────────────────────────────────────────
def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list | None = None,
    save_path: str | None = None,
    title: str = "증상-환경 상관관계 히트맵",
):
    cols = cols or VAS_COLS + ENV_COLS
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr(method="pearson")

    # 한글 레이블
    labels = [COL_KR.get(c, c) for c in available]
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 상삼각 마스크
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, pad=15)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"히트맵 저장: {save_path}")
    plt.show()
    return corr


# ──────────────────────────────────────────────
# 2. 환경 변수 → 증상 상관계수 테이블 + p-value
# ──────────────────────────────────────────────
def compute_symptom_env_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    환경 변수와 각 증상 간 Pearson r, p-value, 유의성 계산.
    Returns: DataFrame (환경변수 × 증상)
    """
    records = []
    for env in ENV_COLS:
        if env not in df.columns:
            continue
        for sym in VAS_COLS:
            if sym not in df.columns:
                continue
            valid = df[[env, sym]].dropna()
            if len(valid) < 5:
                continue
            r, p = stats.pearsonr(valid[env], valid[sym])
            records.append({
                "환경변수": COL_KR.get(env, env),
                "증상":     COL_KR.get(sym, sym),
                "상관계수(r)": round(r, 4),
                "p-value":     round(p, 4),
                "유의성":      "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")),
            })

    result = pd.DataFrame(records)
    logger.info(f"상관관계 분석 완료: {len(result)}개 쌍")
    return result


# ──────────────────────────────────────────────
# 3. 산점도 (환경변수 vs 증상)
# ──────────────────────────────────────────────
def plot_scatter_env_symptom(
    df: pd.DataFrame,
    env_col: str = "pm10",
    sym_col: str = "nasal_congestion",
    save_path: str | None = None,
):
    if env_col not in df.columns or sym_col not in df.columns:
        logger.warning(f"컬럼 없음: {env_col}, {sym_col}")
        return

    valid = df[[env_col, sym_col]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(valid[env_col], valid[sym_col], alpha=0.5, edgecolors="white", s=50)

    # 회귀선
    m, b = np.polyfit(valid[env_col], valid[sym_col], 1)
    x_line = np.linspace(valid[env_col].min(), valid[env_col].max(), 100)
    ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5)

    r, p = stats.pearsonr(valid[env_col], valid[sym_col])
    ax.set_xlabel(COL_KR.get(env_col, env_col))
    ax.set_ylabel(COL_KR.get(sym_col, sym_col))
    ax.set_title(f"{COL_KR.get(env_col, env_col)} vs {COL_KR.get(sym_col, sym_col)}\n(r={r:.3f}, p={p:.4f})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────
# 4. 자동 인사이트 문장 생성
# ──────────────────────────────────────────────
def generate_insights(corr_df: pd.DataFrame, threshold: float = 0.3) -> list[str]:
    """
    상관계수가 임계값 이상인 쌍에 대해 자연어 인사이트 문장 생성.
    """
    insights = []
    for _, row in corr_df.iterrows():
        r = row["상관계수(r)"]
        if abs(r) < threshold:
            continue
        direction = "높을수록" if r > 0 else "낮을수록"
        strength = "강한" if abs(r) >= 0.5 else "중간 수준의"
        sentence = (
            f"• {row['환경변수']}이(가) {direction} "
            f"{row['증상']} 증상이 악화되는 {strength} 상관관계가 있습니다. "
            f"(r={r:+.3f}, {row['유의성']})"
        )
        insights.append(sentence)

    if not insights:
        insights.append("유의미한 상관관계(|r| ≥ 0.3)가 발견되지 않았습니다.")

    return insights


if __name__ == "__main__":
    # 더미 테스트
    np.random.seed(0)
    n = 200
    dummy = pd.DataFrame({
        "nasal_discharge": np.random.randint(0, 10, n),
        "nasal_congestion": np.random.randint(0, 10, n),
        "sneezing":         np.random.randint(0, 10, n),
        "eye_itching":      np.random.randint(0, 10, n),
        "pm10":          np.random.randint(10, 150, n),
        "pm25":          np.random.randint(5, 80, n),
        "temperature":   np.random.uniform(-5, 35, n),
        "humidity":      np.random.uniform(20, 90, n),
        "pollen_index":  np.random.randint(0, 200, n),
    })
    corr_df = compute_symptom_env_correlation(dummy)
    print(corr_df)
    insights = generate_insights(corr_df)
    for s in insights:
        print(s)
