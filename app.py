"""
비염 케어 AI — Streamlit 대시보드

동작 모드 (자동 선택):
  - API 모드  : STREAMLIT_API_URL 또는 config의 api_base_url로 FastAPI 호출
  - 직접 모드 : API 미연결 시 predictor를 직접 임포트해 추론 (Streamlit Cloud 무료 배포용)
"""
import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv

from src.utils.history import save_history, get_recent_history, generate_synthetic_history

load_dotenv()

# Streamlit Cloud secrets → os.environ 브릿지
for _k in ("AIRKOREA_API_KEY", "KMA_API_KEY", "APP_ENV"):
    if _k in st.secrets and not os.getenv(_k):
        os.environ[_k] = st.secrets[_k]


def _hex_rgba(hex_color: str, alpha: float = 0.27) -> str:
    """#RRGGBB → rgba(r,g,b,a)  plotly fillcolor 호환용."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── 페이지 설정 ────────────────────────────────────────
st.set_page_config(
    page_title="비염 케어 AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API URL 결정 ───────────────────────────────────────
def _get_api_url() -> str:
    if url := os.getenv("STREAMLIT_API_URL"):
        return url
    try:
        from src.utils.config import load_config
        return load_config().get("streamlit", {}).get("api_base_url", "")
    except Exception:
        return ""

API_URL = _get_api_url()

# ── 측정 지역 ────────────────────────────────────────
STATIONS = {
    "서울 종로구":   ("종로구",  60, 127),
    "서울 강남구":   ("강남구",  61, 126),
    "서울 마포구":   ("마포구",  59, 127),
    "서울 노원구":   ("노원구",  61, 129),
    "서울 관악구":   ("관악구",  59, 125),
    "인천":          ("인천",    55, 124),
    "수원 (인계동)": ("인계동",  60, 121),
    "파주":          ("파주",    56, 131),
    "대전":          ("대전",    67, 100),
    "대구":          ("대구",    89,  90),
    "부산":          ("부산",    98,  76),
}

# ── 클러스터 메타 (증상 기반 3유형) ───────────────────
CLUSTER_STATS = {
    "콧물·재채기 우세형": {
        "emoji": "💧", "pct": 40.0, "color": "#4A90D9",
        "desc_short": "콧물·재채기 중심 알레르기 비염",
        "symptom_profile": {"콧물": 8, "코막힘": 4, "재채기": 8, "눈 증상": 5},
    },
    "코막힘 우세형": {
        "emoji": "👃", "pct": 33.0, "color": "#E8784A",
        "desc_short": "코막힘 중심 만성 비염",
        "symptom_profile": {"콧물": 3, "코막힘": 9, "재채기": 3, "눈 증상": 3},
    },
    "복합 과민형": {
        "emoji": "👁", "pct": 27.0, "color": "#7B68EE",
        "desc_short": "눈·코 복합 과민 반응형",
        "symptom_profile": {"콧물": 6, "코막힘": 6, "재채기": 6, "눈 증상": 8},
    },
}

# ── 스타일 ────────────────────────────────────────────
st.markdown("""
<style>
.result-card {
    padding: 1.2rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;
}
.guide-item {
    display: flex; align-items: flex-start; gap: 12px;
    background: #f8f9fa; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px;
}
.guide-num {
    min-width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 0.8rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ── 예측 로직 ─────────────────────────────────────────
@st.cache_resource
def _get_predictor():
    """직접 모드용 predictor 싱글톤."""
    from src.api.predictor import predictor
    return predictor


def _check_api() -> bool:
    try:
        import requests
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def run_predict(payload: dict) -> dict:
    """API 모드 → 직접 모드 순서로 시도."""
    if API_URL and _check_api():
        import requests
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["result"]

    predictor = _get_predictor()
    if not predictor.is_loaded:
        st.error(
            "모델 파일을 불러올 수 없습니다.  \n"
            "`outputs/models/kmeans_rhinitis.pkl` 경로를 확인하세요."
        )
        st.stop()
    return predictor.predict(payload)


def _get_guide_info(label: str) -> dict | None:
    if API_URL:
        try:
            import requests
            r = requests.get(f"{API_URL}/guide/{label}", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    try:
        from src.api.predictor import CLUSTER_INFO
        info = CLUSTER_INFO.get(label)
        if info:
            return {"cluster_label": label, **info}
    except Exception:
        pass
    return None


# ── VAS 척도 설명 ────────────────────────────────────
_VAS = {
    "rhinorrhea": [
        (0,  0,  "콧물이 전혀 없음"),
        (1,  2,  "코 안이 약간 촉촉한 느낌, 콧물이 거의 없음"),
        (3,  4,  "콧물이 가끔 나와 휴지로 닦는 정도"),
        (5,  6,  "콧물이 자주 흘러 휴지를 자주 사용함"),
        (7,  8,  "콧물이 계속 흘러 업무·일상이 방해됨"),
        (9,  10, "콧물이 멈추지 않아 휴지 없이 생활 불가"),
    ],
    "congestion": [
        (0,  0,  "코막힘 전혀 없음, 코로 편하게 숨쉬기 가능"),
        (1,  2,  "코가 약간 답답한 느낌이지만 숨쉬기는 괜찮음"),
        (3,  4,  "코가 약간 답답하지만 입으로 숨 쉴 정도는 아님"),
        (5,  6,  "코막힘이 심해 가끔 입으로 숨을 쉬게 됨"),
        (7,  8,  "한쪽 코가 완전히 막혀서 입으로 숨을 쉬어야 함"),
        (9,  10, "양쪽 코가 모두 막혀 수면·집중이 거의 불가능"),
    ],
    "sneezing": [
        (0,  0,  "재채기·가려움 전혀 없음"),
        (1,  2,  "하루 1~2번 가볍게 재채기하는 정도"),
        (3,  4,  "하루 수 차례 재채기, 코·입천장이 가끔 간지러움"),
        (5,  6,  "재채기가 연속으로 나오고 코·목이 자주 가려움"),
        (7,  8,  "발작성 재채기가 멈추지 않고 가려움이 매우 심함"),
        (9,  10, "재채기·가려움으로 일상·수면이 불가능한 수준"),
    ],
    "ocular": [
        (0,  0,  "눈 증상 전혀 없음"),
        (1,  2,  "눈이 가끔 약간 뻑뻑하거나 피로한 느낌"),
        (3,  4,  "눈이 가끔 가렵거나 약간 충혈됨"),
        (5,  6,  "눈이 충혈되고 가려워서 자꾸 손이 감"),
        (7,  8,  "눈이 매우 가렵고 충혈·부종이 심해 일상이 불편"),
        (9,  10, "눈을 뜨기 힘들 정도로 가렵고 충혈·눈물이 심함"),
    ],
}

def _vas_label(symptom: str, score: int) -> str:
    """현재 점수에 해당하는 VAS 설명 반환."""
    for lo, hi, text in _VAS[symptom]:
        if lo <= score <= hi:
            return text
    return ""


# ── PM 등급 ───────────────────────────────────────────
def _pm10_grade(val: float) -> tuple[str, str]:
    if val <= 30:  return "좋음",    "#4CAF50"
    if val <= 80:  return "보통",    "#FFC107"
    if val <= 150: return "나쁨",    "#FF5722"
    return "매우나쁨", "#B71C1C"

def _pm25_grade(val: float) -> tuple[str, str]:
    if val <= 15:  return "좋음",    "#4CAF50"
    if val <= 35:  return "보통",    "#FFC107"
    if val <= 75:  return "나쁨",    "#FF5722"
    return "매우나쁨", "#B71C1C"


has_air_key     = bool(os.getenv("AIRKOREA_API_KEY"))
has_weather_key = bool(os.getenv("KMA_API_KEY"))


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_air(station: str):
    from src.data.api_collector import fetch_airkorea
    return fetch_airkorea(station_name=station)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_weather(nx: int, ny: int):
    from src.data.api_collector import fetch_kma_forecast
    return fetch_kma_forecast(nx=nx, ny=ny)


def _get_current_env(station_name: str, nx: int, ny: int) -> dict:
    """현재 시각 기준 대기·날씨 데이터를 반환. 실패 시 빈 dict."""
    result = {}
    if has_air_key:
        try:
            air_df  = _fetch_air(station_name)
            latest  = air_df.dropna(subset=["pm10"]).iloc[-1]
            result["pm10"]  = latest.get("pm10")  or 0
            result["pm25"]  = latest.get("pm25")  or 0
        except Exception:
            pass
    if has_weather_key:
        try:
            import math as _math
            from datetime import timezone, timedelta as _td
            wx_df  = _fetch_weather(nx, ny)
            tmp_df = wx_df.dropna(subset=["temperature"]).copy()
            tmp_df["_dt"] = pd.to_datetime(
                tmp_df["date"].astype(str) + tmp_df["time"].astype(str).str.zfill(4),
                format="%Y%m%d%H%M", errors="coerce",
            )
            tmp_df  = tmp_df.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
            _KST    = timezone(_td(hours=9))
            now_ts  = pd.Timestamp.now(tz=_KST).replace(tzinfo=None)
            future  = tmp_df[tmp_df["_dt"] >= now_ts]
            row     = future.iloc[0] if not future.empty else tmp_df.iloc[-1]
            result["temperature"] = row.get("temperature")
            h = row.get("humidity")
            result["humidity"]    = None if (h is None or (isinstance(h, float) and _math.isnan(h))) else h
        except Exception:
            pass
    return result


def _risk_messages(label, sr, sc, ss, so, env: dict) -> list[tuple[str, str]]:
    """증상 + 환경 기반 위험 메시지 목록 반환. (type, text) 리스트."""
    msgs = []
    total = sr + sc + ss + so
    sev   = total / 40 * 100

    pm10  = env.get("pm10")
    pm25  = env.get("pm25")
    humid = env.get("humidity")

    # 전체 심각도
    if sev >= 75:
        msgs.append(("error",   "증상이 매우 심각합니다. 이비인후과 방문을 강력히 권장합니다."))
    elif sev >= 50:
        msgs.append(("warning", "증상이 중등도 이상입니다. 증상 악화 시 의사 상담을 권장합니다."))

    # 대기질
    if pm10 is not None:
        if pm10 > 150:
            msgs.append(("error",   f"미세먼지 매우 나쁨 ({pm10:.0f}㎍/㎥) — 외출을 삼가고 공기청정기를 가동하세요."))
        elif pm10 > 80:
            msgs.append(("warning", f"미세먼지 나쁨 ({pm10:.0f}㎍/㎥) — KF80 이상 마스크 착용을 권장합니다."))
        elif pm10 > 30:
            msgs.append(("info",    f"미세먼지 보통 ({pm10:.0f}㎍/㎥) — 민감군은 외출 시 주의하세요."))

    if pm25 is not None and pm25 > 35:
        msgs.append(("warning", f"초미세먼지 나쁨 ({pm25:.0f}㎍/㎥) — 코·목 점막에 직접 영향을 줄 수 있습니다."))

    # 습도
    if humid is not None:
        if humid >= 70:
            msgs.append(("warning", f"습도 높음 ({humid:.0f}%) — 실내 환기와 제습을 권장합니다."))
        elif humid <= 30:
            msgs.append(("warning", f"습도 낮음 ({humid:.0f}%) — 코 점막 건조에 주의하고 가습기를 사용하세요."))

    # 유형별 맞춤
    if label == "콧물·재채기 우세형":
        if pm10 and pm10 > 30:
            msgs.append(("info", "콧물·재채기 우세형은 항원 노출에 민감합니다. 외출 전 예방약 복용을 의사와 상의하세요."))
        if sr >= 7 or ss >= 7:
            msgs.append(("warning", "콧물·재채기가 심합니다. 꽃가루 농도가 높은 오전 6~10시 외출을 피하세요."))
    elif label == "코막힘 우세형":
        if humid and humid < 40:
            msgs.append(("info", "건조한 날씨는 코막힘을 악화시킬 수 있습니다. 실내 가습을 권장합니다."))
        if sc >= 7:
            msgs.append(("warning", "코막힘이 심합니다. 취침 시 베개를 높이고 비강 스프레이를 꾸준히 사용하세요."))
    elif label == "복합 과민형":
        if pm25 and pm25 > 35:
            msgs.append(("error", "초미세먼지가 나쁩니다. 복합 과민형 비염에 특히 위험하니 KF94 마스크를 착용하세요."))
        if so >= 7:
            msgs.append(("warning", "눈 증상이 심합니다. 콘택트렌즈를 자제하고 항히스타민 안약을 사용하세요."))

    return msgs


# ── 사이드바 ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 비염 케어 AI")
    st.divider()

    st.markdown("### 📋 현재 증상 (0~10점)")
    s_rhinorrhea = st.slider("💧 콧물",        0, 10, 5)
    st.caption(f"**{s_rhinorrhea}점** — {_vas_label('rhinorrhea', s_rhinorrhea)}")
    s_congestion = st.slider("👃 코막힘",       0, 10, 3)
    st.caption(f"**{s_congestion}점** — {_vas_label('congestion', s_congestion)}")
    s_sneezing   = st.slider("😤 재채기·가려움", 0, 10, 5)
    st.caption(f"**{s_sneezing}점** — {_vas_label('sneezing', s_sneezing)}")
    s_ocular     = st.slider("👁 눈 증상",      0, 10, 2)
    st.caption(f"**{s_ocular}점** — {_vas_label('ocular', s_ocular)}")

    st.divider()
    st.markdown("### 🏥 동반 질환 (선택)")

    has_asthma       = st.selectbox("천식 보유 여부",    ["없음", "있음"])
    has_atopic_derm  = st.selectbox("아토피 보유 여부",   ["없음", "있음"])
    has_food_allergy = st.selectbox("식품 알레르기 여부", ["없음", "있음"])

    with st.expander("상세 정보 입력"):
        food_count   = st.slider(
            "식품알레르기 종류 수", 0, 10, 0,
            disabled=(has_food_allergy == "없음"),
        )
        onset_age    = st.slider("비염 발병 나이 (세)", 0.0, 20.0, 7.0, 0.5)
        duration     = st.slider("비염 지속 기간 (년)", 0.0, 20.0, 2.0, 0.5)
        atopic_march = st.selectbox("아토픽 마치 여부", ["없음", "있음"])
        st.caption(
            "💡 **아토픽 마치**: 식품알레르기 → 아토피 → 비염 → 천식 순서로 "
            "진행되는 알레르기 자연 경과."
        )

    st.divider()
    st.markdown("### 📍 측정 지역")
    selected_station = st.selectbox("지역 선택", list(STATIONS.keys()), label_visibility="collapsed")
    station_name, nx, ny = STATIONS[selected_station]

    st.divider()
    predict_btn = st.button(
        "🔍 비염 유형 분석하기",
        use_container_width=True,
        type="primary",
    )


# ── 메인 헤더 ─────────────────────────────────────────
st.markdown("# 🌿 비염 케어 AI")
st.caption("현재 증상 점수를 입력하면 비염 유형을 분류하고 맞춤형 관리 가이드를 제공합니다.")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 유형 분석", "📊 유형 현황", "ℹ️ 유형 안내", "🌍 환경 정보"])


# ────────────────────────────────────────────────────
# Tab 1: 유형 분석
# ────────────────────────────────────────────────────
with tab1:
    if predict_btn:
        payload = {
            # 증상 점수 (모델 피처)
            "symptom_rhinorrhea": s_rhinorrhea,
            "symptom_congestion": s_congestion,
            "symptom_sneezing":   s_sneezing,
            "symptom_ocular":     s_ocular,
            # 동반 질환 (참고 정보 — 모델 피처 아님)
            "has_asthma":         1 if has_asthma == "있음" else 0,
            "has_atopic_derm":    1 if has_atopic_derm == "있음" else 0,
            "has_food_allergy":   1 if has_food_allergy == "있음" else 0,
            "food_allergy_count": food_count if has_food_allergy == "있음" else 0,
            "rhinitis_onset_age": onset_age,
            "rhinitis_duration":  duration,
            "atopic_march":       1 if atopic_march == "있음" else 0,
        }

        with st.spinner("증상 패턴 분석 중..."):
            result = run_predict(payload)
            
            # 환경 정보 가져오기 (사전 정의된 함수 활용)
            env_now = _get_current_env(station_name, nx, ny)
            
            # 기록 저장
            record = {
                "cluster_label": result["cluster_label"],
                "symptom_rhinorrhea": s_rhinorrhea,
                "symptom_congestion": s_congestion,
                "symptom_sneezing": s_sneezing,
                "symptom_ocular": s_ocular,
                "pm10": env_now.get("pm10", 0),
                "pm25": env_now.get("pm25", 0),
                "humidity": env_now.get("humidity", 50),
                "temperature": env_now.get("temperature", 20)
            }
            save_history(record)

        label = result["cluster_label"]
        conf  = result["confidence"] * 100
        stat  = CLUSTER_STATS.get(label, {})
        color = stat.get("color", "#888888")
        emoji = stat.get("emoji", "")

        # ── 결과 카드 (유형명 + 설명) ────────────────────────
        st.markdown(
            f"""<div class="result-card" style="background:{color}18;border-left:5px solid {color};">
                <span style="font-size:1.6rem;font-weight:700;color:{color};">{emoji} {label}</span>
                <p style="margin:0.6rem 0 0;color:#444;">{result['description']}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Row 1: 유형 변화 추이 + 레이더 나란히 ─
        top_l, top_r = st.columns([1, 1])

        with top_l:
            recent_df = get_recent_history(days=7)
            
            # 데이터가 부족하면 테스트용 데이터 생성 (최초 실행 시)
            if len(recent_df) < 3:
                generate_synthetic_history()
                recent_df = get_recent_history(days=7)

            if not recent_df.empty:
                # 유형 변화 추이 라인 차트
                recent_df["유형_id"] = recent_df["cluster_label"].map({
                    "콧물·재채기 우세형": 1,
                    "코막힘 우세형": 2,
                    "복합 과민형": 3
                })
                
                fig_trend = px.line(
                    recent_df, x="timestamp", y="유형_id",
                    markers=True,
                    title="최근 7일 유형 변화 추이",
                    labels={"유형_id": "비염 유형", "timestamp": "날짜"}
                )
                
                fig_trend.update_layout(
                    yaxis=dict(
                        tickmode='array',
                        tickvals=[1, 2, 3],
                        ticktext=["콧물·재채기", "코막힘", "복합 과민"],
                        range=[0.5, 3.5]
                    ),
                    height=260,
                    margin=dict(t=50, b=5, l=5, r=5),
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("데이터가 충분하지 않아 추이를 표시할 수 없습니다.")

        with top_r:
            cats   = ["콧물", "코막힘", "재채기", "눈 증상"]
            values = [s_rhinorrhea * 10, s_congestion * 10, s_sneezing * 10, s_ocular * 10]
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor=_hex_rgba(color, 0.27),
                line=dict(color=color, width=2),
                name="내 증상",
            ))
            profile = stat.get("symptom_profile", {})
            if profile:
                pv = [profile.get(c, 0) * 10 for c in cats]
                fig_radar.add_trace(go.Scatterpolar(
                    r=pv + [pv[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    fillcolor=_hex_rgba("#888888", 0.10),
                    line=dict(color="#888888", width=1, dash="dot"),
                    name="유형 평균",
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True, height=280,
                margin=dict(t=20, b=20, l=40, r=40),
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Row Ex: 최근 7일 증상 점수 변화 ──────────────────
        st.markdown("#### 📈 최근 7일 증상 점수 추이")
        if not recent_df.empty:
            # 점수 데이터 리쉐이핑 (Plotly용)
            score_df = recent_df.melt(
                id_vars=["timestamp"],
                value_vars=["symptom_rhinorrhea", "symptom_congestion", "symptom_sneezing", "symptom_ocular"],
                var_name="증상", value_name="점수"
            )
            score_df["증상"] = score_df["증상"].map({
                "symptom_rhinorrhea": "콧물",
                "symptom_congestion": "코막힘",
                "symptom_sneezing": "재채기",
                "symptom_ocular": "눈 증상"
            })
            
            fig_scores = px.line(
                score_df, x="timestamp", y="점수", color="증상",
                markers=True,
                color_discrete_map={"콧물": "#4A90D9", "코막힘": "#E8784A", "재채기": "#2ECC71", "눈 증상": "#7B68EE"}
            )
            fig_scores.update_layout(height=300, margin=dict(t=10, b=10), yaxis_range=[0, 10])
            st.plotly_chart(fig_scores, use_container_width=True)

            # 💡 인사이트 제공
            if len(recent_df) >= 2:
                prev_record = recent_df.iloc[-2]
                curr_record = recent_df.iloc[-1]
                
                prev_label = prev_record["cluster_label"]
                curr_label = curr_record["cluster_label"]
                
                insight_parts = []
                if prev_label != curr_label:
                    # 유형이 변한 경우
                    insight_msg = f"지난 기록에서는 **{prev_label}**이었는데, 현재는 **{curr_label}**로 변화되었습니다."
                    
                    # 환경 요인과의 상관관계 분석
                    if curr_record["pm10"] > prev_record["pm10"] + 20:
                        insight_msg += f" 최근 미세먼지(PM10) 농도가 상승하면서 증상 패턴이 달라진 것으로 보입니다."
                    elif curr_record["humidity"] < prev_record["humidity"] - 10:
                        insight_msg += f" 대기가 건조해지면서 비강 점막에 변화가 생겼을 가능성이 큽니다."
                    
                    st.info(f"💡 **변화 인사이트**: \"{insight_msg}\" (환경과의 상관관계를 체감하실 수 있습니다.)")
                else:
                    st.success(f"💡 **상태 유지**: 비염 유형이 **{curr_label}**로 안정적으로 유지되고 있습니다. 현재의 관리 방식을 지속하세요.")

        # ── Row 2: 오늘의 위험수준 ───────────────────────
        st.divider()
        st.markdown("#### 🚨 오늘의 위험수준")

        total_score = s_rhinorrhea + s_congestion + s_sneezing + s_ocular
        sev_pct     = total_score / 40 * 100
        if sev_pct < 25:   sev_label, sev_color = "경미",   "#4CAF50"
        elif sev_pct < 50: sev_label, sev_color = "보통",   "#FFC107"
        elif sev_pct < 75: sev_label, sev_color = "주의",   "#FF5722"
        else:               sev_label, sev_color = "심각",   "#B71C1C"

        env = _get_current_env(station_name, nx, ny)

        dash_l, dash_r = st.columns([1, 1])

        with dash_l:
            sym_items = [
                ("콧물",    s_rhinorrhea, "#4A90D9"),
                ("코막힘",  s_congestion, "#E8784A"),
                ("재채기",  s_sneezing,   "#2ECC71"),
                ("눈 증상", s_ocular,     "#7B68EE"),
            ]
            nz = [(l, v, c) for l, v, c in sym_items if v > 0]
            if nz:
                fig_risk = go.Figure(go.Pie(
                    labels=[x[0] for x in nz],
                    values=[x[1] for x in nz],
                    marker_colors=[x[2] for x in nz],
                    textinfo="percent+label",
                    hole=0.48,
                ))
            else:
                fig_risk = go.Figure(go.Pie(
                    labels=["증상 없음"], values=[1],
                    marker_colors=["#CCCCCC"], textinfo="label", hole=0.48,
                ))
            fig_risk.update_layout(
                height=260,
                margin=dict(t=30, b=5, l=5, r=5),
                title=dict(text="오늘의 증상 분포", font_size=13),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{sev_label}</b><br>{sev_pct:.0f}%",
                    x=0.5, y=0.5, font_size=13, showarrow=False,
                )],
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with dash_r:
            st.markdown(
                f"""<div style="background:{sev_color}22;border-left:4px solid {sev_color};
                        padding:10px 14px;border-radius:6px;margin-bottom:10px;">
                    <b style="color:{sev_color};">종합 위험도: {sev_label}</b>
                    <div style="margin-top:4px;color:#555;font-size:0.86rem;">
                        증상 합계 {total_score}/40점 ({sev_pct:.0f}%)
                    </div></div>""",
                unsafe_allow_html=True,
            )
            pm10  = env.get("pm10")
            pm25  = env.get("pm25")
            humid = env.get("humidity")
            if pm10 is not None or humid is not None:
                env_cols = st.columns(3)
                if pm10  is not None: env_cols[0].metric("PM10",  f"{pm10:.0f}㎍")
                if pm25  is not None: env_cols[1].metric("PM2.5", f"{pm25:.0f}㎍")
                if humid is not None: env_cols[2].metric("습도",   f"{humid:.0f}%")
            msgs = _risk_messages(label, s_rhinorrhea, s_congestion, s_sneezing, s_ocular, env)
            if msgs:
                for mtype, mtext in msgs:
                    if mtype == "error":     st.error(mtext)
                    elif mtype == "warning": st.warning(mtext)
                    else:                    st.info(mtext)
            else:
                st.success("오늘 증상이 경미하고 환경 조건도 양호합니다. 현재 관리 방법을 유지하세요.")

        # ── Row 3: 맞춤 관리 가이드 ──────────────────────
        st.divider()
        st.markdown("#### 📋 맞춤 관리 가이드")
        for i, tip in enumerate(result["guide"], 1):
            st.markdown(
                f"""<div class="guide-item">
                    <div class="guide-num" style="background:{color};">{i}</div>
                    <span style="padding-top:3px;">{tip}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        with st.expander("입력 정보 요약"):
            st.json(payload)

    else:
        st.info("👈 왼쪽 사이드바에서 현재 증상 점수를 입력하고 **'비염 유형 분석하기'** 버튼을 눌러주세요.")

        st.markdown("#### 💡 유형별 증상 패턴 예시")
        ex_cols = st.columns(3)
        examples = [
            ("💧 콧물·재채기 우세형",
             "콧물: 8/10\n\n재채기·가려움: 8/10\n\n코막힘: 4/10\n\n눈 증상: 5/10"),
            ("👃 코막힘 우세형",
             "코막힘: 9/10\n\n콧물: 3/10\n\n재채기: 3/10\n\n눈 증상: 3/10"),
            ("👁 복합 과민형",
             "눈 증상: 8/10\n\n콧물: 6/10\n\n코막힘: 6/10\n\n재채기: 6/10"),
        ]
        for col, (title, desc) in zip(ex_cols, examples):
            with col:
                st.markdown(f"**{title}**")
                st.markdown(desc)


# ────────────────────────────────────────────────────
# Tab 2: 유형 현황
# ────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 비염 유형 분포")

    labels = list(CLUSTER_STATS.keys())
    pcts   = [v["pct"] for v in CLUSTER_STATS.values()]
    colors = [v["color"] for v in CLUSTER_STATS.values()]

    col_a, col_b = st.columns([1, 1])

    with col_a:
        fig_pie = px.pie(
            names=labels, values=pcts,
            color_discrete_sequence=colors,
            title="유형별 비율",
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### 유형별 대표 증상 프로파일")
        profile_data = []
        for lbl, stat in CLUSTER_STATS.items():
            row = {"유형": f"{stat['emoji']} {lbl}"}
            row.update({k: f"{v}/10" for k, v in stat["symptom_profile"].items()})
            profile_data.append(row)
        st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

    st.markdown("### 🔬 유형별 증상 강도 비교")
    compare_rows = []
    symptom_names = ["콧물", "코막힘", "재채기", "눈 증상"]
    for lbl, stat in CLUSTER_STATS.items():
        for sym in symptom_names:
            compare_rows.append({
                "유형":    f"{stat['emoji']} {lbl}",
                "증상":    sym,
                "점수":    stat["symptom_profile"][sym],
            })
    compare_df = pd.DataFrame(compare_rows)
    fig_bar = px.bar(
        compare_df, x="증상", y="점수", color="유형",
        barmode="group", color_discrete_sequence=colors,
        title="유형별 증상 강도 (0~10)",
    )
    fig_bar.update_layout(height=360, margin=dict(t=50, b=10), yaxis_range=[0, 10])
    st.plotly_chart(fig_bar, use_container_width=True)


# ────────────────────────────────────────────────────
# Tab 3: 유형 안내
# ────────────────────────────────────────────────────
with tab3:
    st.markdown("### ℹ️ 비염 3대 유형 안내")

    for label, stat in CLUSTER_STATS.items():
        with st.expander(f"{stat['emoji']}  {label}  ({stat['pct']}%)", expanded=False):
            info = _get_guide_info(label)

            c1, c2 = st.columns([3, 1])
            with c1:
                if info:
                    st.markdown(f"**설명:** {info['description']}")
                    st.markdown("**관리 가이드:**")
                    for g in info["guide"]:
                        st.markdown(f"- {g}")
                else:
                    st.caption("가이드 정보를 불러올 수 없습니다.")

            with c2:
                for sym, val in stat["symptom_profile"].items():
                    st.metric(sym, f"{val}/10")


# ────────────────────────────────────────────────────
# Tab 4: 환경 정보 (에어코리아 + 기상청)
# ────────────────────────────────────────────────────
with tab4:
    if not has_air_key and not has_weather_key:
        st.warning(
            "API 키가 설정되지 않았습니다.  \n"
            "Streamlit Cloud → 앱 설정 → Secrets 에서 "
            "`AIRKOREA_API_KEY` 와 `KMA_API_KEY` 를 입력해주세요."
        )
    else:
        st.caption(f"📍 측정 지역: **{selected_station}** (사이드바에서 변경)")

        # ── 에어코리아 ─────────────────────────────────
        st.markdown("### 💨 대기 환경")

        if not has_air_key:
            st.caption("에어코리아 API 키 미설정 — 대기 정보를 불러올 수 없습니다.")
        else:
            with st.spinner("대기 정보 불러오는 중..."):
                try:
                    air_df = _fetch_air(station_name)
                    latest = air_df.dropna(subset=["pm10"]).iloc[-1]

                    pm10_val = latest.get("pm10", 0) or 0
                    pm25_val = latest.get("pm25", 0) or 0
                    o3_val   = latest.get("o3",   0) or 0
                    pm10_grade, _ = _pm10_grade(pm10_val)
                    pm25_grade, _ = _pm25_grade(pm25_val)

                    if pm10_val > 80 or pm25_val > 35:
                        st.error("⚠️ 현재 대기 상태가 **나쁨** 수준입니다. 외출 시 KF80 이상 마스크를 착용하세요.")
                    elif pm10_val > 30 or pm25_val > 15:
                        st.warning("현재 대기 상태가 **보통** 수준입니다. 민감군은 주의하세요.")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("PM10 (미세먼지)",    f"{pm10_val:.0f} ㎍/㎥", delta=pm10_grade, delta_color="off")
                    c2.metric("PM2.5 (초미세먼지)", f"{pm25_val:.0f} ㎍/㎥", delta=pm25_grade, delta_color="off")
                    c3.metric("O3 (오존)",          f"{o3_val:.3f} ppm")

                    chart_df = air_df.dropna(subset=["pm10"]).tail(24).copy()
                    if not chart_df.empty:
                        fig_air = px.line(
                            chart_df, x="datetime", y="pm10",
                            title=f"{station_name} PM10 최근 24시간 추이",
                            labels={"datetime": "시각", "pm10": "PM10 (㎍/㎥)"},
                        )
                        fig_air.add_hline(y=30,  line_dash="dot", line_color="#4CAF50", annotation_text="좋음(30)")
                        fig_air.add_hline(y=80,  line_dash="dot", line_color="#FFC107", annotation_text="나쁨(80)")
                        fig_air.add_hline(y=150, line_dash="dot", line_color="#FF5722", annotation_text="매우나쁨(150)")
                        fig_air.update_layout(height=300, margin=dict(t=50, b=10))
                        st.plotly_chart(fig_air, use_container_width=True)

                    st.caption(f"마지막 업데이트: {latest.get('datetime', '-')}  |  캐시 TTL: 60분")

                except Exception as e:
                    st.error(f"대기 정보를 불러오지 못했습니다: {e}")

        st.divider()

        # ── 기상청 ─────────────────────────────────────
        st.markdown("### 🌤 날씨 예보")

        if not has_weather_key:
            st.caption("기상청 API 키 미설정 — 날씨 정보를 불러올 수 없습니다.")
        else:
            with st.spinner("날씨 정보 불러오는 중..."):
                try:
                    import math
                    wx_df = _fetch_weather(nx, ny)
                    # 시간순 정렬 후 현재 시각과 가장 가까운 예보 행 선택
                    tmp_df = wx_df.dropna(subset=["temperature"]).copy()
                    tmp_df["_dt"] = pd.to_datetime(
                        tmp_df["date"].astype(str) + tmp_df["time"].astype(str).str.zfill(4),
                        format="%Y%m%d%H%M", errors="coerce",
                    )
                    tmp_df = tmp_df.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
                    from datetime import timezone, timedelta as _td
                    _KST = timezone(_td(hours=9))
                    now_ts = pd.Timestamp.now(tz=_KST).replace(tzinfo=None)
                    future = tmp_df[tmp_df["_dt"] >= now_ts]
                    latest_wx = (future.iloc[0] if not future.empty else tmp_df.iloc[-1])
                    temp   = latest_wx.get("temperature",  "-")
                    humid  = latest_wx.get("humidity",     "-")
                    precip = latest_wx.get("precipitation", "-")

                    try:
                        precip_f = float(precip)
                        if math.isnan(precip_f) or precip_f == 0:
                            precip_label = "없음"
                        else:
                            precip_label = f"{precip_f:.1f} mm"
                    except (TypeError, ValueError):
                        precip_label = "없음"

                    w1, w2, w3 = st.columns(3)
                    w1.metric("🌡 기온",   f"{temp} °C")
                    w2.metric("💧 습도",   f"{humid} %")
                    w3.metric("🌧 강수량", precip_label)

                    try:
                        if float(humid) >= 70:
                            st.warning("습도가 높습니다. 실내 습도 조절과 환기를 권장합니다.")
                        elif float(humid) <= 30:
                            st.warning("습도가 낮습니다. 코 점막 건조에 주의하세요.")
                    except (ValueError, TypeError):
                        pass

                    if not tmp_df.empty:
                        plot_df = tmp_df.head(24)
                        fig_wx = px.line(
                            plot_df, x="_dt", y="temperature",
                            title=f"{selected} 기온 예보",
                            labels={"temperature": "기온 (°C)", "_dt": "시각"},
                        )
                        fig_wx.update_layout(height=300, margin=dict(t=50, b=10))
                        st.plotly_chart(fig_wx, use_container_width=True)

                except Exception as e:
                    st.error(f"날씨 정보를 불러오지 못했습니다: {e}")
