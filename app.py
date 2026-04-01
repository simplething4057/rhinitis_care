"""
비염 케어 AI — Streamlit 대시보드

동작 모드 (자동 선택):
  - API 모드  : STREAMLIT_API_URL 또는 config의 api_base_url로 FastAPI 호출
  - 직접 모드 : API 미연결 시 predictor를 직접 임포트해 추론 (Streamlit Cloud 무료 배포용)
"""
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud secrets → os.environ 브릿지
# .env는 로컬, st.secrets는 Streamlit Cloud에서 API 키를 공급
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

# ── 클러스터 메타 (차트/안내 탭용 정적 데이터) ─────────
CLUSTER_STATS = {
    "호흡기 알레르기형": {"emoji": "🌿", "size": 26122, "pct": 47.0,
                         "onset": 7.9, "asthma": 0,    "atopy": 18.6, "food": 0,
                         "color": "#4A90D9"},
    "비염+천식 복합형":  {"emoji": "💨", "size": 22342, "pct": 40.2,
                         "onset": 7.0, "asthma": 100,  "atopy": 26.6, "food": 0,
                         "color": "#E8784A"},
    "아토픽 마치형":     {"emoji": "🔶", "size":  7103, "pct": 12.8,
                         "onset": 5.9, "asthma": 66.9, "atopy": 44.0, "food": 100,
                         "color": "#6ABF69"},
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
    """직접 모드용 predictor 싱글톤 (앱 시작 시 한 번만 로드)."""
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

    # 직접 모드
    predictor = _get_predictor()
    if not predictor.is_loaded:
        st.error(
            "모델 파일을 불러올 수 없습니다.  \n"
            "`outputs/models/kmeans_childhood.pkl` 경로를 확인하세요."
        )
        st.stop()
    return predictor.predict(payload)


# ── PM10 등급 판정 ────────────────────────────────────
def _pm10_grade(val: float) -> tuple[str, str]:
    """(등급명, 색상) 반환."""
    if val <= 30:   return "좋음",   "#4CAF50"
    if val <= 80:   return "보통",   "#FFC107"
    if val <= 150:  return "나쁨",   "#FF5722"
    return "매우나쁨", "#B71C1C"

def _pm25_grade(val: float) -> tuple[str, str]:
    if val <= 15:   return "좋음",   "#4CAF50"
    if val <= 35:   return "보통",   "#FFC107"
    if val <= 75:   return "나쁨",   "#FF5722"
    return "매우나쁨", "#B71C1C"


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_air(station: str):
    from src.data.api_collector import fetch_airkorea
    return fetch_airkorea(station_name=station)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_weather(nx: int, ny: int):
    from src.data.api_collector import fetch_kma_forecast
    return fetch_kma_forecast(nx=nx, ny=ny)


def _get_guide_info(label: str) -> dict | None:
    """유형 안내 탭용 가이드. API → predictor 순서로 조회."""
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


# ── 사이드바 ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 비염 케어 AI")
    st.divider()

    st.markdown("### 📋 정보 입력")

    has_asthma       = st.selectbox("천식 보유 여부",     ["없음", "있음"])
    has_atopic_derm  = st.selectbox("아토피 보유 여부",    ["없음", "있음"])
    has_food_allergy = st.selectbox("식품 알레르기 여부",  ["없음", "있음"])
    food_count       = st.slider(
        "식품알레르기 종류 수", 0, 10, 0,
        disabled=(has_food_allergy == "없음"),
    )
    onset_age        = st.slider("비염 발병 나이 (세)",    0.0, 20.0, 7.0, 0.5)
    duration         = st.slider("비염 지속 기간 (년)",    0.0, 20.0, 2.0, 0.5)
    atopic_march     = st.selectbox(
        "아토픽 마치 여부",
        ["없음", "있음"],
    )
    st.caption(
        "💡 **아토픽 마치(Atopic March)**: 영아기 식품알레르기에서 시작해 "
        "아토피 피부염 → 알레르기 비염 → 천식 순서로 진행되는 알레르기 질환의 "
        "자연 경과를 말합니다."
    )

    st.divider()
    predict_btn = st.button(
        "🔍 비염 유형 분석하기",
        use_container_width=True,
        type="primary",
    )


# ── 메인 헤더 ─────────────────────────────────────────
st.markdown("# 🌿 비염 케어 AI")
st.caption("동반 질환 정보를 입력하면 비염 유형을 분류하고 맞춤형 관리 가이드를 제공합니다.")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 유형 분석", "📊 클러스터 현황", "ℹ️ 유형 안내", "🌍 환경 정보"])


# ────────────────────────────────────────────────────
# Tab 1: 유형 분석
# ────────────────────────────────────────────────────
with tab1:
    if predict_btn:
        payload = {
            "has_asthma":         1 if has_asthma == "있음" else 0,
            "has_atopic_derm":    1 if has_atopic_derm == "있음" else 0,
            "has_food_allergy":   1 if has_food_allergy == "있음" else 0,
            "food_allergy_count": food_count if has_food_allergy == "있음" else 0,
            "rhinitis_onset_age": onset_age,
            "rhinitis_duration":  duration,
            "atopic_march":       1 if atopic_march == "있음" else 0,
        }

        with st.spinner("분석 중..."):
            result = run_predict(payload)

        label  = result["cluster_label"]
        conf   = result["confidence"] * 100
        stat   = CLUSTER_STATS.get(label, {})
        color  = stat.get("color", "#888888")
        emoji  = stat.get("emoji", "")

        # 결과 카드
        st.markdown(
            f"""
            <div class="result-card" style="
                background:{color}18; border-left:5px solid {color};
            ">
                <span style="font-size:1.6rem;font-weight:700;color:{color};">
                    {emoji} {label}
                </span>
                <span style="margin-left:14px;color:#555;font-size:0.95rem;">
                    신뢰도 {conf:.1f}%
                </span>
                <p style="margin:0.6rem 0 0;color:#444;">{result['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 아토픽 마치형 용어 설명
        if label == "아토픽 마치형":
            st.info(
                "**아토픽 마치(Atopic March)란?**  \n"
                "영아기 식품알레르기에서 시작해 아토피 피부염 → 알레르기 비염 → 천식 순서로 "
                "알레르기 질환이 단계적으로 진행되는 자연 경과를 말합니다. "
                "조기 발견과 다중 알레르기 통합 관리가 중요합니다."
            )

        # 지표 카드 3개
        col1, col2, col3 = st.columns(3)
        col1.metric("예측 신뢰도",    f"{conf:.1f}%")
        col2.metric("전체 비율",      f"{stat.get('pct', '-')}%")
        col3.metric("평균 발병 나이", f"{stat.get('onset', '-')}세")

        left, right = st.columns([1, 1])

        # 신뢰도 게이지
        with left:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                title={"text": "신뢰도 (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  40], "color": "#FFCDD2"},
                        {"range": [40, 70], "color": "#FFF9C4"},
                        {"range": [70,100], "color": "#C8E6C9"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 동반 질환 레이더
        with right:
            cats   = ["천식 동반율", "아토피 동반율", "식품알레르기율"]
            values = [stat.get("asthma", 0), stat.get("atopy", 0), stat.get("food", 0)]
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor=_hex_rgba(color, 0.27),
                line=dict(color=color, width=2),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False, height=240,
                margin=dict(t=20, b=20, l=40, r=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # 맞춤 가이드
        st.markdown("#### 📋 맞춤 관리 가이드")
        for i, tip in enumerate(result["guide"], 1):
            st.markdown(
                f"""
                <div class="guide-item">
                    <div class="guide-num" style="background:{color};">{i}</div>
                    <span style="padding-top:3px;">{tip}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("입력 정보 요약"):
            st.json(payload)

    else:
        st.info("👈 왼쪽 사이드바에서 정보를 입력하고 **'비염 유형 분석하기'** 버튼을 눌러주세요.")

        st.markdown("#### 💡 입력 예시")
        ex_cols = st.columns(3)
        examples = [
            ("🌿 호흡기 알레르기형", "천식: 없음\n\n아토피: 없음\n\n식품알레르기: 없음\n\n발병 나이: 8세"),
            ("💨 비염+천식 복합형",  "천식: 있음\n\n아토피: 없음\n\n식품알레르기: 없음\n\n발병 나이: 7세"),
            ("🔶 아토픽 마치형",     "천식: 있음\n\n아토피: 있음\n\n식품알레르기: 있음 (2종)\n\n발병 나이: 5세"),
        ]
        for col, (title, desc) in zip(ex_cols, examples):
            with col:
                st.markdown(f"**{title}**")
                st.markdown(desc)


# ────────────────────────────────────────────────────
# Tab 2: 클러스터 현황
# ────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 비염 유형 분포 (55,567명)")

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
        stats_df = pd.DataFrame([
            {
                "유형": k,
                "환자 수": f"{v['size']:,}명",
                "비율": f"{v['pct']}%",
                "평균 발병 나이": f"{v['onset']}세",
                "천식 동반율": f"{v['asthma']}%",
                "아토피 동반율": f"{v['atopy']}%",
                "식품알레르기율": f"{v['food']}%",
            }
            for k, v in CLUSTER_STATS.items()
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("### 🔬 유형별 동반 질환 비율 비교")
    compare_df = pd.DataFrame({
        "유형":      labels * 3,
        "동반 질환": ["천식"] * 3 + ["아토피"] * 3 + ["식품알레르기"] * 3,
        "비율 (%)":  [v["asthma"] for v in CLUSTER_STATS.values()]
                   + [v["atopy"]  for v in CLUSTER_STATS.values()]
                   + [v["food"]   for v in CLUSTER_STATS.values()],
    })
    fig_bar = px.bar(
        compare_df, x="동반 질환", y="비율 (%)", color="유형",
        barmode="group", color_discrete_sequence=colors,
        title="유형별 동반 질환 비율",
    )
    fig_bar.update_layout(height=360, margin=dict(t=50, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)


# ────────────────────────────────────────────────────
# Tab 3: 유형 안내
# ────────────────────────────────────────────────────
with tab3:
    st.markdown("### ℹ️ 3대 비염 유형 안내")

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
                st.metric("전체 비율",    f"{stat['pct']}%")
                st.metric("평균 발병 나이", f"{stat['onset']}세")
                st.metric("천식 동반율",   f"{stat['asthma']}%")


# ────────────────────────────────────────────────────
# Tab 4: 환경 정보 (에어코리아 + 기상청)
# ────────────────────────────────────────────────────
STATIONS = {
    "서울 종로구": ("종로구",  60, 127),
    "서울 강남구": ("강남구",  61, 126),
    "서울 마포구": ("마포구",  59, 127),
    "인천":        ("인천",    55, 124),
    "수원":        ("수원",    60, 121),
    "대전":        ("대전",    67, 100),
    "대구":        ("대구",    89,  90),
    "부산":        ("부산",    98,  76),
}

with tab4:
    has_air_key     = bool(os.getenv("AIRKOREA_API_KEY"))
    has_weather_key = bool(os.getenv("KMA_API_KEY"))

    if not has_air_key and not has_weather_key:
        st.warning(
            "API 키가 설정되지 않았습니다.  \n"
            "Streamlit Cloud → 앱 설정 → Secrets 에서 "
            "`AIRKOREA_API_KEY` 와 `KMA_API_KEY` 를 입력해주세요."
        )
    else:
        selected = st.selectbox("📍 측정 지역", list(STATIONS.keys()))
        station_name, nx, ny = STATIONS[selected]

        # ── 에어코리아 ─────────────────────────────────
        st.markdown("### 💨 대기 환경")

        if not has_air_key:
            st.caption("에어코리아 API 키 미설정 — 대기 정보를 불러올 수 없습니다.")
        else:
            with st.spinner("대기 정보 불러오는 중..."):
                try:
                    air_df = _fetch_air(station_name)
                    latest = air_df.dropna(subset=["pm10"]).iloc[-1]

                    pm10_val  = latest.get("pm10",  0) or 0
                    pm25_val  = latest.get("pm25",  0) or 0
                    o3_val    = latest.get("o3",    0) or 0
                    pm10_grade, pm10_color = _pm10_grade(pm10_val)
                    pm25_grade, pm25_color = _pm25_grade(pm25_val)

                    # 비염 주의 알림
                    if pm10_val > 80 or pm25_val > 35:
                        st.error(
                            f"⚠️ 현재 대기 상태가 **나쁨** 수준입니다. "
                            "외출 시 KF80 이상 마스크를 착용하세요."
                        )
                    elif pm10_val > 30 or pm25_val > 15:
                        st.warning("현재 대기 상태가 **보통** 수준입니다. 민감군은 주의하세요.")

                    # 현재값 지표
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "PM10 (미세먼지)",
                        f"{pm10_val:.0f} ㎍/㎥",
                        delta=pm10_grade,
                        delta_color="off",
                    )
                    c2.metric(
                        "PM2.5 (초미세먼지)",
                        f"{pm25_val:.0f} ㎍/㎥",
                        delta=pm25_grade,
                        delta_color="off",
                    )
                    c3.metric("O3 (오존)", f"{o3_val:.3f} ppm")

                    # 24시간 PM10 추이
                    chart_df = air_df.dropna(subset=["pm10"]).tail(24).copy()
                    if not chart_df.empty:
                        fig_air = px.line(
                            chart_df, x="datetime", y="pm10",
                            title=f"{station_name} PM10 최근 24시간 추이",
                            labels={"datetime": "시각", "pm10": "PM10 (㎍/㎥)"},
                        )
                        fig_air.add_hline(y=30,  line_dash="dot", line_color="#4CAF50",
                                          annotation_text="좋음 기준(30)")
                        fig_air.add_hline(y=80,  line_dash="dot", line_color="#FFC107",
                                          annotation_text="나쁨 기준(80)")
                        fig_air.add_hline(y=150, line_dash="dot", line_color="#FF5722",
                                          annotation_text="매우나쁨(150)")
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
                    wx_df = _fetch_weather(nx, ny)

                    # 가장 가까운 예보 시각 값
                    latest_wx = wx_df.dropna(subset=["temperature"]).iloc[0]
                    temp  = latest_wx.get("temperature",  "-")
                    humid = latest_wx.get("humidity",     "-")
                    precip = latest_wx.get("precipitation", "-")

                    w1, w2, w3 = st.columns(3)
                    w1.metric("🌡 기온",   f"{temp} °C")
                    w2.metric("💧 습도",   f"{humid} %")
                    try:
                        precip_f = float(precip)
                        precip_label = "없음" if precip_f == 0 else f"{precip_f:.1f} mm"
                    except (TypeError, ValueError):
                        precip_label = "없음"
                    w3.metric("🌧 강수량", precip_label)

                    # 습도 비염 주의
                    try:
                        if float(humid) >= 70:
                            st.warning("습도가 높습니다. 실내 습도 조절과 환기를 권장합니다.")
                        elif float(humid) <= 30:
                            st.warning("습도가 낮습니다. 코 점막 건조에 주의하세요.")
                    except (ValueError, TypeError):
                        pass

                    # 오늘 기온 시계열
                    today_df = wx_df.dropna(subset=["temperature"]).copy()
                    if not today_df.empty:
                        # "YYYYMMDD" + "HHMM" → datetime → "MM/DD HH시" 레이블
                        today_df["_dt"] = pd.to_datetime(
                            today_df["date"].astype(str) + today_df["time"].astype(str).str.zfill(4),
                            format="%Y%m%d%H%M",
                            errors="coerce",
                        )
                        today_df["시각"] = today_df["_dt"].dt.strftime("%m/%d %H시")
                        plot_df = today_df.dropna(subset=["_dt"]).head(24)
                        fig_wx = px.line(
                            plot_df, x="시각", y="temperature",
                            title=f"{selected} 기온 예보",
                            labels={"temperature": "기온 (°C)", "시각": ""},
                        )
                        fig_wx.update_xaxes(tickangle=-45)
                        fig_wx.update_layout(height=300, margin=dict(t=50, b=60))
                        st.plotly_chart(fig_wx, use_container_width=True)

                except Exception as e:
                    st.error(f"날씨 정보를 불러오지 못했습니다: {e}")
