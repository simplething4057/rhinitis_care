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

    # 서버 상태 표시
    api_ok = _check_api() if API_URL else False
    if API_URL:
        if api_ok:
            st.success("API 서버 연결됨 ✅")
        else:
            st.warning("API 미연결 — 직접 모드로 동작합니다.")
    else:
        st.info("직접 모드 (API 서버 없음)")

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

tab1, tab2, tab3 = st.tabs(["🔍 유형 분석", "📊 클러스터 현황", "ℹ️ 유형 안내"])


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
