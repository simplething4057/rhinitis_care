"""
비염 케어 AI — Streamlit 대시보드
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── 페이지 설정 ───────────────────────────────────────
st.set_page_config(
    page_title="비염 케어 AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

# ── 스타일 ────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem; font-weight: 700;
    color: #2E7D32; text-align: center; margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem; color: #666;
    text-align: center; margin-bottom: 2rem;
}
.result-box {
    background: #F1F8E9; border-left: 5px solid #66BB6A;
    padding: 1.2rem 1.5rem; border-radius: 8px; margin: 1rem 0;
}
.guide-item {
    background: #fff; border: 1px solid #ddd;
    border-radius: 6px; padding: 0.6rem 1rem; margin: 0.4rem 0;
}
.badge-green  { background:#E8F5E9; color:#2E7D32; padding:3px 10px; border-radius:12px; font-weight:600; }
.badge-blue   { background:#E3F2FD; color:#1565C0; padding:3px 10px; border-radius:12px; font-weight:600; }
.badge-orange { background:#FFF3E0; color:#E65100; padding:3px 10px; border-radius:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── 유틸 ─────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("model_loaded")
    except:
        return False

def call_predict(payload):
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
    return r.json() if r.status_code == 200 else None

BADGE = {
    "호흡기 알레르기형": "badge-green",
    "비염+천식 복합형":  "badge-blue",
    "아토픽 마치형":     "badge-orange",
}
TYPE_EMOJI = {
    "호흡기 알레르기형": "🌿",
    "비염+천식 복합형":  "💨",
    "아토픽 마치형":     "🔶",
}
CLUSTER_STATS = {
    "호흡기 알레르기형": {"size": 26122, "pct": 47.0, "onset": 7.9, "asthma": 0,   "atopy": 18.6, "food": 0},
    "비염+천식 복합형":  {"size": 22342, "pct": 40.2, "onset": 7.0, "asthma": 100, "atopy": 26.6, "food": 0},
    "아토픽 마치형":     {"size":  7103, "pct": 12.8, "onset": 5.9, "asthma": 66.9,"atopy": 44.0, "food": 100},
}


# ── 사이드바 ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 비염 케어 AI")
    st.markdown("---")

    api_ok = check_api()
    if api_ok:
        st.success("서버 연결됨 ✅")
    else:
        st.error("서버 미연결 ❌\n\n`uvicorn src.api.main:app --reload` 를 먼저 실행하세요.")

    st.markdown("---")
    st.markdown("### 📋 정보 입력")

    has_asthma      = st.selectbox("천식 보유 여부",      ["없음", "있음"])
    has_atopic_derm = st.selectbox("아토피 보유 여부",     ["없음", "있음"])
    has_food_allergy= st.selectbox("식품 알레르기 여부",   ["없음", "있음"])
    food_count      = st.slider("식품알레르기 종류 수",    0, 10, 0)
    onset_age       = st.slider("비염 발병 나이 (세)",     0.0, 20.0, 7.0, 0.5)
    duration        = st.slider("비염 지속 기간 (년)",     0.0, 20.0, 2.0, 0.5)
    atopic_march    = st.selectbox("아토픽 마치 여부\n(식품알레르기→아토피→비염 순서)", ["없음", "있음"])

    st.markdown("---")
    predict_btn = st.button("🔍 비염 유형 분석하기", use_container_width=True,
                            type="primary", disabled=not api_ok)


# ── 메인 화면 ─────────────────────────────────────────
st.markdown('<p class="main-title">🌿 비염 케어 AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">동반 질환 정보를 입력하면 비염 유형을 분류하고 맞춤형 관리 가이드를 제공합니다</p>',
            unsafe_allow_html=True)

# ── 탭 구성 ──────────────────────────────────────────
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
            "food_allergy_count": food_count,
            "rhinitis_onset_age": onset_age,
            "rhinitis_duration":  duration,
            "atopic_march":       1 if atopic_march == "있음" else 0,
        }

        with st.spinner("분석 중..."):
            result = call_predict(payload)

        if result:
            r       = result["result"]
            label   = r["cluster_label"]
            conf    = r["confidence"] * 100
            emoji   = TYPE_EMOJI.get(label, "")
            badge   = BADGE.get(label, "badge-green")
            stats   = CLUSTER_STATS.get(label, {})

            # 결과 헤더
            st.markdown(f"""
            <div class="result-box">
                <h2>{emoji} 분류 결과: <span class="{badge}">{label}</span></h2>
                <p style="color:#555; margin:0.5rem 0 0">{r['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            # 지표 카드
            col1, col2, col3 = st.columns(3)
            col1.metric("신뢰도",       f"{conf:.1f}%")
            col2.metric("전체 비율",    f"{stats.get('pct', '-')}%")
            col3.metric("평균 발병 나이", f"{stats.get('onset', '-')}세")

            # 신뢰도 게이지
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                title={"text": "예측 신뢰도 (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#66BB6A"},
                    "steps": [
                        {"range": [0,  40], "color": "#FFCDD2"},
                        {"range": [40, 70], "color": "#FFF9C4"},
                        {"range": [70,100], "color": "#C8E6C9"},
                    ],
                    "threshold": {"line": {"color":"#2E7D32","width":3}, "value": conf}
                }
            ))
            fig_gauge.update_layout(height=220, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # 맞춤 가이드
            st.markdown("### 📋 맞춤 관리 가이드")
            for i, guide in enumerate(r["guide"], 1):
                st.markdown(f"""
                <div class="guide-item">
                    <b>{'✔' if i<=2 else '💡'} {i}.</b> {guide}
                </div>""", unsafe_allow_html=True)

            # 동반 질환 프로파일 레이더
            st.markdown("### 📈 동반 질환 프로파일")
            cats   = ["천식 동반율", "아토피 동반율", "식품알레르기율"]
            values = [
                stats.get("asthma", 0),
                stats.get("atopy", 0),
                stats.get("food", 0),
            ]
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(102,187,106,0.3)",
                line=dict(color="#2E7D32", width=2),
                name=label,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False, height=300,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        else:
            st.error("예측 실패. API 서버 상태를 확인하세요.")

    else:
        st.info("👈 왼쪽 사이드바에서 정보를 입력하고 **'비염 유형 분석하기'** 버튼을 눌러주세요.")

        # 입력 예시 카드
        st.markdown("### 💡 입력 예시")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            st.markdown("""
            **🌿 호흡기 알레르기형**
            - 천식: 없음
            - 아토피: 없음
            - 식품알레르기: 없음
            - 발병 나이: 8세
            """)
        with ex_col2:
            st.markdown("""
            **💨 비염+천식 복합형**
            - 천식: 있음
            - 아토피: 없음
            - 식품알레르기: 없음
            - 발병 나이: 7세
            """)
        with ex_col3:
            st.markdown("""
            **🔶 아토픽 마치형**
            - 천식: 있음
            - 아토피: 있음
            - 식품알레르기: 있음
            - 발병 나이: 5세
            """)


# ────────────────────────────────────────────────────
# Tab 2: 클러스터 현황
# ────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 비염 유형 분포 (55,567명)")

    labels = list(CLUSTER_STATS.keys())
    pcts   = [v["pct"] for v in CLUSTER_STATS.values()]
    colors = ["#66BB6A", "#42A5F5", "#FFA726"]

    col_a, col_b = st.columns(2)

    with col_a:
        fig_pie = px.pie(
            names=labels, values=pcts,
            color_discrete_sequence=colors,
            title="유형별 비율"
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(height=350, margin=dict(t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        stats_df = pd.DataFrame([
            {"유형": k, "환자 수": f"{v['size']:,}명",
             "비율": f"{v['pct']}%", "평균 발병 나이": f"{v['onset']}세",
             "천식 동반율": f"{v['asthma']}%", "아토피 동반율": f"{v['atopy']}%",
             "식품알레르기율": f"{v['food']}%"}
            for k, v in CLUSTER_STATS.items()
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # 동반 질환 비교 바 차트
    st.markdown("### 🔬 유형별 동반 질환 비율 비교")
    compare_df = pd.DataFrame({
        "유형":       labels * 3,
        "동반 질환":  ["천식"]*3 + ["아토피"]*3 + ["식품알레르기"]*3,
        "비율 (%)":   [v["asthma"] for v in CLUSTER_STATS.values()] +
                      [v["atopy"]  for v in CLUSTER_STATS.values()] +
                      [v["food"]   for v in CLUSTER_STATS.values()],
    })
    fig_bar = px.bar(
        compare_df, x="동반 질환", y="비율 (%)", color="유형",
        barmode="group", color_discrete_sequence=colors,
        title="유형별 동반 질환 비율"
    )
    fig_bar.update_layout(height=350, margin=dict(t=50, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)


# ────────────────────────────────────────────────────
# Tab 3: 유형 안내
# ────────────────────────────────────────────────────
with tab3:
    st.markdown("### ℹ️ 3대 비염 유형 안내")

    for label, stat in CLUSTER_STATS.items():
        emoji  = TYPE_EMOJI.get(label, "")
        badge  = BADGE.get(label, "badge-green")

        with st.expander(f"{emoji}  {label}  ({stat['pct']}%)", expanded=False):
            c1, c2 = st.columns([2, 1])
            with c1:
                # API에서 가이드 가져오기
                try:
                    r = requests.get(f"{API_URL}/guide/{label}", timeout=2)
                    if r.status_code == 200:
                        info = r.json()
                        st.markdown(f"**설명:** {info['description']}")
                        st.markdown("**관리 가이드:**")
                        for g in info["guide"]:
                            st.markdown(f"- {g}")
                except:
                    st.markdown("API 서버에 연결하면 상세 가이드를 볼 수 있습니다.")
            with c2:
                st.metric("전체 비율",    f"{stat['pct']}%")
                st.metric("평균 발병 나이", f"{stat['onset']}세")
                st.metric("천식 동반율",   f"{stat['asthma']}%")
