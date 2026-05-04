import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.strategies.core import (
    JP_LABEL,
    JP_TICKERS,
    NJ,
    NU,
    US_LABEL,
    US_TICKERS,
    _prepare_prior,
    _window_state,
    build_V0,
)


def render(us_cc: pd.DataFrame, jp_cc: pd.DataFrame, L: int, lam: float) -> None:
    st.caption(
        "アルゴリズム内部で使用する事前固有ベクトル V₀・エクスポージャー行列 C₀・正則化相関行列 C_reg を可視化します。"
    )
    V0 = build_V0()
    labels_all = [US_LABEL[t] for t in US_TICKERS] + [JP_LABEL[t] for t in JP_TICKERS]
    bar_colors = ["#1f77b4"] * NU + ["#d62728"] * NJ
    factor_titles = ["v₁: グローバル", "v₂: 国スプレッド", "v₃: シクリカル/DF"]

    st.subheader("事前固有ベクトル V₀（青=米国, 赤=日本）")
    st.caption(
        "正則化のアンカーとなる 3 本の事前固有ベクトルです。v₁ はすべての業種が同方向に動くグローバルファクター、v₂ は米国と日本が逆方向に動く国スプレッド、v₃ はシクリカル業種とディフェンシブ業種の対立軸を表します。"
    )
    fig_v0 = make_subplots(rows=1, cols=3, subplot_titles=factor_titles)
    for k in range(3):
        fig_v0.add_trace(
            go.Bar(x=labels_all, y=V0[:, k], marker_color=bar_colors, showlegend=False),
            row=1,
            col=k + 1,
        )
    fig_v0.update_xaxes(tickangle=45, tickfont=dict(size=7))
    fig_v0.update_layout(height=380)
    st.plotly_chart(fig_v0, width="stretch")

    try:
        _, C0, all_cc_full = _prepare_prior(us_cc, jp_cc, "2014-12-31")
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader("事前エクスポージャー行列 C₀")
    st.caption(
        "V₀ を長期相関行列（2015 年以前）に射影して構築した 28×28 の事前エクスポージャー行列です。正則化の「目標値」として機能し、推定ノイズを抑制します。直近 Cₜ との混合比は λ で制御されます。"
    )
    fig_c0 = go.Figure(
        go.Heatmap(
            z=C0,
            x=labels_all,
            y=labels_all,
            colorscale="RdBu_r",
            zmid=0,
            text=C0.round(2),
            texttemplate="%{text}",
            textfont=dict(size=6),
        )
    )
    fig_c0.update_layout(
        height=600,
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
    )
    st.plotly_chart(fig_c0, width="stretch")

    st.subheader("正則化相関行列の比較（直近ウィンドウ）")
    st.caption(
        f"直近 L={L} 日の実測相関行列 Cₜ（左）と、C₀ で正則化した C_reg（右）を並べて比較します。λ が大きいほど右図は C₀ に近づき、ノイズが平滑化された構造になります。"
    )
    recent = all_cc_full.iloc[-L:]
    if len(recent) >= 10:
        _, _, Ct_recent = _window_state(recent, C0)
        C_reg_recent = (1 - lam) * Ct_recent + lam * C0

        c1, c2 = st.columns(2)
        for col, mat, title in [
            (c1, Ct_recent, f"C_t（直近 {L} 日）"),
            (c2, C_reg_recent, f"C_reg（λ={lam}）"),
        ]:
            with col:
                fig = go.Figure(
                    go.Heatmap(
                        z=mat,
                        x=labels_all,
                        y=labels_all,
                        colorscale="RdBu_r",
                        zmid=0,
                        zmin=-1,
                        zmax=1,
                    )
                )
                fig.update_layout(
                    height=500,
                    title=title,
                    xaxis=dict(tickangle=45, tickfont=dict(size=7)),
                    yaxis=dict(tickfont=dict(size=7)),
                )
                st.plotly_chart(fig, width="stretch")
