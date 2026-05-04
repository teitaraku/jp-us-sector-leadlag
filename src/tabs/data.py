import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.strategies.core import JP_LABEL, JP_TICKERS, US_LABEL, US_TICKERS


def render(
    us_cc_view: pd.DataFrame,
    jp_cc_view: pd.DataFrame,
    jp_oc_view: pd.DataFrame,
) -> None:
    st.caption(
        "yfinance で取得した日米業種 ETF の価格データ・リターン分布・相関構造を確認できます。"
    )
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("米国業種 ETF 累積リターン (CC)")
        st.caption(
            "SPDR 11 業種 ETF の Close-to-Close（前日終値→当日終値）リターンを累積した推移です。戦略の入力シグナル源となるデータです。"
        )
        cum_us = (1 + us_cc_view[US_TICKERS]).cumprod()
        fig = go.Figure()
        for t in US_TICKERS:
            if t in cum_us.columns:
                fig.add_trace(
                    go.Scatter(x=cum_us.index, y=cum_us[t], name=US_LABEL[t], mode="lines")
                )
        fig.update_layout(height=380, yaxis_title="累積リターン", legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("日本業種 ETF 累積リターン (OC)")
        st.caption(
            "NEXT FUNDS TOPIX-17 ETF の Open-to-Close（寄付→引け）リターンを累積した推移です。戦略が予測・売買するターゲットデータです。"
        )
        cum_jp = (1 + jp_oc_view[JP_TICKERS]).cumprod()
        fig = go.Figure()
        for t in JP_TICKERS:
            if t in cum_jp.columns:
                fig.add_trace(
                    go.Scatter(x=cum_jp.index, y=cum_jp[t], name=JP_LABEL[t], mode="lines")
                )
        fig.update_layout(height=380, yaxis_title="累積リターン", legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, width="stretch")

    st.subheader("日米業種間相関行列（CC ベース）")
    st.caption(
        "米国 11 業種 × 日本 17 業種の 28×28 全期間相関行列です。赤が正の相関、青が負の相関を示します。右上の米日ブロックがリードラグ構造の可視化です。"
    )
    all_cc = us_cc_view[US_TICKERS].join(jp_cc_view[JP_TICKERS], how="inner").dropna()
    if len(all_cc) > 0:
        corr = all_cc.corr()
        labels_all = [US_LABEL[t] for t in US_TICKERS] + [JP_LABEL[t] for t in JP_TICKERS]
        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=labels_all,
                y=labels_all,
                colorscale="RdBu_r",
                zmid=0,
                text=corr.values.round(2),
                texttemplate="%{text}",
                textfont=dict(size=6),
            )
        )
        fig.update_layout(
            height=620,
            xaxis=dict(tickangle=45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8)),
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("基本統計量")
    st.caption(
        "全期間の年率リターン・ボラティリティ・シャープレシオ・歪度・尖度を業種別に集計しています。銘柄ごとのリターン特性や分布の非対称性を確認できます。"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.write("**米国業種 ETF（CC ベース）**")
        us_st = pd.DataFrame(
            {
                "年率リターン(%)": (us_cc_view[US_TICKERS].mean() * 252 * 100).round(2),
                "年率ボラ(%)": (us_cc_view[US_TICKERS].std() * np.sqrt(252) * 100).round(2),
                "Sharpe": (
                    us_cc_view[US_TICKERS].mean() / us_cc_view[US_TICKERS].std() * np.sqrt(252)
                ).round(2),
                "Skew": us_cc_view[US_TICKERS].skew().round(2),
                "Kurt": us_cc_view[US_TICKERS].kurt().round(2),
            }
        )
        us_st.index = [US_LABEL[t] for t in US_TICKERS]
        st.dataframe(us_st, width="stretch")

    with c2:
        st.write("**日本業種 ETF（OC ベース）**")
        jp_st = pd.DataFrame(
            {
                "年率リターン(%)": (jp_oc_view[JP_TICKERS].mean() * 252 * 100).round(2),
                "年率ボラ(%)": (jp_oc_view[JP_TICKERS].std() * np.sqrt(252) * 100).round(2),
                "Sharpe": (
                    jp_oc_view[JP_TICKERS].mean() / jp_oc_view[JP_TICKERS].std() * np.sqrt(252)
                ).round(2),
                "Skew": jp_oc_view[JP_TICKERS].skew().round(2),
                "Kurt": jp_oc_view[JP_TICKERS].kurt().round(2),
            }
        )
        jp_st.index = [JP_LABEL[t] for t in JP_TICKERS]
        st.dataframe(jp_st, width="stretch")
