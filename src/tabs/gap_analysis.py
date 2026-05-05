"""ギャップ分析タブ — 仮説検証: シグナル対象銘柄の始値ギャップ分析

仮説: 2026年3月以降、ロング候補銘柄が Open 時点で先行して動いており、
OC リターンが減少している。

overnight_gap[t] = jp_open[t] / jp_close[t-1] - 1
                 = (1 + jp_cc[t]) / (1 + jp_oc[t]) - 1  ← 既存データから導出
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.strategies.core import (
    JP_TICKERS,
    NJ,
    NU,
    US_TICKERS,
    StepContext,
    _prepare_prior,
    _window_state,
)
from src.strategies.pca_sub import compute_signal as _pca_sub_signal


def _compute_overnight_gap(jp_cc: pd.DataFrame, jp_oc: pd.DataFrame) -> pd.DataFrame:
    """始値ギャップ（前日終値 → 当日始値）を既存データから導出する。"""
    common = jp_cc.index.intersection(jp_oc.index)
    cc = jp_cc.loc[common, JP_TICKERS]
    oc = jp_oc.loc[common, JP_TICKERS]
    return (1 + cc) / (1 + oc.replace(0, np.nan)) - 1


def _run_analysis(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    overnight_gap: pd.DataFrame,
    L: int,
    lam: float,
    K: int,
    q: float,
    backtest_start: str | None = None,
    on_progress=None,
) -> pd.DataFrame:
    """各取引日について、ロング/ショート対象の始値ギャップと OC リターンを集計する。"""
    _, C0, all_cc = _prepare_prior(us_cc, jp_cc, "2014-12-31")

    jp_dates_arr = jp_oc.index.values
    us_dates = us_cc[US_TICKERS].dropna(how="all").index
    backtest_start_ts = pd.Timestamp(backtest_start) if backtest_start else None

    paired: list[tuple] = []
    for us_date in us_dates:
        if backtest_start_ts is not None and us_date < backtest_start_ts:
            continue
        future = jp_dates_arr[jp_dates_arr > us_date]
        if len(future) == 0:
            continue
        next_jp = future[0]
        if next_jp in jp_oc.index:
            paired.append((us_date, next_jp))

    rows: list[dict] = []
    n_pairs = len(paired)

    for step, (us_date, jp_date) in enumerate(paired):
        if on_progress and step % max(n_pairs // 80, 1) == 0:
            on_progress(step, n_pairs)

        t_idx = all_cc.index.searchsorted(us_date, side="left")
        if t_idx < L:
            continue

        window = all_cc.iloc[t_idx - L : t_idx]
        if window.isna().to_numpy().mean() > 0.3:
            continue

        mu, sigma, Ct = _window_state(window, C0)
        us_today = us_cc.loc[us_date, US_TICKERS].values.astype(float)
        z_us = (us_today - mu[:NU]) / sigma[:NU]
        z_us = np.where(np.isfinite(z_us), z_us, 0.0)

        ctx = StepContext(
            mu=mu, sigma=sigma, z_us=z_us, Ct=Ct, C0=C0, target=None, K=K, lam=lam, q=q
        )
        signal = _pca_sub_signal(ctx)

        n_each = max(1, int(NJ * q))
        rank = np.argsort(signal)
        long_idx = rank[-n_each:]
        short_idx = rank[:n_each]

        if jp_date not in overnight_gap.index or jp_date not in jp_oc.index:
            continue

        gap_row = overnight_gap.loc[jp_date, JP_TICKERS].values.astype(float)
        oc_row = jp_oc.loc[jp_date, JP_TICKERS].values.astype(float)

        valid = np.isfinite(gap_row) & np.isfinite(oc_row)
        long_valid = long_idx[np.isin(long_idx, np.where(valid)[0])]
        short_valid = short_idx[np.isin(short_idx, np.where(valid)[0])]

        if len(long_valid) == 0 or len(short_valid) == 0:
            continue

        overnight_ls = float(gap_row[long_valid].mean() - gap_row[short_valid].mean())
        oc_ls = float(oc_row[long_valid].mean() - oc_row[short_valid].mean())

        rows.append(
            {
                "us_date": us_date,
                "jp_date": jp_date,
                "overnight_ls": overnight_ls,
                "oc_ls": oc_ls,
                "long_tickers": [JP_TICKERS[i] for i in long_valid],
                "short_tickers": [JP_TICKERS[i] for i in short_valid],
                "long_gap_mean": float(gap_row[long_valid].mean()),
                "short_gap_mean": float(gap_row[short_valid].mean()),
                "long_oc_mean": float(oc_row[long_valid].mean()),
                "short_oc_mean": float(oc_row[short_valid].mean()),
            }
        )

    return pd.DataFrame(rows).set_index("us_date") if rows else pd.DataFrame()


def _plot_rolling(df: pd.DataFrame, window: int = 20) -> go.Figure:
    cutoff = df.index.max() - pd.DateOffset(years=1)
    roll = df[["overnight_ls", "oc_ls"]].rolling(window, min_periods=5).mean() * 100
    roll = roll.loc[cutoff:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roll.index,
            y=roll["overnight_ls"],
            name="始値ギャップ LS（前日終値→当日始値）",
            mode="lines",
            line=dict(color="orangered", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=roll.index,
            y=roll["oc_ls"],
            name="OC リターン LS（当日始値→終値）",
            mode="lines",
            line=dict(color="steelblue", width=2),
        )
    )
    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dash")
    fig.add_vrect(
        x0="2026-03-01",
        x1=roll.index.max().isoformat(),
        fillcolor="red",
        opacity=0.05,
        line_width=0,
        annotation_text="2026年3月〜",
        annotation_position="top left",
    )
    fig.update_layout(
        height=400,
        title=f"ローリング {window} 日平均（%）",
        yaxis_title="ロングショート差分 (%)",
        xaxis_title="米国取引日",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def _plot_frontrun_rate(df: pd.DataFrame, window: int = 20) -> go.Figure:
    cutoff = df.index.max() - pd.DateOffset(years=1)
    total = df["overnight_ls"].abs() + df["oc_ls"].abs()
    rate = df["overnight_ls"].abs() / total.replace(0, np.nan)
    rate_roll = rate.rolling(window, min_periods=5).mean().loc[cutoff:] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rate_roll.index,
            y=rate_roll,
            mode="lines",
            line=dict(color="purple", width=2),
            fill="tozeroy",
            fillcolor="rgba(128,0,128,0.1)",
            name="始値で先取りされた割合",
        )
    )
    fig.add_hline(
        y=50,
        line_color="gray",
        line_dash="dash",
        line_width=1,
        annotation_text="50%",
        annotation_position="right",
    )
    fig.add_vrect(
        x0="2026-03-01",
        x1=rate_roll.index.max().isoformat(),
        fillcolor="red",
        opacity=0.05,
        line_width=0,
    )
    fig.update_layout(
        height=300,
        title=f"始値ギャップが LS 変動に占める割合（ローリング {window} 日平均）",
        yaxis=dict(title="割合 (%)", range=[0, 100]),
        xaxis_title="米国取引日",
        hovermode="x unified",
    )
    return fig


def _plot_monthly_bar(df: pd.DataFrame) -> go.Figure:
    cutoff = df.index.max() - pd.DateOffset(years=1)
    monthly = df.loc[cutoff:, ["overnight_ls", "oc_ls"]].resample("ME").mean() * 100
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly["overnight_ls"],
            name="始値ギャップ LS",
            marker_color="orangered",
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly["oc_ls"],
            name="OC リターン LS",
            marker_color="steelblue",
            opacity=0.8,
        )
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        height=380,
        title="月次平均 (%)",
        barmode="group",
        yaxis_title="ロングショート差分 (%)",
        xaxis_title="月",
        hovermode="x unified",
    )
    return fig


def _recent_detail_table(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    recent = df.tail(days).copy()
    tbl = pd.DataFrame(
        {
            "日本取引日": recent["jp_date"].dt.strftime("%Y-%m-%d"),
            "始値ギャップ LS (%)": (recent["overnight_ls"] * 100).round(3),
            "OC リターン LS (%)": (recent["oc_ls"] * 100).round(3),
            "ロング始値ギャップ (%)": (recent["long_gap_mean"] * 100).round(3),
            "ショート始値ギャップ (%)": (recent["short_gap_mean"] * 100).round(3),
            "ロング OC (%)": (recent["long_oc_mean"] * 100).round(3),
            "ショート OC (%)": (recent["short_oc_mean"] * 100).round(3),
        },
        index=recent.index,
    )
    tbl.index = [d.strftime("%Y-%m-%d") for d in tbl.index]
    tbl.index.name = "米国取引日"
    return tbl.sort_index(ascending=False)


def render(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int,
    lam: float,
    K: int,
    q: float,
    start: str,
) -> None:
    st.caption(
        "**仮説**: 2026年3月以降、ロング候補銘柄が Open 時点で先行して動いているため "
        "OC リターンが減少している。始値ギャップ（前日終値→当日始値）と OC リターン（当日始値→終値）を "
        "PCA SUB シグナルのロング/ショート対象に分けて集計し、仮説を検証します。"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("🔍 分析実行", type="primary")
    with col2:
        rolling_w = st.slider("ローリング平均ウィンドウ（日）", 5, 60, 20, key="gap_roll_window")

    if run_btn:
        with st.spinner("分析中…"):
            try:
                prog = st.progress(0.0, text="シグナル計算中…")

                def on_prog(step: int, total: int) -> None:
                    prog.progress(step / total, text=f"シグナル計算中… {step}/{total}")

                overnight_gap = _compute_overnight_gap(jp_cc, jp_oc)
                result = _run_analysis(
                    us_cc,
                    jp_cc,
                    jp_oc,
                    overnight_gap,
                    L=L,
                    lam=lam,
                    K=K,
                    q=q,
                    backtest_start=start,
                    on_progress=on_prog,
                )
                prog.empty()
                st.session_state["gap_result"] = result
                st.session_state["gap_params"] = dict(L=L, lam=lam, K=K, q=q)
                st.success(f"完了！ {len(result)} 日分を分析しました。")
            except Exception as exc:
                import traceback

                st.error(f"エラー: {exc}")
                st.code(traceback.format_exc())

    if "gap_result" not in st.session_state:
        st.info("「分析実行」ボタンを押してください。")
        return

    df: pd.DataFrame = st.session_state["gap_result"]
    if df.empty:
        st.warning("分析結果が空です。")
        return

    params = st.session_state.get("gap_params", {})
    st.caption(
        f"実行時パラメータ: L={params.get('L')}  λ={params.get('lam')}  K={params.get('K')}  q={params.get('q')}"
    )

    # ── サマリー統計 ──────────────────────────────────
    st.subheader("期間別サマリー")
    periods = {
        "全期間": df,
        "〜2026-02": df[df.index < "2026-03-01"],
        "2026-03〜": df[df.index >= "2026-03-01"],
    }
    summary_rows = []
    for label, sub in periods.items():
        if sub.empty:
            continue
        summary_rows.append(
            {
                "期間": label,
                "日数": len(sub),
                "始値ギャップ LS 平均 (%)": round(sub["overnight_ls"].mean() * 100, 3),
                "OC リターン LS 平均 (%)": round(sub["oc_ls"].mean() * 100, 3),
                "始値ギャップ LS 勝率 (%)": round((sub["overnight_ls"] > 0).mean() * 100, 1),
                "OC リターン LS 勝率 (%)": round((sub["oc_ls"] > 0).mean() * 100, 1),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows).set_index("期間"), width="stretch")

    # ── 時系列グラフ ──────────────────────────────────
    st.subheader("ローリング平均推移")
    st.caption(
        "橙: 始値ギャップの LS 差分（シグナルが Open 時点で既に先取りされた分）。"
        "青: OC リターンの LS 差分（実際に戦略が取れるリターン）。"
        "仮説が正しければ、2026年3月以降に橙が上昇・青が低下するはず。"
    )
    st.plotly_chart(_plot_rolling(df, rolling_w), width="stretch")

    # ── 始値先取り率 ──────────────────────────────────
    st.subheader("始値ギャップが LS 変動に占める割合")
    st.caption(
        "シグナルによるロングショート方向の動きのうち、何%が Open 時点で取られているかを示します。"
        "50% を超えると、始値で先取りされた分が OC リターンを上回っています。"
    )
    st.plotly_chart(_plot_frontrun_rate(df, rolling_w), width="stretch")

    # ── 月次棒グラフ ──────────────────────────────────
    st.subheader("月次平均")
    st.plotly_chart(_plot_monthly_bar(df), width="stretch")

    # ── 直近の詳細 ──────────────────────────────────
    st.subheader("直近 30 日の詳細")
    st.caption(
        "始値ギャップ LS がプラス = ロング候補が Open 時点で既に上昇済み（仮説の裏付け）。"
        "OC リターン LS がマイナス = Open 以降に逆転が起きている。"
    )
    tbl = _recent_detail_table(df, 30)
    st.dataframe(
        tbl.style.map(
            lambda v: (
                "background-color: #ffd5cc; color: #1a1a1a"
                if isinstance(v, float) and v < 0
                else "background-color: #d5f5d5; color: #1a1a1a"
                if isinstance(v, float) and v > 0
                else ""
            ),
            subset=["始値ギャップ LS (%)", "OC リターン LS (%)"],
        ).format("{:.3f}", subset=[c for c in tbl.columns if "%" in c]),
        width="stretch",
    )

    # ── CSV ダウンロード ──────────────────────────────
    st.download_button(
        "📥 分析結果 CSV",
        data=df[
            [
                "jp_date",
                "overnight_ls",
                "oc_ls",
                "long_gap_mean",
                "short_gap_mean",
                "long_oc_mean",
                "short_oc_mean",
            ]
        ]
        .to_csv()
        .encode("utf-8-sig"),
        file_name="gap_analysis.csv",
        mime="text/csv",
    )
