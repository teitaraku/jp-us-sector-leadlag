import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.strategies.core import perf_metrics, run_backtest_loop
from src.strategies.double import compute_return as _double
from src.strategies.exp import run_backtest as _exp_run_backtest
from src.strategies.mom import compute_return as _mom
from src.strategies.pca_plain import compute_return as _pca_plain
from src.strategies.pca_sub import compute_return as _pca_sub
from src.tabs.common import color_best

_PAPER_STRATEGIES = {"MOM": _mom, "PCA_PLAIN": _pca_plain, "PCA_SUB": _pca_sub, "DOUBLE": _double}

_RENAME_P = {
    "MOM": "MOM",
    "PCA_PLAIN": "PCA PLAIN",
    "PCA_SUB": "PCA SUB(論文)",
    "DOUBLE": "DOUBLE",
}

_COMBINED_CFG = [
    ("PCA SUB(論文)", "blue", "solid"),
    ("PCA SUB(改)", "cornflowerblue", "dash"),
    ("DOUBLE", "green", "solid"),
    ("PCA PLAIN", "orange", "solid"),
    ("MOM", "red", "solid"),
]

_METRIC_HELP = {
    "AR(%)": "年率リターン。大きいほど高収益。",
    "RISK(%)": "年率ボラティリティ。小さいほど安定。",
    "R/R": "リターン/リスク比（シャープレシオ相当）。大きいほど効率的。",
    "MDD(%)": "最大ドローダウン。小さいほど安定。",
}

_STRAT_HELP = {
    "PCA SUB(論文)": "部分空間正則化 PCA（論文実装）。事前部分空間 V₀ を正則化のアンカーに用いた提案手法。",
    "PCA SUB(改)": "部分空間正則化 PCA（実験中）。論文実装を起点に自由に改変可能な独立コピー。",
    "DOUBLE": "ダブルソート。MOM と PCA SUB の両シグナルで 2 段階スクリーニングを行う複合戦略。",
    "PCA PLAIN": "正則化なし PCA。日米結合相関行列を固有分解してリードラグシグナルを抽出。",
    "MOM": "モメンタム。米国業種リターンをそのまま日本業種シグナルに使うシンプルなベースライン。",
}

_MONTH_COLS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _monthly_pivot(series: pd.Series) -> pd.DataFrame:
    m = series.dropna().resample("ME").sum() * 100
    df_m = pd.DataFrame({"ret": m, "Y": m.index.year, "M": m.index.month})
    piv = df_m.pivot(index="Y", columns="M", values="ret")
    piv.columns = _MONTH_COLS[: len(piv.columns)]
    return piv


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
        "論文戦略と新戦略（実験中）を同時に計算・比較します。「バックテスト実行」を押すと計算が始まります。"
    )

    if st.button("🚀 バックテスト実行", type="primary"):
        with st.spinner("計算中…"):
            try:
                prog = st.progress(0.0, text="論文戦略を計算中…")

                def on_progress_p(step: int, total: int) -> None:
                    prog.progress(step / total * 0.5, text=f"論文戦略を計算中… {step}/{total}")

                def on_progress_e(step: int, total: int) -> None:
                    prog.progress(0.5 + step / total * 0.5, text=f"新戦略を計算中… {step}/{total}")

                rets_p = run_backtest_loop(
                    us_cc,
                    jp_cc,
                    jp_oc,
                    L=L,
                    lam=lam,
                    K=K,
                    q=q,
                    cfull_end="2014-12-31",
                    strategies=_PAPER_STRATEGIES,
                    on_progress=on_progress_p,
                    backtest_start=start,
                )
                rets_e = _exp_run_backtest(
                    us_cc,
                    jp_cc,
                    jp_oc,
                    L=L,
                    lam=lam,
                    K=K,
                    q=q,
                    on_progress=on_progress_e,
                    backtest_start=start,
                )
                prog.empty()
                st.session_state["rets_paper"] = rets_p
                st.session_state["rets_exp"] = rets_e
                st.success(f"完了！ {len(rets_p)} 日分のリターンを計算しました。")
            except Exception as exc:
                st.error(f"エラー: {exc}")
                import traceback

                st.code(traceback.format_exc())

    if "rets_paper" not in st.session_state:
        st.info("「バックテスト実行」ボタンを押してください。")
        return

    rets_p: pd.DataFrame = st.session_state["rets_paper"]
    rets_e: pd.DataFrame = st.session_state["rets_exp"]

    rets_all = rets_p.rename(columns=_RENAME_P).join(
        rets_e[["PCA_SUB"]].rename(columns={"PCA_SUB": "PCA SUB(改)"}), how="outer"
    )

    # ── パフォーマンス指標 ──
    st.subheader("パフォーマンス指標")
    st.caption(
        "MOM・PCA PLAIN・DOUBLE は比較用ベースライン（改版なし）。PCA SUB（提案手法）は論文版と改版を並べて比較します。"
    )
    metrics = perf_metrics(rets_all)
    order = [
        k
        for k in ["PCA SUB(論文)", "PCA SUB(改)", "DOUBLE", "PCA PLAIN", "MOM"]
        if k in metrics.index
    ]
    tbl = metrics.loc[order].copy()
    st.dataframe(
        tbl.style.apply(color_best).format("{:.2f}"),
        width="stretch",
        column_config={
            "AR(%)": st.column_config.NumberColumn("AR (%)", help=_METRIC_HELP["AR(%)"]),
            "RISK(%)": st.column_config.NumberColumn("RISK (%)", help=_METRIC_HELP["RISK(%)"]),
            "R/R": st.column_config.NumberColumn("R/R", help=_METRIC_HELP["R/R"]),
            "MDD(%)": st.column_config.NumberColumn("MDD (%)", help=_METRIC_HELP["MDD(%)"]),
        },
    )
    with st.expander("各戦略の説明"):
        for key in order:
            st.markdown(f"**{key}** — {_STRAT_HELP[key]}")

    # ── 累積リターン ──
    st.subheader("累積リターン推移")
    st.caption("実線が論文戦略、破線が新戦略（改）です。同色で比較できます。")
    cum = (1 + rets_all).cumprod()
    fig_cum = go.Figure()
    for name, color, dash in _COMBINED_CFG:
        if name in cum.columns:
            fig_cum.add_trace(
                go.Scatter(
                    x=cum.index,
                    y=cum[name],
                    name=name,
                    mode="lines",
                    line=dict(color=color, width=2, dash=dash),
                )
            )
    fig_cum.update_layout(
        height=450,
        yaxis_title="累積リターン",
        xaxis_title="日付",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig_cum, width="stretch")
    _, dl1, dl2 = st.columns([2, 1, 1])
    with dl1:
        st.download_button(
            "📥 累積リターン CSV",
            data=cum.to_csv().encode("utf-8-sig"),
            file_name="cumulative_returns.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📥 日次リターン CSV",
            data=rets_all.to_csv().encode("utf-8-sig"),
            file_name="daily_returns.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── ドローダウン ──
    st.subheader("ドローダウン")
    st.caption(
        "累積リターンが過去ピーク比でどれだけ下落したかを示します。実線が論文戦略、破線が新戦略（改）です。"
    )
    fig_dd = go.Figure()
    for name, color, dash in _COMBINED_CFG:
        if name in rets_all.columns:
            r = rets_all[name].dropna()
            cum_r = (1 + r).cumprod()
            dd = (cum_r - cum_r.cummax()) / cum_r.cummax() * 100
            fig_dd.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd,
                    name=name,
                    mode="lines",
                    line=dict(color=color, dash=dash),
                )
            )
    fig_dd.update_layout(
        height=350,
        yaxis_title="ドローダウン (%)",
        xaxis_title="日付",
        hovermode="x unified",
    )
    st.plotly_chart(fig_dd, width="stretch")

    # ── 月次リターン・ヒートマップ（PCA SUB 論文 vs 改） ──
    cols_hm = [c for c in ["PCA SUB(論文)", "PCA SUB(改)"] if c in rets_all.columns]
    if cols_hm:
        st.subheader("PCA SUB 月次リターン (%)")
        st.caption("左が論文戦略、右が新戦略（改）。緑が利益月、赤が損失月です。")
        hm_cols = st.columns(len(cols_hm))
        for col_st, col_name in zip(hm_cols, cols_hm, strict=False):
            with col_st:
                piv = _monthly_pivot(rets_all[col_name])
                fig_m = go.Figure(
                    go.Heatmap(
                        z=piv.values,
                        x=piv.columns.tolist(),
                        y=piv.index.tolist(),
                        colorscale="RdYlGn",
                        zmid=0,
                        text=piv.values.round(1),
                        texttemplate="%{text}",
                        colorbar=dict(title="%"),
                    )
                )
                fig_m.update_layout(height=430, title=col_name, xaxis_title="月", yaxis_title="年")
                st.plotly_chart(fig_m, width="stretch")

    # ── ローリング・シャープレシオ ──
    st.subheader("ローリング・シャープレシオ（252 営業日）")
    st.caption(
        "実線が論文戦略、破線が新戦略（改）です。0 を下回る期間はリスクに対してリターンが出ていない局面です。"
    )
    fig_sh = go.Figure()
    for name, color, dash in _COMBINED_CFG:
        if name in rets_all.columns:
            r = rets_all[name].dropna()
            rs = r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
            fig_sh.add_trace(
                go.Scatter(
                    x=rs.index,
                    y=rs,
                    name=name,
                    mode="lines",
                    line=dict(color=color, dash=dash),
                )
            )
    fig_sh.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_sh.update_layout(
        height=350,
        yaxis_title="シャープレシオ（年率）",
        xaxis_title="日付",
        hovermode="x unified",
    )
    st.plotly_chart(fig_sh, width="stretch")

    # ── 年次リターン比較 ──
    st.subheader("年次リターン比較 (%)")
    st.caption(
        "各年の戦略別リターンを棒グラフで比較します。実線系が論文戦略、破線系（同色）が新戦略（改）です。"
    )
    annual = rets_all.resample("YE").sum() * 100
    annual.index = annual.index.year
    fig_ann = go.Figure()
    for name, color, _ in _COMBINED_CFG:
        if name in annual.columns:
            fig_ann.add_trace(
                go.Bar(
                    x=annual.index,
                    y=annual[name],
                    name=name,
                    marker_color=color,
                    opacity=0.85,
                )
            )
    fig_ann.update_layout(
        height=380,
        barmode="group",
        yaxis_title="年次リターン (%)",
        xaxis_title="年",
    )
    st.plotly_chart(fig_ann, width="stretch")
