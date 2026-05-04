import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.strategies.core import JP_LABEL, US_LABEL, US_TICKERS
from src.strategies.exp import run_live_signal as _exp_run_live_signal
from src.strategies.pca_sub import run_live_signal as _pca_sub_live
from src.tabs.common import highlight_action


def render(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int,
    lam: float,
    K: int,
    q: float,
) -> None:
    st.subheader("🎯 今日の売買シグナル（PCA SUB）")
    strategy_mode = st.radio(
        "使用する戦略を選択",
        ["📄 PCA SUB（論文）", "🧪 PCA SUB 改良版"],
        horizontal=True,
        key="strategy_mode",
        help="「PCA SUB（論文）」は Nakagawa et al. (2026) の実装そのまま。"
        "「PCA SUB 改良版」は自由に改変可能な独立コピー。",
    )
    use_exp = strategy_mode == "🧪 PCA SUB 改良版"
    st.caption(
        f"現在の戦略: {strategy_mode} ｜ "
        "サイドバーの終了日に含まれる最新米国取引日のリターンを使用してシグナルを計算します。"
        "米国市場クローズ（東京時間 5:00〔夏〕/ 6:00〔冬〕）後、東証オープン（9:00）までに確認してください。"
    )

    try:
        sig_result = (
            _exp_run_live_signal(us_cc, jp_cc, jp_oc, L=L, lam=lam, K=K, q=q)
            if use_exp
            else _pca_sub_live(us_cc, jp_cc, jp_oc, L=L, lam=lam, K=K, q=q)
        )
    except Exception as exc:
        st.error(f"シグナル計算に失敗しました: {exc}")
        return

    us_date = sig_result["us_date"]
    signal = sig_result["signal"]
    us_returns = sig_result["us_returns"]
    n_long = sig_result["n_long"]
    n_short = sig_result["n_short"]
    next_jp = sig_result["next_jp_date"]

    next_jp_str = next_jp.strftime("%Y-%m-%d") if next_jp is not None else "データ範囲外"

    st.info(
        f"📅 使用した米国リターン日: **{us_date.strftime('%Y-%m-%d')}**　→　"
        f"日本発注日: **{next_jp_str}**　｜　"
        f"🟢 ロング {n_long} 銘柄 / 🔴 ショート {n_short} 銘柄"
    )

    # ── シグナルランキング表 ──
    signal_sorted = signal.sort_values(ascending=False)
    nj = len(signal_sorted)
    rows = []
    for rank, (ticker, val) in enumerate(signal_sorted.items(), 1):
        if rank <= n_long:
            action = "🟢 ロング"
        elif rank > nj - n_short:
            action = "🔴 ショート"
        else:
            action = "─"
        rows.append(
            {
                "順位": rank,
                "銘柄": JP_LABEL[ticker],
                "コード": ticker,
                "シグナル値": round(float(val), 4),
                "売買": action,
            }
        )

    df_sig = pd.DataFrame(rows)
    st.dataframe(
        df_sig.style.apply(highlight_action, axis=1),
        hide_index=True,
        width="stretch",
        height=36 * len(df_sig) + 38,
    )

    # ── シグナル強度バーチャート ──
    bar_colors = []
    for rank in range(1, nj + 1):
        if rank <= n_long:
            bar_colors.append("#2ecc71")
        elif rank > nj - n_short:
            bar_colors.append("#e74c3c")
        else:
            bar_colors.append("#95a5a6")

    st.subheader("シグナル強度")
    st.caption(
        "今日の米国業種リターン（z スコア）を、日米結合相関行列の上位 K 固有ベクトルで"
        "日本業種空間へ射影した値です。"
        "正の値ほど翌営業日の上昇を、負の値ほど下落を予測します。"
        "緑がロング対象、赤がショート対象の業種です。"
    )
    fig_sig = go.Figure(
        go.Bar(
            x=[JP_LABEL[t] for t in signal_sorted.index],
            y=signal_sorted.values.tolist(),
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in signal_sorted.values],
            textposition="outside",
        )
    )
    fig_sig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_sig.update_layout(
        height=420,
        xaxis_title="業種",
        yaxis_title="シグナル値",
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_sig, width="stretch")

    # ── 米国入力リターン ──
    st.subheader(f"米国業種リターン（入力）: {us_date.strftime('%Y-%m-%d')}")
    st.caption(
        "シグナル計算に使った当日の米国業種 ETF の Close-to-Close リターン実績値です。"
        "上のシグナル強度はこの米国リターンを入力として算出されます。"
        "米国の動きがどの日本業種へ波及しているかを対比して確認できます。"
    )
    us_vals = (us_returns.values * 100).tolist()
    us_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in us_vals]
    fig_us = go.Figure(
        go.Bar(
            x=[US_LABEL[t] for t in US_TICKERS],
            y=us_vals,
            marker_color=us_colors,
            text=[f"{v:.2f}%" for v in us_vals],
            textposition="outside",
        )
    )
    fig_us.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_us.update_layout(
        height=380,
        xaxis_title="業種",
        yaxis_title="リターン (%)",
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_us, width="stretch")
