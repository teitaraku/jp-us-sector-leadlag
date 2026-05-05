"""
日米業種リードラグ投資戦略ダッシュボード
部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略
Nakagawa et al. (2026), JSAI SIG-FIN-036
"""

import pandas as pd
import streamlit as st

from src.strategies.core import load_data
from src.tabs import analysis, backtest, data, gap_analysis, howto, optimize, today
from src.tabs.common import TAB_CSS

st.set_page_config(
    page_title="日米業種リードラグ投資戦略",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_data(start: str, end: str):
    return load_data(start, end)


def _reset_params():
    st.session_state["L"] = 60
    st.session_state["lam"] = 0.9
    st.session_state["K"] = 3
    st.session_state["q"] = 0.30


def _init_params():
    st.session_state.setdefault("L", 60)
    st.session_state.setdefault("lam", 0.9)
    st.session_state.setdefault("K", 3)
    st.session_state.setdefault("q", 0.30)


def main() -> None:
    _init_params()
    st.markdown(TAB_CSS, unsafe_allow_html=True)
    st.title("📈 日米業種リードラグ投資戦略ダッシュボード")
    st.caption(
        "部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略 | "
        "Nakagawa et al. (2026), JSAI SIG-FIN-036"
    )

    # ── サイドバー ────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ パラメータ設定")
        if st.button("🔄 最新データに更新", width="stretch", type="primary"):
            _load_data.clear()
            st.rerun()

        st.markdown("---")
        today_dt = pd.Timestamp.today()
        start = st.date_input(
            "開始日",
            value=pd.Timestamp("2010-01-01"),
            min_value=pd.Timestamp("2000-01-01"),
            max_value=today_dt,
            help="バックテストおよびデータ取得の開始日。2010年以降が推奨（TOPIX-17 ETF の流動性確保のため）。",
        ).strftime("%Y-%m-%d")
        end = st.date_input(
            "終了日",
            value=today_dt,
            min_value=pd.Timestamp("2000-01-01"),
            max_value=today_dt,
            help="バックテストおよびデータ取得の終了日。「今日のシグナル」タブではこの日付以前の最新米国取引日を使用します。",
        ).strftime("%Y-%m-%d")
        L = st.slider(
            "推定ウィンドウ L（営業日）",
            20,
            252,
            step=5,
            key="L",
            help="相関行列 Cₜ を推定するローリングウィンドウの長さ（営業日数）。"
            "短くすると直近の相場変化への追従が速くなるが推定ノイズが増加する。"
            "長くすると安定するが直近トレンドへの反応が遅くなる。論文推奨値: 60。",
        )
        lam = st.slider(
            "正則化パラメータ λ",
            0.0,
            1.0,
            step=0.05,
            key="lam",
            help="事前エクスポージャー行列 C₀ への正則化強度。"
            "C_reg = (1−λ)·Cₜ + λ·C₀ で混合される。"
            "λ=1 で完全に事前知識のみ、λ=0 で正則化なし（PCA PLAIN と同等）。論文推奨値: 0.9。",
        )
        K = st.slider(
            "主成分数 K",
            1,
            5,
            key="K",
            help="正則化相関行列から抽出する上位固有ベクトルの本数。"
            "v₁（グローバルファクター）・v₂（国スプレッド）・v₃（シクリカル/DF）の 3 成分が基本。"
            "増やすと細かい共変動を捉えるが過学習リスクが高まる。論文推奨値: 3。",
        )
        q = st.slider(
            "分位点 q（ロング/ショート比率）",
            0.10,
            0.45,
            step=0.05,
            key="q",
            help="シグナル上位 q% をロング、下位 q% をショートとする閾値。"
            "q=0.30 かつ日本 ETF 17 銘柄の場合、ロング 5 銘柄・ショート 5 銘柄。"
            "大きくすると銘柄数が増え分散効果は高まるが 1 銘柄あたりの期待リターンは低下する。論文推奨値: 0.30。",
        )

        st.markdown("---")
        st.markdown("**PCA SUB(論文) 推奨パラメータ**: L=60, λ=0.9, K=3, q=0.3")
        st.button("PCA SUB(論文) パラメータにリセット", on_click=_reset_params, width="stretch")

        st.markdown("---")
        st.markdown(
            """
**⏰ 取引タイムライン（東京時間）**

| イベント | 時刻 |
|:---|:---:|
| 🇺🇸 NY クローズ（夏時間） | 05:00 |
| 🇺🇸 NY クローズ（冬時間） | 06:00 |
| ✅ シグナル確認・発注準備 | 〜08:30 |
| 🇯🇵 東証オープン | 09:00 |
| 🇯🇵 東証クローズ | 15:30 |
""",
        )

    # ── データ取得 ────────────────────────────────────
    fetch_start = min(start, "2010-01-01")
    with st.spinner("価格データを取得中…"):
        try:
            us_cc, jp_cc, jp_oc, us_close, jp_close = _load_data(fetch_start, end)
        except Exception as exc:
            st.error(f"データ取得に失敗しました: {exc}")
            return
    us_cc_view = us_cc.loc[start:]
    jp_cc_view = jp_cc.loc[start:]
    jp_oc_view = jp_oc.loc[start:]

    # ── タブ ─────────────────────────────────────────
    tab_today, tab_bt, tab_opt, tab_gap, tab_data, tab_sig, tab_over = st.tabs(
        [
            "🎯 今日のシグナル",
            "📈 バックテスト",
            "🔎 パラメータ探索",
            "🔍 ギャップ分析",
            "📊 データ",
            "🔬 モデル分析",
            "📖 使い方",
        ]
    )

    with tab_over:
        howto.render()

    with tab_data:
        data.render(us_cc_view, jp_cc_view, jp_oc_view)

    with tab_sig:
        analysis.render(us_cc, jp_cc, L, lam)

    with tab_bt:
        backtest.render(us_cc, jp_cc, jp_oc, L, lam, K, q, start)

    with tab_gap:
        gap_analysis.render(us_cc, jp_cc, jp_oc, L, lam, K, q, start)

    with tab_opt:
        optimize.render(us_cc, jp_cc, jp_oc, start)

    with tab_today:
        today.render(us_cc, jp_cc, jp_oc, L, lam, K, q)


if __name__ == "__main__":
    main()
