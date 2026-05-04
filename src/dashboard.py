"""
日米業種リードラグ投資戦略ダッシュボード
部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略
Nakagawa et al. (2026), JSAI SIG-FIN-036
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.strategy import (
    JP_LABEL,
    JP_TICKERS,
    NJ,
    NU,
    US_LABEL,
    US_TICKERS,
    build_V0,
    compute_today_signal,
    load_data,
    perf_metrics,
    run_backtest,
)
from src.strategy_core import _prepare_prior, _window_state, compute_live_signal, run_backtest_loop
from src.strategy_exp import compute_return as _exp_return
from src.strategy_exp import compute_signal as _exp_signal

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


def _color_best(s: pd.Series):
    if s.name in ("AR(%)", "R/R"):
        best = s.max()
    elif s.name == "MDD(%)":
        best = s.min()
    else:
        return [""] * len(s)
    return ["background-color:#c8f7c5; color:#000000" if v == best else "" for v in s]


def _highlight_action(row):
    if "ロング" in str(row["売買"]):
        return ["background-color:#c8f7c5; color:#000000"] * len(row)
    if "ショート" in str(row["売買"]):
        return ["background-color:#f7c8c8; color:#000000"] * len(row)
    return [""] * len(row)


_TAB_CSS = """
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: rgba(255,255,255,0.06);
    padding: 4px 6px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 6px 18px;
    color: #888;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(255,255,255,0.08);
    color: #ccc;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(255,255,255,0.18);
    color: #ffffff;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(0,0,0,0.35);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
[data-testid="stMetricValue"] {
    font-size: 1rem;
}
</style>
"""


def main() -> None:
    st.markdown(_TAB_CSS, unsafe_allow_html=True)
    st.title("📈 日米業種リードラグ投資戦略ダッシュボード")
    st.caption(
        "部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略 | "
        "Nakagawa et al. (2026), JSAI SIG-FIN-036"
    )

    # ── サイドバー ────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ パラメータ設定")
        if st.button("🔄 最新データに更新", use_container_width=True, type="primary"):
            _load_data.clear()
            st.rerun()

        strategy_mode = st.radio(
            "戦略モード",
            ["📄 論文戦略", "🧪 新戦略（実験中）"],
            key="strategy_mode",
            help="「論文戦略」は Nakagawa et al. (2026) の実装そのまま。"
            "「新戦略（実験中）」は自由に改変可能な独立コピー。",
            horizontal=True,
        )
        use_exp = strategy_mode == "🧪 新戦略（実験中）"

        st.markdown("---")
        today = pd.Timestamp.today()
        start = st.date_input(
            "開始日",
            value=pd.Timestamp("2010-01-01"),
            min_value=pd.Timestamp("2000-01-01"),
            max_value=today,
            help="バックテストおよびデータ取得の開始日。2010年以降が推奨（TOPIX-17 ETF の流動性確保のため）。",
        ).strftime("%Y-%m-%d")
        end = st.date_input(
            "終了日",
            value=today,
            min_value=pd.Timestamp("2000-01-01"),
            max_value=today,
            help="バックテストおよびデータ取得の終了日。「今日のシグナル」タブではこの日付以前の最新米国取引日を使用します。",
        ).strftime("%Y-%m-%d")
        L = st.slider(
            "推定ウィンドウ L（営業日）",
            20,
            252,
            60,
            5,
            key="L",
            help="相関行列 Cₜ を推定するローリングウィンドウの長さ（営業日数）。"
            "短くすると直近の相場変化への追従が速くなるが推定ノイズが増加する。"
            "長くすると安定するが直近トレンドへの反応が遅くなる。論文推奨値: 60。",
        )
        lam = st.slider(
            "正則化パラメータ λ",
            0.0,
            1.0,
            0.9,
            0.05,
            key="lam",
            help="事前エクスポージャー行列 C₀ への正則化強度。"
            "C_reg = (1−λ)·Cₜ + λ·C₀ で混合される。"
            "λ=1 で完全に事前知識のみ、λ=0 で正則化なし（PCA PLAIN と同等）。論文推奨値: 0.9。",
        )
        K = st.slider(
            "主成分数 K",
            1,
            5,
            3,
            key="K",
            help="正則化相関行列から抽出する上位固有ベクトルの本数。"
            "v₁（グローバルファクター）・v₂（国スプレッド）・v₃（シクリカル/DF）の 3 成分が基本。"
            "増やすと細かい共変動を捉えるが過学習リスクが高まる。論文推奨値: 3。",
        )
        q = st.slider(
            "分位点 q（ロング/ショート比率）",
            0.10,
            0.45,
            0.30,
            0.05,
            key="q",
            help="シグナル上位 q% をロング、下位 q% をショートとする閾値。"
            "q=0.30 かつ日本 ETF 17 銘柄の場合、ロング 5 銘柄・ショート 5 銘柄。"
            "大きくすると銘柄数が増え分散効果は高まるが 1 銘柄あたりの期待リターンは低下する。論文推奨値: 0.30。",
        )

        st.markdown("---")
        st.markdown("**PCA SUB(論文) 推奨パラメータ**: L=60, λ=0.9, K=3, q=0.3")
        st.button(
            "PCA SUB(論文) パラメータにリセット", on_click=_reset_params, use_container_width=True
        )

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
    with st.spinner("価格データを取得中…"):
        try:
            us_cc, jp_cc, jp_oc, us_close, jp_close = _load_data(start, end)
        except Exception as exc:
            st.error(f"データ取得に失敗しました: {exc}")
            return

    # ── タブ ─────────────────────────────────────────
    tab_today, tab_bt, tab_data, tab_sig, tab_over = st.tabs(
        ["🎯 今日のシグナル", "📈 バックテスト", "📊 データ", "🔬 モデル分析", "📖 使い方"]
    )

    # ═══════════════════════════════════════════════════
    # 概要
    # ═══════════════════════════════════════════════════
    with tab_over:
        st.caption(
            "初めての方はここから読んでください。毎日の操作手順から戦略の仕組みまで解説します。"
        )

        # ── 毎日の操作フロー ──
        st.subheader("📋 毎日の操作フロー（3 ステップ）")
        st.markdown(
            """
この戦略は**1 日 1 回**、以下の手順で運用します。
"""
        )

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(
                """
### ① 早朝にシグナル確認
**東京時間 6:00〜8:30 頃**

米国市場がクローズした直後が最適なタイミングです。

1. サイドバー上部の **「🔄 最新データに更新」** を押す
2. **「🎯 今日のシグナル」** タブを開く
3. 表の上部の青いバナーで **使用した米国リターン日** と **日本発注日** を確認する
4. 🟢 **ロング（緑）** の銘柄 → 今日、日本市場で **買う**
5. 🔴 **ショート（赤）** の銘柄 → 今日、日本市場で **空売りする**（または保有中なら売却）
"""
            )
        with col_s2:
            st.markdown(
                """
### ② 日本市場オープン時に発注
**東京時間 9:00 直後**

- 🟢 ロング銘柄の現物 ETF を均等配分で **成行買い**（または寄付近辺の指値）
- 🔴 ショート銘柄は信用売りまたは対応する ETF のインバース商品で対応
- 各銘柄に同額を配分するのが基本（イコールウェイト）
- ポジション数は通常 **ロング 5 銘柄 / ショート 5 銘柄**（q=0.30 時）

> 💡 **注意**: 寄付直後は流動性が低い場合があります。NEXT FUNDS TOPIX-17 ETF は出来高が少ない銘柄もあるため、指値での発注を推奨します。
"""
            )
        with col_s3:
            st.markdown(
                """
### ③ 日本市場クローズ前に決済
**東京時間 15:00〜15:30 頃**

- **全ポジションをクローズ**（ロング銘柄を売却・ショート銘柄を買い戻し）
- この戦略は **1 日完結型**（オーバーナイトポジションは持たない）
- 引け（15:30）に向けて流動性が上がるため、引け前 30 分を目安に決済

> ⚠️ **翌日に持ち越しはしません。** シグナルは翌朝にリセットされるため、毎日クローズが原則です。
"""
            )

        st.markdown("---")

        # ── シグナルの読み方 ──
        st.subheader("🔍 シグナルの読み方")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
**シグナル値とは**

「今日のシグナル」タブの表には、17 の日本業種 ETF が **シグナル値の高い順** に並んでいます。

| 表示 | 意味 | アクション |
|:---:|:---|:---|
| 🟢 ロング | シグナル値が上位 q% | 翌朝オープンで **買い**、引けで **売り** |
| 🔴 ショート | シグナル値が下位 q% | 翌朝オープンで **空売り**、引けで **買い戻し** |
| ─ | 中間 | **取引しない** |

**シグナル値の意味**

米国業種リターン（入力）のパターンが、日本業種 ETF の当日リターンをどれだけ「引っ張る」かを示す値です。値が大きいほど強い上昇シグナル、小さい（負の）ほど強い下落シグナルです。
"""
            )
        with c2:
            st.markdown(
                """
**具体的な読み方の例**

例えば「今日のシグナル」タブでこのような表示が出た場合：

| 順位 | 銘柄 | シグナル値 | 売買 |
|:---:|:---|:---:|:---:|
| 1 | 電機・精密 | +0.082 | 🟢 ロング |
| 2 | 機械 | +0.071 | 🟢 ロング |
| ... | ... | ... | ─ |
| 16 | 電力・ガス | −0.065 | 🔴 ショート |
| 17 | 建設・資材 | −0.078 | 🔴 ショート |

→ **翌朝（日本時間 9:00）に**「電機・精密 ETF (1625.T)」と「機械 ETF (1624.T)」を買い、「電力・ガス ETF (1627.T)」と「建設・資材 ETF (1619.T)」を空売りします。

→ **当日 15:30 までに** すべて反対売買でクローズします。
"""
            )

        st.markdown("---")

        # ── 各タブの説明 ──
        st.subheader("📑 各タブの役割")
        st.markdown(
            """
| タブ | 用途 | 使うタイミング |
|:---|:---|:---|
| 🎯 **今日のシグナル** | 本日の売買指示を確認する | **毎朝**（発注前に必ず確認） |
| 📈 **バックテスト** | 過去の戦略パフォーマンスを検証する | 戦略の信頼性確認・パラメータ調整時 |
| 📊 **データ** | ETF 価格・リターン・相関構造を確認する | 相場環境の把握・銘柄研究時 |
| 🔬 **モデル分析** | アルゴリズム内部の行列・固有ベクトルを可視化する | 戦略の仕組みを深く理解したい時 |
| 📖 **使い方** | このページ（ガイド） | 初回・不明点が生じた時 |
"""
        )

        st.markdown("---")

        # ── リスクと注意事項 ──
        st.subheader("⚠️ リスク・注意事項")
        st.warning(
            "この戦略は学術論文の実装です。過去のバックテスト結果が将来のリターンを保証するものではありません。"
            "実際の取引では市場インパクト・流動性リスク・信用コスト（ショート時の貸株料）・税金が発生します。"
            "投資は自己責任で行ってください。"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
**運用上の注意点**
- NEXT FUNDS TOPIX-17 ETF（1617.T〜1633.T）は**流動性が低い**銘柄があります。大きな資金では執行コストが増大します。
- ショートポジションには**信用口座**が必要です。
- シグナルは**前日の米国クローズ後のデータ**に基づくため、当日の米国市場イベントは反映されません。
- データ更新が遅れた場合、シグナルの日付を必ず確認してから発注してください。
"""
            )
        with c2:
            st.markdown(
                """
**パラメータの影響**
- **q（分位点）を上げる** → 売買銘柄数が増え、分散は上がるが 1 銘柄あたりの期待リターンは下がる傾向
- **L（ウィンドウ）を短くする** → 最近のデータを重視するが推定ノイズが増える
- **λ（正則化）を上げる** → 事前知識への依存度が高まり安定するが、直近の相場変化への追従が遅れる
- 論文推奨値（L=60, λ=0.9, K=3, q=0.30）から大きく外れる場合はバックテストで事前確認を
"""
            )

        st.markdown("---")

        # ── アルゴリズム解説 ──
        st.subheader("🔧 戦略のしくみ（上級者向け）")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
**リードラグ仮説**

日米の取引時間帯の**非同期性**を活用した投資戦略：

- 米国市場（ET 夕方クローズ）→ 翌朝、日本市場オープン
- **情報源**: 米国 11 業種 ETF の当日 Close-to-Close リターン
- **予測対象**: 日本 17 業種 ETF の翌日 Open-to-Close リターン

米国で「テクノロジー業種が上昇した日」の翌朝、日本の「電機・精密業種」が上昇する傾向があるという共分散構造を PCA で抽出します。
"""
            )
        with c2:
            st.markdown(
                r"""
**部分空間正則化 PCA** (式 13–21):

| ステップ | 内容 |
|:---|:---|
| 1. 事前部分空間 | $v_1$: グローバル, $v_2$: 国スプレッド, $v_3$: シクリカル/DF |
| 2. 正則化 | $C^{\rm reg}_t = (1-\lambda)C_t + \lambda C_0$ |
| 3. 固有分解 | 上位 $K$ 固有ベクトルを抽出 |
| 4. シグナル | $\hat{z}_{J,t+1} = V_{J}^{(K)} \bigl(V_{U}^{(K)\top} z_{U,t}\bigr)$ |
| 5. 売買 | 上位 $q$% ロング / 下位 $q$% ショート |
"""
            )

        st.subheader("比較戦略")
        st.dataframe(
            pd.DataFrame(
                {
                    "戦略": ["MOM", "PCA PLAIN", "PCA SUB(論文)", "DOUBLE"],
                    "説明": [
                        "日本業種のウィンドウ内平均リターン（単純モメンタム）",
                        "正則化なし PCA (λ=0) によるリードラグシグナル",
                        "部分空間正則化付き PCA (λ=0.9) によるシグナル",
                        "MOM と PCA SUB の 2×2 ダブルソート",
                    ],
                    "論文の結果（AR / R/R）": [
                        "5.63% / 0.53",
                        "6.24% / 0.62",
                        "**23.79% / 2.22**",
                        "18.86% / 1.69",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    # ═══════════════════════════════════════════════════
    # データ確認
    # ═══════════════════════════════════════════════════
    with tab_data:
        st.caption(
            "yfinance で取得した日米業種 ETF の価格データ・リターン分布・相関構造を確認できます。"
        )
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("米国業種 ETF 累積リターン (CC)")
            st.caption(
                "SPDR 11 業種 ETF の Close-to-Close（前日終値→当日終値）リターンを累積した推移です。戦略の入力シグナル源となるデータです。"
            )
            cum_us = (1 + us_cc[US_TICKERS]).cumprod()
            fig = go.Figure()
            for t in US_TICKERS:
                if t in cum_us.columns:
                    fig.add_trace(
                        go.Scatter(x=cum_us.index, y=cum_us[t], name=US_LABEL[t], mode="lines")
                    )
            fig.update_layout(
                height=380, yaxis_title="累積リターン", legend=dict(font=dict(size=9))
            )
            st.plotly_chart(fig, width="stretch")

        with c2:
            st.subheader("日本業種 ETF 累積リターン (OC)")
            st.caption(
                "NEXT FUNDS TOPIX-17 ETF の Open-to-Close（寄付→引け）リターンを累積した推移です。戦略が予測・売買するターゲットデータです。"
            )
            cum_jp = (1 + jp_oc[JP_TICKERS]).cumprod()
            fig = go.Figure()
            for t in JP_TICKERS:
                if t in cum_jp.columns:
                    fig.add_trace(
                        go.Scatter(x=cum_jp.index, y=cum_jp[t], name=JP_LABEL[t], mode="lines")
                    )
            fig.update_layout(
                height=380, yaxis_title="累積リターン", legend=dict(font=dict(size=9))
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("日米業種間相関行列（CC ベース）")
        st.caption(
            "米国 11 業種 × 日本 17 業種の 28×28 全期間相関行列です。赤が正の相関、青が負の相関を示します。右上の米日ブロックがリードラグ構造の可視化です。"
        )
        all_cc = us_cc[US_TICKERS].join(jp_cc[JP_TICKERS], how="inner").dropna()
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
                    "年率リターン(%)": (us_cc[US_TICKERS].mean() * 252 * 100).round(2),
                    "年率ボラ(%)": (us_cc[US_TICKERS].std() * np.sqrt(252) * 100).round(2),
                    "Sharpe": (
                        us_cc[US_TICKERS].mean() / us_cc[US_TICKERS].std() * np.sqrt(252)
                    ).round(2),
                    "Skew": us_cc[US_TICKERS].skew().round(2),
                    "Kurt": us_cc[US_TICKERS].kurt().round(2),
                }
            )
            us_st.index = [US_LABEL[t] for t in US_TICKERS]
            st.dataframe(us_st, width="stretch")

        with c2:
            st.write("**日本業種 ETF（OC ベース）**")
            jp_st = pd.DataFrame(
                {
                    "年率リターン(%)": (jp_oc[JP_TICKERS].mean() * 252 * 100).round(2),
                    "年率ボラ(%)": (jp_oc[JP_TICKERS].std() * np.sqrt(252) * 100).round(2),
                    "Sharpe": (
                        jp_oc[JP_TICKERS].mean() / jp_oc[JP_TICKERS].std() * np.sqrt(252)
                    ).round(2),
                    "Skew": jp_oc[JP_TICKERS].skew().round(2),
                    "Kurt": jp_oc[JP_TICKERS].kurt().round(2),
                }
            )
            jp_st.index = [JP_LABEL[t] for t in JP_TICKERS]
            st.dataframe(jp_st, width="stretch")

    # ═══════════════════════════════════════════════════
    # シグナル分析
    # ═══════════════════════════════════════════════════
    with tab_sig:
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

    # ═══════════════════════════════════════════════════
    # バックテスト
    # ═══════════════════════════════════════════════════
    with tab_bt:
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
                        prog.progress(
                            0.5 + step / total * 0.5, text=f"新戦略を計算中… {step}/{total}"
                        )

                    rets_p = run_backtest(
                        us_cc, jp_cc, jp_oc, L=L, lam=lam, K=K, q=q, on_progress=on_progress_p
                    )
                    rets_e = run_backtest_loop(
                        us_cc,
                        jp_cc,
                        jp_oc,
                        L=L,
                        lam=lam,
                        K=K,
                        q=q,
                        cfull_end="2014-12-31",
                        strategies={"PCA_SUB": _exp_return},
                        on_progress=on_progress_e,
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
        else:
            rets_p: pd.DataFrame = st.session_state["rets_paper"]
            rets_e: pd.DataFrame = st.session_state["rets_exp"]

            # 論文戦略: 表示名をアンダースコアなしに整形、PCA_SUB のみ "(論文)" 付き
            _RENAME_P = {
                "MOM": "MOM",
                "PCA_PLAIN": "PCA PLAIN",
                "PCA_SUB": "PCA SUB(論文)",
                "DOUBLE": "DOUBLE",
            }
            # 実験戦略: PCA_SUB のみ取り出して "(改)" 付き
            rets_all = rets_p.rename(columns=_RENAME_P).join(
                rets_e[["PCA_SUB"]].rename(columns={"PCA_SUB": "PCA SUB(改)"}), how="outer"
            )

            # (表示名, 色, 線種)
            _COMBINED_CFG = [
                ("PCA SUB(論文)", "blue", "solid"),
                ("PCA SUB(改)", "cornflowerblue", "dash"),
                ("DOUBLE", "green", "solid"),
                ("PCA PLAIN", "orange", "solid"),
                ("MOM", "red", "solid"),
            ]

            # ── パフォーマンス指標 ──
            st.subheader("パフォーマンス指標")
            st.caption(
                "MOM・PCA PLAIN・DOUBLE は比較用ベースライン（改版なし）。PCA SUB（提案手法）は論文版と改版を並べて比較します。"
            )
            metrics = perf_metrics(rets_all)
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
            for key in ["PCA SUB(論文)", "PCA SUB(改)", "DOUBLE", "PCA PLAIN", "MOM"]:
                if key not in metrics.index:
                    continue
                row = metrics.loc[key]
                with st.container(border=True):
                    label_col, ar_col, risk_col, rr_col, mdd_col = st.columns([2, 1, 1, 1, 1])
                    label_col.metric(label=key, value="", help=_STRAT_HELP.get(key, ""))
                    ar_col.metric("AR (%)", f"{row['AR(%)']:.2f}", help=_METRIC_HELP["AR(%)"])
                    risk_col.metric(
                        "RISK (%)", f"{row['RISK(%)']:.2f}", help=_METRIC_HELP["RISK(%)"]
                    )
                    rr_col.metric("R/R", f"{row['R/R']:.2f}", help=_METRIC_HELP["R/R"])
                    mdd_col.metric("MDD (%)", f"{row['MDD(%)']:.2f}", help=_METRIC_HELP["MDD(%)"])

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
            _MONTH_COLS = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]

            def _monthly_pivot(series: pd.Series) -> pd.DataFrame:
                m = series.dropna().resample("ME").sum() * 100
                df_m = pd.DataFrame({"ret": m, "Y": m.index.year, "M": m.index.month})
                piv = df_m.pivot(index="Y", columns="M", values="ret")
                piv.columns = _MONTH_COLS[: len(piv.columns)]
                return piv

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
                        fig_m.update_layout(
                            height=430, title=col_name, xaxis_title="月", yaxis_title="年"
                        )
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

    # ═══════════════════════════════════════════════════
    # 今日のシグナル
    # ═══════════════════════════════════════════════════
    with tab_today:
        today_mode_label = "🧪 新戦略（実験中）" if use_exp else "📄 論文戦略"
        st.subheader(f"🎯 今日の売買シグナル（PCA SUB） — {today_mode_label}")
        st.caption(
            "サイドバーの終了日に含まれる最新米国取引日のリターンを使用してシグナルを計算します。"
            "米国市場クローズ（東京時間 5:00〔夏〕/ 6:00〔冬〕）後、東証オープン（9:00）までに確認してください。"
        )

        try:
            sig_result = (
                compute_live_signal(
                    us_cc,
                    jp_cc,
                    jp_oc,
                    L=L,
                    lam=lam,
                    K=K,
                    q=q,
                    cfull_end="2014-12-31",
                    signal_fn=_exp_signal,
                )
                if use_exp
                else compute_today_signal(us_cc, jp_cc, jp_oc, L=L, lam=lam, K=K, q=q)
            )
        except Exception as exc:
            st.error(f"シグナル計算に失敗しました: {exc}")
        else:
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
                df_sig.style.apply(_highlight_action, axis=1),
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


if __name__ == "__main__":
    main()
