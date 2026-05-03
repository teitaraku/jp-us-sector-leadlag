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

from strategy import (
    JP_LABEL,
    JP_TICKERS,
    NJ,
    NU,
    STRAT_COLORS,
    STRAT_DISP,
    US_LABEL,
    US_TICKERS,
    build_C0,
    build_V0,
    compute_today_signal,
    load_data,
    perf_metrics,
    run_backtest,
)

st.set_page_config(
    page_title="日米業種リードラグ投資戦略",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_data(start: str, end: str):
    return load_data(start, end)


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

        st.markdown("---")
        start = st.date_input("開始日", value=pd.Timestamp("2010-01-01")).strftime("%Y-%m-%d")
        end   = st.date_input("終了日", value=pd.Timestamp.today()).strftime("%Y-%m-%d")
        L   = st.slider("推定ウィンドウ L（営業日）", 20, 252, 60, 5, key="L")
        lam = st.slider("正則化パラメータ λ", 0.0, 1.0, 0.9, 0.05, key="lam")
        K   = st.slider("主成分数 K", 1, 5, 3, key="K")
        q   = st.slider("分位点 q（ロング/ショート比率）", 0.10, 0.45, 0.30, 0.05, key="q")

        st.markdown("---")
        st.markdown("**論文パラメータ**: L=60, λ=0.9, K=3, q=0.3")
        if st.button("論文パラメータにリセット", use_container_width=True):
            st.session_state["L"]   = 60
            st.session_state["lam"] = 0.9
            st.session_state["K"]   = 3
            st.session_state["q"]   = 0.30
            st.rerun()

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
        st.caption("初めての方はここから読んでください。毎日の操作手順から戦略の仕組みまで解説します。")

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
| 16 | 公益事業 | −0.065 | 🔴 ショート |
| 17 | 建設・資材 | −0.078 | 🔴 ショート |

→ **翌朝（日本時間 9:00）に**「電機・精密 ETF (1626.T)」と「機械 ETF (1622.T)」を買い、「公益事業 ETF (1631.T)」と「建設・資材 ETF (1618.T)」を空売りします。

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
            pd.DataFrame({
                "戦略": ["MOM", "PCA PLAIN", "PCA SUB（提案）", "DOUBLE"],
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
            }),
            hide_index=True,
            width='stretch',
        )

    # ═══════════════════════════════════════════════════
    # データ確認
    # ═══════════════════════════════════════════════════
    with tab_data:
        st.caption("yfinance で取得した日米業種 ETF の価格データ・リターン分布・相関構造を確認できます。")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("米国業種 ETF 累積リターン (CC)")
            cum_us = (1 + us_cc[US_TICKERS]).cumprod()
            fig = go.Figure()
            for t in US_TICKERS:
                if t in cum_us.columns:
                    fig.add_trace(go.Scatter(x=cum_us.index, y=cum_us[t], name=US_LABEL[t], mode="lines"))
            fig.update_layout(height=380, yaxis_title="累積リターン", legend=dict(font=dict(size=9)))
            st.plotly_chart(fig, width='stretch')

        with c2:
            st.subheader("日本業種 ETF 累積リターン (OC)")
            cum_jp = (1 + jp_oc[JP_TICKERS]).cumprod()
            fig = go.Figure()
            for t in JP_TICKERS:
                if t in cum_jp.columns:
                    fig.add_trace(go.Scatter(x=cum_jp.index, y=cum_jp[t], name=JP_LABEL[t], mode="lines"))
            fig.update_layout(height=380, yaxis_title="累積リターン", legend=dict(font=dict(size=9)))
            st.plotly_chart(fig, width='stretch')

        st.subheader("日米業種間相関行列（CC ベース）")
        all_cc = us_cc[US_TICKERS].join(jp_cc[JP_TICKERS], how="inner").dropna()
        if len(all_cc) > 0:
            corr = all_cc.corr()
            labels_all = [US_LABEL[t] for t in US_TICKERS] + [JP_LABEL[t] for t in JP_TICKERS]
            fig = go.Figure(
                go.Heatmap(
                    z=corr.values, x=labels_all, y=labels_all,
                    colorscale="RdBu_r", zmid=0,
                    text=corr.values.round(2), texttemplate="%{text}",
                    textfont=dict(size=6),
                )
            )
            fig.update_layout(
                height=620,
                xaxis=dict(tickangle=45, tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8)),
            )
            st.plotly_chart(fig, width='stretch')

        st.subheader("基本統計量")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**米国業種 ETF（CC ベース）**")
            us_st = pd.DataFrame({
                "年率リターン(%)": (us_cc[US_TICKERS].mean() * 252 * 100).round(2),
                "年率ボラ(%)":     (us_cc[US_TICKERS].std() * np.sqrt(252) * 100).round(2),
                "Sharpe":         (us_cc[US_TICKERS].mean() / us_cc[US_TICKERS].std() * np.sqrt(252)).round(2),
                "Skew":           us_cc[US_TICKERS].skew().round(2),
                "Kurt":           us_cc[US_TICKERS].kurt().round(2),
            })
            us_st.index = [US_LABEL[t] for t in US_TICKERS]
            st.dataframe(us_st, width='stretch')

        with c2:
            st.write("**日本業種 ETF（OC ベース）**")
            jp_st = pd.DataFrame({
                "年率リターン(%)": (jp_oc[JP_TICKERS].mean() * 252 * 100).round(2),
                "年率ボラ(%)":     (jp_oc[JP_TICKERS].std() * np.sqrt(252) * 100).round(2),
                "Sharpe":         (jp_oc[JP_TICKERS].mean() / jp_oc[JP_TICKERS].std() * np.sqrt(252)).round(2),
                "Skew":           jp_oc[JP_TICKERS].skew().round(2),
                "Kurt":           jp_oc[JP_TICKERS].kurt().round(2),
            })
            jp_st.index = [JP_LABEL[t] for t in JP_TICKERS]
            st.dataframe(jp_st, width='stretch')

    # ═══════════════════════════════════════════════════
    # シグナル分析
    # ═══════════════════════════════════════════════════
    with tab_sig:
        st.caption("アルゴリズム内部で使用する事前固有ベクトル V₀・エクスポージャー行列 C₀・正則化相関行列 C_reg を可視化します。")
        V0 = build_V0()
        labels_all = [US_LABEL[t] for t in US_TICKERS] + [JP_LABEL[t] for t in JP_TICKERS]
        bar_colors = ["#1f77b4"] * NU + ["#d62728"] * NJ
        factor_titles = ["v₁: グローバル", "v₂: 国スプレッド", "v₃: シクリカル/DF"]

        st.subheader("事前固有ベクトル V₀（青=米国, 赤=日本）")
        fig_v0 = make_subplots(rows=1, cols=3, subplot_titles=factor_titles)
        for k in range(3):
            fig_v0.add_trace(
                go.Bar(x=labels_all, y=V0[:, k], marker_color=bar_colors, showlegend=False),
                row=1, col=k + 1,
            )
        fig_v0.update_xaxes(tickangle=45, tickfont=dict(size=7))
        fig_v0.update_layout(height=380)
        st.plotly_chart(fig_v0, width='stretch')

        all_cc_full = us_cc[US_TICKERS].join(jp_cc[JP_TICKERS], how="inner").dropna()
        cfull_data = all_cc_full[all_cc_full.index < "2015-01-01"]
        if len(cfull_data) < 100:
            cfull_data = all_cc_full.iloc[:500]
        Cfull = np.nan_to_num(cfull_data.corr().values)
        np.fill_diagonal(Cfull, 1.0)
        C0 = build_C0(V0, Cfull)

        st.subheader("事前エクスポージャー行列 C₀")
        fig_c0 = go.Figure(
            go.Heatmap(
                z=C0, x=labels_all, y=labels_all,
                colorscale="RdBu_r", zmid=0,
                text=C0.round(2), texttemplate="%{text}",
                textfont=dict(size=6),
            )
        )
        fig_c0.update_layout(
            height=600,
            xaxis=dict(tickangle=45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8)),
        )
        st.plotly_chart(fig_c0, width='stretch')

        st.subheader("正則化相関行列の比較（直近ウィンドウ）")
        recent = all_cc_full.iloc[-L:]
        if len(recent) >= 10:
            mu_r = recent.mean().values
            sig_r = recent.std().values
            sig_r = np.where(sig_r < 1e-10, 1e-10, sig_r)
            z_r = np.nan_to_num((recent.values - mu_r) / sig_r)
            Ct_recent = np.corrcoef(z_r.T)
            np.fill_diagonal(Ct_recent, 1.0)
            C_reg_recent = (1 - lam) * Ct_recent + lam * C0

            c1, c2 = st.columns(2)
            for col, mat, title in [
                (c1, Ct_recent,    f"C_t（直近 {L} 日）"),
                (c2, C_reg_recent, f"C_reg（λ={lam}）"),
            ]:
                with col:
                    fig = go.Figure(
                        go.Heatmap(
                            z=mat, x=labels_all, y=labels_all,
                            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                        )
                    )
                    fig.update_layout(
                        height=500, title=title,
                        xaxis=dict(tickangle=45, tickfont=dict(size=7)),
                        yaxis=dict(tickfont=dict(size=7)),
                    )
                    st.plotly_chart(fig, width='stretch')

    # ═══════════════════════════════════════════════════
    # バックテスト
    # ═══════════════════════════════════════════════════
    with tab_bt:
        st.caption("指定した期間・パラメータで 4 戦略のパフォーマンスを検証します。「バックテスト実行」を押すと計算が始まります。")
        if st.button("🚀 バックテスト実行", type="primary"):
            with st.spinner("計算中…"):
                try:
                    prog = st.progress(0.0, text="バックテスト実行中…")

                    def on_progress(step: int, total: int) -> None:
                        prog.progress(step / total, text=f"バックテスト実行中… {step}/{total}")

                    rets = run_backtest(
                        us_cc, jp_cc, jp_oc, L=L, lam=lam, K=K, q=q,
                        on_progress=on_progress,
                    )
                    prog.empty()
                    st.session_state["rets"] = rets
                    st.success(f"完了！ {len(rets)} 日分のリターンを計算しました。")
                except Exception as exc:
                    st.error(f"エラー: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

        if "rets" not in st.session_state:
            st.info("「バックテスト実行」ボタンを押してください。")
            return

        rets: pd.DataFrame = st.session_state["rets"]

        # ── パフォーマンス指標 ──
        st.subheader("パフォーマンス指標")
        metrics = perf_metrics(rets)
        metrics.index = [STRAT_DISP.get(i, i) for i in metrics.index]

        def color_best(s: pd.Series):
            if s.name in ("AR(%)", "R/R"):
                best = s.max()
            elif s.name == "MDD(%)":
                best = s.min()
            else:
                return [""] * len(s)
            return ["background-color:#c8f7c5" if v == best else "" for v in s]

        st.dataframe(metrics.style.apply(color_best), width='stretch')

        # ── 累積リターン ──
        st.subheader("累積リターン推移")
        cum = (1 + rets).cumprod()
        fig_cum = go.Figure()
        for col in ["PCA_SUB", "DOUBLE", "PCA_PLAIN", "MOM"]:
            if col in cum.columns:
                fig_cum.add_trace(
                    go.Scatter(
                        x=cum.index, y=cum[col], name=STRAT_DISP[col],
                        mode="lines", line=dict(color=STRAT_COLORS[col], width=2),
                    )
                )
        fig_cum.update_layout(
            height=450, yaxis_title="累積リターン",
            xaxis_title="日付", hovermode="x unified",
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_cum, width='stretch')

        # ── ドローダウン ──
        st.subheader("ドローダウン")
        fig_dd = go.Figure()
        for col in ["PCA_SUB", "DOUBLE", "PCA_PLAIN", "MOM"]:
            if col in rets.columns:
                r = rets[col].dropna()
                cum_r = (1 + r).cumprod()
                dd = (cum_r - cum_r.cummax()) / cum_r.cummax() * 100
                fig_dd.add_trace(
                    go.Scatter(
                        x=dd.index, y=dd, name=STRAT_DISP[col],
                        mode="lines", line=dict(color=STRAT_COLORS[col]),
                    )
                )
        fig_dd.update_layout(
            height=350, yaxis_title="ドローダウン (%)",
            xaxis_title="日付", hovermode="x unified",
        )
        st.plotly_chart(fig_dd, width='stretch')

        # ── 月次リターン・ヒートマップ（PCA SUB） ──
        if "PCA_SUB" in rets.columns:
            st.subheader("PCA SUB（提案）月次リターン (%)")
            monthly = rets["PCA_SUB"].dropna().resample("ME").sum() * 100
            df_m = pd.DataFrame({
                "ret": monthly,
                "Y": monthly.index.year,
                "M": monthly.index.month,
            })
            pivot = df_m.pivot(index="Y", columns="M", values="ret")
            pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            fig_m = go.Figure(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale="RdYlGn", zmid=0,
                    text=pivot.values.round(1), texttemplate="%{text}",
                    colorbar=dict(title="%"),
                )
            )
            fig_m.update_layout(height=430, xaxis_title="月", yaxis_title="年")
            st.plotly_chart(fig_m, width='stretch')

        # ── ローリング・シャープレシオ ──
        st.subheader("ローリング・シャープレシオ（252 営業日）")
        fig_sh = go.Figure()
        for col in ["PCA_SUB", "DOUBLE", "PCA_PLAIN", "MOM"]:
            if col in rets.columns:
                r = rets[col].dropna()
                rs = r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
                fig_sh.add_trace(
                    go.Scatter(
                        x=rs.index, y=rs, name=STRAT_DISP[col],
                        mode="lines", line=dict(color=STRAT_COLORS[col]),
                    )
                )
        fig_sh.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_sh.update_layout(
            height=350, yaxis_title="シャープレシオ（年率）",
            xaxis_title="日付", hovermode="x unified",
        )
        st.plotly_chart(fig_sh, width='stretch')

        # ── 年次リターン比較 ──
        st.subheader("年次リターン比較 (%)")
        annual = rets.resample("YE").sum() * 100
        annual.index = annual.index.year
        fig_ann = go.Figure()
        for col in ["PCA_SUB", "DOUBLE", "PCA_PLAIN", "MOM"]:
            if col in annual.columns:
                fig_ann.add_trace(
                    go.Bar(
                        x=annual.index, y=annual[col], name=STRAT_DISP[col],
                        marker_color=STRAT_COLORS[col], opacity=0.8,
                    )
                )
        fig_ann.update_layout(
            height=380, barmode="group",
            yaxis_title="年次リターン (%)", xaxis_title="年",
        )
        st.plotly_chart(fig_ann, width='stretch')


    # ═══════════════════════════════════════════════════
    # 今日のシグナル
    # ═══════════════════════════════════════════════════
    with tab_today:
        st.subheader("🎯 今日の売買シグナル（PCA SUB）")
        st.caption(
            "サイドバーの終了日に含まれる最新米国取引日のリターンを使用してシグナルを計算します。"
            "米国市場クローズ（東京時間 5:00〔夏〕/ 6:00〔冬〕）後、東証オープン（9:00）までに確認してください。"
        )

        try:
            sig_result = compute_today_signal(
                us_cc, jp_cc, jp_oc, L=L, lam=lam, K=K, q=q
            )
        except Exception as exc:
            st.error(f"シグナル計算に失敗しました: {exc}")
        else:
            us_date     = sig_result["us_date"]
            signal      = sig_result["signal"]
            us_returns  = sig_result["us_returns"]
            n_long      = sig_result["n_long"]
            n_short     = sig_result["n_short"]
            next_jp     = sig_result["next_jp_date"]

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
                rows.append({
                    "順位": rank,
                    "銘柄": JP_LABEL[ticker],
                    "コード": ticker,
                    "シグナル値": round(float(val), 4),
                    "売買": action,
                })

            df_sig = pd.DataFrame(rows)

            def highlight_action(row):
                if "ロング" in str(row["売買"]):
                    return ["background-color:#c8f7c5; color:#000000"] * len(row)
                if "ショート" in str(row["売買"]):
                    return ["background-color:#f7c8c8; color:#000000"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_sig.style.apply(highlight_action, axis=1),
                hide_index=True,
                width='stretch',
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
            st.plotly_chart(fig_sig, width='stretch')

            # ── 米国入力リターン ──
            st.subheader(f"米国業種リターン（入力）: {us_date.strftime('%Y-%m-%d')}")
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
            st.plotly_chart(fig_us, width='stretch')


if __name__ == "__main__":
    main()
