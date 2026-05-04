# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 言語

ユーザーへの応答は原則**日本語**で行うこと。コード中のコメントや変数名は既存の慣習に従う。

## 概要

Nakagawa et al. (2026), JSAI SIG-FIN-036 の論文を実装・可視化する Streamlit ダッシュボード。日米結合相関行列に部分空間正則化付き PCA を適用し、米国業種 ETF の当日 Close-to-Close リターンから翌営業日の日本業種 ETF Open-to-Close リターンを予測する。

実装の詳細や数式を確認する際は、`references/lead-lag-strategies-for-japanese-and-us-sectors-using-subspace-legularization-pca.pdf` を参照すること。

## コマンド

```bash
# 仮想環境作成（初回のみ）
python -m venv .venv

# 仮想環境の有効化
source .venv/bin/activate

# 依存インストール
pip install -r requirements.txt

# ダッシュボード起動（http://localhost:8501）
python -m streamlit run main.py

# テスト実行
pytest tests/

# 特定テストのみ実行
pytest tests/test_strategy.py::TestBuildV0::test_shape

# リント（pyproject.toml に ruff 設定あり）
ruff check .
```

初回起動時は yfinance 経由で ETF データをダウンロードするため数分かかる場合がある。ダウンロード済みデータは `data/` に parquet 形式でキャッシュされる（`us_close.parquet`, `jp_close.parquet`, `jp_open.parquet`）。リクエスト範囲がキャッシュに収まっていればネットワークアクセスは発生しない。Streamlit 側は `@st.cache_data(ttl=3600)` でセッション中追加キャッシュ。

## アーキテクチャ

3 ファイル構成。`main.py` がエントリポイント、`src/` 配下に計算ロジックと UI を分離している。

### `main.py` — エントリポイント

`src.dashboard.main` を呼び出す 2 行だけのラッパー。`streamlit run main.py` で起動する。

### `src/strategy.py` — 計算ロジック（Streamlit に非依存）

定数（ティッカー・ラベル・分類）と以下の関数を含む。

1. **`load_data(start, end)`** — yfinance で米国（SPDR 11 銘柄）と日本（TOPIX-17、`1617.T`〜`1633.T`）の ETF 価格を取得。`us_cc`（米 CC リターン）・`jp_cc`（日 CC リターン、相関推定用）・`jp_oc`（日 OC リターン、予測ターゲット）・`us_close`（米終値）・`jp_close`（日終値）の 5 系列を返す。

2. **`build_V0()`** — N×3（N=28）の事前部分空間行列を構築。3 つの直交正規ベクトル：v₁ グローバルファクター、v₂ 米+/日− 国スプレッド、v₃ シクリカル/ディフェンシブ。正則化のアンカーとして機能する。

3. **`build_C0(V0, Cfull)`** — 長期相関行列（`Cfull`、2015 年以前で推定）を事前部分空間に射影し、事前エクスポージャー行列 C₀ を構築する。

4. **`run_backtest(..., on_progress=None)`** — メインループ。米国取引日 t ごとに対応する次の日本取引日を特定し：
   - L 日ローリングウィンドウで相関行列 `Ct` を推定
   - `C_reg = (1−λ)·Ct + λ·C₀` を形成
   - 上位 K 固有ベクトル `Vk` を固有分解で取得
   - リードラグシグナル `z_JP = Vk[JP] @ (Vk[US].T @ z_US_today)` を計算
   - MOM・PCA_PLAIN・PCA_SUB・DOUBLE の 4 戦略のロングショートリターンを算出
   - 日次リターンの DataFrame を返す
   - `on_progress(step, total)` はオプションのプログレス通知コールバック

5. **`compute_today_signal(...)`** — 最新の米国リターンから日本業種 ETF の当日売買シグナルを計算する。`signal`（JP_TICKERS の z スコア Series）・`us_date`・`next_jp_date`・`us_returns`・`n_long`・`n_short` を含む dict を返す。バックテストを実行せず、リアルタイム運用向けに「今日のシグナル」タブで使用。

6. **`perf_metrics(rets)`** — AR（年率リターン）・RISK・R/R（リターン/リスク比）・MDD を計算する。

### `src/dashboard.py` — Streamlit UI

`src.strategy` の関数をインポートして使用する。

- `_load_data(start, end)` — `load_data` に `@st.cache_data(ttl=3600)` を適用したラッパー
- `main()` — サイドバーでパラメータ（開始/終了日・L・λ・K・q）を設定し、5 タブ（🎯 今日のシグナル・📈 バックテスト・📊 データ・🔬 モデル分析・📖 使い方）を表示する。`run_backtest` には `on_progress` コールバック経由でプログレスバーを渡す。「今日のシグナル」タブは `compute_today_signal` を使用。

## 主要パラメータ（論文の設定値）

| パラメータ | デフォルト | 意味 |
|---|---|---|
| L | 60 | ローリング推定ウィンドウ（営業日） |
| λ (lam) | 0.9 | 事前 C₀ への正則化強度 |
| K | 3 | 抽出する主成分数 |
| q | 0.30 | ロング/ショートの分位点比率 |

## データ詳細

- 米国ティッカー: `XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY`
- 日本ティッカー: `1617.T`〜`1633.T`（NEXT FUNDS TOPIX-17 業種別 ETF）
- `Cfull`（事前相関行列）は 2014-12-31 以前のデータで推定。データ不足時は先頭 500 行にフォールバック
- 日付ペアリング: 各米国取引日に対し次の日本取引日を対応させる（祝日は簡略処理）
- バックテスト結果は `st.session_state["rets"]` に保持。ボタン押下で手動実行
