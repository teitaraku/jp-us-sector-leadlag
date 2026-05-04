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

`main.py` がエントリポイント、`src/strategies/` に計算ロジック、`src/tabs/` に UI を分離している。

### `main.py` — エントリポイント

`src.dashboard.main` を呼び出す 2 行だけのラッパー。

### `src/dashboard.py` — Streamlit UI ルート

- サイドバーでパラメータ（開始/終了日・L・λ・K・q）を設定し、5 タブを表示する
- データ取得は `_load_data`（`@st.cache_data(ttl=3600)` ラッパー）で行う
- 各タブのレンダリングは `src/tabs/` の各モジュールに委譲する

### `src/strategies/` — 計算ロジック（Streamlit 非依存）

#### `core.py` — 共通基盤

定数（ティッカー・ラベル・分類）・`StepContext` データクラス・共通関数を含む。

- **`load_data(start, end)`** — yfinance でデータ取得。`us_cc`・`jp_cc`・`jp_oc`・`us_close`・`jp_close` の 5 系列を返す。キャッシュが部分的に古い場合は差分のみ追加ダウンロードする。
- **`build_V0()`** — N×3（N=28）の事前部分空間行列。v₁ グローバルファクター・v₂ 米+/日− 国スプレッド・v₃ シクリカル/ディフェンシブ の 3 直交正規ベクトル。
- **`build_C0(V0, Cfull)`** — 長期相関行列 `Cfull` を事前部分空間に射影し、事前エクスポージャー行列 C₀ を構築する。
- **`run_backtest_loop(..., strategies: dict[str, Callable[[StepContext], float]])`** — 共通バックテストループ。戦略名→関数の辞書を受け取り、全戦略のロングショートリターン系列を返す。`Cfull` は `cfull_end`（論文版: 2014-12-31）以前のデータで推定する。
- **`compute_live_signal(..., signal_fn: Callable[[StepContext], np.ndarray])`** — 任意の戦略関数で最新シグナルを計算する（「今日のシグナル」タブ用）。
- **`perf_metrics(rets)`** — AR・RISK・R/R・MDD を計算する。

#### `StepContext` — 1 ステップの計算済み中間量

各戦略関数が受け取るデータクラス。`mu`・`sigma`・`z_us`（米国 z スコア）・`Ct`（当日相関行列）・`C0`（事前相関行列）・`target`（バックテスト時のみ）・`K`・`lam`・`q` を持つ。

#### 各戦略モジュール

それぞれ `compute_signal(ctx) -> np.ndarray`・`compute_return(ctx) -> float`・`run_backtest(...)`・`run_live_signal(...)` を実装する。

| モジュール | 戦略 | 概要 |
|---|---|---|
| `pca_sub.py` | PCA SUB（論文） | 手動構築 V₀ による部分空間正則化 PCA。提案手法。`cfull_end="2014-12-31"` 固定。 |
| `pca_plain.py` | PCA PLAIN | 正則化なし PCA（λ=0 相当）。ベースライン。 |
| `mom.py` | MOM | 米国リターンをそのまま日本シグナルに使うモメンタム。ベースライン。 |
| `double.py` | DOUBLE | MOM と PCA SUB の両シグナルで 2 段階スクリーニングする複合戦略。ベースライン。 |
| `exp.py` | PCA SUB（改） | **データ駆動型動的 V₀**。手動ラベルの代わりに直近 2 年の PCA で V₀ を推定。暦年が変わるたびに V₀・C₀ を前 5 年データで再推定するため先読みバイアスがない。論文版を改変する場合はこのファイルを編集する。 |

### `src/tabs/` — 各タブの UI

| モジュール | タブ | 説明 |
|---|---|---|
| `today.py` | 🎯 今日のシグナル | 「論文版」または「改良版」を選択して最新シグナルを表示 |
| `backtest.py` | 📈 バックテスト | 論文戦略（4 種）と改良版を同時実行・比較 |
| `data.py` | 📊 データ | 価格・リターンの閲覧 |
| `analysis.py` | 🔬 モデル分析 | 相関行列・固有ベクトルの可視化 |
| `howto.py` | 📖 使い方 | 手法説明 |
| `common.py` | — | 共通スタイル・ヘルパー（`highlight_action`・`color_best`・`TAB_CSS`） |

バックテスト結果は `st.session_state["rets_paper"]`（論文戦略）と `st.session_state["rets_exp"]`（改良版）に保持。ボタン押下で手動実行する。

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
- `Cfull`（事前相関行列）は 2010-01-01〜2014-12-31 のデータで推定（論文版）。`exp.py` は基準日直前 5 年間でローリング推定する。
- 日付ペアリング: 各米国取引日に対し次の日本取引日を対応させる（祝日は簡略処理）

## 新しい戦略を追加する手順

1. `src/strategies/` に新モジュールを作成し、`compute_signal(ctx)`・`compute_return(ctx)`・`run_backtest(...)`・`run_live_signal(...)` を実装する
2. `src/tabs/backtest.py` の `_PAPER_STRATEGIES` または別変数に追加し、`run_backtest_loop` に渡す
3. `src/tabs/today.py` に選択肢を追加する
