# Repository Guidelines

## プロジェクト構成とモジュール

このリポジトリは、日米業種リードラグ投資戦略を実装し、Streamlit で可視化する Python プロジェクトです。`main.py` はエントリポイントで、`src.dashboard.main()` を呼び出します。Streamlit UI のルートは `src/dashboard.py` にあり、タブ単位の UI は `src/tabs/` に分離されています。戦略計算の中核は `src/strategies/core.py` にあり、個別戦略は `src/strategies/pca_sub.py`、`pca_plain.py`、`mom.py`、`double.py`、`exp.py` に分割されています。戦略計算は Streamlit に依存しない形で保守してください。テストは `tests/` 配下に置き、現在は合成データによる数値計算とバックテスト挙動の検証が中心です。`data/` は parquet キャッシュ用の生成ディレクトリで、`data/.gitkeep` 以外の生成データはコミット対象にしないでください。`references/` には論文などの参考資料があります。

## ビルド・テスト・開発コマンド

初回セットアップ:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ダッシュボード起動:

```bash
python -m streamlit run main.py
```

テスト実行:

```bash
pytest tests/
pytest tests/test_strategy.py::TestRunBacktest::test_output_columns
```

リントとフォーマット:

```bash
ruff check .
ruff check --fix .
ruff format .
```

## コーディングスタイルと命名規則

Python 3.11 以上で動く書き方を基本にしてください。Ruff は `pyproject.toml` で `line-length = 100`、import sort、pyflakes、pycodestyle、bugbear、pyupgrade を有効化しています。計算ロジックは `src/strategies/` に置き、Streamlit から分離したテストしやすい関数として保ってください。タブ UI は `src/tabs/` の各 `render(...)` 関数に閉じ、共通の表示ヘルパーや CSS は `src/tabs/common.py` に寄せてください。関数・変数は `snake_case`、定数は `UPPER_CASE` を使います。`V0`、`C0`、`L`、`K`、`lam`、`q` など、論文や数式に対応する短い名前は既存慣習に合わせて使用できます。

## テスト方針

テストフレームワークは `pytest` です。`pythonpath = ["."]` は `pyproject.toml` に設定済みです。戦略ロジックのテストでは、yfinance やネットワークに依存しない決定的な合成データを優先してください。現在の中心対象は `src/strategies/core.py` の `build_V0`、`build_C0`、`perf_metrics`、`run_backtest_loop` と、各戦略モジュールの `compute_return` です。テストファイルは `test_*.py`、クラスは `Test...`、メソッドは `test_...` で命名します。形状、数値不変条件、境界条件、パラメータ変更時の挙動、米国取引日と翌日本取引日の対応付けを重点的に検証してください。

## コミットとプルリクエスト

直近の履歴では、`bugfix`、`説明の追加` のような短く直接的なコミット件名が使われています。コミットは関心ごとごとに分け、観測できる変更を件名にしてください。Pull Request には変更概要、`pytest tests/` と `ruff check .` の結果、UI 変更がある場合はスクリーンショットを含めてください。キャッシュデータ、yfinance の取得範囲、戦略デフォルト値に影響する変更は明記してください。

## エージェント向け指示

ユーザーへの応答は原則として日本語で行ってください。コード内のコメント、変数名、ドキュメントの言語は既存ファイルの慣習に合わせてください。生成物や修正は、このリポジトリの小さな Python/Streamlit 構成に合わせて簡潔に保ち、無関係なリファクタリングは避けてください。新しい戦略を追加する場合は、`src/strategies/` に戦略モジュールを作成し、`compute_signal(ctx)`、`compute_return(ctx)`、`run_backtest(...)`、`run_live_signal(...)` の既存パターンに合わせてください。バックテスト画面への追加は `src/tabs/backtest.py`、今日のシグナル画面への追加は `src/tabs/today.py` を確認してください。

## セキュリティと設定

`data/` の parquet 生成ファイル、仮想環境、`__pycache__/`、ローカルの出力物はコミットしないでください。初回のダッシュボード起動では yfinance から ETF データを取得するため時間がかかる場合があります。要求期間が parquet キャッシュ内に収まる場合は、以後の実行でネットワークアクセスを避けられます。キャッシュが部分的に古い場合、`src/strategies/core.py` の `load_data` は不足分のみ追加ダウンロードします。
