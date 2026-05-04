# Repository Guidelines

## プロジェクト構成とモジュール

このリポジトリは、日米業種リードラグ投資戦略を実装し、Streamlit で可視化する Python プロジェクトです。`main.py` はエントリポイントで、`src.dashboard.main()` を呼び出します。戦略計算の中核は `src/strategy.py` にあり、Streamlit に依存しない形で保守してください。関連する戦略バリエーションは `src/strategy_*.py`、画面 UI は `src/dashboard.py` にあります。テストは `tests/` 配下に置き、現在は合成データによる数値計算とバックテスト挙動の検証が中心です。`data/` は parquet キャッシュ用の生成ディレクトリで、コミット対象にしないでください。`references/` には論文などの参考資料があります。

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

Python 3.11 互換の書き方を基本にしてください。Ruff は `pyproject.toml` で `line-length = 100`、import sort、pyflakes、pycodestyle、bugbear、pyupgrade を有効化しています。計算ロジックは Streamlit から分離し、テストしやすい純粋な関数として保ってください。関数・変数は `snake_case`、定数は `UPPER_CASE` を使います。`V0`、`C0`、`L`、`K`、`lam` など、論文や数式に対応する短い名前は既存慣習に合わせて使用できます。

## テスト方針

テストフレームワークは `pytest` です。`pythonpath = ["."]` は `pyproject.toml` に設定済みです。戦略ロジックのテストでは、yfinance やネットワークに依存しない決定的な合成データを優先してください。テストファイルは `test_*.py`、クラスは `Test...`、メソッドは `test_...` で命名します。形状、数値不変条件、境界条件、パラメータ変更時の挙動を重点的に検証してください。

## コミットとプルリクエスト

直近の履歴では、`bugfix`、`説明の追加` のような短く直接的なコミット件名が使われています。コミットは関心ごとごとに分け、観測できる変更を件名にしてください。Pull Request には変更概要、`pytest tests/` と `ruff check .` の結果、UI 変更がある場合はスクリーンショットを含めてください。キャッシュデータ、yfinance の取得範囲、戦略デフォルト値に影響する変更は明記してください。

## エージェント向け指示

ユーザーへの応答は原則として日本語で行ってください。コード内のコメント、変数名、ドキュメントの言語は既存ファイルの慣習に合わせてください。生成物や修正は、このリポジトリの小さな Python/Streamlit 構成に合わせて簡潔に保ち、無関係なリファクタリングは避けてください。

## セキュリティと設定

`data/` の生成ファイル、仮想環境、ローカルの出力物はコミットしないでください。初回のダッシュボード起動では yfinance から ETF データを取得するため時間がかかる場合があります。要求期間が parquet キャッシュ内に収まる場合は、以後の実行でネットワークアクセスを避けられます。
