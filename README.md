# 日米業種リードラグ投資戦略ダッシュボード

部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略の実装・可視化ダッシュボード。

> 中川 慧, 竹本 悠城, 久保 健治, 加藤 真大  
> "部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略"  
> 人工知能学会第二種研究会 金融情報学研究会 SIG-FIN-036 (2026)

## 概要

日米の取引時間帯の非同期性を利用した投資戦略です。米国 11 業種 ETF（Select Sector SPDR）の当日 Close-to-Close リターンを情報源として、日本 17 業種 ETF（NEXT FUNDS TOPIX-17）の翌営業日 Open-to-Close リターンを予測します。日米結合相関行列に部分空間正則化付き PCA を適用し、低ランク伝播行列としてシグナルを構成します。

## セットアップ

**前提条件**: Python 3.10 以上、インターネット接続（初回データ取得時）

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

仮想環境を終了するには `deactivate` を実行します。

## 実行

```bash
python -m streamlit run dashboard.py
```

ブラウザで http://localhost:8501 が開きます。初回起動時は yfinance によるデータダウンロードに数分かかる場合があります。ダウンロード済みデータは `data/` にキャッシュされます。

## ディレクトリ構成

```
.
├── dashboard.py        # Streamlit UI
├── strategy.py         # 計算ロジック（Streamlit 非依存）
├── requirements.txt    # 依存パッケージ
├── pyproject.toml      # ruff 設定
├── mise.toml           # Python バージョン管理
├── tests/
│   └── test_strategy.py
├── data/               # parquet キャッシュ（自動生成・git 管理外）
└── references/         # 論文 PDF

```

## テスト

```bash
pytest tests/
```

`strategy.py` の主要関数（`build_V0`・`build_C0`・`perf_metrics`・`run_backtest`）を合成データでユニットテストします。yfinance は使用しません。

## フォーマット・リント

[ruff](https://docs.astral.sh/ruff/) を使用しています（`line-length = 100`、isort 含む）。

```bash
ruff check .        # リント
ruff check --fix .  # 自動修正
ruff format .       # フォーマット
```
