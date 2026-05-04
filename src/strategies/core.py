"""
日米業種リードラグ投資戦略 — コア基盤
定数・データ取得・V0/C0 構築・共通ループ・パフォーマンス指標
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────
US_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
JP_TICKERS = [
    "1617.T",
    "1618.T",
    "1619.T",
    "1620.T",
    "1621.T",
    "1622.T",
    "1623.T",
    "1624.T",
    "1625.T",
    "1626.T",
    "1627.T",
    "1628.T",
    "1629.T",
    "1630.T",
    "1631.T",
    "1632.T",
    "1633.T",
]

US_LABEL = {
    "XLB": "Materials",
    "XLC": "Comm Svcs",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Info Tech",
    "XLP": "Cons Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Cons Discret",
}
JP_LABEL = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融(除く銀行)",
    "1633.T": "不動産",
}

US_CYCLICAL = {"XLB", "XLE", "XLF", "XLRE"}
US_DEFENSIVE = {"XLK", "XLP", "XLU", "XLV"}
JP_CYCLICAL = {"1618.T", "1625.T", "1629.T", "1631.T"}
JP_DEFENSIVE = {"1617.T", "1621.T", "1627.T", "1630.T"}

NU = len(US_TICKERS)
NJ = len(JP_TICKERS)
N = NU + NJ

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PAPER_CFULL_START = pd.Timestamp("2010-01-01")
PAPER_CFULL_END = pd.Timestamp("2014-12-31")
MIN_CFULL_OBS = 100


# ─────────────────────────────────────────────────────────────────────────────
# 戦略コンテキスト
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StepContext:
    """1ステップ分の計算済み中間量。各戦略関数に渡す。"""

    mu: np.ndarray
    sigma: np.ndarray
    z_us: np.ndarray
    Ct: np.ndarray
    C0: np.ndarray
    target: np.ndarray | None  # バックテスト時のみ。live signal では None
    K: int
    lam: float
    q: float


def ls_ret(signal: np.ndarray, target: np.ndarray, q: float) -> float:
    """ロングショートリターンを計算する共通ヘルパー。"""
    mask = ~np.isnan(target)
    s, r = signal[mask], target[mask]
    n = len(s)
    if n < 3:
        return np.nan
    n_each = max(1, int(n * q))
    rank = np.argsort(s)
    return float(r[rank[-n_each:]].mean() - r[rank[:n_each]].mean())


# ─────────────────────────────────────────────────────────────────────────────
# データ取得
# ─────────────────────────────────────────────────────────────────────────────
def load_data(start: str, end: str) -> tuple[pd.DataFrame, ...]:
    DATA_DIR.mkdir(exist_ok=True)

    def extract(raw, tickers, price):
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[price]
            for t in tickers:
                if t not in df.columns:
                    df[t] = np.nan
            return df[tickers].copy()
        df = pd.DataFrame({tickers[0]: raw[price]})
        for t in tickers[1:]:
            df[t] = np.nan
        return df

    def _save(uc, jc, jo):
        uc.to_parquet(DATA_DIR / "us_close.parquet")
        jc.to_parquet(DATA_DIR / "jp_close.parquet")
        jo.to_parquet(DATA_DIR / "jp_open.parquet")

    def _download_and_cache(s: str, e: str):
        us_raw = yf.download(US_TICKERS, start=s, end=e, auto_adjust=True, progress=False)
        jp_raw = yf.download(JP_TICKERS, start=s, end=e, auto_adjust=True, progress=False)
        uc = extract(us_raw, US_TICKERS, "Close")
        jc = extract(jp_raw, JP_TICKERS, "Close")
        jo = extract(jp_raw, JP_TICKERS, "Open")
        _save(uc, jc, jo)
        return uc, jc, jo

    def _incremental_update(uc, jc, jo, cache_max: pd.Timestamp, e: str):
        new_start = (cache_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        us_raw = yf.download(US_TICKERS, start=new_start, end=e, auto_adjust=True, progress=False)
        jp_raw = yf.download(JP_TICKERS, start=new_start, end=e, auto_adjust=True, progress=False)
        if us_raw.empty and jp_raw.empty:
            return uc, jc, jo
        uc_new = extract(us_raw, US_TICKERS, "Close")
        jc_new = extract(jp_raw, JP_TICKERS, "Close")
        jo_new = extract(jp_raw, JP_TICKERS, "Open")

        def merge(old, new):
            combined = pd.concat([old, new])
            return combined[~combined.index.duplicated(keep="last")].sort_index()

        uc, jc, jo = merge(uc, uc_new), merge(jc, jc_new), merge(jo, jo_new)
        _save(uc, jc, jo)
        return uc, jc, jo

    cache = {
        "us_close": DATA_DIR / "us_close.parquet",
        "jp_close": DATA_DIR / "jp_close.parquet",
        "jp_open": DATA_DIR / "jp_open.parquet",
    }
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    if all(p.exists() for p in cache.values()):
        uc = pd.read_parquet(cache["us_close"])
        jc = pd.read_parquet(cache["jp_close"])
        jo = pd.read_parquet(cache["jp_open"])
        cache_min = uc.index.min()
        cache_max = uc.index.max()
        if cache_min > start_ts:
            uc, jc, jo = _download_and_cache(start, end)
        elif cache_max < end_ts:
            uc, jc, jo = _incremental_update(uc, jc, jo, cache_max, end)
        uc = uc.loc[start_ts:end_ts]
        jc = jc.loc[start_ts:end_ts]
        jo = jo.loc[start_ts:end_ts]
    else:
        uc, jc, jo = _download_and_cache(start, end)

    us_cc = uc.pct_change()
    jp_cc = jc.pct_change()
    jp_oc = (jc - jo) / jo.replace(0, np.nan)

    return (
        us_cc.dropna(how="all"),
        jp_cc.dropna(how="all"),
        jp_oc.dropna(how="all"),
        uc,
        jc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 事前部分空間 V₀ と C₀
# ─────────────────────────────────────────────────────────────────────────────
def build_V0() -> np.ndarray:
    v1 = np.ones(N) / np.sqrt(N)

    v2 = np.zeros(N)
    v2[:NU] = 1.0 / np.sqrt(NU)
    v2[NU:] = -1.0 / np.sqrt(NJ)
    v2 -= np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2)

    v3 = np.zeros(N)
    for i, t in enumerate(US_TICKERS):
        if t in US_CYCLICAL:
            v3[i] = 1.0
        elif t in US_DEFENSIVE:
            v3[i] = -1.0
    for i, t in enumerate(JP_TICKERS):
        if t in JP_CYCLICAL:
            v3[NU + i] = 1.0
        elif t in JP_DEFENSIVE:
            v3[NU + i] = -1.0
    v3 -= np.dot(v3, v1) * v1
    v3 -= np.dot(v3, v2) * v2
    norm3 = np.linalg.norm(v3)
    if norm3 > 1e-12:
        v3 /= norm3

    return np.column_stack([v1, v2, v3])


def build_C0(V0: np.ndarray, Cfull: np.ndarray) -> np.ndarray:
    D0 = np.diag(V0.T @ Cfull @ V0)
    C0_raw = V0 @ np.diag(D0) @ V0.T
    diag_sq = np.sqrt(np.maximum(np.diag(C0_raw), 1e-12))
    C0 = C0_raw / np.outer(diag_sq, diag_sq)
    np.fill_diagonal(C0, 1.0)
    return C0


def _joint_cc(us_cc: pd.DataFrame, jp_cc: pd.DataFrame) -> pd.DataFrame:
    return us_cc[US_TICKERS].join(jp_cc[JP_TICKERS], how="inner").dropna(thresh=N // 2)


def _fill_corr_missing(corr: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = corr.copy()
    missing = ~np.isfinite(out)
    out[missing] = fallback[missing]
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 1.0)
    return out


def _window_state(
    window: pd.DataFrame, fallback_corr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = window.mean(skipna=True).values
    sigma = window.std(skipna=True, ddof=0).values
    sigma = np.where((~np.isfinite(sigma)) | (sigma < 1e-10), 1e-10, sigma)
    min_periods = max(3, min(len(window), len(window) // 2))
    Ct = window.corr(min_periods=min_periods).values
    Ct = _fill_corr_missing(Ct, fallback_corr)
    return mu, sigma, Ct


def _prepare_prior(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    cfull_end: str,
    cfull_start: str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    V0 = build_V0()
    all_cc = _joint_cc(us_cc, jp_cc)
    if all_cc.empty:
        raise ValueError("日米の Close-to-Close リターンが対応するデータがありません。")
    cfull_end_ts = pd.Timestamp(cfull_end)
    if cfull_start is not None:
        cfull_start_ts = pd.Timestamp(cfull_start)
    else:
        cfull_start_ts = (
            PAPER_CFULL_START if cfull_end_ts >= PAPER_CFULL_END else all_cc.index.min()
        )
        if (
            cfull_end_ts >= PAPER_CFULL_END
            and all_cc.index.min() > PAPER_CFULL_START + pd.Timedelta(days=31)
        ):
            raise ValueError(
                "Cfull 推定期間の開始データが不足しています。論文版では 2010-01-01 から "
                f"{cfull_end_ts.date()} までで Cfull を推定します "
                f"(現在の最初の利用可能日: {all_cc.index.min().date()})。"
            )
        if cfull_end_ts >= PAPER_CFULL_END and all_cc.index.max() < PAPER_CFULL_END:
            raise ValueError(
                "Cfull 推定期間の終了データが不足しています。論文版では 2010-01-01 から "
                f"{cfull_end_ts.date()} までで Cfull を推定します "
                f"(現在の最後の利用可能日: {all_cc.index.max().date()})。"
            )
    cfull_mask = (all_cc.index >= cfull_start_ts) & (all_cc.index <= cfull_end_ts)
    cfull_data = all_cc[cfull_mask]
    if len(cfull_data) < MIN_CFULL_OBS:
        if cfull_start is not None:
            raise ValueError(
                f"Cfull 推定用データが不足しています（{cfull_start_ts.date()} ～ {cfull_end_ts.date()}）。"
                f"少なくとも {MIN_CFULL_OBS} 行必要ですが現在 {len(cfull_data)} 行です。"
                "バックテスト開始日を推定期間より前に設定してください。"
            )
        raise ValueError(
            "Cfull 推定用データが不足しています。論文版では 2010-01-01 から "
            f"{cfull_end_ts.date()} までのデータが少なくとも {MIN_CFULL_OBS} 行必要です "
            f"(現在: {len(cfull_data)} 行)。開始日を 2010-01-01 以前にしてください。"
        )
    identity = np.eye(N)
    Cfull = _fill_corr_missing(cfull_data.corr(min_periods=MIN_CFULL_OBS).values, identity)
    np.fill_diagonal(Cfull, 1.0)
    C0 = build_C0(V0, Cfull)
    return V0, C0, all_cc


# ─────────────────────────────────────────────────────────────────────────────
# 共通バックテストループ
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest_loop(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int,
    lam: float,
    K: int,
    q: float,
    cfull_end: str,
    strategies: dict[str, Callable[[StepContext], float]],
    on_progress: Callable[[int, int], None] | None = None,
    cfull_start: str | None = None,
    backtest_start: str | None = None,
) -> pd.DataFrame:
    """戦略名→関数の辞書を受け取り、全戦略のロングショートリターン系列を返す。"""
    _, C0, all_cc = _prepare_prior(us_cc, jp_cc, cfull_end, cfull_start)

    jp_dates_arr = jp_oc.index.values
    us_dates = us_cc[US_TICKERS].dropna(how="all").index
    backtest_start_ts = pd.Timestamp(backtest_start) if backtest_start else None
    paired: list[tuple] = []
    for us_date in us_dates:
        if backtest_start_ts is not None and us_date < backtest_start_ts:
            continue
        future = jp_dates_arr[jp_dates_arr > us_date]
        if len(future) == 0:
            continue
        next_jp = future[0]
        if next_jp in jp_oc.index:
            paired.append((us_date, next_jp))

    results = {s: [] for s in strategies}
    dates_out: list = []
    n_pairs = len(paired)

    for step, (us_date, jp_date) in enumerate(paired):
        if on_progress and step % max(n_pairs // 80, 1) == 0:
            on_progress(step, n_pairs)

        t_idx = all_cc.index.searchsorted(us_date, side="left")
        if t_idx < L:
            continue

        window = all_cc.iloc[t_idx - L : t_idx]
        if window.isna().to_numpy().mean() > 0.3:
            continue

        mu, sigma, Ct = _window_state(window, C0)

        us_today = us_cc.loc[us_date, US_TICKERS].values.astype(float)
        z_us = (us_today - mu[:NU]) / sigma[:NU]
        z_us = np.where(np.isfinite(z_us), z_us, 0.0)

        target_row = jp_oc.loc[jp_date, JP_TICKERS]
        target = target_row.values.astype(float)
        if np.isnan(target).mean() > 0.5:
            continue

        ctx = StepContext(
            mu=mu,
            sigma=sigma,
            z_us=z_us,
            Ct=Ct,
            C0=C0,
            target=target,
            K=K,
            lam=lam,
            q=q,
        )

        for name, fn in strategies.items():
            try:
                results[name].append(fn(ctx))
            except Exception:
                results[name].append(np.nan)
        dates_out.append(us_date)

    return pd.DataFrame(results, index=dates_out).dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────────
# 今日のシグナル（共通実装）
# ─────────────────────────────────────────────────────────────────────────────
def compute_live_signal(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int,
    lam: float,
    K: int,
    q: float,
    cfull_end: str,
    signal_fn: Callable[[StepContext], np.ndarray],
    cfull_start: str | None = None,
) -> dict:
    """任意の compute_signal 関数を使い、最新米国リターンから日本業種 ETF のシグナルを計算する。"""
    _, C0, all_cc = _prepare_prior(us_cc, jp_cc, cfull_end, cfull_start)

    us_valid = us_cc[US_TICKERS].dropna(how="all")
    if len(us_valid) == 0:
        raise ValueError("米国リターンデータがありません")

    latest_us_date = us_valid.index[-1]
    us_today_row = us_valid.loc[latest_us_date]

    t_idx = all_cc.index.searchsorted(latest_us_date, side="left")
    all_cc_window = all_cc.iloc[:t_idx]
    if len(all_cc_window) < L:
        raise ValueError(f"ウィンドウ計算に必要なデータが不足しています（必要: {L}日）")

    window = all_cc_window.iloc[-L:]
    mu, sigma, Ct = _window_state(window, C0)

    z_us = (us_today_row.values.astype(float) - mu[:NU]) / sigma[:NU]
    z_us = np.where(np.isfinite(z_us), z_us, 0.0)

    ctx = StepContext(
        mu=mu,
        sigma=sigma,
        z_us=z_us,
        Ct=Ct,
        C0=C0,
        target=None,
        K=K,
        lam=lam,
        q=q,
    )
    z_jp = signal_fn(ctx)

    signal = pd.Series(z_jp, index=JP_TICKERS)
    n_each = max(1, int(NJ * q))

    jp_dates = jp_oc.index
    future_jp = jp_dates[jp_dates > latest_us_date]
    next_jp_date = future_jp[0] if len(future_jp) > 0 else None

    return {
        "signal": signal,
        "us_date": latest_us_date,
        "next_jp_date": next_jp_date,
        "us_returns": us_today_row,
        "n_long": n_each,
        "n_short": n_each,
    }


# ─────────────────────────────────────────────────────────────────────────────
# パフォーマンス指標
# ─────────────────────────────────────────────────────────────────────────────
def perf_metrics(rets: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for col in rets.columns:
        r = rets[col].dropna()
        if len(r) == 0:
            continue
        ar = r.mean() * 252 * 100
        risk = r.std() * np.sqrt(252) * 100
        rr = ar / risk if risk > 0 else 0.0
        cum = (1 + r).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
        rows[col] = {
            "AR(%)": round(ar, 2),
            "RISK(%)": round(risk, 2),
            "R/R": round(rr, 2),
            "MDD(%)": round(abs(mdd), 2),
        }
    return pd.DataFrame(rows).T
