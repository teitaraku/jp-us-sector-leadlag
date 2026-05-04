"""
日米業種リードラグ投資戦略 — 論文実装アセンブリ
Nakagawa et al. (2026), JSAI SIG-FIN-036

4 戦略を組み合わせて run_backtest / compute_today_signal を提供する。
計算ロジックは strategy_core・strategy_*.py に分離されている。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

# コア基盤の再エクスポート（dashboard・テストが strategy から直接 import できるよう維持）
from src.strategy_core import (  # noqa: F401
    JP_LABEL,
    JP_TICKERS,
    NJ,
    NU,
    US_LABEL,
    US_TICKERS,
    N,
    build_C0,
    build_V0,
    compute_live_signal,
    load_data,
    perf_metrics,
    run_backtest_loop,
)
from src.strategy_double import compute_return as _double
from src.strategy_mom import compute_return as _mom
from src.strategy_pca_plain import compute_return as _pca_plain
from src.strategy_pca_sub import compute_return as _pca_sub
from src.strategy_pca_sub import compute_signal as _pca_sub_signal

STRAT_COLORS = {"PCA_SUB": "blue", "DOUBLE": "green", "PCA_PLAIN": "orange", "MOM": "red"}
STRAT_DISP = {
    "PCA_SUB": "PCA SUB（提案）",
    "DOUBLE": "DOUBLE",
    "PCA_PLAIN": "PCA PLAIN",
    "MOM": "MOM",
}

_PAPER_STRATEGIES: dict[str, Callable] = {
    "MOM": _mom,
    "PCA_PLAIN": _pca_plain,
    "PCA_SUB": _pca_sub,
    "DOUBLE": _double,
}


def run_backtest(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int = 60,
    lam: float = 0.9,
    K: int = 3,
    q: float = 0.30,
    cfull_end: str = "2014-12-31",
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    return run_backtest_loop(
        us_cc,
        jp_cc,
        jp_oc,
        L,
        lam,
        K,
        q,
        cfull_end,
        strategies=_PAPER_STRATEGIES,
        on_progress=on_progress,
    )


def compute_today_signal(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int = 60,
    lam: float = 0.9,
    K: int = 3,
    q: float = 0.30,
    cfull_end: str = "2014-12-31",
) -> dict:
    return compute_live_signal(
        us_cc,
        jp_cc,
        jp_oc,
        L,
        lam,
        K,
        q,
        cfull_end,
        signal_fn=_pca_sub_signal,
    )
