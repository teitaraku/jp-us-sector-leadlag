"""DOUBLE 戦略 — MOM と PCA SUB の AND 結合（ベースライン）"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.strategy_core import StepContext, run_backtest_loop
from src.strategy_mom import compute_signal as _mom_signal
from src.strategy_pca_sub import compute_signal as _pca_sub_signal

CFULL_END = "2014-12-31"


def compute_return(ctx: StepContext) -> float:
    target = ctx.target
    mask = ~np.isnan(target)
    if mask.sum() < 4:
        return np.nan
    s1 = _mom_signal(ctx)[mask]
    s2 = _pca_sub_signal(ctx)[mask]
    r = target[mask]
    hi1 = s1 >= np.median(s1)
    hi2 = s2 >= np.median(s2)
    lg, sh = hi1 & hi2, ~hi1 & ~hi2
    if lg.sum() > 0 and sh.sum() > 0:
        return float(r[lg].mean() - r[sh].mean())
    return np.nan


def run_backtest(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int = 60,
    lam: float = 0.9,
    K: int = 3,
    q: float = 0.30,
    on_progress: Callable[[int, int], None] | None = None,
    backtest_start: str | None = None,
) -> pd.DataFrame:
    return run_backtest_loop(
        us_cc, jp_cc, jp_oc, L=L, lam=lam, K=K, q=q,
        cfull_end=CFULL_END,
        strategies={"DOUBLE": compute_return},
        on_progress=on_progress,
        backtest_start=backtest_start,
    )
