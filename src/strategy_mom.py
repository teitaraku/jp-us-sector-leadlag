"""MOM 戦略 — 過去平均リターンによるモメンタムシグナル"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.strategy_core import NU, StepContext, ls_ret, run_backtest_loop

CFULL_END = "2014-12-31"


def compute_signal(ctx: StepContext) -> np.ndarray:
    return ctx.mu[NU:]


def compute_return(ctx: StepContext) -> float:
    return ls_ret(compute_signal(ctx), ctx.target, ctx.q)


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
        strategies={"MOM": compute_return},
        on_progress=on_progress,
        backtest_start=backtest_start,
    )
