"""PCA SUB 戦略 — 部分空間正則化付き PCA（論文提案手法）"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import linalg

from src.strategies.core import NJ, NU, StepContext, compute_live_signal, ls_ret, run_backtest_loop

CFULL_END = "2014-12-31"


def compute_signal(ctx: StepContext) -> np.ndarray:
    try:
        C_reg = (1.0 - ctx.lam) * ctx.Ct + ctx.lam * ctx.C0
        eigvals_r, eigvecs_r = linalg.eigh(C_reg)
        order_r = np.argsort(eigvals_r)[::-1]
        Vk_r = eigvecs_r[:, order_r[: ctx.K]]
        return Vk_r[NU:] @ (Vk_r[:NU].T @ ctx.z_us)
    except Exception:
        return np.zeros(NJ)


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
        us_cc,
        jp_cc,
        jp_oc,
        L=L,
        lam=lam,
        K=K,
        q=q,
        cfull_end=CFULL_END,
        strategies={"PCA_SUB": compute_return},
        on_progress=on_progress,
        backtest_start=backtest_start,
    )


def run_live_signal(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int = 60,
    lam: float = 0.9,
    K: int = 3,
    q: float = 0.30,
) -> dict:
    return compute_live_signal(
        us_cc,
        jp_cc,
        jp_oc,
        L=L,
        lam=lam,
        K=K,
        q=q,
        cfull_end=CFULL_END,
        signal_fn=compute_signal,
    )
