"""MOM 戦略 — 過去平均リターンによるモメンタムシグナル"""

from __future__ import annotations

import numpy as np

from src.strategy_core import NU, StepContext, ls_ret


def compute_signal(ctx: StepContext) -> np.ndarray:
    return ctx.mu[NU:]


def compute_return(ctx: StepContext) -> float:
    return ls_ret(compute_signal(ctx), ctx.target, ctx.q)
