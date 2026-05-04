"""PCA PLAIN 戦略 — 正則化なし PCA によるシグナル（ベースライン）"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from src.strategy_core import NJ, NU, StepContext, ls_ret


def compute_signal(ctx: StepContext) -> np.ndarray:
    try:
        eigvals, eigvecs = linalg.eigh(ctx.Ct)
        order = np.argsort(eigvals)[::-1]
        Vk = eigvecs[:, order[: ctx.K]]
        return Vk[NU:] @ (Vk[:NU].T @ ctx.z_us)
    except Exception:
        return np.zeros(NJ)


def compute_return(ctx: StepContext) -> float:
    return ls_ret(compute_signal(ctx), ctx.target, ctx.q)
