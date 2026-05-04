"""PCA SUB 戦略 — 部分空間正則化付き PCA（論文提案手法）"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from src.strategy_core import NJ, NU, StepContext, ls_ret


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
