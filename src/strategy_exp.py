"""
実験的戦略モジュール — PCA SUB(改) のカスタマイズ用サンドボックス
compute_signal を変更しても src/strategy_pca_sub.py の論文実装には影響しない。
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from src.strategy_core import NJ, NU, StepContext, ls_ret


# ── PCA SUB シグナル（ここを改変する） ──────────────────────────────────────
def compute_signal(ctx: StepContext) -> np.ndarray:
    try:
        C_reg = (1.0 - ctx.lam) * ctx.Ct + ctx.lam * ctx.C0
        eigvals_r, eigvecs_r = linalg.eigh(C_reg)
        order_r = np.argsort(eigvals_r)[::-1]
        Vk_r = eigvecs_r[:, order_r[: ctx.K]]
        return Vk_r[NU:] @ (Vk_r[:NU].T @ ctx.z_us)
    except Exception:
        return np.zeros(NJ)


# ────────────────────────────────────────────────────────────────────────────


def compute_return(ctx: StepContext) -> float:
    return ls_ret(compute_signal(ctx), ctx.target, ctx.q)
