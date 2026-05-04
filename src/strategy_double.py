"""DOUBLE 戦略 — MOM と PCA SUB の AND 結合（ベースライン）"""

from __future__ import annotations

import numpy as np

from src.strategy_core import StepContext
from src.strategy_mom import compute_signal as _mom_signal
from src.strategy_pca_sub import compute_signal as _pca_sub_signal


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
