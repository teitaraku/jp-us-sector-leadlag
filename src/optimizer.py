from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

import pandas as pd

from src.strategies.core import perf_metrics

OBJECTIVES = ("R/R", "AR(%)", "MDD(%)", "RISK(%)")
MINIMIZE_OBJECTIVES = {"MDD(%)", "RISK(%)"}


def parameter_combinations(param_grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """パラメータ候補の直積を辞書リストとして返す。"""
    keys = list(param_grid)
    values = [list(param_grid[key]) for key in keys]
    if not keys or any(len(v) == 0 for v in values):
        return []
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def optimize_parameters(
    runner: Callable[..., pd.DataFrame],
    param_grid: dict[str, Iterable[Any]],
    strategy_name: str,
    objective: str = "R/R",
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """バックテスト runner をグリッド探索し、指標順に並べた結果を返す。"""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective は {OBJECTIVES} から選択してください: {objective}")

    combos = parameter_combinations(param_grid)
    if not combos:
        raise ValueError("パラメータ候補が空です。")

    rows: list[dict[str, Any]] = []
    for i, params in enumerate(combos, start=1):
        if on_progress:
            on_progress(i - 1, len(combos))

        rets = runner(**params)
        metrics = perf_metrics(rets)
        if strategy_name not in metrics.index:
            row = {**params, "AR(%)": pd.NA, "RISK(%)": pd.NA, "R/R": pd.NA, "MDD(%)": pd.NA}
        else:
            row = {**params, **metrics.loc[strategy_name].to_dict()}
        rows.append(row)

    if on_progress:
        on_progress(len(combos), len(combos))

    result = pd.DataFrame(rows)
    ascending = objective in MINIMIZE_OBJECTIVES
    return result.sort_values(objective, ascending=ascending, na_position="last").reset_index(
        drop=True
    )
