import pandas as pd
import streamlit as st

from src.optimizer import optimize_parameters, parameter_combinations
from src.strategies.core import run_backtest_loop
from src.strategies.exp import run_backtest as _exp_run_backtest
from src.strategies.pca_sub import compute_return as _pca_sub

_SEARCH_L_OPTIONS = list(range(20, 253, 5))
_SEARCH_LAM_OPTIONS = [round(i * 0.05, 2) for i in range(21)]
_SEARCH_K_OPTIONS = [1, 2, 3, 4, 5]
_SEARCH_Q_OPTIONS = [round(0.10 + i * 0.05, 2) for i in range(8)]
_SEARCH_MAX_COMBOS = 200
_MAXIMIZE_METRICS = {"AR(%)", "R/R"}
_MINIMIZE_METRICS = {"RISK(%)", "MDD(%)"}
_METRIC_HELP = {
    "AR(%)": "年率リターン。大きいほど高収益。",
    "RISK(%)": "年率ボラティリティ。小さいほど安定。",
    "R/R": "リターン/リスク比（シャープレシオ相当）。大きいほど効率的。",
    "MDD(%)": "最大ドローダウン。小さいほど安定。",
}


def _store_best_params(row: pd.Series) -> None:
    st.session_state["L"] = int(row["L"])
    st.session_state["lam"] = float(row["lam"])
    st.session_state["K"] = int(row["K"])
    st.session_state["q"] = float(row["q"])


def _highlight_good_metric_cells(data: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for col in _MAXIMIZE_METRICS | _MINIMIZE_METRICS:
        if col not in data:
            continue

        values = pd.to_numeric(data[col], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue

        min_value = valid.min()
        max_value = valid.max()
        value_range = max_value - min_value
        if value_range == 0:
            scores = pd.Series(0.5, index=valid.index)
        elif col in _MAXIMIZE_METRICS:
            scores = (valid - min_value) / value_range
        else:
            scores = (max_value - valid) / value_range

        for idx, score in scores.items():
            alpha = 0.12 + 0.32 * float(score)
            styles.at[idx, col] = f"background-color: rgba(34, 139, 34, {alpha:.2f})"

    return styles


def render(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    start: str,
) -> None:
    st.caption(
        "選択した PCA SUB 戦略だけを複数パラメータでバックテストし、R/R が高い候補から並べます。"
    )
    search_cols = st.columns([1.2, 2.8])
    with search_cols[0]:
        search_strategy = st.radio(
            "探索対象",
            ["PCA SUB(論文)", "PCA SUB(改)"],
            horizontal=True,
            help="論文版は固定 Cfull、改版は実験用実装で探索します。",
        )

    grid_row_1 = st.columns(2)
    with grid_row_1[0]:
        search_L = st.multiselect("L 候補", _SEARCH_L_OPTIONS, default=[40, 60, 80])
    with grid_row_1[1]:
        search_lam = st.multiselect("λ 候補", _SEARCH_LAM_OPTIONS, default=[0.75, 0.9, 1.0])

    grid_row_2 = st.columns(2)
    with grid_row_2[0]:
        search_K = st.multiselect("K 候補", _SEARCH_K_OPTIONS, default=[2, 3, 4])
    with grid_row_2[1]:
        search_q = st.multiselect("q 候補", _SEARCH_Q_OPTIONS, default=[0.20, 0.30, 0.40])

    param_grid = {"L": search_L, "lam": search_lam, "K": search_K, "q": search_q}
    n_combos = len(parameter_combinations(param_grid))
    st.caption(f"探索回数: {n_combos} 回（上限 {_SEARCH_MAX_COMBOS} 回）")
    submitted = st.button(
        "🔎 パラメータ探索を実行",
        type="primary",
        disabled=n_combos == 0 or n_combos > _SEARCH_MAX_COMBOS,
    )

    if n_combos > _SEARCH_MAX_COMBOS:
        st.warning("候補数が多すぎます。候補を減らしてから実行してください。")

    if submitted:
        prog = st.progress(0.0, text="パラメータ探索中…")

        def on_search_progress(step: int, total: int) -> None:
            prog.progress(step / total, text=f"パラメータ探索中… {step}/{total}")

        try:
            with st.spinner("パラメータ探索中…"):
                if search_strategy == "PCA SUB(論文)":

                    def runner(**params) -> pd.DataFrame:
                        return run_backtest_loop(
                            us_cc,
                            jp_cc,
                            jp_oc,
                            cfull_end="2014-12-31",
                            strategies={"PCA SUB(論文)": _pca_sub},
                            backtest_start=start,
                            **params,
                        )

                else:

                    def runner(**params) -> pd.DataFrame:
                        rets = _exp_run_backtest(
                            us_cc,
                            jp_cc,
                            jp_oc,
                            backtest_start=start,
                            **params,
                        )
                        return rets.rename(columns={"PCA_SUB": "PCA SUB(改)"})

                search_result = optimize_parameters(
                    runner,
                    param_grid,
                    strategy_name=search_strategy,
                    objective="R/R",
                    on_progress=on_search_progress,
                )
                st.session_state["param_search_result"] = search_result
                st.session_state["param_search_strategy"] = search_strategy
                prog.empty()
                st.success("パラメータ探索が完了しました。")
        except Exception as exc:
            prog.empty()
            st.error(f"パラメータ探索エラー: {exc}")
            import traceback

            st.code(traceback.format_exc())

    if "param_search_result" in st.session_state:
        search_result = st.session_state["param_search_result"]
        search_strategy = st.session_state.get("param_search_strategy", "PCA SUB")
        st.markdown(f"**探索結果: {search_strategy}**")
        st.caption("結果は R/R の高い順です。表の行を選択してパラメータを反映できます。")
        styled_result = search_result.style.format(
            {
                "lam": "{:.2f}",
                "q": "{:.2f}",
                "AR(%)": "{:.2f}",
                "RISK(%)": "{:.2f}",
                "R/R": "{:.2f}",
                "MDD(%)": "{:.2f}",
            }
        ).apply(_highlight_good_metric_cells, axis=None)
        table_state = st.dataframe(
            styled_result,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "lam": st.column_config.NumberColumn("λ", format="%.2f"),
                "q": st.column_config.NumberColumn("q", format="%.2f"),
                "AR(%)": st.column_config.NumberColumn(
                    "AR(%)", help=_METRIC_HELP["AR(%)"], format="%.2f"
                ),
                "RISK(%)": st.column_config.NumberColumn(
                    "RISK(%)", help=_METRIC_HELP["RISK(%)"], format="%.2f"
                ),
                "R/R": st.column_config.NumberColumn(
                    "R/R", help=_METRIC_HELP["R/R"], format="%.2f"
                ),
                "MDD(%)": st.column_config.NumberColumn(
                    "MDD(%)", help=_METRIC_HELP["MDD(%)"], format="%.2f"
                ),
            },
        )
        selected_rows = table_state.selection.rows
        selected_row = search_result.iloc[selected_rows[0]].copy() if selected_rows else None
        st.button(
            "選択したパラメータをサイドバーに反映",
            on_click=_store_best_params,
            args=(selected_row,),
            disabled=selected_row is None,
            width="stretch",
        )
