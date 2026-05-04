import pandas as pd

TAB_CSS = """
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: rgba(255,255,255,0.06);
    padding: 4px 6px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 6px 18px;
    color: #888;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(255,255,255,0.08);
    color: #ccc;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(255,255,255,0.18);
    color: #ffffff;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(0,0,0,0.35);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
[data-testid="stMetricValue"] {
    font-size: 1rem;
}
</style>
"""


def color_best(s: pd.Series):
    if s.name in ("AR(%)", "R/R"):
        best = s.max()
    elif s.name == "MDD(%)":
        best = s.min()
    else:
        return [""] * len(s)
    return ["background-color:#c8f7c5; color:#000000" if v == best else "" for v in s]


def highlight_action(row):
    if "ロング" in str(row["売買"]):
        return ["background-color:#c8f7c5; color:#000000"] * len(row)
    if "ショート" in str(row["売買"]):
        return ["background-color:#f7c8c8; color:#000000"] * len(row)
    return [""] * len(row)
