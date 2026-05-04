import numpy as np
import pandas as pd

from src.tabs.backtest import _recent_daily_returns, _recent_return_summary


class TestRecentDailyReturns:
    def test_recent_daily_returns_takes_last_30_rows(self):
        idx = pd.bdate_range("2024-01-01", periods=35)
        rets = pd.DataFrame(
            {
                "PCA SUB(論文)": np.arange(35) / 1000,
                "PCA SUB(改)": np.arange(35, 70) / 1000,
            },
            index=idx,
        )

        daily = _recent_daily_returns(rets, ["PCA SUB(論文)", "PCA SUB(改)"])

        assert daily.shape == (2, 30)
        assert daily.columns[0] == idx[5].strftime("%m/%d")
        assert daily.columns[-1] == idx[-1].strftime("%m/%d")
        assert daily.loc["PCA SUB(論文)", daily.columns[0]] == rets.iloc[5, 0] * 100

    def test_recent_daily_returns_drops_all_nan_days(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        rets = pd.DataFrame(
            {
                "PCA SUB(論文)": [0.01, np.nan, -0.01],
                "PCA SUB(改)": [0.02, np.nan, -0.02],
            },
            index=idx,
        )

        daily = _recent_daily_returns(rets, ["PCA SUB(論文)", "PCA SUB(改)"])

        assert daily.shape == (2, 2)
        assert idx[1].strftime("%m/%d") not in daily.columns

    def test_recent_return_summary(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        rets = pd.DataFrame({"PCA SUB(論文)": [0.01, -0.02, 0.03]}, index=idx)

        summary = _recent_return_summary(rets, ["PCA SUB(論文)"])

        expected_cum = ((1.01 * 0.98 * 1.03) - 1) * 100
        np.testing.assert_allclose(summary.loc["PCA SUB(論文)", "直近3日累積(%)"], expected_cum)
        assert summary.loc["PCA SUB(論文)", "マイナス日数"] == 1
