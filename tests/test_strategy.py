"""strategy.py のユニットテスト"""

import numpy as np
import pandas as pd

from src.strategy import (
    JP_TICKERS,
    NJ,
    NU,
    US_TICKERS,
    N,
    build_C0,
    build_V0,
    perf_metrics,
    run_backtest,
)


class TestBuildV0:
    def test_shape(self):
        V0 = build_V0()
        assert V0.shape == (N, 3)

    def test_orthonormal_columns(self):
        V0 = build_V0()
        np.testing.assert_allclose(V0.T @ V0, np.eye(3), atol=1e-10)

    def test_v1_uniform_positive(self):
        """v1 は全要素が等しい正値（グローバルファクター）"""
        v1 = build_V0()[:, 0]
        assert np.all(v1 > 0)
        np.testing.assert_allclose(np.std(v1), 0.0, atol=1e-10)

    def test_v2_sign_structure(self):
        """v2 は米国側が正、日本側が負（国スプレッド）"""
        v2 = build_V0()[:, 1]
        assert np.all(v2[:NU] > 0)
        assert np.all(v2[NU:] < 0)


class TestBuildC0:
    def setup_method(self):
        self.V0 = build_V0()
        rng = np.random.default_rng(42)
        A = rng.standard_normal((N, N))
        Cfull = A @ A.T / N
        np.fill_diagonal(Cfull, 1.0)
        self.Cfull = Cfull

    def test_shape(self):
        C0 = build_C0(self.V0, self.Cfull)
        assert C0.shape == (N, N)

    def test_diagonal_ones(self):
        C0 = build_C0(self.V0, self.Cfull)
        np.testing.assert_allclose(np.diag(C0), 1.0, atol=1e-10)

    def test_symmetric(self):
        C0 = build_C0(self.V0, self.Cfull)
        np.testing.assert_allclose(C0, C0.T, atol=1e-10)

    def test_values_bounded(self):
        C0 = build_C0(self.V0, self.Cfull)
        assert np.all(C0 >= -1.0 - 1e-10)
        assert np.all(C0 <= 1.0 + 1e-10)


class TestPerfMetrics:
    def test_ar_calculation(self):
        """定数リターンの AR が正しく計算される"""
        r = pd.Series([0.01] * 252)
        metrics = perf_metrics(pd.DataFrame({"A": r}))
        np.testing.assert_allclose(metrics.loc["A", "AR(%)"], 0.01 * 252 * 100, rtol=1e-5)

    def test_rr_equals_ar_divided_by_risk(self):
        """R/R が round(AR_raw / RISK_raw, 2) と一致する"""
        rng = np.random.default_rng(2)
        r = pd.Series(rng.standard_normal(252) * 0.01)
        metrics = perf_metrics(pd.DataFrame({"X": r}))
        raw_ar = r.mean() * 252 * 100
        raw_risk = r.std() * np.sqrt(252) * 100
        assert metrics.loc["X", "R/R"] == round(raw_ar / raw_risk, 2)

    def test_mdd_nonnegative(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.standard_normal(200) * 0.01)
        metrics = perf_metrics(pd.DataFrame({"X": r}))
        assert metrics.loc["X", "MDD(%)"] >= 0.0

    def test_output_columns(self):
        r = pd.Series(np.random.default_rng(0).standard_normal(100) * 0.01)
        metrics = perf_metrics(pd.DataFrame({"X": r}))
        assert set(metrics.columns) == {"AR(%)", "RISK(%)", "R/R", "MDD(%)"}

    def test_empty_series_excluded(self):
        """リターンが空の系列は出力から除外される"""
        rets = pd.DataFrame({"A": pd.Series(dtype=float), "B": pd.Series([0.01] * 50)})
        metrics = perf_metrics(rets)
        assert "A" not in metrics.index
        assert "B" in metrics.index


class TestRunBacktest:
    """合成データで run_backtest の基本動作を検証（yfinance 不使用）"""

    @staticmethod
    def _make_synthetic(n=200, seed=0):
        rng = np.random.default_rng(seed)
        us_dates = pd.bdate_range("2016-01-04", periods=n, freq="B")
        jp_dates = pd.bdate_range("2016-01-04", periods=n + 1, freq="B")  # 末尾に 1 日余分

        us_cc = pd.DataFrame(
            rng.standard_normal((n, NU)) * 0.01, index=us_dates, columns=US_TICKERS
        )
        jp_cc = pd.DataFrame(
            rng.standard_normal((n + 1, NJ)) * 0.01, index=jp_dates, columns=JP_TICKERS
        )
        jp_oc = pd.DataFrame(
            rng.standard_normal((n + 1, NJ)) * 0.01, index=jp_dates, columns=JP_TICKERS
        )
        return us_cc, jp_cc, jp_oc

    def test_output_columns(self):
        us_cc, jp_cc, jp_oc = self._make_synthetic()
        rets = run_backtest(us_cc, jp_cc, jp_oc, L=30, cfull_end="2015-12-31")
        assert set(rets.columns) == {"MOM", "PCA_PLAIN", "PCA_SUB", "DOUBLE"}

    def test_output_nonempty(self):
        us_cc, jp_cc, jp_oc = self._make_synthetic()
        rets = run_backtest(us_cc, jp_cc, jp_oc, L=30, cfull_end="2015-12-31")
        assert len(rets) > 0

    def test_on_progress_called(self):
        us_cc, jp_cc, jp_oc = self._make_synthetic()
        calls = []
        run_backtest(
            us_cc,
            jp_cc,
            jp_oc,
            L=30,
            cfull_end="2015-12-31",
            on_progress=lambda s, t: calls.append((s, t)),
        )
        assert len(calls) > 0

    def test_lam_zero_vs_one_differ(self):
        """λ=0（正則化なし）と λ=1（事前のみ）で PCA_SUB シグナルが異なる"""
        us_cc, jp_cc, jp_oc = self._make_synthetic()
        r0 = run_backtest(us_cc, jp_cc, jp_oc, L=30, lam=0.0, cfull_end="2015-12-31")
        r1 = run_backtest(us_cc, jp_cc, jp_oc, L=30, lam=1.0, cfull_end="2015-12-31")
        assert not r0["PCA_SUB"].equals(r1["PCA_SUB"])
