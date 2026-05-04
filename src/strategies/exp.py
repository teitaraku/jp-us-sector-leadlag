"""
実験的戦略モジュール — 動的 V₀ による完全データ駆動型 PCA SUB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 論文手法との違い（pca_sub.py との比較）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

論文版（pca_sub.py）では、事前部分空間 V₀ を人間が定義した業種ラベルで
手動構築している。具体的には：

    v₁ = グローバル共通ファクター（全28銘柄を均等ウェイト）
    v₂ = 米国プラス／日本マイナスの国スプレッド
    v₃ = シクリカル業種プラス／ディフェンシブ業種マイナス

これらは「米国と日本の業種間にはグローバルファクター・国差・
景気感応度の3つの共通構造がある」という事前知識を反映したもの。

本モジュールでは、この事前知識をデータから自動学習させる。
直近 V0_YEARS 年の日米業種リターンに PCA を適用し、
分散を最もよく説明する上位3つの主成分ベクトルを V₀ として使用する。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ アルゴリズム全体の流れ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【年次更新フェーズ（暦年が変わるタイミングで実行）】

Step 1. 処理中の暦年 Y の 1月1日を基準日として、直前 CFULL_WINDOW_YEARS 年
        （デフォルト: 5年）のデータで日米28銘柄の相関行列 Cfull を推定する。

            Cfull 推定期間 = [Y-5年1月1日, Y年1月1日)

        基準日以降のデータは一切使わないため、先読みバイアスが生じない。
        データ不足時は利用可能な全履歴にフォールバックする。

Step 2. 直近 V0_YEARS 年（デフォルト: 2年）のリターンデータに PCA を適用し、
        上位3固有ベクトルを V₀ とする。

            corr_v0 = Σ_t r_t r_t^T / T  （直近2年の相関行列）
            [λ₁≥λ₂≥λ₃, v₁, v₂, v₃] = 固有分解(corr_v0) の上位3成分

        論文の手動 V₀ と異なり、v₁ が必ずしも「均等ウェイト」には
        ならない。実際の市場で最も分散を説明する方向が自動で選ばれる。

Step 3. C₀ = V₀ · diag(V₀ᵀ Cfull V₀) · V₀ᵀ  を正規化して事前相関行列を作る。

【取引フェーズ（米国取引日ごとに実行）】

Step 4. 直近 L 営業日（デフォルト: 60日）のローリングウィンドウで
        当日の相関行列 Ct を推定する。

Step 5. 正則化相関行列を合成する：
            C_reg = (1 − λ)·Ct + λ·C₀

        λ（デフォルト: 0.9）が大きいほど C₀ に強く引き寄せられ、
        短期ノイズの影響を抑える。論文の部分空間正則化の核心部分。

Step 6. C_reg を固有分解し、上位 K 本（デフォルト: 3）の固有ベクトル Vk を得る。

            C_reg · Vk[:,i] = μᵢ · Vk[:,i]  （i=1,...,K）

Step 7. リードラグシグナルを計算する：

            z_JP = Vk[JP, :] · (Vk[US, :]ᵀ · z_US_today)

Step 8. z_JP の上位 q 分位を買い・下位 q 分位を売りにして
        ロングショートリターンを計算する。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 動的 V₀ のメリットと注意点
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【メリット】
・業種分類の恣意性を排除できる（どの銘柄がシクリカルかはデータが決める）
・Cfull を年次ローリングで更新するため先読みバイアスがない
・業種間の相関構造が時代とともに変化しても追従できる
  （例: テクノロジー業種の国際的連動性の高まりなど）
・将来、米国または日本のティッカーを入れ替えても再実装不要

【注意点】
・PCA の固有ベクトルには符号の不定性がある（+v と −v は同じ部分空間）。
  C₀ の構築は V₀ V₀ᵀ の射影で行われるため、符号は問題にならない。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import linalg

from src.strategies.core import (
    JP_TICKERS,
    MIN_CFULL_OBS,
    NJ,
    NU,
    US_TICKERS,
    N,
    StepContext,
    _fill_corr_missing,
    _joint_cc,
    _window_state,
    build_C0,
    ls_ret,
)

# V₀ を PCA で推定する直近データの年数
V0_YEARS = 2
# Cfull（長期相関行列）の推定ウィンドウ年数
CFULL_WINDOW_YEARS = 5


def _build_dynamic_V0(data: pd.DataFrame) -> np.ndarray:
    """
    data（N次元の日次リターン DataFrame）の相関行列を固有分解し、
    分散説明量上位3本の固有ベクトルを列に持つ N×3 行列を返す。
    """
    identity = np.eye(N)
    corr = data.corr(min_periods=max(3, len(data) // 2)).values
    corr = _fill_corr_missing(corr, identity)

    eigvals, eigvecs = linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order[:3]]


def _build_prior_for_date(
    all_cc: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """
    as_of_date を基準に直前 CFULL_WINDOW_YEARS 年間のデータから V0 と C0 を構築する。

    as_of_date のデータは含まない（< as_of_date）ため先読みバイアスが生じない。
    データ不足時は as_of_date より前の全データにフォールバックする。
    """
    cfull_start_ts = as_of_date - pd.DateOffset(years=CFULL_WINDOW_YEARS)
    cfull_mask = (all_cc.index >= cfull_start_ts) & (all_cc.index < as_of_date)
    cfull_data = all_cc[cfull_mask]

    if len(cfull_data) < MIN_CFULL_OBS:
        cfull_data = all_cc[all_cc.index < as_of_date]

    if len(cfull_data) < MIN_CFULL_OBS:
        raise ValueError(
            f"Cfull 推定用データが不足しています（基準日: {as_of_date.date()}）。"
            f"少なくとも {MIN_CFULL_OBS} 行必要ですが現在 {len(cfull_data)} 行です。"
        )

    identity = np.eye(N)
    Cfull = _fill_corr_missing(
        cfull_data.corr(min_periods=min(MIN_CFULL_OBS, len(cfull_data))).values,
        identity,
    )
    np.fill_diagonal(Cfull, 1.0)

    v0_start_ts = as_of_date - pd.DateOffset(years=V0_YEARS)
    v0_data = cfull_data[cfull_data.index >= v0_start_ts]
    if len(v0_data) < MIN_CFULL_OBS:
        v0_data = cfull_data

    V0 = _build_dynamic_V0(v0_data)
    C0 = build_C0(V0, Cfull)
    return V0, C0


# ── シグナル計算（論文の pca_sub.py と同一のロジック）─────────────────────────
def compute_signal(ctx: StepContext) -> np.ndarray:
    """
    正則化相関行列 C_reg から上位 K 固有ベクトルを取り出し、
    米国リターンの z スコアを日本業種予測スコアに変換する。

    C_reg = (1 − λ)·Ct + λ·C₀
    z_JP = Vk[JP] @ (Vk[US]ᵀ @ z_US)
    """
    try:
        C_reg = (1.0 - ctx.lam) * ctx.Ct + ctx.lam * ctx.C0
        eigvals_r, eigvecs_r = linalg.eigh(C_reg)
        order_r = np.argsort(eigvals_r)[::-1]
        Vk_r = eigvecs_r[:, order_r[: ctx.K]]
        return Vk_r[NU:] @ (Vk_r[:NU].T @ ctx.z_us)
    except Exception:
        return np.zeros(NJ)


def compute_return(ctx: StepContext) -> float:
    """シグナルからロングショートリターンを計算する（バックテスト用）。"""
    return ls_ret(compute_signal(ctx), ctx.target, ctx.q)


# ── バックテストループ（年次ローリング C0 更新）──────────────────────────────
def _run_loop(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    all_cc: pd.DataFrame,
    L: int,
    lam: float,
    K: int,
    q: float,
    strategies: dict[str, Callable[[StepContext], float]],
    on_progress: Callable[[int, int], None] | None = None,
    backtest_start: str | None = None,
) -> pd.DataFrame:
    """
    米国取引日ごとに次の日本取引日を対応させ、各戦略のロングショートリターンを返す。

    暦年が変わるタイミングで _build_prior_for_date を呼び出し、
    Cfull・V0・C0 を直前 CFULL_WINDOW_YEARS 年のデータで再推定する。
    """
    jp_dates_arr = jp_oc.index.values
    us_dates = us_cc[US_TICKERS].dropna(how="all").index
    backtest_start_ts = pd.Timestamp(backtest_start) if backtest_start else None

    paired: list[tuple] = []
    for us_date in us_dates:
        if backtest_start_ts is not None and us_date < backtest_start_ts:
            continue
        future = jp_dates_arr[jp_dates_arr > us_date]
        if len(future) == 0:
            continue
        next_jp = future[0]
        if next_jp in jp_oc.index:
            paired.append((us_date, next_jp))

    results = {s: [] for s in strategies}
    dates_out: list = []
    n_pairs = len(paired)

    current_year: int | None = None
    C0: np.ndarray | None = None

    for step, (us_date, jp_date) in enumerate(paired):
        if on_progress and step % max(n_pairs // 80, 1) == 0:
            on_progress(step, n_pairs)

        # 暦年が変わったら C0 を再推定
        us_year = pd.Timestamp(us_date).year
        if us_year != current_year:
            as_of = pd.Timestamp(f"{us_year}-01-01")
            try:
                _, C0 = _build_prior_for_date(all_cc, as_of)
                current_year = us_year
            except ValueError:
                # データ不足の場合は current_year を更新しない（毎ステップ再試行）
                pass

        if C0 is None:
            continue

        t_idx = all_cc.index.searchsorted(us_date, side="left")
        if t_idx < L:
            continue

        window = all_cc.iloc[t_idx - L : t_idx]
        if window.isna().to_numpy().mean() > 0.3:
            continue

        mu, sigma, Ct = _window_state(window, C0)

        us_today = us_cc.loc[us_date, US_TICKERS].values.astype(float)
        z_us = (us_today - mu[:NU]) / sigma[:NU]
        z_us = np.where(np.isfinite(z_us), z_us, 0.0)

        target_row = jp_oc.loc[jp_date, JP_TICKERS]
        target = target_row.values.astype(float)
        if np.isnan(target).mean() > 0.5:
            continue

        ctx = StepContext(
            mu=mu,
            sigma=sigma,
            z_us=z_us,
            Ct=Ct,
            C0=C0,
            target=target,
            K=K,
            lam=lam,
            q=q,
        )

        for name, fn in strategies.items():
            try:
                results[name].append(fn(ctx))
            except Exception:
                results[name].append(np.nan)
        dates_out.append(us_date)

    return pd.DataFrame(results, index=dates_out).dropna(how="all")


def run_backtest(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int = 60,
    lam: float = 0.9,
    K: int = 3,
    q: float = 0.30,
    on_progress: Callable[[int, int], None] | None = None,
    backtest_start: str | None = None,
) -> pd.DataFrame:
    all_cc = _joint_cc(us_cc, jp_cc)
    if all_cc.empty:
        raise ValueError("日米の Close-to-Close リターンが対応するデータがありません。")
    return _run_loop(
        us_cc,
        jp_cc,
        jp_oc,
        all_cc=all_cc,
        L=L,
        lam=lam,
        K=K,
        q=q,
        strategies={"PCA_SUB": compute_return},
        on_progress=on_progress,
        backtest_start=backtest_start,
    )


def run_live_signal(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    L: int = 60,
    lam: float = 0.9,
    K: int = 3,
    q: float = 0.30,
) -> dict:
    all_cc = _joint_cc(us_cc, jp_cc)
    if all_cc.empty:
        raise ValueError("日米の Close-to-Close リターンが対応するデータがありません。")

    us_valid = us_cc[US_TICKERS].dropna(how="all")
    if len(us_valid) == 0:
        raise ValueError("米国リターンデータがありません")

    latest_us_date = us_valid.index[-1]
    _, C0 = _build_prior_for_date(all_cc, latest_us_date)

    t_idx = all_cc.index.searchsorted(latest_us_date, side="left")
    all_cc_window = all_cc.iloc[:t_idx]
    if len(all_cc_window) < L:
        raise ValueError(f"ウィンドウ計算に必要なデータが不足しています（必要: {L}日）")

    window = all_cc_window.iloc[-L:]
    mu, sigma, Ct = _window_state(window, C0)

    us_today_row = us_valid.loc[latest_us_date]
    z_us = (us_today_row.values.astype(float) - mu[:NU]) / sigma[:NU]
    z_us = np.where(np.isfinite(z_us), z_us, 0.0)

    ctx = StepContext(
        mu=mu,
        sigma=sigma,
        z_us=z_us,
        Ct=Ct,
        C0=C0,
        target=None,
        K=K,
        lam=lam,
        q=q,
    )
    z_jp = compute_signal(ctx)
    signal = pd.Series(z_jp, index=JP_TICKERS)
    n_each = max(1, int(NJ * q))

    jp_dates = jp_oc.index
    future_jp = jp_dates[jp_dates > latest_us_date]
    next_jp_date = future_jp[0] if len(future_jp) > 0 else None

    return {
        "signal": signal,
        "us_date": latest_us_date,
        "next_jp_date": next_jp_date,
        "us_returns": us_today_row,
        "n_long": n_each,
        "n_short": n_each,
    }
