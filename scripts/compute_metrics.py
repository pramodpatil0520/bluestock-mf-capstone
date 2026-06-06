import logging
from pathlib import Path
 
import numpy as np
import pandas as pd

from scipy import stats
from sqlalchemy import create_engine
 
# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
# ── paths & constants ──────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
DB_PATH   = BASE_DIR / "data" / "db" / "bluestock_mf.db"
PROC_DIR.mkdir(parents=True, exist_ok=True)
 
# RBI repo rate used as risk-free rate proxy (as of mid-2025)
RISK_FREE_ANNUAL = 0.065
RISK_FREE_DAILY  = RISK_FREE_ANNUAL / 252
 
# annualisation factor for daily returns
ANNUALISE = 252
 
 
# ==============================================================================
# Data loading
# ==============================================================================
 
def load_data():
    """
    Pull what we need straight from the SQLite database.
    Running the ETL pipeline first is a prerequisite.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}.\n"
            "Run etl_pipeline.py first."
        )
 
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
 
    nav_df  = pd.read_sql("SELECT amfi_code, date, nav, daily_return_pct FROM fact_nav", engine)
    bench_df = pd.read_sql("SELECT date, index_name, close_value FROM fact_benchmark", engine)
    fund_df  = pd.read_sql("SELECT amfi_code, scheme_name, category, risk_category FROM dim_fund", engine)
 
    nav_df["date"]   = pd.to_datetime(nav_df["date"])
    bench_df["date"] = pd.to_datetime(bench_df["date"])
 
    log.info(f"Loaded NAV history: {len(nav_df):,} rows across {nav_df['amfi_code'].nunique()} schemes")
    log.info(f"Loaded benchmark:   {len(bench_df):,} rows")
 
    return nav_df, bench_df, fund_df
 
 
# ==============================================================================
# Return computation
# ==============================================================================
 
def compute_daily_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute daily returns from scratch rather than trusting the stored column
    (gives us a clean numeric series with no string artefacts).
    """
    nav_df = nav_df.sort_values(["amfi_code", "date"]).copy()
    nav_df["return"] = nav_df.groupby("amfi_code")["nav"].pct_change()
    # first trading day per scheme → NaN, that's expected
    returns_df = nav_df.dropna(subset=["return"]).copy()
    log.info(f"Daily returns computed: {len(returns_df):,} rows")
    returns_df.to_csv(PROC_DIR / "returns_computed.csv", index=False)
    return returns_df
 
 
# ==============================================================================
# CAGR
# ==============================================================================
 
def cagr(start_nav: float, end_nav: float, years: float) -> float:
    if start_nav <= 0 or years <= 0:
        return np.nan
    return (end_nav / start_nav) ** (1 / years) - 1
 
 
def compute_cagr_report(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each scheme compute 1-year, 3-year and 5-year CAGR.
    We work backwards from the latest available NAV date.
    """
    results = []
    latest_date = nav_df["date"].max()
 
    for code, grp in nav_df.groupby("amfi_code"):
        grp = grp.sort_values("date")
        latest_nav = grp.iloc[-1]["nav"]
 
        def get_nav_years_ago(n):
            target = latest_date - pd.DateOffset(years=n)
            sub    = grp[grp["date"] <= target]
            return sub.iloc[-1]["nav"] if not sub.empty else None
 
        nav_1y = get_nav_years_ago(1)
        nav_3y = get_nav_years_ago(3)
        nav_5y = get_nav_years_ago(5)
 
        results.append({
            "amfi_code":      code,
            "cagr_1yr_pct":   round(cagr(nav_1y, latest_nav, 1) * 100, 2) if nav_1y else np.nan,
            "cagr_3yr_pct":   round(cagr(nav_3y, latest_nav, 3) * 100, 2) if nav_3y else np.nan,
            "cagr_5yr_pct":   round(cagr(nav_5y, latest_nav, 5) * 100, 2) if nav_5y else np.nan,
            "latest_nav":     round(latest_nav, 4),
            "as_of_date":     latest_date.strftime("%Y-%m-%d"),
        })
 
    df = pd.DataFrame(results)
    df.to_csv(PROC_DIR / "cagr_report.csv", index=False)
    log.info(f"CAGR report saved → {len(df)} schemes")
    return df
 
 
# ==============================================================================
# Sharpe Ratio
# ==============================================================================
 
def compute_sharpe(returns: pd.Series) -> float:
    """
    Sharpe = (mean_daily_return - risk_free_daily) / std_daily * sqrt(252)
    Using the excess return formulation, annualised.
    """
    excess = returns - RISK_FREE_DAILY
    if returns.std() == 0:
        return np.nan
    return float((excess.mean() / returns.std()) * np.sqrt(ANNUALISE))
 
 
def compute_sharpe_all(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, grp in returns_df.groupby("amfi_code"):
        sharpe = compute_sharpe(grp["return"])
        rows.append({"amfi_code": code, "sharpe_ratio": round(sharpe, 4)})
 
    df = pd.DataFrame(rows)
    df.to_csv(PROC_DIR / "sharpe_values.csv", index=False)
    log.info("Sharpe ratios computed.")
    return df
 
 
# ==============================================================================
# Sortino Ratio
# ==============================================================================
 
def compute_sortino(returns: pd.Series) -> float:
    """
    Sortino uses only downside deviation in the denominator — penalises
    negative volatility but not upside volatility.  Better metric for
    funds with asymmetric return distributions (e.g. sector funds).
    """
    excess = returns - RISK_FREE_DAILY
    downside_diff = np.minimum(excess, 0)
    if len(excess) < 2:
        return np.nan
    # Correct downside deviation formula: penalizes relative to target (RFR), normalized by total observations
    downside_std = np.sqrt(np.sum(downside_diff ** 2) / (len(excess) - 1))
    if downside_std == 0:
        return np.nan
    return float((excess.mean() / downside_std) * np.sqrt(ANNUALISE))
 
 
def compute_sortino_all(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, grp in returns_df.groupby("amfi_code"):
        sortino = compute_sortino(grp["return"])
        rows.append({"amfi_code": code, "sortino_ratio": round(sortino, 4)})
 
    df = pd.DataFrame(rows)
    df.to_csv(PROC_DIR / "sortino_values.csv", index=False)
    log.info("Sortino ratios computed.")
    return df
 
 
# ==============================================================================
# Alpha & Beta  (OLS vs benchmark)
# ==============================================================================
 
def compute_alpha_beta(returns_df: pd.DataFrame, bench_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fund regress daily returns on Nifty 100 daily returns.
      Beta  = slope  (sensitivity to market)
      Alpha = intercept * 252  (annualised excess return above what Beta explains)
 
    We use Nifty 100 as benchmark for Large Cap funds; if not available in the
    table we fall back to Nifty 50.
    """
    # get Nifty 100 daily returns
    # column names may vary — handle both 'close_value' and 'close'
    close_col = "close_value" if "close_value" in bench_df.columns else "close"
 
    nifty100 = bench_df[bench_df["index_name"].str.contains("100", case=False, na=False)].copy()
 
    if nifty100.empty:
        log.warning("Nifty 100 not found in benchmark data — falling back to Nifty 50")
        nifty100 = bench_df[bench_df["index_name"].str.contains("50", case=False, na=False)].copy()
 
    if nifty100.empty:
        log.error("No benchmark data available for Alpha/Beta computation.")
        return pd.DataFrame()
 
    nifty100 = nifty100.sort_values("date")
    nifty100["bench_return"] = nifty100[close_col].pct_change()
    nifty100 = nifty100.dropna(subset=["bench_return"])
 
    rows = []
    for code, grp in returns_df.groupby("amfi_code"):
        merged = grp[["date", "return"]].merge(
            nifty100[["date", "bench_return"]], on="date", how="inner"
        )
 
        if len(merged) < 30:
            # need a minimum sample size for a meaningful regression
            rows.append({
                "amfi_code": code, "alpha_annualised": np.nan,
                "beta": np.nan, "r_squared": np.nan, "obs_count": len(merged)
            })
            continue
 
        slope, intercept, r_val, _, _ = stats.linregress(
            merged["bench_return"], merged["return"]
        )
 
        rows.append({
            "amfi_code":         code,
            "alpha_annualised":  round(intercept * ANNUALISE * 100, 4),   # annualised %
            "beta":              round(slope, 4),
            "r_squared":         round(r_val ** 2, 4),
            "obs_count":         len(merged),
        })
 
    df = pd.DataFrame(rows)
    df.to_csv(PROC_DIR / "alpha_beta.csv", index=False)
    log.info(f"Alpha/Beta computed for {len(df)} schemes.")
    return df
 
 
# ==============================================================================
# Maximum Drawdown
# ==============================================================================
 
def max_drawdown(nav_series: pd.Series) -> float:
    """
    Max drawdown = worst peak-to-trough decline over the observation period.
    Returns a negative number (e.g. -0.32 = 32% drawdown).
    """
    running_max = nav_series.cummax()
    drawdown    = nav_series / running_max - 1
    return float(drawdown.min())
 
 
def compute_max_drawdown_all(nav_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, grp in nav_df.groupby("amfi_code"):
        grp = grp.sort_values("date")
        mdd = max_drawdown(grp["nav"])
 
        # also find the dates of the peak and the trough
        running_max = grp["nav"].cummax()
        dd_series   = grp["nav"] / running_max - 1
        trough_idx  = dd_series.idxmin()
        trough_date = grp.loc[trough_idx, "date"] if trough_idx in grp.index else None
 
        rows.append({
            "amfi_code":          code,
            "max_drawdown_pct":   round(mdd * 100, 2),
            "trough_date":        trough_date.strftime("%Y-%m-%d") if trough_date is not None else None,
        })
 
    df = pd.DataFrame(rows)
    df.to_csv(PROC_DIR / "max_drawdown.csv", index=False)
    log.info("Max drawdown computed.")
    return df
 
 
# ==============================================================================
# VaR & CVaR
# ==============================================================================
 
def compute_var_cvar(returns_df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """
    Historical VaR at 95% confidence:
      VaR  = 5th percentile of daily return distribution
    CVaR (Expected Shortfall):
      Mean of all returns worse than the VaR threshold.
 
    Both are reported as percentage losses (positive = loss magnitude).
    """
    rows = []
    for code, grp in returns_df.groupby("amfi_code"):
        r = grp["return"].dropna()
        if len(r) < 50:
            continue
 
        var_threshold = np.percentile(r, (1 - confidence) * 100)   # e.g. 5th pct
        cvar          = r[r <= var_threshold].mean()
 
        rows.append({
            "amfi_code":           code,
            "var_95_daily_pct":    round(var_threshold * 100, 4),   # negative number
            "cvar_95_daily_pct":   round(cvar * 100, 4),
            "var_95_annual_pct":   round(var_threshold * np.sqrt(252) * 100, 4),
        })
 
    df = pd.DataFrame(rows)
    df.to_csv(PROC_DIR / "var_cvar_report.csv", index=False)
    log.info(f"VaR/CVaR computed for {len(df)} schemes.")
    return df
 
 
# ==============================================================================
# Rolling Sharpe (90-day window)
# ==============================================================================
 
def compute_rolling_sharpe(returns_df: pd.DataFrame,
                            window: int = 90,
                            top_n: int = 5) -> pd.DataFrame:
    """
    Rolling Sharpe is useful for spotting periods when a fund's
    risk-adjusted performance deteriorated even if annual Sharpe looks fine.
    We save the rolling series for the top_n funds by count of NAV records.
    """
    # pick the top_n funds with most data (usually the older ones)
    top_funds = (
        returns_df.groupby("amfi_code")["return"]
                  .count()
                  .nlargest(top_n)
                  .index.tolist()
    )
 
    all_rolling = []
    for code in top_funds:
        grp = returns_df[returns_df["amfi_code"] == code].sort_values("date")
        r   = grp.set_index("date")["return"]
 
        rolling_mean = r.rolling(window, min_periods=window // 2).mean()
        rolling_std  = r.rolling(window, min_periods=window // 2).std()
        rolling_sharpe = (rolling_mean - RISK_FREE_DAILY) / rolling_std * np.sqrt(ANNUALISE)
 
        tmp = pd.DataFrame({
            "amfi_code":      code,
            "date":           r.index,
            "rolling_sharpe": rolling_sharpe.values,
        })
        all_rolling.append(tmp)
 
    df = pd.concat(all_rolling, ignore_index=True)
    df.dropna(subset=["rolling_sharpe"], inplace=True)
    df.to_csv(PROC_DIR / "rolling_sharpe.csv", index=False)
    log.info(f"Rolling {window}-day Sharpe saved for {len(top_funds)} funds.")
    return df
 
 
# ==============================================================================
# Fund Scorecard
# ==============================================================================
 
def build_fund_scorecard(cagr_df, sharpe_df, ab_df, mdd_df, fund_df) -> pd.DataFrame:
    """
    Composite score per the rubric in the project spec:
      30% × 3yr return rank  (higher is better)
      25% × Sharpe rank      (higher is better)
      20% × Alpha rank       (higher is better)
      15% × Expense ratio rank  (lower is better → inverse rank)
      10% × Max drawdown rank   (less negative is better → inverse rank)
 
    All metrics are converted to percentile ranks (0-100) within each
    category before weighting so different scales don't dominate.
    """
    # pull expense ratio from dim_fund
    expense = fund_df[["amfi_code", "scheme_name", "category", "risk_category"]].copy()
 
    engine = create_engine(f"sqlite:///{BASE_DIR / 'data' / 'db' / 'bluestock_mf.db'}", echo=False)
    er_df  = pd.read_sql("SELECT amfi_code, expense_ratio_pct FROM dim_fund", engine)
 
    merged = (
        expense
        .merge(er_df,      on="amfi_code", how="left")
        .merge(cagr_df[["amfi_code", "cagr_3yr_pct"]],   on="amfi_code", how="left")
        .merge(sharpe_df,  on="amfi_code", how="left")
        .merge(ab_df[["amfi_code", "alpha_annualised"]],  on="amfi_code", how="left")
        .merge(mdd_df[["amfi_code", "max_drawdown_pct"]], on="amfi_code", how="left")
    )
 
    n = len(merged)
    if n == 0:
        log.error("No data to build scorecard.")
        return pd.DataFrame()
 
    def pct_rank(series, ascending=True):
        """Rank as 0-100 percentile.  ascending=True → higher value = higher rank."""
        r = series.rank(method="average", ascending=ascending, na_option="bottom")
        return ((r - 1) / max(n - 1, 1)) * 100
 
    merged["rank_3yr_return"]  = pct_rank(merged["cagr_3yr_pct"],     ascending=True)
    merged["rank_sharpe"]      = pct_rank(merged["sharpe_ratio"],      ascending=True)
    merged["rank_alpha"]       = pct_rank(merged["alpha_annualised"],  ascending=True)
    merged["rank_expense"]     = pct_rank(merged["expense_ratio_pct"], ascending=False)  # lower = better
    merged["rank_mdd"]         = pct_rank(merged["max_drawdown_pct"],  ascending=False)  # less neg = better
 
    merged["composite_score"] = (
        0.30 * merged["rank_3yr_return"] +
        0.25 * merged["rank_sharpe"]     +
        0.20 * merged["rank_alpha"]      +
        0.15 * merged["rank_expense"]    +
        0.10 * merged["rank_mdd"]
    ).round(2)
 
    merged.sort_values("composite_score", ascending=False, inplace=True)
    merged["rank_overall"] = range(1, len(merged) + 1)
 
    out_cols = [
        "rank_overall", "amfi_code", "scheme_name", "category", "sub_category", "risk_category",
        "composite_score", "cagr_3yr_pct", "sharpe_ratio",
        "alpha_annualised", "expense_ratio_pct", "max_drawdown_pct",
    ]
    scorecard = merged[[c for c in out_cols if c in merged.columns]]
    scorecard.to_csv(PROC_DIR / "fund_scorecard.csv", index=False)
    log.info(f"Fund scorecard saved → {len(scorecard)} schemes ranked.")
    return scorecard
 
 
# ==============================================================================
# MAIN
# ==============================================================================
 
def run_all():
    log.info("════════════════════════════════════════════════")
    log.info("  Bluestock MF — Metrics Computation")
    log.info("════════════════════════════════════════════════")
 
    nav_df, bench_df, fund_df = load_data()
 
    log.info("Computing daily returns ...")
    returns_df = compute_daily_returns(nav_df)
 
    log.info("Computing CAGR ...")
    cagr_df = compute_cagr_report(nav_df)
 
    log.info("Computing Sharpe ratios ...")
    sharpe_df = compute_sharpe_all(returns_df)
 
    log.info("Computing Sortino ratios ...")
    compute_sortino_all(returns_df)
 
    log.info("Computing Alpha & Beta ...")
    ab_df = compute_alpha_beta(returns_df, bench_df)
 
    log.info("Computing Maximum Drawdown ...")
    mdd_df = compute_max_drawdown_all(nav_df)
 
    log.info("Computing VaR / CVaR ...")
    compute_var_cvar(returns_df)
 
    log.info("Computing rolling Sharpe ...")
    compute_rolling_sharpe(returns_df)
 
    log.info("Building fund scorecard ...")
    scorecard = build_fund_scorecard(cagr_df, sharpe_df, ab_df, mdd_df, fund_df)
 
    log.info("════════════════════════════════════════════════")
    log.info("Top 5 funds by composite score:")
    if not scorecard.empty:
        top5_cols = ["rank_overall", "scheme_name", "composite_score", "cagr_3yr_pct", "sharpe_ratio"]
        available = [c for c in top5_cols if c in scorecard.columns]
        print(scorecard[available].head().to_string(index=False))
    log.info("All metric files written to: %s", PROC_DIR)
    log.info("════════════════════════════════════════════════")
 
 
if __name__ == "__main__":
    run_all()
 