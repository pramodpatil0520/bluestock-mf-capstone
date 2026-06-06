import argparse
import logging
from pathlib import Path
 
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
 
# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
DB_PATH  = BASE_DIR / "data" / "db" / "bluestock_mf.db"
 
 
# ==============================================================================
# Risk appetite mapping
# ==============================================================================
 
# Maps investor-facing terms → SEBI risk category labels in dim_fund
RISK_MAP = {
    "low":      ["Low", "Low to Moderate"],
    "moderate": ["Moderate", "Low to Moderate", "Moderately High"],
    "high":     ["High", "Moderately High", "Very High"],
    "very high":["Very High", "High"],
}
 
# Investment horizon → minimum recommended fund categories
HORIZON_CATEGORIES = {
    "short":  ["Liquid", "Overnight", "Ultra Short Duration", "Money Market"],
    "medium": ["Short Duration", "Corporate Bond", "Hybrid - Conservative",
               "Hybrid - Balanced", "Large Cap"],
    "long":   ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap",
               "Multi Cap", "ELSS", "Hybrid - Aggressive"],
}
 
 
# ==============================================================================
# Load fund data
# ==============================================================================
 
def load_fund_universe() -> pd.DataFrame:
    """
    Pull the fund master + pre-computed metrics from the database.
    Falls back to CSV files in data/processed/ if the DB isn't available.
    """
    scorecard_path = PROC_DIR / "fund_scorecard.csv"
    sharpe_path    = PROC_DIR / "sharpe_values.csv"
    cagr_path      = PROC_DIR / "cagr_report.csv"
 
    # try DB first
    if DB_PATH.exists():
        engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
        fund_df = pd.read_sql(
            """
            SELECT f.amfi_code, f.scheme_name, f.fund_house, f.category,
                   f.sub_category, f.plan, f.expense_ratio_pct,
                   f.risk_category, f.benchmark
            FROM dim_fund f
            """,
            engine
        )
 
        try:
            perf_df = pd.read_sql(
                """
                SELECT amfi_code, return_1yr_pct, return_3yr_pct, return_5yr_pct,
                       sharpe_ratio, sortino_ratio, alpha, beta, max_drawdown_pct
                FROM fact_performance
                """,
                engine
            )
            fund_df = fund_df.merge(perf_df, on="amfi_code", how="left")
            log.info(f"Fund universe loaded from DB: {len(fund_df)} schemes")
        except Exception:
            log.warning("fact_performance table not found in DB — loading from CSV")
            fund_df = _load_from_csvs(fund_df)
 
        return fund_df
 
    # fallback to CSV
    if scorecard_path.exists():
        df = pd.read_csv(scorecard_path)
        log.info(f"Fund universe loaded from scorecard CSV: {len(df)} schemes")
        return df
 
    raise FileNotFoundError(
        "Neither the SQLite database nor the processed CSVs were found.\n"
        "Run etl_pipeline.py and compute_metrics.py first."
    )
 
 
def _load_from_csvs(fund_df: pd.DataFrame) -> pd.DataFrame:
    """Helper: merge scorecard/sharpe CSVs into fund_df when DB perf table is missing."""
    if (PROC_DIR / "fund_scorecard.csv").exists():
        score_df = pd.read_csv(PROC_DIR / "fund_scorecard.csv")
        keep = [c for c in score_df.columns if c not in fund_df.columns or c == "amfi_code"]
        fund_df = fund_df.merge(score_df[keep], on="amfi_code", how="left")
 
    if (PROC_DIR / "sharpe_values.csv").exists():
        sh = pd.read_csv(PROC_DIR / "sharpe_values.csv")
        if "sharpe_ratio" not in fund_df.columns:
            fund_df = fund_df.merge(sh, on="amfi_code", how="left")
 
    if (PROC_DIR / "cagr_report.csv").exists():
        cagr = pd.read_csv(PROC_DIR / "cagr_report.csv")
        cols = ["amfi_code"] + [c for c in cagr.columns if c not in fund_df.columns]
        fund_df = fund_df.merge(cagr[cols], on="amfi_code", how="left")
 
    return fund_df
 
 
# ==============================================================================
# Core recommendation logic
# ==============================================================================
 
def recommend_funds(
    risk_appetite: str,
    horizon: str = "long",
    plan: str = "Direct",
    top_n: int = 3,
    fund_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Main recommendation function.
 
    Parameters
    ----------
    risk_appetite : str
        One of: Low, Moderate, High, Very High  (case-insensitive)
    horizon : str
        Investment horizon: short / medium / long  (default: long)
    plan : str
        'Direct' or 'Regular'  (default: Direct — lower expense ratio)
    top_n : int
        Number of funds to return  (default: 3)
    fund_universe : pd.DataFrame, optional
        Pre-loaded fund data — pass this to avoid re-reading the DB in loops
 
    Returns
    -------
    pd.DataFrame
        Top N recommended funds with key metrics, ranked by Sharpe ratio.
    """
    risk_key    = risk_appetite.strip().lower()
    horizon_key = horizon.strip().lower()
 
    if risk_key not in RISK_MAP:
        raise ValueError(
            f"Invalid risk_appetite '{risk_appetite}'.  "
            f"Choose from: {list(RISK_MAP.keys())}"
        )
    if horizon_key not in HORIZON_CATEGORIES:
        raise ValueError(
            f"Invalid horizon '{horizon}'.  Choose from: short / medium / long"
        )
 
    if fund_universe is None:
        fund_universe = load_fund_universe()
 
    df = fund_universe.copy()
 
    # ── Step 1: filter by risk category ──────────────────────────────────────
    allowed_risk = RISK_MAP[risk_key]
    risk_col = next(
        (c for c in ["risk_category", "risk_grade"] if c in df.columns), None
    )
    if risk_col:
        df = df[df[risk_col].isin(allowed_risk)]
 
    # ── Step 2: filter by investment horizon (category) ──────────────────────
    allowed_cats = HORIZON_CATEGORIES[horizon_key]
    cat_col = next(
        (c for c in ["sub_category", "category"] if c in df.columns), None
    )
    if cat_col:
        horizon_mask = df[cat_col].str.contains(
            "|".join(allowed_cats), case=False, na=False
        )
        horizon_filtered = df[horizon_mask]
        # if the horizon filter removes everything, fall back to risk filter only
        if len(horizon_filtered) > 0:
            df = horizon_filtered
        else:
            log.info("Horizon filter returned 0 results — using risk filter only")
 
    # ── Step 3: prefer Direct plans ──────────────────────────────────────────
    if "plan" in df.columns and plan:
        plan_filtered = df[df["plan"].str.lower() == plan.lower()]
        if len(plan_filtered) > 0:
            df = plan_filtered
 
    # ── Step 4: rank by Sharpe ratio (primary) ───────────────────────────────
    sharpe_col = next(
        (c for c in ["sharpe_ratio", "composite_score"] if c in df.columns), None
    )
    if sharpe_col:
        df = df.dropna(subset=[sharpe_col])
        df = df.sort_values(sharpe_col, ascending=False)
 
    if len(df) == 0:
        log.warning("No funds match the given criteria.  Returning empty DataFrame.")
        return pd.DataFrame()
 
    # ── Step 5: select output columns ────────────────────────────────────────
    display_cols = [
        "amfi_code", "scheme_name", "fund_house", "sub_category",
        "risk_category", "sharpe_ratio", "cagr_3yr_pct", "return_3yr_pct",
        "alpha_annualised", "alpha", "max_drawdown_pct",
        "expense_ratio_pct", "plan",
    ]
    available = [c for c in display_cols if c in df.columns]
    result    = df[available].head(top_n).reset_index(drop=True)
    result.index += 1   # rank starts at 1
 
    return result
 
 
# ==============================================================================
# Display helpers
# ==============================================================================
 
RISK_DESCRIPTIONS = {
    "low":       "Debt / Liquid funds. Capital preservation. Low volatility.",
    "moderate":  "Balanced / Large Cap. Steady growth with manageable risk.",
    "high":      "Mid/Small Cap, Sectoral. Aiming for higher returns; can tolerate dips.",
    "very high": "Small Cap, Thematic. Long horizon; stomach for sharp swings.",
}
 
HORIZON_DESCRIPTIONS = {
    "short":  "< 1 year  - Liquid / Overnight / Ultra Short funds",
    "medium": "1-3 years - Debt / Hybrid / Large Cap funds",
    "long":   "3+ years  - Equity / ELSS / Multi-Cap funds",
}
 
 
def print_recommendation_table(
    df: pd.DataFrame,
    risk: str,
    horizon: str,
):
    """Pretty-print the recommendation output."""
    hr = "-" * 70
    print(f"\n{hr}")
    print(f"  BLUESTOCK FINTECH - FUND RECOMMENDATIONS")
    print(hr)
    print(f"  Risk Profile  : {risk.title()}")
    print(f"  Description   : {RISK_DESCRIPTIONS.get(risk.lower(), '')}")
    print(f"  Horizon       : {horizon.title()}")
    print(f"  Horizon Info  : {HORIZON_DESCRIPTIONS.get(horizon.lower(), '')}")
    print(hr)
 
    if df.empty:
        print("  No matching funds found.  Try adjusting risk or horizon.")
        print(hr)
        return
 
    # rename for display
    rename = {
        "amfi_code":         "AMFI Code",
        "scheme_name":       "Scheme Name",
        "fund_house":        "Fund House",
        "sub_category":      "Category",
        "risk_category":     "Risk",
        "sharpe_ratio":      "Sharpe",
        "cagr_3yr_pct":      "3yr CAGR%",
        "return_3yr_pct":    "3yr Ret%",
        "alpha_annualised":  "Alpha%",
        "alpha":             "Alpha%",
        "max_drawdown_pct":  "Max DD%",
        "expense_ratio_pct": "Exp.Ratio%",
        "plan":              "Plan",
    }
    display = df.rename(columns=rename)
 
    # limit scheme name width for terminal readability
    if "Scheme Name" in display.columns:
        display["Scheme Name"] = display["Scheme Name"].str[:45]
 
    print(display.to_string())
    print(hr)
    print("  * Sharpe > 1 is generally considered good.")
    print("  * Max drawdown shows worst historical loss from peak.")
    print("  * This is not financial advice. Please consult a SEBI-registered advisor.")
    print(f"{hr}\n")
 
 
# ==============================================================================
# Batch mode — generate recommendations for all 3 risk levels
# ==============================================================================
 
def generate_all_recommendations(top_n: int = 3):
    """
    Useful for the final report — generates a recommendation table
    for Low / Moderate / High risk profiles all at once.
    """
    fund_universe = load_fund_universe()
    results       = {}
 
    for risk in ["low", "moderate", "high"]:
        for horizon in ["short", "medium", "long"]:
            key = f"{risk}_{horizon}"
            try:
                rec = recommend_funds(
                    risk_appetite=risk,
                    horizon=horizon,
                    top_n=top_n,
                    fund_universe=fund_universe
                )
                results[key] = rec
            except Exception as e:
                log.warning(f"Recommendation failed for {key}: {e}")
                results[key] = pd.DataFrame()
 
    # save a combined output
    all_frames = []
    for key, df in results.items():
        if not df.empty:
            risk_h, horizon_h = key.split("_")
            df = df.copy()
            df.insert(0, "horizon", horizon_h)
            df.insert(0, "risk_profile", risk_h)
            all_frames.append(df)
 
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        out = PROC_DIR / "all_recommendations.csv"
        combined.to_csv(out, index=False)
        log.info(f"All recommendations saved → {out}")
 
    return results
 
 
# ==============================================================================
# Interactive mode
# ==============================================================================
 
def interactive_mode():
    print("\n" + "=" * 55)
    print("  Bluestock MF - Fund Recommendation Engine")
    print("=" * 55)
 
    # Risk
    print("\n  Risk appetite:")
    risk_options = ["Low", "Moderate", "High", "Very High"]
    for i, r in enumerate(risk_options, 1):
        print(f"    {i}. {r}")
    while True:
        try:
            choice = int(input("  Enter number (1-4): ").strip())
            if 1 <= choice <= 4:
                risk = risk_options[choice - 1]
                break
        except ValueError:
            pass
        print("  Please enter a number between 1 and 4.")
 
    # Horizon
    print("\n  Investment horizon:")
    horizon_options = ["short", "medium", "long"]
    for i, h in enumerate(horizon_options, 1):
        print(f"    {i}. {HORIZON_DESCRIPTIONS[h]}")
    while True:
        try:
            choice = int(input("  Enter number (1-3): ").strip())
            if 1 <= choice <= 3:
                horizon = horizon_options[choice - 1]
                break
        except ValueError:
            pass
        print("  Please enter a number between 1 and 3.")
 
    # Top N
    raw_n = input("\n  How many funds to recommend? [default 3]: ").strip()
    top_n = int(raw_n) if raw_n.isdigit() else 3
 
    print("\n  Fetching recommendations...")
    try:
        rec = recommend_funds(risk_appetite=risk, horizon=horizon, top_n=top_n)
        print_recommendation_table(rec, risk, horizon)
    except FileNotFoundError as e:
        print(f"\n  Error: {e}")
 
 
# ==============================================================================
# CLI
# ==============================================================================
 
def main():
    parser = argparse.ArgumentParser(
        description="Bluestock MF — Fund Recommendation Engine"
    )
    parser.add_argument(
        "--risk",
        type=str,
        choices=["Low", "Moderate", "High", "Very High"],
        help="Investor risk appetite"
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="long",
        choices=["short", "medium", "long"],
        help="Investment horizon (default: long)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of funds to recommend (default: 3)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate recommendations for all risk/horizon combos and save to CSV"
    )
    args = parser.parse_args()
 
    if args.all:
        log.info("Generating all recommendations ...")
        generate_all_recommendations(top_n=args.top)
        return
 
    if args.risk:
        # non-interactive mode
        try:
            rec = recommend_funds(
                risk_appetite=args.risk,
                horizon=args.horizon,
                top_n=args.top
            )
            print_recommendation_table(rec, args.risk, args.horizon)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
    else:
        # fall back to guided interactive session
        interactive_mode()
 
 
if __name__ == "__main__":
    main()
 