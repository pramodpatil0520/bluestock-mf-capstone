import argparse
import logging
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

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


def load_portfolio_nav_data(amfi_codes: list) -> pd.DataFrame:
    """
    Fetch historical NAV returns for a list of AMFI codes.
    Tries SQLite DB first, then falls back to CSV for missing codes.
    """
    clean_codes = [str(c).strip() for c in amfi_codes if c]
    if not clean_codes:
        return pd.DataFrame()

    df = pd.DataFrame()
    missing_codes = list(clean_codes)
    csv_path = PROC_DIR / "clean_nav_history.csv"

    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            codes_str = ",".join([f"'{c}'" for c in clean_codes])
            query = f"""
                SELECT amfi_code, date, nav, daily_return_pct
                FROM fact_nav
                WHERE CAST(amfi_code AS TEXT) IN ({codes_str})
                ORDER BY amfi_code, date
            """
            df = pd.read_sql(query, conn)
            conn.close()
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values(["amfi_code", "date"])
                found_codes = df["amfi_code"].astype(str).unique().tolist()
                missing_codes = [c for c in clean_codes if c not in found_codes]
                log.info(
                    f"Loaded {len(df)} total NAV records from DB for {len(found_codes)} of {len(clean_codes)} requested schemes."
                )
                if not missing_codes:
                    return df
                log.warning(f"Missing NAV data in DB for AMFI codes: {missing_codes}")
        except Exception as e:
            log.warning(f"Error loading portfolio from SQLite DB: {e}")

    if csv_path.exists():
        try:
            csv_df = pd.read_csv(csv_path)
            csv_df["amfi_code"] = csv_df["amfi_code"].astype(str)
            if missing_codes:
                csv_df = csv_df[csv_df["amfi_code"].isin(missing_codes)]
            else:
                csv_df = csv_df[csv_df["amfi_code"].isin(clean_codes)]

            if not csv_df.empty:
                csv_df["date"] = pd.to_datetime(csv_df["date"])
                csv_df = csv_df.sort_values(["amfi_code", "date"])
                if df.empty:
                    df = csv_df
                    csv_schemes = len(csv_df["amfi_code"].unique())
                    log.info(
                        f"Loaded {len(df)} total NAV records from CSV for {csv_schemes} schemes."
                    )
                else:
                    df = pd.concat([df, csv_df], ignore_index=True)
                    df = df.sort_values(["amfi_code", "date"])
                    found_codes = df["amfi_code"].astype(str).unique().tolist()
                    log.info(
                        f"Appended {len(csv_df)} CSV records and now have {len(df)} total NAV records for {len(found_codes)} schemes."
                    )

                final_found = df["amfi_code"].astype(str).unique().tolist()
                if len(final_found) != len(clean_codes):
                    still_missing = [c for c in clean_codes if c not in final_found]
                    log.warning(f"Could not find NAV data for AMFI codes: {still_missing}")
                return df
        except Exception as e:
            log.error(f"Error loading clean_nav_history.csv: {e}")

    return df


def load_fund_names_map() -> dict:
    """
    Get a mapping of AMFI code (as str) to Scheme Name from DB or scorecard CSV.
    """
    # Try DB first
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            df = pd.read_sql("SELECT amfi_code, scheme_name FROM dim_fund", conn)
            conn.close()
            return dict(zip(df["amfi_code"].astype(str), df["scheme_name"]))
        except Exception:
            pass

    # Fallback to CSV
    csv_path = PROC_DIR / "fund_scorecard.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "amfi_code" in df.columns and "scheme_name" in df.columns:
                return dict(zip(df["amfi_code"].astype(str), df["scheme_name"]))
        except Exception:
            pass

    return {}


def optimize_portfolio(
    returns_df: pd.DataFrame,
    num_portfolios: int = 1500,
    rf_rate: float = 0.065,
    seed: int = 42
) -> dict:
    """
    Run Markowitz MPT calculations on a returns DataFrame to find the efficient frontier,
    the Maximum Sharpe Ratio portfolio, and the Minimum Volatility portfolio weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame where columns are fund identifiers and index represents dates, containing daily returns.
    num_portfolios : int
        Number of random weight allocations to simulate (default: 1500).
    rf_rate : float
        Annualized risk-free rate as a decimal (e.g. 0.065 for 6.5%) (default: 0.065).
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    dict
        Dictionary containing mean returns, covariance matrix, simulated results array,
        weights history, and specific details of Max Sharpe & Min Volatility portfolios.
    """
    # Ensure returns are decimals (if returns are stored as percents like 1.5, we scale them to decimal)
    # Check max absolute value across the dataframe.
    if returns_df.abs().max().max() > 0.5:
        daily_returns_decimal = returns_df / 100.0
    else:
        daily_returns_decimal = returns_df

    mean_returns = daily_returns_decimal.mean() * 252
    cov_matrix = daily_returns_decimal.cov() * 252

    num_assets = len(returns_df.columns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    np.random.seed(seed)

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        # Annualized portfolio return and volatility
        p_return = np.sum(weights * mean_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = p_return
        results[1, i] = p_vol
        results[2, i] = (p_return - rf_rate) / p_vol

    # Max Sharpe Portfolio
    max_sharpe_idx = np.argmax(results[2])
    rp_max = results[0, max_sharpe_idx]
    sdp_max = results[1, max_sharpe_idx]
    max_sharpe_weights = weights_record[max_sharpe_idx]

    # Min Volatility Portfolio
    min_vol_idx = np.argmin(results[1])
    rp_min = results[0, min_vol_idx]
    sdp_min = results[1, min_vol_idx]
    min_vol_weights = weights_record[min_vol_idx]

    return {
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix,
        "results": results,
        "weights_record": weights_record,
        "max_sharpe": {
            "return": float(rp_max),
            "volatility": float(sdp_max),
            "sharpe": float(results[2, max_sharpe_idx]),
            "weights": max_sharpe_weights
        },
        "min_volatility": {
            "return": float(rp_min),
            "volatility": float(sdp_min),
            "sharpe": float(results[2, min_vol_idx]),
            "weights": min_vol_weights
        }
    }


def main():
    example_text = (
        "Example:\n"
        "  python scripts\\markowitz.py --amfi 100016,100025 --portfolios 100 --rf 6.5\n"
        "  python scripts\\markowitz.py --amfi 100016,100025\n"
    )

    parser = argparse.ArgumentParser(
        description="Bluestock MF — Markowitz Efficient Frontier Portfolio Optimiser CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=example_text,
    )
    parser.add_argument("--amfi", type=str, default="", help="Comma-separated AMFI codes of funds (e.g. 119598,119605)")
    parser.add_argument("--rf", type=float, default=6.5, help="Annualized risk-free rate in percentage (default: 6.5)")
    parser.add_argument("--portfolios", type=int, default=1500, help="Number of random portfolios to simulate (default: 1500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    if not args.amfi:
        if sys.stdin.isatty():
            try:
                args.amfi = input("Enter comma-separated AMFI codes (e.g. 100016,100025): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
        if not args.amfi:
            parser.print_help()
            return

    amfi_codes = [c.strip() for c in args.amfi.split(",") if c.strip()]
    if len(amfi_codes) < 2:
        log.error("Portfolio optimization requires at least 2 mutual funds.")
        return

    log.info(f"Loading data for AMFI codes: {amfi_codes}...")
    nav_df = load_portfolio_nav_data(amfi_codes)

    if nav_df.empty:
        log.error("Could not fetch historical NAVs for optimization.")
        return

    code_to_name = load_fund_names_map()
    nav_df["fund_name"] = nav_df["amfi_code"].astype(str).map(lambda c: code_to_name.get(c, f"AMFI {c}"))

    # Pivot to get returns aligned by date
    if "daily_return_pct" not in nav_df.columns:
        nav_df["daily_return_pct"] = nav_df.groupby("fund_name")["nav"].pct_change() * 100

    returns_df = nav_df.pivot(index="date", columns="fund_name", values="daily_return_pct")
    returns_df = returns_df.dropna()

    if returns_df.shape[1] < 2:
        log.error(
            "Portfolio optimization requires at least 2 funds with overlapping daily return histories."
        )
        return

    if len(returns_df) < 30:
        log.error(f"Too few overlapping trading days ({len(returns_df)}) between selected funds. Need at least 30 days.")
        return

    rf_rate_decimal = args.rf / 100.0

    try:
        opt_results = optimize_portfolio(
            returns_df=returns_df,
            num_portfolios=args.portfolios,
            rf_rate=rf_rate_decimal,
            seed=args.seed
        )

        hr = "=" * 70
        print(f"\n{hr}")
        print(f"  BLUESTOCK FINTECH - PORTFOLIO OPTIMIZATION REPORT")
        print(f"  Risk-Free Rate: {args.rf:.2f}% | Simulated Portfolios: {args.portfolios}")
        print(hr)
        
        # Max Sharpe Portfolio details
        ms = opt_results["max_sharpe"]
        print(f"\n  🚀 MAXIMUM SHARPE RATIO PORTFOLIO:")
        print(f"  Expected Return (Ann) : {ms['return']*100:.2f}%")
        print(f"  Annualized Volatility : {ms['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio          : {ms['sharpe']:.2f}")
        print("  Asset Allocation Weights:")
        for name, weight in zip(returns_df.columns, ms['weights']):
            print(f"    - {name:<45} : {weight*100:6.2f}%")
            
        # Min Volatility Portfolio details
        mv = opt_results["min_volatility"]
        print(f"\n  🛡️ MINIMUM VOLATILITY PORTFOLIO:")
        print(f"  Expected Return (Ann) : {mv['return']*100:.2f}%")
        print(f"  Annualized Volatility : {mv['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio          : {mv['sharpe']:.2f}")
        print("  Asset Allocation Weights:")
        for name, weight in zip(returns_df.columns, mv['weights']):
            print(f"    - {name:<45} : {weight*100:6.2f}%")
            
        print(f"{hr}\n")

    except Exception as e:
        log.error(f"Failed to run Markowitz optimization: {e}")


if __name__ == "__main__":
    main()
