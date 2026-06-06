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


def load_nav_data(amfi_code: str | int) -> pd.DataFrame:
    """
    Load daily NAV history for a specific fund AMFI code.
    Tries SQLite DB first, then falls back to processed CSV.
    """
    amfi_code_str = str(amfi_code).strip()

    # Try DB first
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            query = """
                SELECT amfi_code, date, nav, daily_return_pct 
                FROM fact_nav 
                WHERE CAST(amfi_code AS TEXT) = ? 
                ORDER BY date
            """
            df = pd.read_sql(query, conn, params=(amfi_code_str,))
            conn.close()
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                log.info(f"Loaded {len(df)} NAV records for AMFI {amfi_code_str} from DB.")
                return df
        except Exception as e:
            log.warning(f"Error loading from SQLite DB for AMFI {amfi_code_str}: {e}")

    # Fallback to CSV
    csv_path = PROC_DIR / "clean_nav_history.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df = df[df["amfi_code"].astype(str) == amfi_code_str]
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
                log.info(f"Loaded {len(df)} NAV records for AMFI {amfi_code_str} from CSV fallback.")
                return df
        except Exception as e:
            log.error(f"Error loading clean_nav_history.csv: {e}")

    log.warning(f"No NAV records found for AMFI code {amfi_code_str}.")
    return pd.DataFrame()


def get_default_amfi() -> str | None:
    """Return a default AMFI code from the database or CSV fallback."""
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            query = "SELECT DISTINCT amfi_code FROM fact_nav LIMIT 1"
            df = pd.read_sql(query, conn)
            conn.close()
            if not df.empty:
                return str(df.iloc[0, 0]).strip()
        except Exception as e:
            log.warning(f"Error reading default AMFI from DB: {e}")

    csv_path = PROC_DIR / "clean_nav_history.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=["amfi_code"], nrows=1000)
            codes = df["amfi_code"].dropna().astype(str).str.strip().unique()
            if len(codes) > 0:
                return str(codes[0])
        except Exception as e:
            log.warning(f"Error reading default AMFI from CSV: {e}")

    return None


def get_sample_amfi_codes(limit: int = 5) -> list[str]:
    """Return a sample list of available AMFI codes for user guidance."""
    codes: list[str] = []
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            query = f"SELECT DISTINCT amfi_code FROM fact_nav LIMIT {limit}"
            df = pd.read_sql(query, conn)
            conn.close()
            codes = [str(x).strip() for x in df["amfi_code"].dropna().tolist()]
            if codes:
                return codes
        except Exception as e:
            log.warning(f"Error reading sample AMFI codes from DB: {e}")

    csv_path = PROC_DIR / "clean_nav_history.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=["amfi_code"], nrows=1000)
            codes = [str(x).strip() for x in df["amfi_code"].dropna().astype(str).unique()[:limit].tolist()]
        except Exception as e:
            log.warning(f"Error reading sample AMFI codes from CSV: {e}")

    return codes


def run_monte_carlo_simulation(
    returns: pd.Series,
    sim_years: int = 5,
    num_sims: int = 250,
    initial_inv: float = 10000.0,
    seed: int = 42
) -> dict:
    """
    Simulate future returns and probability distributions based on daily return parameters.

    Parameters
    ----------
    returns : pd.Series
        Daily return percentages or raw decimal daily returns.
    sim_years : int
        Simulation horizon in years (default: 5).
    num_sims : int
        Number of simulated paths (default: 250).
    initial_inv : float
        Initial investment amount in INR (default: 10000.0).
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    dict
        Dictionary containing simulation paths, median/upper/lower projection arrays,
        annualized statistics, and final projection metrics.
    """
    if len(returns) < 30:
        raise ValueError("Insufficient return data to run simulation (minimum 30 points required).")

    # Clean returns
    clean_returns = returns.dropna()

    # If return percentages are in % (e.g. 0.5% as 0.5), convert to decimals if necessary.
    # We standardise to decimal return (e.g. daily return pct / 100).
    # Check if returns are in percentage scale (e.g., mean standard dev is > 0.05 on daily basis).
    # Standard fact_nav daily_return_pct is usually percentage (e.g., 1.5 for 1.5%).
    # We divide by 100 if the mean daily returns indicate it is in percent format.
    # Usually daily_return_pct is around -5 to 5.
    # Let's inspect: if max absolute value is > 0.5 (e.g. 1.0%), we treat it as percentage and divide by 100.
    if clean_returns.abs().max() > 0.5:
        daily_returns_decimal = clean_returns / 100.0
    else:
        daily_returns_decimal = clean_returns

    mu_daily = daily_returns_decimal.mean()
    sigma_daily = daily_returns_decimal.std()
    
    mu_ann = mu_daily * 252
    sigma_ann = sigma_daily * np.sqrt(252)

    n_days = int(sim_years * 252)
    np.random.seed(seed)

    # Generate daily returns using lognormal assumption: dS = S * (mu*dt + sigma*dW)
    # Log returns: dx = (mu - 0.5*sigma^2)*dt + sigma*dW
    sim_log_returns = np.random.normal(
        loc=(mu_daily - 0.5 * sigma_daily**2),
        scale=sigma_daily,
        size=(n_days, num_sims)
    )

    # Compute cumulative returns
    cum_returns = np.exp(np.cumsum(sim_log_returns, axis=0))
    portfolio_paths = initial_inv * cum_returns

    # Prepend starting investment value (day 0)
    day_zero = np.ones((1, num_sims)) * initial_inv
    portfolio_paths = np.vstack([day_zero, portfolio_paths])

    # Calculate percentiles over paths
    median_path = np.percentile(portfolio_paths, 50, axis=1)
    lower_bound = np.percentile(portfolio_paths, 5, axis=1)
    upper_bound = np.percentile(portfolio_paths, 95, axis=1)

    final_median = median_path[-1]
    final_lower = lower_bound[-1]
    final_upper = upper_bound[-1]
    
    final_values = portfolio_paths[-1, :]
    prob_profit = float(np.mean(final_values > initial_inv) * 100)
    median_cagr = float(((final_median / initial_inv) ** (1 / sim_years) - 1) * 100)

    return {
        "mu_ann": float(mu_ann),
        "sigma_ann": float(sigma_ann),
        "portfolio_paths": portfolio_paths,
        "time_axis": np.arange(len(median_path)) / 252,
        "median_path": median_path,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "final_median": float(final_median),
        "final_lower": float(final_lower),
        "final_upper": float(final_upper),
        "prob_profit": prob_profit,
        "median_cagr": median_cagr
    }


def main():
    parser = argparse.ArgumentParser(description="Bluestock MF — Monte Carlo NAV Growth Simulator CLI")
    parser.add_argument("amfi", nargs="?", help="AMFI code of the scheme to simulate")
    parser.add_argument("-a", "--amfi", dest="amfi", type=str, help="AMFI code of the scheme to simulate")
    parser.add_argument("--list-amfi", action="store_true", help="Show sample AMFI codes and exit")
    parser.add_argument("--years", type=int, default=5, help="Simulation horizon in years (default: 5)")
    parser.add_argument("--sims", type=int, default=250, help="Number of simulated paths (default: 250)")
    parser.add_argument("--initial", type=float, default=10000.0, help="Initial investment in INR (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    if args.list_amfi:
        codes = get_sample_amfi_codes(limit=20)
        if codes:
            print("Available AMFI codes:", ", ".join(codes))
        else:
            print("No AMFI codes available.")
        return

    if not args.amfi:
        default_amfi = get_default_amfi()
        if default_amfi:
            prompt = f"No AMFI provided. Press Enter to use default {default_amfi}, or type another AMFI: "
        else:
            prompt = "No AMFI provided. Enter an AMFI code: "

        try:
            user_amfi = input(prompt).strip()
        except EOFError:
            user_amfi = ""

        if user_amfi:
            args.amfi = user_amfi
        elif default_amfi:
            args.amfi = default_amfi

    if not args.amfi:
        log.error("No AMFI code provided and no AMFI could be selected interactively.")
        sample_codes = get_sample_amfi_codes()
        sample_text = ", ".join(sample_codes) if sample_codes else "<no sample AMFI codes available>"
        log.error("Please provide `--amfi` or positional AMFI. Sample AMFI codes: %s", sample_text)
        return

    if not args.amfi:
        args.amfi = get_default_amfi()
        if args.amfi:
            log.info(f"No AMFI provided, using default AMFI {args.amfi}.")
        else:
            log.error("No AMFI code provided and no default AMFI could be detected.")
            return

    log.info(f"Loading data for AMFI code {args.amfi}...")
    df = load_nav_data(args.amfi)

    if df.empty:
        sample_codes = get_sample_amfi_codes()
        sample_text = ", ".join(sample_codes) if sample_codes else "<no sample AMFI codes available>"
        log.error(
            "Could not run simulation: No data found for AMFI %s. "
            "Try one of these AMFI codes: %s",
            args.amfi,
            sample_text,
        )
        return

    if "daily_return_pct" in df.columns and df["daily_return_pct"].notna().sum() > 30:
        returns = df["daily_return_pct"].dropna()
    else:
        returns = df["nav"].pct_change().dropna()

    try:
        results = run_monte_carlo_simulation(
            returns=returns,
            sim_years=args.years,
            num_sims=args.sims,
            initial_inv=args.initial,
            seed=args.seed
        )

        hr = "=" * 70
        print(f"\n{hr}")
        print(f"  BLUESTOCK FINTECH - MONTE CARLO GROWTH SIMULATION")
        print(f"  AMFI Code            : {args.amfi}")
        print(f"  Simulation Horizon   : {args.years} Years")
        print(f"  Simulated Paths      : {args.sims}")
        print(f"  Initial Investment   : INR {args.initial:,.2f}")
        print(hr)
        print(f"  Annualized Drift (mu)  : {results['mu_ann']*100:.2f}%")
        print(f"  Annualized Vol (sigma) : {results['sigma_ann']*100:.2f}%")
        print(f"  Median Projected Value : INR {results['final_median']:,.2f}")
        print(f"  Pessimistic Value (5%) : INR {results['final_lower']:,.2f}")
        print(f"  Optimistic Value (95%) : INR {results['final_upper']:,.2f}")
        print(f"  Projected Median CAGR  : {results['median_cagr']:.2f}%")
        print(f"  Probability of Profit  : {results['prob_profit']:.1f}%")
        print(f"{hr}\n")

    except Exception as e:
        log.error(f"Failed to run simulation: {e}")


if __name__ == "__main__":
    main()
