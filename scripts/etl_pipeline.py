import os
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
 
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
 
# ── logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # goes up from scripts/ to project root
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
DB_DIR     = BASE_DIR / "data" / "db"
DB_PATH    = DB_DIR / "bluestock_mf.db"
 
for d in [PROC_DIR, DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)
 
 
# ==============================================================================
# CREATE DATABASE + APPLY SCHEMA
# ==============================================================================
 
def create_database():
    """
    Create the SQLite database file and apply schema.sql automatically.
    Safe to re-run — schema.sql uses DROP TABLE IF EXISTS so it won't crash
    if tables already exist.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
 
    schema_path = BASE_DIR / "sql" / "schema.sql"
 
    conn = sqlite3.connect(DB_PATH)
 
    if schema_path.exists():
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        log.info("✅ Schema applied from schema.sql")
    else:
        log.warning("schema.sql not found in sql/ folder — skipping schema creation")
 
    conn.close()
    log.info(f"✅ SQLite database ready → {DB_PATH}")
 
 
# ==============================================================================
# EXTRACT
# ==============================================================================
 
def load_raw_csv(filename: str, **kwargs) -> pd.DataFrame:
    """
    Simple wrapper around pd.read_csv so every load goes through one place.
    Makes it easier to swap to S3/GCS later if needed.
    """
    filepath = RAW_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Expected raw file not found: {filepath}")
    df = pd.read_csv(filepath, **kwargs)
    log.info(f"Loaded  {filename:<45}  shape={df.shape}")
    return df
 
 
def extract_all() -> dict:
    """Read every raw CSV into a dict of DataFrames keyed by logical name."""
    datasets = {}
 
    datasets["fund_master"]          = load_raw_csv("01_fund_master.csv")
    datasets["nav_history"]          = load_raw_csv("02_nav_history.csv")
    datasets["aum_fund_house"]       = load_raw_csv("03_aum_by_fund_house.csv")
    datasets["monthly_sip"]          = load_raw_csv("04_monthly_sip_inflows.csv")
    datasets["category_inflows"]     = load_raw_csv("05_category_inflows.csv")
    datasets["folio_count"]          = load_raw_csv("06_industry_folio_count.csv")
    datasets["scheme_perf"]          = load_raw_csv("07_scheme_performance.csv")
    datasets["investor_txns"]        = load_raw_csv("08_investor_transactions.csv")
    datasets["portfolio_holdings"]   = load_raw_csv("09_portfolio_holdings.csv")
    datasets["benchmark_indices"]    = load_raw_csv("10_benchmark_indices.csv")
 
    return datasets
 
 
# ==============================================================================
# TRANSFORM — individual cleaners
# ==============================================================================
 
def clean_fund_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    fund_master is our dimension table so it needs to be squeaky clean.
    Mostly just type fixes and stripping whitespace from text columns.
    """
    df = df.copy()
    df["amfi_code"] = df["amfi_code"].astype(str).str.strip()
 
    text_cols = ["fund_house", "scheme_name", "category", "sub_category",
                 "plan", "benchmark", "fund_manager", "risk_category"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
 
    # launch_date → proper datetime
    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
 
    # expense ratio sanity check — SEBI cap is 2.5% for regular plans
    bad_expense = df["expense_ratio_pct"].notna() & (
        (df["expense_ratio_pct"] < 0) | (df["expense_ratio_pct"] > 2.5)
    )
    if bad_expense.any():
        log.warning(f"fund_master: {bad_expense.sum()} rows with suspicious expense_ratio_pct")
 
    df.drop_duplicates(subset=["amfi_code"], inplace=True)
    log.info(f"fund_master cleaned → {len(df)} unique schemes")
    return df
 
 
def clean_nav_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    NAV history is the biggest table (~46K rows). Key steps:
      1. Parse dates
      2. Sort so forward-fill works correctly
      3. Forward-fill NAVs over weekends / market holidays
      4. Drop any lingering nulls and obviously wrong values (NAV <= 0)
      5. Compute daily_return_pct
    """
    df = df.copy()
    df["amfi_code"] = df["amfi_code"].astype(str).str.strip()
    df["date"]      = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values(["amfi_code", "date"], inplace=True)
 
    # build a complete business-day date range and reindex so we can ffill
    min_date = df["date"].min()
    max_date = df["date"].max()
    full_bdays = pd.bdate_range(start=min_date, end=max_date)
 
    filled_parts = []
    for code, grp in df.groupby("amfi_code"):
        grp = grp.set_index("date").reindex(full_bdays)
        grp["amfi_code"] = code
        grp["nav"] = grp["nav"].ffill()          # market holiday → use prev NAV
        filled_parts.append(grp.reset_index().rename(columns={"index": "date"}))
 
    df = pd.concat(filled_parts, ignore_index=True)
 
    # remove rows where NAV is still null or negative
    before = len(df)
    df = df[df["nav"] > 0].copy()
    df.dropna(subset=["nav"], inplace=True)
    removed = before - len(df)
    if removed:
        log.warning(f"nav_history: dropped {removed} rows with NAV <= 0 or null")
 
    df.drop_duplicates(subset=["amfi_code", "date"], inplace=True)
 
    # daily return — first day per fund will be NaN, that's fine
    df["daily_return_pct"] = (
        df.groupby("amfi_code")["nav"]
          .pct_change()
          .round(6)
    )
 
    log.info(f"nav_history cleaned → {len(df):,} rows, date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df
 
 
def clean_investor_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction data is synthetically generated but still needs a cleanup pass.
    Standardise the transaction_type values so SQL groupby doesn't split
    'SIP' and 'sip' into two buckets.
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df.dropna(subset=["transaction_date"], inplace=True)
 
    # normalise transaction type
    df["transaction_type"] = (
        df["transaction_type"]
          .str.strip()
          .str.title()
    )
    type_map = {"Sip": "SIP"}
    df["transaction_type"] = df["transaction_type"].replace(type_map)
 
    # amounts must be positive
    df = df[df["amount_inr"] > 0].copy()
 
    # kyc_status — only two valid values
    df["kyc_status"] = df["kyc_status"].str.strip().str.title()
 
    df.drop_duplicates(inplace=True)
    log.info(f"investor_transactions cleaned → {len(df):,} rows")
    return df
 
 
def clean_scheme_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the pre-computed metrics. We don't recompute them here
    (that's compute_metrics.py's job) but we do sanity-check ranges.
    """
    df = df.copy()
    df["amfi_code"] = df["amfi_code"].astype(str).str.strip()
 
    numeric_cols = [
        "return_1yr_pct", "return_3yr_pct", "return_5yr_pct",
        "alpha", "beta", "sharpe_ratio", "sortino_ratio",
        "std_dev_ann_pct", "max_drawdown_pct"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    # expense ratio can't be outside [0.1, 2.5]
    if "expense_ratio_pct" in df.columns:
        bad = (df["expense_ratio_pct"] < 0.1) | (df["expense_ratio_pct"] > 2.5)
        if bad.any():
            log.warning(f"scheme_performance: {bad.sum()} rows with out-of-range expense_ratio_pct")
 
    # negative Sharpe ratios are valid (fund underperformed risk-free rate)
    if "sharpe_ratio" in df.columns:
        neg_sharpe = (df["sharpe_ratio"] < 0).sum()
        if neg_sharpe:
            log.info(f"scheme_performance: {neg_sharpe} funds with negative Sharpe (underperformed RFR)")
 
    df.drop_duplicates(subset=["amfi_code"], inplace=True)
    log.info(f"scheme_performance cleaned → {len(df)} schemes")
    return df
 
 
def clean_generic(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Light-touch cleaning for tables that don't need heavy transformation —
    just strip whitespace from strings and drop full-row duplicates.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    before = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < before:
        log.info(f"{name}: removed {before - len(df)} duplicate rows")
    return df
 
 
def transform_all(raw: dict) -> dict:
    """Run every cleaner and return a dict of cleaned DataFrames."""
    cleaned = {}
    cleaned["fund_master"]        = clean_fund_master(raw["fund_master"])
    cleaned["nav_history"]        = clean_nav_history(raw["nav_history"])
    cleaned["investor_txns"]      = clean_investor_transactions(raw["investor_txns"])
    cleaned["scheme_perf"]        = clean_scheme_performance(raw["scheme_perf"])
 
    # lighter-touch tables
    for key in ["aum_fund_house", "monthly_sip", "category_inflows",
                "folio_count", "portfolio_holdings", "benchmark_indices"]:
        cleaned[key] = clean_generic(raw[key], key)
 
    return cleaned
 
 
# ==============================================================================
# DERIVE — computed columns added before loading
# ==============================================================================
 
def compute_cagr(nav_series: pd.Series, years: float) -> float:
    """
    Standard CAGR formula.
    Uses actual row count as proxy for trading days — more accurate than
    calendar-day division which inflates returns.
    """
    if len(nav_series) < 2:
        return np.nan
    start_nav = nav_series.iloc[0]
    end_nav   = nav_series.iloc[-1]
    if start_nav <= 0:
        return np.nan
    return float((end_nav / start_nav) ** (1 / years) - 1)
 
 
def add_derived_nav_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling 30-day average and a 'year_month' column —
    useful for monthly aggregations in the dashboard.
    """
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
 
    df["nav_30d_avg"] = (
        df.groupby("amfi_code")["nav"]
          .transform(lambda s: s.rolling(30, min_periods=1).mean())
          .round(4)
    )
    return df
 
 
# ==============================================================================
# LOAD
# ==============================================================================
 
def get_engine():
    db_url = f"sqlite:///{DB_PATH}"
    return create_engine(db_url, echo=False)
 
 
def load_to_sqlite(cleaned: dict, engine):
    """
    Write each cleaned DataFrame into the SQLite database.
    Table names mirror the logical names used throughout the project.
    if_exists='replace' so re-running the pipeline is safe.
    """
    table_map = {
        "fund_master":       "dim_fund",
        "nav_history":       "fact_nav",
        "scheme_perf":       "fact_performance",
        "investor_txns":     "fact_transactions",
        "portfolio_holdings":"fact_portfolio",
        "aum_fund_house":    "fact_aum",
        "monthly_sip":       "fact_sip_industry",
        "category_inflows":  "dim_category_inflows",
        "folio_count":       "dim_folio_count",
        "benchmark_indices": "fact_benchmark",
    }
 
    for key, table in table_map.items():
        df = cleaned.get(key)
        if df is None:
            log.warning(f"No data for table '{table}', skipping.")
            continue
        df.to_sql(table, con=engine, if_exists="replace", index=False)
        log.info(f"Loaded  {table:<30}  rows={len(df):,}")
 
 
def create_indexes(engine):
    """
    Add indexes on the columns we'll query most often.
    SQLite doesn't enforce FKs by default but indexes still speed up joins.
    """
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_nav_code_date  ON fact_nav (amfi_code, date);",
        "CREATE INDEX IF NOT EXISTS idx_txn_code       ON fact_transactions (amfi_code);",
        "CREATE INDEX IF NOT EXISTS idx_txn_date       ON fact_transactions (transaction_date);",
        "CREATE INDEX IF NOT EXISTS idx_bench_date     ON fact_benchmark (date);",
    ]
    with engine.connect() as conn:
        for stmt in indexes:
            conn.execute(text(stmt))
        conn.commit()
    log.info("Indexes created.")
 
 
def save_processed_csvs(cleaned: dict):
    """
    Also write cleaned CSVs to data/processed/ — needed for Power BI
    direct-import mode and for sharing with teammates who don't run Python.
    """
    for name, df in cleaned.items():
        out_path = PROC_DIR / f"clean_{name}.csv"
        df.to_csv(out_path, index=False)
    log.info(f"Processed CSVs written to {PROC_DIR}")
 
 
# ==============================================================================
# VALIDATE — quick post-load sanity check
# ==============================================================================
 
def validate_db(engine):
    """
    Run a handful of quick checks after loading to catch obvious problems
    before someone opens the dashboard and sees zeros everywhere.
    """
    checks = {
        "dim_fund row count":          "SELECT COUNT(*) FROM dim_fund",
        "fact_nav row count":          "SELECT COUNT(*) FROM fact_nav",
        "fact_transactions row count": "SELECT COUNT(*) FROM fact_transactions",
        "distinct amfi_codes in NAV":  "SELECT COUNT(DISTINCT amfi_code) FROM fact_nav",
        "null NAVs":                   "SELECT COUNT(*) FROM fact_nav WHERE nav IS NULL",
        "future-dated transactions":   f"SELECT COUNT(*) FROM fact_transactions WHERE transaction_date > '{datetime.today().date()}'",
    }
 
    log.info("── Post-load validation ──────────────────────────────────────────")
    with engine.connect() as conn:
        for label, query in checks.items():
            result = conn.execute(text(query)).scalar()
            log.info(f"  {label:<40}: {result:,}")
    log.info("─────────────────────────────────────────────────────────────────")
 
 
# ==============================================================================
# MAIN
# ==============================================================================
 
def run_pipeline():
    log.info("════════════════════════════════════════════════")
    log.info("  Bluestock MF — ETL Pipeline starting")
    log.info("════════════════════════════════════════════════")
 
    # 0. Create database + apply schema
    log.info("STEP 0/4  Creating database + applying schema.sql")
    create_database()
 
    # 1. Extract
    log.info("STEP 1/4  Extract raw CSVs")
    raw = extract_all()
 
    # 2. Transform
    log.info("STEP 2/4  Clean & transform")
    cleaned = transform_all(raw)
 
    # 3. Derive extra columns on NAV before loading
    log.info("STEP 3/4  Add derived columns to NAV")
    cleaned["nav_history"] = add_derived_nav_columns(cleaned["nav_history"])
 
    # 4. Load
    log.info("STEP 4/4  Load into SQLite + save CSVs")
    engine = get_engine()
    load_to_sqlite(cleaned, engine)
    create_indexes(engine)
    save_processed_csvs(cleaned)
 
    # 5. Quick sanity check
    validate_db(engine)
 
    log.info("Pipeline complete.  DB → %s", DB_PATH)
    log.info("Processed CSVs → %s", PROC_DIR)
 
 
if __name__ == "__main__":
    run_pipeline()
 