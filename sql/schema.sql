-- =============================================================================
-- schema.sql
-- Bluestock Fintech — Mutual Fund Analytics Platform
-- Star Schema Design for SQLite Database (bluestock_mf.db)
--
-- Tables:
--   DIMENSIONS : dim_fund, dim_date
--   FACTS      : fact_nav, fact_transactions, fact_performance,
--                fact_portfolio, fact_aum, fact_sip_industry,
--                fact_benchmark, dim_category_inflows, dim_folio_count
--
-- Run this before loading any data.
-- Author: Pramod | Bluestock Fintech Internship | June 2026
-- =============================================================================


-- drop tables if they already exist (clean slate on re-run)
DROP TABLE IF EXISTS dim_fund;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS fact_nav;
DROP TABLE IF EXISTS fact_transactions;
DROP TABLE IF EXISTS fact_performance;
DROP TABLE IF EXISTS fact_portfolio;
DROP TABLE IF EXISTS fact_aum;
DROP TABLE IF EXISTS fact_sip_industry;
DROP TABLE IF EXISTS fact_benchmark;
DROP TABLE IF EXISTS dim_category_inflows;
DROP TABLE IF EXISTS dim_folio_count;


-- =============================================================================
-- DIMENSION TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- dim_fund
-- Master list of 40 mutual fund schemes.
-- Every fact table joins back to this via amfi_code.
-- -----------------------------------------------------------------------------
CREATE TABLE dim_fund (
    amfi_code           TEXT        PRIMARY KEY,    -- AMFI unique scheme code e.g. 125497
    fund_house          TEXT        NOT NULL,        -- e.g. HDFC Mutual Fund
    scheme_name         TEXT        NOT NULL,        -- full official AMFI scheme name
    category            TEXT,                        -- Equity / Debt / Hybrid
    sub_category        TEXT,                        -- Large Cap / Mid Cap / Small Cap / Liquid etc.
    plan                TEXT,                        -- Direct or Regular
    launch_date         DATE,                        -- fund launch date
    benchmark           TEXT,                        -- official benchmark index
    expense_ratio_pct   REAL,                        -- annual TER in % (0.1 – 2.5)
    exit_load_pct       REAL        DEFAULT 0,       -- exit load % (0 for Liquid / Index funds)
    fund_manager        TEXT,                        -- primary fund manager name
    risk_category       TEXT,                        -- SEBI risk: Low / Moderate / High / Very High
    sebi_category_code  TEXT                         -- internal code: EC01=LargeCap, DC01=Liquid etc.
);


-- -----------------------------------------------------------------------------
-- dim_date
-- Pre-populated date dimension covering Jan 2022 – Dec 2026.
-- Useful for Power BI / Tableau time intelligence functions.
-- -----------------------------------------------------------------------------
CREATE TABLE dim_date (
    date_id     INTEGER     PRIMARY KEY,    -- surrogate key YYYYMMDD e.g. 20240101
    date        DATE        NOT NULL UNIQUE,
    year        INTEGER     NOT NULL,
    month       INTEGER     NOT NULL,       -- 1-12
    month_name  TEXT        NOT NULL,       -- January … December
    quarter     INTEGER     NOT NULL,       -- 1-4
    week_num    INTEGER,                    -- ISO week number
    is_weekday  INTEGER     NOT NULL,       -- 1 = weekday, 0 = weekend
    is_monthend INTEGER     DEFAULT 0       -- 1 if last business day of the month
);


-- =============================================================================
-- FACT TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- fact_nav
-- Daily NAV for all 40 schemes.  Largest table ~46K rows.
-- -----------------------------------------------------------------------------
CREATE TABLE fact_nav (
    amfi_code           TEXT    NOT NULL,
    date                DATE    NOT NULL,
    nav                 REAL    NOT NULL,           -- NAV in Rs. (e.g. 892.4560)
    daily_return_pct    REAL,                       -- (nav_t / nav_t-1) - 1
    nav_30d_avg         REAL,                       -- 30-day rolling average NAV
    year_month          TEXT,                       -- YYYY-MM for easy monthly grouping

    PRIMARY KEY (amfi_code, date),
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code)
);


-- -----------------------------------------------------------------------------
-- fact_transactions
-- Simulated SIP + Lumpsum + Redemption transactions for 5,000 investors.
-- -----------------------------------------------------------------------------
CREATE TABLE fact_transactions (
    tx_id                   TEXT    PRIMARY KEY,    -- unique transaction id
    investor_id             TEXT    NOT NULL,       -- INV000001 … INV005000
    amfi_code               TEXT    NOT NULL,
    transaction_date        DATE    NOT NULL,
    transaction_type        TEXT    NOT NULL,       -- SIP / Lumpsum / Redemption
    amount_inr              INTEGER NOT NULL,       -- transaction amount in Rs.
    state                   TEXT,                   -- investor state
    city                    TEXT,
    city_tier               TEXT,                   -- T30 or B30
    age_group               TEXT,                   -- 18-25 / 26-35 / 36-45 / 46-55 / 56+
    gender                  TEXT,
    annual_income_lakh      REAL,
    payment_mode            TEXT,                   -- UPI / Net Banking / Mandate / Cheque
    kyc_status              TEXT    DEFAULT 'Verified',

    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code)
);


-- -----------------------------------------------------------------------------
-- fact_performance
-- Pre-computed risk-return metrics per scheme (as of latest date).
-- -----------------------------------------------------------------------------
CREATE TABLE fact_performance (
    amfi_code           TEXT    NOT NULL,
    as_of_date          DATE    NOT NULL,
    return_1yr_pct      REAL,                   -- 1-year absolute return %
    return_3yr_pct      REAL,                   -- 3-year CAGR %
    return_5yr_pct      REAL,                   -- 5-year CAGR %
    benchmark_3yr_pct   REAL,                   -- benchmark 3yr CAGR for comparison
    alpha               REAL,                   -- return above benchmark
    beta                REAL,                   -- sensitivity to market (1.0 = in line)
    sharpe_ratio        REAL,                   -- risk-adjusted return (>1 is good)
    sortino_ratio       REAL,                   -- like Sharpe but only penalises downside
    std_dev_ann_pct     REAL,                   -- annualised standard deviation %
    max_drawdown_pct    REAL,                   -- worst peak-to-trough decline (negative)
    morningstar_rating  INTEGER,                -- 1-5 star (based on Sharpe)
    composite_score     REAL,                   -- bluestock scorecard 0-100

    PRIMARY KEY (amfi_code, as_of_date),
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code)
);


-- -----------------------------------------------------------------------------
-- fact_portfolio
-- Top equity holdings for each fund (as of Dec 2025).
-- ~320 rows across all equity schemes.
-- -----------------------------------------------------------------------------
CREATE TABLE fact_portfolio (
    amfi_code       TEXT    NOT NULL,
    stock_symbol    TEXT    NOT NULL,
    stock_name      TEXT,
    weight_pct      REAL,                   -- % of portfolio in this stock
    sector          TEXT,                   -- BFSI / IT / Auto / Pharma etc.
    as_of_date      DATE,

    PRIMARY KEY (amfi_code, stock_symbol, as_of_date),
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code)
);


-- -----------------------------------------------------------------------------
-- fact_aum
-- Quarterly AUM (Rs. crore) for 10 fund houses from 2022 to 2025.
-- -----------------------------------------------------------------------------
CREATE TABLE fact_aum (
    fund_house      TEXT    NOT NULL,
    date            DATE    NOT NULL,           -- quarter-end date
    aum_crore       REAL    NOT NULL,           -- AUM in Rs. crore
    num_schemes     INTEGER,                    -- number of schemes that quarter

    PRIMARY KEY (fund_house, date)
);


-- -----------------------------------------------------------------------------
-- fact_sip_industry
-- Monthly industry-level SIP inflow data (real AMFI Monthly Note figures).
-- 48 rows covering Jan 2022 – Dec 2025.
-- -----------------------------------------------------------------------------
CREATE TABLE fact_sip_industry (
    month                       TEXT    PRIMARY KEY,    -- YYYY-MM
    sip_inflow_crore            REAL,                   -- total SIP inflows Rs. crore
    active_sip_accounts_crore   REAL,                   -- actively contributing SIP accounts in crore
    new_sip_accounts_lakh       REAL,                   -- new SIP registrations that month (lakh)
    sip_aum_lakh_crore          REAL,                   -- total SIP AUM Rs. lakh crore
    yoy_growth_pct              REAL                    -- YoY growth % in SIP inflows
);


-- -----------------------------------------------------------------------------
-- fact_benchmark
-- Daily closing values for Nifty 50, Nifty 100, Nifty Midcap 150,
-- BSE SmallCap, CRISIL Liquid & Gilt indices.
-- -----------------------------------------------------------------------------
CREATE TABLE fact_benchmark (
    date            DATE    NOT NULL,
    index_name      TEXT    NOT NULL,           -- e.g. Nifty 50 / Nifty 100
    close_value     REAL    NOT NULL,           -- index closing value

    PRIMARY KEY (date, index_name)
);


-- -----------------------------------------------------------------------------
-- dim_category_inflows
-- Net inflows by fund category for FY 2024-25.
-- ~144 rows.
-- -----------------------------------------------------------------------------
CREATE TABLE dim_category_inflows (
    month           TEXT    NOT NULL,           -- YYYY-MM
    category        TEXT    NOT NULL,           -- Large Cap / Mid Cap / ELSS / Liquid etc.
    net_inflow_cr   REAL,                       -- net inflow Rs. crore (can be negative = outflow)

    PRIMARY KEY (month, category)
);


-- -----------------------------------------------------------------------------
-- dim_folio_count
-- Total MF folios broken by Equity / Debt / Hybrid — AMFI published milestones.
-- 21 rows.
-- -----------------------------------------------------------------------------
CREATE TABLE dim_folio_count (
    date                DATE    PRIMARY KEY,
    total_folios_crore  REAL,                   -- total folios in crore
    equity_folios_crore REAL,
    debt_folios_crore   REAL,
    hybrid_folios_crore REAL
);


-- =============================================================================
-- INDEXES
-- Added after data load for faster query performance on common join columns.
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_nav_code        ON fact_nav (amfi_code);
CREATE INDEX IF NOT EXISTS idx_nav_date        ON fact_nav (date);
CREATE INDEX IF NOT EXISTS idx_nav_code_date   ON fact_nav (amfi_code, date);
CREATE INDEX IF NOT EXISTS idx_txn_code        ON fact_transactions (amfi_code);
CREATE INDEX IF NOT EXISTS idx_txn_date        ON fact_transactions (transaction_date);
CREATE INDEX IF NOT EXISTS idx_txn_investor    ON fact_transactions (investor_id);
CREATE INDEX IF NOT EXISTS idx_txn_state       ON fact_transactions (state);
CREATE INDEX IF NOT EXISTS idx_perf_code       ON fact_performance (amfi_code);
CREATE INDEX IF NOT EXISTS idx_bench_date      ON fact_benchmark (date);
CREATE INDEX IF NOT EXISTS idx_bench_name      ON fact_benchmark (index_name);
CREATE INDEX IF NOT EXISTS idx_portfolio_code  ON fact_portfolio (amfi_code);
CREATE INDEX IF NOT EXISTS idx_aum_house       ON fact_aum (fund_house);

-- =============================================================================
-- END OF SCHEMA
-- =============================================================================
