-- =============================================================================
-- queries.sql
-- Bluestock Fintech — Mutual Fund Analytics Platform
-- 10 Core Analytical Queries + Bonus Queries
--
-- All queries run against bluestock_mf.db (SQLite).
-- Open DB Browser for SQLite or run via Python / sqlalchemy to test.
--
-- Author: Pramod | Bluestock Fintech Internship | June 2026
-- =============================================================================


-- =============================================================================
-- QUERY 01 — Top 5 Fund Houses by Latest AUM
-- Shows which AMCs manage the most money.
-- Real answer: SBI MF leads at Rs.12.5 lakh crore as of Dec 2025.
-- =============================================================================

SELECT
    fund_house,
    ROUND(MAX(aum_crore), 2)            AS latest_aum_crore,
    ROUND(MAX(aum_crore) / 100000, 4)   AS latest_aum_lakh_crore
FROM fact_aum
WHERE date = (SELECT MAX(date) FROM fact_aum)
GROUP BY fund_house
ORDER BY latest_aum_crore DESC
LIMIT 5;


-- =============================================================================
-- QUERY 02 — Average NAV Per Month for a Specific Fund
-- Change amfi_code to any scheme you want to analyse.
-- HDFC Top 100 Direct = 125497
-- =============================================================================

SELECT
    year_month,
    ROUND(AVG(nav), 4)      AS avg_nav,
    ROUND(MIN(nav), 4)      AS min_nav,
    ROUND(MAX(nav), 4)      AS max_nav,
    COUNT(*)                AS trading_days
FROM fact_nav
WHERE amfi_code = '125497'
GROUP BY year_month
ORDER BY year_month;


-- =============================================================================
-- QUERY 03 — SIP Inflow Year-on-Year Growth
-- Compares total SIP inflows for each calendar year.
-- Highlights the industry growth from 2022 to 2025.
-- =============================================================================

SELECT
    SUBSTR(month, 1, 4)             AS year,
    ROUND(SUM(sip_inflow_crore), 2) AS total_sip_inflow_crore,
    COUNT(*)                        AS months_recorded
FROM fact_sip_industry
GROUP BY SUBSTR(month, 1, 4)
ORDER BY year;


-- =============================================================================
-- QUERY 04 — Total Transaction Amount by State
-- Shows which states contribute the most to MF investments.
-- T30 cities (metros) vs B30 cities (smaller towns) breakdown.
-- =============================================================================

SELECT
    state,
    city_tier,
    COUNT(*)                            AS total_transactions,
    ROUND(SUM(amount_inr) / 1e7, 2)    AS total_amount_crore,
    ROUND(AVG(amount_inr), 0)           AS avg_transaction_inr
FROM fact_transactions
WHERE transaction_type != 'Redemption'
GROUP BY state, city_tier
ORDER BY total_amount_crore DESC;


-- =============================================================================
-- QUERY 05 — Funds with Expense Ratio Below 1%
-- Direct plans with low TER — better for long-term investors.
-- =============================================================================

SELECT
    f.amfi_code,
    f.scheme_name,
    f.fund_house,
    f.category,
    f.sub_category,
    f.expense_ratio_pct,
    f.plan
FROM dim_fund f
WHERE f.expense_ratio_pct < 1.0
  AND f.plan = 'Direct'
ORDER BY f.expense_ratio_pct ASC;


-- =============================================================================
-- QUERY 06 — Best Performing Funds by 3-Year CAGR per Category
-- Uses RANK() window function to find the top fund in each category.
-- =============================================================================

SELECT
    category,
    scheme_name,
    fund_house,
    return_3yr_pct,
    sharpe_ratio,
    rank_in_category
FROM (
    SELECT
        f.category,
        f.scheme_name,
        f.fund_house,
        p.return_3yr_pct,
        p.sharpe_ratio,
        RANK() OVER (
            PARTITION BY f.category
            ORDER BY p.return_3yr_pct DESC
        ) AS rank_in_category
    FROM fact_performance p
    JOIN dim_fund f ON f.amfi_code = p.amfi_code
    WHERE p.return_3yr_pct IS NOT NULL
)
WHERE rank_in_category = 1
ORDER BY return_3yr_pct DESC;


-- =============================================================================
-- QUERY 07 — Monthly Transaction Volume Trend
-- How many SIP / Lumpsum / Redemption transactions happen each month.
-- Useful for spotting seasonal patterns in investor behaviour.
-- =============================================================================

SELECT
    SUBSTR(transaction_date, 1, 7)  AS year_month,
    transaction_type,
    COUNT(*)                        AS num_transactions,
    ROUND(SUM(amount_inr) / 1e6, 2) AS total_amount_lakhs
FROM fact_transactions
GROUP BY year_month, transaction_type
ORDER BY year_month, transaction_type;


-- =============================================================================
-- QUERY 08 — Investor Demographics: Age Group vs Average SIP Amount
-- Shows which age group invests the most per SIP transaction.
-- =============================================================================

SELECT
    age_group,
    gender,
    COUNT(DISTINCT investor_id)         AS unique_investors,
    COUNT(*)                            AS total_sip_transactions,
    ROUND(AVG(amount_inr), 0)           AS avg_sip_amount_inr,
    ROUND(SUM(amount_inr) / 1e7, 2)    AS total_invested_crore
FROM fact_transactions
WHERE transaction_type = 'SIP'
GROUP BY age_group, gender
ORDER BY age_group, gender;


-- =============================================================================
-- QUERY 09 — Funds Beating Their Benchmark (Positive Alpha)
-- Alpha > 0 means the fund delivered returns above what the market gave.
-- The higher the alpha, the better the fund manager's skill.
-- =============================================================================

SELECT
    f.amfi_code,
    f.scheme_name,
    f.fund_house,
    f.sub_category,
    f.benchmark,
    p.alpha,
    p.beta,
    p.return_3yr_pct,
    p.benchmark_3yr_pct,
    ROUND(p.return_3yr_pct - p.benchmark_3yr_pct, 2) AS excess_return_pct
FROM fact_performance p
JOIN dim_fund f ON f.amfi_code = p.amfi_code
WHERE p.alpha > 0
ORDER BY p.alpha DESC;


-- =============================================================================
-- QUERY 10 — Industry AUM + SIP Inflow Combined View (Monthly)
-- Joins AUM growth with SIP inflow data to see if they move together.
-- Good for the Industry Trends dashboard page.
-- =============================================================================

SELECT
    s.month,
    s.sip_inflow_crore,
    s.active_sip_accounts_crore,
    s.sip_aum_lakh_crore,
    s.yoy_growth_pct
FROM fact_sip_industry s
ORDER BY s.month;


-- =============================================================================
-- BONUS QUERY 01 — Fund Scorecard Full Ranking
-- Ranks all 40 funds by composite score (Sharpe + CAGR + Alpha + expense + MDD).
-- =============================================================================

SELECT
    ROW_NUMBER() OVER (ORDER BY p.sharpe_ratio DESC)    AS rank_overall,
    f.scheme_name,
    f.fund_house,
    f.sub_category,
    f.expense_ratio_pct,
    p.return_3yr_pct,
    p.sharpe_ratio,
    p.alpha,
    p.max_drawdown_pct,
    p.composite_score
FROM fact_performance p
JOIN dim_fund f ON f.amfi_code = p.amfi_code
WHERE p.sharpe_ratio IS NOT NULL
ORDER BY p.composite_score DESC NULLS LAST;


-- =============================================================================
-- BONUS QUERY 02 — Top 10 Stocks Held Across All Equity Funds
-- Which stocks appear most frequently in MF portfolios?
-- =============================================================================

SELECT
    stock_name,
    stock_symbol,
    sector,
    COUNT(DISTINCT amfi_code)       AS num_funds_holding,
    ROUND(AVG(weight_pct), 2)       AS avg_weight_pct,
    ROUND(SUM(weight_pct), 2)       AS total_weight_pct
FROM fact_portfolio
GROUP BY stock_symbol, stock_name, sector
ORDER BY num_funds_holding DESC, avg_weight_pct DESC
LIMIT 10;


-- =============================================================================
-- BONUS QUERY 03 — Sector Concentration (HHI) Per Fund
-- Herfindahl-Hirschman Index: sum of squared weights.
-- HHI close to 10000 = very concentrated, close to 0 = well diversified.
-- =============================================================================

SELECT
    fp.amfi_code,
    f.scheme_name,
    f.sub_category,
    COUNT(DISTINCT fp.sector)                           AS num_sectors,
    ROUND(SUM(fp.weight_pct * fp.weight_pct), 2)       AS hhi_score,
    CASE
        WHEN SUM(fp.weight_pct * fp.weight_pct) > 2500 THEN 'Concentrated'
        WHEN SUM(fp.weight_pct * fp.weight_pct) > 1500 THEN 'Moderate'
        ELSE 'Diversified'
    END AS concentration_label
FROM fact_portfolio fp
JOIN dim_fund f ON f.amfi_code = fp.amfi_code
GROUP BY fp.amfi_code, f.scheme_name, f.sub_category
ORDER BY hhi_score DESC;


-- =============================================================================
-- BONUS QUERY 04 — SIP Continuity: At-Risk Investors
-- Investors who have gaps > 35 days between SIP transactions
-- are flagged as 'at-risk' of stopping their SIP.
-- =============================================================================

WITH sip_gaps AS (
    SELECT
        investor_id,
        amfi_code,
        transaction_date,
        LAG(transaction_date) OVER (
            PARTITION BY investor_id, amfi_code
            ORDER BY transaction_date
        ) AS prev_sip_date,
        JULIANDAY(transaction_date) - JULIANDAY(
            LAG(transaction_date) OVER (
                PARTITION BY investor_id, amfi_code
                ORDER BY transaction_date
            )
        ) AS gap_days
    FROM fact_transactions
    WHERE transaction_type = 'SIP'
)
SELECT
    investor_id,
    amfi_code,
    COUNT(*)                    AS total_sips,
    ROUND(AVG(gap_days), 1)     AS avg_gap_days,
    MAX(gap_days)               AS max_gap_days,
    CASE
        WHEN MAX(gap_days) > 35 THEN 'At Risk'
        ELSE 'Regular'
    END AS sip_continuity_status
FROM sip_gaps
WHERE gap_days IS NOT NULL
GROUP BY investor_id, amfi_code
HAVING total_sips >= 6
ORDER BY max_gap_days DESC
LIMIT 20;


-- =============================================================================
-- BONUS QUERY 05 — NAV Performance vs Benchmark (Rolling Comparison)
-- Compares a fund's NAV growth against Nifty 100 for the same date range.
-- Change amfi_code and index_name as needed.
-- =============================================================================

SELECT
    n.date,
    n.nav,
    n.daily_return_pct          AS fund_daily_return,
    b.close_value               AS nifty100_close,
    ROUND(
        (b.close_value - FIRST_VALUE(b.close_value) OVER (ORDER BY b.date))
        / FIRST_VALUE(b.close_value) OVER (ORDER BY b.date) * 100
    , 2)                        AS nifty100_return_pct_from_start,
    ROUND(
        (n.nav - FIRST_VALUE(n.nav) OVER (ORDER BY n.date))
        / FIRST_VALUE(n.nav) OVER (ORDER BY n.date) * 100
    , 2)                        AS fund_return_pct_from_start
FROM fact_nav n
JOIN fact_benchmark b ON b.date = n.date
WHERE n.amfi_code = '125497'          -- HDFC Top 100
  AND b.index_name LIKE '%100%'       -- Nifty 100
  AND n.date >= '2023-01-01'
ORDER BY n.date;


-- =============================================================================
-- END OF QUERIES
-- =============================================================================
