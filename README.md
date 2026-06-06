# 📊 Bluestock Fintech — Mutual Fund Analytics Platform

> **Capstone Internship Project** | Bluestock Fintech Pvt. Ltd. | June 2026  
> **Author:** Pramod Vinayak Patil · Final Year ECE · R.C. Patel Institute of Technology, Shirpur

---

## 🏆 Project Summary

A full-stack **Mutual Fund Analytics Platform** that ingests publicly available AMFI India data, transforms it through a robust Python ETL pipeline, stores it in a normalised SQLite database, and surfaces insights via an interactive Power BI dashboard.

| Metric | Value |
|--------|-------|
| Fund Schemes Covered | 40 |
| CSV Datasets | 10 |
| Total Rows | 87,000+ |
| NAV History | Jan 2022 – May 2026 (4.5 yrs) |
| Database Tables | 8 (star schema) |
| Dashboard Pages | 4 (Power BI) |
| Estimated Effort | 50–55 hours (7 working days) |

---

## 📁 Folder Structure

```
bluestock_mf_capstone/
├── data/
│   ├── raw/                    ← Original downloaded files (15 CSVs + live API)
│   ├── processed/              ← Cleaned, merged CSVs (10 files)
│   └── db/
│       └── bluestock_mf.db     ← SQLite database (add to .gitignore if >100MB)
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_analysis.ipynb
│   ├── 04_performance_analytics.ipynb
│   └── 05_advanced_analytics.ipynb
├── scripts/
│   ├── etl_pipeline.py         ← Master ETL script (run this first)
│   ├── live_nav_fetch.py       ← Fetches live NAV from mfapi.in
│   ├── compute_metrics.py      ← Computes Sharpe, Alpha, Beta, etc.
│   └── recommender.py          ← Rule-based fund recommender
├── sql/
│   ├── schema.sql              ← CREATE TABLE statements (star schema)
│   └── queries.sql             ← 10 analytical SQL queries
├── dashboard/
│   └── bluestock_mf.pbix       ← Power BI Desktop dashboard
├── reports/
│   ├── Final_Report.pdf        ← 15–20 page detailed report
│   └── Presentation.pptx       ← 12-slide capstone deck
├── requirements.txt
├── .gitignore
└── README.md                   ← You are here
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/pramodpatil0520/bluestock-mf-capstone.git
cd bluestock-mf-capstone
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
plotly>=5.0
sqlalchemy>=2.0
requests>=2.30
scipy>=1.10
jupyter
ipykernel
```

### 3. Run the ETL Pipeline

```bash
python scripts/etl_pipeline.py
```

This will:
- Load all 10 CSV datasets from `data/raw/`
- Clean and validate each dataset
- Compute derived fields (daily returns, CAGR)
- Load everything into `data/db/bluestock_mf.db`

### 4. Fetch Live NAV Data (Optional)

```bash
python scripts/live_nav_fetch.py
```

Fetches current NAV for 5 key schemes from mfapi.in (no API key required):

| Scheme | AMFI Code |
|--------|-----------|
| HDFC Top 100 Fund – Direct | 125497 |
| SBI Bluechip Fund – Direct | 119551 |
| ICICI Pru Bluechip Fund – Direct | 120503 |
| Nippon India Large Cap Fund – Direct | 118632 |
| Axis Bluechip Fund – Direct | 119092 |

### 5. Launch Jupyter Notebooks

```bash
jupyter lab
```

Run notebooks in order: `01` → `02` → `03` → `04` → `05`

### 6. Open Power BI Dashboard

Open `dashboard/bluestock_mf.pbix` in Power BI Desktop and:
- Update the SQLite connection path if needed (Data Source Settings)
- Click **Refresh** to reload data from the database
- All 4 pages will populate with current data

---

## 🗄️ Database Schema

The SQLite database uses a **star schema** with `dim_fund` and `dim_date` as dimension tables:

```sql
-- Dimension Tables
CREATE TABLE dim_fund (
    amfi_code TEXT PRIMARY KEY,
    fund_house TEXT,
    scheme_name TEXT,
    category TEXT,
    sub_category TEXT,
    plan TEXT,
    benchmark TEXT,
    expense_ratio_pct REAL,
    risk_category TEXT
);

CREATE TABLE dim_date (
    date_id INTEGER PRIMARY KEY,
    date DATE,
    year INTEGER,
    month INTEGER,
    quarter INTEGER,
    is_weekday INTEGER
);

-- Fact Tables
CREATE TABLE fact_nav (
    amfi_code TEXT REFERENCES dim_fund(amfi_code),
    nav_date DATE,
    nav REAL,
    daily_return_pct REAL
);

CREATE TABLE fact_transactions (
    tx_id TEXT PRIMARY KEY,
    investor_id TEXT,
    amfi_code TEXT REFERENCES dim_fund(amfi_code),
    transaction_date DATE,
    amount_inr INTEGER,
    transaction_type TEXT,   -- SIP / Lumpsum / Redemption
    state TEXT,
    city_tier TEXT,          -- T30 / B30
    age_group TEXT,
    gender TEXT
);

CREATE TABLE fact_performance (
    amfi_code TEXT REFERENCES dim_fund(amfi_code),
    as_of_date DATE,
    return_1yr_pct REAL,
    return_3yr_pct REAL,
    return_5yr_pct REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    alpha REAL,
    beta REAL,
    max_drawdown_pct REAL
);
```

---

## 📐 Performance Metrics — Formulas

| Metric | Formula | Notes |
|--------|---------|-------|
| Daily Return | `nav_t / nav_{t-1} - 1` | Forward-fill weekends/holidays |
| CAGR | `(NAV_end / NAV_start)^(1/n) - 1` | n = trading days / 252 |
| Sharpe Ratio | `(Rp - Rf) / σp × √252` | Rf = 6.5% (RBI repo proxy) |
| Sortino Ratio | `(Rp - Rf) / σ_downside × √252` | σ uses only negative return days |
| Alpha | `OLS intercept × 252` | Regress vs Nifty 100 |
| Beta | `OLS slope` | `scipy.stats.linregress` |
| Max Drawdown | `min(NAV_t / max(NAV_0..t) - 1)` | Most negative value |
| VaR (95%) | `numpy.percentile(returns, 5)` | Historical simulation |
| CVaR | `mean(returns[returns < VaR])` | Expected tail loss |

**Fund Scorecard (0–100):**
```
Score = 0.30 × (3yr_return_rank)
      + 0.25 × (sharpe_rank)
      + 0.20 × (alpha_rank)
      + 0.15 × (expense_ratio_rank_inverse)
      + 0.10 × (max_drawdown_rank_inverse)
```

---

## 📊 Power BI Dashboard — 4 Pages

| Page | Title | Key Visuals |
|------|-------|-------------|
| 1 | Industry Overview | KPI cards (AUM ₹81L Cr, SIP ₹31K Cr, Folios 26.12 Cr), AUM line chart, fund house bar chart |
| 2 | Fund Performance | Return vs Risk scatter, sortable scorecard table, NAV vs benchmark line |
| 3 | Investor Analytics | Transaction map by state, SIP/Lumpsum/Redemption donut, age vs SIP bar |
| 4 | SIP & Market Trends | Dual-axis SIP inflow + Nifty 50, category inflow heatmap, top 5 categories FY25 |

All pages include at least 2 interactive slicers: **Fund House**, **Category**, **Date Range**, **State**.

---

## 📡 Data Sources

| Source | URL | Data |
|--------|-----|------|
| AMFI India | www.amfiindia.com | NAV, AUM, SIP, Folio data |
| mfapi.in | api.mfapi.in/mf/{code} | Historical NAV JSON (no auth) |
| NSE India | nseindia.com/reports | Nifty 50 / Nifty 100 daily prices |
| BSE India | bseindia.com | BSE SmallCap index daily prices |
| AMFI Monthly Notes | amfiindia.com/research | Industry SIP flow data |

**Real Data Points Embedded:**
- SBI MF AUM Dec 2025: **₹12.50 lakh crore** (largest AMC)
- Industry AUM Dec 2025: **₹81 lakh crore**
- SIP Inflow Dec 2025: **₹31,002 crore** (all-time high)
- Total Folios Dec 2025: **26.12 crore**
- HDFC Top 100 NAV anchor: **₹892.45** (Oct 2024, mfapi.in code 125497)

---

## 🔬 Advanced Analytics (Day 6)

| Module | Script | Output |
|--------|--------|--------|
| Historical VaR (95%) + CVaR | `notebooks/05_advanced_analytics.ipynb` | `var_cvar_report.csv` |
| Rolling 90-Day Sharpe | `notebooks/05_advanced_analytics.ipynb` | `rolling_sharpe_chart.png` |
| Investor Cohort Analysis | `notebooks/05_advanced_analytics.ipynb` | `cohort_analysis.csv` |
| SIP Continuation Flags | `notebooks/05_advanced_analytics.ipynb` | `sip_continuity.csv` |
| Fund Recommender | `scripts/recommender.py` | Console + CSV output |
| Sector HHI Concentration | `notebooks/05_advanced_analytics.ipynb` | `sector_hhi.csv` |

**Using the Fund Recommender:**
```python
from scripts.recommender import recommend_funds

# Returns top 3 funds by Sharpe for the given risk appetite
recommend_funds(risk_appetite="Moderate")
# Output:
# ┌──────────────────────────────┬───────────────┬────────────┬──────────────┐
# │ scheme_name                  │ fund_house    │ sharpe_ratio│ risk_category│
# ├──────────────────────────────┼───────────────┼────────────┼──────────────┤
# │ HDFC Balanced Advantage Fund │ HDFC MF       │ 1.42       │ Moderate     │
# │ ICICI Pru Balanced Advantage │ ICICI Pru MF  │ 1.38       │ Moderate     │
# │ SBI Equity Hybrid Fund       │ SBI MF        │ 1.29       │ Moderate     │
# └──────────────────────────────┴───────────────┴────────────┴──────────────┘
```

---

## ⚠️ Common Pitfalls

| ❌ Mistake | ✅ Correct Approach |
|-----------|-------------------|
| Using random NAV without anchoring | Use real AMFI NAV values from mfapi.in |
| Hard-coded file paths | Use `pathlib.Path` for cross-platform paths |
| Ignoring weekends/holidays in NAV | Use `ffill()` after reindexing to full date range |
| CAGR with calendar days | Use trading day count `(252/n_trading_days)` |
| Dashboard without slicers | Every page must have ≥ 2 interactive slicers |
| Confusing AUM units | Use `aum_lakh_crore` vs `aum_crore` in column names |
| Uploading `.db` files to GitHub | Add `*.db` to `.gitignore`; upload `schema.sql` instead |

---

## 📦 Deliverables

| # | Deliverable | Format | Weight |
|---|-------------|--------|--------|
| D1 | ETL Pipeline Script | `etl_pipeline.py` | 15% |
| D2 | SQLite Database | `bluestock_mf.db` + `schema.sql` | 10% |
| D3 | EDA Notebook | `03_eda_analysis.ipynb` | 15% |
| D4 | Performance Metrics Notebook | `04_performance_analytics.ipynb` | 15% |
| D5 | Interactive Power BI Dashboard | `bluestock_mf.pbix` | 20% |
| D6 | Advanced Analytics Notebook | `05_advanced_analytics.ipynb` | 10% |
| D7 | Final Report + Presentation | `Final_Report.pdf` + `Presentation.pptx` | 15% |

---

## 🎁 Bonus Challenges

- [ ] Deploy ETL as scheduled script — auto-fetch NAV from mfapi.in every weekday at 8 PM
- [ ] Build Streamlit web app as alternative to Power BI
- [ ] Monte Carlo simulation — project NAV growth over 5 years with uncertainty bands
- [ ] Markowitz Efficient Frontier — portfolio optimisation for 5 selected funds
- [ ] Automated HTML email report — weekly performance summary

---

## 🔗 Git Workflow

```bash
# Initial setup
git init
git remote add origin https://github.com/pramodpatil0520/bluestock-mf-capstone.git

# Daily commits
git add .
git commit -m "Day 1: Data ingestion complete"
git push origin main

# Final submission
git add .
git commit -m "Final: Complete Bluestock MF Capstone"
git push origin main
git tag v1.0
git push origin v1.0
```

**`.gitignore` (important):**
```
*.db
*.pyc
__pycache__/
venv/
.env
data/raw/*.csv    # optional — exclude large raw files
*.pbix            # optional — Power BI binary
```

---

## 📄 License & Disclaimer

All data used in this project is sourced from publicly available information published by AMFI India, NSE, BSE, and open APIs (mfapi.in). This project is for **educational purposes only** and does not constitute financial advice. Mutual Fund investments are subject to market risks.

---

*Prepared by **Pramod Vinayak Patil**  · June 2026*

**Pramod Patil** —  building hands-on skills in Data Analysis and Machine Learning.


| 📧 **Email** | [pramodpatil0520@gmail.com](mailto:pramodpatil0520@gmail.com) |
| 🔗 **LinkedIn** | [linkedin.com/in/pramod-patil-a00309265](https://linkedin.com/in/pramod-patil-a00309265) |
| 🐙 **GitHub** | [github.com/pramodpatil0520](https://github.com/pramodpatil0520) |

