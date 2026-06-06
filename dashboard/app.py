import sys
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── sys.path extension to import scripts ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from scripts.recommender import recommend_funds, load_fund_universe
    from scripts.monte_carlo import run_monte_carlo_simulation
    from scripts.markowitz import optimize_portfolio
except ImportError:
    st.error("Could not import core analytical engines. Ensure all files in scripts/ are present.")

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bluestock Mutual Fund Analytics Portal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Premium Styling ───────────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* Main background */
        .stApp { background: #0a0f1e; }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1b35 0%, #080d1c 100%);
            border-right: 1px solid #1e2d4a;
        }
        [data-testid="stSidebar"] * { color: #e2e8f0; }
        [data-testid="stSidebar"] .stRadio label { font-size: 0.93rem; }

        /* Metrics container */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #141e36 0%, #1a2540 100%);
            border: 1px solid #2a3d5e;
            padding: 18px 20px;
            border-radius: 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        div[data-testid="metric-container"] label {
            color: #94a3b8 !important;
            font-weight: 500;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #60a5fa !important;
            font-size: 1.7rem;
            font-weight: 700;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
            font-size: 0.82rem;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #141e36;
            border: 1px solid #2a3d5e;
            border-radius: 8px 8px 0 0;
            padding: 10px 22px;
            color: #94a3b8;
            font-weight: 600;
            transition: all 0.25s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #93c5fd;
            background-color: #1e2d4a;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            border-color: #3b82f6 !important;
        }

        /* Section dividers */
        hr { border-color: #1e2d4a; }

        /* Plotly chart containers */
        .js-plotly-plot { border-radius: 12px; }

        /* DataFrames */
        .dataframe thead th {
            background: #141e36;
            color: #93c5fd;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ── Data Loading Helpers (with SQLite & CSV fallback) ───────────────────────
DB_PATH = BASE_DIR / "data" / "db" / "bluestock_mf.db"
PROC_DIR = BASE_DIR / "data" / "processed"

# ── Plotly layout helper (used only by Main Dashboard) ───────────────────────
def _plotly_layout(fig, **kwargs):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(13,27,53,0.0)",
        plot_bgcolor="rgba(13,27,53,0.0)",
        font=dict(family="sans-serif", color="#cbd5e1"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(20,30,54,0.8)", bordercolor="#2a3d5e", borderwidth=1),
        **kwargs
    )
    fig.update_xaxes(gridcolor="#1e2d4a", zerolinecolor="#2a3d5e")
    fig.update_yaxes(gridcolor="#1e2d4a", zerolinecolor="#2a3d5e")
    return fig

@st.cache_resource
def get_db_connection():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(str(DB_PATH))

@st.cache_data
def load_scorecard_data():
    csv_path = PROC_DIR / "fund_scorecard.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql("SELECT * FROM dim_fund", conn)
            # Add placeholders or calculate ranks if scorecard is missing in DB
            return df
        except Exception as e:
            st.error(f"Error loading from Database: {e}")
    return pd.DataFrame()

@st.cache_data
def load_nav_history(amfi_codes=None):
    conn = get_db_connection()
    if conn:
        try:
            if amfi_codes:
                codes_str = ",".join([f"'{c}'" for c in amfi_codes])
                query = f"SELECT amfi_code, date, nav, daily_return_pct FROM fact_nav WHERE amfi_code IN ({codes_str}) ORDER BY date"
            else:
                query = "SELECT amfi_code, date, nav, daily_return_pct FROM fact_nav ORDER BY date"
            df = pd.read_sql(query, conn)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass

    # Fallback to processed clean_nav_history CSV
    csv_path = PROC_DIR / "clean_nav_history.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            if amfi_codes:
                df = df[df["amfi_code"].astype(str).isin([str(c) for c in amfi_codes])]
            return df
        except Exception:
            pass
    return pd.DataFrame()

# ── Dashboard-specific CSV loaders ───────────────────────────────────────────
@st.cache_data
def load_aum_fund_house():
    try:
        df = pd.read_csv(PROC_DIR / "clean_aum_fund_house.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_monthly_sip():
    try:
        df = pd.read_csv(PROC_DIR / "clean_monthly_sip.csv")
        df["month"] = pd.to_datetime(df["month"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_category_inflows():
    try:
        df = pd.read_csv(PROC_DIR / "clean_category_inflows.csv")
        df["month"] = pd.to_datetime(df["month"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_folio_count():
    try:
        df = pd.read_csv(PROC_DIR / "clean_folio_count.csv")
        df["month"] = pd.to_datetime(df["month"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_investor_txns():
    try:
        df = pd.read_csv(PROC_DIR / "clean_investor_txns.csv")
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_portfolio_holdings():
    try:
        return pd.read_csv(PROC_DIR / "clean_portfolio_holdings.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_benchmark_indices():
    try:
        df = pd.read_csv(PROC_DIR / "clean_benchmark_indices.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_scheme_perf():
    try:
        return pd.read_csv(PROC_DIR / "clean_scheme_perf.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_cagr_report():
    try:
        return pd.read_csv(PROC_DIR / "cagr_report.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_risk_metrics():
    try:
        df_var = pd.read_csv(PROC_DIR / "var_cvar_report.csv")
        df_ab  = pd.read_csv(PROC_DIR / "alpha_beta.csv")
        df_sh  = pd.read_csv(PROC_DIR / "sharpe_values.csv")
        df_so  = pd.read_csv(PROC_DIR / "sortino_values.csv")
        df_md  = pd.read_csv(PROC_DIR / "max_drawdown.csv")
        merged = df_var.merge(df_ab, on="amfi_code", how="outer")
        merged = merged.merge(df_sh, on="amfi_code", how="outer")
        merged = merged.merge(df_so, on="amfi_code", how="outer")
        merged = merged.merge(df_md, on="amfi_code", how="outer")
        return merged
    except Exception:
        return pd.DataFrame()

# Load main fund dataset
fund_universe = None
try:
    fund_universe = load_fund_universe()
except Exception as e:
    st.error(f"Error initializing fund universe: {e}")

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://bluestock.in/assets/img/logo.png", width=180)
st.sidebar.markdown("<h2 style='text-align: center; color: #63b3ed;'>Analytics Portal</h2>", unsafe_allow_html=True)
st.sidebar.write("---")

app_tab = st.sidebar.radio(
    "Select Dashboard Tab:",
    [
        "🏠 Main Dashboard",
        "🎯 Fund Recommender",
        "📊 Scorecard Explorer",
        "🔮 NAV Growth Simulator (B3)",
        "⚖️ Portfolio Optimiser (B4)",
        "📈 Visualizations Gallery"
    ]
)

st.sidebar.write("---")
st.sidebar.markdown("""
    **Developer Information**
    - Project: Bluestock MF Capstone
    - Engine: Streamlit Web UI
    - Database: SQLite 3
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 ── MAIN DASHBOARD  (NEW — all charts from CSV)
# ══════════════════════════════════════════════════════════════════════════════
if app_tab == "🏠 Main Dashboard":
    st.title("🏠 Bluestock MF — Industry Analytics Dashboard")
    st.write("Live data from processed datasets · All charts auto-generated from CSV files")
    st.markdown("---")

    # Load all data
    df_aum   = load_aum_fund_house()
    df_sip   = load_monthly_sip()
    df_cat   = load_category_inflows()
    df_folio = load_folio_count()
    df_txn   = load_investor_txns()
    df_hold  = load_portfolio_holdings()
    df_bench = load_benchmark_indices()
    df_score = load_scorecard_data()
    df_cagr  = load_cagr_report()
    df_risk  = load_risk_metrics()

    # ── KPI Row ──────────────────────────────────────────────────────────────
    st.markdown("### 📌 Key Industry Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)

    if not df_aum.empty:
        latest_aum_date = df_aum["date"].max()
        total_aum = df_aum[df_aum["date"] == latest_aum_date]["aum_crore"].sum()
        k1.metric("Total AUM", f"₹{total_aum/100000:.2f}L Cr")
    else:
        k1.metric("Total AUM", "N/A")

    if not df_sip.empty:
        latest_sip = df_sip.sort_values("month").iloc[-1]
        yoy = latest_sip.get("yoy_growth_pct", None)
        k2.metric("Monthly SIP Inflow", f"₹{latest_sip['sip_inflow_crore']:,.0f} Cr",
                  f"{yoy:.1f}% YoY" if pd.notna(yoy) else None)
    else:
        k2.metric("Monthly SIP Inflow", "N/A")

    if not df_folio.empty:
        latest_folio = df_folio.sort_values("month").iloc[-1]
        k3.metric("Total Folios", f"{latest_folio['total_folios_crore']:.2f} Cr")
    else:
        k3.metric("Total Folios", "N/A")

    if not df_txn.empty:
        k4.metric("Investor Transactions", f"{len(df_txn):,}")
    else:
        k4.metric("Investor Transactions", "N/A")

    if not df_score.empty:
        k5.metric("Funds Tracked", f"{len(df_score)}")
    else:
        k5.metric("Funds Tracked", "N/A")

    st.markdown("---")

    # ── Row 1: AUM by Fund House + SIP Trend ─────────────────────────────────
    r1c1, r1c2 = st.columns([1.1, 1])

    with r1c1:
        st.markdown("#### 🏦 Top 10 Fund Houses by AUM")
        if not df_aum.empty:
            latest_date = df_aum["date"].max()
            top_aum = (
                df_aum[df_aum["date"] == latest_date]
                .groupby("fund_house", as_index=False)["aum_crore"]
                .sum()
                .sort_values("aum_crore", ascending=False)
                .head(10)
            )
            fig_aum = px.bar(
                top_aum.sort_values("aum_crore"),
                x="aum_crore", y="fund_house", orientation="h",
                color="aum_crore", color_continuous_scale="Blues",
                labels={"aum_crore": "AUM (₹ Crore)", "fund_house": ""},
                text=top_aum.sort_values("aum_crore")["aum_crore"].apply(lambda v: f"₹{v/1e5:.2f}L Cr")
            )
            fig_aum.update_traces(textposition="outside", marker_line_width=0)
            fig_aum.update_coloraxes(showscale=False)
            _plotly_layout(fig_aum, height=370)
            st.plotly_chart(fig_aum, use_container_width=True)
        else:
            st.info("AUM data unavailable.")

    with r1c2:
        st.markdown("#### 📈 Monthly SIP Inflow Trend")
        if not df_sip.empty:
            df_sip_s = df_sip.sort_values("month")
            fig_sip = go.Figure()
            fig_sip.add_trace(go.Scatter(
                x=df_sip_s["month"], y=df_sip_s["sip_inflow_crore"],
                mode="lines+markers", name="SIP Inflow",
                line=dict(color="#3b82f6", width=2.5),
                marker=dict(size=5, color="#60a5fa"),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.12)"
            ))
            if "sip_aum_lakh_crore" in df_sip_s.columns:
                fig_sip.add_trace(go.Scatter(
                    x=df_sip_s["month"], y=df_sip_s["sip_aum_lakh_crore"] * 1e5,
                    mode="lines", name="SIP AUM (₹ Cr)",
                    line=dict(color="#a78bfa", width=2, dash="dot"),
                    yaxis="y2"
                ))
                fig_sip.update_layout(
                    yaxis2=dict(overlaying="y", side="right",
                                showgrid=False, color="#a78bfa",
                                title="SIP AUM (₹ Cr)")
                )
            _plotly_layout(fig_sip, height=370,
                           xaxis_title="Month", yaxis_title="SIP Inflow (₹ Crore)")
            st.plotly_chart(fig_sip, use_container_width=True)
        else:
            st.info("SIP data unavailable.")

    # ── Row 2: Category Inflow Heatmap + Folio Growth ────────────────────────
    r2c1, r2c2 = st.columns([1.2, 0.9])

    with r2c1:
        st.markdown("#### 🗺️ Category-wise Net Inflow Heatmap")
        if not df_cat.empty:
            df_cat["month_label"] = df_cat["month"].dt.strftime("%b %Y")
            pivot = df_cat.pivot_table(
                index="category", columns="month_label",
                values="net_inflow_crore", aggfunc="sum"
            )
            all_months = df_cat.sort_values("month")["month_label"].unique().tolist()
            pivot = pivot.reindex(columns=[m for m in all_months if m in pivot.columns])
            pivot = pivot.iloc[:, -12:]
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="RdBu", zmid=0,
                text=[[f"{v:,.0f}" for v in row] for row in pivot.values],
                texttemplate="%{text}", textfont=dict(size=9),
                hoverongaps=False, colorbar=dict(title="₹ Crore")
            ))
            _plotly_layout(fig_heat, height=350)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Category inflow data unavailable.")

    with r2c2:
        st.markdown("#### 📂 Industry Folio Growth")
        if not df_folio.empty:
            df_folio_s = df_folio.sort_values("month")
            fig_folio = go.Figure()
            folio_cols = {
                "equity_folios_crore": ("Equity",  "#3b82f6"),
                "debt_folios_crore":   ("Debt",    "#f59e0b"),
                "hybrid_folios_crore": ("Hybrid",  "#10b981"),
                "others_folios_crore": ("Others",  "#a78bfa"),
            }
            for col, (label, color) in folio_cols.items():
                if col in df_folio_s.columns:
                    fig_folio.add_trace(go.Scatter(
                        x=df_folio_s["month"], y=df_folio_s[col],
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2),
                        stackgroup="one", fill="tonexty",
                        marker=dict(size=4)
                    ))
            _plotly_layout(fig_folio, height=350,
                           xaxis_title="", yaxis_title="Folios (Crore)")
            st.plotly_chart(fig_folio, use_container_width=True)
        else:
            st.info("Folio data unavailable.")

    st.markdown("---")

    # ── Row 3: Benchmark Indices + Top 10 Scorecard ───────────────────────────
    r3c1, r3c2 = st.columns([1.2, 0.9])

    with r3c1:
        st.markdown("#### 📊 Benchmark Index Performance")
        if not df_bench.empty:
            indices = df_bench["index_name"].unique().tolist()
            sel_idx = st.multiselect(
                "Select Indices:", indices,
                default=indices[:3] if len(indices) >= 3 else indices,
                key="bench_sel"
            )
            palette = ["#3b82f6", "#f59e0b", "#10b981", "#f43f5e", "#a78bfa"]
            fig_bench = go.Figure()
            for i, idx_name in enumerate(sel_idx):
                sub = df_bench[df_bench["index_name"] == idx_name].sort_values("date")
                base = sub["close_value"].iloc[0]
                fig_bench.add_trace(go.Scatter(
                    x=sub["date"], y=(sub["close_value"] / base) * 100,
                    mode="lines", name=idx_name,
                    line=dict(color=palette[i % len(palette)], width=2)
                ))
            _plotly_layout(fig_bench, height=360,
                           xaxis_title="Date", yaxis_title="Normalised Value (Base=100)")
            st.plotly_chart(fig_bench, use_container_width=True)
        else:
            st.info("Benchmark data unavailable.")

    with r3c2:
        st.markdown("#### 🏆 Top 10 Funds — Composite Score")
        if not df_score.empty:
            top10 = df_score.sort_values("composite_score", ascending=False).head(10)
            short_names = top10["scheme_name"].str.replace(r"Fund.*", "Fund", regex=True).str[:30]
            fig_top = px.bar(
                top10.assign(short_name=short_names.values),
                x="composite_score", y="short_name", orientation="h",
                color="composite_score",
                color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#93c5fd"]],
                text=top10["composite_score"].apply(lambda v: f"{v:.1f}"),
                labels={"composite_score": "Score", "short_name": ""}
            )
            fig_top.update_traces(textposition="outside", marker_line_width=0)
            fig_top.update_coloraxes(showscale=False)
            _plotly_layout(fig_top, height=360)
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("Scorecard data unavailable.")

    st.markdown("---")

    # ── Row 4: Investor Demographics ──────────────────────────────────────────
    st.markdown("### 👥 Investor Demographics & Transaction Insights")
    d1, d2, d3, d4 = st.columns(4)

    if not df_txn.empty:
        with d1:
            age_cnt = df_txn["age_group"].value_counts().reset_index()
            age_cnt.columns = ["age_group", "count"]
            age_order = ["18-25", "26-35", "36-45", "46-55", "56+"]
            age_cnt["age_group"] = pd.Categorical(age_cnt["age_group"], categories=age_order, ordered=True)
            age_cnt = age_cnt.sort_values("age_group")
            fig_age = px.bar(age_cnt, x="age_group", y="count",
                             color="count", color_continuous_scale="Blues",
                             labels={"age_group": "Age Group", "count": "Investors"})
            fig_age.update_coloraxes(showscale=False)
            _plotly_layout(fig_age, height=280, title="Age Distribution")
            st.plotly_chart(fig_age, use_container_width=True)

        with d2:
            tier_cnt = df_txn["city_tier"].value_counts().reset_index()
            tier_cnt.columns = ["tier", "count"]
            fig_tier = px.pie(tier_cnt, names="tier", values="count",
                              hole=0.55, color_discrete_sequence=["#3b82f6", "#f59e0b"])
            fig_tier.update_traces(textinfo="percent+label", pull=[0.03, 0])
            _plotly_layout(fig_tier, height=280, title="T30 vs B30 Investors")
            st.plotly_chart(fig_tier, use_container_width=True)

        with d3:
            gen_cnt = df_txn["gender"].value_counts().reset_index()
            gen_cnt.columns = ["gender", "count"]
            fig_gen = px.pie(gen_cnt, names="gender", values="count",
                             hole=0.55, color_discrete_sequence=["#6366f1", "#ec4899"])
            fig_gen.update_traces(textinfo="percent+label", pull=[0.03, 0])
            _plotly_layout(fig_gen, height=280, title="Gender Split")
            st.plotly_chart(fig_gen, use_container_width=True)

        with d4:
            txn_cnt = df_txn["transaction_type"].value_counts().reset_index()
            txn_cnt.columns = ["type", "count"]
            fig_txnt = px.pie(txn_cnt, names="type", values="count",
                              hole=0.55, color_discrete_sequence=["#10b981", "#f59e0b", "#f43f5e"])
            fig_txnt.update_traces(textinfo="percent+label", pull=[0.03, 0, 0])
            _plotly_layout(fig_txnt, height=280, title="Transaction Type Mix")
            st.plotly_chart(fig_txnt, use_container_width=True)
    else:
        st.info("Investor transaction data unavailable.")

    # ── Row 5: State-wise Investment + Sector Allocation ──────────────────────
    r5c1, r5c2 = st.columns([1.2, 0.9])

    with r5c1:
        st.markdown("#### 🗺️ State-wise Total Investment")
        if not df_txn.empty:
            state_inv = (
                df_txn.groupby("state", as_index=False)["amount_inr"]
                .sum()
                .sort_values("amount_inr", ascending=False)
                .head(15)
            )
            fig_state = px.bar(
                state_inv.sort_values("amount_inr"),
                x="amount_inr", y="state", orientation="h",
                color="amount_inr", color_continuous_scale="Blues",
                labels={"amount_inr": "Total Investment (₹)", "state": ""},
                text=state_inv.sort_values("amount_inr")["amount_inr"].apply(lambda v: f"₹{v/1e7:.1f}Cr")
            )
            fig_state.update_traces(textposition="outside", marker_line_width=0)
            fig_state.update_coloraxes(showscale=False)
            _plotly_layout(fig_state, height=400)
            st.plotly_chart(fig_state, use_container_width=True)
        else:
            st.info("Transaction data unavailable.")

    with r5c2:
        st.markdown("#### 🏭 Portfolio Sector Allocation")
        if not df_hold.empty:
            sector_wt = (
                df_hold.groupby("sector", as_index=False)["weight_pct"]
                .sum()
                .sort_values("weight_pct", ascending=False)
            )
            top_sectors = sector_wt.head(10)
            others_wt = sector_wt.iloc[10:]["weight_pct"].sum()
            if others_wt > 0:
                top_sectors = pd.concat([
                    top_sectors,
                    pd.DataFrame([{"sector": "Others", "weight_pct": others_wt}])
                ], ignore_index=True)
            fig_sector = px.pie(top_sectors, names="sector", values="weight_pct",
                                hole=0.45, color_discrete_sequence=px.colors.qualitative.Bold)
            fig_sector.update_traces(
                textinfo="percent", textposition="outside",
                pull=[0.04 if i == 0 else 0 for i in range(len(top_sectors))]
            )
            _plotly_layout(fig_sector, height=400)
            st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("Portfolio holdings data unavailable.")

    st.markdown("---")

    # ── Row 6: Risk-Return Scatter + CAGR Comparison ─────────────────────────
    r6c1, r6c2 = st.columns(2)

    with r6c1:
        st.markdown("#### ⚡ Risk–Return Scatter (Sharpe vs Max Drawdown)")
        if not df_score.empty:
            merged_risk = df_score.copy()
            fig_scatter = px.scatter(
                merged_risk,
                x="max_drawdown_pct", y="sharpe_ratio",
                color="category" if "category" in merged_risk.columns else None,
                size="composite_score" if "composite_score" in merged_risk.columns else None,
                hover_name="scheme_name" if "scheme_name" in merged_risk.columns else None,
                labels={"max_drawdown_pct": "Max Drawdown (%)", "sharpe_ratio": "Sharpe Ratio"},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_scatter.update_traces(marker=dict(opacity=0.82, line=dict(width=0.5, color="#1e293b")))
            _plotly_layout(fig_scatter, height=360)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Scorecard data unavailable.")

    with r6c2:
        st.markdown("#### 📊 Fund CAGR Comparison (1yr / 3yr / 5yr)")
        if not df_cagr.empty and not df_score.empty:
            cagr_merged = df_cagr.merge(
                df_score[["amfi_code", "scheme_name"]], on="amfi_code", how="left"
            ).dropna(subset=["scheme_name"])
            cagr_merged["short_name"] = cagr_merged["scheme_name"].str[:22] + "…"
            top_cagr = cagr_merged.nlargest(12, "cagr_3yr_pct")
            fig_cagr = go.Figure()
            for col, color, label in [
                ("cagr_1yr_pct", "#f59e0b", "1-Yr CAGR"),
                ("cagr_3yr_pct", "#3b82f6", "3-Yr CAGR"),
                ("cagr_5yr_pct", "#10b981", "5-Yr CAGR"),
            ]:
                if col in top_cagr.columns:
                    fig_cagr.add_trace(go.Bar(
                        name=label, x=top_cagr["short_name"], y=top_cagr[col],
                        marker_color=color, opacity=0.85
                    ))
            fig_cagr.update_layout(barmode="group")
            _plotly_layout(fig_cagr, height=360, xaxis_title="", yaxis_title="CAGR (%)")
            fig_cagr.update_xaxes(tickangle=-35, tickfont=dict(size=9))
            st.plotly_chart(fig_cagr, use_container_width=True)
        else:
            st.info("CAGR data unavailable.")

    st.markdown("---")

    # ── Row 7: Investment by Age + Payment Mode ───────────────────────────────
    r7c1, r7c2 = st.columns(2)

    with r7c1:
        st.markdown("#### 💰 Investment Amount by Age Group")
        if not df_txn.empty:
            age_order = ["18-25", "26-35", "36-45", "46-55", "56+"]
            age_inv = df_txn.groupby("age_group", as_index=False)["amount_inr"].sum()
            age_inv["age_group"] = pd.Categorical(age_inv["age_group"], categories=age_order, ordered=True)
            age_inv = age_inv.sort_values("age_group")
            fig_age_inv = px.bar(
                age_inv, x="age_group", y="amount_inr",
                color="age_group",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={"age_group": "Age Group", "amount_inr": "Total Investment (₹)"},
                text=age_inv["amount_inr"].apply(lambda v: f"₹{v/1e7:.1f}Cr")
            )
            fig_age_inv.update_traces(textposition="outside", marker_line_width=0)
            _plotly_layout(fig_age_inv, height=330, showlegend=False)
            st.plotly_chart(fig_age_inv, use_container_width=True)
        else:
            st.info("Transaction data unavailable.")

    with r7c2:
        st.markdown("#### 💳 Payment Mode Distribution")
        if not df_txn.empty and "payment_mode" in df_txn.columns:
            pay_cnt = df_txn["payment_mode"].value_counts().reset_index()
            pay_cnt.columns = ["mode", "count"]
            fig_pay = px.bar(
                pay_cnt, x="mode", y="count",
                color="mode",
                color_discrete_sequence=px.colors.qualitative.Safe,
                labels={"mode": "Payment Mode", "count": "Count"},
                text=pay_cnt["count"]
            )
            fig_pay.update_traces(textposition="outside", marker_line_width=0)
            _plotly_layout(fig_pay, height=330, showlegend=False)
            st.plotly_chart(fig_pay, use_container_width=True)
        else:
            st.info("Payment mode data unavailable.")

    st.markdown("---")
    st.caption("📌 All charts are generated in real-time from processed CSV datasets.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 ── FUND RECOMMENDER  (original — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
elif app_tab == "🎯 Fund Recommender":
    st.title("🎯 Mutual Fund Recommendation Engine")
    st.write("Enter your financial goals and risk preference to find the most suitable mutual funds ranked by risk-adjusted return (Sharpe Ratio).")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_appetite = st.selectbox(
            "Select Risk Appetite:",
            ["Low", "Moderate", "High", "Very High"],
            index=1,
            help="Low: Capital preservation. Moderate: Balanced. High/Very High: Equity growth oriented."
        )
    with col2:
        horizon = st.selectbox(
            "Investment Horizon:",
            ["Short", "Medium", "Long"],
            index=2,
            help="Short: < 1 year. Medium: 1-3 years. Long: 3+ years."
        )
    with col3:
        plan_type = st.radio(
            "Mutual Fund Plan:",
            ["Direct", "Regular"],
            help="Direct plans have lower expense ratios and higher returns."
        )
    with col4:
        top_n = st.slider(
            "Number of Recommendations:",
            min_value=1, max_value=10, value=3
        )

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Analyzing fund universe..."):
            try:
                rec_df = recommend_funds(
                    risk_appetite=risk_appetite,
                    horizon=horizon,
                    plan=plan_type,
                    top_n=top_n,
                    fund_universe=fund_universe
                )

                if rec_df.empty:
                    st.warning("⚠️ No mutual funds match the selected criteria. Try adjusting filters.")
                else:
                    st.success(f"Top {len(rec_df)} recommended funds for your profile:")

                    display_cols = {
                        "amfi_code": "AMFI Code",
                        "scheme_name": "Scheme Name",
                        "fund_house": "Fund House",
                        "sub_category": "Category",
                        "risk_category": "Risk Category",
                        "sharpe_ratio": "Sharpe Ratio",
                        "cagr_3yr_pct": "3-Yr CAGR (%)",
                        "return_3yr_pct": "3-Yr Return (%)",
                        "alpha_annualised": "Annualized Alpha (%)",
                        "alpha": "Alpha (%)",
                        "max_drawdown_pct": "Max Drawdown (%)",
                        "expense_ratio_pct": "Expense Ratio (%)",
                        "plan": "Plan"
                    }

                    available_cols = [c for c in display_cols.keys() if c in rec_df.columns]
                    formatted_df = rec_df[available_cols].rename(columns=display_cols)

                    st.dataframe(
                        formatted_df.style.format({
                            "Sharpe Ratio": "{:.2f}",
                            "3-Yr CAGR (%)": "{:.2f}%",
                            "3-Yr Return (%)": "{:.2f}%",
                            "Annualized Alpha (%)": "{:.2f}%",
                            "Alpha (%)": "{:.2f}%",
                            "Max Drawdown (%)": "{:.2f}%",
                            "Expense Ratio (%)": "{:.2f}%"
                        }),
                        use_container_width=True
                    )

                    st.info("""
                        * **Sharpe Ratio** measures risk-adjusted performance (higher is better).
                        * **Max Drawdown** shows the worst historical peak-to-trough drop.
                        * **Plan**: Direct plans save agent commissions, boosting your yields.
                    """)
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 ── SCORECARD EXPLORER  (original — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
elif app_tab == "📊 Scorecard Explorer":
    st.title("📊 Scorecard & Performance Explorer")
    st.write("Browse and filter the composite scores and key metrics of all tracked mutual fund schemes.")

    scorecard = load_scorecard_data()

    if scorecard.empty:
        st.error("No scorecard data found. Ensure metrics calculation has run.")
    else:
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            categories = ["All"] + sorted(scorecard["category"].dropna().unique().tolist())
            selected_cat = st.selectbox("Filter by Asset Category:", categories)
        with col2:
            risk_cats = ["All"] + sorted(scorecard["risk_category"].dropna().unique().tolist())
            selected_risk = st.selectbox("Filter by Risk Rating:", risk_cats)
        with col3:
            search_query = st.text_input("Search Fund Scheme Name:", "").strip()

        filtered_df = scorecard.copy()
        if selected_cat != "All":
            filtered_df = filtered_df[filtered_df["category"] == selected_cat]
        if selected_risk != "All":
            filtered_df = filtered_df[filtered_df["risk_category"] == selected_risk]
        if search_query:
            filtered_df = filtered_df[filtered_df["scheme_name"].str.contains(search_query, case=False, na=False)]

        st.markdown("### Selection Summary")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Total Funds", len(filtered_df))
        with mcol2:
            avg_cagr = filtered_df["cagr_3yr_pct"].mean() if "cagr_3yr_pct" in filtered_df.columns else 0.0
            st.metric("Avg 3-Yr CAGR", f"{avg_cagr:.2f}%" if not np.isnan(avg_cagr) else "N/A")
        with mcol3:
            avg_sharpe = filtered_df["sharpe_ratio"].mean() if "sharpe_ratio" in filtered_df.columns else 0.0
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}" if not np.isnan(avg_sharpe) else "N/A")
        with mcol4:
            avg_er = filtered_df["expense_ratio_pct"].mean() if "expense_ratio_pct" in filtered_df.columns else 0.0
            st.metric("Avg Expense Ratio", f"{avg_er:.2f}%" if not np.isnan(avg_er) else "N/A")

        st.write("---")
        st.markdown("### Ranks and Scoring table")
        st.write("Double click columns to sort. Hover over headers for details.")

        display_scorecard_cols = {
            "rank_overall": "Overall Rank",
            "amfi_code": "AMFI Code",
            "scheme_name": "Scheme Name",
            "category": "Category",
            "risk_category": "Risk",
            "composite_score": "Composite Score (0-100)",
            "cagr_3yr_pct": "3-Yr Return CAGR%",
            "sharpe_ratio": "Sharpe Ratio",
            "alpha_annualised": "Annualized Alpha%",
            "expense_ratio_pct": "Expense Ratio%",
            "max_drawdown_pct": "Max Drawdown%"
        }

        present_cols = [c for c in display_scorecard_cols.keys() if c in filtered_df.columns]
        table_df = filtered_df[present_cols].rename(columns=display_scorecard_cols)

        st.dataframe(
            table_df.style.format({
                "Composite Score (0-100)": "{:.2f}",
                "3-Yr Return CAGR%": "{:.2f}%",
                "Sharpe Ratio": "{:.2f}",
                "Annualized Alpha%": "{:.2f}%",
                "Expense Ratio%": "{:.2f}%",
                "Max Drawdown%": "{:.2f}%"
            }),
            use_container_width=True,
            height=400
        )

        st.markdown("---")
        st.markdown("### 🔍 Individual Fund Detailed Breakdown")

        selected_scheme = st.selectbox(
            "Select a specific Fund to inspect details:",
            filtered_df["scheme_name"].tolist() if not filtered_df.empty else ["No funds match filters"]
        )

        if selected_scheme and selected_scheme != "No funds match filters":
            fund_details = filtered_df[filtered_df["scheme_name"] == selected_scheme].iloc[0]

            dcol1, dcol2 = st.columns([1, 1])
            with dcol1:
                st.markdown(f"#### {fund_details['scheme_name']}")
                st.write(f"**AMFI Code**: {fund_details['amfi_code']}")
                st.write(f"**Category**: {fund_details.get('category', 'N/A')} ({fund_details.get('sub_category', 'N/A')})")
                st.write(f"**Fund House**: {fund_details.get('fund_house', 'N/A')}")
                st.write(f"**Risk Level**: {fund_details.get('risk_category', 'N/A')}")
                st.write(f"**Plan**: {fund_details.get('plan', 'N/A')}")

            with dcol2:
                st.markdown("#### Performance Metrics")
                f1, f2, f3 = st.columns(3)
                f1.metric("Composite Score", f"{fund_details.get('composite_score', 0.0):.2f}")
                f2.metric("Sharpe Ratio", f"{fund_details.get('sharpe_ratio', 0.0):.2f}")
                f3.metric("Sortino Ratio", f"{fund_details.get('sortino_ratio', 0.0):.2f}" if 'sortino_ratio' in fund_details else "N/A")

                f4, f5, f6 = st.columns(3)
                f4.metric("3-Yr Return CAGR", f"{fund_details.get('cagr_3yr_pct', 0.0):.2f}%")
                f5.metric("Alpha (vs Bench)", f"{fund_details.get('alpha_annualised', 0.0):.2f}%")
                f6.metric("Max Drawdown", f"{fund_details.get('max_drawdown_pct', 0.0):.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 ── NAV GROWTH SIMULATOR  (original — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
elif app_tab == "🔮 NAV Growth Simulator (B3)":
    st.title("🔮 NAV Monte Carlo Growth Simulator (Bonus Challenge B3)")
    st.write("Simulate 5-year future returns and probability distributions for individual mutual funds based on historical daily return volatility.")

    st.markdown("---")

    nav_history = load_nav_history()
    if nav_history.empty:
        st.error("No historical NAV data found. Please run the ETL pipeline and metrics computer first.")
    else:
        conn = get_db_connection()
        scheme_map = {}
        if conn:
            try:
                schemes_df = pd.read_sql("SELECT amfi_code, scheme_name FROM dim_fund", conn)
                scheme_map = dict(zip(schemes_df["amfi_code"].astype(str), schemes_df["scheme_name"]))
            except Exception:
                pass

        available_codes = nav_history["amfi_code"].astype(str).unique().tolist()
        fund_options = {scheme_map.get(c, f"AMFI {c}"): c for c in available_codes}

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_fund_name = st.selectbox("Select Fund for Simulation:", list(fund_options.keys()))
            selected_fund_code = fund_options[selected_fund_name]
        with col2:
            sim_years = st.slider("Simulation Horizon (Years):", 1, 5, 5)
        with col3:
            num_sims = st.slider("Number of Simulation Paths:", 100, 1000, 250, step=50)

        initial_inv = st.number_input("Initial Investment Amount (INR):", value=10000, min_value=1000, step=1000)

        fund_nav = nav_history[nav_history["amfi_code"].astype(str) == str(selected_fund_code)].sort_values("date")

        if fund_nav.empty or len(fund_nav) < 30:
            st.warning("⚠️ Selected fund has insufficient historical NAV data to calculate returns parameters.")
        else:
            if "daily_return_pct" in fund_nav.columns and fund_nav["daily_return_pct"].notna().sum() > 30:
                returns = fund_nav["daily_return_pct"].dropna()
            else:
                returns = fund_nav["nav"].pct_change().dropna()

            try:
                sim_res = run_monte_carlo_simulation(
                    returns=returns,
                    sim_years=sim_years,
                    num_sims=num_sims,
                    initial_inv=initial_inv,
                    seed=42
                )
                mu_ann = sim_res["mu_ann"]
                sigma_ann = sim_res["sigma_ann"]
                portfolio_paths = sim_res["portfolio_paths"]
                median_path = sim_res["median_path"]
                lower_bound = sim_res["lower_bound"]
                upper_bound = sim_res["upper_bound"]
                time_axis = sim_res["time_axis"]
            except Exception as e:
                st.error(f"Error running Monte Carlo simulation: {e}")
                st.stop()

            st.write(f"**Historical Parameters Calculated (Annualized):** Expected Return (Drift) = **{mu_ann*100:.2f}%**, Volatility = **{sigma_ann*100:.2f}%**")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_axis, y=upper_bound,
                line=dict(color='rgba(72, 187, 120, 0.2)'),
                fill=None, name='Optimistic (95th Pct)'
            ))
            fig.add_trace(go.Scatter(
                x=time_axis, y=lower_bound,
                line=dict(color='rgba(245, 101, 101, 0.2)'),
                fill='tonexty', fillcolor='rgba(99, 179, 237, 0.1)',
                name='Pessimistic (5th Pct)'
            ))
            fig.add_trace(go.Scatter(
                x=time_axis, y=median_path,
                line=dict(color='#3182ce', width=3),
                name='Median Projection'
            ))
            for i in range(min(15, num_sims)):
                fig.add_trace(go.Scatter(
                    x=time_axis, y=portfolio_paths[:, i],
                    line=dict(width=1, dash='dot', color='rgba(160, 174, 192, 0.3)'),
                    name=f'Path {i+1}', showlegend=False
                ))

            fig.update_layout(
                title=f"5-Year Monte Carlo NAV Growth Simulation for {selected_fund_name}",
                xaxis_title="Years",
                yaxis_title="Investment Value (INR)",
                template="plotly_dark",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Projection Summary & Probability Analytics")
            sc1, sc2, sc3, sc4 = st.columns(4)

            final_median = median_path[-1]
            final_lower = lower_bound[-1]
            final_upper = upper_bound[-1]
            final_values = portfolio_paths[-1, :]
            prob_profit = np.mean(final_values > initial_inv) * 100
            median_cagr = ((final_median / initial_inv) ** (1 / sim_years) - 1) * 100

            sc1.metric("Median Projected Value", f"₹{final_median:,.2f}", f"{median_cagr:.2f}% Projected CAGR")
            sc2.metric("Pessimistic Value (5th%)", f"₹{final_lower:,.2f}", help="95% chance the fund value will exceed this amount.")
            sc3.metric("Optimistic Value (95th%)", f"₹{final_upper:,.2f}", help="Only a 5% chance the fund value will exceed this amount.")
            sc4.metric("Probability of Profit", f"{prob_profit:.1f}%", help="Percentage of simulated paths that ended above the initial investment.")

            st.write("---")
            st.caption("Disclaimer: Monte Carlo simulations are based on historical return volatility parameters and assume future distributions will be lognormal. Past performance is not an indicator of future results.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 ── PORTFOLIO OPTIMISER  (original — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
elif app_tab == "⚖️ Portfolio Optimiser (B4)":
    st.title("⚖️ Markowitz Efficient Frontier Portfolio Optimiser (Bonus Challenge B4)")
    st.write("Select mutual funds to run Modern Portfolio Theory (MPT) calculations, generating the Efficient Frontier and optimizing asset weights.")

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 1])

    scorecard = load_scorecard_data()
    if scorecard.empty:
        st.error("No mutual funds found in database scorecard.")
    else:
        fund_list = scorecard["scheme_name"].dropna().unique().tolist()
        default_funds = fund_list[:5] if len(fund_list) >= 5 else fund_list

        with col1:
            selected_funds = st.multiselect(
                "Select Funds to Optimize (Recommend 5):",
                fund_list, default=default_funds, max_selections=10
            )
        with col2:
            rf_rate = st.slider("Risk-Free Rate (% per annum):", 4.0, 9.0, 6.5, step=0.1) / 100.0
        with col3:
            num_portfolios = st.slider("Random Portfolios to Simulate:", 500, 5000, 1500, step=250)

        if len(selected_funds) < 2:
            st.warning("⚠️ Please select at least 2 mutual funds to run portfolio optimization.")
        else:
            with st.spinner("Fetching daily returns and calculating correlation matrices..."):
                conn = get_db_connection()
                if not conn:
                    st.error("Database connection unavailable.")
                else:
                    selected_codes = scorecard[scorecard["scheme_name"].isin(selected_funds)]["amfi_code"].tolist()
                    nav_df = load_nav_history(selected_codes)

                    if nav_df.empty:
                        st.error("Could not fetch historical NAVs for optimization.")
                    else:
                        code_to_name = dict(zip(scorecard["amfi_code"].astype(str), scorecard["scheme_name"]))
                        nav_df["fund_name"] = nav_df["amfi_code"].astype(str).map(code_to_name)

                        returns_df = nav_df.pivot(index="date", columns="fund_name", values="daily_return_pct")
                        returns_df = returns_df.dropna()

                        if len(returns_df) < 30:
                            st.error("Too few overlapping trading days between the selected funds. Try choosing different funds.")
                        else:
                            try:
                                opt_res = optimize_portfolio(
                                    returns_df=returns_df,
                                    num_portfolios=num_portfolios,
                                    rf_rate=rf_rate,
                                    seed=42
                                )
                                results = opt_res["results"]
                                weights_record = opt_res["weights_record"]
                                sdp = opt_res["max_sharpe"]["volatility"]
                                rp = opt_res["max_sharpe"]["return"]
                                max_sharpe_weights = opt_res["max_sharpe"]["weights"]
                                max_sharpe_ratio = opt_res["max_sharpe"]["sharpe"]
                                sdp_min = opt_res["min_volatility"]["volatility"]
                                rp_min = opt_res["min_volatility"]["return"]
                                min_vol_weights = opt_res["min_volatility"]["weights"]
                                min_vol_sharpe = opt_res["min_volatility"]["sharpe"]
                            except Exception as e:
                                st.error(f"Error running Markowitz optimization: {e}")
                                st.stop()

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=results[1], y=results[0], mode='markers',
                                marker=dict(
                                    color=results[2], colorscale='Viridis',
                                    showscale=True, colorbar=dict(title="Sharpe Ratio"), size=5
                                ),
                                name="Simulated Portfolios",
                                text=[f"Sharpe: {sr:.2f}" for sr in results[2]],
                                hoverinfo='text+x+y'
                            ))
                            fig.add_trace(go.Scatter(
                                x=[sdp], y=[rp], mode='markers',
                                marker=dict(color='red', size=15, symbol='star'),
                                name="Max Sharpe Ratio"
                            ))
                            fig.add_trace(go.Scatter(
                                x=[sdp_min], y=[rp_min], mode='markers',
                                marker=dict(color='cyan', size=15, symbol='star'),
                                name="Min Volatility"
                            ))

                            fig.update_layout(
                                title="Modern Portfolio Theory — Efficient Frontier",
                                xaxis_title="Annualized Volatility (Risk)",
                                yaxis_title="Annualized Expected Return",
                                template="plotly_dark", height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.markdown("### 📊 Optimized Asset Allocation Weights")
                            w_col1, w_col2 = st.columns(2)

                            with w_col1:
                                st.markdown("#### 🚀 Maximum Sharpe Ratio Portfolio")
                                st.write(f"**Expected Return**: **{rp*100:.2f}%** | **Risk (Vol)**: **{sdp*100:.2f}%** | **Sharpe**: **{max_sharpe_ratio:.2f}**")
                                max_weights_df = pd.DataFrame({
                                    "Fund Scheme": selected_funds,
                                    "Allocation Weight (%)": max_sharpe_weights * 100
                                }).sort_values("Allocation Weight (%)", ascending=False)
                                st.dataframe(
                                    max_weights_df.style.format({"Allocation Weight (%)": "{:.2f}%"}),
                                    use_container_width=True, hide_index=True
                                )

                            with w_col2:
                                st.markdown("#### 🛡️ Minimum Volatility Portfolio")
                                st.write(f"**Expected Return**: **{rp_min*100:.2f}%** | **Risk (Vol)**: **{sdp_min*100:.2f}%** | **Sharpe**: **{min_vol_sharpe:.2f}**")
                                min_weights_df = pd.DataFrame({
                                    "Fund Scheme": selected_funds,
                                    "Allocation Weight (%)": min_vol_weights * 100
                                }).sort_values("Allocation Weight (%)", ascending=False)
                                st.dataframe(
                                    min_weights_df.style.format({"Allocation Weight (%)": "{:.2f}%"}),
                                    use_container_width=True, hide_index=True
                                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 ── VISUALIZATIONS GALLERY  (original — static images, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
elif app_tab == "📈 Visualizations Gallery":
    st.title("📈 Pre-generated Analytical Charts & Insights")
    st.write("Browse through deep-dive visualizations showing market trends, demographics, and correlation structures.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    charts_metadata = {
        "Investor Demographics": {
            "Investor Age Distribution.png": "Distribution of mutual fund investor ages in the database.",
            "Investment Amount by Age Group.jpeg": "Aggregated investment amount split across defined age buckets.",
            "State-wise Investment Distribution.png": "Geographical split of investment volumes across Indian states.",
            "T30 vs B30 Investor Distribution.png": "Distribution comparing T30 (Top 30 cities) vs B30 (Beyond 30 cities) investors."
        },
        "Market Trends & Inflows": {
            "Monthly SIP Inflow Trend.png": "Aggregated monthly Systematic Investment Plan (SIP) inflow growth trajectory.",
            "Category-wise Inflow Heatmap.png": "Heatmap showing seasonal and monthly inflows split across fund categories.",
            "Industry Folio Growth.png": "Cumulative growth of active mutual fund folios in the industry."
        },
        "Fund & Portfolio Performance": {
            "NAV Trend Analysis (Top 10 Funds).png": "Historical daily NAV trend of the top 10 mutual funds by scoring.",
            "Sector allocation.png": "Sector allocation breakdown across the aggregated portfolio holding database.",
            "Top 10 Fund Houses by AUM.png": "Assets Under Management (AUM) comparison for the top 10 Asset Management Companies (AMCs).",
            "mutual fund correlation matrix.png": "Correlation coefficient matrix based on daily NAV returns of key funds."
        }
    }

    with col1:
        selected_category = st.selectbox("Select Analysis Category:", list(charts_metadata.keys()))

    with col2:
        chart_options = list(charts_metadata[selected_category].keys())
        selected_chart_name = st.selectbox("Select Visual Chart:", chart_options)

    # Load and display chart image
    charts_dir = BASE_DIR / "dashboard" / "python_charts"
    chart_path = charts_dir / selected_chart_name

    if not chart_path.exists():
        st.warning(f"⚠️ Chart file not found at path: `{chart_path}`")
    else:
        st.markdown("### Chart Visualization")
        st.image(str(chart_path), caption=selected_chart_name, use_container_width=True)
        st.markdown("### Key Analytical Insights")
        st.write(charts_metadata[selected_category][selected_chart_name])
