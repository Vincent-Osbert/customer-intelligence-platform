"""
app.py — Streamlit Dashboard
-----------------------------
Customer Intelligence Platform — Live Business Dashboard

Tabs:
  1. Business Overview   — KPIs, revenue trend, daily transaction counter
  2. Customer Segments   — Segment profiles, CLV, recommended actions
  3. Live Feed           — Real-time transaction stream (auto-refresh)

Auto-refresh: the dashboard re-reads data files every N seconds so any
new data added by simulate_live.py appears automatically without restarting.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

REFRESH_INTERVAL = 30   # seconds


# ── Helper loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_INTERVAL)
def load_segment_report() -> pd.DataFrame | None:
    p = REPORTS_DIR / "segment_business_report.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=REFRESH_INTERVAL)
def load_live_feed() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "live_transactions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["InvoiceDate"])
    return df


@st.cache_data(ttl=REFRESH_INTERVAL)
def load_clean_transactions() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "clean_transactions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["InvoiceDate"])
    return df


@st.cache_data(ttl=REFRESH_INTERVAL)
def load_cluster_profiles() -> pd.DataFrame | None:
    p = REPORTS_DIR / "cluster_profiles.csv"
    return pd.read_csv(p) if p.exists() else None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 Customer Intel")
    st.markdown("---")
    st.caption(f"Dashboard refreshes every {REFRESH_INTERVAL}s automatically.")
    st.caption("To simulate new data, run:\n`python src/simulate_live.py`")
    st.markdown("---")
    auto_refresh = st.toggle("Auto-Refresh", value=True)
    if auto_refresh:
        st.caption("✅ Live mode active")


# ── Main content ──────────────────────────────────────────────────────────────

st.title("🧠 Customer Intelligence Platform")
st.caption("Data-driven customer segmentation for smarter marketing decisions.")

tab1, tab2, tab3 = st.tabs(["📊 Business Overview", "👥 Customer Segments", "⚡ Live Feed"])


# ─── Tab 1: Business Overview ────────────────────────────────────────────────

with tab1:
    st.subheader("Business Overview")

    report = load_segment_report()
    if report is None:
        st.warning("Run `python train.py` first to generate reports.")
    else:
        total_customers = int(report["CustomerCount"].sum())
        total_revenue = report["TotalRevenue"].sum()
        avg_clv = report["EstimatedCLV"].mean()
        n_segments = len(report)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Total Revenue (£)", f"£{total_revenue:,.0f}")
        col3.metric("Avg Est. CLV", f"£{avg_clv:,.0f}")
        col4.metric("Active Segments", n_segments)

        st.markdown("---")

        # Revenue share by segment (donut chart)
        fig_donut = px.pie(
            report,
            values="TotalRevenue",
            names="Segment",
            title="Revenue Share by Customer Segment",
            hole=0.45,
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")

        # CLV by segment (bar chart)
        fig_clv = px.bar(
            report.sort_values("EstimatedCLV", ascending=True),
            x="EstimatedCLV",
            y="Segment",
            orientation="h",
            title="Estimated Customer Lifetime Value by Segment (£)",
            color="EstimatedCLV",
            color_continuous_scale="Viridis",
        )

        left, right = st.columns(2)
        left.plotly_chart(fig_donut, use_container_width=True)
        right.plotly_chart(fig_clv, use_container_width=True)

    # Revenue trend from historical data
    df_clean = load_clean_transactions()
    if df_clean is not None:
        monthly = (
            df_clean.groupby(df_clean["InvoiceDate"].dt.to_period("M"))["Revenue"]
            .sum()
            .reset_index()
        )
        monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)
        fig_trend = px.line(
            monthly, x="InvoiceDate", y="Revenue",
            title="Monthly Revenue Trend (Historical Data)",
            markers=True,
        )
        fig_trend.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_trend, use_container_width=True)


# ─── Tab 2: Customer Segments ────────────────────────────────────────────────

with tab2:
    st.subheader("Customer Segment Profiles")

    report = load_segment_report()
    if report is None:
        st.warning("Run `python train.py` first.")
    else:
        # Segment selector
        selected = st.selectbox("Select a segment to view details:", report["Segment"].tolist())
        row = report[report["Segment"] == selected].iloc[0]

        st.markdown(f"### {selected}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{int(row['CustomerCount']):,}")
        c2.metric("Revenue Share", f"{row['RevenueShare_pct']}%")
        c3.metric("Est. CLV", f"£{row['EstimatedCLV']:,.0f}")
        c4.metric("Avg Recency (days)", f"{row['AvgRecency']:.0f}")

        st.markdown("#### Recommended Action")
        priority_color = {
            "HIGH — Protect": "🟢",
            "URGENT — Win Back": "🔴",
            "MEDIUM — Grow": "🟡",
            "LOW — Nurture": "🔵",
            "VERY LOW — Evaluate": "⚫",
        }
        emoji = priority_color.get(row["Priority"], "⚪")
        st.info(f"{emoji} **{row['Priority']}**\n\n{row['RecommendedAction']}")

        st.markdown("---")
        st.markdown("#### All Segments — Side by Side")

        # Radar-style comparison (bar chart per metric)
        metrics_to_show = ["AvgRecency", "AvgFrequency", "AvgMonthlyRevenue", "EstimatedCLV"]
        melted = report.melt(id_vars="Segment", value_vars=metrics_to_show,
                             var_name="Metric", value_name="Value")
        fig_compare = px.bar(
            melted, x="Segment", y="Value", color="Metric",
            barmode="group",
            title="Segment Comparison Across Key Metrics",
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.dataframe(
            report[["Segment", "CustomerCount", "RevenueShare_pct",
                     "AvgRecency", "AvgFrequency", "EstimatedCLV", "Priority"]],
            use_container_width=True,
            hide_index=True,
        )


# ─── Tab 3: Live Feed ────────────────────────────────────────────────────────

with tab3:
    st.subheader("⚡ Live Transaction Feed")
    st.caption(
        "This tab shows simulated new transactions. "
        "Run `python src/simulate_live.py --days 1` in your terminal "
        "and the dashboard will update automatically on next refresh."
    )

    live = load_live_feed()
    if live is None:
        st.warning("No live data yet. Run `python train.py` to seed initial data.")
    else:
        last_date = live["InvoiceDate"].max()
        today_data = live[live["InvoiceDate"].dt.date == last_date.date()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions Today", f"{len(today_data):,}")
        col2.metric("Revenue Today (£)", f"£{today_data['Revenue'].sum():,.2f}")
        col3.metric("Last Update", str(last_date.strftime("%Y-%m-%d %H:%M")))

        # Daily revenue from live data
        daily = (
            live.groupby(live["InvoiceDate"].dt.date)["Revenue"]
            .sum()
            .reset_index()
        )
        daily.columns = ["Date", "Revenue"]
        fig_live = px.bar(
            daily.tail(30), x="Date", y="Revenue",
            title="Daily Revenue — Last 30 Days (Live Feed)",
            color="Revenue", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_live, use_container_width=True)

        st.markdown("#### Recent Transactions")
        st.dataframe(
            live.sort_values("InvoiceDate", ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )

# ── Auto-refresh loop ─────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.cache_data.clear()
    st.rerun()
