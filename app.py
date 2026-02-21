import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AI-Powered Mobile Sales Dashboard",
    page_icon="📱",
    layout="wide",
)

QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]

# =============================================================================
# LOAD DATA
# =============================================================================
CSV_PATH = "Expanded_Dataset.csv"

if not os.path.exists(CSV_PATH):
    st.error("Expanded_Dataset.csv not found.")
    st.stop()

df = pd.read_csv(CSV_PATH)

# Clean columns & types
df.columns = df.columns.str.strip()
df["Quarter"] = df["Quarter"].astype(str).str.upper().str.strip()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# =============================================================================
# SIDEBAR FILTERS (AUTO)
# =============================================================================
st.sidebar.title("🔎 Filters")

# Year options auto from dataset
year_options = ["All"] + sorted(df["Year"].dropna().astype(int).unique().tolist())
selected_year = st.sidebar.selectbox("Select Year", year_options)

# Quarter (allow All so user can see all quarters data too)
selected_quarter = st.sidebar.selectbox("Select Quarter", ["All"] + QUARTER_ORDER)

product_options = ["All"] + sorted(df["Product Model"].dropna().astype(str).unique().tolist())
selected_product = st.sidebar.selectbox("Select Product Model", product_options)

region_options = ["All"] + sorted(df["Region"].dropna().astype(str).unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", region_options)

# =============================================================================
# APPLY FILTERS
# =============================================================================
filtered = df.copy()

if selected_year != "All":
    filtered = filtered[filtered["Year"].astype(int) == int(selected_year)]

if selected_quarter != "All":
    filtered = filtered[filtered["Quarter"].astype(str) == selected_quarter]

if selected_product != "All":
    filtered = filtered[filtered["Product Model"].astype(str) == selected_product]

if selected_region != "All":
    filtered = filtered[filtered["Region"].astype(str) == selected_region]

# make quarter categorical AFTER filtering (safe)
filtered["Quarter"] = pd.Categorical(filtered["Quarter"], categories=QUARTER_ORDER, ordered=True)

# =============================================================================
# HEADER
# =============================================================================
st.title("📱 AI-Powered Mobile Sales Dashboard")
st.caption("Auto-filtered metrics, trends, 5G insights & next-quarter AI forecast")

# =============================================================================
# KEY METRICS
# =============================================================================
st.subheader("Key Metrics")

c1, c2, c3, c4 = st.columns(4)

total_units = int(filtered["Units Sold"].sum()) if not filtered.empty else 0
total_rev = float(filtered["Revenue ($)"].sum()) if not filtered.empty else 0.0
avg_ms = float(filtered["Market Share (%)"].mean()) if not filtered.empty else 0.0
avg_speed = float(filtered["Avg 5G Speed (Mbps)"].mean()) if not filtered.empty else 0.0

c1.metric("📦 Total Units Sold", f"{total_units:,}")
c2.metric("💰 Total Revenue", f"${total_rev:,.2f}")
c3.metric("📈 Avg Market Share", f"{avg_ms:.2f}%")
c4.metric("⚡ Avg 5G Speed", f"{avg_speed:.1f} Mbps")

# =============================================================================
# HISTORICAL YEARLY ANALYSIS (uses FILTERED data)
# =============================================================================
st.subheader("📅 Historical Performance (Year-wise)")

if not filtered.empty and "Year" in filtered.columns:
    yearly_trend = (
        filtered.groupby("Year", as_index=False)
        .agg({"Units Sold": "sum", "Revenue ($)": "sum"})
        .sort_values("Year")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Units Sold by Year")
        st.line_chart(yearly_trend.set_index("Year")["Units Sold"], use_container_width=True)

    with col2:
        st.caption("Revenue by Year")
        st.bar_chart(yearly_trend.set_index("Year")["Revenue ($)"], use_container_width=True)

    st.markdown("### 📊 Yearly Summary Table")
    st.dataframe(yearly_trend, use_container_width=True)
else:
    st.info("No data available for the selected filters.")

# =============================================================================
# NEXT 3 QUARTERS LOGIC (based on selected quarter dropdown)
# =============================================================================
st.subheader("📊 Sales Analytics (Next 3 Quarters Only)")

# If user selected "All", we cannot decide next-three, so show normal quarter trend
if selected_quarter == "All":
    trend = (
        filtered.groupby("Quarter", as_index=False)
        .agg({"Units Sold": "sum", "Revenue ($)": "sum"})
        .sort_values("Quarter")
    )

    left, right = st.columns(2)

    with left:
        st.caption("Units Sold Trend (All Quarters)")
        st.line_chart(trend.set_index("Quarter")["Units Sold"], use_container_width=True)

    with right:
        st.caption("Revenue Trend (All Quarters)")
        st.bar_chart(trend.set_index("Quarter")["Revenue ($)"], use_container_width=True)

else:
    idx = QUARTER_ORDER.index(selected_quarter)
    next_three = [
        QUARTER_ORDER[(idx + 1) % 4],
        QUARTER_ORDER[(idx + 2) % 4],
        QUARTER_ORDER[(idx + 3) % 4],
    ]

    trend = (
        filtered.groupby("Quarter", as_index=False)
        .agg({"Units Sold": "sum", "Revenue ($)": "sum"})
    )

    trend = trend[trend["Quarter"].astype(str).isin(next_three)]
    trend["Quarter"] = pd.Categorical(trend["Quarter"], categories=next_three, ordered=True)
    trend = trend.sort_values("Quarter")

    left, right = st.columns(2)

    with left:
        st.caption(f"Units Sold Trend ({', '.join(next_three)})")
        if not trend.empty:
            st.line_chart(trend.set_index("Quarter")["Units Sold"], use_container_width=True)
        else:
            st.info("No data available for the next 3 quarters with current filters.")

    with right:
        st.caption(f"Revenue Trend ({', '.join(next_three)})")
        if not trend.empty:
            st.bar_chart(trend.set_index("Quarter")["Revenue ($)"], use_container_width=True)
        else:
            st.info("No data available for the next 3 quarters with current filters.")

# =============================================================================
# AI FORECAST (Next 3 quarters prediction)
# =============================================================================
st.subheader("📈 AI Forecast")

forecast_df = (
    filtered.groupby("Quarter", as_index=False)
    .agg({"Units Sold": "sum"})
)

forecast_df["Quarter_Num"] = (
    forecast_df["Quarter"].astype(str).str.replace("Q", "", regex=False)
)
forecast_df["Quarter_Num"] = pd.to_numeric(forecast_df["Quarter_Num"], errors="coerce")
forecast_df = forecast_df.dropna(subset=["Quarter_Num"]).sort_values("Quarter_Num")

if len(forecast_df) >= 2 and selected_quarter != "All":
    X = forecast_df[["Quarter_Num"]]
    y = forecast_df["Units Sold"].values

    model = LinearRegression()
    model.fit(X, y)

    idx = QUARTER_ORDER.index(selected_quarter)
    next_three = [
        QUARTER_ORDER[(idx + 1) % 4],
        QUARTER_ORDER[(idx + 2) % 4],
        QUARTER_ORDER[(idx + 3) % 4],
    ]
    next_three_nums = np.array([int(q.replace("Q", "")) for q in next_three]).reshape(-1, 1)

    predictions = model.predict(next_three_nums)

    st.metric("Predicted Units Sold (Next Quarter)", f"{int(predictions[0]):,}")

    forecast_plot = pd.DataFrame({"Quarter": next_three, "Units Sold": predictions})
    forecast_plot["Quarter"] = pd.Categorical(forecast_plot["Quarter"], categories=next_three, ordered=True)
    forecast_plot = forecast_plot.sort_values("Quarter")

    st.line_chart(forecast_plot.set_index("Quarter")["Units Sold"], use_container_width=True)

elif selected_quarter == "All":
    st.info("Select a specific quarter (Q1–Q4) to see next-3-quarter forecast.")
else:
    st.info("Not enough data to forecast (need at least 2 quarters after filtering).")

# =============================================================================
# 5G INSIGHTS
# =============================================================================
st.subheader("📡 5G Insights")

i1, i2, i3 = st.columns(3)

avg_cov = float(filtered["Regional 5G Coverage (%)"].mean()) if not filtered.empty else 0.0
avg_pref = float(filtered["Preference for 5G (%)"].mean()) if not filtered.empty else 0.0

i1.metric("Avg Regional 5G Coverage", f"{avg_cov:.2f}%")
i2.metric("Avg Preference for 5G", f"{avg_pref:.2f}%")
i3.write("**5G Capability Split**")
if not filtered.empty:
    i3.dataframe(filtered["5G Capability"].value_counts().to_frame("Count"), use_container_width=True, height=180)
else:
    i3.write("No data")

# =============================================================================
# FILTERED TABLE (THIS WILL SHOW ONLY FILTERED ROWS ✅)
# =============================================================================
st.subheader("Filtered Data (Applied Filters)")
st.dataframe(filtered, use_container_width=True, height=420)

st.sidebar.markdown("---")
st.sidebar.caption("Mobile Sales & 5G Analytics • Streamlit Dashboard")