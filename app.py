import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Restaurant Profit AI", layout="wide")

st.title("🍽️ Restaurant Profit AI")
st.caption("AI-powered menu optimization & profit dashboard")

# -------------------------
# DEFAULT DATA
# -------------------------
default_data = pd.DataFrame({
    "Item": ["Coffee", "Sandwich", "Burger"],
    "Price": [5.0, 8.0, 12.0],
    "Cost": [1.5, 3.0, 5.0],
    "Labour": [0.05, 0.1, 0.2],
    "Max Demand": [200, 80, 50]
})

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("⚙️ Controls")

labour_hours = st.sidebar.slider("Total Labour Hours", 1.0, 100.0, 16.0)

run = st.sidebar.button("🚀 Run Optimization")

# -------------------------
# DATA INPUT
# -------------------------
st.subheader("📦 Menu Input Data")

df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

# -------------------------
# OPTIMIZATION LOGIC
# -------------------------
if run:

    df["Profit"] = df["Price"] - df["Cost"]

    c = -df["Profit"].values
    A = [df["Labour"].values]
    b = [labour_hours]
    bounds = [(0, x) for x in df["Max Demand"]]

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    if result.success:

        df["Optimal Qty"] = result.x
        df["Revenue"] = df["Optimal Qty"] * df["Price"]
        df["Total Profit"] = df["Optimal Qty"] * df["Profit"]

        labour_used = (df["Labour"] * result.x).sum()
        total_profit = df["Total Profit"].sum()

        # -------------------------
        # KPI SECTION
        # -------------------------
        st.subheader("📊 Dashboard KPIs")

        c1, c2, c3 = st.columns(3)

        c1.metric("💰 Total Profit", f"${total_profit:.2f}")
        c2.metric("🍽️ Items Sold", f"{int(df['Optimal Qty'].sum())}")
        c3.metric("🕒 Labour Used", f"{labour_used:.2f} / {labour_hours}")

        # -------------------------
        # TABLE
        # -------------------------
        st.subheader("📦 Optimized Plan")
        st.dataframe(df[["Item", "Optimal Qty", "Revenue", "Total Profit"]], use_container_width=True)

        # -------------------------
        # CHARTS
        # -------------------------
        st.subheader("📊 Insights")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(df, x="Item", y="Total Profit", title="Profit per Item")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(df, x="Item", y="Optimal Qty", title="Optimal Quantity")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig = px.pie(df, names="Item", values="Revenue", title="Revenue Share")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.bar(df, x="Item", y="Labour", title="Labour Impact")
            st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # INSIGHTS
        # -------------------------
        st.subheader("🧠 AI Insights")

        best_item = df.loc[df["Profit"].idxmax(), "Item"]
        st.success(f"🔥 Best focus item: {best_item}")

        if labour_used > 0.9 * labour_hours:
            st.warning("⚠️ Labour is a bottleneck")
        else:
            st.info("✅ Labour capacity is sufficient")

    else:
        st.error("Optimization failed")