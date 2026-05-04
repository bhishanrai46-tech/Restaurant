import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Seralung Optimiz", layout="wide")

st.title("🍽️ Seralung Optimiz")
st.caption("Simple Menu Profit & Pricing Optimizer")

# -------------------------
# DEFAULT DATA
# -------------------------
df = pd.DataFrame({
    "Item": ["Burger", "Pasta", "Coffee", "Salad"],
    "Price": [12, 15, 5, 10],
    "Cost": [5, 7, 1.5, 4],
    "Labour (hrs)": [0.2, 0.25, 0.05, 0.1],
    "Max Demand": [80, 60, 200, 100]
})

# Editable table
st.subheader("📋 Menu Data")
df = st.data_editor(df, use_container_width=True)

# -------------------------
# SETTINGS
# -------------------------
st.sidebar.header("⚙️ Settings")
labour = st.sidebar.number_input("Labour Hours", value=16.0)
budget = st.sidebar.number_input("Budget ($)", value=500.0)

# -------------------------
# OPTIMIZATION
# -------------------------
if st.button("🚀 Optimize Menu"):

    df["Profit"] = df["Price"] - df["Cost"]

    c = -df["Profit"].values

    A = [
        df["Labour (hrs)"].values,
        df["Cost"].values
    ]

    b = [labour, budget]

    bounds = [(0, x) for x in df["Max Demand"]]

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    if result.success:

        df["Qty"] = result.x
        df["Total Profit"] = df["Qty"] * df["Profit"]

        total_profit = df["Total Profit"].sum()

        # -------------------------
        # KPI
        # -------------------------
        st.subheader("📊 Summary")

        st.metric("💰 Total Profit", f"${total_profit:,.0f}")

        # -------------------------
        # CHARTS
        # -------------------------
        col1, col2 = st.columns(2)

        fig1 = px.bar(df, x="Item", y="Total Profit", title="Profit by Item")
        fig2 = px.pie(df, names="Item", values="Total Profit", title="Profit Share")

        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

        # -------------------------
        # SIMPLE PRICING (Sensitivity)
        # -------------------------
        st.subheader("💰 Pricing Suggestion")

        df["Suggested Price"] = df["Price"] * 1.1
        df["New Profit"] = (df["Suggested Price"] - df["Cost"]) * df["Qty"]

        best_item = df.loc[df["New Profit"].idxmax(), "Item"]

        st.success(f"👉 Try increasing price of **{best_item}**")

        # -------------------------
        # INSIGHTS
        # -------------------------
        st.subheader("🧠 Insights")

        low_items = df[df["Profit"] < df["Profit"].mean()]["Item"].tolist()

        if low_items:
            st.warning(f"Low margin items: {', '.join(low_items)}")

        st.write("Focus on high-profit items to maximize returns.")

        # -------------------------
        # OWNER REPORT (SELLING FEATURE)
        # -------------------------
        st.subheader("📄 Weekly Summary")

        worst_item = df.loc[df["Total Profit"].idxmin(), "Item"]

        st.info(f"""
        This week you can make **${total_profit:,.0f} profit**

        🔥 Best item: {best_item}  
        ⚠️ Worst item: {worst_item}  

        💡 Recommendation: Adjust pricing & focus on top items
        """)

        # Table
        st.dataframe(df, use_container_width=True)

    else:
        st.error("Optimization failed")
