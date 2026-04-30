import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# PAGE CONFIG (clean UI)
# -------------------------
st.set_page_config(page_title="Restaurant Profit Dashboard", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #0f1117;}
        h1, h2, h3 {color: #ffffff;}
        .stMetric {background-color: #1c1f26; padding: 10px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🍽️ Restaurant Profit Optimization Dashboard")
st.write("Upload your data or use sample data to optimize profits and analyze performance.")

# -------------------------
# UPLOAD CSV
# -------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        "Item": ["Coffee", "Sandwich", "Burger"],
        "Price": [5.0, 8.0, 12.0],
        "Cost": [1.5, 3.0, 5.0],
        "Labour (hrs)": [0.05, 0.1, 0.2],
        "Max Demand": [200, 80, 50]
    })

df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# -------------------------
# SETTINGS
# -------------------------
st.subheader("⚙️ Optimization Settings")

col1, col2 = st.columns(2)

with col1:
    labour_hours = st.number_input("Total Labour Hours", value=16.0)

with col2:
    period = st.selectbox("Planning Period", ["Daily", "Weekly"])

if period == "Weekly":
    labour_hours *= 7
    df["Max Demand"] *= 7

# -------------------------
# CALCULATE
# -------------------------
if st.button("🚀 Run Optimization"):

    df["Profit"] = df["Price"] - df["Cost"]

    c = -df["Profit"].values
    A = [df["Labour (hrs)"].values]
    b = [labour_hours]
    bounds = [(0, x) for x in df["Max Demand"]]

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    if result.success:

        df["Optimal Qty"] = result.x
        df["Revenue"] = df["Optimal Qty"] * df["Price"]
        df["Total Cost"] = df["Optimal Qty"] * df["Cost"]
        df["Total Profit"] = df["Optimal Qty"] * df["Profit"]
        df["Labour Used"] = df["Optimal Qty"] * df["Labour (hrs)"]

        total_profit = df["Total Profit"].sum()
        total_revenue = df["Revenue"].sum()
        total_cost = df["Total Cost"].sum()

        # -------------------------
        # KPI METRICS
        # -------------------------
        st.subheader("📊 Key Performance Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Total Profit", f"${total_profit:,.2f}")
        c2.metric("📈 Revenue", f"${total_revenue:,.2f}")
        c3.metric("💸 Cost", f"${total_cost:,.2f}")

        # -------------------------
        # CHARTS SECTION
        # -------------------------
        st.subheader("📊 Analytics Dashboard")

        col1, col2 = st.columns(2)

        # Bar chart: Profit by item
        with col1:
            fig1 = px.bar(
                df,
                x="Item",
                y="Total Profit",
                text="Total Profit",
                title="Profit by Item"
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Pie chart: profit share
        with col2:
            fig2 = px.pie(
                df,
                names="Item",
                values="Total Profit",
                title="Profit Contribution Share"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Revenue vs Cost
        fig3 = px.bar(
            df,
            x="Item",
            y=["Revenue", "Total Cost"],
            barmode="group",
            title="Revenue vs Cost Analysis"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # -------------------------
        # OPTIMIZED TABLE
        # -------------------------
        st.subheader("📦 Optimized Plan")
        st.dataframe(df, use_container_width=True)

        # -------------------------
        # INSIGHTS SECTION (SMART)
        # -------------------------
        st.subheader("🧠 Business Insights")

        insights = []

        labour_used = df["Labour Used"].sum()

        if labour_used > 0.9 * labour_hours:
            insights.append("⚠️ Labour capacity is nearly fully utilized — this is limiting growth.")
        else:
            insights.append("✅ Labour capacity is sufficient for current demand.")

        best_item = df.loc[df["Total Profit"].idxmax(), "Item"]
        insights.append(f"🔥 Highest profit item: {best_item}")

        low_profit = df[df["Profit"] < df["Profit"].mean()]["Item"].tolist()
        if low_profit:
            insights.append(f"📉 Improve or reduce focus on: {', '.join(low_profit)}")

        profit_margin = (total_profit / total_revenue) * 100 if total_revenue else 0
        insights.append(f"📊 Overall profit margin: {profit_margin:.2f}%")

        for i in insights:
            st.write(i)

        st.success("Optimization completed successfully 🎯")

        # Download
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Report", csv, "restaurant_report.csv", "text/csv")

    else:
        st.error("Optimization failed. Please check your input data.")
