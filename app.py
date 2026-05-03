import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Seralung Optimiz", layout="wide")

# -------------------------
# SIMPLE DARK STYLE (SAFE)
# -------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0D1117;
    color: #E6EDF3;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title("🍽️ Seralung Optimiz")
st.write("Menu Engineering & Weekly Profit Report")

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("Inputs")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
labour_hours = st.sidebar.number_input("Labour Hours", value=16.0)

# -------------------------
# DEFAULT DATA
# -------------------------
default_data = pd.DataFrame({
    "Item": ["Coffee", "Burger", "Pasta"],
    "Price": [5.0, 12.0, 15.0],
    "Cost": [1.5, 5.0, 10.0],
    "Labour (hrs)": [0.05, 0.2, 0.25],
    "Max Demand": [200, 80, 40]
})

# -------------------------
# LOAD DATA
# -------------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except:
        st.error("❌ Failed to read CSV file")
        st.stop()
else:
    df = default_data.copy()

# -------------------------
# VALIDATION
# -------------------------
required_cols = ["Item", "Price", "Cost", "Labour (hrs)", "Max Demand"]

if not all(col in df.columns for col in required_cols):
    st.error("CSV must include: Item, Price, Cost, Labour (hrs), Max Demand")
    st.stop()

# -------------------------
# SHOW DATA
# -------------------------
st.subheader("📋 Menu Data")
st.dataframe(df, use_container_width=True)

# -------------------------
# RUN ANALYSIS BUTTON
# -------------------------
if st.button("🚀 Run Analysis"):

    # Profit calculation
    df["Profit"] = df["Price"] - df["Cost"]

    # Optimization setup
    try:
        c = -df["Profit"].values
        A = [df["Labour (hrs)"].values]
        b = [labour_hours]
        bounds = [(0, x) for x in df["Max Demand"]]

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    except Exception as e:
        st.error(f"Optimization error: {e}")
        st.stop()

    if not result.success:
        st.error("Optimization failed")
        st.stop()

    # Results
    df["Optimal Qty"] = result.x
    df["Total Profit"] = df["Optimal Qty"] * df["Profit"]

    total_profit = df["Total Profit"].sum()

    # -------------------------
    # WEEKLY REPORT
    # -------------------------
    st.subheader("📊 Weekly Owner Report")

    st.metric("💰 Total Profit", f"${total_profit:,.2f}")

    # Chart
    fig = px.bar(df, x="Item", y="Total Profit", title="Profit by Item")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # INSIGHTS
    # -------------------------
    st.subheader("🧠 Key Insights")

    best_item = df.loc[df["Total Profit"].idxmax(), "Item"]
    worst_item = df.loc[df["Total Profit"].idxmin(), "Item"]

    st.success(f"⭐ Best Item: {best_item}")
    st.warning(f"⚠️ Weak Item: {worst_item}")

    # -------------------------
    # PRICING SUGGESTION
    # -------------------------
    st.subheader("💰 Pricing Suggestion")

    best_price = df.loc[df["Total Profit"].idxmax(), "Price"]
    suggested_increase = round(best_price * 0.1, 2)

    st.info(f"Increase {best_item} price by ~${suggested_increase}")

    # -------------------------
    # SIMPLE SUMMARY (SELLABLE PART)
    # -------------------------
    st.subheader("📄 Simple Owner Summary")

    summary = f"""
This week you made ${total_profit:.2f} profit.

Best item: {best_item}
Weak item: {worst_item}

Recommended actions:
- Increase price of {best_item}
- Review or remove {worst_item}
"""

    st.text(summary)

    # Download
    st.download_button("📥 Download Report", summary, "weekly_report.txt")

else:
    st.info("👆 Click 'Run Analysis' to generate report")
