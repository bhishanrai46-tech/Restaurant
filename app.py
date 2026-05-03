import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Seralung Optimiz", layout="wide")

# -------------------------
# CLEAN DARK UI (SAFE)
# -------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0D1117;
    color: #E6EDF3;
}
h1, h2, h3 {
    color: #E6EDF3;
}
.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🍽️ Seralung Optimiz")

uploaded_file = st.sidebar.file_uploader("Upload Menu CSV", type=["csv"])
labour_hours = st.sidebar.number_input("Labour Hours", value=16.0)

# -------------------------
# DEFAULT DATA
# -------------------------
default_data = pd.DataFrame({
    "Item": ["Coffee", "Burger", "Pasta"],
    "Price": [5, 12, 15],
    "Cost": [1.5, 5, 10],
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
        st.error("Invalid CSV format")
        st.stop()
else:
    df = default_data.copy()

# -------------------------
# VALIDATION
# -------------------------
required_cols = ["Item", "Price", "Cost", "Labour (hrs)", "Max Demand"]

if not all(col in df.columns for col in required_cols):
    st.error("CSV must contain: Item, Price, Cost, Labour (hrs), Max Demand")
    st.stop()

# -------------------------
# RUN BUTTON
# -------------------------
if st.sidebar.button("🚀 Run Analysis"):

    df["Profit"] = df["Price"] - df["Cost"]

    try:
        c = -df["Profit"].values
        A = [df["Labour (hrs)"].values]
        b = [labour_hours]
        bounds = [(0, x) for x in df["Max Demand"]]

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    except:
        st.error("Optimization error")
        st.stop()

    if not result.success:
        st.error("Optimization failed")
        st.stop()

    # -------------------------
    # RESULTS
    # -------------------------
    df["Optimal Qty"] = result.x
    df["Total Profit"] = df["Optimal Qty"] * df["Profit"]

    total_profit = df["Total Profit"].sum()

    # -------------------------
    # HEADER
    # -------------------------
    st.title("📊 Weekly Owner Report")

    st.metric("💰 Weekly Profit", f"${total_profit:,.0f}")

    # -------------------------
    # MENU ENGINEERING
    # -------------------------
    st.subheader("📊 Menu Performance")

    fig = px.bar(df, x="Item", y="Total Profit")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # INSIGHTS
    # -------------------------
    st.subheader("🧠 Recommendations")

    best_item = df.loc[df["Total Profit"].idxmax(), "Item"]
    worst_item = df.loc[df["Total Profit"].idxmin(), "Item"]

    st.success(f"⭐ Best item: {best_item} → promote or increase price")
    st.warning(f"⚠️ Weak item: {worst_item} → reprice or remove")

    # -------------------------
    # PRICING SUGGESTION
    # -------------------------
    st.subheader("💰 Pricing Suggestion")

    suggested_increase = round(df.loc[df["Total Profit"].idxmax(), "Price"] * 0.1, 2)

    st.info(f"Increase {best_item} price by ~${suggested_increase} to improve margins")

    # -------------------------
    # SIMPLE OWNER SUMMARY
    # -------------------------
    st.subheader("📄 Weekly Summary")

    summary = f"""
    This week you made ${total_profit:.0f} profit.

    Best item: {best_item}
    Weak item: {worst_item}

    Action:
    - Increase price of {best_item}
    - Review or remove {worst_item}
    """

    st.text(summary)

    st.download_button("📥 Download Report", summary, "weekly_report.txt")

else:
    st.info("👈 Upload your menu CSV and click 'Run Analysis'")
