import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Seralung Optimiz", layout="wide")

st.title("🍽️ Seralung Optimiz")
st.write("Menu Engineering with Budget & Labour Constraints")

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("Constraints")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

labour_hours = st.sidebar.number_input("Available Labour Hours", value=16.0)
budget = st.sidebar.number_input("Total Budget ($)", value=500.0)

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
        st.error("Invalid CSV")
        st.stop()
else:
    df = default_data.copy()

# -------------------------
# VALIDATION
# -------------------------
required_cols = ["Item", "Price", "Cost", "Labour (hrs)", "Max Demand"]

if not all(col in df.columns for col in required_cols):
    st.error("CSV must include required columns")
    st.stop()

# -------------------------
# DISPLAY DATA
# -------------------------
st.subheader("📋 Menu Data")
st.dataframe(df, use_container_width=True)

# -------------------------
# RUN OPTIMIZATION
# -------------------------
if st.button("🚀 Run Optimization"):

    df["Profit"] = df["Price"] - df["Cost"]

    try:
        c = -df["Profit"].values

        # CONSTRAINT MATRIX (IMPORTANT PART)
        A = [
            df["Labour (hrs)"].values,  # labour constraint
            df["Cost"].values           # cost constraint (NEW)
        ]

        b = [
            labour_hours,  # labour limit
            budget         # cost limit
        ]

        bounds = [(0, x) for x in df["Max Demand"]]

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    except Exception as e:
        st.error(f"Optimization error: {e}")
        st.stop()

    if not result.success:
        st.error("Optimization failed")
        st.stop()

    # -------------------------
    # RESULTS
    # -------------------------
    df["Optimal Qty"] = result.x
    df["Total Profit"] = df["Optimal Qty"] * df["Profit"]
    df["Total Cost"] = df["Optimal Qty"] * df["Cost"]
    df["Labour Used"] = df["Optimal Qty"] * df["Labour (hrs)"]

    total_profit = df["Total Profit"].sum()
    total_cost = df["Total Cost"].sum()
    labour_used = df["Labour Used"].sum()

    # -------------------------
    # KPI
    # -------------------------
    st.subheader("📊 Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Profit", f"${total_profit:,.2f}")
    c2.metric("💸 Cost Used", f"${total_cost:,.2f}")
    c3.metric("🕒 Labour Used", f"{labour_used:.2f} hrs")

    # -------------------------
    # CHART
    # -------------------------
    fig = px.bar(df, x="Item", y="Total Profit", title="Profit by Item")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # INSIGHTS
    # -------------------------
    st.subheader("🧠 Insights")

    best_item = df.loc[df["Total Profit"].idxmax(), "Item"]
    worst_item = df.loc[df["Total Profit"].idxmin(), "Item"]

    if labour_used > 0.9 * labour_hours:
        st.warning("⚠️ Labour is limiting your profit")

    if total_cost > 0.9 * budget:
        st.warning("⚠️ Budget constraint is tight")

    st.success(f"⭐ Best item: {best_item}")
    st.error(f"⚠️ Weak item: {worst_item}")

    # -------------------------
    # SUMMARY
    # -------------------------
    st.subheader("📄 Owner Summary")

    summary = f"""
    Profit: ${total_profit:.2f}
    Cost Used: ${total_cost:.2f} / ${budget}
    Labour Used: {labour_used:.2f} / {labour_hours}

    Best Item: {best_item}
    Weak Item: {worst_item}
    """

    st.text(summary)

    st.download_button("📥 Download Report", summary, "report.txt")

else:
    st.info("Click 'Run Optimization' to start")
