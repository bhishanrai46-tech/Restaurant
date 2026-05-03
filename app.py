import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog
import datetime

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Seralung Optimiz", layout="wide")

# -------------------------
# DARK UI
# -------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0B0F1A;
    color: #E5E7EB;
}
.stMetric {
    background-color: #111827;
    padding: 12px;
    border-radius: 10px;
}
.stButton > button {
    background-color: #1F2937;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🍽️ Seralung Optimiz")

page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "💰 Set Price",
    "🧠 Insights",
    "📄 Reports"
])

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
labour_hours = st.sidebar.number_input("Labour Hours", value=16.0)

# -------------------------
# DATA
# -------------------------
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

# -------------------------
# OPTIMIZATION
# -------------------------
df["Profit"] = df["Price"] - df["Cost"]

c = -df["Profit"].values
A = [df["Labour (hrs)"].values]
b = [labour_hours]
bounds = [(0, x) for x in df["Max Demand"]]

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

if result.success:

    df["Optimal Qty"] = result.x
    df["Total Profit"] = df["Optimal Qty"] * df["Profit"]
    df["Revenue"] = df["Optimal Qty"] * df["Price"]
    df["Total Cost"] = df["Optimal Qty"] * df["Cost"]
    df["Labour Used"] = df["Optimal Qty"] * df["Labour (hrs)"]

    total_profit = df["Total Profit"].sum()
    total_revenue = df["Revenue"].sum()

    # -------------------------
    # STORE HISTORY (SESSION)
    # -------------------------
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "date": datetime.datetime.now(),
        "profit": total_profit
    })

    history_df = pd.DataFrame(st.session_state.history)

    # -------------------------
    # DASHBOARD
    # -------------------------
    if page == "📊 Dashboard":

        st.title("📊 Dashboard")

        c1, c2 = st.columns(2)
        c1.metric("💰 Profit", f"${total_profit:,.2f}")
        c2.metric("📈 Revenue", f"${total_revenue:,.2f}")

        fig = px.line(history_df, x="date", y="profit", title="Profit Trend")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # PRICING TOOL
    # -------------------------
    elif page == "💰 Set Price":

        st.title("💰 Pricing Strategy")

        price_changes = [-0.2, -0.1, 0, 0.1, 0.2]
        results = []

        for change in price_changes:
            temp = df.copy()
            temp["Adj Price"] = temp["Price"] * (1 + change)
            temp["Adj Profit"] = temp["Adj Price"] - temp["Cost"]
            temp["Scenario Profit"] = temp["Adj Profit"] * temp["Optimal Qty"]

            results.append({
                "Change": f"{int(change*100)}%",
                "Profit": temp["Scenario Profit"].sum()
            })

        scenario_df = pd.DataFrame(results)

        fig = px.line(scenario_df, x="Change", y="Profit", markers=True)
        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

        best = scenario_df.loc[scenario_df["Profit"].idxmax()]

        st.success(f"💡 Best move: Adjust price by {best['Change']}")

    # -------------------------
    # INSIGHTS (ACTIONABLE)
    # -------------------------
    elif page == "🧠 Insights":

        st.title("🧠 AI Recommendations")

        best_item = df.loc[df["Total Profit"].idxmax(), "Item"]

        st.success(f"Increase focus on **{best_item}** — highest profit contributor")

        # actionable recommendation
        st.info(f"👉 Increase {best_item} price by 5–10% to boost profit")

        low_items = df[df["Profit"] < df["Profit"].mean()]["Item"].tolist()

        if low_items:
            st.warning(f"Reduce or reprice: {', '.join(low_items)}")

        if df["Labour Used"].sum() > 0.9 * labour_hours:
            st.error("Labour is limiting growth — consider hiring or simplifying menu")

    # -------------------------
    # REPORTS (DOWNLOADABLE)
    # -------------------------
    elif page == "📄 Reports":

        st.title("📄 Business Report")

        report = f"""
        Seralung Optimiz Report

        Total Profit: ${total_profit:.2f}
        Revenue: ${total_revenue:.2f}

        Top Item: {df.loc[df['Total Profit'].idxmax(), 'Item']}

        Recommendation:
        Focus on high-margin items and optimize pricing.
        """

        st.text(report)

        st.download_button("📥 Download Report", report, "report.txt")

else:
    st.error("Optimization failed")
