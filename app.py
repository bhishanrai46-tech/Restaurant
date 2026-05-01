import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Seralung Optimiz",
    layout="wide"
)

# -------------------------
# DARK UI
# -------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0B0F1A;
    color: #E5E7EB;
}

h1, h2, h3 {
    color: #E5E7EB;
}

.stMetric {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
}

.stButton > button {
    background-color: #1F2937;
    color: white;
    border-radius: 10px;
    border: 1px solid #374151;
}

.stButton > button:hover {
    background-color: #2563EB;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
st.sidebar.title("🍽️ Seralung Optimiz")

page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "💰 Set Price",
    "🧠 Insights"
])

# -------------------------
# DATA
# -------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

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
# SETTINGS
# -------------------------
labour_hours = st.sidebar.number_input("Labour Hours", value=16.0)

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
    df["Revenue"] = df["Optimal Qty"] * df["Price"]
    df["Total Cost"] = df["Optimal Qty"] * df["Cost"]
    df["Total Profit"] = df["Optimal Qty"] * df["Profit"]
    df["Labour Used"] = df["Optimal Qty"] * df["Labour (hrs)"]

    total_profit = df["Total Profit"].sum()
    total_revenue = df["Revenue"].sum()
    total_cost = df["Total Cost"].sum()
    labour_used = df["Labour Used"].sum()

    # -------------------------
    # DASHBOARD PAGE
    # -------------------------
    if page == "📊 Dashboard":

        st.title("📊 Dashboard")

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Profit", f"${total_profit:,.2f}")
        c2.metric("📈 Revenue", f"${total_revenue:,.2f}")
        c3.metric("💸 Cost", f"${total_cost:,.2f}")

        fig1 = px.bar(df, x="Item", y="Total Profit", title="Profit by Item")
        fig2 = px.pie(df, names="Item", values="Total Profit", title="Revenue Mix")
        fig3 = px.bar(df, x="Item", y=["Revenue", "Total Cost"], barmode="group")

        for fig in [fig1, fig2, fig3]:
            fig.update_layout(template="plotly_dark")

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

        st.plotly_chart(fig3, use_container_width=True)

    # -------------------------
    # SET PRICE PAGE
    # -------------------------
    elif page == "💰 Set Price":

        st.title("💰 Set Price")

        price_changes = [-0.2, -0.1, 0, 0.1, 0.2]

        results = []

        for change in price_changes:
            temp = df.copy()
            temp["Adj Price"] = temp["Price"] * (1 + change)
            temp["Adj Profit"] = temp["Adj Price"] - temp["Cost"]
            temp["Scenario Profit"] = temp["Adj Profit"] * temp["Optimal Qty"]

            results.append({
                "Price Change": f"{int(change*100)}%",
                "Total Profit": temp["Scenario Profit"].sum()
            })

        scenario_df = pd.DataFrame(results)

        fig4 = px.line(
            scenario_df,
            x="Price Change",
            y="Total Profit",
            markers=True,
            title="Profit Sensitivity Curve"
        )

        fig4.update_layout(template="plotly_dark")

        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(scenario_df)

        best = scenario_df.loc[scenario_df["Total Profit"].idxmax()]

        st.success(f"💡 Best pricing strategy: {best['Price Change']}")

    # -------------------------
    # INSIGHTS PAGE
    # -------------------------
    elif page == "🧠 Insights":

        st.title("🧠 Insights & Recommendations")

        best_item = df.loc[df["Total Profit"].idxmax(), "Item"]
        st.success(f"⭐ Focus on {best_item}")

        if labour_used > 0.9 * labour_hours:
            st.warning("🔴 Labour is limiting your profit")

        low_items = df[df["Profit"] < df["Profit"].mean()]["Item"].tolist()
        if low_items:
            st.error(f"⚠️ Low margin items: {', '.join(low_items)}")

else:
    st.error("Optimization failed")
