import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Seralung Optimiz",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# DARK SAAS UI THEME
# -------------------------
st.markdown("""
<style>

/* BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: #0B0F1A;
    color: #E5E7EB;
}

/* TEXT */
h1, h2, h3 {
    color: #E5E7EB;
}

p {
    color: #9CA3AF;
}

/* METRIC CARDS */
.stMetric {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
}

/* BUTTON */
.stButton > button {
    background-color: #1F2937;
    color: #E5E7EB;
    border-radius: 10px;
    border: 1px solid #374151;
    padding: 0.6em 1.2em;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #2563EB;
    color: white;
}

/* TABLE */
[data-testid="stDataFrame"] {
    background-color: #111827;
    border-radius: 10px;
}

/* ALERT BOXES */
.stAlert-success {
    background-color: #064E3B;
    color: #D1FAE5;
    border-radius: 10px;
}

.stAlert-warning {
    background-color: #78350F;
    color: #FEF3C7;
    border-radius: 10px;
}

.stAlert-error {
    background-color: #7F1D1D;
    color: #FECACA;
    border-radius: 10px;
}

/* MOBILE */
@media (max-width: 768px) {
    .block-container {
        padding-left: 10px;
        padding-right: 10px;
    }
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title("🍽️ Seralung Optimiz")
st.write("AI-powered restaurant profit & pricing optimization")

# -------------------------
# CSV UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

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
st.subheader("⚙️ Settings")

col1, col2 = st.columns(2)

with col1:
    labour_hours = st.number_input("Total Labour Hours", value=16.0)

with col2:
    period = st.selectbox("Planning Period", ["Daily", "Weekly"])

if period == "Weekly":
    labour_hours *= 7
    df["Max Demand"] *= 7

# -------------------------
# OPTIMIZATION BUTTON
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
        # KPI
        # -------------------------
        st.subheader("📊 Performance Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Profit", f"${total_profit:,.2f}")
        c2.metric("📈 Revenue", f"${total_revenue:,.2f}")
        c3.metric("💸 Cost", f"${total_cost:,.2f}")

        # -------------------------
        # CHARTS
        # -------------------------
        st.subheader("📊 Analytics")

        fig1 = px.bar(df, x="Item", y="Total Profit", title="Profit by Item")
        fig2 = px.pie(df, names="Item", values="Total Profit", title="Revenue Mix")
        fig3 = px.bar(df, x="Item", y=["Revenue", "Total Cost"], barmode="group")

        for fig in [fig1, fig2, fig3]:
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B0F1A",
                plot_bgcolor="#0B0F1A",
                font=dict(color="#E5E7EB")
            )

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

        st.plotly_chart(fig3, use_container_width=True)

        # -------------------------
        # SET PRICE TOOL
        # -------------------------
        st.subheader("💰 Set Price (AI Pricing Tool)")

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
            title="Pricing Sensitivity Curve"
        )

        fig4.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B0F1A",
            plot_bgcolor="#0B0F1A",
            font=dict(color="#E5E7EB")
        )

        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(scenario_df, use_container_width=True)

        best = scenario_df.loc[scenario_df["Total Profit"].idxmax()]
        st.success(f"💡 Best Pricing Strategy: {best['Price Change']} change maximizes profit")

        # -------------------------
        # INSIGHTS (AI STYLE)
        # -------------------------
        st.subheader("🧠 Insights & Recommendations")

        labour_used = df["Labour Used"].sum()

        if labour_used > 0.9 * labour_hours:
            st.warning("🔴 Labour is your main constraint. Increasing capacity will increase profit.")

        best_item = df.loc[df["Total Profit"].idxmax(), "Item"]
        st.success(f"⭐ Focus on {best_item} — highest profit contributor")

        low_items = df[df["Profit"] < df["Profit"].mean()]["Item"].tolist()
        if low_items:
            st.error(f"⚠️ Low margin items: {', '.join(low_items)}")

        # -------------------------
        # DOWNLOAD
        # -------------------------
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Report", csv, "seralung_optimiz.csv", "text/csv")

    else:
        st.error("Optimization failed.")
