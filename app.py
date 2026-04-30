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
# CUSTOM UI (Fix buttons + clean SaaS look)
# -------------------------
st.markdown("""
    <style>
    .main {
        padding: 10px;
    }

    h1, h2, h3 {
        color: #111;
    }

    /* FIX BUTTON VISIBILITY */
    .stButton > button {
        background-color: #111827;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        border: none;
    }

    .stButton > button:hover {
        background-color: #2563eb;
        color: white;
    }

    /* MOBILE FRIENDLY */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 12px;
            padding-right: 12px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title("🍽️ Seralung Optimiz")
st.write("AI-powered restaurant profit + pricing optimization system")

# -------------------------
# UPLOAD CSV
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
# OPTIMIZATION
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
        st.subheader("📊 Performance Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Profit", f"${total_profit:,.2f}")
        c2.metric("📈 Revenue", f"${total_revenue:,.2f}")
        c3.metric("💸 Cost", f"${total_cost:,.2f}")

        # -------------------------
        # CHARTS
        # -------------------------
        st.subheader("📊 Analytics Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(df, x="Item", y="Total Profit", title="Profit by Item")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.pie(df, names="Item", values="Total Profit", title="Profit Share")
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(df, x="Item", y=["Revenue", "Total Cost"], barmode="group",
                      title="Revenue vs Cost")
        st.plotly_chart(fig3, use_container_width=True)

        # -------------------------
        # SET PRICE (SENSITIVITY ANALYSIS)
        # -------------------------
        st.subheader("💰 Set Price (Pricing Decision Tool)")
        st.write("Test how pricing changes affect profit and choose the best strategy.")

        price_changes = [-0.2, -0.1, 0, 0.1, 0.2]

        results = []

        for change in price_changes:
            temp = df.copy()

            temp["Test Price"] = temp["Price"] * (1 + change)
            temp["Test Profit"] = temp["Test Price"] - temp["Cost"]
            temp["Scenario Profit"] = temp["Test Profit"] * temp["Optimal Qty"]

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

        st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(scenario_df, use_container_width=True)

        best = scenario_df.loc[scenario_df["Total Profit"].idxmax()]

        st.success(f"💡 Recommended Strategy: {best['Price Change']} price change gives highest profit")

        # -------------------------
        # OPTIMIZED TABLE
        # -------------------------
        st.subheader("📦 Optimized Plan")
        st.dataframe(df, use_container_width=True)

        # -------------------------
        # INSIGHTS
        # -------------------------
        st.subheader("🧠 Insights")

        labour_used = df["Labour Used"].sum()

        insights = []

        if labour_used > 0.9 * labour_hours:
            insights.append("⚠️ Labour is your main constraint limiting profit.")
        else:
            insights.append("✅ Labour capacity is sufficient.")

        best_item = df.loc[df["Total Profit"].idxmax(), "Item"]
        insights.append(f"🔥 Best item: {best_item}")

        low_items = df[df["Profit"] < df["Profit"].mean()]["Item"].tolist()
        if low_items:
            insights.append(f"📉 Improve pricing or reduce focus on: {', '.join(low_items)}")

        for i in insights:
            st.write(i)

        # -------------------------
        # DOWNLOAD
        # -------------------------
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Report", csv, "seralung_optimiz.csv", "text/csv")

    else:
        st.error("Optimization failed. Please check inputs.")
