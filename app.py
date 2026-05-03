import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Seralung Optimiz", layout="wide")

# -------------------------
# PREMIUM UI (CLAUDE STYLE)
# -------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0D1117;
    color: #E6EDF3;
}

h1, h2, h3 {
    color: #E6EDF3;
    font-weight: 600;
}

/* CARD STYLE */
.card {
    background-color: #161B22;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 12px;
    border: 1px solid #30363D;
}

/* BUTTON */
.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1.2em;
}

.stButton > button:hover {
    background-color: #2EA043;
}

/* INSIGHT BOXES */
.success-box {
    background-color: #132A1A;
    padding: 12px;
    border-radius: 10px;
}

.warning-box {
    background-color: #2A1A12;
    padding: 12px;
    border-radius: 10px;
}

.danger-box {
    background-color: #2A1212;
    padding: 12px;
    border-radius: 10px;
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
# DATA
# -------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        "Item": ["Coffee", "Burger", "Pasta"],
        "Price": [5, 12, 15],
        "Cost": [1.5, 5, 10],
        "Labour (hrs)": [0.05, 0.2, 0.25],
        "Max Demand": [200, 80, 40]
    })

# -------------------------
# RUN
# -------------------------
if st.sidebar.button("🚀 Run Analysis"):

    df["Profit"] = df["Price"] - df["Cost"]

    c = -df["Profit"].values
    A = [df["Labour (hrs)"].values]
    b = [labour_hours]
    bounds = [(0, x) for x in df["Max Demand"]]

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    if result.success:

        df["Optimal Qty"] = result.x
        df["Total Profit"] = df["Optimal Qty"] * df["Profit"]

        total_profit = df["Total Profit"].sum()

        # -------------------------
        # HEADER
        # -------------------------
        st.title("📊 Weekly Owner Report")

        st.markdown(f"""
        <div class="card">
        <h2>💰 This Week</h2>
        <h1>${total_profit:,.0f} Profit</h1>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # MENU ENGINEERING
        # -------------------------
        st.subheader("📊 Menu Performance")

        fig = px.bar(df, x="Item", y="Total Profit")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # AI INSIGHTS (KEY FEATURE)
        # -------------------------
        st.subheader("🧠 AI Recommendations")

        best_item = df.loc[df["Total Profit"].idxmax(), "Item"]
        worst_item = df.loc[df["Total Profit"].idxmin(), "Item"]

        # BEST
        st.markdown(f"""
        <div class="success-box">
        ⭐ Best performer: <b>{best_item}</b><br>
        Increase promotion or highlight this item.
        </div>
        """, unsafe_allow_html=True)

        # WORST
        st.markdown(f"""
        <div class="danger-box">
        ⚠️ Worst item: <b>{worst_item}</b><br>
        Consider removing or repricing.
        </div>
        """, unsafe_allow_html=True)

        # PRICING SUGGESTION
        st.markdown(f"""
        <div class="warning-box">
        💡 Pricing opportunity:<br>
        Increase <b>{best_item}</b> price by 5–10% to boost profit.
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # SIMPLE OWNER SUMMARY
        # -------------------------
        st.subheader("📄 Simple Summary")

        summary = f"""
        This week you made ${total_profit:.0f} profit.

        Your best item is {best_item}.
        Your worst item is {worst_item}.

        Recommended action:
        - Increase price of {best_item}
        - Review or remove {worst_item}
        """

        st.text(summary)

        st.download_button("📥 Download Report", summary, "weekly_report.txt")

    else:
        st.error("Optimization failed")

else:
    st.info("Upload menu and click Run Analysis")
