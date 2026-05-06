"""
MenuProfit Pro — Restaurant Pricing Tool
Simple. Powerful. Built for restaurant owners.

Run:
    pip install streamlit pandas numpy plotly scipy openpyxl
    streamlit run menuprofit_app.py
"""

import io
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize, differential_evolution, linprog

# ─────────────────────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MenuProfit Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Nunito:wght@400;600;700;800&display=swap');

*, html, body { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    font-size: 15px;
}

/* App background */
[data-testid="stAppViewContainer"] {
    background: #0e0e18;
    color: #e8eaf0;
}
[data-testid="stSidebar"] {
    background: #12121f !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Typography ── */
h1,h2,h3 { font-family:'Playfair Display',serif !important; }

/* ── Top banner ── */
.top-banner {
    background: linear-gradient(135deg,#1a1a30 0%,#0e0e18 100%);
    border-bottom: 1px solid rgba(255,200,80,0.18);
    padding: 18px 24px 14px;
    margin: -1rem -1rem 1.4rem;
}
.top-banner h1 { font-size:1.9rem; margin:0; color:#fff; }
.top-banner p  { margin:4px 0 0; color:#9499b5; font-size:0.85rem; }
.gold { color:#f0c060; }

/* ── Section header ── */
.sec-head {
    font-family:'Playfair Display',serif;
    font-size:1.35rem;
    color:#f0c060;
    border-bottom:1px solid rgba(240,192,96,.2);
    padding-bottom:6px;
    margin:1.8rem 0 .9rem;
}

/* ── KPI cards ── */
.kpi-row { display:flex; gap:12px; flex-wrap:wrap; margin:0 0 1rem; }
.kpi {
    background:linear-gradient(145deg,#181830,#12121f);
    border:1px solid rgba(240,192,96,.15);
    border-radius:14px;
    padding:16px 18px;
    flex:1 1 130px;
    min-width:110px;
    text-align:center;
}
.kpi-lbl { font-size:.72rem; color:#7a7fa0; text-transform:uppercase; letter-spacing:.07em; margin-bottom:5px; }
.kpi-val { font-size:1.6rem; font-weight:800; color:#f0c060; line-height:1.1; }
.kpi-sub { font-size:.72rem; color:#555c7a; margin-top:3px; }
.kpi-val.green { color:#4ade80; }
.kpi-val.red   { color:#f87171; }
.kpi-val.white { color:#e8eaf0; }

/* ── Alert cards ── */
.alert {
    border-radius:10px;
    padding:13px 16px;
    margin-bottom:10px;
    font-size:.88rem;
    line-height:1.6;
    border-left:4px solid #f0c060;
    background:rgba(240,192,96,.07);
}
.alert.red    { border-color:#f87171; background:rgba(248,113,113,.07); }
.alert.green  { border-color:#4ade80; background:rgba(74,222,128,.06); }
.alert.yellow { border-color:#fbbf24; background:rgba(251,191,36,.07); }

/* ── Tip box ── */
.tip {
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.07);
    border-radius:10px;
    padding:11px 15px;
    font-size:.82rem;
    color:#7a7fa0;
    margin:.5rem 0 1rem;
}

/* ── Buttons ── */
.stButton > button {
    background:linear-gradient(135deg,#c97d10,#f0c060) !important;
    color:#0e0e18 !important;
    font-weight:800 !important;
    border:none !important;
    border-radius:12px !important;
    padding:11px 22px !important;
    font-size:.95rem !important;
    width:100% !important;
    letter-spacing:.01em !important;
    transition: opacity .2s;
}
.stButton > button:hover { opacity:.87 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(0,0,0,.35);
    border-radius:12px;
    padding:4px;
    flex-wrap:wrap;
    gap:3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:9px;
    color:#7a7fa0 !important;
    font-size:.82rem !important;
    padding:7px 12px !important;
    font-weight:700 !important;
}
.stTabs [aria-selected="true"] {
    background:rgba(240,192,96,.18) !important;
    color:#f0c060 !important;
}

/* ── Data editor ── */
div[data-testid="stDataEditor"] { border-radius:10px; overflow:hidden; }

/* ── Mobile tweaks ── */
@media (max-width: 640px) {
    .top-banner h1 { font-size:1.4rem; }
    .kpi-val { font-size:1.3rem; }
    .stTabs [data-baseweb="tab"] { font-size:.75rem !important; padding:6px 8px !important; }
    .sec-head { font-size:1.1rem; }
}

/* ── Metric override ── */
[data-testid="stMetric"] {
    background:rgba(18,18,32,.8);
    border-radius:12px;
    border:1px solid rgba(255,255,255,.05);
    padding:12px !important;
}

/* ── Number inputs smaller on mobile ── */
[data-testid="stNumberInput"] input { font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
SENSITIVITY_STEPS = (-20, -15, -10, -5, 0, +5, +10, +15, +20)
MIN_MARGIN    = 0.28
MAX_PRICE_UP  = 0.35

QUAD_COLOR  = {"Star":"#f0c060","Puzzle":"#60a5fa","Plow Horse":"#34d399","Dog":"#f87171"}
QUAD_ICON   = {"Star":"⭐","Puzzle":"🧩","Plow Horse":"🐎","Dog":"🐕"}
QUAD_TIP    = {
    "Star":       "Best sellers with great profit. Protect these — never cut quality.",
    "Puzzle":     "Great profit margin but not ordered much. Push these on the specials board or train staff to recommend them.",
    "Plow Horse": "Ordered a lot but profit is thin. Try raising the price a little — customers love it anyway.",
    "Dog":        "Low orders AND low profit. Consider removing, bundling with a Star item, or running a promotion.",
}

DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Nunito", color="#c0c4dc"),
    margin=dict(l=16,r=16,t=36,b=16),
)

# ─────────────────────────────────────────────────────────────
#  DEFAULT MENU
# ─────────────────────────────────────────────────────────────
DEFAULT = {
    "Dish Name":        ["Wagyu Burger","Truffle Pasta","Caesar Salad","Espresso Martini",
                         "Lava Cake","House Wine","Grilled Salmon","Garlic Bread"],
    "Category":         ["Mains","Mains","Starters","Drinks","Desserts","Drinks","Mains","Starters"],
    "Ingredient Cost($)":[8.50,6.20,2.80,1.90,2.40,3.20,9.80,0.80],
    "Selling Price($)": [24.00,22.00,14.00,16.00,12.00,11.00,28.00,7.00],
    "Sold per Week":    [120,85,200,180,90,250,70,160],
    "Cook Time (min)":  [12,15,5,3,8,2,18,4],
    "Price Sensitivity":[-1.2,-1.0,-0.8,-0.6,-1.1,-0.9,-1.3,-0.7],
}

# ─────────────────────────────────────────────────────────────
#  MATH ENGINE  (plain-English variable names where possible)
# ─────────────────────────────────────────────────────────────

def demand_qty(new_price, base_price, base_qty, sensitivity):
    """How many dishes sell at new_price given price sensitivity."""
    if new_price <= 0: return 0.0
    return float(base_qty) * (new_price / float(base_price)) ** float(sensitivity)

def profit_per_dish(price, cost):
    return price - cost

def weekly_profit_one(price, row):
    qty = demand_qty(price, row["Selling Price($)"], row["Sold per Week"], row["Price Sensitivity"])
    return profit_per_dish(price, row["Ingredient Cost($)"]) * qty

def total_weekly_profit(prices, rows, fixed_costs):
    return sum(weekly_profit_one(prices[i], rows[i]) for i in range(len(rows))) - fixed_costs

def run_price_optimiser(rows, fixed_costs):
    p0     = np.array([r["Selling Price($)"] for r in rows])
    bounds = [(r["Ingredient Cost($)"] * (1+MIN_MARGIN), r["Selling Price($)"] * (1+MAX_PRICE_UP)) for r in rows]
    r1 = minimize(lambda p: -total_weekly_profit(p, rows, fixed_costs),
                  p0, bounds=bounds, method="L-BFGS-B",
                  options={"ftol":1e-12,"maxiter":2000})
    r2 = differential_evolution(lambda p: -total_weekly_profit(p, rows, fixed_costs),
                                bounds, seed=42, maxiter=300, popsize=12, tol=1e-9)
    if -r2.fun >= -r1.fun: return r2.x, -r2.fun
    return r1.x, -r1.fun

def run_lp(df, labour_hrs, budget):
    profit_m = (df["Selling Price($)"] - df["Ingredient Cost($)"]).values
    cook_h   = df["Cook Time (min)"].values / 60.0
    costs    = df["Ingredient Cost($)"].values
    qtys     = df["Sold per Week"].values
    return linprog(-profit_m, A_ub=[cook_h, costs], b_ub=[labour_hrs, budget],
                   bounds=[(0,q) for q in qtys], method="highs")

def sensitivity_sweep(row):
    rows_out = []
    base_cm  = weekly_profit_one(row["Selling Price($)"], row)
    for pct in SENSITIVITY_STEPS:
        p   = row["Selling Price($)"] * (1 + pct/100)
        qty = demand_qty(p, row["Selling Price($)"], row["Sold per Week"], row["Price Sensitivity"])
        cm  = profit_per_dish(p, row["Ingredient Cost($)"]) * qty
        rows_out.append({
            "Price Change": f"{pct:+d}%",
            "New Price":    round(p, 2),
            "Est. Orders":  round(qty, 0),
            "Weekly Revenue": round(p * qty, 2),
            "Weekly Profit":  round(cm, 2),
            "Profit Change":  round(cm - base_cm, 2),
            "Margin":         round((p - row["Ingredient Cost($)"]) / p * 100, 1) if p > 0 else 0,
        })
    return rows_out

def classify_dishes(df):
    df = df.copy()
    df["Profit per Dish"] = df["Selling Price($)"] - df["Ingredient Cost($)"]
    df["Profit Margin %"] = df["Profit per Dish"] / df["Selling Price($)"] * 100
    avg_profit = df["Profit per Dish"].mean()
    avg_qty    = df["Sold per Week"].mean()
    def quad(r):
        if r["Profit per Dish"] >= avg_profit and r["Sold per Week"] >= avg_qty: return "Star"
        if r["Profit per Dish"] >= avg_profit and r["Sold per Week"] <  avg_qty: return "Puzzle"
        if r["Profit per Dish"] <  avg_profit and r["Sold per Week"] >= avg_qty: return "Plow Horse"
        return "Dog"
    df["Type"] = df.apply(quad, axis=1)
    return df, avg_profit, avg_qty

def calc_finances(df, fixed):
    profit_each = df["Selling Price($)"] - df["Ingredient Cost($)"]
    total_profit_pool = (profit_each * df["Sold per Week"]).sum()
    total_revenue     = (df["Selling Price($)"] * df["Sold per Week"]).sum()
    total_covers      = df["Sold per Week"].sum()
    net               = total_profit_pool - fixed
    avg_pp_cover      = total_profit_pool / total_covers if total_covers else 0
    bep_covers        = fixed / avg_pp_cover if avg_pp_cover > 0 else float("inf")
    safety            = net / total_profit_pool * 100 if total_profit_pool > 0 else 0
    food_cost_pct     = (df["Ingredient Cost($)"] * df["Sold per Week"]).sum() / total_revenue * 100 if total_revenue > 0 else 0
    return dict(
        pool=total_profit_pool, revenue=total_revenue, net=net,
        monthly=net*4.33, annual=net*52,
        covers=total_covers, avg_pp_cover=avg_pp_cover,
        bep=bep_covers, surplus=total_covers - bep_covers,
        safety=safety, food_cost=food_cost_pct, fixed=fixed,
    )

def smart_tips(eng_df, fin, opt_prices):
    tips = []
    emoji = "🟢" if fin["safety"] > 25 else ("🟡" if fin["safety"] > 10 else "🔴")
    lbl   = "You're doing well" if fin["safety"] > 25 else ("Watch out" if fin["safety"] > 10 else "Urgent attention needed")
    tips.append(("info",
        f"**{emoji} Safety Buffer: {fin['safety']:.1f}%** — {lbl}. "
        f"You need **{fin['bep']:.0f} orders/week** to cover all costs. Right now you have **{fin['covers']:.0f}**."))
    if fin["food_cost"] > 35:
        tips.append(("yellow",
            f"⚠️ **Your ingredient costs are high ({fin['food_cost']:.1f}% of sales).** "
            f"Industry target is under 32%. Check if any supplier prices have gone up recently."))
    for i, row in eng_df.iterrows():
        p_now = row["Selling Price($)"]
        p_opt = opt_prices[i] if opt_prices is not None else p_now
        chg   = (p_opt - p_now) / p_now * 100
        fc    = row["Ingredient Cost($)"] / p_now * 100
        t     = row["Type"]
        if t == "Dog":
            tips.append(("red",
                f"🐕 **{row['Dish Name']}** earns little AND doesn't sell much. "
                f"Think about removing it, or pair it with a Star dish as a combo deal."))
        elif t == "Plow Horse" and chg > 3:
            tips.append(("yellow",
                f"🐎 **{row['Dish Name']}** is popular but your profit on it is low. "
                f"Try raising the price to **${p_opt:.2f}** (+{chg:.1f}%). "
                f"Customers already love it — a small increase probably won't hurt sales."))
        elif t == "Puzzle":
            tips.append(("yellow",
                f"🧩 **{row['Dish Name']}** has great profit but not many people order it. "
                f"Feature it on the specials board, ask staff to recommend it, or add a good photo to the menu."))
        elif t == "Star":
            tips.append(("green",
                f"⭐ **{row['Dish Name']}** is your money-maker — popular AND profitable. "
                f"Keep the quality high. Could you add a premium version at a higher price?"))
        if fc > 38:
            tips.append(("red",
                f"🔴 **{row['Dish Name']}** — ingredients cost {fc:.0f}% of what you charge. "
                f"Raise price to **${row['Ingredient Cost($)'] / 0.32:.2f}** or reduce the portion cost."))
    return tips

# ─────────────────────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────────────────────

def gauge(value, title, danger=10, caution=25, max_val=100):
    color = "#4ade80" if value > caution else ("#fbbf24" if value > danger else "#f87171")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size":12,"color":"#c0c4dc"}},
        number={"suffix":"%","font":{"size":22,"color":color}},
        gauge=dict(axis=dict(range=[0,max_val],tickcolor="#333"),
                   bar=dict(color=color,thickness=0.22),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[{"range":[0,danger],"color":"rgba(248,113,113,.12)"},
                           {"range":[danger,caution],"color":"rgba(251,191,36,.1)"},
                           {"range":[caution,max_val],"color":"rgba(74,222,128,.07)"}],
                   threshold=dict(line=dict(color=color,width=2),thickness=.75,value=value))))
    fig.update_layout(**DARK, height=190)
    return fig

def bar_chart(names, values, title, color="#f0c060", h=300):
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(color=values, colorscale=[[0,"#f87171"],[0.5,"#fbbf24"],[1,"#4ade80"]],
                    showscale=False, line=dict(width=0)),
        text=[f"${v:,.0f}" for v in values], textposition="outside",
        textfont=dict(size=11, color="#c0c4dc")))
    fig.update_layout(**DARK, title=title, height=h, xaxis_tickangle=-30)
    return fig

def waterfall(items_list, opt_prices):
    curr = [(r["Selling Price($)"] - r["Ingredient Cost($)"]) * r["Sold per Week"] for r in items_list]
    opt  = [weekly_profit_one(opt_prices[i], items_list[i]) for i in range(len(items_list))]
    d    = [opt[i] - curr[i] for i in range(len(items_list))]
    names = [r["Dish Name"] for r in items_list]
    fig = go.Figure(go.Waterfall(
        name="Change", orientation="v", x=names, y=d,
        connector={"line":{"color":"rgba(255,255,255,.08)"}},
        increasing={"marker":{"color":"#4ade80"}},
        decreasing={"marker":{"color":"#f87171"}},
        totals={"marker":{"color":"#f0c060"}}))
    fig.update_layout(**DARK, title="Weekly profit change if you use suggested prices",
                      height=300, xaxis_tickangle=-30)
    return fig

def bcg_scatter(eng_df, avg_p, avg_q):
    fig = go.Figure()
    for t, grp in eng_df.groupby("Type"):
        fig.add_trace(go.Scatter(
            x=grp["Sold per Week"], y=grp["Profit per Dish"],
            mode="markers+text", name=f"{QUAD_ICON[t]} {t}",
            text=grp["Dish Name"], textposition="top center",
            textfont=dict(size=9, color="#c0c4dc"),
            marker=dict(size=14, color=QUAD_COLOR[t], opacity=.85,
                        line=dict(width=1, color="rgba(255,255,255,.15)"))))
    fig.add_hline(y=avg_p, line_dash="dash", line_color="rgba(255,255,255,.15)")
    fig.add_vline(x=avg_q, line_dash="dash", line_color="rgba(255,255,255,.15)")
    for label, ax, ay, xa, ya in [
        ("⭐ STAR",       eng_df["Sold per Week"].max()*1.02, eng_df["Profit per Dish"].max()*1.02, "right","top"),
        ("🧩 PUZZLE",     eng_df["Sold per Week"].min()*.9,   eng_df["Profit per Dish"].max()*1.02, "left", "top"),
        ("🐎 PLOW HORSE", eng_df["Sold per Week"].max()*1.02, eng_df["Profit per Dish"].min()*.9,   "right","bottom"),
        ("🐕 DOG",        eng_df["Sold per Week"].min()*.9,   eng_df["Profit per Dish"].min()*.9,   "left", "bottom"),
    ]:
        fig.add_annotation(x=ax,y=ay,text=label,showarrow=False,
                           font=dict(size=8,color="rgba(255,255,255,.2)"),
                           xanchor=xa,yanchor=ya)
    fig.update_layout(**DARK, height=400,
                      title="Which dishes make money AND sell well?",
                      xaxis_title="Orders per Week  →  (more is better)",
                      yaxis_title="Profit per Dish $  →  (higher is better)")
    return fig

def heatmap_chart(df):
    items  = df.to_dict("records")
    matrix = [[s["Profit Change"] for s in sensitivity_sweep(r)] for r in items]
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"{s:+d}%" for s in SENSITIVITY_STEPS],
        y=[r["Dish Name"] for r in items],
        colorscale=[[0,"#f87171"],[0.5,"#1e1e35"],[1,"#4ade80"]],
        colorbar=dict(
            title=dict(text="Profit Change $", font=dict(color="#c0c4dc")),
            tickfont=dict(color="#c0c4dc")),
        text=[[f"${v:+,.0f}" for v in row] for row in matrix],
        texttemplate="%{text}", textfont=dict(size=8), zmid=0))
    fig.update_layout(**DARK, height=300,
                      title="How much does profit change if you raise or lower prices?")
    return fig

def tornado(df):
    items = df.to_dict("records")
    data  = []
    for r in items:
        sens = sensitivity_sweep(r)
        low  = next(s["Profit Change"] for s in sens if s["Price Change"] == "-20%")
        high = next(s["Profit Change"] for s in sens if s["Price Change"] == "+20%")
        data.append({"Dish": r["Dish Name"], "Low": low, "High": high, "Swing": high - low})
    data = sorted(data, key=lambda x: x["Swing"])
    fig  = go.Figure()
    for d in data:
        fig.add_trace(go.Bar(y=[d["Dish"]],x=[d["Low"]], orientation="h",
                             marker_color="#f87171",showlegend=False,
                             name="Price -20%"))
        fig.add_trace(go.Bar(y=[d["Dish"]],x=[d["High"]],orientation="h",
                             marker_color="#4ade80",showlegend=False,
                             name="Price +20%"))
    fig.update_layout(**DARK, barmode="relative", height=300,
                      title="Which dishes are most affected by price changes?",
                      xaxis_title="Profit Change per Week ($)")
    return fig

def pl_waterfall(fin, fixed):
    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=["Total Sales","Ingredient Costs","Fixed Costs","Your Profit"],
        y=[fin["revenue"], -(fin["revenue"]-fin["pool"]), -fixed, fin["net"]],
        connector={"line":{"color":"rgba(255,255,255,.08)"}},
        increasing={"marker":{"color":"#4ade80"}},
        decreasing={"marker":{"color":"#f87171"}},
        totals={"marker":{"color":"#f0c060"}},
        text=[f"${v:,.0f}" for v in [fin["revenue"],-(fin["revenue"]-fin["pool"]),-fixed,fin["net"]]],
        textposition="outside"))
    fig.update_layout(**DARK, title="Where does your money go each week?", height=300)
    return fig

# ─────────────────────────────────────────────────────────────
#  HTML REPORT GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_report(df, fin, eng_df, opt_prices, tips, period, report_date):
    items_list  = df.to_dict("records")
    curr_profit = total_weekly_profit(np.array([r["Selling Price($)"] for r in items_list]), items_list, fin["fixed"])
    opt_profit  = total_weekly_profit(opt_prices, items_list, fin["fixed"]) if opt_prices is not None else curr_profit
    gain_wk     = opt_profit - curr_profit
    factor      = 4.33 if period == "Monthly" else 1
    p_label     = "Month" if period == "Monthly" else "Week"

    tip_rows = ""
    icons = {"info":"💡","yellow":"⚠️","red":"🔴","green":"✅"}
    colors= {"info":"#1e3a5f","yellow":"#3d2e00","red":"#3d1414","green":"#0f3020"}
    bcolors={"info":"#3b82f6","yellow":"#f59e0b","red":"#ef4444","green":"#22c55e"}
    for kind, msg in tips:
        cleaned = msg.replace("**","").replace("*","")
        tip_rows += f"""
        <div style="border-left:4px solid {bcolors.get(kind,'#f0c060')};background:{colors.get(kind,'#1a1a2e')};
                    padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:10px;font-size:14px;line-height:1.6;">
            {icons.get(kind,'•')} {cleaned}
        </div>"""

    dish_rows = ""
    for i, row in eng_df.iterrows():
        p_opt = round(opt_prices[i]*4)/4 if opt_prices is not None else row["Selling Price($)"]
        fc_p  = row["Ingredient Cost($)"] / row["Selling Price($)"] * 100
        dish_rows += f"""
        <tr>
          <td>{row['Dish Name']}</td>
          <td>{row['Category']}</td>
          <td>${row['Ingredient Cost($)']:.2f}</td>
          <td>${row['Selling Price($)']:.2f}</td>
          <td>${row['Profit per Dish']:.2f} ({row['Profit Margin %']:.0f}%)</td>
          <td>{row['Sold per Week']}</td>
          <td>${row['Profit per Dish']*row['Sold per Week']:,.0f}</td>
          <td style="color:{'#f59e0b' if p_opt != row['Selling Price($)'] else '#9ca3af'};">${p_opt:.2f}</td>
          <td style="color:{'#f87171' if fc_p>35 else '#4ade80'};">{fc_p:.0f}%</td>
          <td><span style="background:{QUAD_COLOR[row['Type']]}22;color:{QUAD_COLOR[row['Type']]};
                padding:3px 10px;border-radius:20px;font-size:12px;">{QUAD_ICON[row['Type']]} {row['Type']}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MenuProfit Report — {period} — {report_date}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Nunito:wght@400;600;700&display=swap');
  *, body {{ box-sizing:border-box; margin:0; padding:0; font-family:'Nunito',sans-serif; }}
  body {{ background:#0e0e18; color:#e0e2ee; padding:0; }}
  .page {{ max-width:960px; margin:0 auto; padding:30px 20px; }}

  /* Header */
  .header {{ background:linear-gradient(135deg,#1a1a30,#0e0e18); border-bottom:1px solid rgba(240,192,96,.2);
             padding:28px 32px; border-radius:16px; margin-bottom:28px; }}
  .header h1 {{ font-family:'Playfair Display',serif; font-size:2rem; color:#fff; }}
  .header h1 span {{ color:#f0c060; }}
  .header p {{ color:#7a7fa0; font-size:.85rem; margin-top:6px; }}

  /* KPI grid */
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:14px; margin-bottom:28px; }}
  .kpi-box {{ background:linear-gradient(145deg,#181830,#12121f); border:1px solid rgba(240,192,96,.14);
              border-radius:12px; padding:18px 16px; text-align:center; }}
  .kpi-box .lbl {{ font-size:.72rem; color:#7a7fa0; text-transform:uppercase; letter-spacing:.07em; margin-bottom:6px; }}
  .kpi-box .val {{ font-size:1.6rem; font-weight:800; color:#f0c060; }}
  .kpi-box .sub {{ font-size:.72rem; color:#555c7a; margin-top:4px; }}
  .kpi-box.green .val {{ color:#4ade80; }}
  .kpi-box.red   .val {{ color:#f87171; }}
  .kpi-box.white .val {{ color:#e8eaf0; }}

  /* Section */
  .sec {{ margin-bottom:28px; }}
  .sec h2 {{ font-family:'Playfair Display',serif; font-size:1.3rem; color:#f0c060;
             border-bottom:1px solid rgba(240,192,96,.18); padding-bottom:6px; margin-bottom:14px; }}

  /* Table */
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  thead tr {{ background:#1a1a30; }}
  th {{ padding:10px 10px; text-align:left; color:#9499b5; font-weight:700;
        font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }}
  td {{ padding:9px 10px; border-bottom:1px solid rgba(255,255,255,.05); vertical-align:middle; }}
  tr:hover {{ background:rgba(255,255,255,.025); }}

  /* Scenario table */
  .sce-table td {{ text-align:center; }}

  /* Footer */
  .footer {{ text-align:center; color:#374151; font-size:.75rem; padding:20px 0; border-top:1px solid rgba(255,255,255,.05); margin-top:32px; }}

  @media print {{
    body {{ background:#fff; color:#111; }}
    .header,.kpi-box,table {{ background:#f9f9f9; color:#111; border-color:#ddd; }}
    .kpi-box .val {{ color:#c97d10; }}
    .sec h2 {{ color:#c97d10; }}
  }}
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <h1>🍽️ MenuProfit <span>Pro</span></h1>
    <p>{period} Performance Report &nbsp;·&nbsp; Generated on {report_date} &nbsp;·&nbsp; Period reference: {p_label}</p>
  </div>

  <!-- KPIs -->
  <div class="sec">
    <h2>At a Glance</h2>
    <div class="kpi-grid">
      <div class="kpi-box"><div class="lbl">Total Sales / {p_label}</div>
        <div class="val">${fin['revenue']*factor:,.0f}</div></div>
      <div class="kpi-box"><div class="lbl">Profit Pool / {p_label}</div>
        <div class="val">${fin['pool']*factor:,.0f}</div></div>
      <div class="kpi-box {'green' if fin['net']*factor > 0 else 'red'}">
        <div class="lbl">Net Profit / {p_label}</div>
        <div class="val">${fin['net']*factor:,.0f}</div></div>
      <div class="kpi-box {'green' if gain_wk*factor > 0 else ''}">
        <div class="lbl">Possible Extra Profit</div>
        <div class="val">${gain_wk*factor:+,.0f}</div>
        <div class="sub">if you use suggested prices</div></div>
      <div class="kpi-box {'red' if fin['food_cost'] > 35 else 'green'}">
        <div class="lbl">Food Cost %</div>
        <div class="val">{fin['food_cost']:.1f}%</div>
        <div class="sub">target is under 32%</div></div>
      <div class="kpi-box {'green' if fin['safety'] > 25 else 'red'}">
        <div class="lbl">Safety Buffer</div>
        <div class="val">{fin['safety']:.1f}%</div>
        <div class="sub">how far above breakeven</div></div>
      <div class="kpi-box white"><div class="lbl">Breakeven Orders</div>
        <div class="val">{fin['bep']:.0f}</div>
        <div class="sub">orders/week needed</div></div>
      <div class="kpi-box white"><div class="lbl">Annual Profit</div>
        <div class="val">${fin['annual']:,.0f}</div></div>
    </div>
  </div>

  <!-- Tips -->
  <div class="sec">
    <h2>Smart Tips for This {p_label}</h2>
    {tip_rows}
  </div>

  <!-- Dish breakdown -->
  <div class="sec">
    <h2>Full Dish Breakdown</h2>
    <table>
      <thead><tr>
        <th>Dish</th><th>Category</th><th>Cost</th><th>Price</th>
        <th>Profit/Dish</th><th>Sold/Wk</th><th>Wkly Profit</th>
        <th>Suggested $</th><th>Food Cost%</th><th>Status</th>
      </tr></thead>
      <tbody>{dish_rows}</tbody>
    </table>
  </div>

  <!-- Breakeven scenarios -->
  <div class="sec">
    <h2>What If Your Fixed Costs Change?</h2>
    <table class="sce-table">
      <thead><tr>
        <th>Fixed Costs / Week</th><th>Orders Needed to Break Even</th>
        <th>Your Surplus Orders</th><th>Net Profit / {p_label}</th><th>Annual Profit</th>
      </tr></thead>
      <tbody>"""
    avg_ppc = fin["avg_pp_cover"]
    for m in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
        fc = fin["fixed"] * m
        bep_c = fc / avg_ppc if avg_ppc > 0 else 0
        net_m = (fin["pool"] - fc) * factor
        ann_m = (fin["pool"] - fc) * 52
        surplus = fin["covers"] - bep_c
        c_s = "#4ade80" if surplus > 0 else "#f87171"
        c_n = "#4ade80" if net_m > 0 else "#f87171"
        html += f"""<tr>
          <td>${fc:,.0f}</td><td>{bep_c:.0f}</td>
          <td style="color:{c_s};">{surplus:+.0f}</td>
          <td style="color:{c_n};">${net_m:+,.0f}</td>
          <td style="color:{c_n};">${ann_m:+,.0f}</td>
        </tr>"""
    html += f"""
      </tbody>
    </table>
  </div>

  <div class="footer">
    MenuProfit Pro &nbsp;·&nbsp; {period} Report &nbsp;·&nbsp; {report_date} &nbsp;·&nbsp;
    Built for independent restaurants
  </div>
</div>
</body>
</html>"""
    return html

# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
for k, v in [("opt_prices",None),("eng_df",None),("fin",None),("tips",[]),("ran",False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
#  HEADER BANNER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <h1>🍽️ MenuProfit <span class="gold">Pro</span></h1>
  <p>Simple pricing tool for restaurants & cafes — find your best prices, cut waste, grow profit</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR — business settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Your Business Numbers")
    weekly_fixed  = st.number_input("Fixed costs per week ($)", value=3500, step=100,
                                     help="Rent + wages + electricity + anything you pay every week regardless of sales")
    labour_hrs    = st.number_input("Staff hours per week", value=120, step=5,
                                     help="Total kitchen + front-of-house hours available")
    weekly_budget = st.number_input("Ingredient budget per week ($)", value=4000, step=100,
                                     help="How much you can spend on food & drinks each week")
    st.divider()
    st.markdown("### 📐 Price Rules")
    min_mg = st.slider("Minimum profit margin %", 20, 60, 30,
                        help="Never suggest a price that gives less than this margin")
    max_up = st.slider("Max price increase allowed %", 5, 50, 35,
                        help="The tool won't suggest prices higher than your current price + this %")
    st.divider()
    st.markdown("### 📂 Your Menu File")
    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                 help="Columns needed: Dish Name, Category, Ingredient Cost($), Selling Price($), Sold per Week, Cook Time (min), Price Sensitivity")
    if uploaded:
        try:
            imported = pd.read_csv(uploaded)
            missing  = [c for c in DEFAULT.keys() if c not in imported.columns]
            if missing:
                st.error(f"Missing columns: {missing}"); uploaded = None
            else:
                st.success("File loaded ✓")
        except Exception as e:
            st.error(f"Could not read file: {e}"); uploaded = None
    st.divider()
    st.caption(
        "💡 **Price Sensitivity guide:**\n\n"
        "-0.5 = customers barely notice price changes (e.g. coffee, cocktails)\n\n"
        "-1.0 = balanced\n\n"
        "-1.5 = customers very sensitive to price (e.g. daily staples)"
    )

MIN_MARGIN   = min_mg / 100
MAX_PRICE_UP = max_up / 100

# ─────────────────────────────────────────────────────────────
#  MENU TABLE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">📋 Your Menu</div>', unsafe_allow_html=True)
st.markdown('<div class="tip">✏️ Edit any number below. Add or remove rows with the + button at the bottom.</div>',
            unsafe_allow_html=True)

base_df  = pd.DataFrame(imported if uploaded else DEFAULT)
menu_df  = st.data_editor(base_df, use_container_width=True, num_rows="dynamic",
    column_config={
        "Ingredient Cost($)": st.column_config.NumberColumn("Ingredient Cost ($)", format="$%.2f", min_value=0.0),
        "Selling Price($)":   st.column_config.NumberColumn("Selling Price ($)",   format="$%.2f", min_value=0.0),
        "Sold per Week":      st.column_config.NumberColumn("Sold / Week",         min_value=0),
        "Cook Time (min)":    st.column_config.NumberColumn("Cook Time (min)",     min_value=0),
        "Price Sensitivity":  st.column_config.NumberColumn("Price Sensitivity",   format="%.1f",
                                                             min_value=-3.0, max_value=-0.1,
                                                             help="-0.5 = low sensitivity  |  -1.5 = high sensitivity"),
    }, hide_index=True)

# Quick health check before running
if not menu_df.empty:
    issues = menu_df[menu_df["Ingredient Cost($)"] >= menu_df["Selling Price($)"]]
    if not issues.empty:
        st.warning(f"⚠️ {', '.join(issues['Dish Name'].tolist())} — ingredient cost is higher than selling price! Fix this first.")

# Run button
col_run, col_exp = st.columns([3,1])
with col_run:
    run_btn = st.button("🚀  Analyse My Menu & Find Best Prices")
with col_exp:
    if st.session_state.eng_df is not None:
        buf = io.StringIO()
        st.session_state.eng_df.to_csv(buf, index=False)
        st.download_button("⬇️ Export CSV", buf.getvalue(),
                           "menuprofit_results.csv", "text/csv")

# ─────────────────────────────────────────────────────────────
#  RUN ENGINE
# ─────────────────────────────────────────────────────────────
if run_btn:
    if menu_df.empty:
        st.error("Please add at least one dish."); st.stop()
    with st.spinner("Crunching the numbers..."):
        rows = menu_df.to_dict("records")
        run_lp(menu_df, labour_hrs, weekly_budget)
        opt_p, _ = run_price_optimiser(rows, weekly_fixed)
        st.session_state.opt_prices = opt_p
        eng_df, _, _ = classify_dishes(menu_df)
        st.session_state.eng_df = eng_df
        fin = calc_finances(menu_df, weekly_fixed)
        st.session_state.fin  = fin
        st.session_state.tips = smart_tips(eng_df, fin, opt_p)
        st.session_state.ran  = True
    st.success("Done! Scroll down to see your results ↓")

# ─────────────────────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────────────────────
if st.session_state.ran and st.session_state.eng_df is not None:
    eng_df     = st.session_state.eng_df
    opt_prices = st.session_state.opt_prices
    fin        = st.session_state.fin
    tips       = st.session_state.tips
    rows       = menu_df.to_dict("records")

    curr_profit = total_weekly_profit(np.array([r["Selling Price($)"] for r in rows]), rows, weekly_fixed)
    opt_profit  = total_weekly_profit(opt_prices, rows, weekly_fixed)
    wk_gain     = opt_profit - curr_profit
    yr_gain     = wk_gain * 52

    # TABS
    t1,t2,t3,t4,t5,t6,t7 = st.tabs([
        "📊 Overview",
        "💰 Best Prices",
        "📈 Price Testing",
        "🍽️ Dish Ratings",
        "📉 Breakeven",
        "💡 Tips",
        "📄 Report",
    ])

    # ── TAB 1: OVERVIEW ──────────────────────────────────────
    with t1:
        st.markdown('<div class="sec-head">This Week at a Glance</div>', unsafe_allow_html=True)

        # KPI row 1
        k1,k2,k3,k4 = st.columns(4)
        for col, lbl, val, sub, cls in [
            (k1,"Total Sales",    f"${fin['revenue']:,.0f}",    "this week",    "white"),
            (k2,"Your Profit Pool",f"${fin['pool']:,.0f}",      "before fixed costs","white"),
            (k3,"Net Profit",     f"${fin['net']:,.0f}",        "after all costs", "green" if fin['net']>0 else "red"),
            (k4,"Food Cost",      f"{fin['food_cost']:.1f}%",   "target under 32%","green" if fin['food_cost']<=32 else "red"),
        ]:
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-lbl">{lbl}</div>'
                            f'<div class="kpi-val {cls}">{val}</div>'
                            f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # KPI row 2
        k5,k6,k7,k8 = st.columns(4)
        for col, lbl, val, sub, cls in [
            (k5,"Monthly Profit",  f"${fin['monthly']:,.0f}", "×4.33 weeks",   "green" if fin['monthly']>0 else "red"),
            (k6,"Annual Profit",   f"${fin['annual']:,.0f}",  "×52 weeks",     "green" if fin['annual']>0 else "red"),
            (k7,"Extra if Repriced",f"${wk_gain:+,.0f}/wk",  f"${yr_gain:+,.0f}/yr","green" if wk_gain>0 else "white"),
            (k8,"Orders / Week",   f"{fin['covers']:.0f}",    f"need {fin['bep']:.0f} to break even","white"),
        ]:
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-lbl">{lbl}</div>'
                            f'<div class="kpi-val {cls}">{val}</div>'
                            f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            wk_profits = [(r["Selling Price($)"] - r["Ingredient Cost($)"]) * r["Sold per Week"] for r in rows]
            st.plotly_chart(bar_chart([r["Dish Name"] for r in rows], wk_profits,
                                      "Weekly profit by dish ($)"), use_container_width=True)
        with col_b:
            # Category revenue pie
            cat_rev = {}
            for r in rows:
                cat = r["Category"]
                cat_rev[cat] = cat_rev.get(cat, 0) + r["Selling Price($)"] * r["Sold per Week"]
            fig_pie = go.Figure(go.Pie(
                labels=list(cat_rev.keys()), values=list(cat_rev.values()),
                hole=.45, marker=dict(colors=["#f0c060","#60a5fa","#34d399","#f87171","#a78bfa","#fb923c"]),
                textfont=dict(size=11)))
            fig_pie.update_layout(**DARK, title="Sales split by category", height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Food cost bar
        fc_vals = [r["Ingredient Cost($)"] / r["Selling Price($)"] * 100 for r in rows]
        fig_fc  = go.Figure(go.Bar(
            x=[r["Dish Name"] for r in rows], y=fc_vals,
            marker=dict(color=fc_vals,
                        colorscale=[[0,"#4ade80"],[0.45,"#fbbf24"],[1,"#f87171"]],
                        showscale=False),
            text=[f"{v:.0f}%" for v in fc_vals], textposition="outside",
            textfont=dict(size=10, color="#c0c4dc")))
        fig_fc.add_hline(y=32, line_dash="dash", line_color="#f0c060",
                          annotation_text="Target 32%", annotation_font_color="#f0c060")
        fig_fc.update_layout(**DARK, height=270, xaxis_tickangle=-30,
                              title="Ingredient cost as % of selling price  (lower is better)")
        st.plotly_chart(fig_fc, use_container_width=True)

    # ── TAB 2: BEST PRICES ───────────────────────────────────
    with t2:
        st.markdown('<div class="sec-head">💰 Suggested Prices</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip">These prices are calculated to maximise your total weekly profit '
                    'while keeping demand realistic. You don\'t have to use all of them — '
                    'start with the ones that show the biggest gain.</div>', unsafe_allow_html=True)

        o1,o2,o3 = st.columns(3)
        for col,lbl,val,sub,cls in [
            (o1,"Current weekly profit",    f"${curr_profit:,.0f}","",          "white"),
            (o2,"Profit with new prices",   f"${opt_profit:,.0f}", "",          "green" if opt_profit>curr_profit else "red"),
            (o3,"Extra profit per year",    f"${yr_gain:+,.0f}",   "if you reprice today","green" if yr_gain>0 else "white"),
        ]:
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-lbl">{lbl}</div>'
                            f'<div class="kpi-val {cls}">{val}</div>'
                            f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(waterfall(rows, opt_prices), use_container_width=True)

        price_rows = []
        for i, r in enumerate(rows):
            p_c   = r["Selling Price($)"]
            p_o   = round(opt_prices[i] * 4) / 4
            chg   = (p_o - p_c) / p_c * 100
            cm_c  = weekly_profit_one(p_c, r)
            cm_o  = weekly_profit_one(p_o, r)
            diff  = cm_o - cm_c
            price_rows.append({
                "Dish":              r["Dish Name"],
                "Current Price":     f"${p_c:.2f}",
                "Suggested Price":   f"${p_o:.2f}",
                "Price Change":      f"{chg:+.1f}%",
                "Current Profit/Wk":f"${cm_c:,.0f}",
                "New Profit/Wk":    f"${cm_o:,.0f}",
                "Extra Per Week":   f"${diff:+,.0f}",
            })
        st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)

    # ── TAB 3: PRICE TESTING ─────────────────────────────────
    with t3:
        st.markdown('<div class="sec-head">📈 What Happens If I Change a Price?</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip">See how raising or lowering your price changes orders, revenue and profit '
                    'for any dish — before you actually do it.</div>', unsafe_allow_html=True)

        st.plotly_chart(heatmap_chart(menu_df), use_container_width=True)
        st.plotly_chart(tornado(menu_df), use_container_width=True)

        st.markdown("#### Try a specific price")
        sel = st.selectbox("Pick a dish", menu_df["Dish Name"].tolist(), key="pt_sel")
        sel_row = menu_df[menu_df["Dish Name"] == sel].to_dict("records")[0]
        new_price = st.slider(
            "Move the price",
            min_value=float(sel_row["Ingredient Cost($)"] * 1.05),
            max_value=float(sel_row["Selling Price($)"] * 2.0),
            value=float(sel_row["Selling Price($)"]),
            step=0.25,
        )
        new_qty    = demand_qty(new_price, sel_row["Selling Price($)"], sel_row["Sold per Week"], sel_row["Price Sensitivity"])
        new_cm     = (new_price - sel_row["Ingredient Cost($)"]) * new_qty
        base_cm    = (sel_row["Selling Price($)"] - sel_row["Ingredient Cost($)"]) * sel_row["Sold per Week"]
        delta_cm   = new_cm - base_cm
        margin_pct = (new_price - sel_row["Ingredient Cost($)"]) / new_price * 100

        wi1,wi2,wi3,wi4 = st.columns(4)
        wi1.metric("Estimated Orders/Week", f"{new_qty:.0f}", f"{new_qty - sel_row['Sold per Week']:+.0f}")
        wi2.metric("Weekly Profit",         f"${new_cm:,.0f}", f"${delta_cm:+,.0f}")
        wi3.metric("Annual Extra Profit",   f"${delta_cm*52:+,.0f}")
        wi4.metric("Your Margin",           f"{margin_pct:.1f}%")

        # Sensitivity table
        sens_data = sensitivity_sweep(sel_row)
        sens_df   = pd.DataFrame(sens_data)
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=sens_df["Price Change"], y=sens_df["Weekly Profit"],
                                    mode="lines+markers", name="Weekly Profit",
                                    line=dict(color="#f0c060",width=2), marker=dict(size=8)))
        fig_s.add_trace(go.Scatter(x=sens_df["Price Change"], y=sens_df["Weekly Revenue"],
                                    mode="lines+markers", name="Weekly Revenue",
                                    line=dict(color="#60a5fa",width=2,dash="dot"), marker=dict(size=6)))
        fig_s.update_layout(**DARK, title=f"Profit & revenue at different prices — {sel}",
                             xaxis_title="Price Change", yaxis_title="$ per Week", height=300)
        st.plotly_chart(fig_s, use_container_width=True)
        sens_df["New Price"]       = sens_df["New Price"].map("${:.2f}".format)
        sens_df["Weekly Revenue"]  = sens_df["Weekly Revenue"].map("${:,.0f}".format)
        sens_df["Weekly Profit"]   = sens_df["Weekly Profit"].map("${:,.0f}".format)
        sens_df["Profit Change"]   = sens_df["Profit Change"].map("${:+,.0f}".format)
        st.dataframe(sens_df, use_container_width=True, hide_index=True)

    # ── TAB 4: DISH RATINGS ──────────────────────────────────
    with t4:
        st.markdown('<div class="sec-head">🍽️ Dish Report Card</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip">Every dish gets rated based on how popular it is AND how much profit it makes. '
                    'Use this to decide what to promote, reprice, or remove.</div>', unsafe_allow_html=True)

        _, avg_p, avg_q = classify_dishes(menu_df)
        st.plotly_chart(bcg_scatter(eng_df, avg_p, avg_q), use_container_width=True)

        cl, cr = st.columns(2)
        for quad in ["Star","Puzzle","Plow Horse","Dog"]:
            sub = eng_df[eng_df["Type"] == quad]
            col = cl if quad in ("Star","Plow Horse") else cr
            with col:
                if sub.empty: continue
                names = ", ".join(sub["Dish Name"].tolist())
                st.markdown(f"""
                <div class="alert {'green' if quad=='Star' else 'yellow' if quad in ('Puzzle','Plow Horse') else 'red'}">
                  <b style="color:{QUAD_COLOR[quad]};">{QUAD_ICON[quad]} {quad}</b> — {names}<br>
                  <span style="color:#9499b5;font-size:.82rem;">{QUAD_TIP[quad]}</span>
                </div>""", unsafe_allow_html=True)

        # Margin bar
        fig_mg = px.bar(eng_df.sort_values("Profit Margin %", ascending=True),
                        x="Profit Margin %", y="Dish Name", orientation="h",
                        color="Type", color_discrete_map=QUAD_COLOR,
                        title="Profit margin per dish  (how much of each dollar you keep)")
        fig_mg.add_vline(x=50, line_dash="dash", line_color="#f0c060",
                          annotation_text="50% target", annotation_font_color="#f0c060")
        fig_mg.update_layout(**DARK, height=300)
        st.plotly_chart(fig_mg, use_container_width=True)

        # Full table
        tbl = eng_df[["Dish Name","Category","Ingredient Cost($)","Selling Price($)",
                       "Profit per Dish","Profit Margin %","Sold per Week","Type"]].copy()
        tbl["Weekly Profit"] = (tbl["Profit per Dish"] * tbl["Sold per Week"]).map("${:,.0f}".format)
        tbl["Profit per Dish"] = tbl["Profit per Dish"].map("${:.2f}".format)
        tbl["Profit Margin %"] = tbl["Profit Margin %"].map("{:.1f}%".format)
        tbl["Ingredient Cost($)"] = tbl["Ingredient Cost($)"].map("${:.2f}".format)
        tbl["Selling Price($)"]   = tbl["Selling Price($)"].map("${:.2f}".format)
        tbl = tbl.rename(columns={"Dish Name":"Dish","Ingredient Cost($)":"Cost",
                                    "Selling Price($)":"Price","Profit per Dish":"Profit/Dish",
                                    "Profit Margin %":"Margin","Sold per Week":"Orders/Wk","Type":"Rating"})
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── TAB 5: BREAKEVEN ─────────────────────────────────────
    with t5:
        st.markdown('<div class="sec-head">📉 Can You Cover Your Costs?</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip">Breakeven = the minimum number of orders you need each week '
                    'just to pay your fixed costs (rent, wages, electricity). '
                    'Everything above that is profit.</div>', unsafe_allow_html=True)

        b1,b2,b3 = st.columns(3)
        with b1: st.plotly_chart(gauge(fin["safety"], "Safety Buffer %", danger=10, caution=25), use_container_width=True)
        with b2: st.plotly_chart(gauge(max(0, 100-fin["food_cost"]), "Gross Margin Health", danger=60, caution=68), use_container_width=True)
        with b3:
            sp = min(fin["surplus"] / fin["bep"] * 100, 100) if fin["bep"] > 0 else 0
            st.plotly_chart(gauge(sp, "Orders Buffer Above Breakeven", danger=20, caution=50), use_container_width=True)

        st.plotly_chart(pl_waterfall(fin, weekly_fixed), use_container_width=True)

        st.markdown("#### What if my fixed costs go up or down?")
        avg_ppc = fin["avg_pp_cover"]
        sce = []
        for m in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
            fc = weekly_fixed * m
            bep_c = fc / avg_ppc if avg_ppc > 0 else 0
            sce.append({
                "Your Fixed Costs/Wk": f"${fc:,.0f}",
                "Orders to Break Even": f"{bep_c:.0f}",
                "Your Surplus Orders":  f"{fin['covers'] - bep_c:+.0f}",
                "Net Profit / Week":   f"${fin['pool'] - fc:+,.0f}",
                "Net Profit / Year":   f"${(fin['pool'] - fc)*52:+,.0f}",
            })
        st.dataframe(pd.DataFrame(sce), use_container_width=True, hide_index=True)

        st.markdown("#### Your key numbers")
        kn = pd.DataFrame({
            "What": ["Total Sales / Week", "Profit Pool / Week", "Fixed Costs / Week",
                     "Net Profit / Week", "Net Profit / Month", "Net Profit / Year",
                     "Average Profit per Order", "Orders to Break Even",
                     "Your Orders / Week", "Extra Orders Above Breakeven",
                     "Safety Buffer", "Food Cost %"],
            "Amount": [f"${fin['revenue']:,.2f}", f"${fin['pool']:,.2f}", f"${fin['fixed']:,.2f}",
                       f"${fin['net']:,.2f}", f"${fin['monthly']:,.2f}", f"${fin['annual']:,.0f}",
                       f"${fin['avg_pp_cover']:.2f}", f"{fin['bep']:.0f}",
                       f"{fin['covers']:.0f}", f"{fin['surplus']:+.0f}",
                       f"{fin['safety']:.1f}%", f"{fin['food_cost']:.1f}%"],
        })
        st.dataframe(kn, use_container_width=True, hide_index=True)

    # ── TAB 6: TIPS ──────────────────────────────────────────
    with t6:
        st.markdown('<div class="sec-head">💡 Smart Tips</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip">These tips are based on your actual menu data — not generic advice.</div>',
                    unsafe_allow_html=True)

        TYPE_CSS = {"info":"","yellow":"yellow","red":"red","green":"green"}
        TYPE_ICO = {"info":"💡","yellow":"⚠️","red":"🔴","green":"✅"}
        for kind, msg in tips:
            st.markdown(f'<div class="alert {TYPE_CSS.get(kind,"")}">'
                        f'{TYPE_ICO.get(kind,"•")} {msg}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🧪 Try a price — see the impact instantly")
        st.caption("Drag the slider and watch the numbers update in real time.")
        wi_dish  = st.selectbox("Which dish?", menu_df["Dish Name"].tolist(), key="wi_dish")
        wi_r     = menu_df[menu_df["Dish Name"] == wi_dish].to_dict("records")[0]
        wi_price = st.slider("New price ($)",
                              min_value=float(wi_r["Ingredient Cost($)"] * 1.05),
                              max_value=float(wi_r["Selling Price($)"] * 2.0),
                              value=float(wi_r["Selling Price($)"]), step=0.25,
                              key="wi_slider")
        wi_qty  = demand_qty(wi_price, wi_r["Selling Price($)"], wi_r["Sold per Week"], wi_r["Price Sensitivity"])
        wi_cm   = (wi_price - wi_r["Ingredient Cost($)"]) * wi_qty
        wi_base = (wi_r["Selling Price($)"] - wi_r["Ingredient Cost($)"]) * wi_r["Sold per Week"]
        wi_d    = wi_cm - wi_base

        c1,c2,c3 = st.columns(3)
        c1.metric("Estimated Orders / Week", f"{wi_qty:.0f}", f"{wi_qty - wi_r['Sold per Week']:+.0f}")
        c2.metric("Weekly Profit from this dish", f"${wi_cm:,.0f}", f"${wi_d:+,.0f}")
        c3.metric("Extra per Year", f"${wi_d*52:+,.0f}", "vs current price")

    # ── TAB 7: REPORT ─────────────────────────────────────────
    with t7:
        st.markdown('<div class="sec-head">📄 Download Your Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="tip">Generate a clean, printable HTML report to save, share with your accountant, '
                    'or review with your team. Open it in any browser and use Ctrl+P to print as PDF.</div>',
                    unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            period = st.radio("Report period", ["Weekly", "Monthly"], horizontal=True)
        with rc2:
            report_date = st.date_input("Report date", datetime.date.today())

        st.markdown("<br>", unsafe_allow_html=True)

        # Live preview of key numbers in report
        factor = 4.33 if period == "Monthly" else 1
        p_lbl  = "month" if period == "Monthly" else "week"
        st.markdown(f"**Preview — {period} figures:**")
        rv1,rv2,rv3,rv4 = st.columns(4)
        rv1.metric(f"Sales / {p_lbl}",  f"${fin['revenue']*factor:,.0f}")
        rv2.metric(f"Net Profit / {p_lbl}", f"${fin['net']*factor:,.0f}")
        rv3.metric("Annual Profit",     f"${fin['annual']:,.0f}")
        rv4.metric("Extra if Repriced", f"${wk_gain*factor*(-1 if period=='Monthly' else 1):+,.0f}/{p_lbl}")

        if st.button("📄  Generate & Download Report"):
            html_report = generate_report(
                menu_df, fin, eng_df, opt_prices, tips, period, str(report_date)
            )
            st.download_button(
                label=f"⬇️  Download {period} Report (HTML)",
                data=html_report,
                file_name=f"MenuProfit_{period}_Report_{report_date}.html",
                mime="text/html",
            )
            st.success("Report ready! Click the button above to download.")
            st.markdown('<div class="tip">Tip: Open the downloaded file in Chrome or Edge, '
                        'then press Ctrl+P → Save as PDF to get a PDF version.</div>', unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div style="text-align:center;color:#374151;font-size:.75rem;padding:24px 0 8px;
                border-top:1px solid rgba(255,255,255,.05);margin-top:24px;">
      MenuProfit Pro &nbsp;·&nbsp; Made for independent restaurants &nbsp;·&nbsp; Real numbers, not guesswork
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:50px 16px;color:#4b5563;">
      <div style="font-size:2.6rem;margin-bottom:14px;">🍽️</div>
      <div style="font-size:1.05rem;line-height:1.7;">
        Fill in your menu above (or keep the example data),<br>
        then tap <b style="color:#f0c060;">Analyse My Menu</b> to get your results.
      </div>
    </div>
    """, unsafe_allow_html=True)
