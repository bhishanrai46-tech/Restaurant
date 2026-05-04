#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          MENUPROFIT — Restaurant Pricing Optimization Engine                ║
║          Real calculations: NLP, Sensitivity Analysis, Menu Engineering     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dependencies:  pip install numpy scipy matplotlib
Optional:      pip install tabulate  (prettier tables)

How it works:
  - Demand model:      Power-law  Q(p) = Q₀·(p/p₀)^ε   (ε = price elasticity)
  - Optimization:      scipy L-BFGS-B (constrained nonlinear programming)
  - Sensitivity:       Closed-form ±5/10/15/20% price sweep per item
  - Breakeven:         Contribution margin vs. fixed-cost structure
  - Menu Engineering:  BCG-style quadrant (Star/Puzzle/PlowHorse/Dog)
  - Visualisation:     matplotlib charts (saved as PNG + shown if GUI available)
"""

import sys
import json
import os
from datetime import datetime

import numpy as np
from scipy.optimize import minimize, differential_evolution

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # headless-safe; remove if you want live window
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not installed — charts will be skipped.\n"
          "       Run:  pip install matplotlib\n")

try:
    from tabulate import tabulate
    HAS_TAB = True
except ImportError:
    HAS_TAB = False

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WEEKLY_FIXED_COSTS = 3_500.00   # rent, wages, utilities, etc.
MIN_MARGIN_PCT     = 0.30       # never price below cost + 30 %
MAX_PRICE_INCREASE = 0.35       # optimiser cap: +35 % above current price
REPORT_DIR         = "."        # where to save JSON report & PNG charts

# ── Default menu (edit freely) ─────────────────────────────────────────────
DEFAULT_MENU: list[dict] = [
    # elasticity: -1.2 means a 1 % price rise → 1.2 % demand drop (elastic)
    #             -0.6 means a 1 % price rise → 0.6 % demand drop (inelastic)
    dict(id=1, name="Wagyu Burger",      category="Mains",    cost=8.50,  price=24.00, weekly_qty=120, elasticity=-1.2),
    dict(id=2, name="Truffle Pasta",     category="Mains",    cost=6.20,  price=22.00, weekly_qty=85,  elasticity=-1.0),
    dict(id=3, name="Caesar Salad",      category="Starters", cost=2.80,  price=14.00, weekly_qty=200, elasticity=-0.8),
    dict(id=4, name="Espresso Martini",  category="Drinks",   cost=1.90,  price=16.00, weekly_qty=180, elasticity=-0.6),
    dict(id=5, name="Lava Cake",         category="Desserts", cost=2.40,  price=12.00, weekly_qty=90,  elasticity=-1.1),
    dict(id=6, name="House Wine",        category="Drinks",   cost=3.20,  price=11.00, weekly_qty=250, elasticity=-0.9),
    dict(id=7, name="Grilled Salmon",    category="Mains",    cost=9.80,  price=28.00, weekly_qty=70,  elasticity=-1.3),
    dict(id=8, name="Garlic Bread",      category="Starters", cost=0.80,  price=7.00,  weekly_qty=160, elasticity=-0.7),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  DEMAND & PROFIT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def demand(p: float, p0: float, q0: float, eps: float) -> float:
    """
    Power-law demand curve: Q(p) = Q₀ · (p / p₀)^ε
    ε < 0: price ↑ → quantity ↓  (standard downward-sloping demand)
    ε ∈ (-1, 0): inelastic   |ε| > 1: elastic
    """
    if p <= 0:
        return 0.0
    return q0 * (p / p0) ** eps


def item_contribution_margin(p: float, item: dict) -> float:
    """Weekly contribution margin = (price − cost) × quantity"""
    q = demand(p, item["price"], item["weekly_qty"], item["elasticity"])
    return (p - item["cost"]) * q


def total_weekly_profit(prices: np.ndarray, items: list[dict], fixed: float) -> float:
    """Sum of all items' contribution margins minus weekly fixed costs."""
    return sum(
        item_contribution_margin(prices[i], items[i])
        for i in range(len(items))
    ) - fixed


def optimal_price_closed_form(item: dict) -> float:
    """
    Analytical profit-maximising price for power-law demand.

    Derivation:
        π(p) = (p − c) · Q₀ · (p/p₀)^ε
        dπ/dp = 0  →  1 + ε·(1 − c/p) = 0
               →  p* = ε·c / (ε + 1)      [valid for ε < −1]

    For inelastic items (−1 < ε < 0) the unconstrained optimal is ∞;
    we cap the suggestion at +20 % above current price.
    """
    eps = item["elasticity"]
    c   = item["cost"]
    p0  = item["price"]

    if eps < -1.0:
        p_star = (eps * c) / (eps + 1.0)
    else:
        p_star = p0 * 1.20        # inelastic: push up conservatively

    p_star = max(p_star, c * (1 + MIN_MARGIN_PCT))   # floor: margin floor
    p_star = min(p_star, p0 * (1 + MAX_PRICE_INCREASE))  # cap: market ceiling
    return round(p_star * 4) / 4   # snap to nearest $0.25


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMISATION ENGINE  (scipy constrained NLP)
# ═══════════════════════════════════════════════════════════════════════════════

def run_optimisation(items: list[dict], fixed: float) -> tuple:
    """
    Multi-variable constrained optimisation over all item prices simultaneously.

    Objective  : maximise Σ (pᵢ − cᵢ) · Q(pᵢ)  −  FixedCosts
    Constraints:
        pᵢ ≥ cᵢ · (1 + MIN_MARGIN_PCT)          [minimum gross margin]
        pᵢ ≤ current_pᵢ · (1 + MAX_PRICE_INCREASE) [market cap]
    Method: L-BFGS-B  (gradient-based, handles bounds natively)
    """
    n  = len(items)
    p0 = np.array([item["price"] for item in items], dtype=float)

    bounds = [
        (item["cost"] * (1 + MIN_MARGIN_PCT),
         item["price"] * (1 + MAX_PRICE_INCREASE))
        for item in items
    ]

    result = minimize(
        fun=lambda p: -total_weekly_profit(p, items, fixed),
        x0=p0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 2000},
    )
    return result, -result.fun     # (OptimizeResult, optimised profit)


def run_global_optimisation(items: list[dict], fixed: float) -> tuple:
    """
    Global optimisation via differential evolution — escapes local optima.
    Slower but more thorough; useful when item count is large.
    """
    n = len(items)
    bounds = [
        (item["cost"] * (1 + MIN_MARGIN_PCT),
         item["price"] * (1 + MAX_PRICE_INCREASE))
        for item in items
    ]
    result = differential_evolution(
        func=lambda p: -total_weekly_profit(p, items, fixed),
        bounds=bounds,
        seed=42,
        maxiter=300,
        popsize=12,
        tol=1e-8,
    )
    return result, -result.fun


# ═══════════════════════════════════════════════════════════════════════════════
#  SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

PRICE_SCENARIOS = (-20, -15, -10, -5, 0, 5, 10, 15, 20)  # %

def sensitivity_single(item: dict, scenarios=PRICE_SCENARIOS) -> list[dict]:
    """
    For one menu item: sweep price across scenarios, compute all KPIs.
    Returns list of dicts (one per scenario).
    """
    c   = item["cost"]
    p0  = item["price"]
    q0  = item["weekly_qty"]
    eps = item["elasticity"]
    base_cm = item_contribution_margin(p0, item)

    rows = []
    for pct in scenarios:
        p   = p0 * (1 + pct / 100)
        q   = max(0.0, demand(p, p0, q0, eps))
        cm  = max(0.0, (p - c) * q)
        rev = p * q
        margin = (p - c) / p * 100 if p > 0 else 0
        rows.append(dict(
            pct_change = pct,
            price      = p,
            quantity   = q,
            revenue    = rev,
            weekly_cm  = cm,
            delta_cm   = cm - base_cm,
            margin_pct = margin,
        ))
    return rows


def sensitivity_all(items: list[dict], fixed: float, scenarios=PRICE_SCENARIOS) -> list[dict]:
    """
    Global sensitivity: vary ALL prices by the same % simultaneously.
    Shows how total profit shifts with a uniform pricing move.
    """
    rows = []
    base_profit = total_weekly_profit(
        np.array([i["price"] for i in items]), items, fixed
    )
    for pct in scenarios:
        prices = np.array([i["price"] * (1 + pct / 100) for i in items])
        profit = total_weekly_profit(prices, items, fixed)
        rev    = sum(prices[i] * demand(prices[i], items[i]["price"],
                                        items[i]["weekly_qty"], items[i]["elasticity"])
                     for i in range(len(items)))
        rows.append(dict(
            pct_change   = pct,
            total_profit = profit,
            delta_profit = profit - base_profit,
            total_revenue= rev,
        ))
    return rows


def tornado_data(items: list[dict]) -> list[dict]:
    """
    Profit swing range for each item (−20 % to +20 %).
    Used to rank items by profit sensitivity.
    """
    result = []
    for item in items:
        sens = sensitivity_single(item)
        low  = next(s["delta_cm"] for s in sens if s["pct_change"] == -20)
        high = next(s["delta_cm"] for s in sens if s["pct_change"] == +20)
        result.append(dict(
            name  = item["name"],
            low   = low,
            high  = high,
            swing = high - low,
        ))
    return sorted(result, key=lambda x: x["swing"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MENU ENGINEERING  (BCG Matrix)
# ═══════════════════════════════════════════════════════════════════════════════

QUADRANT_ICON = {"Star": "⭐", "Puzzle": "🧩", "Plow Horse": "🐎", "Dog": "🐕"}
QUADRANT_STRATEGY = {
    "Star"      : "PROTECT & PROMOTE — feature prominently, maintain quality",
    "Puzzle"    : "ANALYSE & PUSH — upsell via staff, add to specials board",
    "Plow Horse": "STREAMLINE — cut ingredient cost OR raise price to boost margin",
    "Dog"       : "REVIEW — consider removing, bundling, or replacing",
}

def menu_engineering(items: list[dict]) -> tuple[list[dict], float, float]:
    """
    Classify each item into BCG quadrant.
    X-axis: weekly quantity   (proxy for popularity)
    Y-axis: contribution margin per unit  (proxy for profitability)
    """
    cms  = np.array([i["price"] - i["cost"] for i in items])
    qtys = np.array([i["weekly_qty"]        for i in items])
    avg_cm  = float(np.mean(cms))
    avg_qty = float(np.mean(qtys))

    enriched = []
    for item, cm, qty in zip(items, cms, qtys):
        cm_pct = cm / item["price"] * 100
        if cm >= avg_cm and qty >= avg_qty:
            quad = "Star"
        elif cm >= avg_cm and qty < avg_qty:
            quad = "Puzzle"
        elif cm < avg_cm and qty >= avg_qty:
            quad = "Plow Horse"
        else:
            quad = "Dog"
        enriched.append({**item, "cm": float(cm), "cm_pct": float(cm_pct), "quadrant": quad})
    return enriched, avg_cm, avg_qty


# ═══════════════════════════════════════════════════════════════════════════════
#  BREAKEVEN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def breakeven_analysis(items: list[dict], fixed: float) -> dict:
    """
    Breakeven weekly covers and operating safety margin.
    Safety margin = (actual CM − fixed costs) / actual CM × 100
    """
    total_cm     = sum((i["price"] - i["cost"]) * i["weekly_qty"] for i in items)
    total_covers = sum(i["weekly_qty"] for i in items)
    net_profit   = total_cm - fixed

    # Weighted avg CM per cover
    avg_cm_per_cover = total_cm / total_covers if total_covers else 0
    be_covers = fixed / avg_cm_per_cover if avg_cm_per_cover > 0 else float("inf")

    safety_margin = (net_profit / total_cm * 100) if total_cm > 0 else 0
    return dict(
        total_weekly_cm      = total_cm,
        total_weekly_revenue = sum(i["price"] * i["weekly_qty"] for i in items),
        net_profit           = net_profit,
        monthly_profit       = net_profit * 4.33,
        annual_profit        = net_profit * 52,
        total_weekly_covers  = total_covers,
        avg_cm_per_cover     = avg_cm_per_cover,
        breakeven_covers     = be_covers,
        surplus_covers       = total_covers - be_covers,
        safety_margin_pct    = safety_margin,
        fixed_costs          = fixed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TERMINAL FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

W = 86   # terminal width

RED    = "\033[91m"
GRN    = "\033[92m"
YLW    = "\033[93m"
GOLD   = "\033[33m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RST    = "\033[0m"

def clr(v: float, s: str) -> str:
    if v > 0:  return f"{GRN}{s}{RST}"
    if v < 0:  return f"{RED}{s}{RST}"
    return s

def banner(title: str) -> str:
    pad = (W - len(title) - 2) // 2
    return (f"\n{'═' * W}\n"
            f"{GOLD}{'║'}{RST}{' ' * pad}{BOLD}{title}{RST}{' ' * (W - pad - len(title) - 2)}{GOLD}{'║'}{RST}\n"
            f"{'═' * W}")

def section(title: str) -> str:
    return f"\n{DIM}{'─' * W}{RST}\n  {BOLD}{GOLD}{title}{RST}\n{DIM}{'─' * W}{RST}"

def row_sep(width=W): return f"{DIM}{'─' * width}{RST}"

def pct_bar(pct: float, width: int = 40) -> str:
    filled = int(min(100, max(0, pct)) * width / 100)
    color  = GRN if pct > 30 else (YLW if pct > 10 else RED)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RST}  {pct:5.1f}%"

def simple_table(headers: list, rows: list, col_fmt: list | None = None) -> str:
    """Minimal table without tabulate dependency."""
    if HAS_TAB:
        return tabulate(rows, headers=headers, tablefmt="simple", floatfmt=".2f")
    # Fallback
    lines = ["  " + "  ".join(f"{h:<14}" for h in headers)]
    lines.append("  " + "─" * (16 * len(headers)))
    for r in rows:
        lines.append("  " + "  ".join(f"{str(c):<14}" for c in r))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_menu_overview(items: list[dict]) -> None:
    print(section("📋  MENU OVERVIEW — Current State"))
    fmt = f"  {{:<23}} {{:<12}} {{:>7}} {{:>8}} {{:>8}} {{:>9}} {{:>10}}"
    print(fmt.format("Item", "Category", "Cost", "Price", "Margin%", "Qty/wk", "Wkly CM"))
    print(row_sep())
    total_rev = total_cm = 0.0
    for i in items:
        cm     = i["price"] - i["cost"]
        margin = cm / i["price"] * 100
        w_cm   = cm * i["weekly_qty"]
        rev    = i["price"] * i["weekly_qty"]
        total_cm  += w_cm
        total_rev += rev
        c_margin   = GRN if margin >= 65 else (YLW if margin >= 50 else RED)
        print(fmt.format(
            i["name"], i["category"],
            f"${i['cost']:.2f}", f"${i['price']:.2f}",
            f"{c_margin}{margin:.1f}%{RST}",
            i["weekly_qty"],
            f"${w_cm:,.0f}",
        ))
    print(row_sep())
    print(fmt.format("TOTALS", "", "", "", "", sum(i["weekly_qty"] for i in items), f"${total_cm:,.0f}"))
    print(f"\n  Weekly Revenue: {BOLD}${total_rev:,.2f}{RST}  │  "
          f"Weekly CM: {BOLD}${total_cm:,.2f}{RST}  │  "
          f"Avg Food-Cost Ratio: {BOLD}{(1 - total_cm / total_rev) * 100:.1f}%{RST}")


def print_optimisation(items: list[dict], fixed: float) -> tuple:
    print(section("⚡  PRICE OPTIMISATION — Constrained Nonlinear Programming"))

    curr_prices = np.array([i["price"] for i in items])
    curr_profit = total_weekly_profit(curr_prices, items, fixed)

    print(f"\n  {DIM}Running L-BFGS-B solver…{RST}", end="", flush=True)
    result_lbfgs, opt_profit_lbfgs = run_optimisation(items, fixed)
    print(f"  ✓  (converged={result_lbfgs.success}, iters={result_lbfgs.nit})")

    print(f"  {DIM}Running Differential Evolution (global)…{RST}", end="", flush=True)
    result_de, opt_profit_de = run_global_optimisation(items, fixed)
    print(f"  ✓  (iters={result_de.nit})")

    # Pick the better result
    if opt_profit_de >= opt_profit_lbfgs:
        best_prices = result_de.x
        best_profit = opt_profit_de
        method_used = "Differential Evolution"
    else:
        best_prices = result_lbfgs.x
        best_profit = opt_profit_lbfgs
        method_used = "L-BFGS-B"

    gain_wk = best_profit - curr_profit
    gain_yr = gain_wk * 52

    print(f"\n  {'Current weekly profit':<30} {BOLD}${curr_profit:>10,.2f}{RST}")
    print(f"  {'Optimised weekly profit':<30} {BOLD}{GRN}${best_profit:>10,.2f}{RST}")
    print(f"  {'Weekly improvement':<30} {clr(gain_wk, f'${gain_wk:>+10,.2f}')}")
    print(f"  {'Annual improvement':<30} {clr(gain_yr, f'${gain_yr:>+10,.0f}')}")
    print(f"  {'Best method':<30} {method_used}")

    print(f"\n  {row_sep()}")
    fmt = f"  {{:<23}} {{:>10}} {{:>10}} {{:>8}} {{:>11}} {{:>11}} {{:>11}}"
    print(fmt.format("Item", "Curr $", "Opt $", "Δ %", "Curr CM/wk", "Opt CM/wk", "Δ CM/wk"))
    print(row_sep())

    for i, item in enumerate(items):
        p_c  = item["price"]
        p_o  = best_prices[i]
        d_pct = (p_o - p_c) / p_c * 100
        cm_c = item_contribution_margin(p_c, item)
        cm_o = item_contribution_margin(p_o, item)
        d_cm = cm_o - cm_c
        print(fmt.format(
            item["name"],
            f"${p_c:.2f}",
            f"{GRN if p_o > p_c else YLW}${p_o:.2f}{RST}",
            clr(d_pct, f"{d_pct:+.1f}%"),
            f"${cm_c:,.2f}",
            f"${cm_o:,.2f}",
            clr(d_cm, f"${d_cm:+,.2f}"),
        ))
    print(row_sep())
    return best_prices, best_profit


def print_sensitivity(items: list[dict], fixed: float) -> list:
    print(section("📊  SENSITIVITY ANALYSIS — Per-Item Price Elasticity"))

    tornado = tornado_data(items)

    for item in items:
        sens  = sensitivity_single(item)
        lerner = 1 / abs(item["elasticity"])          # Lerner index = 1/|ε|
        print(f"\n  {BOLD}{item['name']}{RST}  "
              f"{DIM}(ε = {item['elasticity']}, Lerner index = {lerner:.2f}, "
              f"base CM = ${item['price'] - item['cost']:.2f}/unit){RST}")
        fmt = f"  {{:>6}}  {{:>9}}  {{:>9}}  {{:>10}}  {{:>10}}  {{:>12}}  {{:>8}}"
        print(fmt.format("Δ Price", "New $", "Qty/wk", "Revenue", "Wkly CM", "Δ CM", "Margin%"))
        print(f"  {row_sep(74)}")
        for s in sens:
            tag   = f"  {YLW}◀ CURRENT{RST}" if s["pct_change"] == 0 else ""
            d_str = clr(s["delta_cm"], f"${s['delta_cm']:>+,.2f}")
            print(fmt.format(
                f"{s['pct_change']:>+.0f}%",
                f"${s['price']:.2f}",
                f"{s['quantity']:,.1f}",
                f"${s['revenue']:,.2f}",
                f"${s['weekly_cm']:,.2f}",
                d_str,
                f"{s['margin_pct']:.1f}%",
            ) + tag)

    # Global uniform price sweep
    print(f"\n  {BOLD}Global Uniform Price Sweep{RST}  {DIM}(all items moved by same %){RST}")
    g_sens = sensitivity_all(items, fixed)
    fmt = f"  {{:>8}}  {{:>14}}  {{:>14}}  {{:>14}}"
    print(fmt.format("Δ All Prices", "Total Profit", "Δ Profit", "Total Revenue"))
    print(f"  {row_sep(60)}")
    for s in g_sens:
        tag = f"  {YLW}◀ CURRENT{RST}" if s["pct_change"] == 0 else ""
        print(fmt.format(
            f"{s['pct_change']:>+.0f}%",
            clr(s["total_profit"], f"${s['total_profit']:,.2f}"),
            clr(s["delta_profit"], f"${s['delta_profit']:>+,.2f}"),
            f"${s['total_revenue']:,.2f}",
        ) + tag)

    # Tornado chart (text)
    print(f"\n  {BOLD}🌪  Tornado Chart — CM Sensitivity (−20 % → +20 %){RST}")
    print(f"  Items ranked by profit swing (most sensitive first):\n")
    max_swing = tornado[0]["swing"] if tornado else 1
    for t in tornado:
        bar_len  = int(t["swing"] / max_swing * 36)
        bar_fill = "█" * bar_len + "░" * (36 - bar_len)
        low_s    = clr(t["low"],  f"${t['low']:>+8,.2f}")
        high_s   = clr(t["high"], f"${t['high']:>+8,.2f}")
        print(f"  {t['name']:<23} {low_s}  {DIM}{bar_fill}{RST}  {high_s}")

    return tornado


def print_menu_engineering(items: list[dict]) -> list:
    print(section("🗺   MENU ENGINEERING — BCG Profitability Matrix"))
    enriched, avg_cm, avg_qty = menu_engineering(items)

    print(f"\n  Thresholds:  Avg CM per unit = ${avg_cm:.2f}  │  "
          f"Avg weekly qty = {avg_qty:.0f}\n")
    fmt = f"  {{:<23}} {{:<12}} {{:>7}} {{:>8}} {{:>9}} {{:<14}}"
    print(fmt.format("Item", "Category", "CM", "Margin%", "Qty/wk", "Quadrant"))
    print(row_sep())
    for it in sorted(enriched, key=lambda x: x["quadrant"]):
        icon = QUADRANT_ICON[it["quadrant"]]
        c    = (GRN if it["quadrant"] == "Star" else
                YLW if it["quadrant"] == "Puzzle" else
                YLW if it["quadrant"] == "Plow Horse" else RED)
        print(fmt.format(
            it["name"], it["category"],
            f"${it['cm']:.2f}", f"{it['cm_pct']:.1f}%",
            it["weekly_qty"],
            f"{c}{icon} {it['quadrant']}{RST}",
        ))
    print(row_sep())
    print(f"\n  {BOLD}Action Guide:{RST}")
    for q, strat in QUADRANT_STRATEGY.items():
        print(f"  {QUADRANT_ICON[q]}  {q:<12} → {strat}")
    return enriched


def print_breakeven(items: list[dict], fixed: float) -> None:
    print(section("📉  BREAKEVEN & SAFETY MARGIN"))
    b = breakeven_analysis(items, fixed)

    net_s     = clr(b['net_profit'],    f"${b['net_profit']:>10,.2f}")
    month_s   = clr(b['monthly_profit'], f"${b['monthly_profit']:>10,.2f}")
    annual_s  = clr(b['annual_profit'],  f"${b['annual_profit']:>10,.0f}")
    surplus_s = clr(b['surplus_covers'], f"{b['surplus_covers']:>+10.0f} covers")
    print(f"\n  {'Weekly Fixed Costs':<32} ${b['fixed_costs']:>10,.2f}")
    print(f"  {'Weekly Contribution Margin':<32} ${b['total_weekly_cm']:>10,.2f}")
    print(f"  {'Weekly Net Profit':<32} {net_s}")
    print(f"  {'Monthly Net Profit (x4.33)':<32} {month_s}")
    print(f"  {'Annual Net Profit (x52)':<32}  {annual_s}")
    print()
    print(f"  {'Breakeven Covers / Week':<32} {b['breakeven_covers']:>10.0f}")
    print(f"  {'Current Covers / Week':<32} {b['total_weekly_covers']:>10}")
    print(f"  {'Surplus Above Breakeven':<32} {surplus_s}")
    print(f"  {'Avg CM per Cover':<32} ${b['avg_cm_per_cover']:>10.2f}")
    print()
    print(f"  Operating Safety Margin: [{pct_bar(b['safety_margin_pct'])}]")
    if b["safety_margin_pct"] < 10:
        print(f"\n  {RED}{BOLD}⚠  DANGER ZONE{RST}{RED}: safety margin < 10 % — review costs immediately{RST}")
    elif b["safety_margin_pct"] < 20:
        print(f"\n  {YLW}⚡  Caution: safety margin < 20 % — limited buffer for demand downturns{RST}")
    else:
        print(f"\n  {GRN}✓  Healthy safety margin — business is resilient to short-term demand dips{RST}")


def print_recommendations(items: list[dict], fixed: float,
                           opt_prices: np.ndarray,
                           enriched: list[dict]) -> None:
    print(section("💡  RECOMMENDATIONS"))
    recs = []

    b = breakeven_analysis(items, fixed)

    # Safety margin warnings
    if b["safety_margin_pct"] < 10:
        recs.append((RED, "CRITICAL", "Safety margin < 10 %. Raise prices or cut costs urgently."))
    elif b["safety_margin_pct"] < 20:
        recs.append((YLW, "WARNING",  "Safety margin < 20 %. Identify 2–3 items to reprice upwards."))
    else:
        recs.append((GRN, "HEALTHY",  f"Safety margin {b['safety_margin_pct']:.1f}% — maintain discipline."))

    for i, item in enumerate(enriched):
        opt_p      = opt_prices[i]
        curr_p     = item["price"]
        chg_pct    = (opt_p - curr_p) / curr_p * 100
        lerner     = 1 / abs(item["elasticity"])

        if item["quadrant"] == "Dog":
            recs.append((RED,  item["name"],
                         f"Dog item (low margin {item['cm_pct']:.0f}%, low volume). "
                         f"Remove, bundle with a Star, or reprice to ${opt_p:.2f}."))

        elif item["quadrant"] == "Plow Horse" and chg_pct > 2:
            recs.append((YLW, item["name"],
                         f"High volume but thin margin. Raise price to ${opt_p:.2f} "
                         f"(+{chg_pct:.1f}%) — demand inelastic enough (ε={item['elasticity']})."))

        elif item["quadrant"] == "Puzzle":
            recs.append((YLW, item["name"],
                         f"Great margin ({item['cm_pct']:.0f}%) but low volume. "
                         f"Add to specials board / train staff to upsell actively."))

        elif item["quadrant"] == "Star":
            recs.append((GRN, item["name"],
                         f"Top performer. Protect quality & price. "
                         f"Consider a premium variant at higher price point."))

        if item["elasticity"] > -0.8:
            recs.append(("", item["name"],
                         f"Inelastic demand (ε={item['elasticity']}). "
                         f"Lerner markup = {lerner:.0%}. Can absorb a price rise with minimal volume loss."))

    # Annual uplift call-out
    curr_profit = total_weekly_profit(
        np.array([i["price"] for i in items]), items, fixed
    )
    opt_profit = total_weekly_profit(opt_prices, items, fixed)
    annual_gain = (opt_profit - curr_profit) * 52

    recs.append((GRN, "OPPORTUNITY",
                 f"Applying all optimised prices could add ~${annual_gain:,.0f} / year "
                 f"(${(opt_profit - curr_profit):,.0f}/wk)."))

    for color, label, msg in recs:
        print(f"\n  {color}{BOLD}{label}{RST}")
        # word-wrap at ~70 chars
        words = msg.split()
        line = "    "
        for w in words:
            if len(line) + len(w) > 78:
                print(line)
                line = "    " + w + " "
            else:
                line += w + " "
        print(line.rstrip())


# ═══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

CHART_COLORS = {
    "Star"      : "#c9a14c",
    "Puzzle"    : "#5b8df0",
    "Plow Horse": "#3dba82",
    "Dog"       : "#e05555",
    "bar_pos"   : "#3dba82",
    "bar_neg"   : "#e05555",
    "line"      : "#c9a14c",
    "bg"        : "#0f0f1a",
    "surface"   : "#161622",
    "text"      : "#dde0f0",
    "grid"      : "#222235",
}

def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(CHART_COLORS["surface"])
    ax.tick_params(colors=CHART_COLORS["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_COLORS["grid"])
    ax.yaxis.label.set_color(CHART_COLORS["text"])
    ax.xaxis.label.set_color(CHART_COLORS["text"])
    ax.grid(color=CHART_COLORS["grid"], linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=CHART_COLORS["text"], fontsize=10, pad=8)


def plot_dashboard(items: list[dict], fixed: float,
                   opt_prices: np.ndarray, enriched: list[dict]) -> str | None:
    if not HAS_MPL:
        return None

    fig = plt.figure(figsize=(18, 12), facecolor=CHART_COLORS["bg"])
    fig.suptitle("MenuProfit — Restaurant Pricing Dashboard",
                 color=CHART_COLORS["line"], fontsize=14, fontweight="bold", y=0.98)
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)

    names   = [i["name"] for i in items]
    cms     = [i["price"] - i["cost"] for i in items]
    margins = [(i["price"] - i["cost"]) / i["price"] * 100 for i in items]
    qtys    = [i["weekly_qty"] for i in items]
    w_cms   = [c * q for c, q in zip(cms, qtys)]

    # ── 1. Weekly Contribution Margin bar ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    colors1 = [CHART_COLORS["bar_pos"] if c >= 0 else CHART_COLORS["bar_neg"] for c in w_cms]
    bars = ax1.bar(names, w_cms, color=colors1, alpha=0.85, edgecolor=CHART_COLORS["bg"], linewidth=0.5)
    for bar, val in zip(bars, w_cms):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 f"${val:,.0f}", ha="center", va="bottom",
                 color=CHART_COLORS["text"], fontsize=7)
    ax1.tick_params(axis="x", rotation=30, labelsize=7)
    _style_ax(ax1, "Weekly Contribution Margin by Item ($)")
    ax1.set_ylabel("$ / week")

    # ── 2. Gross Margin % ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    colors2 = [CHART_COLORS["bar_pos"] if m >= 60 else
               CHART_COLORS["line"] if m >= 45 else
               CHART_COLORS["bar_neg"] for m in margins]
    ax2.barh(names, margins, color=colors2, alpha=0.85, edgecolor=CHART_COLORS["bg"])
    ax2.axvline(50, color=CHART_COLORS["line"], linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(51, len(names) - 0.5, "Target 50%", color=CHART_COLORS["line"], fontsize=7)
    _style_ax(ax2, "Gross Margin % per Item")
    ax2.set_xlabel("Margin %")

    # ── 3. Sensitivity — uniform price sweep ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    g_sens = sensitivity_all(items, fixed)
    pcts    = [s["pct_change"] for s in g_sens]
    profits = [s["total_profit"] for s in g_sens]
    line_c  = [CHART_COLORS["bar_pos"] if p >= 0 else CHART_COLORS["bar_neg"] for p in profits]
    ax3.plot(pcts, profits, color=CHART_COLORS["line"], linewidth=2, zorder=3)
    ax3.scatter(pcts, profits, c=line_c, s=40, zorder=4)
    ax3.axhline(0, color=CHART_COLORS["grid"], linewidth=1)
    ax3.axvline(0, color=CHART_COLORS["text"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.fill_between(pcts, profits, 0,
                     where=[p >= 0 for p in profits],
                     alpha=0.15, color=CHART_COLORS["bar_pos"])
    ax3.fill_between(pcts, profits, 0,
                     where=[p < 0 for p in profits],
                     alpha=0.15, color=CHART_COLORS["bar_neg"])
    _style_ax(ax3, "Sensitivity: Total Weekly Profit vs Uniform Price Change")
    ax3.set_xlabel("Price Change (%)")
    ax3.set_ylabel("Weekly Profit ($)")

    # ── 4. Tornado chart ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    tornado = tornado_data(items)
    t_names = [t["name"][:14] for t in tornado]
    t_swings = [t["swing"] for t in tornado]
    bars4 = ax4.barh(t_names, t_swings, color=CHART_COLORS["line"], alpha=0.8)
    _style_ax(ax4, "Tornado: CM Swing −20%→+20%")
    ax4.set_xlabel("CM swing $ / week")

    # ── 5. Menu Engineering Scatter ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    _, avg_cm, avg_qty = menu_engineering(items)
    for it in enriched:
        col = CHART_COLORS[it["quadrant"]]
        ax5.scatter(it["weekly_qty"], it["cm"], s=120, color=col, alpha=0.9, zorder=3)
        ax5.annotate(it["name"][:14], (it["weekly_qty"], it["cm"]),
                     textcoords="offset points", xytext=(6, 4),
                     color=CHART_COLORS["text"], fontsize=7)
    ax5.axhline(avg_cm,  color=CHART_COLORS["grid"], linestyle="--", linewidth=1)
    ax5.axvline(avg_qty, color=CHART_COLORS["grid"], linestyle="--", linewidth=1)
    patches = [mpatches.Patch(color=CHART_COLORS[q], label=f"{QUADRANT_ICON[q]} {q}")
               for q in ["Star", "Puzzle", "Plow Horse", "Dog"]]
    ax5.legend(handles=patches, fontsize=7,
               facecolor=CHART_COLORS["surface"], labelcolor=CHART_COLORS["text"],
               edgecolor=CHART_COLORS["grid"])
    _style_ax(ax5, "Menu Engineering Matrix (BCG)")
    ax5.set_xlabel("Weekly Qty (popularity)")
    ax5.set_ylabel("Contribution Margin per Unit ($)")

    # ── 6. Current vs Optimised profit per item ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    curr_cms = [item_contribution_margin(i["price"], i) for i in items]
    opt_cms  = [item_contribution_margin(opt_prices[j], items[j]) for j in range(len(items))]
    x = np.arange(len(items))
    ax6.bar(x - 0.2, curr_cms, 0.38, label="Current", color=CHART_COLORS["grid"],   alpha=0.9)
    ax6.bar(x + 0.2, opt_cms,  0.38, label="Optimised", color=CHART_COLORS["line"], alpha=0.9)
    ax6.set_xticks(x)
    ax6.set_xticklabels([i["name"][:8] for i in items], rotation=35, fontsize=6)
    ax6.legend(fontsize=7, facecolor=CHART_COLORS["surface"],
               labelcolor=CHART_COLORS["text"], edgecolor=CHART_COLORS["grid"])
    _style_ax(ax6, "Current vs Optimised CM/wk ($)")
    ax6.set_ylabel("$ / week")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REPORT_DIR, f"menuprofit_dashboard_{ts}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=CHART_COLORS["bg"])
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_json_report(items, fixed, opt_prices, enriched, be) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORT_DIR, f"menuprofit_report_{ts}.json")

    curr_profit = total_weekly_profit(np.array([i["price"] for i in items]), items, fixed)
    opt_profit  = total_weekly_profit(opt_prices, items, fixed)

    payload = {
        "generated_at"           : datetime.now().isoformat(),
        "weekly_fixed_costs"     : fixed,
        "current_weekly_profit"  : round(curr_profit, 2),
        "optimised_weekly_profit": round(opt_profit, 2),
        "weekly_improvement"     : round(opt_profit - curr_profit, 2),
        "annual_improvement"     : round((opt_profit - curr_profit) * 52, 2),
        "breakeven"              : {k: round(v, 2) if isinstance(v, float) else v
                                    for k, v in be.items()},
        "items": [
            {
                "id"             : item["id"],
                "name"           : item["name"],
                "category"       : item["category"],
                "cost"           : item["cost"],
                "current_price"  : item["price"],
                "optimal_price"  : round(float(opt_prices[j]) * 4) / 4,
                "weekly_qty"     : item["weekly_qty"],
                "elasticity"     : item["elasticity"],
                "cm_per_unit"    : round(item["price"] - item["cost"], 2),
                "margin_pct"     : round((item["price"] - item["cost"]) / item["price"] * 100, 1),
                "weekly_cm_curr" : round(item_contribution_margin(item["price"], item), 2),
                "weekly_cm_opt"  : round(item_contribution_margin(opt_prices[j], item), 2),
                "quadrant"       : enriched[j]["quadrant"],
            }
            for j, item in enumerate(items)
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(banner("MENUPROFIT — RESTAURANT PRICING INTELLIGENCE ENGINE"))
    print(f"\n  {DIM}{datetime.now().strftime('%A %d %B %Y  %H:%M')}"
          f"  │  Items: {len(DEFAULT_MENU)}"
          f"  │  Weekly Fixed Costs: ${WEEKLY_FIXED_COSTS:,.0f}{RST}\n")

    items = DEFAULT_MENU

    # ── Sections ──────────────────────────────────────────────────────────────
    print_menu_overview(items)
    opt_prices, opt_profit = print_optimisation(items, WEEKLY_FIXED_COSTS)
    tornado               = print_sensitivity(items, WEEKLY_FIXED_COSTS)
    enriched, *_          = menu_engineering(items)
    print_menu_engineering(items)
    be = breakeven_analysis(items, WEEKLY_FIXED_COSTS)
    print_breakeven(items, WEEKLY_FIXED_COSTS)
    print_recommendations(items, WEEKLY_FIXED_COSTS, opt_prices, enriched)

    # ── Save outputs ──────────────────────────────────────────────────────────
    json_path  = save_json_report(items, WEEKLY_FIXED_COSTS, opt_prices, enriched, be)
    chart_path = plot_dashboard(items, WEEKLY_FIXED_COSTS, opt_prices, enriched)

    print(f"\n{'═' * W}")
    print(f"  {GRN}{BOLD}✅  Analysis complete{RST}")
    print(f"  📄  JSON report  →  {json_path}")
    if chart_path:
        print(f"  📊  Dashboard    →  {chart_path}")
    else:
        print(f"  {YLW}📊  Charts skipped (install matplotlib){RST}")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
