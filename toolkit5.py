# streamlit_ai_suite.py — ERP vs AI/ML + PED (navy/gold, PED badge, profit-max toggle)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

# ---------- Page + Global CSS (full-width) ----------
st.set_page_config(page_title="Real Analytics 101 — AI Toolkit", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 96vw; padding-top: 1rem; padding-bottom: 1rem;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Brand palette ----------
NAVY = "#0A1A2F"
GOLD = "#C9A038"
LIGHT = "#F5F5F5"

# ---------- Styles ----------
st.markdown(f"""
<style>
  .ra-card {{
    border: 2px solid {NAVY};
    border-radius: 16px;
    padding: 16px;
    background: #fff;
  }}
  .ra-title {{ color: {NAVY}; font-size: 22px; font-weight: 800; margin: 0; }}
  .ra-sub   {{ color: #333; font-size: 14px; margin: 4px 0 0 0; }}
  .stTabs [data-baseweb="tab-list"] > div[data-baseweb="tab"] {{ font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(
    f'<div class="ra-card"><div class="ra-title">Real Analytics 101 — AI/ML Add-Ons for ERP</div>'
    f'<div class="ra-sub">ERP vs AI/ML Comparison • Price Elasticity (PED)</div></div>',
    unsafe_allow_html=True,
)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["ERP vs AI/ML", "Price Elasticity (PED)"])

# ======== Tab 1: ERP vs AI/ML ========
with tab1:
    rows = [
        ("Inventory Management",
         "You have 1,200 units of Product X in stock.",
         "You’ll sell out of Product X in ~9 days based on sales velocity & seasonal uplift. Reorder ~800 now to avoid ±R120k revenue loss."),
        ("Pricing",
         "Product Y sold 500 units last month.",
         "A +7% price move is forecast to cut demand ~2% — net +R18k profit this month."),
        ("Sales Forecasting",
         "Last quarter sales were R1.5M.",
         "Next quarter forecast R1.62M (≈82% confidence). A 10% promo on slow movers could lift to ~R1.75M."),
        ("Supplier Performance",
         "Supplier A delivered 95% on time.",
         "On-time fell 8% in 3 months — risk of stockouts in peak. Shift ~30% of volume to Supplier B."),
        ("Customer Insights",
         "Customer Z bought 5 times in a year.",
         "Churn risk ~65% in 90 days — send targeted R50 voucher + cross-sell bundle."),
        ("Whitespace / New Opportunities",
         "ERP shows current SKUs only.",
         "Add 3 complements to top sellers — modeled +R250k/yr with minimal marketing."),
        ("Cash Flow",
         "Outstanding invoices: R500k.",
         "Collect top 10 debtors 10 days earlier → free ~R150k working capital.")
    ]

    # Build table HTML (wrapping, fixed column fractions)
    table_rows_html = []
    for area, erp, ai in rows:
        table_rows_html.append(
            "<tr>"
            f"<td class='area'>{area}</td>"
            f"<td class='erp'>{erp}</td>"
            f"<td class='ai'>{ai}</td>"
            "</tr>"
        )
    table_html = "\n".join(table_rows_html)

    html = f"""
    <div class='wrap'>
      <table role='table' aria-label='ERP vs AI/ML'>
        <thead>
          <tr>
            <th>Area</th>
            <th>Typical ERP Output (Static / Historical)</th>
            <th>AI/ML-Driven Insight (Dynamic / Predictive)</th>
          </tr>
        </thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
      <div class='brand'>© Real Analytics 101 • promotions@realanalytics101.co.za</div>
    </div>
    <style>
      :root {{ --navy:{NAVY}; --gold:{GOLD}; --light:{LIGHT}; }}
      .wrap {{ border:2px solid var(--navy); border-radius:16px; overflow:hidden; }}
      table {{ width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; }}
      thead th {{ background:var(--navy); color:#fff; text-align:left; padding:12px; font-weight:700; }}
      tbody td {{ padding:12px; vertical-align:top; white-space:normal; overflow-wrap:anywhere; word-break:break-word; }}
      td.area {{ font-weight:700; background:#fafafa; width:17%; }}
      td.erp  {{ background:var(--light); width:33%; }}
      td.ai   {{ background:var(--gold);  width:50%; }}
      .brand {{ text-align:right; color:#333; padding:8px 6px; background:#fff; }}
    </style>
    """

    row_height = 78
    header_height = 60
    margin = 120
    adaptive_height = header_height + row_height * len(rows) + margin

    st.components.v1.html(html, height=adaptive_height, scrolling=True)

# ======== Tab 2: Price Elasticity (PED) ========
with tab2:
    st.subheader("Price Elasticity (PED) — Simulator")

    # Plain-English explainer (elastic vs inelastic) with examples
    st.markdown(
f"""
**PED in plain English**  
Price Elasticity of Demand measures **how sensitive demand is to price**.

- **Elastic products** (|PED| > 1): demand changes **a lot** when price moves.  
  *Examples:* restaurant meals, fashion, electronics accessories.  
  *If PED = -1.5 and you raise price 10% → units drop ≈ 15%.*  

- **Inelastic products** (|PED| < 1): demand changes **a little** when price moves.  
  *Examples:* basic groceries, utilities, essential meds, cigarettes.  
  *If PED = -0.3 and you raise price 10% → units drop ≈ 3%.*  

We estimate PED from your historical **price** & **quantity** pairs using a simple log-log regression.
"""
    )

    st.write("**Upload data (optional):** CSV with columns `price`, `qty` for a single SKU. If blank, we use demo data.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="ped_csv")
    if up:
        df = pd.read_csv(up)
    else:
        rng = np.random.default_rng(42)
        price = np.linspace(10, 40, 20)
        qty = (1200 - 20*price) + rng.normal(0, 30, size=price.shape[0])
        df = pd.DataFrame({"price": price, "qty": np.maximum(qty, 1)})

    st.dataframe(df.head(), use_container_width=True, height=260)

    # Fit elasticity (log-log)
    X = np.log(df[["price"]].values)
    y = np.log(df["qty"].values)
    model = LinearRegression().fit(X, y)
    elasticity = float(model.coef_[0])
    st.markdown(f"**Estimated Price Elasticity:** **{elasticity:.2f}**  (negative means demand falls as price rises)")

    # Category badge
    abs_elasticity = abs(elasticity)
    if abs_elasticity > 1:
        category = "Elastic — demand changes a lot when price moves."
    elif abs_elasticity < 1:
        category = "Inelastic — demand changes little when price moves."
    else:
        category = "Unit Elastic — demand changes proportionally to price."
    st.markdown(
        f"<div style='background-color:{GOLD};padding:6px 10px;border-radius:8px;display:inline-block;"
        f"color:white;font-weight:bold;'>{category}</div>",
        unsafe_allow_html=True
    )

    # Scenario metrics
    base_price = float(df["price"].mean())
    base_qty = float(np.exp(model.predict([[np.log(base_price)]]))[0])
    base_rev = base_price * base_qty

    pct_change = st.slider("Proposed price change (%)", min_value=-30, max_value=30, value=5, step=1)
    new_price = base_price * (1 + pct_change/100.0)
    new_qty = float(np.exp(model.predict([[np.log(new_price)]]))[0])
    new_rev = new_price * new_qty
    rev_diff = new_rev - base_rev

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Price", f"R{base_price:.2f}")
    c2.metric("Projected Price", f"R{new_price:.2f}", f"{pct_change}%")
    c3.metric("Projected Qty", f"{new_qty:.0f}", f"Δ {new_qty-base_qty:.0f}")
    c4.metric("Revenue Impact", f"R{new_rev:.0f}", f"R{rev_diff:.0f}")

    # ----- Profit-maximising price (optional toggle, needs unit cost) -----
    with st.expander("Show profit-maximising price (requires unit cost)"):
        st.caption(
            "Uses constant-elasticity (Lerner) rule: (P − c)/P = 1/|PED|. "
            "A finite optimum exists only when |PED| > 1."
        )
        default_cost = round(base_price * 0.6, 2) if base_price > 0 else 1.0
        unit_cost = st.number_input("Unit cost (R)", min_value=0.0, value=default_cost, step=0.1)

        if abs_elasticity > 1 and unit_cost >= 0:
            # P* = c / (1 - 1/|ε|) ; valid only for |ε|>1
            p_star = unit_cost / (1.0 - 1.0/abs_elasticity)
            if p_star > 0:
                q_star = float(np.exp(model.predict([[np.log(p_star)]]))[0])
                profit_star = (p_star - unit_cost) * q_star
                s1, s2, s3 = st.columns(3)
                s1.metric("Profit-maximising price (P*)", f"R{p_star:.2f}")
                s2.metric("Qty at P*", f"{q_star:.0f}")
                s3.metric("Profit at P*", f"R{profit_star:,.0f}")
            else:
                st.info("Computed P* is not positive with current inputs. Adjust unit cost or review data.")
        else:
            st.info("For |PED| ≤ 1, the constant-elasticity model does not yield a finite P*. "
                    "In practice, apply business constraints (caps, competition, budget).")

    # Branded chart (navy & gold)
    fig = plt.figure(figsize=(10, 4.8))
    p_grid = np.linspace(df["price"].min()*0.8, df["price"].max()*1.2, 100)
    q_grid = np.exp(model.predict(np.log(p_grid).reshape(-1, 1)))

    plt.plot(df["price"], df["qty"], 'o', label="Observed", color=GOLD, markersize=6)
    plt.plot(p_grid, q_grid, label="Fitted (log-log)", linewidth=2.5, color=NAVY)
    plt.axvline(base_price, linestyle="--", label="Base price", color=NAVY, alpha=0.75)
    plt.axvline(new_price, linestyle="--", label="New price", color=GOLD, alpha=0.85)

    # If we computed P*, add it to the chart (best-effort)
    try:
        if 'p_star' in locals() and p_star > 0:
            plt.axvline(p_star, linestyle=":", linewidth=2.5, color="#8B6A00", label="Profit-max price (P*)")  # darker gold
    except Exception:
        pass

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4, color=NAVY)
    plt.xlabel("Price")
    plt.ylabel("Quantity")
    plt.legend(frameon=False)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:right; color:#333;'>© Real Analytics 101 • "
    f"<a href='mailto:promotions@realanalytics101.co.za' style='color:{NAVY};text-decoration:none;'>promotions@realanalytics101.co.za</a></div>",
    unsafe_allow_html=True
)
