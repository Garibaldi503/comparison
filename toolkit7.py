import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Page + Global CSS ----------
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

# ======== Tab 1: ERP vs AI/ML (dynamic height + robust wrapping) ========
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

    table_rows_html = "\n".join(
        f"<tr><td class='area'>{area}</td><td class='erp'>{erp}</td><td class='ai'>{ai}</td></tr>"
        for area, erp, ai in rows
    )

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
          {table_rows_html}
        </tbody>
      </table>
      <div class='brand'>© Real Analytics 101 • promotions@realanalytics101.co.za</div>
    </div>
    <style>
      :root {{ --navy:{NAVY}; --gold:{GOLD}; --light:{LIGHT}; }}
      .wrap {{ border:2px solid var(--navy); border-radius:16px; overflow:hidden; width:100%; }}
      table {{ width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; }}
      thead th {{
        background:var(--navy); color:#fff; padding:12px; font-weight:700; text-align:left;
        position:sticky; top:0; /* keeps header visible if scrolling */
      }}
      tbody td {{
        padding:12px; vertical-align:top;
        white-space:normal; overflow-wrap:anywhere; word-break:break-word;
      }}
      td.area {{ font-weight:700; background:#fafafa; width:17%; }}
      td.erp  {{ background:var(--light); width:33%; }}
      td.ai   {{ background:var(--gold);  width:50%; }}
      .brand {{ text-align:right; color:#333; padding:8px 6px; background:#fff; }}
    </style>
    """

    # Adaptive height to fit content (no clipping, no “half page”)
    row_height = 92         # generous per-row allowance for wrapped text
    header_height = 64
    margins = 140
    adaptive_height = header_height + row_height * len(rows) + margins

    # Cap to a reasonable max; enable scrolling if very long
    adaptive_height = min(adaptive_height, 1000)

    st.components.v1.html(html, height=adaptive_height, scrolling=True)

# ======== Tab 2: Price Elasticity (PED) ========
with tab2:
    st.subheader("Price Elasticity (PED) — Simulator")
    st.markdown(
f"""
**In plain English**  
Price Elasticity of Demand measures **how sensitive demand is to price**.

- **Elastic** (|PED| > 1): demand changes **a lot** when price moves.  
  *Example:* restaurant meals — PED = -1.5, raise price 10% → units drop ~15%.  
- **Inelastic** (|PED| < 1): demand changes **a little** when price moves.  
  *Example:* bread — PED = -0.3, raise price 10% → units drop ~3%.  
"""
    )

    st.write("Upload CSV with `price` and `qty`, or use demo data.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
    else:
        rng = np.random.default_rng(42)
        price = np.linspace(10, 40, 20)
        qty = (1200 - 20*price) + rng.normal(0, 30, size=price.shape[0])
        df = pd.DataFrame({"price": price, "qty": np.maximum(qty, 1)})

    st.dataframe(df.head(), use_container_width=True)

    # NumPy regression (log-log): ln(q) = a + b ln(p) -> b is elasticity
    ln_p, ln_q = np.log(df["price"]), np.log(df["qty"])
    b, a = np.polyfit(ln_p, ln_q, 1)
    elasticity = float(b)
    st.markdown(f"**Estimated PED:** {elasticity:.2f}")

    # Category badge
    abs_e = abs(elasticity)
    if abs_e > 1:
        category = "Elastic — demand changes a lot"
    elif abs_e < 1:
        category = "Inelastic — demand changes little"
    else:
        category = "Unit Elastic — demand changes proportionally"
    st.markdown(
        f"<div style='background-color:{GOLD};padding:6px 10px;border-radius:8px;color:white;font-weight:bold;'>{category}</div>",
        unsafe_allow_html=True
    )

    def q_hat(p): return float(np.exp(a + elasticity * np.log(p)))

    base_p = float(df["price"].mean())
    base_q = q_hat(base_p)
    base_rev = base_p * base_q

    pct_change = st.slider("Proposed price change (%)", -30, 30, 5)
    new_p = base_p * (1 + pct_change/100)
    new_q = q_hat(new_p)
    new_rev = new_p * new_q
    rev_diff = new_rev - base_rev

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Price", f"R{base_p:.2f}")
    c2.metric("New Price", f"R{new_p:.2f}", f"{pct_change}%")
    c3.metric("New Qty", f"{new_q:.0f}", f"Δ {new_q-base_q:.0f}")
    c4.metric("Revenue Δ", f"R{new_rev:.0f}", f"R{rev_diff:.0f}")

    # Chart
    fig, ax = plt.subplots(figsize=(10,4.8))
    p_grid = np.linspace(df["price"].min()*0.8, df["price"].max()*1.2, 100)
    q_grid = [q_hat(p) for p in p_grid]
    ax.plot(df["price"], df["qty"], 'o', label="Observed", color=GOLD)
    ax.plot(p_grid, q_grid, label="Fitted", color=NAVY, linewidth=2.2)
    ax.axvline(base_p, ls="--", color=NAVY, alpha=0.75, label="Base price")
    ax.axvline(new_p,  ls="--", color=GOLD, alpha=0.85, label="New price")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4, color=NAVY)
    ax.set_xlabel("Price"); ax.set_ylabel("Quantity"); ax.legend(frameon=False)
    st.pyplot(fig, use_container_width=True)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:right; color:#333;'>© Real Analytics 101 • "
    f"<a href='mailto:promotions@realanalytics101.co.za' style='color:{NAVY};text-decoration:none;'>promotions@realanalytics101.co.za</a></div>",
    unsafe_allow_html=True
)
