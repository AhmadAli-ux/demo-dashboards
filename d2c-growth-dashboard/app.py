"""D2C Brand Growth Dashboard — DEMO built with sample data.

Shows what a real client dashboard looks like: revenue trend + forecast,
ad-spend efficiency by channel, and inventory health. Built with Python,
pandas, NumPy, Plotly and Streamlit.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="D2C Growth Dashboard — Demo", layout="wide")

# ---------------------------------------------------------------- data ----
@st.cache_data
def load_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    days = pd.date_range(end=pd.Timestamp.today().normalize(), periods=120, freq="D")

    # revenue with weekly seasonality + upward trend + noise
    trend = np.linspace(900, 1600, 120)
    weekly = 180 * np.sin(2 * np.pi * np.arange(120) / 7)
    revenue = trend + weekly + rng.normal(0, 120, 120)
    revenue = np.clip(revenue, 300, None).round(2)

    # ad spend by channel
    channels = ["Meta Ads", "Google Ads", "TikTok", "Email/SMS"]
    spend = pd.DataFrame({
        "date": np.repeat(days, len(channels)),
        "channel": channels * len(days),
        "spend": np.concatenate([
            np.clip(rng.normal(180, 30, 120), 60, None),   # Meta
            np.clip(rng.normal(120, 25, 120), 40, None),   # Google
            np.clip(rng.normal(70, 20, 120), 20, None),    # TikTok
            np.clip(rng.normal(25, 8, 120), 5, None),      # Email
        ]).round(2),
    })
    # revenue attributed per channel (Email has best ROAS, TikTok lowest)
    roas_base = {"Meta Ads": 4.2, "Google Ads": 5.1, "TikTok": 3.1, "Email/SMS": 9.4}
    spend["attributed_revenue"] = (
        spend["spend"] * spend["channel"].map(roas_base)
        * rng.normal(1, 0.15, len(spend))
    ).round(2)

    # inventory snapshot
    products = ["Hoodie — Charcoal", "Tee — Sand", "Cap — Black",
                "Sweatpants — Navy", "Tee — White", "Hoodie — Olive"]
    inventory = pd.DataFrame({
        "product": products,
        "units_left": rng.integers(15, 900, len(products)),
        "avg_daily_sales": rng.integers(4, 40, len(products)),
    })
    inventory["days_of_stock"] = (inventory["units_left"] / inventory["avg_daily_sales"]).round(1)

    return pd.DataFrame({"date": days, "revenue": revenue}), spend, inventory


def simple_forecast(series: pd.Series, horizon: int = 14) -> pd.DataFrame:
    """Naive-but-honest forecast: 7-day rolling mean extended forward.
    Labelled illustrative — a real build would train on the client's data."""
    last = series.rolling(7).mean().iloc[-1]
    slope = (series.tail(14).mean() - series.tail(42).mean()) / 28
    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1),
                               periods=horizon, freq="D")
    values = last + slope * np.arange(1, horizon + 1)
    return pd.DataFrame({"date": future_idx, "forecast": values.round(2)})


# ------------------------------------------------------------------ ui ----
daily, spend, inventory = load_data()

st.title("📈 D2C Brand Growth Dashboard")
st.caption("DEMO — built with sample data to show what this looks like on your real store data.")

k1, k2, k3, k4 = st.columns(4)
last30 = daily.tail(30)
k1.metric("Revenue (30d)", f"£{last30['revenue'].sum():,.0f}",
          f"{(last30['revenue'].sum() / daily.tail(60).head(30)['revenue'].sum() - 1) * 100:+.1f}% vs prior")
k2.metric("Orders (30d, est.)", f"{int(last30['revenue'].sum() / 42):,}")
spend30 = spend[spend["date"] >= daily["date"].max() - pd.Timedelta(days=29)]
k3.metric("Ad spend (30d)", f"£{spend30['spend'].sum():,.0f}")
roas = spend30["attributed_revenue"].sum() / spend30["spend"].sum()
k4.metric("Blended ROAS", f"{roas:.1f}x")

st.divider()
st.subheader("Revenue trend + 14-day outlook")
hist = daily.set_index("date")["revenue"]
fc = simple_forecast(hist)
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"],
                         name="Daily revenue", line=dict(color="#1B5FCC")))
fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast (illustrative)",
                         line=dict(color="#F0B429", dash="dash")))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis_title="Revenue (£)", legend=dict(orientation="h", y=1.05))
st.plotly_chart(fig, use_container_width=True)
st.caption("Forecast uses a simple trend projection — on your data this becomes a trained model (Prophet/GBM).")

c1, c2 = st.columns(2)
with c1:
    st.subheader("ROAS by channel (30d)")
    ch = (spend30.groupby("channel")[["spend", "attributed_revenue"]].sum()
          .assign(roas=lambda d: d["attributed_revenue"] / d["spend"]).reset_index())
    fig2 = px.bar(ch, x="channel", y="roas", color="roas",
                  color_continuous_scale=["#d62728", "#F0B429", "#2ca02c"],
                  labels={"roas": "ROAS (x)", "channel": ""})
    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    st.subheader("Daily ad spend by channel")
    fig3 = px.area(spend30, x="date", y="spend", color="channel",
                   labels={"spend": "Spend (£)", "date": ""})
    fig3.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Inventory health")
inv = inventory.copy()
inv["status"] = pd.cut(inv["days_of_stock"], [0, 14, 45, 9999],
                       labels=["🔴 Reorder now", "🟡 Watch", "🟢 Healthy"])
st.dataframe(inv[["product", "units_left", "avg_daily_sales", "days_of_stock", "status"]],
             use_container_width=True, hide_index=True)

st.divider()
st.info("**What the real version includes:** connected to your Shopify + ad accounts, daily auto-refresh, "
        "trained demand-forecast model per product, and low-stock alerts before you run out.")
st.caption("Demo dashboard by Ahmad Ali — Data Scientist & ML Engineer · Python · scikit-learn · Streamlit")
