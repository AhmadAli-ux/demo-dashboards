"""E-commerce Product Performance Dashboard — DEMO built with sample data.

Bestsellers, category share, returns analysis and rating-vs-sales view.
Built with Python, pandas, NumPy, Plotly and Streamlit.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Product Performance Dashboard — Demo", layout="wide")


@st.cache_data
def load_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    products = [
        ("Linen Midi Dress", "Dresses"), ("Oversized Blazer", "Outerwear"),
        ("Ribbed Knit Top", "Tops"), ("Pleated Skirt", "Skirts"),
        ("Wide-Leg Trousers", "Bottoms"), ("Silk Scarf", "Accessories"),
        ("Denim Jacket", "Outerwear"), ("Satin Slip Dress", "Dresses"),
        ("Cotton Poplin Shirt", "Tops"), ("Wool Tote Bag", "Accessories"),
        ("Maxi Summer Dress", "Dresses"), ("Cropped Cardigan", "Tops"),
    ]
    n = len(products)
    units = rng.integers(40, 900, n)
    price = np.round(rng.uniform(18, 140, n), 2)
    revenue = (units * price).round(2)
    return_rate = np.clip(rng.normal(0.08, 0.05, n), 0.01, 0.28).round(3)
    rating = np.clip(rng.normal(4.4, 0.35, n), 3.2, 5.0).round(2)
    reviews = rng.integers(8, 420, n)
    df = pd.DataFrame({
        "product": [p[0] for p in products],
        "category": [p[1] for p in products],
        "units_sold": units,
        "price": price,
        "revenue": revenue,
        "return_rate": return_rate,
        "rating": rating,
        "reviews": reviews,
    }).sort_values("revenue", ascending=False).reset_index(drop=True)
    return df


df = load_data()

st.title("🛍️ Product Performance Dashboard")
st.caption("DEMO — built with sample data to show what this looks like on your real store data.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue (90d)", f"£{df['revenue'].sum():,.0f}")
k2.metric("Units sold", f"{df['units_sold'].sum():,}")
k3.metric("Avg return rate", f"{(df['return_rate'] * df['units_sold']).sum() / df['units_sold'].sum():.1%}")
k4.metric("Avg rating", f"{np.average(df['rating'], weights=df['reviews']):.2f} ★")

st.divider()
c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Revenue by product")
    fig = px.bar(df, x="revenue", y="product", orientation="h", color="category",
                 labels={"revenue": "Revenue (£)", "product": ""},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Revenue share by category")
    cat = df.groupby("category")["revenue"].sum().reset_index()
    fig2 = px.pie(cat, values="revenue", names="category", hole=0.45,
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Rating vs sales")
    fig3 = px.scatter(df, x="rating", y="units_sold", size="reviews",
                      hover_name="product", color="return_rate",
                      color_continuous_scale=["#2ca02c", "#F0B429", "#d62728"],
                      labels={"rating": "Rating ★", "units_sold": "Units sold",
                              "return_rate": "Return rate"})
    fig3.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Returns spotlight — where margin leaks")
ret = df.nlargest(5, "return_rate")[["product", "category", "units_sold", "return_rate", "rating"]]
ret["return_rate"] = (ret["return_rate"] * 100).round(1).astype(str) + "%"
st.dataframe(ret, use_container_width=True, hide_index=True)
worst = df.loc[df["return_rate"].idxmax()]
st.warning(f"⚠️ **{worst['product']}** has the highest return rate "
           f"({worst['return_rate']:.1%}) at a {worst['rating']}★ rating — worth checking sizing/fit notes in reviews.")

st.divider()
st.info("**What the real version includes:** connected to your store, automatic bestseller/decliner alerts, "
        "size-curve analysis, and review-text mining to explain *why* products get returned.")
st.caption("Demo dashboard by Ahmad Ali — Data Scientist & ML Engineer · Python · scikit-learn · Streamlit")
