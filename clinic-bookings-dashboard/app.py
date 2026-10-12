"""Clinic Bookings & No-Show Dashboard — DEMO built with sample data.

Multi-site view: bookings, no-show rates by weekday, chair utilisation and
a simple no-show risk flag per appointment slot. Built with Python, pandas,
NumPy, Plotly and Streamlit.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Clinic Bookings Dashboard — Demo", layout="wide")

SITES = ["Grange Dental — Site A", "Grange Dental — Site B", "Skin Clinic"]


@st.cache_data
def load_data(seed: int = 21):
    rng = np.random.default_rng(seed)
    days = pd.date_range(end=pd.Timestamp.today().normalize(), periods=90, freq="D")

    rows = []
    for site in SITES:
        base = {"Grange Dental — Site A": 46, "Grange Dental — Site B": 38, "Skin Clinic": 22}[site]
        for d in days:
            dow = d.dayofweek
            weekend_dip = 0.35 if dow >= 5 else 1.0
            booked = int(base * weekend_dip * rng.normal(1, 0.12))
            # no-show probability higher on Mondays & Fridays, lower mid-week
            p = 0.06 + 0.05 * (dow in (0, 4)) + rng.normal(0, 0.015)
            p = float(np.clip(p, 0.02, 0.22))
            noshow = rng.binomial(max(booked, 0), p)
            rows.append({"date": d, "site": site, "booked": booked,
                         "no_show": noshow, "weekday": d.strftime("%a")})
    appts = pd.DataFrame(rows)

    # practitioner utilisation snapshot (this week)
    pract = pd.DataFrame({
        "practitioner": ["Dr. A. Khan", "Dr. S. Patel", "Hygienist J. Lee",
                         "Dr. M. Osei", "Aesthetician R. Ali", "Dr. T. Ng"],
        "site": ["Grange Dental — Site A", "Grange Dental — Site A",
                 "Grange Dental — Site B", "Grange Dental — Site B",
                 "Skin Clinic", "Skin Clinic"],
        "slots": [40, 40, 32, 32, 24, 24],
        "filled": [37, 31, 30, 24, 22, 15],
    })
    pract["utilisation"] = (pract["filled"] / pract["slots"]).round(3)

    # upcoming week risk flags (illustrative scoring)
    upcoming = pd.DataFrame({
        "date": pd.date_range(pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=7).strftime("%a %d %b"),
        "site": rng.choice(SITES, 7),
        "patient_type": rng.choice(["New patient", "Recall", "Treatment plan", "Hygiene"], 7),
        "risk": rng.choice(["🟢 Low", "🟡 Medium", "🔴 High"], 7, p=[0.55, 0.3, 0.15]),
    })
    return appts, pract, upcoming


appts, pract, upcoming = load_data()

st.title("🦷 Clinic Bookings & No-Show Dashboard")
st.caption("DEMO — built with sample data to show what this looks like on your real booking data.")

tot_booked = appts["booked"].sum()
tot_noshow = appts["no_show"].sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Appointments (90d)", f"{tot_booked:,}")
k2.metric("No-show rate", f"{tot_noshow / tot_booked:.1%}")
k3.metric("Lost slots (90d)", f"{tot_noshow:,}")
k4.metric("Est. lost revenue", f"£{tot_noshow * 65:,}")

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("No-show rate by weekday")
    wd = (appts.groupby("weekday")[["booked", "no_show"]].sum()
          .assign(rate=lambda d: d["no_show"] / d["booked"]).reset_index())
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wd["weekday"] = pd.Categorical(wd["weekday"], order, ordered=True)
    wd = wd.sort_values("weekday")
    fig = px.bar(wd, x="weekday", y="rate", color="rate",
                 color_continuous_scale=["#2ca02c", "#F0B429", "#d62728"],
                 labels={"rate": "No-show rate", "weekday": ""})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      coloraxis_showscale=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Bookings by site (90d trend)")
    trend = appts.groupby(["date", "site"])["booked"].sum().reset_index()
    fig2 = px.line(trend, x="date", y="booked", color="site",
                   labels={"booked": "Booked", "date": ""})
    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Chair utilisation — this week")
    fig3 = px.bar(pract, x="utilisation", y="practitioner", orientation="h",
                  color="utilisation", color_continuous_scale=["#d62728", "#F0B429", "#2ca02c"],
                  range_x=[0, 1], labels={"utilisation": "Utilisation", "practitioner": ""},
                  hover_data=["site", "filled", "slots"])
    fig3.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       coloraxis_showscale=False, xaxis_tickformat=".0%",
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)
with c4:
    st.subheader("No-show risk — upcoming appointments")
    st.dataframe(upcoming, use_container_width=True, hide_index=True)
    st.caption("Risk flags come from a trained model on your real data (history, lead time, patient type). "
               "High-risk slots get an extra SMS reminder automatically.")

st.divider()
st.info("**What the real version includes:** live link to your booking system, no-show prediction model per site, "
        "automatic reminder prioritisation, and a morning huddle view for the front desk.")
st.caption("Demo dashboard by Ahmad Ali — Data Scientist & ML Engineer · Python · scikit-learn · Streamlit")
