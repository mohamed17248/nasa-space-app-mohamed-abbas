"""Streamlit app: explore Egypt fire hotspots 2000-present.

Pick a date -> see that day's detection-density heatmap.
Data: MODIS (Terra+Aqua) through 2011, VIIRS (Suomi-NPP) from 2012.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt

st.set_page_config(page_title="Egypt Fire Hotspots", layout="wide")

DATA = "hotspots_harmonized.parquet"


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA)
    df["date"] = df["acq_datetime"].dt.date
    return df


@st.cache_data
def egypt_boundary():
    try:
        import geopandas as gpd

        world = gpd.read_file(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_110m_admin_0_countries.geojson"
        )
        return world[world["ISO_A3"] == "EGY"]
    except Exception:
        return None


df = load_data()
egypt = egypt_boundary()

st.title("Fire Hotspots — Egypt (2000–present)")
st.caption(
    "MODIS (Terra + Aqua) through 2011 · VIIRS (Suomi-NPP) from 2012 · "
    f"{len(df):,} detections"
)

# ------------------------------------------------------------- sidebar --

min_date = df["date"].min()
max_date = df["date"].max()

with st.sidebar:
    sel_date = st.date_input(
        "Date", value=date(2020, 8, 15), min_value=min_date, max_value=max_date
    )
    window = st.slider("Days around date (0 = single day)", 0, 15, 0)
    weight_conf = st.checkbox("Weight by confidence", value=True)
    show_points = st.checkbox("Show individual detections", value=True)

start = sel_date - timedelta(days=window)
end = sel_date + timedelta(days=window)
day = df[(df["date"] >= start) & (df["date"] <= end)].copy()

st.subheader(f"{start} → {end}" if window else f"{sel_date}")

# ---------------------------------------------------------------- stats --

c1, c2, c3, c4 = st.columns(4)
c1.metric("Detections", f"{len(day):,}")
if len(day):
    c2.metric("Mean FRP (MW)", f"{day['frp'].mean():.1f}")
    c3.metric("Day / Night", f"{(day['daynight'] == 'D').sum()} / {(day['daynight'] == 'N').sum()}")
    c4.metric("Sensor", day["sensor"].iloc[0] if window == 0 else "mixed")
else:
    c2.metric("Mean FRP (MW)", "—")
    c3.metric("Day / Night", "—")
    c4.metric("Sensor", "—")

# --------------------------------------------------------------- heatmap --

if len(day) == 0:
    st.warning("No detections in this window — try nearby dates.")
else:
    fig, ax = plt.subplots(figsize=(10, 8))

    weights = (day["confidence_pct"] / 100.0) if weight_conf else None
    h = ax.hist2d(
        day["longitude"], day["latitude"], bins=80,
        range=[[24.5, 36.5], [21.5, 32.5]],
        cmap="inferno", weights=weights,
    )
    plt.colorbar(h[3], ax=ax, label="confidence-weighted detections" if weight_conf else "detections")

    if egypt is not None:
        egypt.boundary.plot(ax=ax, color="white", linewidth=1.2, alpha=0.9)

    if show_points and len(day) <= 5000:
        ax.scatter(
            day["longitude"], day["latitude"], s=4, c="cyan",
            alpha=0.35, linewidths=0, zorder=5,
        )

    ax.set_xlim(24.5, 36.5)
    ax.set_ylim(21.5, 32.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Hotspot density — {sel_date}")
    ax.set_aspect("equal")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------------- monthly trend --

st.subheader("Monthly detections over time")
monthly = df.groupby(["month", "sensor"]).size().unstack(fill_value=0)
monthly.index = pd.PeriodIndex(monthly.index, freq="M").to_timestamp()
fig, ax = plt.subplots(figsize=(12, 3.5))
if "MODIS" in monthly:
    ax.plot(monthly.index, monthly["MODIS"], color="tab:orange",
            marker=".", markersize=3, label="MODIS (≤2011)")
if "VIIRS" in monthly:
    ax.plot(monthly.index, monthly["VIIRS"], color="tab:blue",
            marker=".", markersize=3, label="VIIRS (≥2012)")
ax.set_ylabel("detections / month")
ax.legend()
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
