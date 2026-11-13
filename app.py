"""Burning Activity Calendar — Egypt fire hotspots 2000-present.

Tabs:
  1. Burning Calendar   - day-of-year x year activity heatmap for an AOI
  2. Historical Patterns - seasonal climatology + monthly/yearly trends
  3. Early Warnings      - 7-day anomaly detection vs climatology
  4. Daily Explorer      - single-date density heatmap (original view)

Sensor eras are analyzed separately (MODIS <=2011, VIIRS >=2012) because
their detection counts are not directly comparable.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

st.set_page_config(page_title="Egypt Fire Activity Calendar", layout="wide")

DATA = "hotspots_harmonized.parquet"

AOIS = {
    "All Egypt": (21.5, 32.0, 24.5, 36.0),
    "Nile Delta": (30.4, 31.6, 29.5, 32.6),
    "Nile Valley (Upper Egypt)": (24.0, 30.4, 29.5, 33.0),
    "Western Desert": (22.0, 31.0, 25.0, 29.5),
    "Eastern Desert": (22.0, 30.5, 32.5, 35.5),
    "Sinai": (27.0, 31.3, 32.3, 34.9),
    "North Coast": (30.6, 31.7, 24.5, 30.0),
    "Red Sea Coast": (21.8, 28.5, 33.5, 36.5),
}


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA)
    df["date"] = df["acq_datetime"].dt.date
    df["doy"] = df["acq_datetime"].dt.dayofyear
    df["year"] = df["acq_datetime"].dt.year
    df["month"] = df["acq_datetime"].dt.to_period("M")
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

st.title("Burning Activity Calendar — Egypt")
st.caption(
    "Fire hotspot record harmonized across sensors: MODIS (Terra+Aqua) "
    "2000–2011, VIIRS (Suomi-NPP) 2012–present · "
    f"{len(df):,} detections · NASA FIRMS data"
)

# ------------------------------------------------------------ AOI select --

with st.sidebar:
    st.header("Area of Interest")
    aoi_name = st.selectbox("Preset region", list(AOIS.keys()), index=0)
    if aoi_name == "Custom":
        pass
    custom = st.checkbox("Custom bounding box")
    if custom:
        la1 = st.number_input("Lat min", 21.5, 32.0, 21.5, 0.1)
        la2 = st.number_input("Lat max", 21.5, 32.0, 32.0, 0.1)
        lo1 = st.number_input("Lon min", 24.5, 36.5, 24.5, 0.1)
        lo2 = st.number_input("Lon max", 24.5, 36.5, 36.0, 0.1)
        bbox = (min(la1, la2), max(la1, la2), min(lo1, lo2), max(lo1, lo2))
    else:
        bbox = AOIS[aoi_name]
    st.caption(f"Bounds: lat {bbox[0]:.1f}–{bbox[1]:.1f}, lon {bbox[2]:.1f}–{bbox[3]:.1f}")

aoi = df[
    (df.latitude >= bbox[0]) & (df.latitude <= bbox[1])
    & (df.longitude >= bbox[2]) & (df.longitude <= bbox[3])
].copy()


def daily_activity(d):
    """Daily detection count and FRP sum indexed by date."""
    g = d.groupby("date").agg(count=("frp", "size"), frp=("frp", "sum"))
    return g


tab_cal, tab_hist, tab_warn, tab_day = st.tabs(
    ["Burning Calendar", "Historical Patterns", "Early Warnings", "Daily Explorer"]
)

# ============================================================ CALENDAR ==

with tab_cal:
    st.subheader("Burning activity calendar")
    st.caption(
        "Each cell = total activity on that day-of-year for that year. "
        "Bright vertical bands = fire season; bright horizontal lines = unusual years."
    )
    metric = st.radio("Activity metric", ["Detections", "FRP (MW)"], horizontal=True)

    pivot = (
        aoi.pivot_table(index="year", columns="doy", values="frp",
                        aggfunc="count" if metric == "Detections" else "sum",
                        fill_value=0)
    )
    years = list(range(int(df.year.min()), int(df.year.max()) + 1))
    pivot = pivot.reindex(index=years, columns=range(1, 367), fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 7))
    if metric == "Detections":
        norm = PowerNorm(gamma=0.5, vmin=0.5)
    else:
        norm = PowerNorm(gamma=0.5, vmin=1)
    im = ax.imshow(
        pivot.to_numpy(), aspect="auto", origin="lower", cmap="inferno",
        extent=[1, 366, years[0], years[-1] + 1], norm=norm,
    )
    fig.colorbar(im, ax=ax, label=metric)

    # Month boundaries on x axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names)
    ax.set_yticks(years[::2])
    ax.set_ylabel("Year")
    ax.set_xlabel("Day of year")
    ax.set_title(f"{aoi_name} — {metric.lower()} by day of year")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================ HISTORY ==

with tab_hist:
    st.subheader("Historical fire patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Seasonal cycle** (mean daily activity by day of year)")
        daily = daily_activity(aoi).reindex(
            pd.date_range(str(df.date.min()), str(df.date.max())), fill_value=0
        )
        daily["doy"] = daily.index.dayofyear
        daily["year"] = daily.index.year
        daily["sensor_era"] = np.where(daily["year"] <= 2011, "MODIS", "VIIRS")
        clim = daily.groupby(["sensor_era", "doy"])["count"].mean().unstack(0)
        clim = clim.reindex(range(1, 367)).interpolate()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = {"MODIS": "tab:orange", "VIIRS": "tab:blue"}
        for era in clim.columns:
            ax.plot(clim.index, clim[era], color=colors.get(era, "gray"),
                    label=f"{era} climatology")
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_names)
        ax.set_ylabel("mean detections / day")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Sensor eras plotted separately — raw counts are not comparable across the 2012 transition.")

    with col2:
        st.markdown("**Yearly totals**")
        yearly = aoi.groupby(["year", "sensor"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 4))
        x = yearly.index
        if "MODIS" in yearly:
            ax.bar(x - 0.2, yearly["MODIS"], 0.4, color="tab:orange", label="MODIS")
        if "VIIRS" in yearly:
            ax.bar(x + 0.2, yearly["VIIRS"], 0.4, color="tab:blue", label="VIIRS")
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=90, fontsize=7)
        ax.set_ylabel("detections / year")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Monthly totals**")
    monthly = aoi.groupby(["month", "sensor"]).size().unstack(fill_value=0)
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(13, 3.5))
    if "MODIS" in monthly:
        ax.plot(monthly.index, monthly["MODIS"], color="tab:orange",
                marker=".", markersize=3, label="MODIS")
    if "VIIRS" in monthly:
        ax.plot(monthly.index, monthly["VIIRS"], color="tab:blue",
                marker=".", markersize=3, label="VIIRS")
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.set_ylabel("detections / month")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =========================================================== WARNINGS ==

with tab_warn:
    st.subheader("Early warning — unusual fire activity")
    st.caption(
        "7-day rolling detection totals are compared against the climatology "
        "for the same day-of-year within each sensor era. Periods with "
        "z-score ≥ 2 are flagged as unusual."
    )

    daily = daily_activity(aoi)
    # full date range so rolling windows don't break at gaps
    idx = pd.date_range(str(daily.index.min()), str(daily.index.max()))
    daily = daily.reindex(idx, fill_value=0)
    daily["year"] = daily.index.year
    daily["sensor_era"] = np.where(daily["year"] <= 2011, "MODIS", "VIIRS")
    daily["roll7"] = daily.groupby("sensor_era")["count"].transform(
        lambda s: s.rolling(7, center=True, min_periods=1).sum()
    )
    daily["doy"] = daily.index.dayofyear

    clim = daily.groupby(["sensor_era", "doy"])["roll7"].agg(["mean", "std"])
    clim["std"] = clim["std"].replace(0, np.nan)
    daily = daily.join(clim, on=["sensor_era", "doy"])
    daily["z"] = (daily["roll7"] - daily["mean"]) / daily["std"]

    z_thresh = st.slider("Alert threshold (z-score)", 1.0, 4.0, 2.0, 0.25)
    alerts = daily[daily["z"] >= z_thresh].copy()

    if alerts.empty:
        st.success("No periods exceed the threshold in this AOI.")
    else:
        # group consecutive alert days (gap > 2 days starts a new period)
        rows = []
        grp = (alerts.index.to_series().diff() > pd.Timedelta(days=2)).cumsum()
        for _, g in alerts.groupby(grp):
            rows.append({
                "start": g.index.min().date(),
                "end": g.index.max().date(),
                "days_flagged": len(g),
                "peak_z": round(g["z"].max(), 1),
                "7d_detections": int(g.loc[g["z"].idxmax(), "roll7"]),
                "sensor_era": g["sensor_era"].iloc[0],
            })
        periods_df = pd.DataFrame(rows).sort_values("peak_z", ascending=False)

        st.markdown(f"**{len(periods_df)} unusual periods** detected")
        st.dataframe(periods_df, use_container_width=True, hide_index=True)

        st.markdown("**Top critical periods by raw 7-day activity**")
        top = daily.nlargest(10, "roll7")[
            ["count", "roll7", "z", "frp", "sensor_era"]
        ].rename(columns={
            "count": "day_detections", "roll7": "7d_detections",
            "frp": "7d_FRP_MW",
        })
        top.index = top.index.date
        top.index.name = "window_center"
        st.dataframe(top.round(1), use_container_width=True)

# ========================================================= DAILY VIEW ==

with tab_day:
    st.subheader("Daily hotspot explorer")
    min_date, max_date = df["date"].min(), df["date"].max()
    sel_date = st.date_input("Date", value=date(2020, 8, 15),
                             min_value=min_date, max_value=max_date)
    window = st.slider("Days around date", 0, 15, 0, key="daywin")
    weight_conf = st.checkbox("Weight by confidence", value=True, key="wc")

    start = sel_date - timedelta(days=window)
    end = sel_date + timedelta(days=window)
    day = aoi[(aoi["date"] >= start) & (aoi["date"] <= end)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Detections", f"{len(day):,}")
    c2.metric("Mean FRP (MW)", f"{day['frp'].mean():.1f}" if len(day) else "—")
    if len(day):
        c3.metric("Day / Night",
                  f"{(day['daynight'] == 'D').sum()} / {(day['daynight'] == 'N').sum()}")
    else:
        c3.metric("Day / Night", "—")

    if len(day) == 0:
        st.warning("No detections in this window — try nearby dates.")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        weights = (day["confidence_pct"] / 100.0) if weight_conf else None
        h = ax.hist2d(
            day["longitude"], day["latitude"], bins=80,
            range=[[24.5, 36.5], [21.5, 32.5]], cmap="inferno", weights=weights,
        )
        plt.colorbar(h[3], ax=ax,
                     label="confidence-weighted" if weight_conf else "detections")
        if egypt is not None:
            egypt.boundary.plot(ax=ax, color="white", linewidth=1.2, alpha=0.9)
        ax.scatter(day["longitude"], day["latitude"], s=4, c="cyan",
                   alpha=0.35, linewidths=0, zorder=5)
        ax.set_xlim(24.5, 36.5)
        ax.set_ylim(21.5, 32.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{aoi_name} — {start} → {end}" if window else f"{aoi_name} — {sel_date}")
        ax.set_aspect("equal")
        st.pyplot(fig)
        plt.close(fig)
