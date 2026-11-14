"""Fire Activity Explorer.

1. Search a country/region -> the map zooms to it.
2. If data/<region>.csv (or .parquet) exists, fire bubbles, a burning calendar
   heatmap, a "current month vs historical median" metric and a CSV download appear.

Expected data file columns (FIRMS download): latitude, longitude, acq_date
Optional columns: instrument, satellite, confidence, frp, brightness, bright_ti4
Put files in a "data" folder next to this script, named after the region:
    data/egypt.csv   or   data/egypt.parquet
"""

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
import geopandas as gpd
from streamlit_folium import st_folium
from streamlit_autocomplete import st_textcomplete_autocomplete
from matplotlib import colormaps, colors

st.set_page_config(page_title="Fire Activity Explorer", layout="wide")

NE_URL = "ne_countries.geojson"  # local copy of Natural Earth 110m countries
DATA_DIR = Path("data")

# Scaling factor to turn MODIS counts into VIIRS-equivalent counts.
# 1.0 = no adjustment. Replace it with the factor your analysts compute
# from the overlap period (VIIRS total / MODIS total).
MODIS_TO_VIIRS_FACTOR = 1.0

# Common names that differ from Natural Earth's NAME field
ALIASES = {
    "ivory coast": "côte d'ivoire",
    "east timor": "timor-leste",
    "türkiye": "turkey",
    "turkiye": "turkey",
    "vatican city": "vatican",
    "united states": "united states of america",
    "sao tome": "são tomé and principe",
    "bosnia and herzegovina": "bosnia and herz.",
    "central african republic": "central african rep.",
    "democratic republic of the congo": "dem. rep. congo",
    "republic of the congo": "congo",
    "dominican republic": "dominican rep.",
    "solomon islands": "solomon is.",
    "south sudan": "s. sudan",
    "equatorial guinea": "eq. guinea",
    "dr congo": "dem. rep. congo",
    "drc": "dem. rep. congo",
    "congo republic": "congo",
    "czech republic": "czechia",
}

# Bboxes for microstates too small for the 110m dataset
# (lat_min, lat_max, lon_min, lon_max)
MANUAL_REGIONS = {
    "andorra": (42.4, 42.75, 1.4, 1.8),
    "liechtenstein": (47.0, 47.3, 9.5, 9.65),
    "monaco": (43.72, 43.78, 7.4, 7.5),
    "san marino": (43.88, 43.99, 12.4, 12.5),
    "vatican": (41.89, 41.93, 12.44, 12.47),
    "malta": (35.8, 36.1, 14.2, 14.65),
    "singapore": (1.2, 1.5, 103.6, 104.1),
    "bahrain": (25.7, 26.4, 50.3, 50.95),
    "maldives": (-0.7, 7.4, 72.6, 73.9),
    "mauritius": (-20.6, -19.9, 57.3, 57.95),
    "seychelles": (-4.8, -4.5, 55.2, 55.7),
    "comoros": (-13.0, -11.3, 43.1, 45.3),
    "sao tome and principe": (0.0, 2.3, 6.2, 7.6),
    "dominica": (15.15, 15.7, -61.55, -61.15),
    "grenada": (11.9, 12.3, -61.85, -61.55),
    "saint lucia": (13.7, 14.1, -61.1, -60.85),
    "saint vincent and the grenadines": (12.5, 13.45, -61.55, -61.05),
    "antigua and barbuda": (16.95, 17.75, -61.95, -61.6),
    "saint kitts and nevis": (17.2, 17.45, -62.9, -62.55),
    "barbados": (13.0, 13.45, -59.75, -59.35),
    "trinidad and tobago": (10.0, 11.35, -61.95, -60.5),
    "tonga": (-23.0, -15.4, -176.0, -173.0),
    "samoa": (-14.05, -13.4, -172.85, -171.3),
    "vanuatu": (-20.3, -13.0, 166.4, 170.2),
    "kiribati": (-11.5, 5.0, 168.0, 180.0),
    "tuvalu": (-11.2, -5.6, 175.0, 180.0),
    "palau": (2.7, 8.2, 134.0, 134.8),
    "marshall islands": (4.5, 15.0, 160.7, 172.2),
    "nauru": (-0.55, -0.45, 166.85, 166.95),
    "cape verde": (14.8, 17.2, -25.4, -22.6),
}

# Countries whose full bounding box zooms out too far (date line, far territories).
# These always win over the automatic boxes.
OVERRIDES = {
    "france": (41.0, 51.5, -5.5, 9.8),
    "united states of america": (24.0, 50.0, -125.0, -66.5),
    "russia": (41.0, 70.0, 27.0, 180.0),
    "fiji": (-20.3, -15.5, 176.5, 180.0),
    "new zealand": (-47.5, -34.0, 166.0, 179.0),
}

WANTED_COLUMNS = {
    "latitude", "longitude", "acq_date", "satellite", "instrument",
    "confidence", "frp", "daynight", "brightness", "bright_ti4", "type",
}


# --------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------
@st.cache_data
def load_regions():
    """Return {search_name_lower: (lat_min, lat_max, lon_min, lon_max)}."""
    import geopandas as gpd

    world = gpd.read_file(NE_URL)
    name_col = next(c for c in ("NAME", "name", "NAME_EN", "ADMIN") if c in world.columns)

    regs = {}
    for _, row in world.iterrows():
        name = str(row[name_col]).lower()
        minx, miny, maxx, maxy = row.geometry.bounds
        regs[name] = (miny, maxy, minx, maxx)

    for alias, target in ALIASES.items():
        if target in regs and alias not in regs:
            regs[alias] = regs[target]

    for name, bbox in MANUAL_REGIONS.items():
        regs.setdefault(name, bbox)

    regs.update(OVERRIDES)
    return regs


# --------------------------------------------------------------------------
# Hotspot data
# --------------------------------------------------------------------------
def infer_sensor(df: pd.DataFrame) -> pd.Series:
    """Return a 'MODIS' / 'VIIRS' label for each row."""
    if "instrument" in df.columns:
        return df["instrument"].astype(str).str.upper().str.strip()
    if "bright_ti4" in df.columns:
        return pd.Series("VIIRS", index=df.index)
    if "brightness" in df.columns:
        return pd.Series("MODIS", index=df.index)
    return pd.Series("UNKNOWN", index=df.index)


@st.cache_data
def load_hotspots(region_key: str):
    """Load data/<region>.parquet or .csv, or return None if missing."""
    stem = region_key.replace(" ", "_")
    parquet = DATA_DIR / f"{stem}.parquet"
    csv = DATA_DIR / f"{stem}.csv"

    if parquet.exists():
        df = pd.read_parquet(parquet)
        df = df[[c for c in df.columns if c in WANTED_COLUMNS]]
    elif csv.exists():
        df = pd.read_csv(csv, usecols=lambda c: c in WANTED_COLUMNS)
    else:
        return None

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["sensor"] = infer_sensor(df)
    # weight = how much each hotspot counts after harmonization
    df["weight"] = np.where(df["sensor"] == "MODIS", MODIS_TO_VIIRS_FACTOR, 1.0)
    df["year"] = df["acq_date"].dt.year
    df["month"] = df["acq_date"].dt.month
    return df.drop_duplicates()



# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("Fire Activity Explorer")
st.caption(
    "Search a country or region to zoom in. Fire data appears when a data file "
    "for that region exists in the data folder."
)

try:
    regions = load_regions()
except Exception as e:
    st.error(f"Could not load country boundaries: {e}")
    st.stop()

OPTIONS = sorted(k.title() for k in regions.keys())

selected = st_textcomplete_autocomplete(
    "Search region",
    OPTIONS,
    key="region_pick",
    placeholder="Type to search, e.g. 'egy', 'jap', 'bra'...",
    height=300,
)

query = (selected or "").strip().lower()
match = None
if query:
    if query in regions:
        match = query
    else:
        cands = [k for k in regions if query in k]
        if cands:
            match = min(cands, key=len)
bbox = regions[match] if match else None

if match:
    st.success(f"Showing: **{match.title()}**")

# ---- Load and filter fire data -------------------------------------------
sub = None

if match:
    hot = load_hotspots(match)
    if hot is None or hot.empty:
        st.info(
            f"No fire data for {match.title()} yet. Add `data/{match.replace(' ', '_')}.csv` "
            "(a FIRMS download) to see hotspots."
        )
    else:
        dmin = hot["acq_date"].min().date()
        dmax = hot["acq_date"].max().date()
        if dmin < dmax:
            d0, d1 = st.slider("Date range", dmin, dmax, (dmin, dmax))
        else:
            d0, d1 = dmin, dmax

        sub = hot[(hot["acq_date"].dt.date >= d0) & (hot["acq_date"].dt.date <= d1)]
        if sub.empty:
            st.warning("No hotspots in this date range.")
            sub = None

# ---- Map -----------------------------------------------------------------
view = "Points"
if sub is not None:
    view = st.radio("Map view", ["Points", "Density grid"], horizontal=True,
                    key="mapview")

if view == "Points" and sub is not None:
    pts = sub[["latitude", "longitude", "acq_date", "sensor"]].copy()
    pts["acq_date"] = pts["acq_date"].dt.strftime("%Y-%m-%d")   # explore() can't serialize datetimes
    if len(pts) > 40000:                                        # keep the browser fast
        pts = pts.sample(40000, random_state=0)
        st.caption("Showing a random sample of 40,000 hotspots on the map.")

    gdf = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts["longitude"], pts["latitude"]),
        crs="EPSG:4326",
    )
    m = gdf.explore(
        column="sensor",
        categorical=True,
        cmap=["red", "blue", "orange"],          # one color per sensor
        marker_kwds=dict(radius=3),
        style_kwds=dict(fillOpacity=0.15, weight=0),  # low opacity for overlaps
        tooltip=["acq_date", "sensor"],
        tiles="OpenStreetMap",
        legend=True,
    )
elif view == "Density grid" and sub is not None:
    # Aggregate detections into ~0.05° grid cells, colored by count
    cell = 0.05

    def _grid(c):
        return (sub.assign(
                    glat=(sub.latitude // c) * c + c / 2,
                    glon=(sub.longitude // c) * c + c / 2,
                )
                .groupby(["glat", "glon"]).size()
                .reset_index(name="count"))

    grid = _grid(cell)
    if len(grid) > 8000:                        # too many circles -> coarsen
        cell = 0.1
        grid = _grid(cell)

    center = [20, 0]
    zoom = 2
    if bbox:
        center = [(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2]
        zoom = 6
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    maxc = int(grid["count"].max())
    norm = colors.LogNorm(vmin=1, vmax=maxc)
    cmap = colormaps["YlOrRd"]
    radius_m = int(cell * 111320 * 0.6)
    for r in grid.itertuples():
        folium.Circle(
            location=[r.glat, r.glon],
            radius=radius_m,
            color=colors.to_hex(cmap(norm(r.count))),
            fill=True,
            fill_opacity=0.55,
            weight=0,
        ).add_to(m)
    st.caption(
        f"Grid cell ≈ {cell}° (~{int(cell * 111)} km per side) · "
        f"max {maxc} detections per cell"
    )
else:
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")

if bbox:
    m.fit_bounds([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])    # zoom to the selected country

st_folium(m, height=650, use_container_width=True, returned_objects=[], key="map")

# ---- Calendar, median comparison, download -------------------------------
if sub is not None:
    st.subheader(f"Burning activity calendar: {match.title()}")

    monthly = sub.groupby(["year", "month"])["weight"].sum().reset_index(name="count")
    pivot = monthly.pivot(index="year", columns="month", values="count")
    pivot = pivot.reindex(columns=range(1, 13)).fillna(0)
    pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]

    heat = px.imshow(
        pivot,
        color_continuous_scale="Viridis",  # colorblind-friendly
        labels=dict(x="Month", y="Year", color="Hotspots"),
        aspect="auto",
        text_auto=".0f",
    )
    heat.update_yaxes(type="category")
    heat.update_layout(height=max(250, 60 * len(pivot) + 120))
    st.plotly_chart(heat, use_container_width=True, key="calendar")

    # Current (latest) month vs the historical median for that same month
    latest = monthly.sort_values(["year", "month"]).iloc[-1]
    ly, lm = int(latest["year"]), int(latest["month"])
    current = float(latest["count"])
    history = monthly[(monthly["month"] == lm) & (monthly["year"] < ly)]["count"]

    c1, c2 = st.columns(2)
    if len(history) > 0:
        median = float(history.median())
        c1.metric(
            f"{calendar.month_name[lm]} {ly}",
            f"{current:.0f} hotspots",
            delta=f"{current - median:+.0f} vs historical median ({median:.0f})",
            delta_color="inverse",  # more fire = worse = red
        )
    else:
        c1.metric(f"{calendar.month_name[lm]} {ly}", f"{current:.0f} hotspots")
        c2.info("Add more years of data to compare against the historical median.")

    # Sensor breakdown (useful to check harmonization)
    with st.expander("Hotspots per month by sensor (before harmonization factor)"):
        by_sensor = (
            sub.groupby([sub["acq_date"].dt.to_period("M").astype(str), "sensor"])
            .size()
            .unstack(fill_value=0)
        )
        st.line_chart(by_sensor)

    st.download_button(
        "Download filtered hotspots (CSV)",
        sub.drop(columns=["weight", "year", "month"]).to_csv(index=False),
        file_name=f"{match.replace(' ', '_')}_hotspots.csv",
        mime="text/csv",
    )
