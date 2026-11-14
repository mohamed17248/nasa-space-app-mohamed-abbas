"""Fire Activity Explorer — Competition Edition.

Multi-decade harmonized wildfire & thermal anomaly intelligence platform
powered by NASA FIRMS (MODIS + VIIRS) and Natural Earth boundaries.
"""

import calendar
import json
from pathlib import Path

import folium
from folium.plugins import FastMarkerCluster, HeatMap
import geopandas as gpd
from matplotlib import colormaps, colors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Fire Activity Explorer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle dark-accent styling for KPI cards
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(255, 87, 34, 0.06);
        border: 1px solid rgba(255, 87, 34, 0.22);
        padding: 12px 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NE_URL = "ne_countries.geojson"
DATA_DIR = Path("data")
MODIS_TO_VIIRS_FACTOR = 1.0

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

OVERRIDES = {
    "france": (41.0, 51.5, -5.5, 9.8),
    "united states of america": (24.0, 50.0, -125.0, -66.5),
    "russia": (41.0, 70.0, 27.0, 180.0),
    "fiji": (-20.3, -15.5, 176.5, 180.0),
    "new zealand": (-47.5, -34.0, 166.0, 179.0),
}

WANTED_COLUMNS = {
    "latitude", "longitude", "acq_date", "acq_time", "satellite",
    "instrument", "confidence", "frp", "daynight", "brightness",
    "bright_ti4", "bright_ti5", "type",
}

FIRE_TYPE_LABELS = {
    0: "Vegetation / Wildfire",
    1: "Active Volcano",
    2: "Static / Industrial / Flare",
    3: "Offshore",
}


# --------------------------------------------------------------------------
# Regions & Geometries
# --------------------------------------------------------------------------
@st.cache_data
def load_regions():
    """Return (bboxes_dict, geojson_polygons_dict)."""
    world = gpd.read_file(NE_URL)
    name_col = next(c for c in ("NAME", "name", "NAME_EN", "ADMIN") if c in world.columns)

    regs = {}
    polys = {}
    for _, row in world.iterrows():
        name = str(row[name_col]).lower()
        minx, miny, maxx, maxy = row.geometry.bounds
        regs[name] = (miny, maxy, minx, maxx)
        polys[name] = json.loads(gpd.GeoSeries([row.geometry]).to_json())

    for alias, target in ALIASES.items():
        if target in regs and alias not in regs:
            regs[alias] = regs[target]
            polys[alias] = polys[target]

    for name, bbox in MANUAL_REGIONS.items():
        regs.setdefault(name, bbox)

    regs.update(OVERRIDES)
    return regs, polys


# --------------------------------------------------------------------------
# Data Loading & Scientific Harmonization
# --------------------------------------------------------------------------
def infer_sensor(df: pd.DataFrame) -> pd.Series:
    if "instrument" in df.columns:
        return df["instrument"].astype(str).str.upper().str.strip()
    if "bright_ti4" in df.columns:
        return pd.Series("VIIRS", index=df.index)
    if "brightness" in df.columns:
        return pd.Series("MODIS", index=df.index)
    return pd.Series("UNKNOWN", index=df.index)


def normalize_confidence(series: pd.Series) -> pd.Series:
    """Unify MODIS (0-100) and VIIRS ('l'/'n'/'h') confidence values."""
    def _map_val(v):
        if pd.isna(v):
            return "Nominal"
        s = str(v).strip().lower()
        if s in ("l", "low"):
            return "Low"
        if s in ("n", "nominal"):
            return "Nominal"
        if s in ("h", "high"):
            return "High"
        try:
            num = float(s)
            if num < 30:
                return "Low"
            if num < 80:
                return "Nominal"
            return "High"
        except ValueError:
            return "Nominal"

    return series.map(_map_val)


def prepare_hotspots(df: pd.DataFrame):
    """Clean, enrich, and harmonize MODIS/VIIRS records."""
    df = df[[c for c in df.columns if c in WANTED_COLUMNS]].copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["sensor"] = infer_sensor(df)
    df["year"] = df["acq_date"].dt.year
    df["month"] = df["acq_date"].dt.month
    df = df.drop_duplicates()

    # Enrich NASA FIRMS metadata columns
    if "type" in df.columns:
        df["fire_type"] = (
            pd.to_numeric(df["type"], errors="coerce")
            .map(FIRE_TYPE_LABELS)
            .fillna("Unclassified")
        )
    else:
        df["fire_type"] = "Unclassified"

    if "confidence" in df.columns:
        df["conf_tier"] = normalize_confidence(df["confidence"])
    else:
        df["conf_tier"] = "Nominal"

    if "daynight" in df.columns:
        df["period"] = (
            df["daynight"].astype(str).str.upper().str.strip()
            .map({"D": "Daytime", "N": "Nighttime"})
            .fillna("Unknown")
        )
    else:
        df["period"] = "Unknown"

    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["frp"] = 0.0

    # Harmonize MODIS -> VIIRS equivalent
    factor = MODIS_TO_VIIRS_FACTOR
    viirs_years = df.loc[df["sensor"] == "VIIRS", "year"].unique()
    if len(viirs_years) and (df["sensor"] == "MODIS").any():
        split = int(min(viirs_years))
        overlap = df[df["year"] >= split]
        n_modis = int((overlap["sensor"] == "MODIS").sum())
        n_viirs = int((overlap["sensor"] == "VIIRS").sum())
        if n_modis and n_viirs:
            factor = n_viirs / n_modis
        df = df[~((df["sensor"] == "MODIS") & (df["year"] >= split))]

    df["weight"] = np.where(df["sensor"] == "MODIS", factor, 1.0)
    df["weighted_frp"] = df["frp"] * df["weight"]
    return df, factor


@st.cache_data
def load_local_hotspots(region_key: str):
    """Check both data/<region>.* and ./<region>.* for Parquet or CSV."""
    stem = region_key.replace(" ", "_")
    candidates = [
        DATA_DIR / f"{stem}.parquet",
        Path(f"{stem}.parquet"),
        DATA_DIR / f"{stem}.csv",
        Path(f"{stem}.csv"),
    ]
    for path in candidates:
        if path.exists():
            raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            return prepare_hotspots(raw)
    return None, MODIS_TO_VIIRS_FACTOR


@st.cache_data(ttl=3600)
def fetch_firms_nrt(map_key: str, source: str, bbox: tuple, days: int):
    """Fetch near-real-time hotspots directly from NASA FIRMS Area API."""
    lat_min, lat_max, lon_min, lon_max = bbox
    area = f"{lon_min:.4f},{lat_min:.4f},{lon_max:.4f},{lat_max:.4f}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key.strip()}/{source}/{area}/{days}"
    raw = pd.read_csv(url)
    if raw.empty or "latitude" not in raw.columns:
        return None, 1.0
    return prepare_hotspots(raw)


# --------------------------------------------------------------------------
# Sidebar Controls
# --------------------------------------------------------------------------
try:
    regions, polygons = load_regions()
except Exception as e:
    st.error(f"Could not load country boundaries (`{NE_URL}`): {e}")
    st.stop()

OPTIONS = sorted(k.title() for k in regions.keys())
default_idx = OPTIONS.index("Egypt") if "Egypt" in OPTIONS else None

with st.sidebar:
    st.header("🛰️ Explorer Controls")
    selected = st.selectbox(
        "Country / Region",
        OPTIONS,
        index=default_idx,
        placeholder="Type to search (e.g. Egypt, Brazil)...",
    )

    st.divider()
    st.subheader("Data Source")
    uploaded_file = st.file_uploader(
        "Upload FIRMS CSV or Parquet",
        type=["csv", "parquet"],
        help="Override local files with any NASA FIRMS download.",
    )

    with st.expander("🌐 Live NASA FIRMS API (Optional)"):
        firms_key = st.text_input("FIRMS MAP_KEY", type="password", help="Free key from firms.modaps.eosdis.nasa.gov")
        firms_source = st.selectbox(
            "Satellite Feed",
            ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"],
        )
        firms_days = st.slider("Lookback window (days)", 1, 10, 5)
        use_live_api = st.checkbox("Fetch live NRT detections", value=False, disabled=not bool(firms_key))

query = (selected or "").strip().lower()
match = query if query in regions else None
bbox = regions[match] if match else None
poly_geojson = polygons.get(match) if match else None

# --------------------------------------------------------------------------
# Header & Data Resolution
# --------------------------------------------------------------------------
st.title("🔥 Fire Activity Explorer")
st.caption(
    "Harmonized multi-sensor satellite thermal intelligence (MODIS + VIIRS) "
    "for wildfire monitoring, anomaly detection, and intensity analysis."
)

hot, factor, source_label = None, 1.0, ""
if uploaded_file is not None:
    raw_up = (
        pd.read_parquet(uploaded_file)
        if uploaded_file.name.endswith(".parquet")
        else pd.read_csv(uploaded_file)
    )
    hot, factor = prepare_hotspots(raw_up)
    source_label = f"Uploaded file: `{uploaded_file.name}`"
elif match:
    if use_live_api and firms_key and bbox:
        with st.spinner(f"Querying NASA FIRMS ({firms_source}) for {match.title()}..."):
            try:
                hot, factor = fetch_firms_nrt(firms_key, firms_source, bbox, firms_days)
                source_label = f"Live NASA FIRMS API (`{firms_source}`, last {firms_days}d)"
            except Exception as exc:
                st.warning(f"Live API request failed ({exc}); falling back to local archive.")
    if hot is None:
        hot, factor = load_local_hotspots(match)
        source_label = f"Local archive (`{match.replace(' ', '_')}`)"

# --------------------------------------------------------------------------
# Sidebar Analytical Filters (when data exists)
# --------------------------------------------------------------------------
sub = None
if hot is not None and not hot.empty:
    with st.sidebar:
        st.divider()
        st.subheader("Telemetry Filters")

        dmin, dmax = hot["acq_date"].min().date(), hot["acq_date"].max().date()
        if dmin < dmax:
            d0, d1 = st.slider("Date range", dmin, dmax, (dmin, dmax))
        else:
            d0, d1 = dmin, dmax

        avail_types = sorted(hot["fire_type"].unique())
        sel_types = st.multiselect("Thermal source type", avail_types, default=avail_types)

        avail_conf = [c for c in ("High", "Nominal", "Low") if c in hot["conf_tier"].unique()]
        sel_conf = st.multiselect("Detection confidence", avail_conf, default=avail_conf)

        avail_periods = sorted(hot["period"].unique())
        sel_periods = st.multiselect("Day / Night pass", avail_periods, default=avail_periods)

        min_frp = 0.0
        if hot["frp"].max() > 0:
            min_frp = st.number_input("Minimum FRP (MW)", min_value=0.0, value=0.0, step=5.0)

    mask = (
        (hot["acq_date"].dt.date >= d0)
        & (hot["acq_date"].dt.date <= d1)
        & (hot["fire_type"].isin(sel_types))
        & (hot["conf_tier"].isin(sel_conf))
        & (hot["period"].isin(sel_periods))
        & (hot["frp"] >= min_frp)
    )
    sub = hot[mask].copy()

# --------------------------------------------------------------------------
# Executive KPIs & Automated Insight Banner
# --------------------------------------------------------------------------
if match and (hot is None or hot.empty):
    st.info(
        f"Showing boundaries for **{match.title()}**. No local hotspot file found "
        f"(`data/{match.replace(' ', '_')}.parquet`). Upload a FIRMS file or enable the "
        "Live NASA FIRMS API in the sidebar to populate analytics."
    )
elif sub is not None and sub.empty:
    st.warning("No hotspots match the current sidebar filters. Broaden your filter selection.")

if sub is not None and not sub.empty:
    total_weighted = sub["weight"].sum()
    has_frp = sub["frp"].sum() > 0
    total_frp_gw = sub["weighted_frp"].sum() / 1000.0
    veg_share = (sub["fire_type"] == "Vegetation / Wildfire").mean() * 100

    # Latest month anomaly vs historical same day-of-month median
    latest_date = sub["acq_date"].max()
    ly, lm, last_day = latest_date.year, latest_date.month, latest_date.day
    current_mo = sub[(sub["year"] == ly) & (sub["month"] == lm)]["weight"].sum()
    past_years = sorted(y for y in sub["year"].unique() if y < ly)
    hist = (
        sub[(sub["month"] == lm) & (sub["year"] < ly) & (sub["acq_date"].dt.day <= last_day)]
        .groupby("year")["weight"]
        .sum()
        .reindex(past_years, fill_value=0)
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Harmonized Detections",
        f"{total_weighted:,.0f}",
        help=f"MODIS detections scaled ×{factor:.2f} to VIIRS-equivalent.",
    )
    k2.metric(
        "Radiative Energy (FRP)",
        f"{total_frp_gw:,.2f} GW" if has_frp else "N/A",
        delta=f"Median {sub['frp'].median():.1f} MW / event" if has_frp else None,
        delta_color="off",
    )
    k3.metric(
        "Vegetation Fire Share",
        f"{veg_share:.1f}%" if "Unclassified" not in sub["fire_type"].unique() else "N/A",
        delta="vs. static industrial/flares" if "Unclassified" not in sub["fire_type"].unique() else None,
        delta_color="off",
    )
    latest_label = f"{calendar.month_abbr[lm]} {ly} (1–{last_day})"
    if len(hist) > 0:
        med = float(hist.median())
        pct = ((current_mo - med) / med * 100) if med > 0 else 0.0
        k4.metric(
            latest_label,
            f"{current_mo:,.0f}",
            delta=f"{current_mo - med:+,.0f} ({pct:+.0f}%) vs median",
            delta_color="inverse",
        )
    else:
        k4.metric(latest_label, f"{current_mo:,.0f}")

    # Automated narrative takeaway
    peak_mo_num = int(sub.groupby("month")["weight"].sum().idxmax())
    peak_mo_name = calendar.month_name[peak_mo_num]
    st.caption(
        f"Source: {source_label} · Harmonization factor: **×{factor:.2f}** · "
        f"Seasonal peak occurs in **{peak_mo_name}**."
    )

# --------------------------------------------------------------------------
# Main Analytical Tabs
# --------------------------------------------------------------------------
tab_map, tab_season, tab_intensity, tab_qa = st.tabs(
    [
        "🗺️ Spatial Map Explorer",
        "📅 Seasonality & Anomalies",
        "⚡ Fire Intensity (FRP) & Diurnal",
        "🔬 Sensor Harmonization & Export",
    ]
)

# ---- TAB 1: MAP ----------------------------------------------------------
with tab_map:
    c_mode, c_tiles, c_weight = st.columns([2, 2, 2])
    map_mode = c_mode.radio(
        "Visualization layer",
        ["Density grid", "Clustered points", "Thermal heatmap"],
        horizontal=True,
    )
    basemap_choice = c_tiles.selectbox(
        "Basemap",
        ["CartoDB dark_matter", "Esri Satellite", "OpenStreetMap"],
    )
    weight_metric = c_weight.selectbox(
        "Intensity weighting",
        ["Harmonized count", "Fire Radiative Power (FRP)"] if (sub is not None and sub["frp"].sum() > 0) else ["Harmonized count"],
    )

    center = [(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2] if bbox else [20, 0]
    m = folium.Map(location=center, zoom_start=5 if bbox else 2, tiles=None)

    if basemap_choice == "CartoDB dark_matter":
        folium.TileLayer("CartoDB dark_matter", name="Dark Matter").add_to(m)
    elif basemap_choice == "Esri Satellite":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite",
        ).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # Highlight selected country boundary
    if poly_geojson is not None:
        folium.GeoJson(
            poly_geojson,
            name="Region boundary",
            style_function=lambda _: {
                "fillColor": "transparent",
                "color": "#ff7043",
                "weight": 2,
                "dashArray": "5, 5",
            },
        ).add_to(m)

    if sub is not None and not sub.empty:
        val_col = "weighted_frp" if weight_metric == "Fire Radiative Power (FRP)" else "weight"

        if map_mode == "Density grid":
            cell = 0.05

            def _agg_grid(c):
                return (
                    sub.assign(
                        glat=(sub["latitude"] // c) * c + c / 2,
                        glon=(sub["longitude"] // c) * c + c / 2,
                    )
                    .groupby(["glat", "glon"], as_index=False)
                    .agg(
                        score=(val_col, "sum"),
                        detections=("weight", "sum"),
                        total_frp=("frp", "sum"),
                        top_type=("fire_type", lambda s: s.mode().iloc[0]),
                    )
                )

            grid = _agg_grid(cell)
            while len(grid) > 1500:
                cell *= 2
                grid = _agg_grid(cell)

            max_score = max(1.0, float(grid["score"].max()))
            norm = colors.LogNorm(vmin=1.0, vmax=max_score)
            cmap = colormaps["YlOrRd"]
            radius_m = int(cell * 111320 * 0.58)

            for r in grid.itertuples():
                tip = (
                    f"<b>Detections:</b> {r.detections:,.0f}<br>"
                    f"<b>Total FRP:</b> {r.total_frp:,.1f} MW<br>"
                    f"<b>Dominant type:</b> {r.top_type}"
                )
                folium.Circle(
                    location=[r.glat, r.glon],
                    radius=radius_m,
                    color=colors.to_hex(cmap(norm(max(1.0, r.score)))),
                    fill=True,
                    fill_opacity=0.62,
                    weight=0,
                    tooltip=folium.Tooltip(tip),
                ).add_to(m)
            st.caption(f"Grid resolution: ~{cell:.2f}° (~{int(cell * 111)} km) · Hover any cell for telemetry.")

        elif map_mode == "Clustered points":
            pts = sub.sample(min(len(sub), 25000), random_state=42)
            coords = pts[["latitude", "longitude"]].values.tolist()
            FastMarkerCluster(coords).add_to(m)
            if len(sub) > 25000:
                st.caption("Showing 25,000 sampled points in interactive clusters.")

        elif map_mode == "Thermal heatmap":
            pts = sub.sample(min(len(sub), 30000), random_state=42)
            heat_data = list(
                zip(
                    pts["latitude"],
                    pts["longitude"],
                    np.log1p(pts[val_col]),
                )
            )
            HeatMap(heat_data, radius=14, blur=18, min_opacity=0.35).add_to(m)

    if bbox:
        m.fit_bounds([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])

    st_folium(m, height=600, use_container_width=True, returned_objects=[], key="main_map")

# ---- TAB 2: SEASONALITY & ANOMALIES --------------------------------------
with tab_season:
    if sub is None or sub.empty:
        st.info("Select a region with fire data to view multi-year seasonality.")
    else:
        st.subheader(f"Burning Activity Calendar — {match.title() if match else 'Dataset'}")
        monthly = sub.groupby(["year", "month"])["weight"].sum().reset_index(name="count")
        pivot = monthly.pivot(index="year", columns="month", values="count").reindex(columns=range(1, 13)).fillna(0)
        pivot.columns = [calendar.month_abbr[mo] for mo in pivot.columns]

        heat = px.imshow(
            pivot,
            color_continuous_scale="Inferno",
            labels=dict(x="Month", y="Year", color="Hotspots"),
            aspect="auto",
            text_auto=".0f",
        )
        heat.update_yaxes(type="category")
        heat.update_layout(height=max(280, 52 * len(pivot) + 120))
        st.plotly_chart(heat, use_container_width=True)

        # Monthly climatology envelope (Min / Median / Max across years)
        if len(pivot) >= 2:
            st.subheader("Seasonal Regime (Historical Envelope)")
            clim = pd.DataFrame(
                {
                    "Month": pivot.columns,
                    "Median": pivot.median(axis=0).values,
                    "Min": pivot.min(axis=0).values,
                    "Max": pivot.max(axis=0).values,
                }
            )
            fig_clim = go.Figure()
            fig_clim.add_trace(
                go.Scatter(
                    x=clim["Month"], y=clim["Max"], mode="lines",
                    line=dict(width=0), showlegend=False, name="Max",
                )
            )
            fig_clim.add_trace(
                go.Scatter(
                    x=clim["Month"], y=clim["Min"], mode="lines",
                    line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(255, 112, 67, 0.2)", name="Historical Min–Max Range",
                )
            )
            fig_clim.add_trace(
                go.Scatter(
                    x=clim["Month"], y=clim["Median"], mode="lines+markers",
                    line=dict(color="#ff7043", width=3), name="Historical Median",
                )
            )
            fig_clim.update_layout(yaxis_title="Harmonized Detections", height=340)
            st.plotly_chart(fig_clim, use_container_width=True)

# ---- TAB 3: FIRE INTENSITY & DIURNAL -------------------------------------
with tab_intensity:
    if sub is None or sub.empty:
        st.info("Select a region with fire data to analyze radiative power and source types.")
    else:
        c_left, c_right = st.columns(2)

        with c_left:
            st.subheader("Detections by Thermal Source Type")
            by_type = (
                sub.groupby(["year", "fire_type"])["weight"]
                .sum()
                .reset_index(name="detections")
            )
            fig_type = px.bar(
                by_type,
                x="year",
                y="detections",
                color="fire_type",
                barmode="stack",
                labels={"year": "Year", "detections": "Harmonized Detections", "fire_type": "Source Type"},
            )
            fig_type.update_xaxes(type="category")
            st.plotly_chart(fig_type, use_container_width=True)

        with c_right:
            st.subheader("Diurnal Cycle (Day vs. Night Passes)")
            by_dn = (
                sub.groupby(["month", "period"])["weight"]
                .sum()
                .reset_index(name="detections")
            )
            by_dn["Month"] = by_dn["month"].map(lambda m: calendar.month_abbr[m])
            fig_dn = px.bar(
                by_dn,
                x="Month",
                y="detections",
                color="period",
                barmode="group",
                category_orders={"Month": list(calendar.month_abbr[1:])},
                labels={"detections": "Harmonized Detections", "period": "Overpass"},
            )
            st.plotly_chart(fig_dn, use_container_width=True)

        if sub["frp"].sum() > 0:
            st.subheader("Monthly Fire Radiative Power (Total MW)")
            frp_mo = (
                sub.groupby(["year", "month"])["weighted_frp"]
                .sum()
                .reset_index()
            )
            frp_mo["Date"] = pd.to_datetime(
                frp_mo["year"].astype(str) + "-" + frp_mo["month"].astype(str).str.zfill(2) + "-01"
            )
            fig_frp = px.area(
                frp_mo,
                x="Date",
                y="weighted_frp",
                labels={"weighted_frp": "Harmonized FRP (MW)"},
                color_discrete_sequence=["#ff5722"],
            )
            st.plotly_chart(fig_frp, use_container_width=True)

# ---- TAB 4: HARMONIZATION & EXPORT ---------------------------------------
with tab_qa:
    if sub is None or sub.empty:
        st.info("Select a region with fire data to inspect sensor calibration and export records.")
    else:
        st.subheader("MODIS → VIIRS Cross-Sensor Calibration")
        st.markdown(
            f"Because VIIRS (375 m resolution) detects significantly more small thermal anomalies "
            f"than MODIS (1 km resolution), raw multi-decade counts create an artificial jump at the "
            f"sensor transition. Over the overlap period, MODIS detections are scaled by **×{factor:.2f}** "
            f"and superseded by VIIRS once full coverage begins."
        )
        yearly_sensor = (
            sub.groupby(["year", "sensor"])["weight"]
            .sum()
            .reset_index(name="harmonized_detections")
        )
        fig_cal = px.line(
            yearly_sensor,
            x="year",
            y="harmonized_detections",
            color="sensor",
            markers=True,
            color_discrete_map={"MODIS": "#ef5350", "VIIRS": "#42a5f5"},
            labels={"year": "Year", "harmonized_detections": "Annual Harmonized Detections"},
        )
        st.plotly_chart(fig_cal, use_container_width=True)

        export_df = sub.drop(columns=["weight", "weighted_frp", "year", "month"], errors="ignore")
        st.download_button(
            "⬇️ Download Filtered & Enriched Dataset (CSV)",
            export_df.to_csv(index=False),
            file_name=f"{(match or 'region').replace(' ', '_')}_hotspots_enriched.csv",
            mime="text/csv",
        )
