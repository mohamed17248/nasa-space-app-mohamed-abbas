"""NASA-style fire density heatmap for Egypt 2020.

- UTM zone 36N projection (EPSG:32636), Egypt's central zone
- Confidence-weighted detection density (low-confidence VIIRS/MODIS down-weighted)
- Gaussian-smoothed density grid, counts per 100 km^2
- Egypt boundary, graticule, scale bar, north arrow, colorbar
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter
from pyproj import Transformer
import geopandas as gpd

# ------------------------------------------------------------------ data --

viirs = pd.read_csv("viirs_clean.csv", parse_dates=["acq_datetime"])
modis = pd.read_csv("modis_clean.csv", parse_dates=["acq_datetime"])

# Weight each detection by confidence (0-100 scale -> 0.33-1.0)
for df in (viirs, modis):
    df["weight"] = df["confidence_pct"] / 100.0

lon = pd.concat([viirs["longitude"], modis["longitude"]]).to_numpy()
lat = pd.concat([viirs["latitude"], modis["latitude"]]).to_numpy()
w = pd.concat([viirs["weight"], modis["weight"]]).to_numpy()

# ------------------------------------------------------------- projection --

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
x, y = transformer.transform(lon, lat)

world = gpd.read_file(
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)
egypt = world[world["ISO_A3"] == "EGY"].to_crs("EPSG:32636")

# ------------------------------------------------------------ density grid --

# 5 km cells over Egypt's extent
res = 5000.0
pad = 60000.0  # 60 km padding around data
x0, x1 = x.min() - pad, x.max() + pad
y0, y1 = y.min() - pad, y.max() + pad
nx = int((x1 - x0) / res) + 1
ny = int((y1 - y0) / res) + 1

H, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[x0, x1], [y0, y1]], weights=w)
cell_km2 = (res / 1000.0) ** 2
density = gaussian_filter(H / cell_km2, sigma=2.0) * cell_km2  # -> detections / 100 km^2

# ------------------------------------------------------------------ plot --

fig, ax = plt.subplots(figsize=(10, 9))

pcm = ax.pcolormesh(
    np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1), density.T,
    cmap="inferno", norm=PowerNorm(gamma=0.5, vmin=0.01, vmax=np.percentile(density, 99.5)),
    shading="auto",
)
cbar = fig.colorbar(pcm, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Confidence-weighted detections per 100 km² (smoothed)")

egypt.boundary.plot(ax=ax, color="white", linewidth=1.5, alpha=0.9)

# Graticule: transform lon/lat gridlines into UTM
lon_lines = np.arange(25, 37, 1)
lat_lines = np.arange(22, 33, 1)
for lon0 in lon_lines:
    xs, ys = transformer.transform(
        np.full_like(lat_lines, lon0, dtype=float), lat_lines.astype(float)
    )
    ax.plot(xs, ys, color="white", linewidth=0.4, alpha=0.4, linestyle=":")
for lat0 in lat_lines:
    xs, ys = transformer.transform(
        lon_lines.astype(float), np.full_like(lon_lines, lat0, dtype=float)
    )
    ax.plot(xs, ys, color="white", linewidth=0.4, alpha=0.4, linestyle=":")

# Tick labels in lon/lat at projected positions
tick_lon = np.arange(26, 36, 2)
tick_lat = np.arange(22, 33, 2)
xt, yt = transformer.transform(tick_lon, np.full_like(tick_lon, 26.0))
ax.set_xticks(xt)
ax.set_xticklabels([f"{v}°E" for v in tick_lon])
xl, yl = transformer.transform(np.full_like(tick_lat, 25.0), tick_lat)
ax.set_yticks(yl)
ax.set_yticklabels([f"{v}°N" for v in tick_lat])

# Scale bar (~200 km) at lower-left inside plot
cx = x0 + 0.12 * (x1 - x0)
cy = y0 + 0.06 * (y1 - y0)
bar_km = 200
ax.plot([cx, cx + bar_km * 1000], [cy, cy], color="white", linewidth=3)
ax.text(cx + bar_km * 500, cy + 4000, f"{bar_km} km", color="white",
        fontsize=9, ha="center", va="bottom")

# North arrow
ax.annotate("N", xy=(x1 - 0.06 * (x1 - x0), y1 - 0.05 * (y1 - y0)),
            xytext=(x1 - 0.06 * (x1 - x0), y1 - 0.14 * (y1 - y0)),
            color="white", fontsize=14, ha="center", va="center",
            arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))

ax.set_title("Fire hotspot density — Egypt, 2020\n"
             "VIIRS (Suomi-NPP) + MODIS (Terra/Aqua), confidence-weighted", fontsize=13)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")
ax.set_facecolor("#1a1a1a")
fig.patch.set_facecolor("#1a1a1a")

fig.tight_layout()
fig.savefig("nasa_heatmap.png", dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)

print(f"Wrote nasa_heatmap.png ({len(viirs) + len(modis)} detections)")
