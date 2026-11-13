"""Match MODIS detections to VIIRS, overlay heatmap, and mapped comparison."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

viirs = pd.read_csv("viirs_clean.csv", parse_dates=["acq_datetime"])
modis = pd.read_csv("modis_clean.csv", parse_dates=["acq_datetime"])

# ------------------------------------------------- spatial-temporal match --
# For each MODIS detection find the nearest VIIRS detection on the same day.
# MODIS pixel ~1 km, VIIRS ~375 m, so 2 km is a reasonable same-fire radius.

RADIUS_KM = 2.0
EARTH_R = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


viirs["date"] = viirs["acq_datetime"].dt.date
modis["date"] = modis["acq_datetime"].dt.date
viirs_by_day = {d: g for d, g in viirs.groupby("date")}

matches = []
for _, m in modis.iterrows():
    day = viirs_by_day.get(m["date"])
    if day is None or len(day) == 0:
        matches.append(np.nan)
        continue
    dist = haversine_km(
        m["latitude"], m["longitude"],
        day["latitude"].to_numpy(), day["longitude"].to_numpy(),
    )
    i = dist.argmin()
    if dist[i] <= RADIUS_KM:
        dt_min = abs((day.iloc[i]["acq_datetime"] - m["acq_datetime"]).total_seconds()) / 60
        matches.append((dist[i], dt_min))
    else:
        matches.append(np.nan)

modis["viirs_match_km"] = [np.nan if x is np.nan else x[0] for x in matches]
modis["viirs_match_dt_min"] = [np.nan if x is np.nan else x[1] for x in matches]
matched = modis["viirs_match_km"].notna()

print(f"MODIS detections: {len(modis)}")
print(f"Matched to a VIIRS detection within {RADIUS_KM} km same day: {matched.sum()} "
      f"({matched.mean() * 100:.1f}%)")
print(f"Match distance: median {modis.loc[matched, 'viirs_match_km'].median():.2f} km, "
      f"mean {modis.loc[matched, 'viirs_match_km'].mean():.2f} km")
print(f"Time gap to matched VIIRS: median {modis.loc[matched, 'viirs_match_dt_min'].median():.0f} min, "
      f"mean {modis.loc[matched, 'viirs_match_dt_min'].mean():.0f} min")
print(f"Matched within 1 h: {(modis.loc[matched, 'viirs_match_dt_min'] <= 60).sum()}")

modis.to_csv("modis_matched.csv", index=False)

# ------------------------------------------------------- overlay heatmap --

combined = pd.concat([viirs, modis], ignore_index=True)

fig, ax = plt.subplots(figsize=(9, 7))
h = ax.hist2d(
    combined["longitude"], combined["latitude"], bins=80,
    range=[[24.5, 36.5], [21.5, 32.5]], cmap="YlOrRd",
)
plt.colorbar(h[3], ax=ax, label="detections per cell")
ax.scatter(
    modis["longitude"], modis["latitude"], s=12, c="navy",
    marker="x", linewidths=1.2, label=f"MODIS (n={len(modis)})",
    zorder=5,
)
ax.set_xlim(24.5, 36.5)
ax.set_ylim(21.5, 32.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Combined hotspot density with MODIS detections overlaid")
ax.legend(loc="lower left")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("heatmap_overlay.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------- basemap --

try:
    import geopandas as gpd

    world = gpd.read_file(
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/ne_110m_admin_0_countries.geojson"
    )
    egypt = world[world["ISO_A3"] == "EGY"]
    if egypt.empty:
        egypt = world[world["NAME"].str.contains("Egypt", case=False, na=False)]
    has_basemap = not egypt.empty
except Exception as e:
    print(f"Basemap unavailable ({e}); plotting without country outline.")
    egypt = None
    has_basemap = False

fig, ax = plt.subplots(figsize=(9, 7))
h = ax.hist2d(
    combined["longitude"], combined["latitude"], bins=80,
    range=[[24.5, 36.5], [21.5, 32.5]], cmap="YlOrRd",
)
plt.colorbar(h[3], ax=ax, label="detections per cell")
colors = {"MODIS": "tab:orange", "VIIRS": "tab:blue"}
for sensor, g in combined.groupby("sensor"):
    ax.scatter(
        g["longitude"], g["latitude"], s=3, alpha=0.4,
        c=colors[sensor], label=f"{sensor} (n={len(g)})",
    )
if has_basemap:
    egypt.boundary.plot(ax=ax, color="black", linewidth=1.2)
ax.set_xlim(24.5, 36.5)
ax.set_ylim(21.5, 32.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Hotspot density in Egypt (2020)")
ax.legend(markerscale=6, loc="lower left")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("heatmap_egypt.png", dpi=150)
plt.close(fig)

print("Saved: heatmap_overlay.png, heatmap_egypt.png")
