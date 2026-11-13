"""Clean VIIRS + MODIS Egypt fire hotspot data, compare, map, and heatmap."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- cleaning --

def load_and_clean(path, sensor):
    df = pd.read_csv(path)

    # acq_time is HHMM stored as int -> leading zeros lost (e.g. 957 -> 09:57)
    t = df["acq_time"].astype(str).str.zfill(4)
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"] + " " + t.str[:2] + ":" + t.str[2:],
        format="%Y-%m-%d %H:%M",
    )

    # Harmonize sensor-specific brightness column names
    rename = {"brightness": "bright_ti4", "bright_t31": "bright_ti5"}
    df = df.rename(columns=rename)

    # Harmonize confidence: MODIS is 0-100, VIIRS is l/n/h -> map to same scale
    viirs_conf = {"l": 33, "n": 66, "h": 100}
    if sensor == "VIIRS":
        df["confidence_pct"] = df["confidence"].map(viirs_conf)
    else:
        df["confidence_pct"] = df["confidence"]

    df["sensor"] = sensor
    df["month"] = df["acq_datetime"].dt.to_period("M").astype(str)

    keep = [
        "sensor", "acq_datetime", "month", "latitude", "longitude",
        "bright_ti4", "bright_ti5", "frp", "confidence", "confidence_pct",
        "daynight", "type",
    ]
    return df[keep].sort_values("acq_datetime").reset_index(drop=True)


viirs = load_and_clean("viirs-jpss1_2020_Egypt.csv", "VIIRS")
modis = load_and_clean("modis_2020_Egypt.csv", "MODIS")

viirs.to_csv("viirs_clean.csv", index=False)
modis.to_csv("modis_clean.csv", index=False)

combined = pd.concat([viirs, modis], ignore_index=True)
combined.to_csv("hotspots_combined.csv", index=False)

print(f"VIIRS: {len(viirs)} rows | MODIS: {len(modis)} rows")

# ------------------------------------------------------- monthly counts --

monthly = (
    combined.groupby(["month", "sensor"]).size().unstack(fill_value=0)
)
monthly.to_csv("monthly_counts.csv")
print(monthly.to_string())

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(monthly))
width = 0.38
ax.bar(x - width / 2, monthly["MODIS"], width, label="MODIS", color="tab:orange")
ax.bar(x + width / 2, monthly["VIIRS"], width, label="VIIRS", color="tab:blue")
ax.set_xticks(x)
ax.set_xticklabels(monthly.index, rotation=45)
ax.set_ylabel("Hotspot count")
ax.set_title("Monthly fire hotspot detections in Egypt (2020)")
ax.legend()
fig.tight_layout()
fig.savefig("monthly_counts.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ map --

fig, ax = plt.subplots(figsize=(8, 7))
colors = {"MODIS": "tab:orange", "VIIRS": "tab:blue"}
for sensor, g in combined.groupby("sensor"):
    ax.scatter(
        g["longitude"], g["latitude"], s=4, alpha=0.5,
        c=colors[sensor], label=f"{sensor} (n={len(g)})",
    )
ax.set_xlim(24.5, 36.5)
ax.set_ylim(21.5, 32.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Fire hotspot detections in Egypt (2020)")
ax.legend(markerscale=5)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("hotspot_map.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------- heatmap --

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
for ax, (name, data) in zip(
    axes, [("MODIS", modis), ("VIIRS", viirs), ("Combined", combined)]
):
    h = ax.hist2d(
        data["longitude"], data["latitude"], bins=60,
        range=[[24.5, 36.5], [21.5, 32.5]], cmap="YlOrRd",
    )
    ax.set_title(f"{name} (n={len(data)})")
    ax.set_xlabel("Longitude")
fig.axes[0].set_ylabel("Latitude")
fig.suptitle("Hotspot density heatmap, Egypt 2020")
fig.tight_layout()
fig.savefig("heatmap.png", dpi=150)
plt.close(fig)

print("Saved: monthly_counts.png, hotspot_map.png, heatmap.png")
