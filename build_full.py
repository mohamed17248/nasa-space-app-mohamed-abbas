"""Build the full 2000-present Egypt fire dataset.

- Loads all FIRMS CSVs from MODIS/ and VIIRS/ folders
- Cleans + harmonizes both sensors into one schema
- Merged series: MODIS for <=2011, VIIRS (SNPP) for >=2012
- Writes hotspots_full.parquet for the Streamlit app + summary plots
"""

import glob
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ load --

def load_sensor(pattern, sensor):
    frames = []
    for path in sorted(glob.glob(pattern)):
        if " (" in path:  # skip browser-style duplicate downloads
            continue
        df = pd.read_csv(path)
        df["source_file"] = path
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # acq_time is HHMM stored as int -> leading zeros lost (e.g. 957 -> 09:57)
    t = df["acq_time"].astype(str).str.zfill(4)
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"] + " " + t.str[:2] + ":" + t.str[2:],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    bad = df["acq_datetime"].isna().sum()
    if bad:
        print(f"  WARNING {sensor}: {bad} rows failed datetime parse, dropping")
        df = df.dropna(subset=["acq_datetime"])

    # Harmonize sensor-specific column names
    df = df.rename(columns={"brightness": "bright_ti4", "bright_t31": "bright_ti5"})

    # Confidence -> 0-100 scale (VIIRS l/n/h, MODIS already 0-100)
    viirs_map = {"l": 33, "n": 66, "h": 100}
    if sensor == "VIIRS":
        df["confidence_pct"] = df["confidence"].map(viirs_map)
    else:
        df["confidence_pct"] = df["confidence"].astype(float)

    df["confidence"] = df["confidence"].astype(str)
    df["sensor"] = sensor
    df["year"] = df["acq_datetime"].dt.year
    df["month"] = df["acq_datetime"].dt.to_period("M").astype(str)
    return df


print("Loading MODIS files...")
modis = load_sensor("MODIS/*.csv", "MODIS")
print(f"  {len(modis)} rows, {modis['acq_datetime'].min()} -> {modis['acq_datetime'].max()}")

print("Loading VIIRS files...")
viirs = load_sensor("VIIRS/*.csv", "VIIRS")
print(f"  {len(viirs)} rows, {viirs['acq_datetime'].min()} -> {viirs['acq_datetime'].max()}")

# ------------------------------------------------------------------ merge --

full = pd.concat([modis, viirs], ignore_index=True)
full = full.drop_duplicates(
    subset=["latitude", "longitude", "acq_datetime", "sensor"]
).reset_index(drop=True)
print(f"Combined: {len(full)} rows "
      f"({full['sensor'].value_counts().to_dict()})")

# Harmonized series: MODIS <=2011, VIIRS >=2012
harmonized = pd.concat(
    [full[(full.sensor == "MODIS") & (full.year <= 2011)],
     full[(full.sensor == "VIIRS") & (full.year >= 2012)]],
    ignore_index=True,
).sort_values("acq_datetime").reset_index(drop=True)
print(f"Harmonized (MODIS<=2011 + VIIRS>=2012): {len(harmonized)} rows")

full.to_parquet("hotspots_full.parquet", index=False)
harmonized.to_parquet("hotspots_harmonized.parquet", index=False)

# ------------------------------------------------------- monthly counts --

monthly = (
    full.groupby(["month", "sensor"]).size().unstack(fill_value=0)
    .reindex(columns=["MODIS", "VIIRS"], fill_value=0)
)
monthly.index = pd.PeriodIndex(monthly.index, freq="M").to_timestamp()

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
axes[0].bar(monthly.index, monthly["MODIS"], width=25,
            color="tab:orange", label="MODIS (Terra+Aqua)")
axes[0].set_ylabel("detections / month")
axes[0].set_title("MODIS monthly fire hotspots, Egypt 2000–2025")
axes[0].legend()

axes[1].bar(monthly.index, monthly["VIIRS"], width=25,
            color="tab:blue", label="VIIRS (Suomi-NPP)")
axes[1].set_ylabel("detections / month")
axes[1].set_title("VIIRS monthly fire hotspots, Egypt 2012–2025")
axes[1].legend()
for ax in axes:
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig("monthly_counts_full.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------- yearly totals --

yearly = full.groupby(["year", "sensor"]).size().unstack(fill_value=0)
yearly.to_csv("yearly_counts.csv")
print(yearly.to_string())

print("Saved: hotspots_full.parquet, hotspots_harmonized.parquet, "
      "monthly_counts_full.png, yearly_counts.csv")
