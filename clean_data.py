import pandas as pd

INPUT = "viirs-jpss1_2020_Egypt.csv"
OUTPUT = "viirs-jpss1_2020_Egypt_clean.csv"

df = pd.read_csv(INPUT)

# Combine acq_date (YYYY-MM-DD) and acq_time (HHMM, stored as int so leading
# zeros are lost) into a proper datetime column.
time_str = df["acq_time"].astype(str).str.zfill(4)
df["acq_datetime"] = pd.to_datetime(
    df["acq_date"] + " " + time_str.str[:2] + ":" + time_str.str[2:],
    errors="coerce",
    format="%Y-%m-%d %H:%M",
)
assert df["acq_datetime"].isna().sum() == 0, "some acq_datetime values failed to parse"

# Sensible dtypes
df["confidence"] = df["confidence"].astype("category")
df["daynight"] = df["daynight"].astype("category")

# Low-value constant columns are kept but flagged as category
df["satellite"] = df["satellite"].astype("category")
df["instrument"] = df["instrument"].astype("category")

# Sort chronologically for easier time-series work
df = df.sort_values("acq_datetime").reset_index(drop=True)

df.to_csv(OUTPUT, index=False)
print(f"Wrote {len(df)} rows to {OUTPUT}")
print(df.dtypes)
