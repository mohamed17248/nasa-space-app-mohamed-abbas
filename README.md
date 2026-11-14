# Fire Activity Explorer — NASA Space Apps

Interactive dashboard for exploring satellite fire-hotspot detections
(NASA FIRMS: MODIS Terra/Aqua + VIIRS Suomi-NPP), harmonized across sensors.

## Features

- Searchable world map of countries/regions (Natural Earth boundaries)
- Per-region fire data via `data/<region>.parquet` or `.csv` (FIRMS format)
- Folium map with Points / Density-grid views, colored by sensor
- Burning-activity calendar (year × month heatmap)
- MODIS→VIIRS harmonization: scale factor computed from overlap years
- "Latest month vs historical median" anomaly metric (same day-range)
- CSV export of filtered hotspots

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.12. Open http://localhost:8501.

## Data

Fire detections are loaded from `data/<region>.parquet|.csv`, named after
the searchable region (e.g. `data/egypt.parquet`). Expected columns follow
the FIRMS export format: `latitude`, `longitude`, `acq_date`, plus optional
`satellite`, `instrument`, `confidence`, `frp`, `daynight`, `type`.

Processing scripts (`build_full.py`, `analyze.py`, `match_and_map.py`,
`nasa_heatmap.py`) regenerate the parquet files from raw FIRMS CSVs in
`MODIS/` and `VIIRS/`.

## Contributors

- mohamed17248
- queriesonhold
