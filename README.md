# 30 Day Map Challenge 2026

Gautami Kasat · Weitzman School of Design, MUSA · [#30DayMapChallenge](https://30daymapchallenge.com)

**[See all the maps →](https://gautamik01.github.io/30-day-map-challenge-2026/output/)**

Two threads run through the month:

- **The Cloud Has a Footprint.** America's data centers, the ones running today and the giant AI campuses on the way,
  followed into the water, power and climate risk around them.
- **City Senses.** The lighter side: how New York, Philadelphia, DC, San Francisco, Austin and Seattle look,
  sound and taste on a map.

Every map is reproducible: one notebook per map in `scripts/`.

---

## Repository layout

```
scripts/
  _template_ai.ipynb       copy for an AI day
  _template_city.ipynb     copy for a city day
  dayNN_prompt_series.ipynb   one notebook per map
  mapkit/                  shared code: styles, data loading, map builders, publishing
data/
  raw/                     downloads (not on GitHub, see "Data" to rebuild)
  processed/               small cleaned files (on GitHub)
output/                    live interactive maps + gallery (GitHub Pages)
png_thumbnail/             one PNG per map
captions/                  one caption + MUSA blurb per map (_template.md to start)
PLANNER.md                 the month at a glance
```

## Setup (once)

```bash
conda activate geospatial
conda install -c conda-forge geopandas folium osmnx mapclassify matplotlib
pip install playwright
python -m playwright install chromium     # for automatic PNG thumbnails
```

**Mapbox:** make a free account at mapbox.com, copy your **Default public token** (starts with `pk.`)
and paste it into a file called `mapbox_token.txt` in this folder. It's git-ignored.
No token? Maps still build, using a free MapLibre + CARTO fallback.

**GitHub Pages:** Settings → Pages → Deploy from a branch → `main` / `(root)`.

## Daily loop

1. Open `scripts/_template_ai.ipynb` or `_template_city.ipynb` → **File → Save As** `dayNN_prompt_series.ipynb`
2. Edit cell 1 (settings), write cell 3 (analysis), **Run All**
3. Open the new file in `captions/`, write the post
4. GitHub Desktop → summary "Day NN" → **Commit to main** → **Push origin**
5. Post on LinkedIn, live link in the first comment
6. MUSA: copy the PNG + your `.md` into your fork of `weitzman-musa/30DayMapChallenge` → **Contribute → Open pull request**

## Data

Download into `data/raw/`:

| File | Source | Used for |
|---|---|---|
| `im3_open_source_data_center_atlas_v2026.02.09.csv` | [MSD-LIVE record](https://data.msdlive.org/records/p147s-4h760), open the folder under Files | Existing data centers (ODbL, built on OpenStreetMap) |
| `data_centers.csv` | [Epoch AI data page](https://epoch.ai/data/data-centers), Download section | Frontier AI campuses (CC BY 4.0) |
| `data_center_timelines.csv` | same page, same download | Build-out timelines, power today vs. at full build |

**For later maps** (also into `data/raw/`):

| File | Source | Used for |
|---|---|---|
| Aqueduct 4.0 download, **unzipped** (the `.gdb` folder) | [WRI Aqueduct 4.0 global maps data](https://www.wri.org/data/aqueduct-global-maps-40-data) | Day 3 · water stress (CC BY 4.0) |
| EIA-860M generator inventory (`.xlsx`) | [EIA-860M](https://www.eia.gov/electricity/data/eia860M/) → latest month's Excel | Day 12 · power plants (public domain) |
| `NRI_Table_Counties.csv` | FEMA RAPT page → *National Risk Index Dataset* → county Table (CSV) | Day 15 · climate risk (public domain) |

Census county and state shapes download automatically the first time they're needed.

City maps pull from OpenStreetMap automatically and cache in `data/raw/`.

## Data credits

- **Existing data centers:** Mongird, K., Vernon, C., Burleyson, C., Akdemir, K. Z., & Rice, J. (2026).
  *IM3 Open Source Data Center Atlas* (v2026.02.09). MSD-LIVE Data Repository. https://doi.org/10.57931/3017294.
  Derived from © OpenStreetMap contributors, Open Database License (ODbL). IM3 is led by Pacific Northwest
  National Laboratory with support from the U.S. Department of Energy's Office of Science.
- **Frontier AI data centers:** Epoch AI, *Frontier Data Centers*, https://epoch.ai/data/data-centers, CC BY 4.0.
- **City maps:** © OpenStreetMap contributors, ODbL.
- **Water stress:** Kuzma, S. et al. (2023). *Aqueduct 4.0*. World Resources Institute, CC BY 4.0.
- **Power plants:** U.S. Energy Information Administration, Form EIA-860M monthly generator inventory, public domain.
- **Climate risk:** FEMA, National Risk Index (county level), public domain.
- **Boundaries:** US Census Bureau, Cartographic Boundary Files.
- **Basemaps:** © Mapbox, © CARTO, © Esri, © OpenStreetMap contributors.

Each map's caption file in `captions/` lists the exact sources used for that map. Code: MIT.
