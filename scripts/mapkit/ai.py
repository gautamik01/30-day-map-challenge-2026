"""
ai.py : the AI series template, "The Cloud Has a Footprint".

The idea: data centers stay constant, and each map adds ONE thing they press on
(power, water, homes, heat, flood...). These helpers do the repeat work:

    dc = ai.load_datacenters()                    # every data center (PNNL atlas)
    big = ai.load_ai_campuses()                   # the giant AI campuses, built + coming (Epoch AI)
    dc = ai.clip(dc, "dc_nova")                   # zoom to a state list or a metro
    counties = ai.load_counties()                 # Census county shapes
    counties = ai.count_in(counties, dc)          # how many per county
    hexes = ai.hexbin(dc, size_km=50)             # hexagon density
    dc = ai.attach(dc, water_stress, ["bws_label"])   # give each point an overlay's value
    dc = ai.count_near(dc, homes, km=1.6)         # e.g. homes within a mile
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box

from .paths import RAW, PROCESSED, EQUAL_AREA, WGS84

# ---------------------------------------------------------------- places
FOCUS_STATES = ["NY", "PA", "CA", "TX", "WA", "DC"]
DC_METRO_STATES = ["DC", "VA", "MD"]          # the DC job market spills into NoVA and MD

# Rough bounding boxes (lon_min, lat_min, lon_max, lat_max). Tweak freely.
REGIONS = {
    "us48":     (-125.0, 24.4, -66.9, 49.5),
    "nyc":      (-74.30, 40.45, -73.65, 40.95),
    "philly":   (-75.35, 39.85, -74.95, 40.15),
    "dc_nova":  (-77.75, 38.70, -76.90, 39.15),   # includes Ashburn / Loudoun County
    "sf_bay":   (-122.60, 37.20, -121.75, 37.95),
    "dallas":   (-97.40, 32.55, -96.55, 33.10),
    "austin":   (-98.00, 30.10, -97.55, 30.55),
    "seattle":  (-122.50, 47.35, -122.10, 47.80),
    "quincy_wa": (-120.10, 47.10, -119.70, 47.35),  # central WA, hydro-powered cluster
}

CENSUS_COUNTIES = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip"
CENSUS_STATES = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"


# ---------------------------------------------------------------- loading
def _find(pattern):
    hits = sorted(RAW.glob(pattern))
    return hits[0] if hits else None


def load_datacenters(path=None, states=None):
    """
    Read the PNNL / IM3 Open Source Data Center Atlas (CSV or GeoPackage) from data/raw.
    Works with the original CSV and the enriched *_power_sources.csv.
    Returns one point per facility (buildings/campuses become their centroid), WGS84.
    """
    if path is None:
        path = (_find("*data_center_atlas*power_sources*.csv") or _find("*data_center_atlas*.csv")
                or _find("*data_center_atlas*.gpkg") or _find("*datacenter*.csv"))
    if path is None:
        raise FileNotFoundError(
            "No data center file in data/raw. Download the IM3 Data Center Atlas CSV "
            "(see README, 'Data') and drop it into data/raw/.")
    path = Path(path)
    if not path.exists():
        path = RAW / path

    if str(path).endswith(".gpkg"):
        layers = gpd.list_layers(path)["name"].tolist() if hasattr(gpd, "list_layers") else [None]
        frames = [gpd.read_file(path, layer=l) for l in layers]
        gdf = pd.concat(frames, ignore_index=True).to_crs(WGS84)
        gdf["geometry"] = gdf.geometry.representative_point()
    else:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        lon = next(c for c in df.columns if c in ("lon", "longitude", "x"))
        lat = next(c for c in df.columns if c in ("lat", "latitude", "y"))
        df = df.dropna(subset=[lon, lat])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=WGS84)

    if "id" in gdf.columns:
        gdf = gdf.drop_duplicates(subset="id")
    for col in ("name", "operator"):
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna("Unnamed" if col == "name" else "Unknown")
    if "sqft" in gdf.columns:
        gdf["sqft"] = pd.to_numeric(gdf["sqft"], errors="coerce")
    if states:
        gdf = gdf[gdf["state_abb"].isin(states)]
    return gdf.reset_index(drop=True)


def load_ai_campuses(country="United States", geocode=True, check=True):
    """
    Epoch AI 'Frontier Data Centers': the big AI campuses, built and coming.
    Needs data/raw/data_centers.csv and data/raw/data_center_timelines.csv (see README).

    Adds:
      power_now_mw      power today
      power_peak_mw     the most power the site reaches in Epoch's timeline (incl. projections)
      peak_date         when it gets there
      status            Operating | Expanding | Under construction
      water_mgd         water use (million gallons/day) where Epoch has it, mostly blank
    Addresses are geocoded once with OpenStreetMap and cached in data/processed (committed).
    """
    main = pd.read_csv(RAW / "data_centers.csv")
    tl = pd.read_csv(RAW / "data_center_timelines.csv")
    main = main[main["Country"] == country].copy() if country else main.copy()
    tl["Date"] = pd.to_datetime(tl["Date"], errors="coerce")
    today = pd.Timestamp.today().normalize()

    rows = []
    for _, r in main.iterrows():
        t = tl[tl["Data center"] == r["Name"]].sort_values("Date")
        now = pd.to_numeric(r.get("Current power (MW)"), errors="coerce")
        p = pd.to_numeric(t["Power (MW)"], errors="coerce")
        peak = p.max() if len(p.dropna()) else now
        peak_date = t.loc[p.idxmax(), "Date"] if len(p.dropna()) else pd.NaT
        water = pd.to_numeric(t["Water use (MGD)"], errors="coerce").dropna()
        if pd.isna(now) or now == 0:
            status = "Under construction"
        elif pd.notna(peak) and peak > now * 1.05 and pd.notna(peak_date) and peak_date > today:
            status = "Expanding"
        else:
            status = "Operating"
        rows.append(dict(name=r["Name"], owner=str(r.get("Owner", "")).split(" #")[0] or "Unknown",
                         users=str(r.get("Users", "") if pd.notna(r.get("Users")) else "").split(" #")[0],
                         address=r.get("Address"), power_now_mw=now, power_peak_mw=peak,
                         peak_date=peak_date.date().isoformat() if pd.notna(peak_date) else None,
                         status=status, water_mgd=water.iloc[-1] if len(water) else None,
                         h100e=pd.to_numeric(r.get("Current H100 equivalents"), errors="coerce")))
    df = pd.DataFrame(rows)

    cache = PROCESSED / "ai_campuses_geocoded.csv"
    known = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["name", "address", "lat", "lon", "placement"])
    todo = df[~df["name"].isin(known["name"])]
    if geocode and len(todo):
        import time
        import osmnx as ox
        print(f"geocoding {len(todo)} addresses with OpenStreetMap (one time, ~1 per second) ...")
        new = []
        for _, r in todo.iterrows():
            addr = r["address"] if isinstance(r["address"], str) and r["address"].strip() else None
            lat = lon = None
            if addr is None:
                print(f"  no address in Epoch's file: {r['name']} -> add lat/lon by hand in {cache.name}")
            else:
                parts = [p.strip() for p in addr.replace("\n", " ").split(",")]
                for q in [addr, ", ".join(parts[-3:]), ", ".join(parts[-2:])]:
                    try:
                        lat, lon = ox.geocode(q)
                        break
                    except Exception:
                        time.sleep(1)
                if lat is None:
                    print(f"  could not place: {r['name']} ({addr}) -> add lat/lon by hand in {cache.name}")
            new.append(dict(name=r["name"], address=addr, lat=lat, lon=lon, placement="address" if lat else None))
            time.sleep(1)
        known = pd.concat([known, pd.DataFrame(new)], ignore_index=True)
        known.to_csv(cache, index=False)

    cols = ["name", "lat", "lon"] + (["placement"] if "placement" in known else [])
    df = df.merge(known[cols], on="name", how="left")
    missing = df[df["lat"].isna()]
    if len(missing):
        print(f"{len(missing)} campus(es) left off the map until you add lat/lon in {cache.name}: "
              + ", ".join(missing["name"]))
    df = df.dropna(subset=["lat", "lon"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=WGS84)
    gdf["state_abb"] = gdf["address"].str.extract(r",\s*([A-Z]{2})\s*\d{5}")[0]
    if check:
        _check_states(gdf, cache.name)
    return gdf.reset_index(drop=True)


def _check_states(gdf, cache_name):
    """Warn about any campus that lands outside the state named in its address (wrong geocoder match)."""
    try:
        states = load_states()[["state_abb", "state_name", "geometry"]]
    except Exception as e:
        print("state check skipped:", e)
        return
    g = gdf
    if "placement" in g:
        g = g[~g["placement"].fillna("").str.startswith("approximate")]   # placed by hand on purpose
    j = gpd.sjoin(g[["name", "address", "lat", "lon", "geometry"]], states, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    import re
    names = set(states["state_name"]) | set(states["state_abb"])
    names_in = lambda a: any(re.search(rf"\b{re.escape(x)}\b", a) for x in names) if isinstance(a, str) else False
    bad = [names_in(a) and not ((isinstance(s, str) and re.search(rf"\b{s}\b", a)) or (isinstance(n, str) and n in a))
           for a, s, n in zip(j["address"], j["state_abb"], j["state_name"])]
    if any(bad):
        print(f"check these in {cache_name}, they sit outside the state in their address:")
        print(j.loc[bad, ["name", "address", "lat", "lon", "state_abb"]].to_string())
    else:
        print("location check: every AI campus sits inside the state named in its address")


def _cached(url, name):
    local = RAW / name
    if not local.exists():
        print(f"downloading {name} (one time) ...")
        gpd.read_file(url).to_file(local, driver="GPKG")
    return gpd.read_file(local)


def load_counties():
    c = _cached(CENSUS_COUNTIES, "census_counties_2023.gpkg").to_crs(WGS84)
    return c.rename(columns={"STUSPS": "state_abb", "NAME": "county_name", "GEOID": "geoid"})


def load_states():
    s = _cached(CENSUS_STATES, "census_states_2023.gpkg").to_crs(WGS84)
    return s.rename(columns={"STUSPS": "state_abb", "NAME": "state_name"})


def load_layer(name, **kw):
    """Any extra overlay you dropped into data/raw (shp, gpkg, geojson, ...)."""
    return gpd.read_file(RAW / name, **kw).to_crs(WGS84)


# ---------------------------------------------------------------- filtering
def clip(gdf, where):
    """where = a REGIONS key ('dc_nova'), a list of state codes (['TX','WA']), or a bbox tuple."""
    if isinstance(where, (list, tuple)) and where and isinstance(where[0], str):
        col = "state_abb" if "state_abb" in gdf.columns else "STUSPS"
        return gdf[gdf[col].isin(where)].reset_index(drop=True)
    bbox = REGIONS[where] if isinstance(where, str) else where
    return gpd.clip(gdf, box(*bbox)).reset_index(drop=True)


# ---------------------------------------------------------------- analysis
def count_in(polygons, points, name="n_datacenters", sum_col=None):
    """Count points (and optionally sum a column like sqft) inside each polygon."""
    j = gpd.sjoin(points[[c for c in [sum_col] if c] + ["geometry"]], polygons[["geometry"]],
                  predicate="within", how="inner")
    out = polygons.copy()
    out[name] = j.groupby("index_right").size().reindex(out.index).fillna(0).astype(int)
    if sum_col:
        out[f"{sum_col}_total"] = j.groupby("index_right")[sum_col].sum().reindex(out.index).fillna(0)
    return out


def hexbin(points, size_km=50, bounds_gdf=None, weight=None, keep_empty=False):
    """Hexagon grid (equal-area) with a count of points per hex. size_km = hex width, flat to flat."""
    pts = points.to_crs(EQUAL_AREA)
    ref = (bounds_gdf.to_crs(EQUAL_AREA) if bounds_gdf is not None else pts)
    minx, miny, maxx, maxy = ref.total_bounds
    r = size_km * 1000 / math.sqrt(3)          # circumradius from flat-to-flat width
    w, h = 1.5 * r, math.sqrt(3) * r
    cells = []
    for i, x in enumerate(np.arange(minx - r, maxx + r, w)):
        off = h / 2 if i % 2 else 0
        for y in np.arange(miny - h, maxy + h, h):
            cy = y + off
            cells.append(Polygon([(x + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
                                  for a in range(0, 360, 60)]))
    grid = gpd.GeoDataFrame(geometry=cells, crs=EQUAL_AREA)
    grid = count_in(grid, pts, name="count", sum_col=weight)
    if bounds_gdf is not None:
        grid = grid[grid.intersects(ref.union_all() if hasattr(ref, "union_all") else ref.unary_union)]
    if not keep_empty:
        grid = grid[grid["count"] > 0]
    return grid.to_crs(WGS84).reset_index(drop=True)


def attach(points, polygons, columns):
    """Give each point the value(s) of the polygon it sits in (e.g. water stress, flood zone)."""
    j = gpd.sjoin(points, polygons[columns + ["geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated(keep="first")].drop(columns="index_right")
    return j


def count_near(points, others, km=1.6, name=None):
    """How many `others` (homes, schools, rivers...) within `km` of each point."""
    name = name or f"n_within_{km}km"
    p = points.to_crs(EQUAL_AREA)
    o = others.to_crs(EQUAL_AREA)
    buf = p[["geometry"]].copy()
    buf["geometry"] = p.buffer(km * 1000)
    j = gpd.sjoin(o[["geometry"]], buf, predicate="within")
    out = points.copy()
    out[name] = j.groupby("index_right").size().reindex(out.index).fillna(0).astype(int)
    return out


def distance_to_nearest(points, others, name="dist_km"):
    """Distance (km) from each point to the nearest feature in `others` (a city, a river...)."""
    p = points.to_crs(EQUAL_AREA)
    o = others.to_crs(EQUAL_AREA)
    j = gpd.sjoin_nearest(p, o[["geometry"]], distance_col="_d", how="left")
    j = j[~j.index.duplicated(keep="first")]
    out = points.copy()
    out[name] = (j["_d"] / 1000).round(1).values
    return out


def top_operators(points, n=6):
    """Most common operators, everyone else becomes 'Other'. Handy for coloring points."""
    top = points["operator"].value_counts().head(n).index
    return points["operator"].where(points["operator"].isin(top), "Other")


def save(gdf, name):
    """Save a cleaned layer to data/processed (small files are committed, so maps rebuild)."""
    out = PROCESSED / f"{name}.geojson"
    gdf.to_crs(WGS84).to_file(out, driver="GeoJSON")
    return out


# ================================================================ the master table
# One table, every data center (existing + AI campuses). Each map ADDS columns to it,
# so by the end of a study the analysis is already in one place.
MASTER = PROCESSED / "datacenters_master.geojson"
MASTER_LOG = PROCESSED / "master_columns.json"
BASE_COLS = ["uid", "kind", "name", "operator", "status", "state_abb", "state", "county", "sqft",
             "power_now_mw", "power_peak_mw", "peak_date", "placement", "users", "source"]


def _slug(s):
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def build_master():
    """Combine the IM3 atlas + Epoch AI campuses into one table. Keeps any columns earlier maps added."""
    dc = load_datacenters()
    big = load_ai_campuses(check=False)
    e = gpd.GeoDataFrame(dict(
        uid="im3_" + dc["id"].astype(str), kind="existing", name=dc["name"], operator=dc["operator"],
        status="Existing", state_abb=dc.get("state_abb"), state=dc.get("state"), county=dc.get("county"),
        sqft=dc.get("sqft"), source="IM3 Data Center Atlas"), geometry=dc.geometry, crs=WGS84)
    a = gpd.GeoDataFrame(dict(
        uid="epoch_" + big["name"].map(_slug), kind="ai_campus", name=big["name"], operator=big["owner"],
        status=big["status"], state_abb=big["state_abb"], power_now_mw=big["power_now_mw"],
        power_peak_mw=big["power_peak_mw"], peak_date=big["peak_date"],
        placement=big.get("placement"), users=big["users"], source="Epoch AI"), geometry=big.geometry, crs=WGS84)
    m = pd.concat([e, a], ignore_index=True)
    m = gpd.GeoDataFrame(m, geometry="geometry", crs=WGS84)

    # fill state / county from Census shapes where the source left them blank
    try:
        st = load_states()[["state_abb", "state_name", "geometry"]]
        j = gpd.sjoin(m[["geometry"]], st, how="left", predicate="within")
        j = j[~j.index.duplicated()]
        m["state_abb"] = m["state_abb"].fillna(j["state_abb"])
        m["state"] = m["state"].fillna(j["state_name"])
    except Exception as ex:
        print("state fill skipped:", ex)

    if MASTER.exists():                      # keep what earlier maps added
        old = gpd.read_file(MASTER)
        extra = [c for c in old.columns if c not in BASE_COLS + ["geometry"]]
        if extra:
            m = m.merge(pd.DataFrame(old[["uid"] + extra]), on="uid", how="left")
    m.to_file(MASTER, driver="GeoJSON")
    print(f"master table: {len(m):,} rows ({(m.kind == 'existing').sum():,} existing, "
          f"{(m.kind == 'ai_campus').sum()} AI campuses) -> data/processed/{MASTER.name}")
    return m


def master(rebuild=False):
    """The master table. Built once from data/raw, then read from data/processed (fast, and on GitHub)."""
    if rebuild or not MASTER.exists():
        return build_master()
    return gpd.read_file(MASTER)


def update_master(gdf, columns, about="", source="", day=None):
    """
    Save new columns from today's map into the master table, matched on uid.
        dcs = ai.attach(dcs, water_stress, ["bws_label"])
        ai.update_master(dcs, ["bws_label"], about="Baseline water stress", source="WRI Aqueduct 4.0", day=20)
    Works with a clipped subset too (only those rows are updated).
    """
    import json
    from datetime import date
    m = master().set_index("uid")
    g = pd.DataFrame(gdf).drop_duplicates("uid").set_index("uid")
    for c in columns:
        if c not in m.columns:
            m[c] = None
        m.loc[g.index.intersection(m.index), c] = g[c]
    gpd.GeoDataFrame(m.reset_index(), geometry="geometry", crs=WGS84).to_file(MASTER, driver="GeoJSON")
    log = json.loads(MASTER_LOG.read_text()) if MASTER_LOG.exists() else {}
    for c in columns:
        log[c] = dict(about=about, source=source, day=day, updated=date.today().isoformat())
    MASTER_LOG.write_text(json.dumps(log, indent=2))
    print(f"master table: saved {', '.join(columns)} for {len(g):,} data centers")


def base_layers(dcs, extra_popup=None, show_existing=True):
    """
    The standard cast, same look on every map:
        Existing data centers · New AI campuses running · New AI campuses coming (glowing)
    extra_popup: master columns to add to the hover cards, e.g. {"bws_label": "Water stress"}
    """
    from .render import Layer
    from . import style as S
    d = dcs.copy()
    extra = extra_popup or {}
    d["label"] = d["name"].where(~d["name"].isin(["Unnamed", "", None]) & d["name"].notna(),
                                 d["operator"].where(~d["operator"].isin(["Unknown", ""]) & d["operator"].notna(),
                                                     "Data center"))
    d["operator_show"] = d["operator"].where(~d["operator"].isin(["Unknown"]), "")
    now, peak = pd.to_numeric(d.get("power_now_mw"), errors="coerce"), pd.to_numeric(d.get("power_peak_mw"), errors="coerce")
    d["power"] = [f"{a:,.0f} MW now · {b:,.0f} MW planned" if pd.notna(b) else "" for a, b in zip(now.fillna(0), peak)]
    d["full_by"] = pd.to_datetime(d.get("peak_date"), errors="coerce").dt.year.astype("Int64").astype(str).replace("<NA>", "")
    pl = d.get("placement", pd.Series("", index=d.index)).fillna("")
    d["where"] = [p.replace("approximate: ", "") + " (approx.)" if p.startswith("approximate") else "" for p in pl]

    ex = d[d["kind"] == "existing"]
    ai_ = d[d["kind"] == "ai_campus"]
    campus_popup = {"operator_show": "Owner", "status": "Status", "power": "Power", "full_by": "Full build",
                    "where": "Placed at", **extra}
    layers = [
        Layer(ex, S.LABELS["existing"], color=S.LOOK["existing"], radius=2.4, opacity=0.6, tooltip="label",
              popup={"operator_show": "Operator", "state": "State", **extra}, show=show_existing),
        Layer(ai_[ai_["status"] == "Operating"], S.LABELS["ai_running"], color=S.LOOK["ai_running"],
              size_column="power_peak_mw", radius=6, opacity=0.95, tooltip="label", popup=campus_popup),
        Layer(ai_[ai_["status"] != "Operating"], S.LABELS["ai_coming"], color=S.LOOK["ai_coming"],
              size_column="power_peak_mw", radius=6, opacity=0.95, glow=True, tooltip="label", popup=campus_popup),
    ]
    return [l for l in layers if len(l.gdf)]
