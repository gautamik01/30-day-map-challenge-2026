"""
fetch_week3.py : trims the big downloads for the Day 13 to Day 23 maps into small files Claude can read.

    conda activate geospatial
    python scripts/fetch_week3.py day13 day15     (just those days)
    python scripts/fetch_week3.py                 (every step)

Days 13 and 15 only read files you downloaded into data/raw/. Days 17 to 19 download for you. Every step writes
small versions to data/processed/ (commit those). Safe to run again: finished steps are skipped,
and a failed step does not stop the others. Delete a file in data/processed/ to redo its step.

What each step needs in data/raw/
    day13  EIA-860M generator inventory (.xlsx), the one you used for Day 12
           USGS solar footprints (USPVDB), unzipped, any folder with "uspvdb" in its name
    day15  NOAA Coastal Relief Model export for Monterey Bay (exportImage.tiff, GeoTIFF)
    day17  nothing: downloads EIA's hourly grid file itself (big, about 150 MB, kept in data/raw)
    day18  nothing: downloads BTS transit stops, Amtrak, intercity terminals and Census counties itself
    day19  nothing: asks OpenStreetMap for Philadelphia's cafés (if it fails, the server is busy: run again later)
    day20  the WRI Aqueduct 4.0 folder you unzipped for the water map (the .gdb)
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapkit.paths import PROCESSED, RAW, WGS84, EQUAL_AREA  # noqa: E402

sys.stdout.reconfigure(errors="replace")
REPORT_FILE = PROCESSED / "fetch_report.json"
REPORT = json.loads(REPORT_FILE.read_text()) if REPORT_FILE.exists() else {}


# ------------------------------------------------------------------ helpers
def step(name, *outputs):
    """Run a step once. Skip it if its files exist. Keep going if it fails."""
    def wrap(fn):
        def run():
            files = [PROCESSED / o for o in outputs]
            if files and all(f.exists() for f in files):
                print(f"[skip] {name}: already done")
                return
            print(f"[....] {name}")
            t = time.time()
            try:
                info = fn() or {}
                REPORT[name] = {"status": "ok", "seconds": round(time.time() - t), **info}
                print(f"[ ok ] {name} ({round(time.time() - t)}s)")
                for k, v in info.items():
                    print(f"         {k}: {v}")
            except Exception as e:
                REPORT[name] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
            REPORT_FILE.write_text(json.dumps(REPORT, indent=2, default=str))
        run.step_name = name
        return run
    return wrap


def find(*patterns):
    """First file in data/raw (any depth) matching one of the patterns, newest first."""
    for pat in patterns:
        hits = sorted(RAW.rglob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        hits = [h for h in hits if h.is_file()]
        if hits:
            return hits[0]
    return None


def size_mb(path):
    return round(Path(path).stat().st_size / 1e6, 2)


# ------------------------------------------------------------------ Day 13 · EIA-860M
# Fuel groups, the same codes mapkit.ai uses for Day 12, so the two maps agree
EIA_CODES = {"NG": "Gas", "OG": "Gas", "BFG": "Gas", "LFG": "Gas", "OBG": "Gas",
             "BIT": "Coal", "SUB": "Coal", "LIG": "Coal", "ANT": "Coal", "RC": "Coal", "WC": "Coal", "SGC": "Coal",
             "NUC": "Nuclear", "WAT": "Hydro", "WND": "Wind", "SUN": "Solar", "MWH": "Storage",
             "DFO": "Oil", "RFO": "Oil", "JF": "Oil", "KER": "Oil", "PC": "Oil", "WO": "Oil", "SGP": "Oil"}


def _sheet(raw, prefix):
    name = next((k for k in raw if k.strip().lower().startswith(prefix)), None)
    if name is None:
        raise ValueError(f"No sheet starting with '{prefix}' in the 860M file. Sheets: {list(raw)}")
    df = raw[name]
    hdr = next(i for i in range(20) if df.iloc[i].astype(str).str.contains("Plant Name", case=False).any())
    df.columns = [str(c).strip() for c in df.iloc[hdr]]
    return df.iloc[hdr + 1:].reset_index(drop=True)


def _col(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    for n in names:                                   # loose match, e.g. "Planned Operation Year"
        hit = next((c for c in df.columns if n.lower() in c.lower()), None)
        if hit:
            return hit
    return None


def _generators(df, group):
    c = lambda *n: _col(df, *n)  # noqa: E731
    out = pd.DataFrame({
        "status_group": group,
        "status": df[c("Status")].astype(str).str.strip() if c("Status") else "",
        "plant_id": pd.to_numeric(df[c("Plant ID", "Plant Code")], errors="coerce"),
        "plant": df[c("Plant Name")].astype(str).str.strip(),
        "gen_id": df[c("Generator ID")].astype(str).str.strip() if c("Generator ID") else "",
        "state": df[c("Plant State", "State")].astype(str).str.strip(),
        "county": df[c("County")].astype(str).str.strip() if c("County") else "",
        "ba": df[c("Balancing Authority Code")].astype(str).str.strip() if c("Balancing Authority Code") else "",
        "tech": df[c("Technology")].astype(str).str.strip() if c("Technology") else "",
        "source": df[c("Energy Source Code")].astype(str).str.strip().str.upper(),
        "mw": pd.to_numeric(df[c("Nameplate Capacity (MW)", "Net Summer Capacity (MW)")], errors="coerce"),
        "year": pd.to_numeric(df[c("Planned Operation Year", "Operating Year")], errors="coerce"),
        "month": pd.to_numeric(df[c("Planned Operation Month", "Operating Month")], errors="coerce"),
        "lat": pd.to_numeric(df[c("Latitude")], errors="coerce"),
        "lon": pd.to_numeric(df[c("Longitude")], errors="coerce"),
    })
    ry = c("Planned Retirement Year")
    out["retire_year"] = pd.to_numeric(df[ry], errors="coerce") if ry else np.nan
    out["fuel"] = out["source"].map(EIA_CODES).fillna("Other")
    return out


@step("day13_eia860m", "eia860m_generators.csv")
def day13_eia():
    xl = find("*860M*.xlsx", "*860m*.xlsx", "*generator*inventory*.xlsx", "*Generator*Inventory*.xlsx",
              "*_generator20*.xlsx", "*generator*.xlsx", "*Generator*.xlsx")
    if xl is None:
        raise FileNotFoundError("No EIA-860M .xlsx in data/raw. Use the one from Day 12, or download it "
                                "from eia.gov/electricity/data/eia860m")
    print(f"         reading {xl.name} (a minute or two, it's a big Excel file) ...")
    raw = pd.read_excel(xl, sheet_name=None, header=None)
    g = pd.concat([_generators(_sheet(raw, "operating"), "operating"),
                   _generators(_sheet(raw, "planned"), "planned")], ignore_index=True)
    g = g[g["lat"].notna() & g["lon"].notna() & g["mw"].notna()]
    g["year"] = g["year"].astype("Int64")
    g["month"] = g["month"].astype("Int64")
    g["retire_year"] = g["retire_year"].astype("Int64")
    g.to_csv(PROCESSED / "eia860m_generators.csv", index=False)
    by = g.groupby(["status_group", "fuel"])["mw"].agg(["count", "sum"]).round(0)
    return {"file": xl.name, "generators": len(g), "size_mb": size_mb(PROCESSED / "eia860m_generators.csv"),
            "mw_by_status_and_fuel": {f"{s} {f}": f"{int(r['count'])} gens, {int(r['sum']):,} MW"
                                      for (s, f), r in by.iterrows() if f in ("Gas", "Solar", "Wind", "Storage")}}


# ------------------------------------------------------------------ Day 13 · USGS solar footprints
FOOTPRINT_MIN_MW = 20        # polygons kept only for farms this big or bigger (keeps the file small)


@step("day13_uspvdb", "uspvdb_sites.csv", "uspvdb_footprints.geojson")
def day13_uspvdb():
    src = None
    for folder in [p for p in RAW.rglob("*") if p.is_dir() and "uspvdb" in p.name.lower()] + [RAW]:
        src = next((f for f in folder.rglob("*.geojson") if "uspvdb" in str(f).lower()), None) or \
              next((f for f in folder.rglob("*.shp") if "uspvdb" in str(f).lower()), None)
        if src:
            break
    if src is None:
        raise FileNotFoundError("No USPVDB .geojson or .shp in data/raw. Download it from energy.usgs.gov/uspvdb "
                                "and unzip it into data/raw/")
    print(f"         reading {src.relative_to(RAW)} ...")
    g = gpd.read_file(src)
    if g.crs is None:
        g = g.set_crs(WGS84)
    ea = g.to_crs(EQUAL_AREA)
    g["area_m2_calc"] = ea.geometry.area.round(0)          # our own area, to check against USGS's p_area
    c = ea.geometry.representative_point().to_crs(WGS84)
    g["lon_c"], g["lat_c"] = c.x.round(5), c.y.round(5)

    sites = pd.DataFrame(g.drop(columns="geometry"))
    sites.to_csv(PROCESSED / "uspvdb_sites.csv", index=False)

    cap = next((k for k in ("p_cap_ac", "p_cap_dc") if k in g.columns), None)
    big = g if cap is None else g[pd.to_numeric(g[cap], errors="coerce").fillna(0) >= FOOTPRINT_MIN_MW]
    keep = [k for k in ("case_id", "eia_id", "p_name", "p_year", "p_state", "p_cap_ac", "p_cap_dc", "p_area",
                        "area_m2_calc") if k in big.columns]
    shp = big[keep + ["geometry"]].to_crs(EQUAL_AREA)
    shp["geometry"] = shp.geometry.simplify(3, preserve_topology=True)   # 3 m: invisible at any map scale we use
    shp = shp.to_crs(WGS84)
    shp.to_file(PROCESSED / "uspvdb_footprints.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)

    info = {"file": str(src.relative_to(RAW)), "columns": list(sites.columns), "sites": len(sites),
            f"footprints_kept_{FOOTPRINT_MIN_MW}MW_plus": len(shp),
            "size_mb_sites": size_mb(PROCESSED / "uspvdb_sites.csv"),
            "size_mb_footprints": size_mb(PROCESSED / "uspvdb_footprints.geojson")}
    if cap and "p_area" in g.columns:
        a = pd.to_numeric(g["p_area"], errors="coerce")
        info["area_check_median_ratio_ours_vs_usgs"] = round(float((g["area_m2_calc"] / a).median()), 3)
    return info


# ------------------------------------------------------------------ Day 15 · Monterey bathymetry
MAX_PIXELS_WIDE = 1600       # enough for a 1080 px wide map with room to crop


def _read_tiff(path):
    """Returns (z as float array, west, north, x_step, y_step, crs, nodata). Works with or without rasterio."""
    try:
        import rasterio
        with rasterio.open(path) as r:
            z = r.read(1).astype("float32")
            t = r.transform
            return z, t.c, t.f, t.a, -t.e, (r.crs.to_string() if r.crs else ""), r.nodata, r.count
    except ImportError:
        pass
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    bands = len(im.getbands())
    tags = im.tag_v2
    scale = tags.get(33550)                 # ModelPixelScaleTag
    tie = tags.get(33922)                   # ModelTiepointTag
    if not scale or not tie:
        raise ValueError("This TIFF has no map position (no GeoTIFF tags). Export again as GeoTIFF.")
    keys = tags.get(34735) or ()            # GeoKeyDirectory, used to spot Web Mercator
    crs = "EPSG:3857" if (3857 in keys or 102100 in keys) else ("EPSG:4326" if abs(scale[0]) < 1 else "projected")
    nodata = tags.get(42113)
    z = np.array(im if bands == 1 else im.getchannel(0), dtype="float32")
    west = tie[3] - tie[0] * scale[0]
    north = tie[4] + tie[1] * scale[1]
    return z, west, north, scale[0], scale[1], crs, (float(nodata) if nodata not in (None, "") else None), bands


@step("day15_monterey_bathy", "monterey_bathy.npz")
def day15():
    src = find("exportImage*.tif*", "*monterey*.tif*", "*crm*.tif*", "*CRM*.tif*")
    if src is None:
        raise FileNotFoundError("No exportImage.tiff in data/raw. Export the Coastal Relief Model box "
                                "from ncei.noaa.gov/maps/grid-extract as GeoTIFF.")
    z, west, north, dx, dy, crs, nodata, bands = _read_tiff(src)
    if bands > 1 or z.max() <= 255 and z.min() >= 0:
        raise ValueError(f"{src.name} looks like a picture ({bands} colour bands, values 0 to 255), not depths. "
                         "In Grid Extract pick 'Coastal Relief Model (CRM) Mosaic' (not Hillshade) and GeoTIFF.")
    if nodata is not None:
        z[z == nodata] = np.nan
    z[(z < -12000) | (z > 9000)] = np.nan
    h, w = z.shape
    step_px = max(1, int(np.ceil(w / MAX_PIXELS_WIDE)))
    if step_px > 1:                                          # block average, keeps the canyon walls honest
        hh, ww = h // step_px * step_px, w // step_px * step_px
        z = np.nanmean(z[:hh, :ww].reshape(hh // step_px, step_px, ww // step_px, step_px), axis=(1, 3))
        dx, dy = dx * step_px, dy * step_px
    h2, w2 = z.shape
    east, south = west + w2 * dx, north - h2 * dy
    np.savez_compressed(PROCESSED / "monterey_bathy.npz",
                        z=np.where(np.isnan(z), -32768, np.round(z)).astype("int16"),
                        bounds=np.array([west, south, east, north]), crs=np.array(crs),
                        nodata=np.array(-32768), source=np.array("NOAA NCEI Coastal Relief Model, Grid Extract"))
    sea = z[z < 0]
    return {"file": src.name, "crs": crs, "original_px": f"{w} x {h}", "saved_px": f"{w2} x {h2}",
            "pixel_size": f"{dx:.6g} x {dy:.6g} ({'degrees' if crs == 'EPSG:4326' else 'map units'})",
            "bounds_w_s_e_n": [round(v, 4) for v in (west, south, east, north)],
            "deepest_m": float(np.nanmin(z)), "highest_land_m": float(np.nanmax(z)),
            "share_sea": round(float(sea.size / np.isfinite(z).sum()), 3),
            "size_mb": size_mb(PROCESSED / "monterey_bathy.npz")}


# ------------------------------------------------------------------ download helpers (Days 17 to 19)
def _get(url, dest=None, params=None, timeout=300):
    import requests
    r = requests.get(url, params=params, timeout=timeout, stream=dest is not None,
                     headers={"User-Agent": "30DayMapChallenge (gautamik01)"})
    r.raise_for_status()
    if dest is None:
        return r
    tmp = Path(str(dest) + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def _arcgis(layer_url, fields="*", where="1=1", page=2000):
    """Every feature of an ArcGIS feature layer, in pages, as a GeoDataFrame (WGS84)."""
    parts, offset = [], 0
    while True:
        r = _get(f"{layer_url}/query", params={"where": where, "outFields": fields, "outSR": 4326, "f": "geojson",
                                                "resultOffset": offset, "resultRecordCount": page,
                                                "orderByFields": "OBJECTID"}).json()
        feats = r.get("features", [])
        if feats:
            parts.append(gpd.GeoDataFrame.from_features(feats, crs=WGS84))
        if len(feats) < page and not r.get("properties", {}).get("exceededTransferLimit"):
            break
        offset += len(feats)
        if offset % 20000 < page:
            print(f"         ... {offset:,} so far")
    return pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame(geometry=[], crs=WGS84)


BTS = "https://services.arcgis.com/xOi1kZaI0eWDREZv/ArcGIS/rest/services"
def _counties_l48():
    """Census counties for the Lower 48 (cached in data/raw by mapkit)."""
    from mapkit import ai
    c = ai.load_counties()
    return c[~c.state_abb.isin(["AK", "HI", "PR", "VI", "GU", "AS", "MP"])].to_crs(WGS84).reset_index(drop=True)


# ------------------------------------------------------------------ Day 17 · ERCOT hour by hour
EIA930 = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{y}_{h}.csv"
EIA930_TRY = [(2026, "Jul_Dec"), (2026, "Jan_Jun"), (2025, "Jul_Dec")]


@step("day17_ercot_hourly", "ercot_hourly_fuel.csv")
def day17():
    src = find("EIA930_BALANCE_*.csv")
    if src is None:
        for y, h in EIA930_TRY:
            dest = RAW / f"EIA930_BALANCE_{y}_{h}.csv"
            try:
                print(f"         downloading {dest.name} (big, a few minutes) ...")
                src = _get(EIA930.format(y=y, h=h), dest, timeout=900)
                break
            except Exception as e:
                print(f"         {dest.name} not available ({e}), trying an older one")
    if src is None:
        raise FileNotFoundError("Couldn't download EIA-930. On eia.gov/electricity/gridmonitor choose Download data > "
                                "Six-month files > Balance, save the .csv into data/raw/, then run again.")
    print(f"         reading {src.name} ...")
    df = pd.read_csv(src, low_memory=False, thousands=",")
    df = df[df["Balancing Authority"].astype(str).str.strip() == "ERCO"]
    base = {c.replace(" (Adjusted)", "") for c in df.columns
            if c.startswith("Net Generation (MW) from") and "Imputed" not in c}
    gen_cols = [f"{c} (Adjusted)" if f"{c} (Adjusted)" in df.columns else c for c in sorted(base)]   # EIA's cleaned numbers when they exist
    adjusted = [c for c in gen_cols if "(Adjusted)" in c]
    def fuel_of(c):
        c = c.lower()
        for key, name in [("solar", "Solar"), ("wind", "Wind"), ("natural gas", "Gas"), ("coal", "Coal"),
                          ("nuclear", "Nuclear"), ("hydro", "Hydro"), ("storage", "Storage"), ("battery", "Storage"),
                          ("petroleum", "Oil")]:
            if key in c:
                return name
        return "Other"
    out = pd.DataFrame({"utc_end": pd.to_datetime(df["UTC Time at End of Hour"]),
                        "local_date": df["Data Date"], "hour": df["Hour Number"]})
    for c in gen_cols:
        out[fuel_of(c)] = out.get(fuel_of(c), 0) + pd.to_numeric(df[c], errors="coerce").fillna(0).values
    if "Demand (MW)" in df.columns:
        out["demand"] = pd.to_numeric(df["Demand (MW)"], errors="coerce").values
    out = out.sort_values("utc_end")
    out.to_csv(PROCESSED / "ercot_hourly_fuel.csv", index=False)
    day = out.groupby("local_date")[["Solar"]].sum()
    return {"file": src.name, "adjusted_numbers": bool(adjusted), "hours": len(out), "from": str(out.utc_end.min()), "to": str(out.utc_end.max()),
            "fuel_columns": [c for c in out.columns if c not in ("utc_end", "local_date", "hour", "demand")],
            "sunniest_day": str(day.Solar.idxmax()), "peak_solar_mw": int(out.Solar.max())}


# ------------------------------------------------------------------ Day 18 · road only
@step("day18_road_only", "counties_connectivity.geojson", "amtrak_routes.geojson")
def day18():
    counties = _counties_l48()[["geoid", "state_abb", "county_name", "geometry"]]
    print(f"         {len(counties):,} counties")

    print("         National Transit Map stops (fixed-route transit, a few hundred thousand points) ...")
    stops = _arcgis(f"{BTS}/NTAD_National_Transit_Map_Stops/FeatureServer/0", fields="OBJECTID")
    print("         Amtrak stations, Amtrak routes, intercity terminals ...")
    amtrak = _arcgis(f"{BTS}/NTAD_Amtrak_Stations/FeatureServer/0", page=1000)
    routes = _arcgis(f"{BTS}/NTAD_Amtrak_Routes/FeatureServer/0")
    ipcd = _arcgis(f"{BTS}/NTAD_Intermodal_Passenger_Connectivity_Database/FeatureServer/0")

    def count_in(points, name):
        j = gpd.sjoin(points[["geometry"]], counties[["geoid", "geometry"]], predicate="within")
        counties[name] = counties.geoid.map(j.groupby("geoid").size()).fillna(0).astype(int)

    count_in(stops, "transit_stops")
    count_in(amtrak, "amtrak_stations")
    # IPCD: keep the columns that say which modes serve each terminal
    mode_cols = [c for c in ipcd.columns if c.lower() not in ("geometry",) and ipcd[c].astype(str).isin(["0", "1"]).mean() > 0.9]
    for c in mode_cols:
        pts = ipcd[pd.to_numeric(ipcd[c], errors="coerce") == 1]
        count_in(pts, f"ipcd_{c.lower()}")
    counties["geometry"] = counties.to_crs(EQUAL_AREA).geometry.simplify(400).to_crs(WGS84)
    counties.to_file(PROCESSED / "counties_connectivity.geojson", driver="GeoJSON", COORDINATE_PRECISION=4)
    routes_s = routes.to_crs(EQUAL_AREA)
    routes_s["geometry"] = routes_s.geometry.simplify(300)
    routes_s.to_crs(WGS84)[["geometry"] + [c for c in ("name", "NAME", "Name") if c in routes.columns]].to_file(
        PROCESSED / "amtrak_routes.geojson", driver="GeoJSON", COORDINATE_PRECISION=4)
    amtrak[[c for c in amtrak.columns if c.lower() in ("code", "stncode", "name", "stationname", "city", "state", "stntype", "geometry")]].to_file(
        PROCESSED / "amtrak_stations.geojson", driver="GeoJSON")
    return {"counties": len(counties), "transit_stops": len(stops), "amtrak_stations": len(amtrak),
            "amtrak_route_pieces": len(routes), "ipcd_terminals": len(ipcd), "ipcd_mode_columns": mode_cols,
            "counties_with_no_transit_stop": int((counties.transit_stops == 0).sum()),
            "size_mb": size_mb(PROCESSED / "counties_connectivity.geojson")}


# ------------------------------------------------------------------ Day 19 · Philly coffee
@step("day19_philly_coffee", "philly_coffee.geojson")
def day19():
    import osmnx as ox
    tags = {"amenity": "cafe", "shop": "coffee", "craft": ["coffee_roaster", "roaster", "roastery"]}
    g = ox.features_from_place("Philadelphia, Pennsylvania, USA", tags=tags)
    g = g.reset_index()
    keep = [c for c in ("osmid", "id", "element", "element_type", "name", "brand", "brand:wikidata", "amenity", "shop",
                        "craft", "cuisine", "operator", "addr:street", "opening_hours") if c in g.columns]
    pts = g[keep + ["geometry"]].copy()
    pts["geometry"] = pts.to_crs(EQUAL_AREA).geometry.representative_point().to_crs(WGS84)
    pts.to_file(PROCESSED / "philly_coffee.geojson", driver="GeoJSON")
    return {"places": len(pts), "named": int(pts["name"].notna().sum()) if "name" in pts else 0,
            "with_brand": int(pts["brand"].notna().sum()) if "brand" in pts else 0,
            "top_brands": pts["brand"].value_counts().head(6).to_dict() if "brand" in pts else {},
            "tags": {k: int(pts[k].notna().sum()) for k in ("amenity", "shop", "craft") if k in pts}}


# ------------------------------------------------------------------ Day 20 · Aqueduct basins with groundwater
@step("day20_aqueduct_basins", "aqueduct_us_basins.geojson")
def day20():
    src = next((p for pat in ("*Aq40*.gdb", "*aqueduct*.gdb", "*Aqueduct*.gdb", "*.gdb") for p in RAW.rglob(pat)), None)
    if src is None:
        raise FileNotFoundError("No Aqueduct .gdb folder in data/raw. Unzip 'Aqueduct 4.0 global maps data' from WRI "
                                "into data/raw/ (the same one the water map used).")
    layers = gpd.list_layers(src)["name"].tolist()
    layer = next((l for l in layers if "baseline_annual" in l.lower()), layers[0])
    print(f"         reading {src.name} / {layer} for the Lower 48 (a minute or two) ...")
    g = gpd.read_file(src, layer=layer, bbox=(-125.0, 24.0, -66.5, 49.6))
    want = ["pfaf_id", "name_0", "name_1", "bws_raw", "bws_score", "bws_cat", "bws_label",
            "gtd_raw", "gtd_score", "gtd_cat", "gtd_label", "drr_cat", "iav_cat", "sev_cat", "rfr_cat", "cfr_cat"]
    keep = [c for c in want if c in g.columns]
    g = g[keep + ["geometry"]].to_crs(WGS84)
    if "name_0" in g.columns:
        g = g[g["name_0"].astype(str).str.contains("United States", na=False)]
    g["geometry"] = g.to_crs(EQUAL_AREA).geometry.simplify(800, preserve_topology=True).to_crs(WGS84)
    g.to_file(PROCESSED / "aqueduct_us_basins.geojson", driver="GeoJSON", COORDINATE_PRECISION=4)
    return {"file": f"{src.name} / {layer}", "polygons": len(g), "columns": keep,
            "missing_groundwater_column": "gtd_cat" not in g.columns,
            "size_mb": size_mb(PROCESSED / "aqueduct_us_basins.geojson")}


# ------------------------------------------------------------------ run
STEPS = [day13_eia, day13_uspvdb, day15, day17, day18, day19, day20]

if __name__ == "__main__":
    wanted = [a.lower() for a in sys.argv[1:]]
    chosen = [fn for fn in STEPS if not wanted or any(fn.step_name.startswith(w + "_") for w in wanted)]
    if not chosen:
        sys.exit(f"No step matches {wanted}. Try day13, day15, day17, day18, day19 or day20.")
    for fn in chosen:
        fn()
    print("\nDone. Commit the new files in data/processed/ (GitHub Desktop: commit, then Push origin).")
