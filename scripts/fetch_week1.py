"""
fetch_week1.py : one run gets the data for the Day 2 to Day 7 maps.

    conda activate geospatial
    pip install scikit-learn        (only needed once, the Day 4 notebook uses it)
    python scripts/fetch_week1.py

Safe to run again. Finished steps are skipped, and a failed step does not stop the others.
Delete a file in data/processed/ to fetch it again.
Results land in data/processed/, small enough to commit, with a fetch_report.json summary.
The 1843 Bombay scan stays in data/raw/. This script copies a lighter version to data/processed/.

Sources: OpenStreetMap (Overpass), USGS 3DEP and EPQS, US Census, City of Philadelphia.
"""
import json
import math
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import requests
import shapely
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapkit.paths import PROCESSED, RAW, WGS84  # noqa: E402

WEEK = PROCESSED
sys.stdout.reconfigure(errors="replace")   # odd characters in names should never stop a run
REPORT_FILE = WEEK / "fetch_report.json"
REPORT = json.loads(REPORT_FILE.read_text()) if REPORT_FILE.exists() else {}

ox.settings.use_cache = True
ox.settings.cache_folder = str(RAW / "osm_cache")
ox.settings.requests_timeout = 300

STREETS = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"]
METERS_PER_MILE = 1609.344

# Census place ids for the six target cities
PLACES = {"nyc": "3651000", "philly": "4260000", "dc": "1150000",
          "sf": "0667000", "austin": "4805000", "seattle": "5363000"}
PLACE_LABELS = {"nyc": "New York", "philly": "Philadelphia", "dc": "Washington DC",
                "sf": "San Francisco", "austin": "Austin", "seattle": "Seattle"}

CITY_HALL = (39.95222, -75.16361)     # lat, lon


# ------------------------------------------------------------------ helpers
def step(name, *outputs):
    """Run a fetch step once. Skip it if its files exist. Keep going if it fails."""
    def wrap(fn):
        def run():
            files = [WEEK / o for o in outputs]
            if files and all(f.exists() for f in files):
                print(f"[skip] {name}: already fetched")
                return
            print(f"[....] {name}")
            t = time.time()
            try:
                info = fn() or {}
                REPORT[name] = {"status": "ok", "seconds": round(time.time() - t), **info}
                print(f"[ ok ] {name} ({round(time.time() - t)}s) {info}")
            except Exception as e:
                REPORT[name] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
            REPORT_FILE.write_text(json.dumps(REPORT, indent=2))
        return run
    return wrap


def round_coords(gdf, decimals=5):
    """Five decimals is about a meter. Keeps the files small."""
    out = gdf.copy()
    out["geometry"] = shapely.transform(out.geometry.values, lambda a: np.round(a, decimals))
    return out[~out.geometry.is_empty]


def save(gdf, name):
    round_coords(gdf).to_file(WEEK / name, driver="GeoJSON")


def osm(polygon, tags, columns=()):
    """OSM features inside a polygon, with only the columns we want."""
    g = ox.features_from_polygon(polygon, tags).reset_index()
    keep = [c for c in columns if c in g.columns]
    return g[keep + ["geometry"]].copy()


def only(gdf, *kinds):
    return gdf[gdf.geometry.geom_type.isin(kinds)].copy()


def nearest_node(G, lat, lon):
    ids = np.array(list(G.nodes))
    ys = np.array([G.nodes[n]["y"] for n in ids])
    xs = np.array([G.nodes[n]["x"] for n in ids])
    d = (ys - lat) ** 2 + ((xs - lon) * math.cos(math.radians(lat))) ** 2
    return ids[int(d.argmin())]


def drive_route(polygon, filters, start, end):
    """Ordered points along a road from start to end. Tries each OSM filter until one connects."""
    last = None
    for f in filters:
        try:
            G = ox.graph_from_polygon(polygon, custom_filter=f, simplify=False,
                                      retain_all=True, truncate_by_edge=True)
            U = G.to_undirected()
            path = nx.shortest_path(U, nearest_node(U, *start), nearest_node(U, *end), weight="length")
            points, bridges = [], []
            for u, v in zip(path[:-1], path[1:]):
                data = min(U[u][v].values(), key=lambda d: d.get("length", 1e12))
                points.append((U.nodes[u]["x"], U.nodes[u]["y"]))
                bridges.append("bridge" in data and str(data["bridge"]).lower() not in ("no", "nan"))
            points.append((U.nodes[path[-1]]["x"], U.nodes[path[-1]]["y"]))
            return points, bridges, f
        except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError) as e:
            last = e
            print(f"       filter did not connect, trying the next one: {f}")
    raise RuntimeError(f"no filter connected the route: {last}")


def resample(points, bridges, spacing_m):
    """Even samples every spacing_m along a path. Returns lon, lat, distance and bridge flag."""
    pts = np.array(points)
    k = math.cos(math.radians(pts[:, 1].mean()))
    seg = np.hypot(np.diff(pts[:, 0]) * k, np.diff(pts[:, 1])) * 111320.0
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = np.arange(0.0, cum[-1], spacing_m)
    d = np.append(d, cum[-1])
    idx = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    return pd.DataFrame({"lon": np.interp(d, cum, pts[:, 0]), "lat": np.interp(d, cum, pts[:, 1]),
                         "dist_m": d, "bridge": np.array(bridges)[idx]})


def epqs(lat, lon):
    """Ground elevation in feet from the USGS point query service."""
    url = "https://epqs.nationalmap.gov/v1/json"
    for attempt in range(4):
        try:
            r = requests.get(url, params={"x": lon, "y": lat, "units": "Feet", "wkid": 4326}, timeout=40)
            r.raise_for_status()
            return float(r.json()["value"])
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return float("nan")


def add_elevation(df):
    with ThreadPoolExecutor(max_workers=8) as pool:
        df["elev_ft"] = list(pool.map(epqs, df.lat, df.lon))
    missing = int(df.elev_ft.isna().sum())
    if missing:
        print(f"       {missing} points had no elevation, filling from their neighbors")
        df["elev_ft"] = df.elev_ft.interpolate(limit_direction="both")
    return df, missing


def places():
    f = WEEK / "places_six.geojson"
    if f.exists():
        return gpd.read_file(f).set_index("city")
    z = RAW / "cb_2023_us_place_500k.zip"
    if not z.exists():
        url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_place_500k.zip"
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        z.write_bytes(r.content)
    g = gpd.read_file(z).to_crs(WGS84)
    g = g[g.GEOID.isin(PLACES.values())].copy()
    back = {v: k for k, v in PLACES.items()}
    g["city"] = g.GEOID.map(back)
    g["label"] = g.city.map(PLACE_LABELS)
    g = g[["city", "label", "NAME", "ALAND", "geometry"]]
    assert len(g) == 6, f"expected 6 cities, found {sorted(g.NAME)}"
    save(g, "places_six.geojson")
    return gpd.read_file(f).set_index("city")


def query_shape(geom, tolerance=0.002):
    """A simpler shape to ask OSM with. We clip to the real boundary afterward."""
    s = geom.buffer(tolerance).simplify(tolerance)
    return s


# ------------------------------------------------------------------ Day 2
BIG_SUR = box(-122.06, 36.20, -121.62, 36.58)


@step("day2_big_sur_route", "bigsur_profile.csv")
def day2_route():
    start = (36.2465, -121.7807)      # Big Sur Station
    finish = (36.5367, -121.9089)     # Highway 1 at Rio Road
    filters = ['["highway"]["ref"~"(^|;) *CA 1 *(;|$)"]', '["highway"]["ref"="CA 1"]',
               '["highway"~"trunk|primary|secondary"]']
    pts, br, used = drive_route(BIG_SUR, filters, start, finish)
    df, missing = add_elevation(resample(pts, br, 100))
    df["dist_mi"] = df.dist_m / METERS_PER_MILE
    df.round({"lon": 5, "lat": 5, "dist_m": 1, "dist_mi": 3, "elev_ft": 1}).to_csv(WEEK / "bigsur_profile.csv", index=False)
    return {"miles": round(float(df.dist_mi.max()), 2), "start_ft": round(float(df.elev_ft.iloc[0])),
            "top_ft": round(float(df.elev_ft.max())), "finish_ft": round(float(df.elev_ft.iloc[-1])),
            "bridge_points": int(df.bridge.sum()), "missing": missing, "filter": used,
            "official": "26.2 mi, start 356 ft, Hurricane Point 560 ft, finish 10 ft"}


@step("day2_boston_profile", "boston_comm_ave.csv")
def day2_boston():
    area = box(-71.225, 42.322, -71.155, 42.350)
    start = (42.3376, -71.2092)       # Commonwealth Ave at Walnut St
    finish = (42.3388, -71.1672)      # Boston College
    filters = ['["highway"]["name"="Commonwealth Avenue"]', '["highway"~"primary|secondary|tertiary"]']
    pts, br, used = drive_route(area, filters, start, finish)
    df, missing = add_elevation(resample(pts, br, 25))
    df["dist_mi"] = df.dist_m / METERS_PER_MILE
    df.round({"lon": 5, "lat": 5, "dist_m": 1, "dist_mi": 3, "elev_ft": 1}).to_csv(WEEK / "boston_comm_ave.csv", index=False)
    return {"miles": round(float(df.dist_mi.max()), 2), "min_ft": round(float(df.elev_ft.min())),
            "max_ft": round(float(df.elev_ft.max())), "missing": missing, "filter": used}


@step("day2_terrain", "bigsur_dem.npz")
def day2_terrain():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    x0, y0 = (-122.06 * 20037508.34 / 180,
              math.log(math.tan((90 + 36.20) * math.pi / 360)) * 20037508.34 / math.pi)
    x1, y1 = (-121.62 * 20037508.34 / 180,
              math.log(math.tan((90 + 36.58) * math.pi / 360)) * 20037508.34 / math.pi)
    width = 1800
    height = round(width * (y1 - y0) / (x1 - x0))
    url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
    r = requests.get(url, params={"bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": 3857, "imageSR": 3857,
                                  "size": f"{width},{height}", "format": "tiff", "pixelType": "F32",
                                  "interpolation": "RSP_BilinearInterpolation", "f": "image"}, timeout=300)
    r.raise_for_status()
    tif = RAW / "bigsur_dem.tif"
    tif.write_bytes(r.content)
    dem = np.array(Image.open(tif), dtype="float32")
    dem[~np.isfinite(dem) | (dem < -100) | (dem > 5000)] = 0
    np.savez_compressed(WEEK / "bigsur_dem.npz", dem_dm=np.round(dem * 10).astype("int16"),
                        bbox_3857=np.array([x0, y0, x1, y1]))
    return {"shape": list(dem.shape), "max_m": round(float(dem.max())), "ocean_share": round(float((dem <= 0.5).mean()), 3)}


# ------------------------------------------------------------------ Day 3
@step("day3_sf", "sf_parking.geojson", "sf_streets.geojson", "sf_ggpark.geojson")
def day3():
    sf = places().loc["sf"].geometry
    area = box(-122.53, 37.70, -122.35, 37.835)
    land = sf.intersection(area)
    p = only(osm(area, {"amenity": "parking"}, ("id", "name", "parking", "access", "capacity", "building")),
             "Polygon", "MultiPolygon")
    p = p[p.representative_point().within(land)]
    save(p, "sf_parking.geojson")
    s = only(osm(area, {"highway": STREETS}, ("highway",)), "LineString", "MultiLineString")
    s = s[s.intersects(land)]
    save(s, "sf_streets.geojson")
    g = osm(area, {"leisure": "park"}, ("name",))
    g = only(g[g["name"].astype(str) == "Golden Gate Park"], "Polygon", "MultiPolygon")
    save(g, "sf_ggpark.geojson")
    ea = p.to_crs(32610).area / 1e6
    return {"parking_polygons": len(p), "parking_km2": round(float(ea.sum()), 3),
            "by_type": {str(k): int(v) for k, v in p.get("parking", pd.Series(dtype=str)).fillna("untagged").value_counts().items()},
            "golden_gate_park_km2": round(float(g.to_crs(32610).area.sum() / 1e6), 3), "street_ways": len(s)}


# ------------------------------------------------------------------ Day 4
INDIAN = r"(?:^|[;, _])(?:indian|south_indian|north_indian|punjabi|gujarati|keralan?|tamil|hyderabadi|bengali|andhra|chettinad|dosa|udupi)(?:$|[;, _])"


@step("day4_restaurants", "indian_restaurants.geojson")
def day4():
    frames, totals = [], {}
    pl = places()
    for key, row in pl.iterrows():
        print(f"       {row['label']}")
        g = osm(query_shape(row.geometry), {"amenity": ["restaurant", "fast_food"]},
                ("id", "amenity", "name", "cuisine", "brand"))
        pts = g.copy()
        pts["geometry"] = pts.representative_point()
        pts = pts[pts.within(row.geometry)]
        for col in ("name", "cuisine"):
            if col not in pts:
                pts[col] = None
        by_cuisine = pts.cuisine.fillna("").str.lower().str.contains(INDIAN, regex=True)
        by_name = pts.name.fillna("").str.lower().str.contains(r"\bdosas?\b|\bdosai\b", regex=True)
        hit = pts[by_cuisine | by_name].copy()
        hit["city"] = key
        hit["dosa_named"] = by_name[hit.index]
        frames.append(hit)
        totals[key] = {"all_eateries": len(pts), "with_cuisine_tag": int(pts.cuisine.notna().sum()), "indian": len(hit)}
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=WGS84)
    save(out, "indian_restaurants.geojson")
    return {"by_city": totals}


# ------------------------------------------------------------------ Day 5
PHILLY_API = "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/LI_BUILDING_FOOTPRINTS/FeatureServer/0"


@step("day5_city_footprints", "philly_tall.geojson")
def day5_city():
    meta = requests.get(PHILLY_API, params={"f": "json"}, timeout=60).json()
    fields = {f["name"].lower(): f["name"] for f in meta["fields"]}
    h = fields["max_hgt"]
    wanted = [fields[k] for k in ("max_hgt", "approx_hgt", "base_elevation", "building_name", "address", "bin") if k in fields]
    frames, offset = [], 0
    while True:
        r = requests.get(PHILLY_API + "/query", timeout=180, params={
            "where": f"{h} >= 100", "outFields": ",".join(wanted), "outSR": 4326, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": 2000, "orderByFields": h + " DESC"})
        r.raise_for_status()
        feats = r.json().get("features", [])
        if not feats:
            break
        frames.append(gpd.GeoDataFrame.from_features(feats, crs=WGS84))
        offset += len(feats)
        if len(feats) < 2000:
            break
    g = pd.concat(frames, ignore_index=True)
    g.columns = [c.lower() if c != "geometry" else c for c in g.columns]
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=WGS84)
    save(g, "philly_tall.geojson")
    top = g.sort_values("max_hgt", ascending=False).head(25)
    return {"footprints": len(g), "max_hgt_top": [
        [str(a) if a else None, str(b), round(float(c))] for a, b, c in
        zip(top.get("building_name", [None] * 25), top.get("address", [""] * 25), top.max_hgt)]}


@step("day5_osm_towers", "philly_osm_tall.geojson")
def day5_osm():
    lat, lon = CITY_HALL
    area = box(lon - 0.06, lat - 0.045, lon + 0.06, lat + 0.045)
    cols = ("id", "name", "height", "building:levels", "building", "building:part", "man_made")
    g = only(osm(area, {"building": True, "building:part": True}, cols), "Polygon", "MultiPolygon")
    g["height_m"] = pd.to_numeric(g.get("height", pd.Series(index=g.index, dtype=str)).astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    near = g.geometry.representative_point().distance(shapely.Point(lon, lat)) < 0.003
    keep = g[(g.height_m >= 60) | ((g.height_m >= 30) & near)]
    save(keep, "philly_osm_tall.geojson")
    return {"kept": len(keep), "top": [[str(a), float(b)] for a, b in keep.sort_values("height_m", ascending=False)[["name", "height_m"]].head(25).values]}


@step("day5_backdrop", "philly_streets.geojson", "philly_water.geojson")
def day5_backdrop():
    philly = places().loc["philly"].geometry
    area = box(*philly.buffer(0.01).bounds)
    s = only(osm(area, {"highway": STREETS}, ("highway",)), "LineString", "MultiLineString")
    s = s[s.intersects(philly)]
    save(s, "philly_streets.geojson")
    w = only(osm(area, {"natural": "water"}, ("name", "water")), "Polygon", "MultiPolygon")
    a = w.to_crs(32618).area
    w = w[a > 20000]
    save(w, "philly_water.geojson")
    return {"street_ways": len(s), "water_polygons": len(w)}


# ------------------------------------------------------------------ Day 6
@step("day6_coastline", "mumbai_coastline.geojson")
def day6():
    area = box(72.74, 18.86, 72.95, 19.14)
    g = only(osm(area, {"natural": "coastline"}, ("id",)), "LineString", "MultiLineString")
    save(g, "mumbai_coastline.geojson")
    return {"coast_ways": len(g)}


@step("day6_scan", "bombay_1843.jpg")
def day6_scan():
    """Finds the scan in data/raw/ and saves a lighter copy we can commit."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    found = [f for f in RAW.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]
    if not found:
        raise FileNotFoundError("no image in data/raw/. Download the 1843 map there and run this again.")
    named = [f for f in found if any(w in f.name.lower() for w in ("1843", "bombay", "seven", "island"))]
    src = max(named or found, key=lambda f: f.stat().st_mtime)
    im = Image.open(src).convert("RGB")
    original = im.size
    im.thumbnail((3000, 3000))
    im.save(WEEK / "bombay_1843.jpg", quality=88)
    return {"source_file": src.name, "original_px": list(original), "saved_px": list(im.size)}


# ------------------------------------------------------------------ Day 7
TEN_MINUTES_M = 804.7                 # half a mile, the distance the Trust for Public Land uses for a 10-minute walk
PARK_SIZES_ACRES = {"d_any": 0, "d_1ac": 1, "d_10ac": 10}
PHILLY_GIS = "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services"
WALKED_STREETS = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
                  "unclassified", "residential", "living_street", "trunk", "trunk_link"}


def arcgis_all(url, fields="*"):
    """Every feature in an ArcGIS layer, one page at a time."""
    frames, offset = [], 0
    while True:
        r = requests.get(url + "/query", timeout=180, params={
            "where": "1=1", "outFields": fields, "outSR": 4326, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": 1000})
        r.raise_for_status()
        feats = r.json().get("features", [])
        if not feats:
            break
        frames.append(gpd.GeoDataFrame.from_features(feats, crs=WGS84))
        offset += len(feats)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=WGS84)


def block_groups():
    """Philadelphia block groups with population from the latest ACS 5-year release we can reach."""
    z = RAW / "cb_2023_42_bg_500k.zip"
    if not z.exists():
        r = requests.get("https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_42_bg_500k.zip", timeout=300)
        r.raise_for_status()
        z.write_bytes(r.content)
    bg = gpd.read_file(f"zip://{z.as_posix()}").to_crs(WGS84)
    bg = bg[bg.COUNTYFP == "101"][["GEOID", "geometry"]].copy()
    for year in (2024, 2023):
        r = requests.get(f"https://api.census.gov/data/{year}/acs/acs5", timeout=120, params={
            "get": "B01003_001E", "for": "block group:*", "in": "state:42 county:101 tract:*"})
        if r.status_code == 200:
            rows = r.json()
            acs = pd.DataFrame(rows[1:], columns=rows[0])
            acs["GEOID"] = acs.state + acs.county + acs.tract + acs["block group"]
            acs["population"] = pd.to_numeric(acs.B01003_001E)
            out = bg.merge(acs[["GEOID", "population"]], on="GEOID", how="left")
            return out, year
    raise RuntimeError(f"Census API answered {r.status_code}: {r.text[:200]}")


@step("day7_ten_minute_test", "tenmin_streets.geojson", "tenmin_parks.geojson", "heat_tracts.geojson", "philly_bg_pop.geojson")
def day7():
    for old in ("meyerson_walk.geojson", "meyerson_water.geojson", "meyerson_point.json"):
        (WEEK / old).unlink(missing_ok=True)      # left over from the earlier Day 7 idea

    heat = arcgis_all(f"{PHILLY_GIS}/heat_vulnerability_ct/FeatureServer/0")
    heat.columns = [c.lower() if c != "geometry" else c for c in heat.columns]
    save(heat, "heat_tracts.geojson")

    bg, acs_year = block_groups()
    save(bg, "philly_bg_pop.geojson")

    # Parks: neighboring parcels count as one park, so Fairmount Park is not split into pieces
    ppr = arcgis_all(f"{PHILLY_GIS}/PPR_Properties/FeatureServer/0").to_crs(32618)
    merged = ppr.geometry.buffer(5).union_all().buffer(-5)
    parks = gpd.GeoDataFrame(geometry=gpd.GeoSeries([merged], crs=32618).explode(index_parts=False).values, crs=32618)
    parks = parks[parks.area > 0].reset_index(drop=True)
    parks["acres"] = parks.area / 4046.856
    save(parks.round({"acres": 2}).to_crs(WGS84), "tenmin_parks.geojson")

    # Walking network, a little past the city line so border streets can reach nearby parks
    philly = places().loc["philly"].geometry
    G = ox.graph_from_polygon(philly.buffer(0.012), network_type="walk", simplify=True, retain_all=False)
    U = G.to_undirected()
    nodes = gpd.GeoDataFrame({"node": list(U.nodes)}, crs=WGS84,
                             geometry=gpd.points_from_xy([U.nodes[n]["x"] for n in U.nodes],
                                                         [U.nodes[n]["y"] for n in U.nodes])).to_crs(32618)

    # Shortest walk from every node to the nearest park of each size. A park counts once you reach its edge.
    dists = {}
    for col, acres in PARK_SIZES_ACRES.items():
        chosen = parks[parks.acres >= acres]
        door = gpd.GeoDataFrame(geometry=chosen.buffer(25), crs=32618)
        sources = set(gpd.sjoin(nodes, door, predicate="within").node)
        dists[col] = nx.multi_source_dijkstra_path_length(U, sources, weight="length")
        print(f"       parks of {acres}+ acres: {len(chosen)} parks, {len(sources)} entry nodes")

    # Keep ordinary streets, since that is where people live. Footpaths only helped people get there.
    rows = []
    for u, v, d in U.edges(data=True):
        kind = d.get("highway")
        kind = kind[0] if isinstance(kind, list) else kind
        if kind not in WALKED_STREETS:
            continue
        geom = d.get("geometry") or LineString([(U.nodes[u]["x"], U.nodes[u]["y"]), (U.nodes[v]["x"], U.nodes[v]["y"])])
        row = {"length": round(d.get("length", 0), 1), "geometry": geom}
        for col, table in dists.items():
            ends = [table[n] for n in (u, v) if n in table]
            row[col] = round(min(ends) + d.get("length", 0) / 2) if ends else 99999
        rows.append(row)
    streets = gpd.GeoDataFrame(rows, crs=WGS84)
    middle = gpd.GeoDataFrame(streets.drop(columns="geometry"), crs=WGS84, geometry=streets.geometry.interpolate(0.5, normalized=True))
    joined = gpd.sjoin(middle, bg[["GEOID", "geometry"]], predicate="within", how="left")
    streets["GEOID"] = joined.groupby(level=0).GEOID.first()
    streets = streets[streets.GEOID.notna()]
    save(streets, "tenmin_streets.geojson")

    # Share of people within 10 minutes: each block group's people spread along its streets
    pop = bg.set_index("GEOID").population
    served = {}
    for col in PARK_SIZES_ACRES:
        near = streets[streets[col] <= TEN_MINUTES_M].groupby("GEOID").length.sum()
        share = (near / streets.groupby("GEOID").length.sum()).reindex(pop.index).fillna(0)
        served[col] = round(float((share * pop).sum() / pop.sum() * 100), 1)
    return {"share_of_people_within_10_min": served, "published_tpl_any_park": 97, "parks": len(parks),
            "street_edges": len(streets), "acs_year": acs_year, "people": int(pop.sum()),
            "heat_fields": [c for c in heat.columns if c != "geometry"]}


# ------------------------------------------------------------------ run
STEPS = [day2_route, day2_boston, day2_terrain, day3, day4, day5_city, day5_osm, day5_backdrop, day6, day6_scan, day7]

if __name__ == "__main__":
    t0 = time.time()
    for fn in STEPS:
        fn()
    REPORT_FILE.write_text(json.dumps(REPORT, indent=2))
    print("\n--- summary ---")
    failed = [k for k, v in REPORT.items() if v.get("status") != "ok"]
    for k, v in REPORT.items():
        print(f"  {v.get('status', '?'):7} {k}")
    print(f"  total time {round((time.time() - t0) / 60, 1)} min")
    print("\nFailed steps: " + ", ".join(failed) + "\nRun the script again to retry them." if failed else "\nAll good. Commit and push on week1-maps.")
