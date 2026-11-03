"""
fetch_week1.py : one run gets the data for the Day 2 to Day 7 maps, plus Day 21 (skiing).

    conda activate geospatial
    pip install scikit-learn        (only needed once, the Day 4 notebook uses it)
    python scripts/fetch_week1.py

Safe to run again. Finished steps are skipped, and a failed step does not stop the others.
Delete a file in data/processed/ to fetch it again.
Results land in data/processed/, small enough to commit, with a fetch_report.json summary.
The 1843 Bombay scan stays in data/raw/. This script copies a lighter version to data/processed/.

Sources: OpenStreetMap (Overpass), USGS 3DEP and EPQS, US Census, City of Philadelphia, Mapbox drive times.
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


# ------------------------------------------------------------------ Day 21
SKI_BOX = (-78.6, 39.4, -72.3, 43.1)   # west, south, east, north: a long day trip from either city
ORIGINS = {"nyc": (-73.9855, 40.7580), "philly": (-75.1636, 39.9522)}   # Times Square, City Hall (lon, lat)
LIFTS = ["chair_lift", "gondola", "cable_car", "mixed_lift", "t-bar", "j-bar", "platter", "drag_lift", "rope_tow", "magic_carpet"]


def first_value(v):
    return str(v).split(";")[0].strip().lower() if pd.notna(v) else "unrated"


@step("day21_ski_osm", "ski_runs_raw.geojson", "ski_lifts.geojson", "ski_area_shapes.geojson")
def day21_osm():
    area = box(*SKI_BOX)
    usual = ox.settings.max_query_area_size
    ox.settings.max_query_area_size = 1e11        # ski tags are sparse, so bigger query tiles are fine here
    try:
        runs = osm(area, {"piste:type": True}, ("id", "name", "piste:name", "piste:type", "piste:difficulty", "piste:grooming"))
        lifts = only(osm(area, {"aerialway": LIFTS}, ("id", "name", "aerialway")), "LineString", "MultiLineString")
        shapes = only(osm(area, {"landuse": "winter_sports"}, ("id", "name")), "Polygon", "MultiPolygon")
    finally:
        ox.settings.max_query_area_size = usual
    runs = only(runs[runs["piste:type"].astype(str).str.contains("downhill")], "LineString", "MultiLineString", "Polygon", "MultiPolygon")
    runs["difficulty"] = runs.get("piste:difficulty", pd.Series(index=runs.index, dtype=object)).map(first_value)
    utm = runs.to_crs(32618).geometry
    polygons = utm.geom_type.isin(["Polygon", "MultiPolygon"])
    runs["len_m"] = np.where(polygons, utm.length / 2, utm.length).round(1)  # an area's outline is roughly twice its run
    keep = [c for c in ("id", "name", "piste:name", "difficulty", "len_m", "geometry") if c in runs.columns]
    save(runs[keep].rename(columns={"piste:name": "piste_name"}), "ski_runs_raw.geojson")
    save(lifts, "ski_lifts.geojson")
    save(shapes, "ski_area_shapes.geojson")
    return {"downhill_runs": len(runs), "km": round(float(runs.len_m.sum() / 1000), 1), "lifts": len(lifts),
            "named_area_shapes": int(shapes.get("name", pd.Series(dtype=str)).notna().sum()),
            "difficulty": {k: int(v) for k, v in runs.difficulty.value_counts().items()}}


def drive_minutes(lons, lats):
    """Drive time and distance from each origin to each place. Mapbox when there is a token, else the public OSRM server."""
    from mapkit.paths import mapbox_token
    token = mapbox_token()
    keys = list(ORIGINS)
    out = {f"{k}_{m}": [] for k in keys for m in ("min", "miles")}
    for i in range(0, len(lons), 23):
        batch = list(zip(lons[i:i + 23], lats[i:i + 23]))
        coords = ";".join(f"{x:.5f},{y:.5f}" for x, y in [ORIGINS[k] for k in keys] + batch)
        params = {"sources": "0;1", "destinations": ";".join(str(j + 2) for j in range(len(batch))),
                  "annotations": "duration,distance"}
        if token:
            url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coords}"
            params["access_token"] = token
        else:
            url = f"https://router.project-osrm.org/table/v1/driving/{coords}"
        try:
            r = requests.get(url, params=params, timeout=90)
        except requests.RequestException as e:
            raise RuntimeError(f"drive time service unreachable ({type(e).__name__})") from None   # keeps the token out of logs
        if r.status_code != 200:
            raise RuntimeError(f"drive time service answered {r.status_code}")
        data = r.json()
        for s, k in enumerate(keys):
            out[f"{k}_min"] += [None if v is None else round(v / 60) for v in data["durations"][s]]
            out[f"{k}_miles"] += [None if v is None else round(v / METERS_PER_MILE, 1) for v in data["distances"][s]]
        time.sleep(1.1)
    return out, ("Mapbox" if token else "OSRM")


@step("day21_ski_areas", "ski_runs.geojson", "ski_areas.geojson")
def day21_areas():
    from sklearn.cluster import DBSCAN
    runs = gpd.read_file(WEEK / "ski_runs_raw.geojson").to_crs(32618)
    lifts = gpd.read_file(WEEK / "ski_lifts.geojson").to_crs(32618)
    shapes = gpd.read_file(WEEK / "ski_area_shapes.geojson").to_crs(32618)
    shapes = shapes[shapes["name"].notna()].reset_index(drop=True) if "name" in shapes else shapes.iloc[0:0]

    # Each run joins the named ski area within 1.5 km. Runs with no named shape nearby get grouped by distance.
    middle = gpd.GeoDataFrame(runs[["len_m"]], geometry=runs.representative_point(), crs=32618)
    near = gpd.sjoin_nearest(middle, shapes[["name", "geometry"]], max_distance=1500, how="left")
    near = near[~near.index.duplicated()]
    runs["ski_area"] = near["name"]
    loose = runs["ski_area"].isna()
    if loose.any():
        xy = np.c_[middle.geometry.x[loose], middle.geometry.y[loose]]
        labels = DBSCAN(eps=1500, min_samples=3).fit_predict(xy)
        runs.loc[loose, "ski_area"] = [f"Unnamed area {n}" if n >= 0 else None for n in labels]
    runs = runs[runs["ski_area"].notna()].copy()

    rows = []
    for name, g in runs.groupby("ski_area"):
        km = g.groupby("difficulty").len_m.sum() / 1000
        rated = float(km.drop("unrated", errors="ignore").sum())
        beginner = float(km.get("novice", 0) + km.get("easy", 0))
        hull = g.geometry.union_all().convex_hull
        rows.append({"ski_area": name, "runs": len(g), "total_km": round(float(km.sum()), 2),
                     "bunny_km": round(float(km.get("novice", 0)), 2), "green_km": round(float(km.get("easy", 0)), 2),
                     "blue_km": round(float(km.get("intermediate", 0)), 2),
                     "black_km": round(float(km.get("advanced", 0) + km.get("expert", 0) + km.get("freeride", 0)), 2),
                     "unrated_km": round(float(km.get("unrated", 0)), 2), "beginner_km": round(beginner, 2),
                     "beginner_share": round(beginner / rated, 3) if rated else None,
                     "lifts": int(lifts.distance(hull).lt(500).sum()), "geometry": hull.centroid})
    areas = gpd.GeoDataFrame(rows, crs=32618).to_crs(WGS84)
    times, service = drive_minutes(areas.geometry.x.tolist(), areas.geometry.y.tolist())
    for col, values in times.items():
        areas[col] = values
    areas["closest_min"] = areas[["nyc_min", "philly_min"]].min(axis=1)
    save(areas, "ski_areas.geojson")
    save(runs.to_crs(WGS84), "ski_runs.geojson")
    (WEEK / "ski_runs_raw.geojson").unlink(missing_ok=True)
    day_trip = areas[areas.closest_min <= 180].sort_values("beginner_km", ascending=False)
    return {"areas": len(areas), "within_3h_of_either": len(day_trip), "drive_service": service,
            "unnamed": int(areas["ski_area"].str.startswith("Unnamed").sum()),
            "top_beginner": day_trip[["ski_area", "beginner_km", "total_km", "nyc_min", "philly_min"]].head(15).values.tolist()}


@step("day21_ski_backdrop", "ski_relief.npz", "ski_states.geojson")
def day21_backdrop():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    west, south, east, north = SKI_BOX
    merc_x = lambda lon: lon * 20037508.34 / 180
    merc_y = lambda lat: math.log(math.tan((90 + lat) * math.pi / 360)) * 20037508.34 / math.pi
    x0, y0, x1, y1 = merc_x(west), merc_y(south), merc_x(east), merc_y(north)
    width = 1800
    height = round(width * (y1 - y0) / (x1 - x0))
    url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
    r = requests.get(url, timeout=300, params={"bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": 3857, "imageSR": 3857,
                                               "size": f"{width},{height}", "format": "tiff", "pixelType": "F32",
                                               "interpolation": "RSP_BilinearInterpolation", "f": "image"})
    r.raise_for_status()
    tif = RAW / "ski_dem.tif"
    tif.write_bytes(r.content)
    dem = np.array(Image.open(tif), dtype="float32")
    dem[~np.isfinite(dem) | (dem < -100) | (dem > 5000)] = 0
    np.savez_compressed(WEEK / "ski_relief.npz", dem_dm=np.round(dem * 10).astype("int16"), bbox_3857=np.array([x0, y0, x1, y1]))

    z = RAW / "cb_2023_us_state_500k.zip"
    if not z.exists():
        s = requests.get("https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip", timeout=300)
        s.raise_for_status()
        z.write_bytes(s.content)
    states = gpd.read_file(f"zip://{z.as_posix()}").to_crs(WGS84)
    states = states[states.intersects(box(*SKI_BOX))][["NAME", "STUSPS", "geometry"]]
    save(states, "ski_states.geojson")
    return {"relief_shape": list(dem.shape), "states": sorted(states.STUSPS)}


# ------------------------------------------------------------------ run
STEPS = [day2_route, day2_boston, day2_terrain, day3, day4, day5_city, day5_osm, day5_backdrop, day6, day6_scan, day7,
         day21_osm, day21_areas, day21_backdrop]

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
