"""
day21_landscape.py : real water and tree cover for the zoomed-in view of each ski area on the Day 21 map.

For each mountain in data/processed/ski_terrain_index.json it:
  1. asks the USGS National Hydrography Dataset (NHD) for the lakes, ponds, reservoirs, rivers and streams in the box
  2. asks OpenStreetMap for water polygons and keeps the ponds NHD does not have (snowmaking ponds are usually OSM-only),
     cut so nothing is drawn twice
  3. cuts NLCD Tree Canopy Cover 2025 (USDA Forest Service, published through MRLC) to the box, keeping the percent values
  4. saves data/raw/day21_terrain_<slug>_water.json and data/raw/day21_terrain_<slug>_trees.png (not committed) and
     records the sources, source years and licenses in ski_terrain_index.json

Run it in the conda geospatial env, from the repo folder:

  python scripts\\day21_landscape.py --only "Blue Mountain" --preview
  python scripts\\day21_landscape.py

Finished mountains are skipped, so it's safe to run again. --redo rebuilds them. --preview also writes
data/raw/day21_landscape_preview_<slug>.png: hillshade + canopy in green + water in blue + ski runs in red, to check by eye.

The tree file: an 8-bit grayscale PNG the same size as the mountain's _dem.bin grid, so it sits on the same bounds_3857.
Pixel value = tree canopy percent, 0 to 100. 255 = no data. Nothing is thresholded into forest / not forest.

Where things go (no new folders):
  data/raw/        the water and trees files, preview PNGs, and the canopy zip only if it has to be downloaded (git-ignored)
  data/processed/  the terrain files this script reads, and ski_terrain_index.json, which gains water_file, trees_file,
                   canopy_range and landscape_sources
"""
import argparse
import contextlib
import json
import math
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import requests
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box, shape
from shapely.ops import unary_union
import rasterio
from rasterio import features as rfeatures
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import from_bounds as window_from_bounds
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PROCESSED, RAW = ROOT / "data" / "processed", ROOT / "data" / "raw"
LANDSCAPE = RAW         # where the water and trees files go; they are not committed
INDEX = PROCESSED / "ski_terrain_index.json"
WGS84, UTM18N, WEB_MERCATOR = 4326, 32618, 3857

# ---- water: USGS National Hydrography Dataset, the "Large Scale" layers are the high-resolution NHD
NHD = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
NHD_LAYERS = {"waterbody": 12, "area": 9, "flowline": 6}
NHD_POLYGON_KINDS = {390: "lake_pond", 436: "reservoir", 493: "estuary", 312: "bay_inlet", 445: "sea_ocean", 460: "river", 336: "canal"}
NHD_LINE_KINDS = {460: "stream", 336: "canal"}
# Dropped on purpose: 558 artificial path (the centerline NHD draws through every lake and wide river, so it would double the
# water), 466 swamp/marsh, 364 connector, 428 pipeline, 566 coastline, 343 dam, 431 rapids, 403 inundation area, 378 ice.
INTERMITTENT = {39001, 39005, 39006, 46003, 46007}      # NHD fcodes for intermittent or ephemeral water

# ---- water: OpenStreetMap, for the ponds NHD lacks
OVERPASS = ["https://overpass-api.de/api", "https://overpass.kumi.systems/api", "https://overpass.private.coffee/api"]
OSM_STILL_WATER = {None, "pond", "reservoir", "lake", "basin", "lagoon", "oxbow", "pool", "reflecting_pool", "fishpond"}
MIN_POND_M2 = 100          # an OSM pond with less than this left after cutting away NHD water is the same pond

# ---- trees: NLCD Tree Canopy Cover, the newest release on the Forest Service gateway MRLC links to
TCC_URL = "https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/docs/v2025-6/nlcd_tcc_conus_2025_v2025-6_wgs84.zip"
TCC_NAME = "NLCD Tree Canopy Cover 2025, v2025-6 (USDA Forest Service, published through MRLC)"
TCC_PAGE = "https://www.mrlc.gov/data/type/nlcd-tree-canopy-cover"
PNG_NODATA = 255
TCC_MASK = 254             # NLCD masks some areas out of the canopy model (its notes list water and non-tree farmland)
TREES_SCALE = "pixel value = tree canopy percent 0 to 100 as published; 254 = masked by NLCD (water and the like); 255 = no data"
TCC_CITATION = "USDA Forest Service. 2026. USFS NLCD Percent Tree Canopy CONUS v2025-6. Salt Lake City, UT."
TCC_LICENSE = ("Public domain (U.S. Government work). The file's metadata: can be used without additional permissions or fees; "
               "cite " + TCC_CITATION)
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".zip,.tif",
            "GDAL_INGESTED_BYTES_AT_OPEN": 4_000_000, "CPL_VSIL_CURL_CHUNK_SIZE": 1_000_000, "CPL_VSIL_CURL_CACHE_SIZE": 300_000_000,
            "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
            "GDAL_HTTP_MAX_RETRY": 5, "GDAL_HTTP_RETRY_DELAY": 10, "GDAL_HTTP_RETRY_CODES": "ALL",
            "GDAL_HTTP_CONNECTTIMEOUT": 60, "GDAL_HTTP_TIMEOUT": 300}

SIMPLIFY_DEG = 0.00002     # about 2 m, the same as the contour files


# ------------------------------------------------------------------ small helpers
def lonlat_box(b):
    """A Web Mercator box as west, south, east, north in degrees."""
    x0, y0, x1, y1 = b
    lon = lambda x: x / 20037508.34 * 180
    lat = lambda y: math.degrees(2 * math.atan(math.exp(y / 20037508.34 * math.pi)) - math.pi / 2)
    return lon(x0), lat(y0), lon(x1), lat(y1)


def mercator(lon, lat):
    return lon * 20037508.34 / 180, math.log(math.tan((90 + lat) * math.pi / 360)) * 20037508.34 / math.pi


def tag(row, key):
    """One OSM tag as text, or None. osmnx fills missing tags with NaN."""
    v = row.get(key)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def get_json(url, params, tries=5, timeout=120):
    """GET that returns parsed JSON. The USGS gateway answers 502/504 now and then, so it waits and asks again."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if "error" not in data:
                    return data
                why = f"service error {data['error'].get('code')}: {data['error'].get('message')}"
            else:
                why = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:      # ValueError: an HTML error page where JSON was expected
            why = type(e).__name__
        print(f"       NHD attempt {attempt}: {why}")
        time.sleep(10 * attempt)
    raise RuntimeError(f"no answer from {url}")


# ------------------------------------------------------------------ water from NHD
def nhd_query(layer, bounds_3857):
    """Every feature of one NHD layer that touches the box, as lon/lat GeoJSON features. Pages through long answers."""
    x0, y0, x1, y1 = bounds_3857
    envelope = {"xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, "spatialReference": {"wkid": WEB_MERCATOR}}
    params = {"where": "1=1", "geometry": json.dumps(envelope), "geometryType": "esriGeometryEnvelope", "inSR": WEB_MERCATOR,
              "spatialRel": "esriSpatialRelIntersects", "outFields": "permanent_identifier,gnis_name,ftype,fcode",
              "returnGeometry": "true", "outSR": WGS84, "f": "geojson", "resultRecordCount": 1000}
    feats, offset = [], 0
    while True:
        data = get_json(f"{NHD}/{layer}/query", {**params, "resultOffset": offset})
        page = data.get("features", [])
        feats += page
        if not page or not data.get("properties", {}).get("exceededTransferLimit"):
            return feats
        offset += len(page)


def nhd_water(bounds_3857):
    """NHD lakes, ponds, reservoirs, river polygons and stream lines in the box, in lon/lat. Also what was dropped, by ftype."""
    rows, dropped = [], {}
    for layer, kinds in (("waterbody", NHD_POLYGON_KINDS), ("area", NHD_POLYGON_KINDS), ("flowline", NHD_LINE_KINDS)):
        for f in nhd_query(NHD_LAYERS[layer], bounds_3857):
            p = {k.lower(): v for k, v in f["properties"].items()}     # the three layers spell their fields differently
            fcode = int(p["fcode"])
            ftype = int(p["ftype"]) if p.get("ftype") is not None else fcode // 100
            kind = kinds.get(ftype)
            if not kind:
                dropped[ftype] = dropped.get(ftype, 0) + 1
                continue
            rows.append({"geometry": shape(f["geometry"]), "kind": kind, "name": p.get("gnis_name"), "source": "nhd",
                         "id": p.get("permanent_identifier"), "fcode": fcode, "intermittent": fcode in INTERMITTENT, "nhd_layer": layer})
    columns = ["geometry", "kind", "name", "source", "id", "fcode", "intermittent", "nhd_layer"]
    return gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs=WGS84), dropped


# ------------------------------------------------------------------ water from OpenStreetMap
def osm_features(bbox, tags):
    """OSM features in a lon/lat box (west, south, east, north). Tries the other Overpass servers when one is busy."""
    import osmnx as ox
    ox.settings.requests_timeout = 300
    ox.settings.use_cache = (RAW / "osm_cache").is_dir()      # reuse the fetch script's cache if it is there, make no folder
    ox.settings.cache_folder = str(RAW / "osm_cache")
    last = None
    for url in OVERPASS:
        ox.settings.overpass_url = url
        ox.settings.overpass_rate_limit = url == OVERPASS[0]
        try:
            return ox.features_from_bbox(bbox=bbox, tags=tags).reset_index()
        except Exception as e:
            if type(e).__name__ == "InsufficientResponseError":        # the server answered: there is just nothing here
                return gpd.GeoDataFrame({"element": [], "id": []}, geometry=[], crs=WGS84)
            last = e
            print(f"       {url} failed ({type(e).__name__}), trying the next server")
    raise RuntimeError(f"every Overpass server failed, last error: {last}")


def osm_check(bbox, nhd):
    """Compare OSM water with NHD. Returns the OSM ponds NHD misses, already cut so they never overlap NHD water, and the tallies."""
    polys = osm_features(bbox, {"natural": "water", "landuse": "reservoir", "water": True})
    polys = polys[polys.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    lines = osm_features(bbox, {"waterway": ["river", "stream", "canal", "ditch", "drain"]})
    lines = lines[lines.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if len(lines):
        lines = lines.clip(box(*bbox))                # OSM hands back whole ways, which can run far outside the box

    nhd_utm = nhd.to_crs(UTM18N)
    is_poly = nhd_utm.geom_type.isin(["Polygon", "MultiPolygon"])
    nhd_poly = unary_union(nhd_utm[is_poly].geometry.values) if is_poly.any() else shapely.Polygon()
    # Any NHD water within 25 m counts, polygons too: a river NHD draws as a shape has no line of its own
    nhd_near = unary_union(nhd_utm.geometry.values).buffer(25) if len(nhd_utm) else shapely.Polygon()

    added, tally = [], {"osm_polygons": len(polys), "already_in_nhd": 0, "added": 0, "left_to_nhd": 0}
    for _, r in polys.to_crs(UTM18N).iterrows():
        water = tag(r, "water")
        if water not in OSM_STILL_WATER:
            tally["left_to_nhd"] += 1                 # rivers, streams, canals: NHD draws those
            continue
        g = r.geometry
        if g.is_empty or g.area == 0:
            continue
        if g.intersection(nhd_poly).area / g.area >= 0.5:
            tally["already_in_nhd"] += 1
            continue
        rest = g.difference(nhd_poly)                 # keep only the part NHD does not already draw
        if rest.area < MIN_POND_M2:
            tally["already_in_nhd"] += 1
            continue
        words = " ".join(filter(None, (tag(r, k) for k in ("name", "description", "note", "usage", "reservoir_type")))).lower()
        added.append({"geometry": rest, "kind": water or "water", "name": tag(r, "name"), "source": "osm",
                      "id": f"{tag(r, 'element')}/{tag(r, 'id')}", "intermittent": tag(r, "intermittent") == "yes",
                      "snowmaking": "snow" in words})
        tally["added"] += 1

    # The check the other way: OSM streams with more than half their length away from any NHD line. Counted, not added.
    missing = 0
    for g in lines.to_crs(UTM18N).geometry:
        if g.length and g.difference(nhd_near).length / g.length > 0.5:
            missing += 1
    tally["osm_waterways"] = len(lines)
    tally["osm_waterways_not_in_nhd"] = missing
    columns = ["geometry", "kind", "name", "source", "id", "intermittent", "snowmaking"]
    ponds = gpd.GeoDataFrame(added, columns=columns, geometry="geometry", crs=UTM18N).to_crs(WGS84)
    return ponds, tally


def plain(v):
    """A pandas cell as a plain JSON value: numpy numbers become Python numbers, whole floats become ints. None for empty."""
    if v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v) or None                      # False is left out of the file to keep it small
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v) if float(v).is_integer() else float(v)
    return v


def write_water(path, gdf):
    """One small GeoJSON in lon/lat, five decimals, lines and outlines simplified by about 2 m. Empty properties are left out."""
    feats = []
    for _, r in gdf.iterrows():
        g = r.geometry.simplify(SIMPLIFY_DEG, preserve_topology=True)
        g = shapely.transform(g, lambda a: np.round(a, 5))
        if g.is_empty:
            continue
        props = {k: plain(v) for k, v in r.items() if k != "geometry"}
        props = {k: v for k, v in props.items() if v is not None}
        feats.append({"type": "Feature", "properties": props, "geometry": shapely.geometry.mapping(g)})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")), encoding="utf-8")
    return len(feats)


# ------------------------------------------------------------------ trees from NLCD Tree Canopy Cover
def fetch(url, tries=4, **kw):
    """One HTTP request with retries, because the Forest Service server sometimes stops answering for a minute."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.request(kw.pop("method", "GET"), url, timeout=kw.pop("timeout", 60), allow_redirects=True, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == tries:
                raise
            print(f"       Forest Service attempt {attempt}: {type(e).__name__}, waiting {15 * attempt} s")
            time.sleep(15 * attempt)


class HttpFile:
    """A read-only file over HTTP byte ranges: enough for zipfile to read a 3.6 GB zip's table of contents in a few KB."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(fetch(url, method="HEAD").headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        r = fetch(self.url, headers={"Range": f"bytes={self.pos}-{min(self.size, self.pos + n) - 1}"}, timeout=120)
        self.pos += len(r.content)
        return r.content


def download(url, path):
    """Stream a big file into data/raw. Writes to .part first so a dropped download never leaves a half file."""
    part = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(part, "wb") as out:
            for chunk in r.iter_content(1 << 22):
                out.write(chunk)
                done += len(chunk)
                if total and done % (1 << 28) < (1 << 22):
                    print(f"       {done / 1e9:.1f} of {total / 1e9:.1f} GB")
    part.replace(path)


def canopy_path():
    """Where to read the canopy raster from. If the TIFF sits uncompressed inside the zip, GDAL reads just the tiles under
    each mountain by byte range from the Forest Service server. Otherwise the zip is downloaded once into data/raw."""
    f = HttpFile(TCC_URL)
    with zipfile.ZipFile(f) as z:
        tif = next(i for i in z.infolist() if i.filename.lower().endswith(".tif"))
    if tif.compress_type == zipfile.ZIP_STORED:
        return f"/vsizip//vsicurl/{TCC_URL}/{tif.filename}", "read by byte range from the Forest Service server"
    local = RAW / Path(TCC_URL).name
    if not local.exists():
        print(f"       the TIFF is compressed inside the zip, downloading {TCC_URL.rsplit('/', 1)[-1]} once ({f.size / 1e9:.1f} GB)")
        download(TCC_URL, local)
    return f"/vsizip/{local.as_posix()}/{tif.filename}", f"downloaded once to {local.relative_to(ROOT).as_posix()}"


def cut_canopy(src, bounds_3857, size):
    """The canopy percent grid for one mountain on the same Web Mercator grid as its _dem.bin. Nearest-neighbour keeps the
    percent values exactly as published. 254 (NLCD's mask) stays 254; anything else above 100 becomes 255, no data."""
    # The file is in Albers Equal Area meters ("wgs84" in its name is the datum), so the box goes into that system first.
    left, bottom, right, top = transform_bounds(f"EPSG:{WEB_MERCATOR}", src.crs, *bounds_3857)
    pad = 3 * src.res[0]
    win = window_from_bounds(left - pad, bottom - pad, right + pad, top + pad, src.transform).round_offsets().round_lengths()
    a = src.read(1, window=win)
    out = np.full((size[1], size[0]), PNG_NODATA, "uint8")
    reproject(a, out, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=src.nodata,
              dst_transform=from_bounds(*bounds_3857, size[0], size[1]), dst_crs=f"EPSG:{WEB_MERCATOR}",
              dst_nodata=PNG_NODATA, resampling=Resampling.nearest)
    out[(out > 100) & (out != TCC_MASK)] = PNG_NODATA
    return out


def trail_check(canopy, bounds_3857, size, runs):
    """Mean canopy percent on the ski trails against the woods 50 to 150 m beside them. A wide gap means the trails read
    as cuts through the trees."""
    if runs.empty:
        return None
    utm = runs.to_crs(UTM18N).geometry
    on = gpd.GeoSeries([unary_union(utm.buffer(10).values)], crs=UTM18N).to_crs(WEB_MERCATOR).iloc[0]
    ring = unary_union(utm.buffer(150).values).difference(unary_union(utm.buffer(50).values))
    beside = gpd.GeoSeries([ring], crs=UTM18N).to_crs(WEB_MERCATOR).iloc[0]
    tr = from_bounds(*bounds_3857, size[0], size[1])
    valid = canopy <= 100
    out = {}
    for key, geom in (("on_trail", on), ("beside_trail", beside)):
        mask = rfeatures.rasterize([(geom, 1)], out_shape=canopy.shape, transform=tr, fill=0, dtype="uint8").astype(bool) & valid
        out[key] = round(float(canopy[mask].mean()), 1) if mask.any() else None
    return out


# ------------------------------------------------------------------ preview
def draw_preview(m, canopy, water, runs, path):
    """Hillshade with the canopy tinted green by percent, water in blue and ski runs in red. For checking by eye only."""
    shade = Image.open(PROCESSED / f"day21_terrain_{m['slug']}_shade.jpg").convert("RGB")
    W = 1024
    H = round(shade.height * W / shade.width)
    base = np.array(shade.resize((W, H), Image.BILINEAR), dtype="float32")
    if canopy is not None:
        pct = np.array(Image.fromarray(canopy).resize((W, H), Image.NEAREST), dtype="float32")
        alpha = np.where(pct <= 100, pct / 100 * 0.6, 0)[..., None]
        green = np.array([40, 120, 50], "float32")
        base = base * (1 - alpha) + green * alpha
    img = Image.fromarray(base.astype("uint8"))
    draw = ImageDraw.Draw(img, "RGBA")
    x0, y0, x1, y1 = m["bounds_3857"]

    def px(coords):
        out = []
        for lon, lat in coords:
            X, Y = mercator(lon, lat)
            out.append(((X - x0) / (x1 - x0) * W, (y1 - Y) / (y1 - y0) * H))
        return out

    for g in water.geometry:
        for part in shapely.get_parts(g):
            if part.geom_type == "Polygon":
                draw.polygon(px(part.exterior.coords), fill=(40, 110, 230, 210), outline=(20, 60, 160, 255))
            elif part.geom_type == "LineString":
                draw.line(px(part.coords), fill=(40, 110, 230, 230), width=2)
    for g in runs.geometry:
        for part in shapely.get_parts(g):
            pts = px(part.exterior.coords if part.geom_type == "Polygon" else part.coords)
            draw.line(pts, fill=(235, 60, 60, 230), width=2)
    img.save(path, optimize=True)


# ------------------------------------------------------------------ one mountain
def build(m, tcc, tcc_how, nhd_version, runs_all, preview):
    s, bounds = m["slug"], m["bounds_3857"]
    w, so, e, n = lonlat_box(bounds)
    clip = box(w, so, e, n)
    today = date.today().isoformat()
    out = {"landscape_sources": [], "landscape_date": today}

    # 1. water from NHD, then the OSM ponds NHD lacks
    nhd, dropped = nhd_water(bounds)
    counts = {"nhd_lakes_ponds": int((nhd.nhd_layer == "waterbody").sum()), "nhd_river_areas": int((nhd.nhd_layer == "area").sum()),
              "nhd_streams": int((nhd.nhd_layer == "flowline").sum()), "nhd_dropped_by_ftype": dropped}
    out["landscape_sources"].append({"layer": "water", "name": "USGS National Hydrography Dataset, large-scale layers (The National Map)",
                                     "url": NHD, "version": nhd_version, "license": "Public domain, U.S. Geological Survey"})
    try:
        ponds, tally = osm_check((w, so, e, n), nhd)
        counts.update({f"osm_{k}" if not k.startswith("osm") else k: v for k, v in tally.items()})
        out["landscape_sources"].append({"layer": "water", "name": "OpenStreetMap water polygons, only the ponds NHD lacks (Overpass API)",
                                         "url": "https://www.openstreetmap.org/copyright", "version": f"fetched {today}",
                                         "license": "ODbL 1.0, (c) OpenStreetMap contributors"})
    except RuntimeError as err:
        print(f"       OpenStreetMap unreachable, water is NHD only: {err}")
        ponds, counts["osm"] = gpd.GeoDataFrame(columns=nhd.columns, geometry="geometry", crs=WGS84), f"unreachable: {err}"
    water = pd.concat([nhd.drop(columns="nhd_layer"), ponds], ignore_index=True)
    water = gpd.GeoDataFrame(water, geometry="geometry", crs=WGS84).clip(clip)
    water_file = LANDSCAPE / f"day21_terrain_{s}_water.json"
    counts["written"] = write_water(water_file, water)
    out.update({"water_file": water_file.name, "water_counts": counts})
    print(f"       water: {counts['nhd_lakes_ponds']} NHD lakes/ponds, {counts['nhd_river_areas']} river areas, {counts['nhd_streams']} streams; "
          f"OSM ponds added {counts.get('osm_added', 'n/a')}, already in NHD {counts.get('osm_already_in_nhd', 'n/a')}; "
          f"{counts['written']} features, {water_file.stat().st_size / 1e3:.0f} KB")

    # 2. trees
    runs = runs_all[runs_all.area_id == m["area_id"]]
    canopy = cut_canopy(tcc, bounds, m["dem_size"]) if tcc is not None else None
    valid = canopy[canopy <= 100] if canopy is not None else np.empty(0)
    if tcc is None:
        print(f"       trees: left out this run, {tcc_how}")
        out.update({"trees_file": None, "trees_note": tcc_how})
    elif valid.size == 0:
        print("       trees: the canopy raster has no data here, no trees file written")
        out.update({"trees_file": None, "trees_note": "the NLCD canopy raster has no data inside this box"})
    else:
        trees_file = LANDSCAPE / f"day21_terrain_{s}_trees.png"
        Image.fromarray(canopy).save(trees_file, optimize=True)          # a 2-D uint8 array saves as 8-bit grayscale
        out.pop("trees_note", None)
        out.update({"trees_file": trees_file.name, "trees_px": [int(canopy.shape[1]), int(canopy.shape[0])], "trees_scale": TREES_SCALE,
                    "canopy_range": [int(valid.min()), int(valid.max())], "canopy_mean": round(float(valid.mean()), 1),
                    "canopy_percent_cells": round(float(valid.size / canopy.size), 3), "canopy_masked_cells": round(float((canopy == TCC_MASK).mean()), 3),
                    "trail_canopy": trail_check(canopy, bounds, m["dem_size"], runs)})
        out["landscape_sources"].append({"layer": "trees", "name": TCC_NAME, "url": TCC_URL, "page": TCC_PAGE,
                                         "version": "v2025-6, data year 2025, released 2026", "access": tcc_how,
                                         "citation": TCC_CITATION, "license": TCC_LICENSE})
        print(f"       trees: canopy {out['canopy_range'][0]} to {out['canopy_range'][1]}%, mean {out['canopy_mean']}%, "
              f"trails {out['trail_canopy']}, {trees_file.stat().st_size / 1e3:.0f} KB")
    if preview:
        p = RAW / f"day21_landscape_preview_{s}.png"
        draw_preview(m, canopy, water, runs, p)
        print(f"       preview: {p.relative_to(ROOT).as_posix()}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help='one mountain, e.g. "Blue Mountain"')
    ap.add_argument("--redo", action="store_true", help="rebuild mountains that are already finished")
    ap.add_argument("--preview", action="store_true", help="also write a preview PNG per mountain into data/raw")
    args = ap.parse_args()

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    mountains = [m for m in index.values() if not args.only or args.only.lower() in m["ski_area"].lower()]
    if not mountains:
        raise SystemExit(f"No mountain in the terrain index matches {args.only!r}. Run day21_terrain_arcgis.py first.")
    runs_all = gpd.read_file(PROCESSED / "ski_runs.geojson")

    nhd_version = get_json(NHD, {"f": "pjson"}).get("copyrightText", "").strip()
    print(f"NHD: {nhd_version}")

    # The canopy raster is opened once for the whole run. If the Forest Service server cannot be reached, the run still
    # does the water, and every mountain's index entry says the trees were left out, so the next run picks them up.
    t = time.time()
    tcc, stack = None, contextlib.ExitStack()
    try:
        tcc_path, tcc_how = canopy_path()
        stack.enter_context(rasterio.Env(**GDAL_ENV))
        tcc = stack.enter_context(rasterio.open(tcc_path))
        print(f"canopy: {tcc_how}; {tcc.width} x {tcc.height} cells of {tcc.res[0]:.0f} m, {tcc.crs.to_dict().get('proj', tcc.crs)}, "
              f"nodata {tcc.nodata}, opened in {time.time() - t:.0f} s\n")
    except Exception as err:
        tcc_how = f"Forest Service canopy server unreachable ({type(err).__name__}: {err})"
        print(f"canopy: {tcc_how}\n")

    failed = []
    with stack:
        for i, m in enumerate(mountains, 1):
            print(f"[{i}/{len(mountains)}] {m['ski_area']}")
            done = m.get("water_file") and m.get("trees_file") and (LANDSCAPE / m["water_file"]).exists() and (LANDSCAPE / m["trees_file"]).exists()
            if done and not args.redo:
                print("       already done")
                continue
            try:
                index[m["slug"]].update(build(m, tcc, tcc_how, nhd_version, runs_all, args.preview))
                INDEX.write_text(json.dumps(index, indent=1), encoding="utf-8")
            except Exception as err:
                failed.append(m["ski_area"])
                print(f"       FAILED: {type(err).__name__}: {err}")

    done = [m for m in index.values() if m.get("water_file")]
    size = sum((LANDSCAPE / m[k]).stat().st_size for m in done for k in ("water_file", "trees_file") if m.get(k) and (LANDSCAPE / m[k]).exists())
    print(f"\nDone: {len(done)} of {len(index)} mountains have landscape files, {size / 1e6:.2f} MB in all.")
    if failed:
        print("Failed, run again to retry: " + ", ".join(failed))


if __name__ == "__main__":
    main()
