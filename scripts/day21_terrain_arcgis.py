"""
day21_terrain_arcgis.py : precise terrain for the ski map, made with ArcGIS Pro.

For each ski area it:
  1. downloads USGS 3DEP elevation at a 3 m cell (real meters, UTM zone 18N)
  2. makes a multidirectional hillshade in ArcGIS Pro
  3. makes contours in ArcGIS Pro, every 20 m with every 100 m marked as an index line
  4. saves small web files the interactive page loads when you click that mountain

Run it with ArcGIS Pro's Python, not the conda geospatial env, because only Pro's Python has arcpy.
From the repo folder, in PowerShell:

  & "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\Scripts\\propy.bat" scripts\\day21_terrain_arcgis.py --only "Blue Mountain"
  & "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\Scripts\\propy.bat" scripts\\day21_terrain_arcgis.py

It needs data/processed/ski_areas.geojson and ski_runs.geojson first (python scripts\\fetch_week1.py day21).
Finished mountains are skipped, so it's safe to run again.

Where things go (no new folders):
  data/raw/       the raw USGS grids (git-ignored)
  data/arcgis/    full-size DEM, hillshade and contours to open in ArcGIS Pro (git-ignored)
  output/         day21_terrain_<mountain>_shade.jpg, _dem.bin, _contours.json for the web page
  data/processed/ ski_terrain_index.json, which lists every mountain's files, bounds and elevations
"""
import argparse
import io
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
PROCESSED, RAW, ARCGIS, OUTPUT = ROOT / "data" / "processed", ROOT / "data" / "raw", ROOT / "data" / "arcgis", ROOT / "output"
INDEX = PROCESSED / "ski_terrain_index.json"
SERVICE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"

# ---- the settings worth playing with
CELL_M = 3             # elevation cell size. Smaller is sharper and slower.
MARGIN_M = 1500        # terrain kept around the outermost trail
CONTOUR_M = 20         # contour interval
INDEX_M = 100          # every Nth meter is drawn heavier
SMOOTH_CELLS = 4       # smooths the DEM before contouring, so lines don't jitter over every lidar bump
SHADE_PX = 2048        # hillshade size for the web
DEM_PX = 512           # 3D surface size for the web
MAX_MINUTES = 180      # your 3-hour day trip rule
NODATA = -9999.0
UTM18N, WEB_MERCATOR = 32618, 3857


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def to_mercator(lon, lat):
    return lon * 20037508.34 / 180, math.log(math.tan((90 + lat) * math.pi / 360)) * 20037508.34 / math.pi


def walk_coords(c):
    """Every lon, lat pair in a GeoJSON coordinate list, however deeply nested."""
    if isinstance(c[0], (int, float)):
        yield c[0], c[1]
    else:
        for part in c:
            yield from walk_coords(part)


def load_mountains(only=None):
    """The ski areas within the day trip rule, each with the lon/lat box around its trails."""
    areas = json.loads((PROCESSED / "ski_areas.geojson").read_text(encoding="utf-8"))["features"]
    runs = json.loads((PROCESSED / "ski_runs.geojson").read_text(encoding="utf-8"))["features"]
    boxes = {}
    for f in runs:
        area_id = f["properties"].get("area_id")
        if area_id:
            pts = list(walk_coords(f["geometry"]["coordinates"]))
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            b = boxes.setdefault(area_id, [180, 90, -180, -90])
            boxes[area_id] = [min(b[0], *xs), min(b[1], *ys), max(b[2], *xs), max(b[3], *ys)]
    out = []
    for f in areas:
        p = f["properties"]
        if p["area_id"] not in boxes or (p.get("closest_min") is not None and p["closest_min"] > MAX_MINUTES):
            continue
        if only and only.lower() not in (p.get("ski_area") or "").lower():
            continue
        out.append({"name": p["ski_area"], "area_id": p["area_id"], "slug": slug(p["ski_area"]), "box": boxes[p["area_id"]]})
    return out


def service_limit():
    """The largest image the USGS service hands back in one request. The service times out now and then, so it retries,
    then falls back to 4000 px, which every 3DEP image service has allowed for years."""
    for attempt in range(1, 5):
        try:
            info = requests.get(SERVICE, params={"f": "json"}, timeout=60).json()
            return int(min(info.get("maxImageWidth", 4000), info.get("maxImageHeight", 4000)))
        except (requests.RequestException, ValueError) as e:       # ValueError covers a non-JSON answer such as a 504 page
            print(f"       USGS service info attempt {attempt}: {type(e).__name__}")
            time.sleep(20 * attempt)
    print("       USGS service info unavailable, assuming 4000 px per image")
    return 4000


def download_dem(x0, y0, x1, y1, limit):
    """USGS elevation for a UTM box, in meters. The cell grows only if the box is bigger than the service allows."""
    cell = max(CELL_M, (x1 - x0) / limit, (y1 - y0) / limit)
    w, h = math.ceil((x1 - x0) / cell), math.ceil((y1 - y0) / cell)
    x1, y0 = x0 + w * cell, y1 - h * cell        # snap the box to whole cells
    params = {"bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": UTM18N, "imageSR": UTM18N, "size": f"{w},{h}", "format": "tiff",
              "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    for attempt in range(1, 5):
        try:
            r = requests.get(f"{SERVICE}/exportImage", params=params, timeout=300)
            if r.status_code == 200 and r.content[:2] in (b"II", b"MM"):
                dem = np.array(Image.open(io.BytesIO(r.content)), dtype="float32")
                if dem.shape == (h, w):
                    dem[~np.isfinite(dem) | (dem < -100) | (dem > 3000)] = NODATA
                    return dem, cell, x0, y0, x1, y1
            print(f"       USGS attempt {attempt}: unexpected answer ({r.status_code}, {len(r.content)} bytes)")
        except requests.RequestException as e:
            print(f"       USGS attempt {attempt}: {type(e).__name__}")
        time.sleep(20 * attempt)
    raise RuntimeError("USGS elevation service did not return a valid grid")


def fill_gaps(a, missing):
    """Fill missing cells with their nearest real neighbor."""
    if not missing.any():
        return a
    nearest = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
    return a[tuple(nearest)]


def multidirectional(dem, cell):
    """Backup hillshade in plain numpy, used only if Pro's Image Analyst function isn't licensed. Six suns, equally weighted."""
    dy, dx = np.gradient(dem, cell)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    alt = math.radians(45)
    total = sum(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(math.radians(az) - aspect)
                for az in (225, 270, 315, 360, 45, 90))
    return np.clip(total / 6, 0, 1) * 255


def simplify(line, tol):
    """Douglas-Peucker: drops vertices that don't change a line's shape."""
    pts = np.asarray(line, float)
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ab = pts[b] - pts[a]
        d = np.abs(ab[0] * (pts[a + 1:b, 1] - pts[a, 1]) - ab[1] * (pts[a + 1:b, 0] - pts[a, 0])) / (np.hypot(*ab) or 1)
        i = int(d.argmax())
        if d[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return pts[keep]


def resize(a, px):
    """Shrink a grid so its long side is px. Returns float32."""
    h, w = a.shape
    k = px / max(h, w)
    size = (max(2, round(w * k)), max(2, round(h * k)))
    return np.array(Image.fromarray(a.astype("float32")).resize(size, Image.BILINEAR), dtype="float32")


def build(m, arcpy, limit):
    name, s = m["name"], m["slug"]
    # 1. the box around the trails, in UTM meters
    wgs = arcpy.SpatialReference(4326)
    utm = arcpy.SpatialReference(UTM18N)
    w, so, e, n = m["box"]
    corners = [arcpy.PointGeometry(arcpy.Point(x, y), wgs).projectAs(utm).firstPoint for x, y in ((w, so), (w, n), (e, so), (e, n))]
    x0, x1 = min(c.X for c in corners) - MARGIN_M, max(c.X for c in corners) + MARGIN_M
    y0, y1 = min(c.Y for c in corners) - MARGIN_M, max(c.Y for c in corners) + MARGIN_M

    # 2. USGS elevation
    raw = RAW / f"ski_dem_{s}.npy"
    meta = RAW / f"ski_dem_{s}.json"
    if raw.exists() and meta.exists():
        dem = np.load(raw)
        cell, x0, y0, x1, y1 = json.loads(meta.read_text())
    else:
        dem, cell, x0, y0, x1, y1 = download_dem(x0, y0, x1, y1, limit)
        np.save(raw, dem)
        meta.write_text(json.dumps([cell, x0, y0, x1, y1]))
    missing = dem == NODATA
    print(f"       {dem.shape[1]} x {dem.shape[0]} cells at {cell:.2f} m, elevation {dem[~missing].min():.0f} to {dem[~missing].max():.0f} m, {missing.mean():.1%} missing")

    # 3. into ArcGIS Pro
    dem_tif = ARCGIS / f"ski_dem_{s}.tif"
    ras = arcpy.NumPyArrayToRaster(dem, arcpy.Point(x0, y0), cell, cell, NODATA)
    ras.save(str(dem_tif))
    arcpy.management.DefineProjection(str(dem_tif), utm)

    # 4. multidirectional hillshade
    shade_tif = ARCGIS / f"ski_hillshade_{s}.tif"
    try:
        arcpy.CheckOutExtension("ImageAnalyst")
        arcpy.ia.Hillshade(str(dem_tif), z_factor=1, hillshade_type=1).save(str(shade_tif))
        method = "ArcGIS Pro multidirectional hillshade"
    except Exception as e:
        print(f"       Image Analyst hillshade unavailable ({type(e).__name__}), using the numpy backup")
        arcpy.NumPyArrayToRaster(multidirectional(fill_gaps(dem, missing), cell).astype("float32"), arcpy.Point(x0, y0), cell, cell).save(str(shade_tif))
        arcpy.management.DefineProjection(str(shade_tif), utm)
        method = "numpy multidirectional hillshade, six suns"

    # 5. contours from a lightly smoothed DEM
    gdb = ARCGIS / "ski_terrain.gdb"
    if not arcpy.Exists(str(gdb)):
        arcpy.management.CreateFileGDB(str(ARCGIS), gdb.name)
    arcpy.CheckOutExtension("Spatial")
    smooth = arcpy.sa.FocalStatistics(str(dem_tif), arcpy.sa.NbrCircle(SMOOTH_CELLS, "CELL"), "MEAN", "DATA")
    lines = str(gdb / f"contours_{s}")
    arcpy.sa.Contour(smooth, lines, CONTOUR_M, 0)
    tmp = RAW / f"ski_contours_{s}.geojson"
    arcpy.conversion.FeaturesToJSON(lines, str(tmp), "NOT_FORMATTED", "NO_Z_VALUES", "NO_M_VALUES", True, True)

    feats = []
    for f in json.loads(tmp.read_text(encoding="utf-8"))["features"]:
        elev = f["properties"]["Contour"]
        g = f["geometry"]
        parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        for part in parts:
            if len(part) >= 2:
                c = np.round(simplify([p[:2] for p in part], 0.00002), 5).tolist()
                feats.append({"type": "Feature", "properties": {"e": elev, "i": int(elev % INDEX_M == 0)},
                              "geometry": {"type": "LineString", "coordinates": c}})
    contour_file = OUTPUT / f"day21_terrain_{s}_contours.json"
    contour_file.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")), encoding="utf-8")

    # 6. web files, in web mercator so they sit exactly on the map
    merc = arcpy.SpatialReference(WEB_MERCATOR)
    dem_m, shade_m = ARCGIS / f"ski_dem_{s}_3857.tif", ARCGIS / f"ski_hillshade_{s}_3857.tif"
    arcpy.management.ProjectRaster(str(dem_tif), str(dem_m), merc, "BILINEAR")
    arcpy.management.ProjectRaster(str(shade_tif), str(shade_m), merc, "BILINEAR")
    out_dem = arcpy.Raster(str(dem_m))
    ex = out_dem.extent
    a = arcpy.RasterToNumPyArray(out_dem, nodata_to_value=NODATA).astype("float32")
    a = fill_gaps(a, a == NODATA)
    small = resize(a, DEM_PX)
    dem_file = OUTPUT / f"day21_terrain_{s}_dem.bin"
    np.round(small * 10).astype("<i2").tofile(dem_file)         # whole decimeters, north row first

    b = arcpy.RasterToNumPyArray(arcpy.Raster(str(shade_m)), nodata_to_value=NODATA).astype("float32")
    b = fill_gaps(b, b == NODATA)
    shade = resize(b, SHADE_PX)
    shade_file = OUTPUT / f"day21_terrain_{s}_shade.jpg"
    Image.fromarray(np.clip(shade, 0, 255).astype("uint8")).save(shade_file, quality=86, optimize=True)

    size_mb = sum(f.stat().st_size for f in (contour_file, dem_file, shade_file)) / 1e6
    print(f"       {method}. {len(feats)} contour lines. Web files {size_mb:.2f} MB")
    return {"ski_area": name, "area_id": m["area_id"], "slug": s, "bounds_3857": [ex.XMin, ex.YMin, ex.XMax, ex.YMax],
            "dem_size": [int(small.shape[1]), int(small.shape[0])], "cell_m": round(cell, 2), "hillshade": method,
            "elev_min_m": round(float(a.min()), 1), "elev_max_m": round(float(a.max()), 1), "web_mb": round(size_mb, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help='one mountain, e.g. "Blue Mountain"')
    ap.add_argument("--redo", action="store_true", help="rebuild mountains that are already finished")
    args = ap.parse_args()

    import arcpy      # only Pro's Python has this, so a wrong interpreter fails here with a clear error
    arcpy.env.overwriteOutput = True
    for ext in ("Spatial", "ImageAnalyst"):
        print(f"{ext} license: {arcpy.CheckExtension(ext)}")
    if arcpy.CheckExtension("Spatial") != "Available":
        raise SystemExit("Contours need the Spatial Analyst license. Turn it on under Project > Licensing in ArcGIS Pro.")

    mountains = load_mountains(args.only)
    if not mountains:
        raise SystemExit(f"No ski area matches {args.only!r} within {MAX_MINUTES} minutes.")
    index = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    limit = service_limit()
    print(f"{len(mountains)} ski areas. USGS allows {limit} px per image.\n")

    failed = []
    for i, m in enumerate(mountains, 1):
        print(f"[{i}/{len(mountains)}] {m['name']}")
        if m["slug"] in index and not args.redo:
            print("       already done")
            continue
        try:
            index[m["slug"]] = build(m, arcpy, limit)
            INDEX.write_text(json.dumps(index, indent=1))
        except Exception as e:
            failed.append(m["name"])
            print(f"       FAILED: {type(e).__name__}: {e}")
    total = sum(v["web_mb"] for v in index.values())
    print(f"\nDone: {len(index)} mountains, {total:.0f} MB of web files.")
    if failed:
        print("Failed, run again to retry: " + ", ".join(failed))


if __name__ == "__main__":
    main()
