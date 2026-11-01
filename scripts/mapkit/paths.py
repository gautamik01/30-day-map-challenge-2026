"""paths.py : where things live, so notebooks work no matter where they run from."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the repo folder
SCRIPTS = ROOT / "scripts"
RAW = ROOT / "data" / "raw"                          # big downloads (kept off GitHub)
PROCESSED = ROOT / "data" / "processed"              # small cleaned files (committed)
ARCGIS = ROOT / "data" / "arcgis"                    # GeoPackages for ArcGIS Pro (kept off GitHub)
OUTPUT = ROOT / "output"                             # live maps, served by GitHub Pages
THUMBS = ROOT / "png_thumbnail"                      # one PNG per map, for MUSA + LinkedIn
CAPTIONS = ROOT / "captions"                         # one caption file per map

EQUAL_AREA = "EPSG:5070"   # US Albers: use for areas, distances, hexagons
WGS84 = "EPSG:4326"        # lon/lat: use for web maps

for p in (RAW, PROCESSED, ARCGIS, OUTPUT, THUMBS, CAPTIONS):
    p.mkdir(parents=True, exist_ok=True)


def mapbox_token():
    """Reads MAPBOX_TOKEN from your environment, or from mapbox_token.txt in the repo root (git-ignored)."""
    tok = os.environ.get("MAPBOX_TOKEN", "").strip()
    f = ROOT / "mapbox_token.txt"
    if not tok and f.exists():
        tok = f.read_text().strip()
    return tok
