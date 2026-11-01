"""
city.py : the city series template, "Five Cities, Five Senses".

Everything comes from OpenStreetMap, so the only thing you change each day
is WHAT you pull (the OSM tags). Same cities, same look, new sense.

    cafes = city.fetch({"amenity": "cafe"}, name="cafes")          # all cities
    cafes = city.fetch({"amenity": "cafe"}, name="cafes", cities=["philly", "nyc"])
    stats = city.per_km2(cafes)                                     # cafes per km² per city

Find tags at https://wiki.openstreetmap.org/wiki/Map_features or taginfo.openstreetmap.org
Handy ones:
    cafes        {"amenity": "cafe"}
    bakeries     {"shop": "bakery"}
    bookstores   {"shop": "books"}
    murals       {"artwork_type": "mural"}
    viewpoints   {"tourism": "viewpoint"}
    benches      {"amenity": "bench"}
    parks        {"leisure": "park"}
    food trucks  {"amenity": "fast_food", "street_vendor": "yes"}
    bike lanes   {"highway": "cycleway"}
    rail lines   {"railway": ["subway", "light_rail", "tram"]}
"""
import pandas as pd
import geopandas as gpd

from .paths import RAW, EQUAL_AREA, WGS84

# Comment out any city you don't want. Order = order of the switcher buttons.
CITIES = {
    "nyc":     dict(label="New York",      place="New York City, New York, USA",          center=(40.7128, -74.0060), zoom=11),
    "philly":  dict(label="Philadelphia",  place="Philadelphia, Pennsylvania, USA",       center=(39.9526, -75.1652), zoom=12),
    "dc":      dict(label="Washington DC", place="Washington, District of Columbia, USA", center=(38.9072, -77.0369), zoom=12),
    "sf":      dict(label="San Francisco", place="San Francisco, California, USA",        center=(37.7749, -122.4194), zoom=12),
    "austin":  dict(label="Austin",        place="Austin, Texas, USA",                    center=(30.2672, -97.7431), zoom=11),
    "seattle": dict(label="Seattle",       place="Seattle, Washington, USA",              center=(47.6062, -122.3321), zoom=11),
}


def _ox():
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(RAW / "osm_cache")
    ox.settings.requests_timeout = 300
    return ox


def boundary(city_key):
    """City boundary polygon (cached)."""
    f = RAW / f"boundary_{city_key}.geojson"
    if not f.exists():
        _ox().geocode_to_gdf(CITIES[city_key]["place"])[["geometry"]].to_file(f, driver="GeoJSON")
    g = gpd.read_file(f).to_crs(WGS84)
    g["city"] = city_key
    return g


def boundaries(cities=None):
    return pd.concat([boundary(c) for c in (cities or CITIES)], ignore_index=True)


def fetch(tags, name, cities=None, as_points=True, refresh=False):
    """
    Pull OSM features for each city, cache them in data/raw/osm_<name>_<city>.geojson.
    as_points=True turns shops/buildings drawn as shapes into dots (what you want for most maps).
    """
    frames = []
    for key in (cities or CITIES):
        f = RAW / f"osm_{name}_{key}.geojson"
        if refresh or not f.exists():
            print(f"  {CITIES[key]['label']}: asking OpenStreetMap for {name} ...")
            try:
                g = _ox().features_from_place(CITIES[key]["place"], tags=tags).reset_index()
            except Exception as e:                      # nothing found is not an error for us
                print(f"    nothing found ({e.__class__.__name__})")
                g = gpd.GeoDataFrame(geometry=[], crs=WGS84)
            keep = [c for c in ["osmid", "id", "element", "element_type", "name", "cuisine", "brand",
                                "opening_hours", "website", "artist_name", "geometry"] + list(tags) if c in g.columns]
            g = g[keep]
            if not g.empty:
                g.to_file(f, driver="GeoJSON")
            else:
                gpd.GeoDataFrame({"name": []}, geometry=[], crs=WGS84).to_file(f, driver="GeoJSON")
        g = gpd.read_file(f)
        if g.empty:
            continue
        g = g.set_crs(WGS84, allow_override=True)
        g["city"] = key
        g["city_label"] = CITIES[key]["label"]
        frames.append(g)
    if not frames:
        return gpd.GeoDataFrame({"city": []}, geometry=[], crs=WGS84)
    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=WGS84)
    if as_points:
        out["geometry"] = out.geometry.representative_point()
    if "name" in out.columns:
        out["name"] = out["name"].fillna("(no name)")
    return out


def per_km2(gdf, cities=None, label="per_km2"):
    """Count of features per km² of city land, one row per city. Great for captions."""
    rows = []
    for key in (cities or sorted(gdf["city"].unique())):
        area = boundary(key).to_crs(EQUAL_AREA).area.sum() / 1e6
        n = int((gdf["city"] == key).sum())
        rows.append(dict(city=key, city_label=CITIES[key]["label"], count=n,
                         area_km2=round(area, 1), **{label: round(n / area, 2)}))
    return pd.DataFrame(rows).sort_values(label, ascending=False).reset_index(drop=True)


def top_values(gdf, column, n=6):
    """Most common values of a column (e.g. cuisine), the rest become 'other'."""
    top = gdf[column].dropna().value_counts().head(n).index
    return gdf[column].where(gdf[column].isin(top), "other").fillna("other")
