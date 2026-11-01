"""
style.py : one look for the whole series.

Two series, two palettes:
  "ai"   -> The Cloud Has a Footprint   (dark, teal + amber, a bit serious)
  "city" -> Five Cities, Five Senses    (warm paper, playful)

Change colors here once and every map follows.
"""

AUTHOR = "Gautami Kasat"
HASHTAG = "#30DayMapChallenge"
GITHUB_USER = "gautamik01"
REPO = "30-day-map-challenge-2026"
PAGES_URL = f"https://{GITHUB_USER}.github.io/{REPO}"
CODE_URL = f"https://github.com/{GITHUB_USER}/{REPO}/blob/main/scripts"

# Data credits. SHORT goes on the map card, FULL goes in the caption file + README.
CREDITS = {
    "im3": dict(
        short="IM3 Data Center Atlas (PNNL) from © OpenStreetMap contributors",
        full="Mongird, K., Vernon, C., Burleyson, C., Akdemir, K. Z., & Rice, J. (2026). IM3 Open Source Data Center "
             "Atlas (v2026.02.09). MSD-LIVE Data Repository. https://doi.org/10.57931/3017294 · "
             "derived from © OpenStreetMap contributors, ODbL"),
    "epoch": dict(
        short="Epoch AI Frontier Data Centers (CC BY 4.0)",
        full="Epoch AI, Frontier Data Centers, https://epoch.ai/data/data-centers · CC BY 4.0"),
    "osm": dict(
        short="© OpenStreetMap contributors",
        full="© OpenStreetMap contributors, https://www.openstreetmap.org/copyright · ODbL"),
    "census": dict(
        short="US Census Bureau",
        full="US Census Bureau, Cartographic Boundary Files 2023"),
}


def credits(*keys, full=False):
    """credits("im3", "epoch") -> one line for the map card; full=True -> citation list for captions."""
    if full:
        return "\n".join(f"- {CREDITS[k]['full']}" for k in keys)
    return " · ".join(CREDITS[k]["short"] for k in keys)

SERIES = {
    "ai": dict(
        name="The Cloud Has a Footprint",
        hashtags="#30DayMapChallenge #DataCenters #ClimateRisk #UrbanPlanning #GIS",
        basemap="dark",
        paper="#0e1418", ink="#eef3f4", muted="#9fb0b6", card="rgba(14,20,24,0.88)",
        accent="#f2b134",              # data centers
        accent2="#2fb5a8",             # water / resources
        ramp=["#13343b", "#1f6f74", "#2fb5a8", "#f2b134", "#f25f3a"],   # low -> high
        categorical=["#f2b134", "#2fb5a8", "#e86a92", "#8fb3ff", "#c3e88d", "#ff9f6e", "#b39ddb"],
    ),
    "city": dict(
        name="City Senses",            # rename anytime, it updates the gallery + cards
        hashtags="#30DayMapChallenge #Cities #UrbanDesign #OpenStreetMap #GIS",
        basemap="light",
        paper="#f7f2ea", ink="#24201c", muted="#7a6f63", card="rgba(255,252,247,0.94)",
        accent="#d1495b",
        accent2="#00798c",
        ramp=["#fbe8d3", "#f4b183", "#e0705a", "#b03a48", "#5c1a33"],
        categorical=["#d1495b", "#00798c", "#edae49", "#30638e", "#6a994e", "#9c6ade", "#e07a5f"],
    ),
}

# Raster basemaps for folium (Leaflet). Labels drawn above your data.
BASEMAPS = {
    "light":   ("light_nolabels", "light_only_labels"),
    "dark":    ("dark_nolabels", "dark_only_labels"),
    "voyager": ("rastertiles/voyager_nolabels", "rastertiles/voyager_only_labels"),
}
CARTO_URL = "https://{s}.basemaps.cartocdn.com/{path}/{z}/{x}/{y}{r}.png"
CARTO_ATTR = ('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
              '&copy; <a href="https://carto.com/attributions">CARTO</a>')

# Esri basemaps (no labels-on-top trick, but nice for Esri-flavoured days)
ESRI_TILES = {
    "esri_dark":    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    "esri_light":   "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    "esri_imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "esri_ocean":   "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
}
ESRI_ATTR = "Tiles &copy; Esri"

# Mapbox GL styles (needs a free public token, see README)
MAPBOX_STYLES = {
    "dark": "mapbox://styles/mapbox/dark-v11",
    "light": "mapbox://styles/mapbox/light-v11",
    "standard": "mapbox://styles/mapbox/standard",
    "satellite": "mapbox://styles/mapbox/satellite-streets-v12",
}
# Free fallback when there is no Mapbox token (MapLibre + CARTO vector styles)
MAPLIBRE_FALLBACK = {
    "dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "light": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "standard": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "satellite": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
}

FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600'
         '&family=Inter:wght@400;600&display=swap" rel="stylesheet">')


def card_html(day, prompt, title, subtitle="", story="", legend_html="", sources="", series="ai",
              extra_html=""):
    """The floating title card used on every interactive map."""
    s = SERIES[series]
    return f"""
<style>
  .mk-card {{position:fixed;top:16px;left:16px;z-index:9999;max-width:340px;padding:18px 20px 14px;
    background:{s['card']};color:{s['ink']};border-radius:14px;backdrop-filter:blur(6px);
    box-shadow:0 8px 28px rgba(0,0,0,.22);font-family:Inter,system-ui,sans-serif}}
  .mk-chip {{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:{s['accent']};margin-bottom:6px}}
  .mk-title {{font-family:Fraunces,Georgia,serif;font-size:26px;line-height:1.08;font-weight:600;margin:0 0 5px}}
  .mk-sub {{font-size:13.5px;color:{s['muted']};margin-bottom:10px}}
  .mk-story {{font-size:13px;line-height:1.48;margin-bottom:10px}}
  .mk-legend {{font-size:12px;margin:8px 0}}
  .mk-legend .row {{display:flex;align-items:center;gap:8px;margin:3px 0}}
  .mk-legend .dot {{width:11px;height:11px;border-radius:50%;flex:none}}
  .mk-legend .bar {{height:9px;border-radius:5px}}
  .mk-legend .ends {{display:flex;justify-content:space-between;color:{s['muted']};font-size:11px;margin-top:2px}}
  .mk-foot {{font-size:10.5px;color:{s['muted']};margin-top:10px;line-height:1.45}}
  .mk-btns {{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 4px}}
  .mk-btns button {{font:600 12px Inter,system-ui,sans-serif;padding:6px 10px;border-radius:999px;cursor:pointer;
    border:1px solid {s['muted']};background:transparent;color:{s['ink']}}}
  .mk-btns button.on, .mk-btns button:hover {{background:{s['accent']};border-color:{s['accent']};color:#111}}
  @media (max-width:640px) {{ .mk-card {{left:10px;right:10px;top:10px;max-width:none;padding:14px}}
    .mk-title {{font-size:21px}} .mk-story {{display:none}} }}
</style>
<div class="mk-card">
  <div class="mk-chip">Day {day:02d} &middot; {prompt}</div>
  <div class="mk-title">{title}</div>
  {f'<div class="mk-sub">{subtitle}</div>' if subtitle else ''}
  {f'<div class="mk-story">{story}</div>' if story else ''}
  {extra_html}
  {f'<div class="mk-legend">{legend_html}</div>' if legend_html else ''}
  <div class="mk-foot">{('Data: ' + sources + '<br>') if sources else ''}{AUTHOR} &middot; {HASHTAG}</div>
</div>"""


def legend_categories(items):
    """items: list of (label, color)"""
    return "".join(f'<div class="row"><span class="dot" style="background:{c}"></span>{l}</div>' for l, c in items)


def legend_ramp(colors, low="low", high="high"):
    return (f'<div class="bar" style="background:linear-gradient(90deg,{", ".join(colors)})"></div>'
            f'<div class="ends"><span>{low}</span><span>{high}</span></div>')
