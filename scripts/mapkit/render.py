"""
render.py : three ways to draw a map, same inputs.

    layer = Layer(gdf, "Data centers", column="operator_top", popup={"name": "Name", "operator": "Operator"})

    m   = interactive([layer], day=1, prompt="Points", title="...", series="ai")   # Leaflet / folium
    html = mapbox([layer], day=1, prompt="Points", title="...", projection="globe") # Mapbox GL (sponsor!)
    fig = static([layer], day=1, prompt="Points", title="...")                      # PNG for quick days

Colors come from style.SERIES so every map in a series matches.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import geopandas as gpd
import folium

from . import style as S
from .paths import WGS84


# ======================================================================= layers
@dataclass
class Layer:
    gdf: gpd.GeoDataFrame
    name: str
    column: str = None          # color by this column (numbers -> 5 classes, text -> categories)
    color: str = None           # or one fixed color
    size_column: str = None     # points only: scale dot size by this column (e.g. sqft)
    radius: float = 4           # points: base dot size
    popup: dict = None          # {"column": "Label", ...}
    tooltip: str = None         # column shown on hover (defaults to 'name' if present)
    opacity: float = 0.85
    classes: int = 5
    order: list = None          # fixed category order (optional)
    legend: bool = True
    show: bool = True           # visible when the map opens
    fmt: dict = field(default_factory=dict)   # {"sqft": "{:,.0f}"} number formats in popups
    glow: bool = False          # points only: soft halo around each dot (use for "what's coming")
    colors: dict = None         # fixed colors per category, e.g. {"High": "#2fb5a8", ...}


def _geom_kind(gdf):
    t = gdf.geom_type.dropna().iloc[0] if len(gdf) else "Point"
    return "point" if "Point" in t else "line" if "Line" in t else "polygon"


def _fmt_num(v, f=None):
    if pd.isna(v):
        return "n/a"
    if f:
        return f.format(v)
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}"


def prepare(layer, series="ai"):
    """Adds _color / _radius / _tip / popup columns. Returns (gdf, legend_html)."""
    pal = S.SERIES[series]
    g = layer.gdf.to_crs(WGS84).copy()
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    legend = ""

    if layer.column and pd.api.types.is_numeric_dtype(g[layer.column]):
        vals = g[layer.column]
        n = min(layer.classes, max(vals.nunique(), 1))
        ramp = pal["ramp"] if n == len(pal["ramp"]) else [pal["ramp"][int(i)] for i in np.linspace(0, len(pal["ramp"]) - 1, n)]
        if n > 1:
            bins = pd.qcut(vals.rank(method="first"), n, labels=False)
            g["_color"] = [ramp[int(b)] if pd.notna(b) else "#777777" for b in bins]
            edges = [vals[bins == i] for i in range(n)]
            items = [(f"{_fmt_num(e.min(), layer.fmt.get(layer.column))} to {_fmt_num(e.max(), layer.fmt.get(layer.column))}", ramp[i])
                     for i, e in enumerate(edges)]
        else:
            g["_color"] = ramp[-1]
            items = [(_fmt_num(vals.iloc[0]), ramp[-1])]
        legend = f"<b>{layer.name}</b>" + S.legend_categories(items)
    elif layer.column:
        counts = g[layer.column].fillna("n/a").value_counts()
        cats = [c for c in layer.order if counts.get(c, 0) > 0] if layer.order else \
            [c for c in counts.index if c not in ("Other", "other", "n/a")] + \
            [c for c in ("Other", "other", "n/a") if c in counts.index]
        cmap = {c: (pal["categorical"][i] if i < len(pal["categorical"]) else "#8a8a8a") for i, c in enumerate(cats)}
        if layer.colors:
            cmap.update({c: col for c, col in layer.colors.items() if c in cmap})
        for c in ("Other", "other", "n/a"):
            if c in cmap:
                cmap[c] = "#7d8a8f"
        g["_cat"] = g[layer.column].fillna("n/a")
        g["_color"] = g["_cat"].map(cmap).fillna("#7d8a8f")
        legend = f"<b>{layer.name}</b>" + S.legend_categories(
            [(f"{c} <span style='opacity:.6'>({counts.get(c, 0):,})</span>", cmap[c]) for c in cats])
    else:
        g["_color"] = layer.color or pal["accent"]
        legend = S.legend_categories([(layer.name, g["_color"].iloc[0] if len(g) else pal["accent"])])

    if layer.size_column and layer.size_column in g:
        v = pd.to_numeric(g[layer.size_column], errors="coerce")
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 98)
        g["_radius"] = (layer.radius * 0.6 + np.sqrt(((v.clip(lo, hi) - lo) / (hi - lo + 1e-9)).fillna(0)) * layer.radius * 2.6).round(1)
    else:
        g["_radius"] = layer.radius

    tip = layer.tooltip or ("name" if "name" in g.columns else None)
    g["_tip"] = g[tip].astype(str) if tip else layer.name
    for col in (layer.popup or {}):
        if col in g.columns:
            g[f"_p_{col}"] = [(_fmt_num(v, layer.fmt.get(col)) if isinstance(v, (int, float, np.number)) else
                               ("n/a" if pd.isna(v) else str(v))) for v in g[col]]
    keep = ["_color", "_radius", "_tip"] + (["_cat"] if "_cat" in g else []) + [c for c in g.columns if c.startswith("_p_")] + ["geometry"]
    return g[keep], (legend if layer.legend else "")


def _bounds(layers, bounds=None):
    if bounds is not None:
        w, s, e, n = bounds
    else:
        tb = np.array([l.gdf.to_crs(WGS84).total_bounds for l in layers if len(l.gdf)])
        w, s, e, n = tb[:, 0].min(), tb[:, 1].min(), tb[:, 2].max(), tb[:, 3].max()
    return [[s, w], [n, e]]


# ======================================================================= folium
def interactive(layers, *, day, prompt, title, subtitle="", story="", sources="", series="ai",
                bounds=None, basemap=None, switcher=None, layer_control=None, start=None):
    """
    Leaflet map with the series title card.
    basemap: 'light' | 'dark' | 'voyager' | 'esri_dark' | 'esri_light' | 'esri_imagery' | 'esri_ocean'
    switcher: {"nyc": (w, s, e, n), ...} or True to use the city list -> buttons that fly between places
    start: (lat, lon, zoom) to open on one spot instead of fitting all the data (handy with a switcher)
    """
    pal = S.SERIES[series]
    basemap = basemap or pal["basemap"]
    m = folium.Map(location=start[:2] if start else None, zoom_start=start[2] if start else None,
                   tiles=None, control_scale=True, zoom_control="bottomright", prefer_canvas=True)

    if basemap in S.ESRI_TILES:
        folium.TileLayer(S.ESRI_TILES[basemap], attr=S.ESRI_ATTR, name="Basemap", control=False, max_zoom=19).add_to(m)
        label_tiles = None
    else:
        base, label_tiles = S.BASEMAPS[basemap]
        folium.TileLayer(S.CARTO_URL.replace("{path}", base), attr=S.CARTO_ATTR, name="Basemap",
                         control=False, max_zoom=19).add_to(m)

    legends = []
    for layer in layers:
        g, legend = prepare(layer, series)
        if legend:
            legends.append(legend)
        kind = _geom_kind(g)
        fields = [c for c in g.columns if c.startswith("_p_")]
        aliases = [layer.popup[c[3:]] for c in fields]
        popup = folium.GeoJsonPopup(fields=fields, aliases=aliases, labels=True, localize=True) if fields else None
        tooltip = folium.GeoJsonTooltip(fields=["_tip"], labels=False, sticky=True)
        op = layer.opacity
        if kind == "point":
            style = lambda f, op=op: dict(color=f["properties"]["_color"], fillColor=f["properties"]["_color"],
                                          radius=f["properties"]["_radius"], weight=0.6, fillOpacity=op, opacity=op)
            marker = folium.CircleMarker(radius=4, fill=True)
        elif kind == "line":
            style = lambda f, op=op: dict(color=f["properties"]["_color"], weight=f["properties"]["_radius"] / 2, opacity=op)
            marker = None
        else:
            style = lambda f, op=op: dict(color=f["properties"]["_color"], weight=0.6, fillColor=f["properties"]["_color"],
                                          fillOpacity=op * 0.75, opacity=op)
            marker = None
        if kind == "point" and layer.glow:
            for scale, fo in ((3.2, 0.10), (2.2, 0.18)):
                folium.GeoJson(g[["_color", "_radius", "geometry"]].to_json(), name=f"{layer.name} glow", control=False,
                               show=layer.show, marker=folium.CircleMarker(radius=4, fill=True),
                               style_function=lambda f, s=scale, fo=fo: dict(
                                   color=f["properties"]["_color"], fillColor=f["properties"]["_color"], weight=0,
                                   radius=f["properties"]["_radius"] * s, fillOpacity=fo)).add_to(m)
        folium.GeoJson(g.to_json(), name=layer.name, style_function=style, marker=marker, popup=popup,
                       tooltip=tooltip, show=layer.show,
                       highlight_function=(lambda f: dict(weight=2.5)) if kind != "point" else None).add_to(m)

    if label_tiles:
        folium.map.CustomPane("labels", z_index=650, pointer_events=False).add_to(m)
        folium.TileLayer(S.CARTO_URL.replace("{path}", label_tiles), attr=S.CARTO_ATTR, name="Labels",
                         pane="labels", control=False, overlay=True).add_to(m)

    if layer_control if layer_control is not None else len(layers) > 1:
        folium.LayerControl(position="topright", collapsed=False).add_to(m)

    if not start:
        m.fit_bounds(_bounds(layers, bounds), padding=(30, 30))

    extra = ""
    if switcher:
        from .city import CITIES
        if switcher is True:
            switcher = {k: None for k in CITIES}
        btns = []
        for i, (k, bb) in enumerate(switcher.items()):
            if bb is None:
                c = CITIES[k]
                js = f"{m.get_name()}.flyTo([{c['center'][0]}, {c['center'][1]}], {c['zoom']}, {{duration: 1.4}})"
                label = c["label"]
            else:
                js = f"{m.get_name()}.flyToBounds([[{bb[1]}, {bb[0]}], [{bb[3]}, {bb[2]}]], {{duration: 1.4}})"
                label = k
            btns.append(f"<button onclick=\"document.querySelectorAll('.mk-btns button').forEach(b=>b.classList.remove('on'));"
                        f"this.classList.add('on');{js}\">{label}</button>")
        extra = f'<div class="mk-btns">{"".join(btns)}</div>'

    m.get_root().header.add_child(folium.Element(S.FONTS + f"<title>Day {day:02d} | {title}</title>"
                                                 f"<style>.leaflet-container{{background:{pal['paper']}}}</style>"))
    m.get_root().html.add_child(folium.Element(
        S.card_html(day, prompt, title, subtitle, story, "".join(legends), sources, series, extra)))
    return m


# ======================================================================= Mapbox GL
_MAP_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
__LIB__
<style>
  /* ---- THEME: dark panel by default; body.light switches everything to the light version ---- */
  :root { --panel: rgba(13,17,21,.84); --ink:#eef2f4; --muted:#93a1a8; --line:rgba(255,255,255,.08);
          --title:#f7e7c6; --tip:rgba(10,13,16,.94); --about:#c9d3d7; --hover:rgba(255,255,255,.06);
          --shadow:0 10px 40px rgba(0,0,0,.35); --glow:#ff5c8a;
          --tile:rgba(255,255,255,.035); --tile-line:rgba(255,255,255,.07); --kicker:#c9b79a; }
  body.light { --panel: rgba(248,245,239,.93); --ink:#22262a; --muted:#6f6a62; --line:rgba(60,48,30,.10);
               --title:#1f2326; --tip:rgba(252,250,246,.98); --about:#3f3b35; --hover:rgba(60,48,30,.05);
               --shadow:0 1px 2px rgba(60,48,30,.06), 0 12px 32px rgba(60,48,30,.12); --glow:#e8436f;
               --tile:rgba(255,255,255,.72); --tile-line:rgba(60,48,30,.09); --kicker:#9a7b4f; }
  .hl { color:var(--glow); text-shadow:0 0 10px color-mix(in srgb, var(--glow) 85%, transparent),
        0 0 26px color-mix(in srgb, var(--glow) 50%, transparent); }
  body.light .hl { text-shadow:0 0 14px color-mix(in srgb, var(--glow) 40%, transparent); }
  html, body, #map { margin:0; height:100%; background:__PAPER__; }
  body { font-family: Inter, system-ui, sans-serif; color: var(--ink); -webkit-font-smoothing: antialiased; }
  .panel { position:fixed; top:20px; left:20px; z-index:5; width:290px; padding:20px 20px 14px;
           background:var(--panel); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
           border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); }
  .panel h1 { color:var(--title); font-family: Fraunces, Georgia, serif; font-weight:600; font-size:25px; line-height:1.08;
              letter-spacing:-.01em; margin:0; }
  .panel .sub { font-size:13px; line-height:1.45; color:var(--muted); margin:8px 0 0; }
  .panel header { display:block; }
  .kicker { display:flex; align-items:center; gap:8px; font-size:10.5px; font-weight:600; letter-spacing:.14em;
            text-transform:uppercase; color:var(--kicker); margin:0 0 10px; }
  .kicker::before { content:""; width:16px; height:2px; border-radius:2px; background:var(--glow); }
  .legend.tiles { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; margin:14px 0 2px; }
  .tile { display:flex; flex-direction:column; align-items:flex-start; gap:6px; padding:11px 10px 10px; border-radius:12px;
          background:var(--tile); border:1px solid var(--tile-line); color:var(--ink); cursor:pointer; text-align:left;
          font:500 11.5px/1.3 Inter, system-ui, sans-serif; transition:transform .15s, opacity .15s, border-color .15s; }
  .tile:hover { transform:translateY(-1px); border-color:var(--line); }
  .tile.off { opacity:.38; }
  .tile .num { font:600 21px/1 Inter, system-ui, sans-serif; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
  .tile .lab { color:var(--muted); }
  .tile .bar { width:100%; height:3px; border-radius:3px; }
  .tile .bar.glow { box-shadow:0 0 10px 1px currentColor; }
  .foot { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:12px; }
  .foot .hint { margin:0; }
  .legend { margin:16px 0 4px; display:flex; flex-direction:column; gap:2px; }
  .item { display:flex; align-items:center; gap:10px; width:100%; padding:8px 8px; margin:0 -8px;
          width:calc(100% + 16px); border:0; border-radius:9px; background:transparent; color:var(--ink);
          font:500 13px Inter, system-ui, sans-serif; text-align:left; cursor:pointer; transition:background .15s, opacity .15s; }
  .item:hover { background:var(--hover); }
  .item.off { opacity:.35; }
  .item .dot { width:10px; height:10px; border-radius:50%; flex:none; }
  .item .dot.glow { box-shadow:0 0 0 3px rgba(255,255,255,.04), 0 0 12px 3px currentColor; }
  .item .label { flex:1; }
  .item .n { color:var(--muted); font-variant-numeric: tabular-nums; font-size:12px; }
  .cats { margin:0 0 6px 20px; display:flex; flex-wrap:wrap; gap:4px 12px; font-size:11.5px; color:var(--muted); }
  .cats span::before { content:""; display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px;
                       background:var(--c); vertical-align:1px; }
  .hint { font-size:11px; color:var(--muted); margin:6px 0 0; }
  .about-btn { margin-top:12px; padding:0; border:0; background:none; color:var(--muted); cursor:pointer;
               font:500 12px Inter, system-ui, sans-serif; display:flex; align-items:center; gap:6px; }
  .about-btn:hover { color:var(--ink); }
  .about { font-size:12px; line-height:1.55; color:var(--about); margin-top:10px; padding-top:10px; border-top:1px solid var(--line); }
  .about .credits { color:var(--muted); font-size:11px; margin-top:8px; }
  .about .by { color:var(--muted); font-size:11px; margin-top:6px; }
  .tip { position:fixed; z-index:10; pointer-events:none; display:none; max-width:260px; padding:11px 13px;
         background:var(--tip); color:var(--ink); border:1px solid var(--line); border-radius:11px;
         box-shadow:var(--shadow); font-size:12px; }
  .tip .t-name { font-weight:600; font-size:14px; line-height:1.25; }
  .tip .t-kind { font-size:11px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; margin:3px 0 7px; }
  .tip .row { display:flex; justify-content:space-between; gap:14px; padding:2px 0; }
  .tip .row span:first-child { color:var(--muted); }
  .tip .row span:last-child { text-align:right; font-variant-numeric: tabular-nums; }
  .mapboxgl-ctrl-group, .maplibregl-ctrl-group { background:var(--panel)!important; border:1px solid var(--line); }
  body:not(.light) .mapboxgl-ctrl-group button span, body:not(.light) .maplibregl-ctrl-group button span { filter: invert(1); }
  @media (max-width: 640px) {
    .panel { top:auto; bottom:12px; left:12px; right:12px; width:auto; padding:14px 16px 10px; }
    .panel h1 { font-size:20px; } .panel .sub { display:none; } .hint { display:none; }
    .tile .num { font-size:18px; } .kicker { margin-bottom:6px; }
  }
</style></head>
<body class="__THEME__">
<div id="map"></div>
<aside class="panel">
  <header>__KICKER__
  <h1>__TITLE_HTML__</h1>
  __SUB__</header>
  <div class="legend" id="legend"></div>
  <p class="hint">Hover a point for details. Click the legend to show or hide a layer.</p>
  <button class="about-btn" id="aboutBtn" aria-expanded="false">&#9432;&nbsp; About this map</button>
  <div class="about" id="about" hidden>__ABOUT__<div class="credits">__CREDITS__</div><div class="by">__AUTHOR__</div></div>
</aside>
<div class="tip" id="tip"></div>
<script>
const C = __CONFIG__;
const D = __DATA__;
const lib = C.mapbox ? mapboxgl : maplibregl;
if (C.mapbox) mapboxgl.accessToken = C.token;
const narrow = window.innerWidth < 640;
const pad = narrow ? {top:30, bottom:200, left:20, right:20} : {top:50, bottom:50, left:340, right:50};
const map = new lib.Map(Object.assign({container:'map', style:C.style, bounds:C.bounds,
  fitBoundsOptions:{padding:pad}, pitch:C.pitch, bearing:C.bearing},
  C.mapbox ? {projection:C.projection} : {}));
if (!C.mapbox && C.projection === 'globe') map.on('style.load', () => { try { map.setProjection({type:'globe'}); } catch(e) {} });
map.addControl(new lib.NavigationControl({showCompass:false}), 'bottom-right');

const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const zr = k => ['interpolate', ['linear'], ['zoom'], 2, ['*', ['get','_radius'], k[0]], 10, ['*', ['get','_radius'], k[1]]];

map.on('load', () => {
  C.layers.forEach(L => {
    map.addSource(L.id, {type:'geojson', data:D[L.id]});
    const vis = L.visible ? 'visible' : 'none';
    if (L.kind === 'point') {
      if (L.glow) map.addLayer({id:L.id+'-glow', type:'circle', source:L.id, layout:{visibility:vis},
        paint:{'circle-color':['get','_color'], 'circle-radius':zr([2.6, 5]), 'circle-blur':1, 'circle-opacity':0.55}});
      map.addLayer({id:L.id, type:'circle', source:L.id, layout:{visibility:vis},
        paint:{'circle-color':['get','_color'], 'circle-radius':zr([0.75, 1.6]), 'circle-opacity':L.opacity,
               'circle-stroke-width':L.glow || L.size ? 0.8 : 0, 'circle-stroke-color':'rgba(255,255,255,0.55)'}});
    } else if (L.kind === 'line') {
      map.addLayer({id:L.id, type:'line', source:L.id, layout:{visibility:vis},
        paint:{'line-color':['get','_color'], 'line-width':['/', ['get','_radius'], 2], 'line-opacity':L.opacity}});
    } else {
      map.addLayer({id:L.id, type:'fill', source:L.id, layout:{visibility:vis},
        paint:{'fill-color':['get','_color'], 'fill-opacity':L.opacity * 0.7, 'fill-outline-color':'rgba(255,255,255,0.15)'}});
    }
  });

  // legend = layer switches
  const legend = document.getElementById('legend');
  const asTiles = !!C.tiles;
  if (asTiles) legend.classList.add('tiles');
  C.layers.forEach(L => {
    const b = document.createElement('button');
    b.className = (asTiles ? 'tile' : 'item') + (L.visible ? '' : ' off');
    b.innerHTML = asTiles
      ? `<span class="bar ${L.glow ? 'glow' : ''}" style="background:${L.color};color:${L.color}"></span>
         <span class="num">${L.count.toLocaleString()}</span><span class="lab">${esc(L.name).replace(' · ', '<br>')}</span>`
      : `<span class="dot ${L.glow ? 'glow' : ''}" style="background:${L.color};color:${L.color}"></span>
         <span class="label">${esc(L.name)}</span><span class="n">${L.count.toLocaleString()}</span>`;
    b.title = 'Show / hide ' + L.name;
    b.onclick = () => {
      const on = b.classList.toggle('off');
      [L.id, L.id + '-glow'].forEach(id => map.getLayer(id) && map.setLayoutProperty(id, 'visibility', on ? 'none' : 'visible'));
    };
    legend.appendChild(b);
    if (L.cats && L.cats.length > 1) {
      const c = document.createElement('div'); c.className = 'cats';
      c.innerHTML = L.cats.map(([n, col]) => `<span style="--c:${col}">${esc(n)}</span>`).join('');
      legend.appendChild(c);
    }
  });

  // hover cards
  const tip = document.getElementById('tip');
  const byId = Object.fromEntries(C.layers.map(L => [L.id, L]));
  const ids = C.layers.slice().reverse().map(L => L.id);
  const show = e => {
    const live = ids.filter(id => map.getLayoutProperty(id, 'visibility') !== 'none');
    const f = live.length ? map.queryRenderedFeatures(e.point, {layers: live}) : [];
    if (!f.length) { tip.style.display = 'none'; map.getCanvas().style.cursor = ''; return; }
    const p = f[0].properties, L = byId[f[0].layer.id];
    const rows = Object.entries(L.fields).map(([k, lab]) => [lab, p[k]])
      .filter(([, v]) => v !== undefined && v !== null && !['', 'n/a', 'nan', 'None', 'null'].includes(String(v)));
    tip.innerHTML = `<div class="t-name">${esc(p._tip)}</div><div class="t-kind" style="color:${L.color}">${esc(L.name)}</div>`
      + rows.map(([a, b]) => `<div class="row"><span>${esc(a)}</span><span>${esc(b)}</span></div>`).join('');
    tip.style.display = 'block';
    const r = tip.getBoundingClientRect(), x = e.originalEvent.clientX, y = e.originalEvent.clientY;
    tip.style.left = (x + 16 + r.width > innerWidth ? x - r.width - 16 : x + 16) + 'px';
    tip.style.top = (y + 16 + r.height > innerHeight ? y - r.height - 16 : y + 16) + 'px';
    map.getCanvas().style.cursor = 'pointer';
  };
  map.on('mousemove', show);
  map.on('click', show);
  map.getCanvas().addEventListener('mouseleave', () => tip.style.display = 'none');
});

const btn = document.getElementById('aboutBtn'), about = document.getElementById('about');
btn.onclick = () => { about.hidden = !about.hidden; btn.setAttribute('aria-expanded', String(!about.hidden)); };
</script></body></html>"""


def mapbox(layers, *, title, subtitle="", about="", sources="", series="ai", bounds=None,
           style="dark", projection="mercator", pitch=0, bearing=0, token=None, author=True,
           highlight=None, kicker=None, tiles=False, story=None, **_ignored):
    """
    A clean, portfolio-ready Mapbox GL page (returns HTML text; publish.save() writes it).

    title / subtitle  short heading on the panel (keep the subtitle to one line)
    about             a few sentences, hidden behind "About this map"
    sources           data credits, shown inside About
    Each Layer becomes a legend switch (click to show/hide) and points get hover cards built from Layer.popup.
    projection: 'mercator' | 'globe' | 'albers' | 'equalEarth' | 'naturalEarth' | 'winkelTripel'
    style: 'dark' keeps the dark panel; 'light' / 'standard' / 'satellite' switch the panel to light
    highlight: a word in the title to make glow, e.g. highlight="AI"
    kicker:    optional small label above the title, e.g. kicker=style.UMBRELLA
    tiles:     True shows the legend as stat tiles instead of a list (off by default)
    No Mapbox token -> falls back to MapLibre + CARTO so it still works.
    """
    import html as _h
    light = style in ("light", "standard")
    pal = S.SERIES[series]
    from .paths import mapbox_token
    token = token or mapbox_token()
    use_mapbox = token.startswith("pk.")
    about = about or story or ""

    cfg_layers, data = [], {}
    for i, layer in enumerate(layers):
        g, _ = prepare(layer, series)
        if light and layer.color == S.LOOK["existing"]:       # grey-blue is too faint on a light map
            g["_color"] = S.LOOK["existing_on_light"]
        sid = f"l{i}"
        data[sid] = json.loads(g.to_json())
        cats = []
        if "_cat" in g:
            cats = g[["_cat", "_color"]].drop_duplicates().values.tolist()
            if layer.order:
                cats.sort(key=lambda c: layer.order.index(c[0]) if c[0] in layer.order else 99)
        cfg_layers.append(dict(
            id=sid, name=layer.name, kind=_geom_kind(g), count=int(len(g)),
            color=(g["_color"].iloc[0] if (layer.color and len(g)) else layer.color or
                   (g["_color"].iloc[0] if len(g) else pal["accent"])),
            glow=bool(layer.glow), size=bool(layer.size_column), opacity=layer.opacity, visible=bool(layer.show),
            cats=cats, fields={f"_p_{k}": v for k, v in (layer.popup or {}).items() if f"_p_{k}" in g},
        ))

    (s_, w_), (n_, e_) = _bounds(layers, bounds)
    if use_mapbox:
        lib = ('<link href="https://api.mapbox.com/mapbox-gl-js/v3.7.0/mapbox-gl.css" rel="stylesheet">'
               '<script src="https://api.mapbox.com/mapbox-gl-js/v3.7.0/mapbox-gl.js"></script>')
        style_url, basemap = S.MAPBOX_STYLES[style], "Basemap © Mapbox © OpenStreetMap"
    else:
        lib = ('<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">'
               '<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>')
        style_url, basemap = S.MAPLIBRE_FALLBACK[style], "Basemap © CARTO © OpenStreetMap"
        print("No Mapbox token found, using the free MapLibre fallback. Add mapbox_token.txt for real Mapbox.")

    cfg = dict(tiles=bool(tiles), mapbox=use_mapbox, token=token, style=style_url, projection=projection, pitch=pitch,
               bearing=bearing, bounds=[[w_, s_], [e_, n_]], layers=cfg_layers)
    credits = " · ".join(x for x in [sources, basemap] if x)
    title_html = _h.escape(title)
    if highlight and highlight in title:
        title_html = title_html.replace(_h.escape(highlight), f'<span class="hl">{_h.escape(highlight)}</span>', 1)
    out = _MAP_HTML
    kicker = kicker or ""
    for k, v in {"__KICKER__": f'<p class="kicker">{_h.escape(kicker)}</p>' if kicker else "", "__TITLE_HTML__": title_html, "__THEME__": "light" if light else "dark", "__TITLE__": _h.escape(title), "__SUB__": f'<p class="sub">{_h.escape(subtitle)}</p>' if subtitle else "",
                 "__ABOUT__": about, "__CREDITS__": credits, "__AUTHOR__": S.AUTHOR if author else "",
                 "__PAPER__": pal["paper"], "__LIB__": lib}.items():
        out = out.replace(k, v)
    return out.replace("__CONFIG__", json.dumps(cfg)).replace("__DATA__", json.dumps(data))


# ======================================================================= static PNG
def static(layers, *, day, prompt, title, subtitle="", sources="", series="ai", bounds=None,
           size=(10.8, 13.5), basemap=False, crs=None):
    """
    Matplotlib poster, 4:5 portrait (LinkedIn-friendly at 100 dpi = 1080x1350).
    crs: e.g. "EPSG:5070" for a proper US Albers look. basemap=True adds CARTO tiles (needs contextily).
    """
    import matplotlib.pyplot as plt
    pal = S.SERIES[series]
    fig, ax = plt.subplots(figsize=size, dpi=100)
    fig.patch.set_facecolor(pal["paper"])
    ax.set_facecolor(pal["paper"])
    target = crs or ("EPSG:3857" if basemap else WGS84)
    legends = []
    for layer in layers:
        g, _ = prepare(layer, series)
        g = g.to_crs(target)
        kind = _geom_kind(g)
        if kind == "point":
            if layer.glow:
                for s, a in ((9, 0.08), (4.5, 0.16)):
                    g.plot(ax=ax, color=g["_color"], markersize=(g["_radius"] ** 2) * 1.4 * s, alpha=a, linewidth=0)
            g.plot(ax=ax, color=g["_color"], markersize=(g["_radius"] ** 2) * 1.4, alpha=layer.opacity, linewidth=0)
        elif kind == "line":
            g.plot(ax=ax, color=g["_color"], linewidth=g["_radius"] / 3, alpha=layer.opacity)
        else:
            g.plot(ax=ax, color=g["_color"], edgecolor=pal["paper"], linewidth=0.3, alpha=layer.opacity)
        if layer.legend and "_cat" in g:
            legends += list(g[["_cat", "_color"]].drop_duplicates().itertuples(index=False, name=None))
        elif layer.legend and not layer.column:
            legends.append((layer.name, g["_color"].iloc[0] if len(g) else pal["accent"]))
    if bounds is not None:
        b = gpd.GeoSeries.from_xy([bounds[0], bounds[2]], [bounds[1], bounds[3]], crs=WGS84).to_crs(target)
        ax.set_xlim(b.x.min(), b.x.max()); ax.set_ylim(b.y.min(), b.y.max())
    if basemap:
        try:
            import contextily as cx
            src = cx.providers.CartoDB.DarkMatterNoLabels if pal["basemap"] == "dark" else cx.providers.CartoDB.PositronNoLabels
            cx.add_basemap(ax, source=src, attribution_size=6)
        except Exception as e:
            print("basemap skipped:", e)
    ax.set_axis_off()
    fig.text(0.06, 0.93, title, color=pal["ink"], fontsize=26, family="serif", weight="bold")
    if subtitle:
        fig.text(0.06, 0.905, subtitle, color=pal["muted"], fontsize=13, family="sans-serif")
    top = 0.05 + len(legends) * 0.022
    for i, (label, c) in enumerate(legends):
        fig.text(0.06, top - i * 0.022, "●", color=c, fontsize=13)
        fig.text(0.085, top - i * 0.022, label, color=pal["ink"], fontsize=11)
    fig.text(0.06, 0.025, f"{('Data: ' + sources + '   ') if sources else ''}{S.AUTHOR}",
             color=pal["muted"], fontsize=9)
    ax.set_position([0.03, 0.06 + len(legends) * 0.022, 0.94, 0.8 - len(legends) * 0.022])
    return fig
