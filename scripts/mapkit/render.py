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
def mapbox(layers, *, day, prompt, title, subtitle="", story="", sources="", series="ai",
           bounds=None, style="dark", projection="mercator", pitch=0, bearing=0, token=None):
    """
    Mapbox GL JS page (returns HTML text; publish() saves it).
    projection: 'mercator' | 'globe' | 'albers' | 'equalEarth' | 'naturalEarth' | 'winkelTripel'
    token: your public Mapbox token, or set the MAPBOX_TOKEN environment variable.
           With no token it falls back to MapLibre + CARTO so the map still works.
    """
    pal = S.SERIES[series]
    from .paths import mapbox_token
    token = token or mapbox_token()
    use_mapbox = token.startswith("pk.")
    sources_js, layers_js, legends, pop_layers = [], [], [], []
    for i, layer in enumerate(layers):
        g, legend = prepare(layer, series)
        if legend:
            legends.append(legend)
        kind = _geom_kind(g)
        sid = f"l{i}"
        sources_js.append(f"map.addSource('{sid}', {{type:'geojson', data:{g.to_json()}}});")
        vis = "visible" if layer.show else "none"
        if kind == "point" and layer.glow:
            layers_js.append(f"""map.addLayer({{id:'{sid}-glow', type:'circle', source:'{sid}', layout:{{visibility:'{vis}'}},
              paint:{{'circle-color':['get','_color'], 'circle-radius':['*',['get','_radius'],['interpolate',['linear'],['zoom'],2,2.4,10,5]],
              'circle-blur':1, 'circle-opacity':0.6}}}});""")
        if kind == "point":
            layers_js.append(f"""map.addLayer({{id:'{sid}', type:'circle', source:'{sid}', layout:{{visibility:'{vis}'}},
              paint:{{'circle-color':['get','_color'], 'circle-radius':['*',['get','_radius'],['interpolate',['linear'],['zoom'],2,0.7,10,1.6]],
              'circle-opacity':{layer.opacity}, 'circle-stroke-width':0.4, 'circle-stroke-color':'{pal['paper']}'}}}});""")
        elif kind == "line":
            layers_js.append(f"""map.addLayer({{id:'{sid}', type:'line', source:'{sid}', layout:{{visibility:'{vis}'}},
              paint:{{'line-color':['get','_color'], 'line-width':['/',['get','_radius'],2], 'line-opacity':{layer.opacity}}}}});""")
        else:
            layers_js.append(f"""map.addLayer({{id:'{sid}', type:'fill', source:'{sid}', layout:{{visibility:'{vis}'}},
              paint:{{'fill-color':['get','_color'], 'fill-opacity':{layer.opacity * 0.75}, 'fill-outline-color':'{pal['paper']}'}}}});""")
        fields = {c: layer.popup[c[3:]] for c in g.columns if c.startswith("_p_")}
        pop_layers.append((sid, fields))

    popup_js = "".join(f"""
      map.on('click','{sid}',e=>{{const p=e.features[0].properties;
        new {"mapboxgl" if use_mapbox else "maplibregl"}.Popup().setLngLat(e.lngLat).setHTML(
          '<div style="font:13px Inter,sans-serif;color:#1e2a2f"><b>'+p._tip+'</b>'
          +Object.entries({json.dumps(fields)}).map(([k,v])=>'<br>'+v+': <b>'+p[k]+'</b>').join('')+'</div>').addTo(map);}});
      map.on('mouseenter','{sid}',()=>map.getCanvas().style.cursor='pointer');
      map.on('mouseleave','{sid}',()=>map.getCanvas().style.cursor='');""" for sid, fields in pop_layers)

    (s_, w_), (n_, e_) = _bounds(layers, bounds)
    if use_mapbox:
        lib = ('<link href="https://api.mapbox.com/mapbox-gl-js/v3.7.0/mapbox-gl.css" rel="stylesheet">'
               '<script src="https://api.mapbox.com/mapbox-gl-js/v3.7.0/mapbox-gl.js"></script>')
        init = (f"mapboxgl.accessToken='{token}'; const map=new mapboxgl.Map({{container:'map', style:'{S.MAPBOX_STYLES[style]}', "
                f"projection:'{projection}', pitch:{pitch}, bearing:{bearing}}});")
        credit = "Basemap &copy; Mapbox &copy; OpenStreetMap"
    else:
        lib = ('<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">'
               '<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>')
        proj = "globe" if projection == "globe" else "mercator"
        init = (f"const map=new maplibregl.Map({{container:'map', style:'{S.MAPLIBRE_FALLBACK[style]}', pitch:{pitch}, bearing:{bearing}}});"
                + (f"map.on('style.load',()=>{{try{{map.setProjection({{type:'{proj}'}})}}catch(e){{}}}});" if proj == "globe" else ""))
        credit = "Basemap &copy; CARTO &copy; OpenStreetMap"
        print("No Mapbox token found, using the free MapLibre fallback. Set MAPBOX_TOKEN for real Mapbox.")

    card = S.card_html(day, prompt, title, subtitle, story, "".join(legends),
                       (sources + " &middot; " if sources else "") + credit, series)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Day {day:02d} | {title}</title>
{S.FONTS}{lib}
<style>html,body,#map{{margin:0;height:100%;background:{pal['paper']}}}</style></head>
<body><div id="map"></div>{card}
<script>
{init}
map.addControl(new {"mapboxgl" if use_mapbox else "maplibregl"}.NavigationControl(), 'bottom-right');
map.on('load',()=>{{
  {''.join(sources_js)}
  {''.join(layers_js)}
  {popup_js}
  map.fitBounds([[{w_},{s_}],[{e_},{n_}]], {{padding:40, duration:0}});
}});
</script></body></html>"""


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
    fig.text(0.06, 0.955, f"DAY {day:02d} · {prompt.upper()}", color=pal["accent"], fontsize=11, family="sans-serif", weight="bold")
    fig.text(0.06, 0.915, title, color=pal["ink"], fontsize=26, family="serif", weight="bold")
    if subtitle:
        fig.text(0.06, 0.89, subtitle, color=pal["muted"], fontsize=13, family="sans-serif")
    top = 0.05 + len(legends) * 0.022
    for i, (label, c) in enumerate(legends):
        fig.text(0.06, top - i * 0.022, "●", color=c, fontsize=13)
        fig.text(0.085, top - i * 0.022, label, color=pal["ink"], fontsize=11)
    fig.text(0.06, 0.025, f"{('Data: ' + sources + '   ') if sources else ''}{S.AUTHOR} · {S.HASHTAG}",
             color=pal["muted"], fontsize=9)
    ax.set_position([0.03, 0.06 + len(legends) * 0.022, 0.94, 0.8 - len(legends) * 0.022])
    return fig
