"""
publish.py : the last cell of every notebook.

    publish.save(m, day=1, slug="points-ai", prompt="Points", title="...", series="ai",
                 tool="Python + Mapbox", notebook="day01_points_ai.ipynb")

What save() does in one go:
  1. writes the map into output/            (interactive HTML, or a PNG for static maps)
  2. makes a PNG thumbnail in png_thumbnail/ (for MUSA + LinkedIn; needs Playwright, see README)
  3. starts captions/dayNN_slug.md from captions/_template.md, links already filled in
  4. records the map in output/maps.json and rebuilds the gallery, output/index.html

For ArcGIS Pro days:
    publish.to_arcgis({"datacenters": dc, "ai_campuses": big}, day=12)   -> data/arcgis/day12.gpkg
"""
import json
import subprocess
import sys
from datetime import date

from . import style as S
from .paths import ROOT, OUTPUT, THUMBS, CAPTIONS, ARCGIS, WGS84

REGISTRY = OUTPUT / "maps.json"


def _registry():
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []


def snapshot(html_path, png_path, size=(1600, 1000), wait_ms=5000):
    """Screenshot an HTML map into a PNG with a headless browser. Runs in a separate process so it works in Jupyter."""
    code = f"""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--enable-webgl", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={{"width": {size[0]}, "height": {size[1]}}}, device_scale_factor=1)
    pg.goto({html_path.resolve().as_uri()!r}, wait_until="networkidle", timeout=60000)
    pg.wait_for_timeout({wait_ms})
    pg.screenshot(path={str(png_path)!r})
    b.close()
"""
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and png_path.exists():
            return True
        print("thumbnail skipped:", (r.stderr or "").strip().splitlines()[-1:] or "unknown error")
    except Exception as e:
        print("thumbnail skipped:", e)
    print(f"  -> take a screenshot yourself (Win+Shift+S) and save it as png_thumbnail/{png_path.name}")
    return False


def _caption_stub(stem, day, prompt, title, subtitle, series, interactive, live_url, code_url, thumb, credits="", study=None):
    out = CAPTIONS / f"{stem}.md"
    tpl = CAPTIONS / "_template.md"
    if out.exists() or not tpl.exists():
        return out
    values = dict(day=f"{day:02d}", day_n=str(day), hashtags=S.SERIES[series]["hashtags"], prompt=prompt, title=title, subtitle=subtitle,
                  series=S.SERIES[series]["name"], interactive="yes" if interactive else "no",
                  live_url=live_url, code_url=code_url, thumbnail=thumb, credits=credits or "(add data credits)",
                  study=f"{study} · {S.STUDIES.get(study, '')}" if study else "(none yet)")
    text = tpl.read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    out.write_text(text, encoding="utf-8")
    return out


def save(obj, *, day, slug, prompt, title, subtitle="", series="ai", tool="Python",
         notebook=None, credits="", study=None, thumbnail=True):
    """credits: full citation text for the caption file, e.g. style.credits("im3", "epoch", full=True)"""
    stem = f"day{day:02d}_{slug}"
    kind = type(obj).__name__
    thumb = THUMBS / f"{stem}.png"
    interactive = True

    if kind == "Map":                                   # folium
        out = OUTPUT / f"{stem}.html"
        obj.save(out)
    elif isinstance(obj, str):                          # Mapbox HTML text
        out = OUTPUT / f"{stem}.html"
        out.write_text(obj, encoding="utf-8")
    elif kind == "Figure":                              # matplotlib
        out = OUTPUT / f"{stem}.png"
        obj.savefig(out, dpi=100, facecolor=obj.get_facecolor())
        obj.savefig(thumb, dpi=100, facecolor=obj.get_facecolor())
        interactive = False
    else:
        raise TypeError(f"Don't know how to save a {kind}")

    if interactive and thumbnail:
        snapshot(out, thumb)

    live_url = f"{S.PAGES_URL}/output/{out.name}"
    code_url = f"{S.CODE_URL}/{notebook}" if notebook else S.CODE_URL
    cap = _caption_stub(stem, day, prompt, title, subtitle, series, interactive, live_url, code_url, thumb.name, credits, study)

    reg = [r for r in _registry() if r["file"] != out.name]
    reg.append(dict(day=day, prompt=prompt, title=title, subtitle=subtitle, series=series, study=study, tool=tool,
                    interactive=interactive, file=out.name,
                    preview=f"../png_thumbnail/{thumb.name}" if thumb.exists() else None,
                    updated=date.today().isoformat()))
    reg.sort(key=lambda r: (r["day"], r["file"]))
    REGISTRY.write_text(json.dumps(reg, indent=2))
    build_gallery()

    print(f"map        output/{out.name}")
    print(f"thumbnail  png_thumbnail/{thumb.name}" + ("" if thumb.exists() else "  (missing, see above)"))
    print(f"caption    captions/{cap.name}")
    print(f"live link  {live_url}  (after you push)")


def to_arcgis(layers, day):
    """
    One GeoPackage for ArcGIS Pro: data/arcgis/dayNN.gpkg (one layer per entry).
    In Pro: Catalog -> Folders -> Add Folder Connection -> data/arcgis -> drag layers onto the map.
    """
    out = ARCGIS / f"day{day:02d}.gpkg"
    for name, gdf in layers.items():
        g = gdf.to_crs(WGS84).copy()
        g = g[[c for c in g.columns if not c.startswith("_")]]
        g.to_file(out, layer=name, driver="GPKG")
    print(f"ArcGIS Pro package: data/arcgis/{out.name}  ({', '.join(layers)})")
    return out


def build_gallery():
    reg = _registry()
    names = {k: v["name"] for k, v in S.SERIES.items()}
    cards = "".join(f"""
      <a class="card {r['series']}" data-study="{r.get('study') or ''}" href="{r['file']}">
        {f'<img src="{r["preview"]}" alt="">' if r.get('preview') else ''}
        <div class="chip">Day {r['day']:02d} · {r['prompt']}{' · interactive' if r.get('interactive') else ''}</div>
        <h2>{r['title']}</h2>
        <p>{r.get('subtitle', '')}</p>
        <div class="meta">{S.STUDIES.get(r.get('study'), names.get(r['series'], r['series']))} · {r.get('tool', '')}</div>
      </a>""" for r in reg)
    (OUTPUT / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>30 Day Map Challenge 2026 | {S.AUTHOR}</title>{S.FONTS}
<style>
  :root {{ --ink:#1f1d1a; --muted:#6f675e; --paper:#f6f2eb; --card:#fffdf9; --line:#e5ddd0; --ai:#b57a08; --city:#c43d50; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --ink:#eef2f3; --muted:#a3b0b5; --paper:#11161a; --card:#1a2126;
    --line:#29333a; --ai:#f2b134; --city:#ef7a88; }} }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Inter,system-ui,sans-serif }}
  header, main, footer {{ max-width:1120px; margin:0 auto; padding-left:16px; padding-right:16px }}
  header {{ padding-top:64px; padding-bottom:20px }}
  h1 {{ font-family:Fraunces,Georgia,serif; font-size:clamp(34px,6vw,58px); line-height:1.02; margin:0 0 14px; font-weight:600 }}
  .lede {{ max-width:680px; color:var(--muted); font-size:17px; line-height:1.55; margin:0 0 18px }}
  .tabs button {{ font:600 13px Inter,sans-serif; padding:7px 13px; border-radius:999px; border:1px solid var(--line);
    background:var(--card); color:var(--ink); cursor:pointer; margin:0 6px 6px 0 }}
  .tabs button.on {{ background:var(--ink); color:var(--paper) }}
  main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:16px; padding-bottom:72px }}
  .card {{ display:block; text-decoration:none; color:inherit; background:var(--card); border:1px solid var(--line);
    border-radius:14px; padding:18px; transition:transform .15s, border-color .15s }}
  .card:hover {{ transform:translateY(-2px) }}
  .card.ai:hover {{ border-color:var(--ai) }} .card.city:hover {{ border-color:var(--city) }}
  .card img {{ width:100%; aspect-ratio:16/10; object-fit:cover; border-radius:8px; margin-bottom:12px; display:block }}
  .chip {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase }}
  .card.ai .chip {{ color:var(--ai) }} .card.city .chip {{ color:var(--city) }}
  h2 {{ font-family:Fraunces,Georgia,serif; font-size:21px; margin:8px 0 6px; font-weight:600 }}
  p {{ margin:0; color:var(--muted); font-size:14px; line-height:1.45 }}
  .meta {{ margin-top:12px; font-size:12px; color:var(--muted) }}
  footer {{ color:var(--muted); font-size:13px; padding-bottom:48px }}
  footer a {{ color:inherit }}
</style></head><body>
<header>
  <h1>30 Day Map Challenge 2026</h1>
  <p class="lede"><b>{names['ai']}</b> follows America's data centers, the ones running today and the giant AI campuses
  on the way, into the water, power and climate risk around them. <b>{names['city']}</b> is the lighter thread:
  how New York, Philadelphia, DC, San Francisco, Austin and Seattle look, sound and taste on a map.</p>
  <div class="tabs"><button class="on" data-f="all">All</button>{"".join(
      f'<button data-f="{k}">{v}</button>' for k, v in S.STUDIES.items() if any(r.get("study") == k for r in reg))}</div>
</header>
<main>{cards}
</main>
<footer>{S.AUTHOR} · Python, GeoPandas, Mapbox, ArcGIS Pro ·
  <a href="https://github.com/{S.GITHUB_USER}/{S.REPO}">code for every map</a></footer>
<script>
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {{
  document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.card').forEach(c => c.style.display = (b.dataset.f === 'all' || c.dataset.study === b.dataset.f) ? '' : 'none');
}});
</script></body></html>""", encoding="utf-8")
