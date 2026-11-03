"""
poster.py : shared pieces for the static 4:5 maps (1080 x 1350).

Fonts live next to this file, so every laptop draws the same poster.
Each map picks the type that suits its subject. All are under the SIL Open Font License.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
SERIF = HERE / "FrauncesMap-SemiBold.ttf"
SANS = HERE / "InterMap-Regular.ttf"
SANS_BOLD = HERE / "InterMap-SemiBold.ttf"
OLD = HERE / "IMFellEnglish-Regular.ttf"
OLD_ITALIC = HERE / "IMFellEnglish-Italic.ttf"
SLAB = HERE / "ZillaSlab-Bold.ttf"             # trail signs and park maps
BARLOW = HERE / "Barlow-Regular.ttf"
BARLOW_BOLD = HERE / "Barlow-SemiBold.ttf"
CINZEL = HERE / "Cinzel-Bold.ttf"              # carved in stone, for monuments
LATO = HERE / "Lato-Regular.ttf"
LATO_BOLD = HERE / "Lato-Bold.ttf"
ABRIL = HERE / "AbrilFatface-Regular.ttf"      # high-contrast poster type
YATRA = HERE / "YatraOne-Regular.ttf"          # hand-painted Indian signage
POPPINS = HERE / "Poppins-Regular.ttf"
POPPINS_BOLD = HERE / "Poppins-SemiBold.ttf"
DM_SERIF = HERE / "DMSerifDisplay-Regular.ttf"
BEBAS = HERE / "BebasNeue-Regular.ttf"         # vintage ski poster

INK = "#0e1418"
PAPER = "#eef3f4"
MUTED = "#9fb0b6"
PINK = "#ff5c8a"
CREDIT_COLOR = "#7d8c92"


def font(path, size):
    return FontProperties(fname=str(path), size=size)


def page(bg=INK):
    """A blank poster with one axes filling the whole page."""
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=bg)
    ax.set_axis_off()
    return fig, ax


def title(fig, text, subtitle=None, x=0.06, y=0.945, size=46, color=PAPER, sub_color=MUTED, sub_size=17, line=1.04,
          path=SERIF, sub_path=SANS):
    """Serif title, one short line under it. Returns the y just below the block."""
    lines = text.split("\n")
    step = size * line / 1350 * 100 / 72
    for i, t in enumerate(lines):
        fig.text(x, y - i * step, t, color=color, fontproperties=font(path, size), va="top")
    below = y - len(lines) * step - 0.008
    if subtitle:
        for j, s in enumerate(subtitle.split("\n")):
            fig.text(x, below - j * sub_size * 1.36 / 1350 * 100 / 72, s, color=sub_color,
                     fontproperties=font(sub_path, sub_size), va="top")
    return below


def credit(fig, text, x=0.06, y=0.02, color=CREDIT_COLOR, ha="left", size=9.5, path=SANS, name_path=SANS_BOLD, name_color=None):
    """Her name, with the data credits under it."""
    lines = text.count("\n") + 1
    fig.text(x, y, text, color=color, fontproperties=font(path, size), ha=ha, va="bottom", linespacing=1.35)
    fig.text(x, y + lines * size * 1.42 / 1350 * 100 / 72 + 0.003, "Map by Gautami Kasat", color=name_color or color,
             fontproperties=font(name_path, size + 2.5), ha=ha, va="bottom")


def glow(ax, x, y, color=PINK, width=2.4, zorder=6):
    """A line with a soft halo."""
    for w, a in ((width * 6, 0.07), (width * 3.4, 0.14), (width * 1.9, 0.28)):
        ax.plot(x, y, color=color, lw=w, alpha=a, solid_capstyle="round", zorder=zorder)
    ax.plot(x, y, color=color, lw=width, solid_capstyle="round", zorder=zorder + 1)


def halo_text(ax, x, y, text, color=PAPER, path=SANS, size=12, halo=INK, halo_width=3, **kw):
    t = ax.text(x, y, text, color=color, fontproperties=font(path, size), **kw)
    t.set_path_effects([pe.withStroke(linewidth=halo_width, foreground=halo)])
    return t


def merc(lon, lat):
    """Lon/lat to web mercator meters."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return lon * 20037508.34 / 180, np.log(np.tan((90 + lat) * np.pi / 360)) * 20037508.34 / np.pi


def fit_frame(ax, west, south, east, north):
    """Set the view in mercator, widening one side so it fills 4:5 exactly."""
    x0, y0 = merc(west, south)
    x1, y1 = merc(east, north)
    w, h = x1 - x0, y1 - y0
    if w / h < 0.8:
        pad = (h * 0.8 - w) / 2
        x0, x1 = x0 - pad, x1 + pad
    else:
        pad = (w / 0.8 - h) / 2
        y0, y1 = y0 - pad, y1 + pad
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return x0, y0, x1, y1
