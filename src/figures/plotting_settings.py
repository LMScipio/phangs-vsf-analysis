"""Shared matplotlib settings of the paper figures, used through plt.rc_context(rc=paper_rcparams()).
"""

from matplotlib import patheffects

COLUMN_WIDTH_PT = 417.47307
SAVEFIG_PAD_INCHES = 0.12

def column_width_inches():
    return COLUMN_WIDTH_PT * (1.0 / 72.27)

def paper_rcparams():
    return {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman", "Nimbus Roman", "serif"],
        "font.size": 11.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.linewidth": 1.35,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "figure.dpi": 150,
    }

def add_subplot_label(ax, letter, corner, fontsize, stroke=False):
    """Bold panel label such as (a) inside one corner of an axes."""
    if "left" in corner:
        x, ha = 0.03, "left"
    else:
        x, ha = 1.0 - 0.03, "right"
    if "upper" in corner:
        y, va = 1.0 - 0.03, "top"
    else:
        y, va = 0.03, "bottom"
    t = ax.text(
        x, y, f"({letter})",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        zorder=25,
        clip_on=False,
    )
    if stroke:
        t.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="w"), patheffects.Normal()])
