"""Number of galaxies per Mach number classification against the Pan and Scannapieco (2011) curves.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.figures.plotting_settings import SAVEFIG_PAD_INCHES, paper_rcparams

REGIME_COLORS = {
    "0.9": "#1F4E79", "1.4": "#1F4E79", "2.1": "#4F8F8B", "3.0": "#4F8F8B", "4.6": "#D7B46A", "6.1": "#D7B46A",
}

def histogram_bin(result):
    """Sort key, tick label, colour and hatching of the histogram bin of one classification."""
    lo, hi = result["range_lo"], result["range_hi"]
    if lo == hi:
        return (0, float(lo)), lo, REGIME_COLORS[lo], None
    if lo is None:
        return (1, 0.0, float(hi)), f"≤{hi}", REGIME_COLORS["0.9"], "///"
    if hi is None:
        return (1, float(lo), float("inf")), f"≥{lo}", REGIME_COLORS[lo], "///"
    return (1, float(lo), float(hi)), f"{lo}–{hi}", REGIME_COLORS[lo], "///"

def plot_mach_classification_histogram(results, outpath):
    counts = {}
    bins = {}
    for r in results:
        key, tick, color, hatch = histogram_bin(r)
        counts[tick] = counts.get(tick, 0) + 1
        bins[tick] = (key, color, hatch)
    ticks = sorted(counts, key=lambda t: bins[t][0])
    x = np.arange(len(ticks))
    y = np.zeros(len(ticks), dtype=int)
    for i in range(len(ticks)):
        y[i] = counts[ticks[i]]

    rc = paper_rcparams()
    rc.update({
        "font.size": 9.9,
        "axes.labelsize": 9.9,
        "xtick.labelsize": 8.9,
        "ytick.labelsize": 8.9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "hatch.linewidth": 0.6,
    })
    with plt.rc_context(rc=rc):
        fig, ax = plt.subplots(figsize=(3.375, 2.7))
        for i in range(len(ticks)):
            _, color, hatch = bins[ticks[i]]
            ax.bar(x[i], y[i], width=0.55, color=color, edgecolor="0.25", linewidth=0.7, hatch=hatch, zorder=3)
            ax.text(x[i], y[i] + y.max() * 0.03, str(int(y[i])), ha="center", va="bottom", fontsize=8.9, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlabel(r"Mach number $\mathcal{M}_s$")
        ax.set_ylabel("Number of galaxies")
        ax.set_ylim(0, y.max() * 1.22)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.text(0.98, 0.95, rf"$N={int(y.sum())}$", transform=ax.transAxes, ha="right", va="top", fontsize=8.9)
        fig.tight_layout()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, pad_inches=SAVEFIG_PAD_INCHES)
        plt.close(fig)
    return outpath
