"""First to third order scaling exponents of the 100 molecular clouds of Ma et al. (2025),
ApJ 979, 65, Table A1, plotted against their consecutive exponent, with a least squares fit.
Only the clouds with a significant power law (type S) are used.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.figures.plotting_settings import SAVEFIG_PAD_INCHES, add_subplot_label, column_width_inches, paper_rcparams
from src.helpers import GMC_TABLE_PATH

def load_gmc_table():
    """zeta_1, zeta_2 and zeta_3 of the clouds with a significant power law (type S)."""
    table = np.loadtxt(GMC_TABLE_PATH, dtype=str, delimiter="\t", encoding="utf-8")
    zeta = table[table[:, 15] == "S", 12:15].astype(float)
    return {"zeta_1": zeta[:, 0], "zeta_2": zeta[:, 1], "zeta_3": zeta[:, 2]}

def plot_consecutive_pairs(data, outpath):
    rc = paper_rcparams()
    rc.update({
        "font.size": 14.7,
        "axes.labelsize": 14.7,
        "xtick.labelsize": 13.1,
        "ytick.labelsize": 13.1,
        "legend.fontsize": 14.7,
    })
    with plt.rc_context(rc=rc):
        w = column_width_inches()
        fig, axes = plt.subplots(1, 2, figsize=(2.0 * w * 0.95, w * 0.62))
        for p, letter in ((1, "a"), (2, "b")):
            q = p + 1
            ax = axes[p - 1]
            x = data[f"zeta_{q}"]
            y = data[f"zeta_{p}"]
            ax.scatter(x, y, s=14, color="#0072B2", alpha=0.75, edgecolors="k", linewidths=0.25, zorder=3)

            m, b = np.polyfit(x, y, 1)
            y_fit = m * x + b
            r2 = 1.0 - np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                x_line, m * x_line + b, color="k", lw=1.4, zorder=4,
                label=rf"$\zeta_{p}={m:.3f}\,\zeta_{q}{b:+.3f}$, $R^2={r2:.3f}$, $N={x.size}$",
            )
            ax.set_xlabel(rf"$\zeta_{q}$")
            ax.set_ylabel(rf"$\zeta_{p}$")
            ax.grid(True, alpha=0.22, ls=":", lw=0.6, zorder=0)
            ax.legend(
                loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.7", handlelength=1.1,
                handletextpad=0.4, labelspacing=0.3, borderpad=0.4,
            )
            add_subplot_label(ax, letter, corner="lower right", stroke=True, fontsize=14.7)

        fig.tight_layout(pad=0.3, w_pad=1.2)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, pad_inches=SAVEFIG_PAD_INCHES)
        plt.close(fig)
    return outpath
