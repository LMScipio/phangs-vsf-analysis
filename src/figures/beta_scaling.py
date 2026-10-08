"""The ESS exponents beta_p against order p, with the predictions of K41, She and Leveque,
Kolmogorov-Burgers and Pan and Scannapieco (2011).
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import ess_beta, k41, kolmogorov_burgers, pan_scannapieco_beta, she_leveque
from src.figures.plotting_settings import SAVEFIG_PAD_INCHES, column_width_inches, paper_rcparams
from src.compute_structure_functions import ORDERS_P

def plot_beta_scaling(rows, outpath):
    """beta_p of every galaxy, their median with 16th to 84th percentile range, and the model predictions."""
    p_arr = np.array(ORDERS_P)
    p_fine = np.linspace(0.5, 6.5, 200)
    beta_all = np.zeros((len(rows), len(ORDERS_P)))
    for j in range(len(rows)):
        for p in ORDERS_P:
            beta_all[j, p - 1] = rows[j]["beta"][p][0]
    data_color = "0.45"
    thin = 0.72

    with plt.rc_context(rc=paper_rcparams()):
        w = column_width_inches()
        fig, ax = plt.subplots(figsize=(w, w * 0.71))
        ax.set_box_aspect(0.8)
        label_fs, tick_fs, legend_fs = 12.9, 11.5, 6.6

        med = np.median(beta_all, axis=0)
        lo = np.percentile(beta_all, 16, axis=0)
        hi = np.percentile(beta_all, 84, axis=0)

        label = "Individual galaxies"
        for curve in beta_all:
            ax.plot(p_arr, curve, ls="none", marker="o", ms=2.5, color=data_color, alpha=0.45, zorder=2, label=label)
            label = None

        ax.plot(p_fine, k41(p_fine), color="#333333", ls=":", lw=thin, alpha=0.88, label="K41", zorder=4)
        ax.plot(
            p_fine, ess_beta(she_leveque, p_fine), color="#332288", ls="-", lw=thin, alpha=0.88, label="She-Leveque",
            zorder=4,
        )
        ax.plot(
            p_fine, ess_beta(kolmogorov_burgers, p_fine), color="#117733", ls="--", lw=thin, alpha=0.88,
            label="Kolmogorov-Burgers", zorder=4,
        )
        for M, color, ls in ((1.4, "#88CCEE", (0, (4, 2))), (6.1, "#CC6677", "-.")):
            ax.plot(
                p_fine, pan_scannapieco_beta(p_fine, M), color=color, lw=thin, alpha=0.88, ls=ls,
                label=rf"Pan & Scannapieco, $\mathcal{{M}}_\mathrm{{s}} = {M}$", zorder=4,
            )

        ax.plot(
            p_arr, med, "-", color="k", lw=1.15, zorder=6, label="Median",
            marker="o", ms=3.4, markerfacecolor="white", markeredgecolor="k", markeredgewidth=0.95,
        )
        y_mid = 0.5 * (lo + hi)
        ax.errorbar(
            p_arr, y_mid, yerr=[y_mid - lo, hi - y_mid], fmt="none", ecolor="k",
            elinewidth=1.15, capsize=3.0, capthick=1.15, zorder=5, label="16th-84th percentile",
        )
        ax.axhline(1.0, color="#888888", lw=0.45, ls=":", alpha=0.5, zorder=0)

        ax.set_xlabel("Order  $p$", fontsize=label_fs, labelpad=8)
        ax.set_ylabel(r"$\beta_p$", fontsize=label_fs, labelpad=10)
        ax.set_xlim(0.8, 6.3)
        ax.set_ylim(0.2, 2.2)
        ax.set_yticks(np.arange(0.2, 2.2 + 1e-9, 0.2))
        ax.set_xticks(ORDERS_P)
        ax.tick_params(labelsize=tick_fs, pad=4)
        ax.grid(True, alpha=0.12, lw=0.45, zorder=0)
        leg = ax.legend(
            fontsize=legend_fs, loc="lower right", bbox_to_anchor=(0.99, 0.02), frameon=True, framealpha=0.95,
            fancybox=True, shadow=False, edgecolor="0.65", facecolor="white", handlelength=1.6,
            handletextpad=0.45, columnspacing=0.55, ncol=2,
        )
        leg.get_frame().set_linewidth(0.65)
        fig.tight_layout()

        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, pad_inches=SAVEFIG_PAD_INCHES)
        plt.close(fig)
    return outpath
