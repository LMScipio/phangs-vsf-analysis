"""Linear interdependence of consecutive scaling exponents zeta_p against zeta_{p+1} for
the selected galaxies, with the free fit, the fit with slope p/(p+1) and the generalized She and
Leveque prediction.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.figures.plotting_settings import SAVEFIG_PAD_INCHES, add_subplot_label, column_width_inches, paper_rcparams

FREE_FIT_LABEL = r"Free fit, slope and intercept fitted"
FIXED_SLOPE_LABEL = r"Slope fixed to $p/(p+1)$, intercept fitted"
PREDICTION_LABEL = r"Generalized She and Leveque prediction"

def plot_zeta_interdependence(rows, fits, outpath):
    """Five panels of zeta_p against zeta_{p+1} with the free fit, the fit with slope p/(p+1), and the
    line predicted by the generalized She and Leveque framework, as fitted by analysis.fit_interdependence."""
    pair_colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
    panel_keys = ["a", "b", "c", "d", "e"]

    with plt.rc_context(rc=paper_rcparams()):
        w = column_width_inches()
        fig, axd = plt.subplot_mosaic([panel_keys[:3], ["d", "e", "legend"]], figsize=(2.0 * w, 4.0 * w / 3.0))

        axis_label_fs, tick_fs, panel_letter_fs = 17.6, 15.8, 17.6
        equation_fs, n_fs, legend_fs = 11.4, 13.4, 12.6

        for p in (1, 2, 3, 4, 5):
            q = p + 1
            idx = p - 1
            ax = axd[panel_keys[idx]]
            col = pair_colors[idx]
            lo_max, hi_max = 1.0 + idx * 0.5, 1.5 + idx * 0.5
            if hi_max <= 2.0:
                tick_step = 0.5
            else:
                tick_step = 1.0

            zp, zq = np.zeros((len(rows), 3)), np.zeros((len(rows), 3))
            for j in range(len(rows)):
                zp[j], zq[j] = rows[j]["zeta"][p], rows[j]["zeta"][q]
            ax.errorbar(
                zq[:, 0], zp[:, 0], xerr=zq[:, 1:].T, yerr=zp[:, 1:].T,
                fmt="none", ecolor=col, elinewidth=0.65, capsize=1.5, alpha=0.38, zorder=2,
            )
            ax.scatter(zq[:, 0], zp[:, 0], s=12, color=col, edgecolors="k", linewidths=0.22, alpha=0.85, zorder=3)

            x_line = np.linspace(0.0, hi_max, 100)
            slope = p / (p + 1)
            f = fits[p]
            line_gamma = ax.plot(x_line, slope * x_line + f["gamma"], color="#332288", lw=1.0,
                                 ls=(0, (4, 1.5)), zorder=3)[0]
            line_prediction = ax.plot(x_line, slope * x_line + f["gamma_pred"], color="#882255", lw=1.0,
                                      ls=(0, (1, 1)), zorder=3)[0]
            line_free = ax.plot(x_line, f["slope"] * x_line + f["intercept"], color="k", lw=1.5, zorder=4)[0]
            entries = [
                (rf"$\zeta_{{{p}}} = {f['slope']:.2f}\,\zeta_{{{q}}} {f['intercept']:+.2f}$", "k"),
                (rf"$\zeta_{{{p}}} = {slope:.2f}\,\zeta_{{{q}}} {f['gamma']:+.2f}$", "#332288"),
                (rf"$\zeta_{{{p}}} = {slope:.2f}\,\zeta_{{{q}}} {f['gamma_pred']:+.2f}$", "#882255"),
            ]

            line_gap = equation_fs * 1.9
            for i in range(len(entries)):
                text, color = entries[i]
                ax.annotate(
                    text, xy=(0.03, 0.97), xycoords="axes fraction",
                    xytext=(0, -i * line_gap), textcoords="offset points",
                    va="top", ha="left", fontsize=equation_fs, zorder=6, color=color,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.82, edgecolor="0.72",
                              linewidth=0.45),
                )
            ax.text(
                0.97, 0.03, rf"$N={len(rows)}$", transform=ax.transAxes, va="bottom", ha="right",
                fontsize=n_fs, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78, edgecolor="0.72", linewidth=0.35),
            )

            ax.set_xlabel(rf"$\zeta_{q}$", fontsize=axis_label_fs, labelpad=4)
            ax.set_ylabel(rf"$\zeta_{p}$", fontsize=axis_label_fs, labelpad=6)
            ax.tick_params(axis="both", labelsize=tick_fs, pad=6)
            ax.grid(True, alpha=0.22, ls=":", lw=0.6)
            ax.set_xlim((0.0, hi_max))
            ax.set_ylim((0.0, lo_max))
            ax.set_xticks(np.arange(0.0, hi_max + 1e-9, tick_step))
            ax.set_yticks(np.arange(tick_step, lo_max + 1e-9, tick_step))
            ax.set_box_aspect(1.0)

        for key in panel_keys:
            add_subplot_label(axd[key], key, corner="lower left", fontsize=panel_letter_fs, stroke=True)

        axd["legend"].axis("off")
        axd["legend"].legend(
            [line_free, line_gamma, line_prediction], [FREE_FIT_LABEL, FIXED_SLOPE_LABEL, PREDICTION_LABEL],
            loc="center", frameon=False, fontsize=legend_fs,
            handlelength=1.6, labelspacing=1.1, labelcolor="linecolor",
        )

        fig.subplots_adjust(left=0.06, right=0.985, top=0.99, bottom=0.08, wspace=0.28, hspace=0.27)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, pad_inches=SAVEFIG_PAD_INCHES)
        plt.close(fig)
    return outpath
