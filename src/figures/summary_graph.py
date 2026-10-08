"""Summary graph: the analysis pipeline illustrated for one galaxy. (a) observed "moment 1" map, (b) residual velocity
map after subtraction of the rotational model and cleaning, (c) structure functions S_p(ell)^(1/p) with the fit
range shaded, (d) ESS regression of log S_p against log S_3, (e) beta-test.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import ODR, Model, RealData
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from src.analysis import beta_test
from src.bootstrap import linear
from src.figures.plotting_settings import SAVEFIG_PAD_INCHES, add_subplot_label, paper_rcparams
from src.helpers import load, load_bootstrap, load_structure_functions, processed_dir
from src.compute_structure_functions import ORDERS_P

EXAMPLE_GALAXY = "ngc3621"
ORDER_COLORS = {1: "#1f4e79", 2: "#4f8f8b", 3: "#d7b46a", 4: "#c66a4a", 5: "#7b5a87", 6: "#2d6e5e"}
CAPTION = (
    r"For each galaxy, in each of 1000 bootstrap iterations the scaling "
    r"exponents $\zeta_p$, $\beta_p$, and cascade ratio $\beta_{\mathrm{SL}}$ are computed."
)

MAP_A = [-0.117, 0.24, 0.1575, 0.56]
MAP_B = [0.087, 0.24, 0.1575, 0.56]
PANEL_C = [0.3124, 0.24, 0.1573, 0.56]
PANEL_D = [0.4971, 0.24, 0.1573, 0.56]
PANEL_E = [0.6949, 0.24, 0.1573, 0.56]
BOX = [0.2692, 0.13, 0.599, 0.78]
BRACE_X = 0.2622

def map_panel(ax, arr):
    """Velocity map cropped to its finite pixels, on a diverging colour scale centred on its median."""
    finite_mask = np.isfinite(arr)
    rows = np.where(finite_mask.any(axis=1))[0]
    cols = np.where(finite_mask.any(axis=0))[0]
    r0, r1 = rows.min() - 3, rows.max() + 3 + 1
    c0, c1 = cols.min() - 3, cols.max() + 3 + 1
    arr = arr[r0:r1, c0:c1]
    finite = arr[np.isfinite(arr)]

    center = np.median(finite)
    vlim = np.percentile(np.abs(finite - center), 96)
    im = ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=center - vlim, vmax=center + vlim, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.4")
        spine.set_linewidth(0.6)
    return im

def sf_panel(ax, galaxy):
    """S_p(ell)^(1/p), which brings all orders to a velocity-like scale, with the per-lag bootstrap
    16th and 84th percentiles as error bars; these exist only around the fit range."""
    data = load_structure_functions(galaxy)
    lags_pc = data["lags_pc"]
    fit_mask = data["fit_mask"]
    err_lo = np.full(data["sp"].shape, np.nan)
    err_hi = np.full(data["sp"].shape, np.nan)
    boot = load_bootstrap(galaxy)
    idx = np.searchsorted(data["lags_px"], boot["lags_px_ext"])
    for p in ORDERS_P:
        err_lo[p - 1, idx] = boot[f"sp_boot_lo_{p}"]
        err_hi[p - 1, idx] = boot[f"sp_boot_hi_{p}"]

    ax.axvspan(lags_pc[fit_mask].min(), lags_pc[fit_mask].max(), color="steelblue", alpha=0.10, zorder=0)

    for p in ORDERS_P:
        sp = data["sp"][p - 1]
        y = sp ** (1.0 / p)
        color = ORDER_COLORS[p]
        if p == 3:
            lw, alpha, ms, z = 1.6, 1.0, 3.0, 4
        else:
            lw, alpha, ms, z = 0.9, 0.75, 2.4, 2
        ax.plot(lags_pc, y, color=color, lw=lw, alpha=alpha, zorder=z)

        has_err = np.isfinite(err_lo[p - 1]) & np.isfinite(err_hi[p - 1])
        y_lo = np.where(has_err, sp - err_lo[p - 1], sp) ** (1.0 / p)
        y_hi = np.where(has_err, sp + err_hi[p - 1], sp) ** (1.0 / p)
        ax.errorbar(
            lags_pc, y, yerr=[y - y_lo, y_hi - y], fmt="o",
            ms=ms, mfc=color, mec="none", ecolor=color,
            elinewidth=0.6, capsize=1.2, capthick=0.6,
            alpha=alpha, zorder=z + 1,
        )
        if p == 1 or p == 6:
            if p == 6:
                offset = 9
            else:
                offset = -11
            ax.annotate(f"$S_{p}$", xy=(lags_pc[-1] * 0.94, y[-1]), xycoords="data",
                        xytext=(0, offset), textcoords="offset points",
                        color=color, ha="right", va="center", zorder=5, clip_on=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$ [pc]")
    ax.set_ylabel(r"$S_p(\ell)^{1/p}$ [km/s]")
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi * 1.18)

def ess_panel(ax, galaxy):
    """log S_p against log S_3 within the fit range, one line per order, fitted with the orthogonal
    distance regression of the bootstrap, applied to the structure functions of all pairs."""
    data = load_structure_functions(galaxy)
    fit_mask = data["fit_mask"]
    s3 = data["sp"][2]
    boot = load_bootstrap(galaxy)
    in_fit = boot["fit_mask_ext"]
    sigma3 = (boot["sp_boot_lo_3"][in_fit] + boot["sp_boot_hi_3"][in_fit]) / (2.0 * s3[fit_mask])
    for p in (1, 2, 4, 5, 6):
        sp = data["sp"][p - 1]
        sigma = (boot[f"sp_boot_lo_{p}"][in_fit] + boot[f"sp_boot_hi_{p}"][in_fit]) / (2.0 * sp[fit_mask])
        x, y = np.log10(s3[fit_mask]), np.log10(sp[fit_mask])
        odr = ODR(RealData(x, y, sx=sigma3 / np.log(10.0), sy=sigma / np.log(10.0)),
                  Model(linear), beta0=[1.0, 0.0])
        slope, intercept = odr.run().beta
        x_line = np.array([x.min(), x.max()])
        color = ORDER_COLORS[p]
        ax.scatter(x, y, s=8, color=color, alpha=0.55, edgecolors="none", zorder=1)
        ax.plot(x_line, slope * x_line + intercept, color=color, lw=1.1, alpha=0.9, zorder=2)
        if p == 1 or p == 6:
            if p == 1:
                offset = 7
            else:
                offset = -7
            x_lab = x_line[-1] - 0.02 * (x_line[-1] - x_line[0])
            ax.annotate(f"$p{{=}}{p}$", xy=(x_lab, slope * x_lab + intercept), xycoords="data",
                        xytext=(0, offset), textcoords="offset points",
                        color=color, ha="right", va="center", zorder=3, clip_on=True)
    ax.set_xlabel(r"$\log S_3$")
    ax.set_ylabel(r"$\log S_p$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

def beta_test_panel(ax, galaxy):
    result = beta_test(galaxy)
    x_line = np.array([result["x"].min(), result["x"].max()])
    ax.plot(x_line, result["beta"] * x_line + result["intercept"], "-", color="#B03A2E", lw=1.3, zorder=3)
    ax.scatter(result["x"], result["y"], s=13, marker="s", color="#1B2A4A", edgecolors="none", alpha=0.85, zorder=4)
    ax.set_xlabel(r"$\ln(F_p/F_1)$")
    ax.set_ylabel(r"$\ln(F_{p+1}/F_2)$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

def rotation_subtraction_arrow(fig, ax_left, ax_right, label):
    p_left, p_right = ax_left.get_position(), ax_right.get_position()
    y = p_left.y0 + p_left.height * 0.5
    x_start, x_end = p_left.x0 + p_left.width, p_right.x0
    fig.add_artist(FancyArrowPatch(
        (x_start, y), (x_end, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
        color="0.28", zorder=25, shrinkA=2, shrinkB=2, clip_on=False,
    ))
    fig.text((x_start + x_end) / 2, y + 0.010, label, ha="center", va="bottom", fontsize=13.0, color="black", zorder=25)

def curly_brace(fig, x, y0, y1, width):
    """Curly brace spanning [y0, y1] at x from cubic Bezier curves, pointing left for a negative width."""
    h = y1 - y0
    y_mid = (y0 + y1) / 2.0
    verts = [
        (x, y0),
        (x + width * 0.9, y0 + h * 0.02), (x + width, y0 + h * 0.18), (x + width, y0 + h * 0.25),
        (x + width, y0 + h * 0.32), (x + width * 0.55, y_mid - h * 0.06), (x + width * 1.35, y_mid),
        (x + width * 0.55, y_mid + h * 0.06), (x + width, y1 - h * 0.32), (x + width, y1 - h * 0.25),
        (x + width, y1 - h * 0.18), (x + width * 0.9, y1 - h * 0.02), (x, y1),
    ]
    fig.add_artist(PathPatch(
        MplPath(verts, [MplPath.MOVETO] + [MplPath.CURVE4] * 12), transform=fig.transFigure, fill=False,
        linewidth=1.1, edgecolor="0.28", zorder=25, clip_on=False,
    ))

def horizontal_colorbar(fig, ax, im, label):
    p = ax.get_position()
    cax = fig.add_axes([p.x0, p.y0 - 0.045 - 0.020, p.width, 0.020])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=16.0)
    cb.ax.tick_params(labelsize=11.0, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("0.5")
    for spine in cax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("0.5")

def make_summary_graph(outpath, galaxy=EXAMPLE_GALAXY):
    moment1 = load(processed_dir(galaxy) / "moment1.pkl")["moment1"]
    residual = load(processed_dir(galaxy) / "residual_map.pkl")["residual"]

    rc = paper_rcparams()
    rc.update({
        "axes.linewidth": 1.1,
        "axes.edgecolor": "0.15",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 2.6,
        "ytick.minor.size": 2.6,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })
    with plt.rc_context(rc=rc):
        fig = plt.figure(figsize=(16.0, 4.5))
        ax_map1, ax_map2 = fig.add_axes(MAP_A), fig.add_axes(MAP_B)
        ax_sf, ax_ess, ax_beta = fig.add_axes(PANEL_C), fig.add_axes(PANEL_D), fig.add_axes(PANEL_E)
        im1 = map_panel(ax_map1, moment1)
        im2 = map_panel(ax_map2, residual)
        ax_map1.set_box_aspect(1)
        ax_map2.set_box_aspect(1)
        sf_panel(ax_sf, galaxy)
        ess_panel(ax_ess, galaxy)
        beta_test_panel(ax_beta, galaxy)

        horizontal_colorbar(fig, ax_map1, im1, "$v$ [km/s]")
        horizontal_colorbar(fig, ax_map2, im2, r"$v_{\rm res}$ [km/s]")
        rotation_subtraction_arrow(fig, ax_map1, ax_map2, label=r"$-V_{\rm model}$")

        x0, y0, w, h = BOX
        curly_brace(fig, BRACE_X, y0, y0 + h, width=-0.010)
        fig.add_artist(Rectangle(
            (x0, y0), w, h, transform=fig.transFigure, fill=False, linestyle="--", linewidth=1.0, edgecolor="0.5",
            zorder=15,
        ))
        fig.text(x0 + w / 2, y0 + h - 0.025, CAPTION, ha="center", va="top", color="black", zorder=16)

        add_subplot_label(ax_map1, "a", corner="upper left", stroke=True, fontsize=11.0)
        add_subplot_label(ax_map2, "b", corner="upper left", stroke=True, fontsize=11.0)
        add_subplot_label(ax_sf, "c", corner="upper left", fontsize=11.0)
        add_subplot_label(ax_ess, "d", corner="upper left", fontsize=11.0)
        add_subplot_label(ax_beta, "e", corner="upper left", fontsize=11.0)

        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, pad_inches=SAVEFIG_PAD_INCHES)
        plt.close(fig)
    return outpath
