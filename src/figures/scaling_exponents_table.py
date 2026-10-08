"""Table of the cascade ratio, singularity exponent, Hausdorff dimension, scaling exponents, Mach number
classification and fit range of each selected galaxy.
"""

import numpy as np

EMPTY = r"$-$"
HEADER = (
    r"Galaxy & $\beta_{\mathrm{SL}}$ & $\alpha$ & $D$ & $\zeta_2$ & $\beta_2$ & $\zeta_3$ "
    r"& $\mathcal{M}_s$ classification & Fit range [pc] \\"
)

def caption(n_galaxies):
    return (
        rf"This table lists the {n_galaxies} selected galaxies, which are the galaxies with a reliable kinematic fit "
        r"and a smooth analytic fit to the rotation curve in Lang et al. (2020) and an inclination "
        r"$i + \sigma_i \leq 70^\circ$. "
        r"It contains the cascade ratio $\beta_{\mathrm{SL}}$, singularity exponent $\alpha$, Hausdorff dimension $D$, "
        r"and the scaling exponents $\zeta_2$ and $\zeta_3$, and ESS exponents $\beta_2$ from the spatial bootstrap, "
        r"along with their errors as the 16th and 84th percentiles of the bootstrap distribution. The row labelled "
        r"\textit{Median} gives the median across the galaxy population of each of the exponents, and the "
        r"corresponding errors report the distance from that median to the 16th and 84th percentiles. The "
        r"$\mathcal{M}_s$ classification column gives each galaxy's classification against the six Pan and "
        r"Scannapieco (2011) Mach number curves. A single Mach number is provided where one curve fits well, else the "
        r"range it falls between is reported. Lastly, the table includes the range in which the scaling exponents are "
        r"fitted for each galaxy."
    )

def galaxy_name(galaxy):
    for prefix in ("ngc", "ic"):
        if galaxy.startswith(prefix):
            return f"{prefix.upper()}\\,{galaxy[len(prefix):]}"

def asymmetric(med, lo, hi):
    """$med^{+hi}_{-lo}$ rounded to two decimals."""
    return f"${med:.2f}^{{+{hi:.2f}}}_{{-{lo:.2f}}}$"

def mach_cell(result):
    lo, hi = result["range_lo"], result["range_hi"]
    if lo == hi:
        return rf"${lo}$"
    if lo is None:
        return rf"$\leq{hi}$"
    if hi is None:
        return rf"$\geq{lo}$"
    return rf"${lo}\text{{--}}{hi}$"

def fit_range_cell(r_min, r_max):
    return rf"${r_min:.0f}\text{{--}}{r_max:.0f}$"

def scriptsize(tex):
    return r"{\scriptsize " + tex + "}"

def write_scaling_exponents_table(rows, mach_results, outpath):
    """rows are the galaxy_results rows of the selected galaxies, mach_results their Mach number classifications."""
    mach_by_galaxy = {}
    for r in mach_results:
        mach_by_galaxy[r["galaxy"]] = r
    lines = [
        r"% Scaling exponents per selected galaxy. Requires \usepackage{booktabs} and \usepackage{longtable}.",
        r"\begingroup",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\LTcapwidth}{\textwidth}",
        r"\setlength{\tabcolsep}{1.9pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable*}{@{\extracolsep{\fill}}l c c c c c c c c@{}}",
        r"\toprule", HEADER, r"\midrule", r"\endfirsthead",
        r"\toprule", HEADER, r"\midrule", r"\endhead",
        r"\midrule", r"\endfoot",
        r"\midrule", r"\endlastfoot",
    ]

    medians = []
    for row in rows:
        values = [row["beta_SL"], row["alpha"], row["D"], row["zeta"][2], row["beta"][2], row["zeta"][3]]
        cells = [galaxy_name(row["galaxy"])]
        row_medians = []
        for v in values:
            cells.append(asymmetric(v[0], v[1], v[2]))
            row_medians.append(v[0])
        cells.append(mach_cell(mach_by_galaxy[row["galaxy"]]))
        cells.append(fit_range_cell(row["r_min_pc"], row["r_max_pc"]))
        lines.append(" & ".join(cells) + r" \\")
        medians.append(row_medians)
    medians = np.array(medians)

    median_cells = []
    for k in range(6):
        v = medians[:, k]
        p16, p84 = np.percentile(v, [16, 84])
        m = np.median(v)
        median_cells.append(scriptsize(asymmetric(m, m - p16, p84 - m)))
    median_cells.append(scriptsize(EMPTY))
    r_min, r_max = [], []
    for r in rows:
        r_min.append(r["r_min_pc"])
        r_max.append(r["r_max_pc"])
    median_cells.append(scriptsize(fit_range_cell(np.median(r_min), np.median(r_max))))

    lines += [
        r"\midrule",
        r"\midrule",
        r"{\scriptsize\textit{Median}} & " + " & ".join(median_cells) + r" \\",
        r"\bottomrule",
        r"\caption{" + caption(len(rows)) + "}",
        r"\label{tab:scaling-exponents}",
        r"\end{longtable*}",
        r"\endgroup",
    ]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines), encoding="utf-8")
    return outpath
