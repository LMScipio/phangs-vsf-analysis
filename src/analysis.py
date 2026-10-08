"""Selection of the galaxies, the turbulence model predictions, the Mach number classification and the
population fits of consecutive scaling exponents.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from src.helpers import (ALL_GALAXIES_PATH, SELECTED_GALAXIES_PATH, TAB2_TABLE_PATH, load_bootstrap,
    load_structure_functions, print_summary, read_galaxy_list, write_galaxy_list)
from src.compute_structure_functions import ORDERS_P
from src.bootstrap import beta_test_fit

# Selection option: make sure high inclination galaxies are excluded.
INCLINATION_MAX = 70.0

# Mach number M_s, zeta_3, beta and D of each simulation of Pan and Scannapieco (2011)
PAN_SCANNAPIECO_TABLE = (
    (0.9, 0.98, 0.88, 1.0),
    (1.4, 1.07, 0.85, 1.0),
    (2.1, 1.18, 0.77, 1.2),
    (3.0, 1.22, 0.63, 1.7),
    (4.6, 1.22, 0.54, 1.8),
    (6.1, 1.22, 0.52, 1.8),
)

def median_and_errors(values):
    """Median over the bootstrap iterations and its distances to the 16th and 84th percentiles. Iterations in which
    alpha and D are undefined (zeta_3 <= 0) are skipped."""
    median = np.nanmedian(values)
    # median, lower and upper error
    return median, median - np.nanpercentile(values, 16), np.nanpercentile(values, 84) - median

def beta_test(galaxy):
    """beta-test on S_p of all pixel pairs in the fit range: the points x, y, beta_SL, the intercept and R^2."""
    sf_results = load_structure_functions(galaxy)
    beta, intercept, x, y = beta_test_fit(sf_results["sp"], np.where(sf_results["fit_mask"])[0])
    correlation = np.corrcoef(x, y)[0, 1] # Pearson correlation r of the points, R^2 = r^2
    return {"x": x, "y": y, "beta": beta, "intercept": intercept, "r2": correlation ** 2}

def galaxy_results():
    """Fit range and the medians and percentile distances of zeta_p, beta_p, beta_SL, alpha and D of every galaxy
    with a valid fit range."""
    rows = []
    for galaxy in read_galaxy_list(ALL_GALAXIES_PATH):
        sf_results = load_structure_functions(galaxy)
        if not sf_results["fit_mask"].any(): # no valid fit range, so no bootstrap either
            continue
        bootstrap_results = load_bootstrap(galaxy)
        fit_range_lags = sf_results["lags_pc"][sf_results["fit_mask"]]
        row = {"galaxy": galaxy, "r_min_pc": fit_range_lags.min(), "r_max_pc": fit_range_lags.max(),
               "zeta": {}, "beta": {}}
        for p in ORDERS_P:
            row["zeta"][p] = median_and_errors(bootstrap_results[f"zeta_{p}"])
            row["beta"][p] = median_and_errors(bootstrap_results[f"beta_{p}"])
        for quantity in ("beta_SL", "alpha", "D"):
            row[quantity] = median_and_errors(bootstrap_results[quantity])
        row["beta_test_r2"] = beta_test(galaxy)["r2"]
        rows.append(row)
    return rows

def select_galaxies(rows):
    """Select galaxies with a reliable kinematic fit and a smooth analytic fit to the rotation curve in Lang et al.
    (2020), and inclination i + sigma_i <= 70 degrees."""
    tab2 = np.loadtxt(TAB2_TABLE_PATH, dtype=str, encoding="utf-8")
    selected = []
    for row in rows:
        lang_row = tab2[tab2[:, 0] == row["galaxy"]][0] # row of this galaxy in Table 2
        inclination = float(lang_row[3]) + float(lang_row[4]) # i + sigma_i
        # columns reliable and smooth_fit
        if lang_row[7] == "yes" and lang_row[8] == "yes" and inclination <= INCLINATION_MAX:
            selected.append(row)
    return selected

def k41(p):
    return p / 3.0

def she_leveque(p):
    return p / 9.0 + 2.0 * (1.0 - (2.0 / 3.0) ** (p / 3.0))

def kolmogorov_burgers(p):
    return p / 9.0 + 1.0 - (1.0 / 3.0) ** (p / 3.0)

def ess_beta(zeta_model, p):
    return zeta_model(p) / zeta_model(3.0)

def predicted_intercept(p, beta_sl, C):
    """gamma_p of the generalized She and Leveque framework."""
    # gamma_p = C[(1 - beta_SL^p) - p/(p+1) (1 - beta_SL^(p+1))]
    return C * ((1.0 - beta_sl ** p) - (p / (p + 1)) * (1.0 - beta_sl ** (p + 1)))

def pan_scannapieco_beta(p, mach):
    """beta_p of the Pan and Scannapieco (2011) simulation for each Mach number."""
    for simulation in PAN_SCANNAPIECO_TABLE:
        if simulation[0] == mach:
            _, zeta3, beta, d = simulation
    C = (3.0 - d) / zeta3
    gamma = (1.0 - C * (1.0 - beta ** 3)) / 3.0
    return gamma * p + C * (1.0 - beta ** p)

def classify_mach_number(galaxy):
    """Classify the beta_p vs p curves of the galaxies by the Pan and Scannapieco (2011) predictions."""
    bootstrap_results = load_bootstrap(galaxy)
    orders = (1, 2, 4, 5, 6)
    beta_draws = []
    for p in orders:
        beta_draws.append(bootstrap_results[f"beta_{p}"])
    beta_16, beta_median, beta_84 = np.percentile(np.column_stack(beta_draws), [16, 50, 84], axis=0)
    sigma = (beta_84 - beta_16) / 2.0 # average of the lower and upper error of beta_p
    mach_labels, curves, p_values = [], [], []
    for simulation in PAN_SCANNAPIECO_TABLE:
        curve = pan_scannapieco_beta(np.array(orders), simulation[0])
        mach_labels.append(f"{simulation[0]:.1f}")
        curves.append(curve)
        p_values.append(stats.chi2.sf(np.sum(((beta_median - curve) / sigma) ** 2), 5)) # p-value of the weighted chi^2
    curves = np.array(curves)
    # the two curves with the highest p-value
    best, second = sorted(range(len(curves)), key=lambda i: p_values[i], reverse=True)[:2]

    if p_values[best] >= 0.05: # the best curve is acceptable
        return {"galaxy": galaxy, "range_lo": mach_labels[best], "range_hi": mach_labels[best]}
    lower, upper = np.minimum(curves[best], curves[second]), np.maximum(curves[best], curves[second])
    if np.all((beta_median >= lower) & (beta_median <= upper)): # the data lies between two curves
        range_lo, range_hi = sorted([mach_labels[best], mach_labels[second]], key=float)
        return {"galaxy": galaxy, "range_lo": range_lo, "range_hi": range_hi}

    lowest, highest = len(mach_labels), -1 # start from an empty range
    for order_index in range(len(orders)):
        sorted_curves = np.argsort(curves[:, order_index]) # sorting by beta_p at this order
        # where the measured beta_p falls among them
        position = np.searchsorted(curves[sorted_curves, order_index], beta_median[order_index])
        if 0 < position < len(sorted_curves): # between two curves
            bracket = sorted((sorted_curves[position - 1], sorted_curves[position]))
        else: # below the lowest or above the highest curve
            if position == 0:
                outermost = sorted_curves[0]
            else:
                outermost = sorted_curves[-1]
            if outermost == 0: # open end below M = 0.9
                bracket = (-1, 0)
            else: # open end above M = 6.1
                bracket = (5, 6)
        lowest, highest = min(lowest, bracket[0]), max(highest, bracket[1]) # widen the range to contain this order
    range_lo, range_hi = None, None
    if lowest != -1:
        range_lo = mach_labels[lowest]
    if highest != len(mach_labels):
        range_hi = mach_labels[highest]
    return {"galaxy": galaxy, "range_lo": range_lo, "range_hi": range_hi}

def population_parameters(galaxies):
    """Medians over the galaxies of beta_SL and alpha, and the co-dimension C = -alpha / (1 - beta_SL^3)."""
    beta_sl_medians, alpha_medians = [], []
    for galaxy in galaxies:
        bootstrap_results = load_bootstrap(galaxy)
        beta_sl_medians.append(median_and_errors(bootstrap_results["beta_SL"])[0])
        alpha_medians.append(median_and_errors(bootstrap_results["alpha"])[0])
    beta_sl, alpha = np.median(beta_sl_medians), np.median(alpha_medians)
    return beta_sl, -alpha / (1.0 - beta_sl ** 3)

def negative_log_likelihood(ln_V, projection, projection_variance):
    """-ln L for an intrinsic scatter V, with b_perp at its best value for this V."""
    weights = 1.0 / (projection_variance + np.exp(ln_V)) # 1 / (Sigma_i^2 + V)
    b_perp = np.sum(weights * projection) / np.sum(weights) # best perpendicular offset for this V
    return 0.5 * np.sum(weights * (projection - b_perp) ** 2 - np.log(weights)) # -ln L up to a constant

def fit_at_angle(theta, Z, S):
    """b_perp and -ln L of the maximum likelihood fit with orthogonal displacements for a line at angle theta. For a
    given intrinsic scatter V the best b_perp is the weighted mean of the projections, so only V is searched."""
    v_hat = np.array([-np.sin(theta), np.cos(theta)]) # unit vector perpendicular to the line
    projection = np.dot(Z, v_hat) # projection of each galaxy's (zeta_{p+1}, zeta_p) onto it
    projection_variance = np.zeros(len(S)) # Sigma_i^2 = v^T S_i v
    for j in range(2):
        for k in range(2):
            projection_variance += v_hat[j] * S[:, j, k] * v_hat[k]
    # searched in ln V so that V stays positive
    ln_V = minimize_scalar(negative_log_likelihood, bounds=(-40.0, 10.0), args=(projection, projection_variance),
                           method="bounded", options={"xatol": 1e-10}).x
    weights = 1.0 / (projection_variance + np.exp(ln_V))
    b_perp = np.sum(weights * projection) / np.sum(weights)
    return b_perp, negative_log_likelihood(ln_V, projection, projection_variance)

def negative_log_likelihood_at_angle(theta, Z, S):
    """-ln L of the best line at angle theta."""
    return fit_at_angle(theta, Z, S)[1]

def fit_line(Z, S, slope=None):
    """Slope m and intercept b of zeta_p = m zeta_{p+1} + b of maximum likelihood (Hogg, Bovy and Lang 2010),
    from a grid of angles refined by a bounded search. With a given slope only b and V are fitted."""
    if slope is None: # free fit of slope and intercept
        angles = np.linspace(0.0, np.arctan(2.0), 41) # grid of line angles, slopes from 0 to 2
        neg_log_likelihoods = []
        for angle in angles:
            neg_log_likelihoods.append(negative_log_likelihood_at_angle(angle, Z, S))
        best = int(np.argmin(neg_log_likelihoods))
        # refine between the neighbouring grid angles
        theta = minimize_scalar(negative_log_likelihood_at_angle, bounds=(angles[best - 1], angles[best + 1]),
                                args=(Z, S), method="bounded", options={"xatol": 1e-10}).x
    else: # slope fixed, e.g. to p/(p+1)
        theta = np.arctan(slope)
    b_perp = fit_at_angle(theta, Z, S)[0]
    return np.tan(theta), b_perp / np.cos(theta) # m = tan(theta), b = b_perp / cos(theta)

def fit_interdependence(galaxies, prediction):
    """For p = 1 to 5 the free fit of zeta_p against zeta_{p+1}, the intercept gamma_p of the fit with slope p/(p+1)
    and the predicted gamma_p."""
    bootstrap_results = []
    for galaxy in galaxies:
        bootstrap_results.append(load_bootstrap(galaxy))
    fits = {}
    for p in range(1, 6):
        # median (zeta_{p+1}, zeta_p) of each galaxy and their 2 x 2 covariance over the bootstrap iterations
        Z, S = [], []
        for results in bootstrap_results:
            Z.append([np.median(results[f"zeta_{p + 1}"]), np.median(results[f"zeta_{p}"])])
            S.append(np.cov(results[f"zeta_{p + 1}"], results[f"zeta_{p}"]))
        Z, S = np.array(Z), np.array(S)
        slope, intercept = fit_line(Z, S)
        gamma = fit_line(Z, S, slope=p / (p + 1))[1] # intercept with the slope fixed to p/(p+1)
        fits[p] = {"slope": slope, "intercept": intercept, "gamma": gamma,
                   "gamma_pred": predicted_intercept(p, prediction[0], prediction[1])}
    return fits

def run_analysis():
    """Select and classify the galaxies and fit the interdependence of consecutive scaling exponents."""
    rows = galaxy_results()
    selected_rows = select_galaxies(rows)
    all_galaxies, selected_galaxies, mach_results = [], [], []
    for row in rows:
        all_galaxies.append(row["galaxy"])
    for row in selected_rows:
        selected_galaxies.append(row["galaxy"])
        mach_results.append(classify_mach_number(row["galaxy"]))
    write_galaxy_list(SELECTED_GALAXIES_PATH, sorted(selected_galaxies))

    prediction = population_parameters(selected_galaxies)
    fits = fit_interdependence(selected_galaxies, prediction)
    fits_all = fit_interdependence(all_galaxies, prediction)
    print_summary(rows, selected_rows, fits)
    return {
        "rows": rows,
        "selected_rows": selected_rows,
        "prediction": prediction,
        "fits": fits,
        "fits_all": fits_all,
        "mach_results": mach_results,
    }
