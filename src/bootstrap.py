"""The spatial bootstrap of the scaling exponents zeta_p, the ESS exponents beta_p and the cascade ratio beta_SL,
and the singularity exponent alpha, co-dimension C and Hausdorff dimension D derived from them.
"""

import numpy as np
from scipy.odr import ODR, Model, RealData
from scipy.spatial import cKDTree
from src.helpers import bootstrap_path, load, load_structure_functions, processed_dir, save
from src.compute_structure_functions import ORDERS_P, structure_functions

N_ITERATIONS = 1000 # number of bootstrap iterations
BOOTSTRAP_SEED = 13294 # seed for reproducibility

def beta_test_fit(sp, lag_indices):
    """Compute the points (ln(F_p/F_1), ln(F_{p+1}/F_2)) with F_p = S_{p+1}/S_p for p = 2, 3, 4.
    The slope of these points is the cascade ratio beta_SL."""
    S = sp[:, lag_indices]
    F = S[1:] / S[:-1]
    x, y = [], []
    for p in (2, 3, 4):
        x.append(np.log(F[p - 1] / F[0])) # ln(F_p / F_1)
        y.append(np.log(F[p] / F[1])) # ln(F_{p+1} / F_2)
    x, y = np.concatenate(x), np.concatenate(y)
    mean_x, mean_y = x.mean(), y.mean()
    # simple least squares slope gives the cascade ratio beta_SL
    slope = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
    intercept = mean_y - slope * mean_x
    return slope, intercept, x, y

def linear(parameters, x):
    """Straight line for the ODR fit, which passes the slope and intercept together in one array."""
    slope, intercept = parameters
    return slope * x + intercept

def sampling_blocks(residual, block_size):
    """Split the map into squares of block_size x block_size pixels and provide
    the coordinates of its valid (having a velocity) pixels."""
    pixels_in_block = {}
    for y, x in np.argwhere(np.isfinite(residual)):
        block_position = (y // block_size, x // block_size) # the beam sized block that contains this pixel
        if block_position not in pixels_in_block:
            pixels_in_block[block_position] = []
        pixels_in_block[block_position].append((y, x))
    blocks = []
    for block_position in sorted(pixels_in_block): # fixed order of the blocks for reproducibility
        blocks.append(np.array(pixels_in_block[block_position]))
    return blocks

def block_pairs(blocks, block_size, largest_lag):
    """Pairs of blocks that can give a pixel pair at one of the lags."""
    centres = []
    for pixels in blocks:
        centres.append(pixels.mean(axis=0)) # mean position of the valid pixels of the block
    # finds block pairs close enough to have a pixel pair
    pairs = cKDTree(np.array(centres)).query_pairs(r=largest_lag + 2 * block_size, output_type="ndarray")
    return pairs[:, 0], pairs[:, 1]

def bootstrap_iteration(rng, blocks, first_block, second_block, residual, error, extended_lags, fit_positions):
    """One pixel is drawn from each block, and its velocity is perturbed by Gaussian noise of its measurement
    uncertainty."""
    drawn_pixels = []
    for pixels in blocks:
        drawn_pixels.append(pixels[rng.integers(0, len(pixels))])
    drawn_y, drawn_x = np.array(drawn_pixels).T
    drawn_error = error[drawn_y, drawn_x]
    drawn_velocity = rng.normal(loc=residual[drawn_y, drawn_x], scale=drawn_error)
    dy = drawn_y[second_block] - drawn_y[first_block]
    dx = drawn_x[second_block] - drawn_x[first_block]
    separation = np.rint(np.sqrt(dy ** 2 + dx ** 2)) # separation of each pixel pair rounded to whole pixels
    dv = drawn_velocity[second_block] - drawn_velocity[first_block] # velocity increments of all pixel pairs
    # sigma_pair = sqrt(sigma_1^2 + sigma_2^2)
    sigma = np.sqrt(drawn_error[second_block] ** 2 + drawn_error[first_block] ** 2)

    n_orders, n_lags = len(ORDERS_P), len(extended_lags)
    sp = np.full((n_orders, n_lags), np.nan)
    sp_weighting_error = np.full((n_orders, n_lags), np.nan) # error of S_p is used for the weights of the zeta_p fit
    for lag_index in range(n_lags):
        at_lag = separation == extended_lags[lag_index]
        sp[:, lag_index] = structure_functions(dv[at_lag], sigma[at_lag])
        for p in ORDERS_P:
            # std(|dv|^p) / sqrt(N_pairs)
            sp_weighting_error[p - 1, lag_index] = np.std(np.abs(dv[at_lag]) ** p) / np.sqrt(at_lag.sum())

    # computation of zeta_p
    ln_lags = np.log(extended_lags[fit_positions])
    zeta = np.full(n_orders, np.nan)
    for p in ORDERS_P:
        ln_sp = np.log(sp[p - 1, fit_positions])
        weights = (sp[p - 1, fit_positions] / sp_weighting_error[p - 1, fit_positions]) ** 2
        # weighted means
        mean_ln_lag, mean_ln_sp = np.dot(weights, ln_lags) / weights.sum(), np.dot(weights, ln_sp) / weights.sum()
        covariance = np.dot(weights, (ln_lags - mean_ln_lag) * (ln_sp - mean_ln_sp))
        variance = np.dot(weights, (ln_lags - mean_ln_lag) ** 2)
        zeta[p - 1] = covariance / variance # weighted least squares slope
    beta_sl = beta_test_fit(sp, fit_positions)[0] # cascade ratio beta_SL from the beta-test
    return sp, zeta, beta_sl

def ess_and_singularity_exponents(sp_all_iterations, zeta, beta_sl, fit_positions):
    """Error bars of S_p, and beta_p, alpha and C of all iterations."""
    n_orders, n_lags = len(ORDERS_P), sp_all_iterations.shape[2]

    # error bars of S_p
    sp_error_low = np.full((n_orders, n_lags), np.nan) # median S_p - its 16th percentile
    sp_error_high = np.full((n_orders, n_lags), np.nan) # 84th percentile - the median S_p
    sigma_ln_sp = np.full((n_orders, n_lags), np.nan)
    for p in ORDERS_P:
        for lag_index in range(n_lags):
            sp_values = sp_all_iterations[:, p - 1, lag_index] # S_p at this lag in all iterations
            median = np.median(sp_values)
            sp_error_low[p - 1, lag_index] = median - np.percentile(sp_values, 16)
            sp_error_high[p - 1, lag_index] = np.percentile(sp_values, 84) - median
            sigma_ln_sp[p - 1, lag_index] = np.std(np.log(sp_values), ddof=1)

    # computation of beta_p
    beta = np.full((n_orders, N_ITERATIONS), np.nan)
    beta[2] = 1.0
    for iteration in range(N_ITERATIONS):
        ln_s3 = np.log(sp_all_iterations[iteration, 2, fit_positions])
        for p in (1, 2, 4, 5, 6):
            ln_sp = np.log(sp_all_iterations[iteration, p - 1, fit_positions])
            # uncertainties on both axes
            odr_data = RealData(ln_s3, ln_sp, sx=sigma_ln_sp[2, fit_positions], sy=sigma_ln_sp[p - 1, fit_positions])
            # slope of ln S_p against ln S_3
            beta[p - 1, iteration] = ODR(odr_data, Model(linear), beta0=[1.0, 0.0]).run().beta[0]

    # computation of alpha, using beta_SL of the same iteration
    orders = np.array([1, 2, 4, 5, 6]) # beta_3 is excluded because it is 1 for all cases
    alpha = np.full(N_ITERATIONS, np.nan)
    for iteration in range(N_ITERATIONS):
        if zeta[2, iteration] > 0: # alpha is only defined for a positive zeta_3
            x_p = ((orders / 3.0 - (1.0 - beta_sl[iteration] ** orders) / (1.0 - beta_sl[iteration] ** 3))
                   / zeta[2, iteration])
            # least squares slope
            alpha[iteration] = np.sum(x_p * (beta[orders - 1, iteration] - orders / 3.0)) / np.sum(x_p ** 2)
    C = -alpha / (1.0 - beta_sl ** 3) # co-dimension
    return sp_error_low, sp_error_high, beta, alpha, C

def run_galaxy_bootstrap(galaxy):
    """The spatial bootstrap of one galaxy."""
    sf_results = load_structure_functions(galaxy)
    maps = load(processed_dir(galaxy) / "residual_map.pkl")
    residual, error = maps["residual"], maps["error"]
    lags_px = sf_results["lags_px"]
    block_size = round(sf_results["beam_fwhm_pc"] / sf_results["pc_per_pix"]) # the beam FWHM in whole pixels

    # S_p is also computed one lag either side of the fit range, where that lag exists
    fit_indices = np.where(sf_results["fit_mask"])[0]
    extended_indices = np.arange(fit_indices[0] - 1, min(fit_indices[-1] + 2, len(lags_px)))
    extended_lags = lags_px[extended_indices]
    in_fit_range = np.isin(extended_indices, fit_indices) # which of the extended lags lie in the fit range
    fit_positions = np.where(in_fit_range)[0] # their positions among the extended lags

    blocks = sampling_blocks(residual, block_size)
    first_block, second_block = block_pairs(blocks, block_size, extended_lags.max())

    n_orders, n_lags = len(ORDERS_P), len(extended_lags)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    zeta = np.full((n_orders, N_ITERATIONS), np.nan)
    beta_sl = np.full(N_ITERATIONS, np.nan)
    sp_all_iterations = np.full((N_ITERATIONS, n_orders, n_lags), np.nan)
    for iteration in range(N_ITERATIONS):
        sp_all_iterations[iteration], zeta[:, iteration], beta_sl[iteration] = bootstrap_iteration(
            rng, blocks, first_block, second_block, residual, error, extended_lags, fit_positions)
    sp_error_low, sp_error_high, beta, alpha, C = ess_and_singularity_exponents(
        sp_all_iterations, zeta, beta_sl, fit_positions)

    output = {"beta_SL": beta_sl, "alpha": alpha, "D": 3.0 - C,
              "lags_px_ext": extended_lags, "fit_mask_ext": in_fit_range}
    for p in ORDERS_P:
        output[f"zeta_{p}"], output[f"beta_{p}"] = zeta[p - 1], beta[p - 1]
        output[f"sp_boot_lo_{p}"], output[f"sp_boot_hi_{p}"] = sp_error_low[p - 1], sp_error_high[p - 1]
    save(bootstrap_path(galaxy), output)
