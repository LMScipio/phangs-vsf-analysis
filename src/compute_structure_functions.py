"""Computation of the velocity increments and the structure functions S_p(ell) of orders p = 1 to 6.
Subsequently, the fit range of the structure functions is determined.
"""

import numpy as np
from scipy import stats
from src.helpers import load, processed_dir, save

ORDERS_P = (1, 2, 3, 4, 5, 6)
MAX_LAG_KPC = 3.0 # largest lag in kpc (the fit ranges will be below this range, thus this avoids needless computation)
MIN_PAIRS_PER_LAG = 1000 # the computation stops at the first lag with fewer pixel pairs than this
FIT_RANGE_ORDERS = (1, 2, 3) # orders that are used to determine the fit range
R2_THRESHOLD = 0.99 # the log-log power law for orders 1 to 3 should have R^2 >= 0.99
MIN_LAGS = 5 # minimum number of lags in a fit range

def structure_functions(dv, sigma):
    """Structure function definition, with the noise bias subtracted from S_2."""
    sp = []
    for p in ORDERS_P:
        sp.append(np.mean(np.abs(dv) ** p)) # S_p = <|dv|^p>, averaged over all pairs at this lag
    sp[1] -= np.mean(sigma ** 2) # noise bias correction of S_2
    return sp

def compute_structure_functions(galaxy):
    """Structure functions of all pixel pairs at each lag, and the fit range."""
    maps = load(processed_dir(galaxy) / "residual_map.pkl")
    residual, error = maps["residual"], maps["error"]
    pc_per_pix, beam_fwhm_pc = maps["pc_per_pix"], maps["beam_fwhm_pc"]

    max_lag = int(np.floor(MAX_LAG_KPC * 1000.0 / pc_per_pix - 0.5))
    ny, nx = residual.shape
    has_velocity = np.isfinite(residual)
    residual_pad = np.pad(residual, max_lag, constant_values=np.nan)
    error_pad = np.pad(error, max_lag, constant_values=np.nan)
    lags, sp = [], []
    for ell in range(1, max_lag + 1):
        # all pixel pairs whose separation rounds to ell pixels
        # note that (dx,dy) gives the same pairs as (-dx,-dy), so we don't count them twice
        dv, sigma = [], []
        for dx in range(0, ell + 1):
            for dy in range(-ell, ell + 1):
                if dx == 0 and dy <= 0: # same pair as (0, -dy)
                    continue
                if round((dx * dx + dy * dy) ** 0.5) != ell: # falls outside of this lag
                    continue
                y, x = max_lag + dy, max_lag + dx # slightly shift map due to padding
                # velocity and error at r + (dx, dy) for every pixel r
                v1, e1 = residual_pad[y:y + ny, x:x + nx], error_pad[y:y + ny, x:x + nx]
                ok = has_velocity & np.isfinite(v1) # good if both pixels of the pair are inside the mask
                dv.append(v1[ok] - residual[ok]) # velocity increments dv = v(r + ell) - v(r)
                sigma.append(np.sqrt(error[ok] ** 2 + e1[ok] ** 2)) # sigma_pair = sqrt(sigma_1^2 + sigma_2^2)
        dv, sigma = np.concatenate(dv), np.concatenate(sigma)
        if len(dv) < MIN_PAIRS_PER_LAG:
            break
        lags.append(ell)
        sp.append(structure_functions(dv, sigma))
    lags_px = np.array(lags)
    sp = np.array(sp).T
    lags_pc = lags_px * pc_per_pix

    # fit range starts at twice the beam FWHM and adds lags one by one while S_1, S_2 and S_3 still have R^2 >= 0.99
    fit_mask = np.ones(len(lags_pc), dtype=bool)
    for p in FIT_RANGE_ORDERS:
        candidates = np.where(lags_pc >= 2.0 * beam_fwhm_pc)[0] # lags >= 2x beam FWHM
        in_range = np.zeros(len(lags_pc), dtype=bool)
        # start with the first range of 5 lags and add one lag at a time
        for end in range(MIN_LAGS, len(candidates) + 1):
            r_squared = stats.linregress(np.log(lags_pc[candidates[:end]]),
                                         np.log(sp[p - 1, candidates[:end]])).rvalue ** 2
            if r_squared < R2_THRESHOLD: # R^2 of ln S_p against ln ell falls below 0.99
                break
            in_range[candidates[:end]] = True
        fit_mask &= in_range

    save(processed_dir(galaxy) / "structure_functions.npz", {
        "lags_px": lags_px, "lags_pc": lags_pc, "sp": sp, "fit_mask": fit_mask,
        "pc_per_pix": pc_per_pix, "beam_fwhm_pc": beam_fwhm_pc,
    })
