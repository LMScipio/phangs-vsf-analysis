"""Subtract a rotational model from the moment 1 map to obtain the residual velocity map.
The rotational model is V_model = V_sys + V_rot(R) cos(theta) sin(i), which uses the parameters from Lang et al.
(2020) Table 1, Table 2 and Table D1, which are extracted through extract_galaxy_data.py.
Note that these distances are not the same as provided by the newer PHANGS-ALMA data release.
To subtract the proper model, we use the distances used in Lang et al. (2020).
Additonally, the residual map is cleaned as in Chen et al. (2024).  
I.e. pixels with an uncertain velocity and the outliers on either side of the residual velocity distribution are excluded.
"""

import numpy as np
from src.helpers import load, processed_dir, save

def subtract_rotational_model(galaxy):
    """Calculate the residual velocity map by subtracting the rotational model from the moment 1 map, and clean it."""
    data = load(processed_dir(galaxy) / "moment1.pkl")
    mom1 = data["moment1"]
    lang_phi, lang_inc = np.deg2rad(data["lang_phi_deg"]), np.deg2rad(data["lang_inc_deg"])

    ny, nx = mom1.shape
    yy, xx = np.mgrid[:ny, :nx]
    x_c, y_c = data["center_pix"]
    dx = xx - x_c
    dy = yy - y_c
    # coordinate along the kinematic major axis, positive on the receding side
    x_gal = -dx * np.sin(lang_phi) + dy * np.cos(lang_phi)
    # coordinate along the minor axis, divided by cos(i) to deproject it
    y_gal = (-dx * np.cos(lang_phi) - dy * np.sin(lang_phi)) / np.cos(lang_inc)
    # galactocentric radius in the disc plane in kpc (4.848e-3 kpc per arcsec per Mpc)
    R = np.sqrt(x_gal**2 + y_gal**2) * data["pix_scale_arcsec"] * 4.848e-3 * data["lang_distance_mpc"]
    theta = np.arctan2(y_gal, x_gal) # azimuth in the disc plane, measured from the receding major axis

    # rotation velocity at each pixel by linear interpolation of the rotation curve
    # the curve starts at V_rot(0) = 0 and is flat beyond the last point provided in Table D1
    v_rot = np.interp(R, np.concatenate(([0.0], data["lang_rc_r_kpc"])), np.concatenate(([0.0], data["lang_rc_v_kms"])))
    # line of sight velocity of the rotating disc, V_model = V_sys + V_rot(R) cos(theta) sin(i)
    model = data["lang_v_sys_kms"] + v_rot * np.cos(theta) * np.sin(lang_inc)
    residual = mom1 - model
    error = data["moment1_error"].astype(float)

    # cleaning of the residual map, first the uncertain pixels and then the 2% tails on either side
    keep = np.isfinite(residual)
    keep &= error <= 3.0 * np.median(error[keep]) # pixels with a velocity uncertainty above 3 times the median uncertainty are excluded
    low, high = np.percentile(residual[keep], [2.0, 100.0 - 2.0]) # percentage of the residual velocities excluded on either side of the distribution
    keep &= (residual >= low) & (residual <= high)

    save(processed_dir(galaxy) / "residual_map.pkl", {
        "model": model,
        "residual": np.where(keep, residual, np.nan),
        "error": np.where(keep, error, np.nan),
        "pc_per_pix": data["pc_per_pix"],
        "beam_fwhm_pc": data["beam_fwhm_pc"],
    })
