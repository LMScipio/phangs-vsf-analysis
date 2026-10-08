"""For a chosen galaxy extract the strict moment 1 map and its corresponding error map from the PHANGS-ALMA data,
along with its centre, distance, pixel size and beam size, and the kinematic orientation, distance and rotation curve of
Lang et al. (2020)."""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from src.helpers import D1_TABLE_PATH, PHANGS_DATA_DIR, TAB1_TABLE_PATH, TAB2_TABLE_PATH, processed_dir, save

def extract_galaxy_data(galaxy):
    """Extract the strict moment 1 map, error map, centre coordinates and beam, and compute the pixel size in
    parsecs."""
    map_path = PHANGS_DATA_DIR / galaxy / f"{galaxy}_12m+7m+tp_co21_strict"
    mom1, header = fits.getdata(f"{map_path}_mom1.fits", header=True)
    emom1 = fits.getdata(f"{map_path}_emom1.fits")
    table = fits.getdata(PHANGS_DATA_DIR / "phangs_sample_table_v1p6.fits", 1)
    row = table[table["name"] == galaxy][0]

    tab1 = np.loadtxt(TAB1_TABLE_PATH, dtype=str, encoding="utf-8")
    tab2 = np.loadtxt(TAB2_TABLE_PATH, dtype=str, encoding="utf-8")
    d1 = np.loadtxt(D1_TABLE_PATH, dtype=str, encoding="utf-8")
    lang_distance = float(tab1[tab1[:, 0] == galaxy][0, 3]) # distance to galaxy in Mpc used by Lang et al. (2020)
    lang_row = tab2[tab2[:, 0] == galaxy][0] # row of this galaxy in Table 2
    # position angle phi and inclination i in degrees, systemic velocity V_sys in km/s
    lang_phi, lang_inc, lang_v_sys = lang_row[[1, 3, 5]].astype(float)
    # rotation curve: radius R in kpc and rotation velocity V_rot in km/s
    lang_rc_r, lang_rc_v = d1[d1[:, 0] == galaxy][:, 1:3].astype(float).T

    pc_per_arcsec = 4.848 * row["dist"]
    pix_ra = abs(header["CDELT1"]) * 3600
    pix_dec = abs(header["CDELT2"]) * 3600
    # turn galaxy centre to pixel coordinates
    cx, cy = WCS(header).world_to_pixel(SkyCoord(float(row["orient_ra"]) * u.deg, float(row["orient_dec"]) * u.deg))

    save(processed_dir(galaxy) / "moment1.pkl", {
        "moment1": mom1,
        "moment1_error": emom1,
        "center_pix": (float(cx), float(cy)),
        "pix_scale_arcsec": (pix_ra + pix_dec) / 2.0, # average pixel size in arcseconds
        "pc_per_pix": (pix_ra * pc_per_arcsec + pix_dec * pc_per_arcsec) / 2.0, # average pixel size in parsecs
        "beam_fwhm_pc": header["BMAJ"] * 3600 * pc_per_arcsec, # beam size in parsecs
        "lang_distance_mpc": lang_distance,
        "lang_phi_deg": lang_phi,
        "lang_inc_deg": lang_inc,
        "lang_v_sys_kms": lang_v_sys,
        "lang_rc_r_kpc": lang_rc_r,
        "lang_rc_v_kms": lang_rc_v,
    })
