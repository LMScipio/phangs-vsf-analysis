"""Paths of the input tables, results and figures, saving and loading of the results, and the printed summary of
the numbers quoted in the paper."""
import os
import pickle
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT_DIR / "results"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", ROOT_DIR / "figures"))

PHANGS_DATA_DIR = DATA_DIR / "PHANGSALMA"
PHANGS_RESULTS_DIR = RESULTS_DIR / "PHANGSALMA"
TXT_DIR = ROOT_DIR / "txt_files"

ALL_GALAXIES_PATH = TXT_DIR / "all_galaxies.txt" # the 67 galaxies with a kinematic model in Lang et al. (2020)
SELECTED_GALAXIES_PATH = TXT_DIR / "selected_galaxies.txt" # the selected galaxies
D1_TABLE_PATH = TXT_DIR / "D1_table.txt" # rotation curves, Lang et al. (2020) Table D1
TAB1_TABLE_PATH = TXT_DIR / "tab1_table.txt" # distances, Lang et al. (2020) Table 1
# kinematic orientations and reliability of the fit, Lang et al. (2020) Table 2
TAB2_TABLE_PATH = TXT_DIR / "tab2_table.txt"
GMC_TABLE_PATH = TXT_DIR / "GMC_table.txt" # scaling exponents of the molecular clouds, Ma et al. (2025) Table A1

def processed_dir(galaxy):
    """Folder with the saved maps and structure functions of one galaxy."""
    return PHANGS_RESULTS_DIR / "processed" / galaxy

def read_galaxy_list(path):
    """Galaxy names in a text file, one per line."""
    return path.read_text(encoding="utf-8").split()

def write_galaxy_list(path, galaxies):
    """Galaxy names to a text file, one per line."""
    path.write_text("\n".join(galaxies) + "\n", encoding="utf-8", newline="\n")

def save(path, data):
    """Write a dictionary to a .npz (arrays) or .pkl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        np.savez_compressed(path, **data)
    else:
        with open(path, "wb") as f:
            pickle.dump(data, f)

def load(path):
    """Read a dictionary written by save."""
    if path.suffix == ".npz":
        with np.load(path) as d:
            return dict(d)
    with open(path, "rb") as f:
        return pickle.load(f)

def bootstrap_path(galaxy):
    return PHANGS_RESULTS_DIR / "bootstrap" / f"{galaxy}.npz"

def load_structure_functions(galaxy):
    return load(processed_dir(galaxy) / "structure_functions.npz")

def load_bootstrap(galaxy):
    return load(bootstrap_path(galaxy))

def print_summary(rows, selected_rows, fits):
    """The numbers quoted in the paper."""
    print(f"  {len(selected_rows)} selected galaxies out of {len(rows)} with a valid fit range, "
          f"written to {SELECTED_GALAXIES_PATH}")
    beta_test_r2 = []
    n_physical = 0
    for row in selected_rows:
        beta_test_r2.append(row["beta_test_r2"])
        if 0 <= row["D"][0] <= 3:
            n_physical += 1
    print(f"  beta-test R^2 of the selected galaxies between {min(beta_test_r2):.3f} and {max(beta_test_r2):.3f}; "
          f"0 <= D <= 3 in {n_physical} of {len(selected_rows)} selected galaxies")
    beta_median = []
    for p in range(1, 7):
        beta_p = []
        for row in selected_rows:
            beta_p.append(row["beta"][p][0])
        beta_median.append(f"{np.median(beta_p):.3f}") # median of the galaxy medians
    print("  Median beta_p, p = 1 to 6: " + ", ".join(beta_median))
    for p in range(1, 6):
        fit = fits[p]
        print(f"  p = {p}: slope {fit['slope']:.3f} (p/(p+1) = {p / (p + 1):.3f}), intercept {fit['intercept']:+.3f}, "
              f"gamma_p {fit['gamma']:.3f}, predicted {fit['gamma_pred']:.3f}")
