# Evidence for Hierarchical Turbulent Scaling in Nearby Star-Forming Galaxies

This pipeline computes velocity structure functions of the CO(2-1) "moment 1" maps of 67 PHANGS-ALMA galaxies with kinematic models from Lang et al. (2020).

## Requirements

Python 3.10 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data

Download the PHANGS-ALMA data from https://www.canfar.net/storage/vault/list/phangs/RELEASES and put it in `data/PHANGSALMA/`:

- the strict 12m+7m+TP moment 1 map and its error map of each galaxy in `data/PHANGSALMA/{galaxy}/`, e.g. `ic1954_12m+7m+tp_co21_strict_mom1.fits` and `ic1954_12m+7m+tp_co21_strict_emom1.fits`
- the sample table `phangs_sample_table_v1p6.fits`

## Running the pipeline

Everything runs through `main.py`. To run the full pipeline:

```bash
python main.py --galaxies-file txt_files/all_galaxies.txt --all
```

The pipeline has six steps, each with its own flag to run:

1. `--step-extract`: read the moment 1 map, its error map and the galaxy information
2. `--step-rotation`: subtract the rotational model and clean the residual map
3. `--step-structure-functions`: velocity increments, structure functions and fit range
4. `--step-bootstrap`: 1000 bootstrap iterations for zeta_p, beta_p, beta_SL, alpha and D
5. `--step-analysis`: selection, Mach classification and the population fits
6. `--step-plots`: the figures and table of the paper

`--data` runs steps 1 to 3, `--bootstrap` runs step 4 and `--plots` runs steps 5 and 6. Steps 1 to 4 need a list of galaxies (`-g ngc3621` or `--galaxies-file`). They save their output per galaxy in `results/` and skip a galaxy whose output already exists, unless you add `--force`. To try it on one galaxy:

```bash
python main.py -g ngc3621 --data --bootstrap
```

Add `--workers N` to process N galaxies at the same time on different cores, at most 5. 

