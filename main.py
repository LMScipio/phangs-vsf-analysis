"""
Main file from which you can run the full pipeline, see README.md for instructions.
"""

import argparse
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path
from src.helpers import (PHANGS_DATA_DIR, ROOT_DIR, bootstrap_path, load_structure_functions, processed_dir,
    read_galaxy_list)
from src.extract_galaxy_data import extract_galaxy_data
from src.subtract_rotational_model import subtract_rotational_model
from src.compute_structure_functions import compute_structure_functions
from src.bootstrap import run_galaxy_bootstrap
from src.analysis import run_analysis
from src.make_figures import make_figures

def banner(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60, flush=True)

def process_galaxy(galaxy, stages, force):
    """Run the selected stages for one galaxy, skipping a stage whose output already exists. Returns False if the
    galaxy failed."""
    folder = processed_dir(galaxy)
    try:
        if "extract" in stages and (force or not (folder / "moment1.pkl").exists()):
            print(f"  {galaxy}: reading the moment 1 map", flush=True)
            extract_galaxy_data(galaxy)
        if "rotation" in stages and (force or not (folder / "residual_map.pkl").exists()):
            print(f"  {galaxy}: subtracting the rotational model and cleaning the residual map", flush=True)
            subtract_rotational_model(galaxy)
        if "structure functions" in stages and (force or not (folder / "structure_functions.npz").exists()):
            print(f"  {galaxy}: structure functions and fit range", flush=True)
            compute_structure_functions(galaxy)
        if "bootstrap" in stages and (force or not bootstrap_path(galaxy).exists()):
            if load_structure_functions(galaxy)["fit_mask"].any():
                print(f"  {galaxy}: bootstrap", flush=True)
                run_galaxy_bootstrap(galaxy)
            else:
                print(f"  {galaxy}: bootstrap skipped, no valid fit range", flush=True)
    except Exception:
        print(f"  {galaxy}: failed\n{traceback.format_exc()}", flush=True)
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="PHANGS-ALMA velocity structure function pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("-g", "--galaxies", nargs="+", metavar="GALAXY", help="One or more galaxy names")
    parser.add_argument("--galaxies-file", metavar="FILE", help="Text file with one galaxy name per line")
    parser.add_argument("--limit", type=int, metavar="N", help="Process only the first N galaxies")
    parser.add_argument("--force", action="store_true",
                        help="Recompute the results of each galaxy even when they exist")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 6), metavar="N",
                        help="Number of galaxies processed at the same time, at most 5")
    parser.add_argument("--all", action="store_true", help="Run the whole pipeline")
    parser.add_argument("--data", action="store_true",
                        help="Read the maps, subtract the rotational model, clean the residual map and compute the "
                        "structure functions")
    parser.add_argument("--bootstrap", action="store_true", help="Run the bootstrap")
    parser.add_argument("--plots", action="store_true",
                        help="Run the analysis and make the figures and table of the paper")
    parser.add_argument("--step-extract", action="store_true",
                        help="Read the moment 1 map, its error map and the galaxy information")
    parser.add_argument("--step-rotation", action="store_true",
                        help="Subtract the rotational model and clean the residual map")
    parser.add_argument("--step-structure-functions", action="store_true",
                        help="Velocity increments, structure functions and fit range")
    parser.add_argument("--step-bootstrap", action="store_true",
                        help="Bootstrap of zeta_p, beta_p, beta_SL, alpha and D")
    parser.add_argument("--step-analysis", action="store_true",
                        help="Selection, Mach classification and the population fits")
    parser.add_argument("--step-plots", action="store_true", help="The figures and table of the paper")
    args = parser.parse_args()

    stages = []
    if args.all or args.data or args.step_extract:
        stages.append("extract")
    if args.all or args.data or args.step_rotation:
        stages.append("rotation")
    if args.all or args.data or args.step_structure_functions:
        stages.append("structure functions")
    if args.all or args.bootstrap or args.step_bootstrap:
        stages.append("bootstrap")
    do_analysis = args.all or args.plots or args.step_analysis or args.step_plots
    do_plots = args.all or args.plots or args.step_plots
    if not stages and not do_analysis:
        print("Nothing selected. Use --all, --data, --bootstrap, --plots or the --step-* flags; see --help.")
        sys.exit(0)

    if stages:
        if args.galaxies:
            galaxies = []
            for g in args.galaxies:
                galaxies.append("".join(g.split()).lower())
        elif args.galaxies_file:
            path = Path(args.galaxies_file)
            if not path.is_absolute():
                path = ROOT_DIR / path
            if not path.exists():
                sys.exit(f"error: --galaxies-file not found: {path}")
            galaxies = read_galaxy_list(path)
        else:
            sys.exit("error: no galaxies specified, pass -g/--galaxies NAME [NAME ...] or --galaxies-file PATH")
        with_data, skipped = [], []
        for g in galaxies[: args.limit]:
            if (PHANGS_DATA_DIR / g).is_dir() or (processed_dir(g) / "moment1.pkl").exists():
                with_data.append(g)
            else:
                skipped.append(g)
        galaxies = with_data
        if skipped:
            print(f"Skipping {len(skipped)} galaxies without data: {', '.join(skipped)}")
        if not galaxies:
            sys.exit("No galaxies with data found.")

        banner(f"{', '.join(stages).capitalize()} for {len(galaxies)} galaxies "
               f"(--force: {args.force}, --workers: {args.workers})")
        if args.workers > 1:
            jobs = []
            for galaxy in galaxies:
                jobs.append((galaxy, stages, args.force))
            with Pool(args.workers) as pool:
                # each worker takes the next galaxy as soon as it is free
                succeeded = pool.starmap(process_galaxy, jobs, chunksize=1)
        else:
            succeeded = []
            for galaxy in galaxies:
                succeeded.append(process_galaxy(galaxy, stages, args.force))
        failed = []
        for i in range(len(galaxies)):
            if not succeeded[i]:
                failed.append(galaxies[i])
        if failed:
            print(f"\n  Failed: {', '.join(failed)}")

    analysis = None
    if do_analysis:
        banner("Selection, Mach classification and the population fits")
        analysis = run_analysis()
    if do_plots:
        banner("Figures and table of the paper")
        make_figures(analysis)
    banner("Pipeline complete.")

if __name__ == "__main__":
    main()
