"""Make all the figures and table of the paper in figures/. The code of each figure is in src/figures/."""
import matplotlib
from src.figures.beta_scaling import plot_beta_scaling
from src.figures.mach_classification_histogram import plot_mach_classification_histogram
from src.figures.molecular_cloud_interdependence import load_gmc_table, plot_consecutive_pairs
from src.figures.scaling_exponents_table import write_scaling_exponents_table
from src.figures.summary_graph import make_summary_graph
from src.figures.zeta_interdependence import plot_zeta_interdependence
from src.helpers import FIGURES_DIR
matplotlib.use("Agg")

def make_figures(analysis):
    """Make all figures and write the table of scaling exponents."""
    rows, selected_rows, mach_results = analysis["rows"], analysis["selected_rows"], analysis["mach_results"]

    outputs = [
        make_summary_graph(FIGURES_DIR / "summary_graph.pdf"),
        plot_zeta_interdependence(selected_rows, analysis["fits"], FIGURES_DIR / "zeta_interdependence.pdf"),
        plot_consecutive_pairs(load_gmc_table(), FIGURES_DIR / "molecular_cloud_interdependence.pdf"),
        plot_beta_scaling(selected_rows, FIGURES_DIR / "beta_scaling.pdf"),
        plot_mach_classification_histogram(mach_results, FIGURES_DIR / "mach_classification_histogram.pdf"),
        plot_zeta_interdependence(rows, analysis["fits_all"], FIGURES_DIR / "zeta_interdependence_full_sample.pdf"),
        write_scaling_exponents_table(selected_rows, mach_results, FIGURES_DIR / "scaling_exponents_table.tex"),
    ]
    for path in outputs:
        print(f"  Saved: {path}")
