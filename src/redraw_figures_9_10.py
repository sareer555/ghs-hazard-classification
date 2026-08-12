"""
REDRAW FIGURES 9 AND 10 FROM THEIR SAVED RESULTS
================================================
Figures 9 and 10 are produced by learning_curve.py and
controlled_size_experiment.py, both of which train models and take hours. When
only the drawing changes - a colour scheme, a resolution, a label - there is no
reason to retrain anything: both scripts write their results to CSV, and both
now expose the plotting as make_figure().

This script loads those tables and redraws the two figures. It touches no model
and recomputes no number, so the figures it produces describe exactly the same
experiment as before.

Author : Sareer Ahmad
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghs_config import stamped

import learning_curve
import controlled_size_experiment


def main():
    """Redraw both figures from the stored result tables."""
    print("=" * 78)
    print("REDRAWING FIGURES 9 AND 10 FROM SAVED RESULTS")
    print("=" * 78)

    curve_path = stamped("EXTRA_learning_curve.csv")
    size_path = stamped("EXTRA_controlled_size_experiment.csv")
    missing = [p for p in (curve_path, size_path) if not os.path.exists(p)]
    if missing:
        raise SystemExit("Missing result tables, so the figures cannot be "
                         "redrawn without re-running the experiments:\n  "
                         + "\n  ".join(missing))

    table = pd.read_csv(curve_path)
    # main() passes the final mean AUC as the reference line.
    learning_curve.make_figure(table, table["mean_auc"].iloc[-1])
    print(f"   Figure 9  redrawn from {os.path.basename(curve_path)} "
          f"({len(table)} training sizes)")

    table = pd.read_csv(size_path)
    # main() passes the smallest condition as the reference line.
    controlled_size_experiment.make_figure(table, table.iloc[0])
    print(f"   Figure 10 redrawn from {os.path.basename(size_path)} "
          f"({len(table)} conditions)")

    print("\nBoth figures rewritten at the shared publication resolution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
