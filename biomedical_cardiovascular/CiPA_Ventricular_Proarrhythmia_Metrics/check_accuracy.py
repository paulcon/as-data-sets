"""Recompute the accuracy numbers quoted in DATA.md for CiPA_Ventricular_Proarrhythmia_Metrics (needs Myokit).

    python check_accuracy.py            # about 2.5 min of CPU plus compiling the model

Prints, beside the values DATA.md quotes:

1. Rows 0, 112 and 223 of cipaord_virtual_drugs.csv rerun twice with the shipped tolerances
   (CVODES abs/rel 1e-6/1e-4): from the exact Sobol' design (generate_data.design()) and from the
   CSV's 10-significant-digit inputs. For each, the largest relative difference of the eight
   features from the CSV. The first shows that the scripts reproduce the file; the second, how far
   CVODES's adaptive steps move the features for a 1e-10 perturbation of the inputs (the numerical
   noise of the data).

The other accuracy numbers in DATA.md come from `generate_data.py --check` (rerun at a tighter
tolerance, control steady state) and from the notebook (convergence of the train). The comparison
of the right-hand sides with the FDA C code is an external run, not reproducible from the
repository (DATA.md).

Deterministic. Needs numpy, scipy and Myokit (pip install myokit; the shipped data used Myokit
1.39.2) with SUNDIALS (CVODES) and a C compiler, which Myokit uses to compile the model. Without
Myokit the script stops with an error that says what to install. It reads the files next to it
and writes nothing.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cipaord_driver as cm  # noqa: E402
import generate_data as gd  # noqa: E402

ROWS = (0, 112, 223)


def main():
    try:
        import myokit
    except ImportError:
        sys.exit("Myokit is not installed. Install it (pip install myokit; the shipped data used Myokit 1.39.2) "
                 "together with SUNDIALS and a C compiler (see DATA.md), then rerun check_accuracy.py.")
    try:
        cm._sim(gd.ABS_TOL, gd.REL_TOL)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Myokit cannot compile a simulation ({type(e).__name__}: {str(e).splitlines()[0]}). "
                 "Install SUNDIALS and a C compiler (see DATA.md), then rerun check_accuracy.py.")
    print(f"Myokit {myokit.__version__}, numpy {np.__version__}; CVODES abs/rel tolerance {gd.ABS_TOL:g}/{gd.REL_TOL:g}")
    D = np.loadtxt(os.path.join(HERE, "cipaord_virtual_drugs.csv"), delimiter=",")
    m, nq = len(cm.NAMES), len(cm.QOI_NAMES)
    X = gd.design()
    t0 = time.time()
    print("\n1. Reruns of three rows: largest relative difference of the 8 features from the CSV")
    worst = {"design": 0.0, "csv": 0.0}
    for i in ROWS:
        f = D[i, m:m + nq]
        line = []
        for lab, x in (("design", X[i]), ("csv", D[i, :m])):
            a = np.array(gd.evaluate(x)[:nq])
            d = float(np.nanmax(np.abs(a - f) / np.abs(f)))
            worst[lab] = max(worst[lab], d)
            line.append(f"from the {'exact design' if lab == 'design' else 'CSV inputs'} {d:.1e}")
        print(f"   row {i:3d}: " + "; ".join(line) + f" (max input difference {np.max(np.abs(X[i] - D[i, :m]) / np.maximum(np.abs(X[i]), 1e-300)):.1e} relative)")
    print(f"   worst: exact design {worst['design']:.1e}; CSV inputs {worst['csv']:.1e}")
    print(f"\nwall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
