"""Write the stored data of Marine_Carbonate_System_PyCO2SYS with PyCO2SYS v1.8.3.

Code: MIT. Needs PyCO2SYS 1.8.3 (GPLv3, not bundled), autograd and pandas, which the notebook's environment
does not need; without PyCO2SYS the script stops with an install message. The shipped CSVs are the record
(stored data): rerun this only if a problem is found in them.

    python -m venv pyco2-venv && pyco2-venv/bin/pip install PyCO2SYS==1.8.3 pandas
    pyco2-venv/bin/python generate_data.py                              # writes the seven CSVs here
    pyco2-venv/bin/python generate_data.py --out <scratch dir> --check  # writes elsewhere, compares with these

The accuracy checks (override identity, step sizes, PyCO2SYS's own derivatives) are in check_accuracy.py.

Two studies, each M = 1000 independent Gaussian draws x = NOMINAL + SD * xi, xi ~ N(0, I_14), with the
saved seeds below: "measurement" (measurement and constant uncertainties, at the paper's Table 4 test
conditions, 1234 dbar) and "oceanwide" (a wide spread of the measured inputs, same constant uncertainties,
at the sea surface, 0 dbar). See DATA.md.
"""
import argparse
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carbonate_driver as cm  # noqa: E402

pyco2 = cm.pyco2
M = 1000
STUDIES = {
    "measurement": dict(sd=cm.SD_MEASUREMENT, seed=20220104, held={}),
    "oceanwide": dict(sd=cm.SD_OCEANWIDE, seed=20220105, held=cm.HELD_OCEANWIDE),
}



def draw(study):
    s = STUDIES[study]
    xi = np.random.default_rng(s["seed"]).standard_normal((M, len(cm.NAMES)))
    return cm.NOMINAL + s["sd"] * xi


def outside_physical(X):
    """Counts of samples outside physical meaning (none are dropped)."""
    return {
        "total_silicate < 0": int(np.sum(X[:, 4] < 0)),
        "total_phosphate < 0": int(np.sum(X[:, 5] < 0)),
        "salinity < 0": int(np.sum(X[:, 3] < 0)),
        "temperature < -1.9 degC (below seawater freezing)": int(np.sum(X[:, 2] < -1.9)),
        "temperature > 30 degC": int(np.sum(X[:, 2] > 30)),
        "temperature > 35 degC (above open-ocean maximum)": int(np.sum(X[:, 2] > 35)),
        "f_total_borate <= 0": int(np.sum(X[:, 13] <= 0)),
        "alkalinity < dic": int(np.sum(X[:, 0] < X[:, 1])),
        "alkalinity - dic < 50 umol/kg": int(np.sum(X[:, 0] - X[:, 1] < 50)),
    }


def write_csv(path, A, description, columns):
    np.savetxt(path, A, delimiter=",", fmt="%.17g",
               header=description + "\n" + ",".join(columns), comments="# ")


def column_names(study):
    xcols = [f"{n} [{u}]" for n, u in zip(cm.NAMES, cm.UNITS)]
    ycols = [f"{o} [{u}]" for o, u in zip(cm.OUTPUTS, cm.OUTPUT_UNITS)]
    gcols = [f"d {o} / d {n}" for o in cm.OUTPUTS for n in cm.NAMES]
    return xcols, ycols, gcols


def generate(outdir):
    os.makedirs(outdir, exist_ok=True)
    for study in STUDIES:
        t0 = time.time()
        X = draw(study)
        print(f"[{study}] outside physical meaning (kept):", outside_physical(X))
        held = STUDIES[study]["held"]
        Y = cm.evaluate(X, held=held)
        G = cm.gradients(X, held=held)  # (M, 4, 14)
        print(f"[{study}] held overrides {held}; pCO2 range [{Y[:, 1].min():.1f}, {Y[:, 1].max():.1f}] uatm, "
              f"pCO2 > 1000 uatm: {int(np.sum(Y[:, 1] > 1000))}, Omega_ar < 1: {int(np.sum(Y[:, 2] < 1))}")
        xcols, ycols, gcols = column_names(study)
        tag = (f"PyCO2SYS {pyco2.__version__}, {study} study, M = {M}, seed {STUDIES[study]['seed']}, "
               f"pressure {cm.HELD['pressure'] if 'pressure' not in held else held['pressure']:g} dbar, "
               "as-data-sets Marine_Carbonate_System_PyCO2SYS")
        write_csv(os.path.join(outdir, f"inputs_{study}.csv"), X, "Inputs (physical units); " + tag, xcols)
        write_csv(os.path.join(outdir, f"outputs_{study}.csv"), Y, "Outputs; " + tag, ycols)
        write_csv(os.path.join(outdir, f"gradients_{study}.csv"), G.reshape(M, -1),
                  "Gradients w.r.t. physical inputs, column 14*q + j = d output_q / d input_j "
                  "(central differences); " + tag, gcols)
        print(f"[{study}] wrote {M} samples in {time.time() - t0:.1f} s")


REF_KEYS = ("pH_total", "pCO2", "saturation_aragonite", "saturation_calcite", "fCO2", "xCO2",
            "carbonate", "bicarbonate", "aqueous_CO2")
REF_UNITS = ("1", "uatm", "1", "1", "uatm", "umol/mol", "umol/kg", "umol/kg", "umol/kg")
UNC_FROM = ("par1", "par2", "temperature", "salinity", "total_silicate", "total_phosphate",
            "pk_CO2", "pk_carbonic_1", "pk_carbonic_2", "pk_borate", "pk_water",
            "pk_aragonite", "pk_calcite", "total_borate__f")


# The paper's round-robin script (manuscript/round_robin.py, PyCO2SYS v1.8.0) adds two hypothetical
# acid-base systems to the Table 4 conditions; with them, the v1.8.0 default options reproduce every
# printed Table 4 value. They are not seawater constituents and are not used in the studies.
ROUND_ROBIN_EXTRA = dict(total_alpha=5, k_alpha=1e-4, total_beta=6, k_beta=1e-8)


def reference_rows():
    """The five rows of nominal_reference.csv (see reference)."""
    import warnings
    warnings.simplefilter("ignore")
    rows = []
    for study in STUDIES:
        _, kw = cm._base_kwargs(cm.NOMINAL[None, :], STUDIES[study]["held"])
        r = pyco2.sys(**kw)
        rows.append([float(np.ravel(r[k])[0]) for k in REF_KEYS])
        sd = STUDIES[study]["sd"]
        r = pyco2.sys(**kw, uncertainty_into=list(REF_KEYS), uncertainty_from=dict(zip(UNC_FROM, sd)))
        rows.append([float(np.ravel(r["u_" + k])[0]) for k in REF_KEYS])
    _, kw = cm._base_kwargs(cm.NOMINAL[None, :])
    r = pyco2.sys(**kw, **ROUND_ROBIN_EXTRA)
    rows.append([float(np.ravel(r[k])[0]) for k in REF_KEYS])
    return np.array(rows)


def reference(outdir):
    """Write nominal_reference.csv. Rows: 0 PyCO2SYS at the nominal point at the measurement study's
    conditions (Table 4, 1234 dbar); 1 PyCO2SYS's own linear propagation (its Eq. 24, forward differences)
    of the measurement study's sds there; 2 the nominal point at the ocean-wide study's conditions (0 dbar);
    3 Eq. 24 of the ocean-wide sds there; 4 row 0 plus the round-robin script's two hypothetical acid-base
    systems (ROUND_ROBIN_EXTRA), which reproduces the paper's Table 4."""
    write_csv(os.path.join(outdir, "nominal_reference.csv"), reference_rows(),
              f"PyCO2SYS {pyco2.__version__}: nominal point at 1234 dbar (row 0) and Eq. 24 propagation of the "
              "measurement sds (row 1); nominal point at 0 dbar (row 2) and Eq. 24 of the oceanwide sds (row 3); "
              "row 0 plus the round-robin script's hypothetical acid-base systems alpha and beta (row 4, = Table 4)",
              [f"{k} [{u}]" for k, u in zip(REF_KEYS, REF_UNITS)])


def check_against(outdir):
    worst = 0.0
    for study in STUDIES:
        for kind in ("inputs", "outputs", "gradients"):
            a = np.loadtxt(os.path.join(HERE, f"{kind}_{study}.csv"), delimiter=",")
            b = np.loadtxt(os.path.join(outdir, f"{kind}_{study}.csv"), delimiter=",")
            d = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300 + np.max(np.abs(a)) * 1e-12))
            worst = max(worst, d)
            print(f"{kind}_{study}.csv: identical={np.array_equal(a, b)}, max rel diff {d:.2e}")
    a = np.loadtxt(os.path.join(HERE, "nominal_reference.csv"), delimiter=",")
    b = np.loadtxt(os.path.join(outdir, "nominal_reference.csv"), delimiter=",")
    print(f"nominal_reference.csv: identical={np.array_equal(a, b)}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reference-only", action="store_true")
    a = ap.parse_args()
    if os.path.abspath(a.out) == HERE and a.check:
        ap.error('--check needs --out DIR outside this directory, so that the shipped files are not overwritten')
    if pyco2 is None:
        print(cm.INSTALL.replace("and rerun.", "and rerun generate_data.py."), file=sys.stderr)
        sys.exit(3)
    print(f"PyCO2SYS {pyco2.__version__}, numpy {np.__version__}, Python {platform.python_version()}")
    t = time.time()
    os.makedirs(a.out, exist_ok=True)
    if a.reference_only:
        reference(a.out)
        raise SystemExit
    generate(a.out)
    reference(a.out)
    if a.check:
        check_against(a.out)
    print(f"total {time.time() - t:.1f} s")
