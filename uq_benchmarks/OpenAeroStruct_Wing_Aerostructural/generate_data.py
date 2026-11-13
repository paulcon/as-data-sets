"""Write inputs.csv, outputs.csv and gradients.csv with OpenAeroStruct 2.x.

    python generate_data.py                                   # all 2000 samples, writes here
    python generate_data.py --check                           # all 2000 samples in a temporary directory,
                                                              # compared with the shipped files
    python generate_data.py --rows 0:10 --check               # subset, compared with the shipped files

    # the shipped files were made in chunks (one process at a time), then merged:
    python generate_data.py --rows 0:500    --out chunks/c0
    python generate_data.py --rows 500:1000 --out chunks/c1   # ... and so on up to 2000
    python generate_data.py --merge chunks/c0 chunks/c1 chunks/c2 chunks/c3

Needs `openaerostruct` (pip install openaerostruct==2.12.0; the shipped files used
openaerostruct 2.12.0, openmdao 3.45.2, numpy 2.5.4, scipy 1.18.1, Python 3.13), which the
notebook does not need. Without it the script stops with an error that says what to install
(`--merge` needs only numpy). Another OpenAeroStruct version stops it too, unless
`--any-oas` is given.

Inputs: M = 2000 independent uniform samples of the 12 psdr inputs (Monte Carlo,
numpy default_rng(SEED).random((M, 12)) scaled to the ranges), in physical units,
columns in oas_wing_driver.NAMES. The whole design is drawn from the seed every time, so a
chunk --rows a:b runs exactly rows a:b of it.
Outputs: the 5 QoIs in oas_wing_driver.QOIS for each sample. Gradients: physical-unit
gradients from `OASWing.evaluate_with_gradient` (OpenMDAO total derivatives for the 9
geometric/flight inputs, central differences for E and G, exact for mrho), column
12 * i + j = d QOI_i / d NAMES_j.

One process; no multiprocessing. `--check` compares the rows written with the same
rows of the shipped files and prints the largest differences relative to each column's
largest entry; it fails only on structural mismatches (shapes, NaN pattern, inputs). `--merge`
concatenates chunk directories (in the order given), checks that their inputs are the
design rows, and writes the three files to --out.
"""
import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oas_wing_driver as ow  # noqa: E402

SEED = 2018
M = 2000
OAS_VERSION = "2.12.0"   #: the version that wrote the shipped files
# --check reports |new - shipped| / column max (E, G columns are central differences)
INPUT_RTOL = 1e-8  #: --check: the inputs (deterministic draws) may differ from the shipped ones by round-off only
UNITS = {"twist_1": "deg", "twist_2": "deg", "twist_3": "deg", "thickness_1": "m",
         "thickness_2": "m", "thickness_3": "m", "root_chord": "-", "taper_ratio": "-",
         "alpha": "deg", "E": "Pa", "G": "Pa", "mrho": "kg/m^3"}


def draw_inputs(m=M, seed=SEED):
    rng = np.random.default_rng(seed)
    return ow.LOWER + (ow.UPPER - ow.LOWER) * rng.random((m, len(ow.NAMES)))


def provenance():
    import openaerostruct
    import openmdao
    return ("OpenAeroStruct %s / OpenMDAO %s / numpy %s, oas_wing_driver.py (psdr OpenAeroStruct "
            "test function rebuilt on OAS 2.x; NLBGS atol 1e-10)" %
            (openaerostruct.__version__, openmdao.__version__, np.__version__))


def write(path, A, head1, cols, fmt):
    with open(path, "w") as fh:
        fh.write("# " + head1 + "\n# " + ",".join(cols) + "\n")
        np.savetxt(fh, A, delimiter=",", fmt=fmt)


def header_line(path):
    with open(path) as fh:
        return fh.readline()[2:].rstrip("\n")


def write_all(out, X, F, D, prov, what):
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, "inputs.csv"), X,
          "%s Monte Carlo samples (numpy default_rng(%d)), independent uniform on the psdr "
          "OpenAeroStruct ranges; physical units" % (what, SEED),
          ["%s [%s]" % (n, UNITS[n]) for n in ow.NAMES], "%.15e")
    write(os.path.join(out, "outputs.csv"), F, prov,
          ["%s [%s]" % (q, u) for q, u in zip(ow.QOIS, ow.QOI_UNITS)], "%.12e")
    write(os.path.join(out, "gradients.csv"), D,
          "column 12*i+j = d QOI_i / d x_j in physical units; " + prov,
          ["d%s/d%s" % (q, n) for q in ow.QOIS for n in ow.NAMES], "%.10e")


def merge(dirs, out):
    X = draw_inputs()
    parts = {k: [] for k in ("inputs", "outputs", "gradients")}
    for d in dirs:
        for k in parts:
            parts[k].append(np.loadtxt(os.path.join(d, k + ".csv"), delimiter=",", ndmin=2))
    Xi = np.vstack(parts["inputs"])
    n = len(Xi)
    err = np.max(np.abs(Xi - X[:n]) / (ow.UPPER - ow.LOWER))
    print("merged %d rows; inputs vs design rows 0:%d, max normalized difference %.1e" % (n, n, err))
    if err > 1e-14:
        raise SystemExit("chunks are not the design rows in order")
    prov = header_line(os.path.join(dirs[0], "outputs.csv"))
    write_all(out, X[:n], np.vstack(parts["outputs"]), np.vstack(parts["gradients"]), prov, str(n))


def compare(out, rows):
    """Print |new - shipped| / column max for the rows written to out; return the structural failures."""
    bad, worst = [], (-1.0, "")
    for name in ("inputs.csv", "outputs.csv", "gradients.csv"):
        ref = np.loadtxt(os.path.join(HERE, name), delimiter=",", ndmin=2)
        new = np.loadtxt(os.path.join(out, name), delimiter=",", ndmin=2)
        if len(ref) != M or ref.shape[1:] != new.shape[1:] or len(new) != len(rows):
            bad.append("FAIL: %s: shape %s, shipped %s" % (name, new.shape, ref.shape))
            continue
        ref = ref[rows]
        if not np.array_equal(np.isnan(new), np.isnan(ref)):
            bad.append("FAIL: %s: the NaN pattern differs from the shipped file" % name)
            continue
        scale = np.maximum(np.nanmax(np.abs(ref), axis=0), 1e-300)
        d = np.nanmax(np.abs(new - ref) / scale)
        print("%s: max |new - shipped| / column max = %.2e" % (name, d))
        if name == "inputs.csv" and d > INPUT_RTOL:
            bad.append("FAIL: %s differs from the shipped inputs by %.1e > %g" % (name, d, INPUT_RTOL))
        worst = max(worst, (d, name))
    print("largest difference from the shipped files: %.1e (relative to the column max, %s)" % worst)
    for line in bad:
        print(line)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None, help="slice a:b of the design to run")
    ap.add_argument("--out", default=None,
                    help="output directory (default: this directory; with --check, a temporary directory)")
    ap.add_argument("--check", action="store_true", help="compare the rows written with the shipped files")
    ap.add_argument("--merge", nargs="+", default=None, help="chunk directories to concatenate")
    ap.add_argument("--any-oas", action="store_true", help="run with an OpenAeroStruct version other than %s" % OAS_VERSION)
    a = ap.parse_args()
    if a.out is None:
        if a.check:
            a.out = tempfile.mkdtemp(prefix="check_")
            atexit.register(shutil.rmtree, a.out, ignore_errors=True)
        else:
            a.out = HERE
    if os.path.abspath(a.out) == HERE and (a.check or a.rows):
        ap.error('--check and --rows need --out DIR outside this directory, so that the shipped files are not overwritten')
    if a.merge:
        merge(a.merge, a.out)
        return
    try:
        import openaerostruct
    except ImportError:
        print("OpenAeroStruct is not installed. Install it (pip install openaerostruct==%s, the version "
              "that wrote the shipped files; it pulls OpenMDAO) and rerun this script." % OAS_VERSION, file=sys.stderr)
        sys.exit(3)
    if openaerostruct.__version__ != OAS_VERSION and not a.any_oas:
        print("openaerostruct %s installed; the shipped files used %s (pip install openaerostruct==%s), "
              "or pass --any-oas" % (openaerostruct.__version__, OAS_VERSION, OAS_VERSION), file=sys.stderr)
        sys.exit(3)
    X = draw_inputs()
    rows = np.arange(M)
    if a.rows:
        i0, i1 = (int(v) for v in a.rows.split(":"))
        rows = rows[i0:i1]
    model = ow.OASWing()
    F = np.zeros((len(rows), 5))
    D = np.zeros((len(rows), 5 * 12))
    t0 = time.time()
    c0 = time.process_time()
    for n, r in enumerate(rows):
        y, g = model.evaluate_with_gradient(X[r])
        F[n] = y
        D[n] = g.ravel()
        if (n + 1) % 50 == 0:
            print("%d/%d  %.0f s wall, %.0f s CPU" % (n + 1, len(rows), time.time() - t0,
                                                     time.process_time() - c0), flush=True)
    elapsed = time.time() - t0
    cpu = time.process_time() - c0
    what = str(len(rows)) if len(rows) == M else "rows %d:%d of the %d" % (rows[0], rows[-1] + 1, M)
    write_all(a.out, X[rows], F, D, provenance(), what)
    print("%d samples in %.1f s wall, %.1f s CPU (%.2f s CPU per sample)" %
          (len(rows), elapsed, cpu, cpu / len(rows)))
    if a.check and compare(a.out, rows):
        sys.exit(1)

if __name__ == "__main__":
    main()
