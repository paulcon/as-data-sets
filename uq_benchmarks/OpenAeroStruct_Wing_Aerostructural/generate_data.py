"""Write inputs.csv, outputs.csv and gradients.csv with OpenAeroStruct 2.x.

    python generate_data.py                                   # all 2000 samples, writes here
    python generate_data.py --rows 0:10 --out /tmp/oas --check    # subset, compared with the shipped files

    # the shipped files were made in chunks (one process at a time), then merged:
    python generate_data.py --rows 0:500    --out chunks/c0
    python generate_data.py --rows 500:1000 --out chunks/c1   # ... and so on up to 2000
    python generate_data.py --merge chunks/c0 chunks/c1 chunks/c2 chunks/c3

Needs `openaerostruct` (pip; the shipped files used openaerostruct 2.12.0, openmdao
3.45.2, numpy 2.5.4, scipy 1.18.1, Python 3.13), which the notebook does not need.

Inputs: M = 2000 independent uniform samples of the 12 psdr inputs (Monte Carlo,
numpy default_rng(SEED).random((M, 12)) scaled to the ranges), in physical units,
columns in oas_wing.NAMES. The stream is drawn row by row, so the first 500 rows are
exactly the 500-sample design of the first release of this data set.
Outputs: the 5 QoIs in oas_wing.QOIS for each sample. Gradients: physical-unit
gradients from `OASWing.evaluate_with_gradient` (OpenMDAO total derivatives for the 9
geometric/flight inputs, central differences for E and G, exact for mrho), column
12 * i + j = d QOI_i / d NAMES_j.

One process; no multiprocessing. `--check` compares the rows written with the same
rows of the shipped files and prints the largest relative differences. `--merge`
concatenates chunk directories (in the order given), checks that their inputs are the
design rows, and writes the three files to --out.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oas_wing as ow  # noqa: E402

SEED = 2018
M = 2000
UNITS = {"twist_1": "deg", "twist_2": "deg", "twist_3": "deg", "thickness_1": "m",
         "thickness_2": "m", "thickness_3": "m", "root_chord": "-", "taper_ratio": "-",
         "alpha": "deg", "E": "Pa", "G": "Pa", "mrho": "kg/m^3"}


def draw_inputs(m=M, seed=SEED):
    rng = np.random.default_rng(seed)
    return ow.LOWER + (ow.UPPER - ow.LOWER) * rng.random((m, len(ow.NAMES)))


def provenance():
    import openaerostruct
    import openmdao
    return ("OpenAeroStruct %s / OpenMDAO %s / numpy %s, oas_wing.py (psdr OpenAeroStruct "
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=None, help="slice a:b of the design to run")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--merge", nargs="+", default=None, help="chunk directories to concatenate")
    a = ap.parse_args()
    if os.path.abspath(a.out) == HERE and (a.check or a.rows):
        ap.error('--check and --rows need --out DIR outside this directory, so that the shipped files are not overwritten')
    if a.merge:
        merge(a.merge, a.out)
        return
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
    if a.check:
        for name in ("outputs.csv", "gradients.csv"):
            ref = np.loadtxt(os.path.join(HERE, name), delimiter=",")[rows]
            new = np.loadtxt(os.path.join(a.out, name), delimiter=",", ndmin=2)
            scale = np.maximum(np.abs(ref).max(axis=0), 1e-300)
            print("%s: max |new - shipped| / column max = %.2e" % (name, np.max(np.abs(new - ref) / scale)))


if __name__ == "__main__":
    main()
