"""Write the data files of NACA2412_Flapped_Airfoil by running XFOIL 6.99.

Kind: stored data. The numbers come from XFOIL (M. Drela and H. Youngren, MIT, GPL), an
outside program that this script drives through xfoil_run.py; see DATA.md for the build of
the executable that wrote the shipped files. numpy is the only Python dependency.

    python generate_data.py --xfoil /path/to/xfoil             # all XFOIL files (about 10 min)
    python generate_data.py --xfoil /path/to/xfoil --part primary
    python generate_data.py --xfoil /path/to/xfoil --check 25  # rerun 25 rows of each design,
                                                               # plus every ITER 100 row
    python generate_data.py --davis-src /path/to/uqchallenge2/training/training_100.csv

Files written (all CSV with two '#' header lines, read with np.loadtxt(path, delimiter=",")):
  inputs.csv, outputs.csv                     challenge study, m = 3 (alpha, flap, Re)
  extension_inputs.csv, extension_outputs.csv our extension, m = 6 (+ Ncrit, two trip x/c)
  reference_points.csv                        XFOIL at the 7 calibration + 4 prediction points
  davis_rerun_outputs.csv                     XFOIL at the 98 inputs of the authors' file
  davis_training_98.csv                       the authors' training_100.csv, text copied as is
  re_sensitivity.csv                          XFOIL at the 7 calibration points, 5 Re values
                                              including the paper's printed 665,000-735,000
  convergence_ladder.csv                      how each point that needed the retry ladder
                                              (status 3 or 4) or failed (status -1) went
A point where XFOIL does not converge is kept, with nan outputs and status -1. The status
column of every output file says how each point converged (xfoil_run.STATUS_NOTE).
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xfoil_run  # noqa: E402

SEED_PRIMARY = 2412
SEED_EXTENSION = 24120
M_PRIMARY = 1000
M_EXTENSION = 1000

# Challenge envelope (Davis and Renganathan, Sec. 1.2.1, p. 3; xfoil_training.py)
PRIMARY_BOUNDS = [("alpha", "deg", -5.0, 10.0),
                  ("flap", "deg", -5.0, 15.0),
                  ("Re", "-", 7e5 * 0.995, 7e5 * 1.005)]
# Our extension: XFOIL transition settings the challenge holds fixed (Ncrit = 0.01, no trip)
EXTENSION_BOUNDS = PRIMARY_BOUNDS + [("ncrit", "-", 0.01, 9.0),
                                     ("xtr_top", "x/c", 0.05, 1.0),
                                     ("xtr_bot", "x/c", 0.05, 1.0)]

# Calibration points (Table 1, p. 2) and prediction points (Fig. 1; desired_truths.csv)
REFERENCE = [  # set (0 calibration, 1 prediction), alpha, flap, Re
    (0, 0.0, 0.0, 7e5), (0, 5.0, 0.0, 7e5), (0, 10.0, 0.0, 7e5), (0, 0.0, 5.0, 7e5),
    (0, 0.0, 10.0, 7e5), (0, 5.0, 5.0, 7e5), (0, -5.0, -5.0, 7e5),
    (1, 0.0, 0.0, 7e5), (1, 2.0, 2.0, 7e5), (1, 7.0, 15.0, 7e5), (1, -3.0, -3.0, 7e5),
]

OUT_COLS = "CL (-),CD (-),CDp (-),CM (-),xtr_top_out (x/c),xtr_bot_out (x/c),status"
OUT_FMT = ["%.4f", "%.5f", "%.5f", "%.4f", "%.4f", "%.4f", "%d"]
STATUS_NOTE = xfoil_run.STATUS_NOTE

# Re sensitivity (labelled check, not a design): the paper's printed range 665,000-735,000
# (Sec. 1.2.1, p. 3), the code's range 696,500-703,500, and the nominal 700,000
RE_SENSITIVITY = [665000.0, 696500.0, 700000.0, 703500.0, 735000.0]

# convergence_ladder.csv: one row per point with status 3, 4 or -1
LADDER_FILES = {"outputs.csv": 0, "extension_outputs.csv": 1, "reference_points.csv": 2,
                "davis_rerun_outputs.csv": 3, "re_sensitivity.csv": 4}
LADDER_COLS = ("file (0 outputs.csv; 1 extension_outputs.csv; 2 reference_points.csv; "
               "3 davis_rerun_outputs.csv; 4 re_sensitivity.csv),row (0-based data row),status,"
               "seq_side (-1 start below alpha; +1 above; nan),seq_start (deg),"
               "seq_step_min (deg),seq_n_steps,seq_n_runs (XFOIL processes)")
LADDER_FMT = ["%d", "%d", "%d", "%g", "%.17g", "%.17g", "%g", "%g"]


def latin_hypercube(n, bounds, seed):
    """Latin hypercube on the box: one point in each of n equal strata per input."""
    rng = np.random.default_rng(seed)
    d = len(bounds)
    u = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
    lo = np.array([b[2] for b in bounds])
    hi = np.array([b[3] for b in bounds])
    return lo + u * (hi - lo)


def run_rows(X, names, exe, label, ladder=None):
    """Run XFOIL row by row. If ladder is a list, append one convergence_ladder row
    (without the file column) for every point with status 3, 4 or -1."""
    out = np.full((X.shape[0], 7), np.nan)
    t0 = time.time()
    for i, x in enumerate(X):
        kw = dict(zip(names, x))
        seq = None
        try:
            r = xfoil_run.run(exe=exe, **kw)
            out[i] = [r["cl"], r["cd"], r["cdp"], r["cm"], r["xtr_top_out"],
                      r["xtr_bot_out"], r["status"]]
            seq = r.get("seq")
        except xfoil_run.XfoilFailure:
            out[i, 6] = -1
        if ladder is not None and (out[i, 6] >= 3 or out[i, 6] < 0):
            if seq:
                ladder.append([i, out[i, 6], -1.0 if seq["side"] == "below" else 1.0,
                               seq["start"], seq["step_min"], seq["n_steps"], seq["n_runs"]])
            else:
                ladder.append([i, out[i, 6]] + [np.nan] * 5)
        if (i + 1) % 100 == 0:
            print("  %s: %d/%d, %.0f s" % (label, i + 1, X.shape[0], time.time() - t0),
                  flush=True)
    st = out[:, 6].astype(int)
    print("  %s: %d points in %.1f s; status counts %s" % (
        label, X.shape[0], time.time() - t0,
        {int(k): int(v) for k, v in zip(*np.unique(st, return_counts=True))}), flush=True)
    return out


def save(path, data, fmt, line1, line2):
    np.savetxt(path, data, fmt=fmt, delimiter=",", header=line1 + "\n" + line2,
               comments="# ")


def xfoil_names(bounds):
    return [b[0].lower() for b in bounds]   # xfoil_run.run keyword names


def save_ladder(fname, rows):
    """Replace the convergence_ladder.csv rows of one output file, keep the others."""
    path = os.path.join(HERE, "convergence_ladder.csv")
    fid = LADDER_FILES[fname]
    old = np.empty((0, 8))
    if os.path.exists(path):
        old = np.loadtxt(path, delimiter=",", ndmin=2).reshape(-1, 8)
        old = old[old[:, 0] != fid]
    new = np.array([[fid] + list(r) for r in rows], dtype=float).reshape(-1, 8)
    allr = np.vstack([old, new])
    allr = allr[np.lexsort((allr[:, 1], allr[:, 0]))]
    save(path, allr, LADDER_FMT,
         "Points of the XFOIL files that needed the retry ladder of xfoil_run.py (status 3 = "
         "ITER 100 directly, 4 = ITER 100 continuation) or failed every attempt (-1); seq_* "
         "describe the continuation path of status 4 rows", LADDER_COLS)


def part_design(bounds, m_rows, seed, stem, exe):
    X = latin_hypercube(m_rows, bounds, seed)
    cols = ",".join("%s (%s)" % (b[0], b[1]) for b in bounds)
    save(os.path.join(HERE, stem + "inputs.csv"), X, "%.17g",
         "Latin hypercube, %d points, numpy default_rng(%d), inputs in physical units"
         % (m_rows, seed), cols)
    lad = []
    Y = run_rows(X, xfoil_names(bounds), exe, stem + "outputs", lad)
    save(os.path.join(HERE, stem + "outputs.csv"), Y, OUT_FMT,
         "XFOIL 6.99 outputs for %sinputs.csv (row by row); %s" % (stem, STATUS_NOTE),
         OUT_COLS)
    save_ladder(stem + "outputs.csv", lad)


def part_reference(exe):
    R = np.array(REFERENCE)
    lad = []
    Y = run_rows(R[:, 1:], ["alpha", "flap", "re"], exe, "reference_points", lad)
    save_ladder("reference_points.csv", lad)
    save(os.path.join(HERE, "reference_points.csv"), np.hstack([R, Y]),
         ["%d", "%.17g", "%.17g", "%.17g"] + OUT_FMT,
         "XFOIL 6.99 at the challenge's 7 calibration (set 0, Table 1) and 4 prediction "
         "(set 1, Fig. 1) points; " + STATUS_NOTE,
         "set,alpha (deg),flap (deg),Re (-)," + OUT_COLS)


def part_davis_rerun(exe):
    D = np.loadtxt(os.path.join(HERE, "davis_training_98.csv"), delimiter=",")
    X = D[:, [1, 2, 0]]                       # file order is re, alpha, flap
    lad = []
    Y = run_rows(X, ["alpha", "flap", "re"], exe, "davis_rerun_outputs", lad)
    save(os.path.join(HERE, "davis_rerun_outputs.csv"), Y, OUT_FMT,
         "XFOIL 6.99 (this build) at the 98 inputs of davis_training_98.csv, row by row; "
         + STATUS_NOTE, OUT_COLS)
    save_ladder("davis_rerun_outputs.csv", lad)


def part_re_sensitivity(exe):
    """The 7 calibration points of Table 1 at 5 Reynolds numbers (a labelled check)."""
    cal = [r for r in REFERENCE if r[0] == 0]
    X = np.array([[a, b, re] for re in RE_SENSITIVITY for (_, a, b, _) in cal])
    lad = []
    Y = run_rows(X, ["alpha", "flap", "re"], exe, "re_sensitivity", lad)
    save(os.path.join(HERE, "re_sensitivity.csv"), np.hstack([X, Y]),
         ["%.17g"] * 3 + OUT_FMT,
         "XFOIL 6.99 at the 7 calibration points (Table 1) for Re = 665,000 and 735,000 (the "
         "paper's printed range, p. 3), 696,500 and 703,500 (the code's range) and 700,000; "
         "a sensitivity check, not a design; " + STATUS_NOTE,
         "alpha (deg),flap (deg),Re (-)," + OUT_COLS)
    save_ladder("re_sensitivity.csv", lad)


def part_davis_copy(src):
    """Copy the authors' training file, adding two '#' lines; data text is not changed."""
    with open(src) as fh:
        lines = fh.read().splitlines()
    if lines[0].strip() != "re,alpha,flap,cl,cd,cm":
        raise ValueError("unexpected header in %s: %r" % (src, lines[0]))
    body = [ln for ln in lines[1:] if ln.strip()]
    dst = os.path.join(HERE, "davis_training_98.csv")
    with open(dst, "w") as fh:
        fh.write("# training/training_100.csv of github.com/csdlpsu/uqchallenge2 (commit "
                 "73eaec9), %d XFOIL rows written by the authors; values copied as text\n"
                 % len(body))
        fh.write("# Re (-),alpha (deg),flap (deg),CL (-),CD (-),CM (-)\n")
        fh.write("\n".join(body) + "\n")
    a = np.loadtxt(dst, delimiter=",")
    b = np.array([[float(v) for v in ln.split(",")] for ln in body])
    assert np.array_equal(a, b), "lossless copy check failed"
    print("  davis_training_98.csv: %d rows copied" % len(body))


def check(exe, nrows):
    """Rerun the first nrows rows of each design, plus every row that converged in the retry
    ladder (status 3 or 4), and compare every value. Rows that failed every attempt are not
    rerun (each takes up to a minute)."""
    ok = True
    cases = [("inputs.csv", "outputs.csv", ["alpha", "flap", "re"]),
             ("extension_inputs.csv", "extension_outputs.csv",
              ["alpha", "flap", "re", "ncrit", "xtr_top", "xtr_bot"])]
    for fin, fout, names in cases:
        Xa = np.loadtxt(os.path.join(HERE, fin), delimiter=",")
        Ya = np.loadtxt(os.path.join(HERE, fout), delimiter=",")
        idx = sorted(set(range(min(nrows, len(Xa)))) |
                     set(np.where(Ya[:, 6] >= 3)[0].tolist()))
        X, Y = Xa[idx], Ya[idx]
        Z = run_rows(X, names, exe, "check " + fout)
        same = np.array_equal(np.nan_to_num(Y, nan=-9.0), np.nan_to_num(Z, nan=-9.0))
        print("  %s: first %d rows and %d ladder rows %s" % (
            fout, min(nrows, len(Xa)), len(idx) - min(nrows, len(Xa)),
            "identical" if same else "DIFFER"))
        if not same:
            ok = False
            bad = np.where(np.any(np.nan_to_num(Y, nan=-9) != np.nan_to_num(Z, nan=-9), 1))[0]
            print("    rows differing:", [idx[b] for b in bad])
    X0 = latin_hypercube(M_PRIMARY, PRIMARY_BOUNDS, SEED_PRIMARY)
    Xs = np.loadtxt(os.path.join(HERE, "inputs.csv"), delimiter=",")
    print("  inputs.csv regenerated from the seed:", np.array_equal(X0, Xs))
    X1 = latin_hypercube(M_EXTENSION, EXTENSION_BOUNDS, SEED_EXTENSION)
    Xs = np.loadtxt(os.path.join(HERE, "extension_inputs.csv"), delimiter=",")
    print("  extension_inputs.csv regenerated from the seed:", np.array_equal(X1, Xs))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xfoil", default=os.environ.get("XFOIL_EXE", "xfoil"))
    p.add_argument("--part", default="all",
                   choices=["all", "primary", "extension", "reference", "davis_rerun",
                            "re_sensitivity"])
    p.add_argument("--check", type=int, default=0, metavar="N")
    p.add_argument("--davis-src", default=None,
                   help="copy the authors' training_100.csv to davis_training_98.csv")
    p.add_argument("--tmp", default=None, help="directory for XFOIL scratch runs")
    a = p.parse_args()
    if a.tmp:
        os.makedirs(a.tmp, exist_ok=True)
        tempfile.tempdir = a.tmp
    t0 = time.time()
    if a.davis_src:
        part_davis_copy(a.davis_src)
        if a.part == "all" and not a.check:
            print("copied; rerun without --davis-src for the XFOIL files")
            return
    if a.check:
        ok = check(a.xfoil, a.check)
        print("check %s in %.1f s" % ("passed" if ok else "FAILED", time.time() - t0))
        sys.exit(0 if ok else 1)
    if a.part in ("all", "primary"):
        part_design(PRIMARY_BOUNDS, M_PRIMARY, SEED_PRIMARY, "", a.xfoil)
    if a.part in ("all", "extension"):
        part_design(EXTENSION_BOUNDS, M_EXTENSION, SEED_EXTENSION, "extension_", a.xfoil)
    if a.part in ("all", "reference"):
        part_reference(a.xfoil)
    if a.part in ("all", "davis_rerun"):
        part_davis_rerun(a.xfoil)
    if a.part in ("all", "re_sensitivity"):
        part_re_sensitivity(a.xfoil)
    print("done in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
