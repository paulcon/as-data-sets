"""Generate the DTMB 5415 hull-resistance data with the L2-Sea code (an outside tool).

    python generate_data.py design --l2sea-home DIR --raw RAW --stop 384 [--max-minutes 19]
    python generate_data.py check  --l2sea-home DIR --raw RAW
    python generate_data.py reference --l2sea-home DIR --raw RAW
    python generate_data.py quadrature --l2sea-home DIR --raw RAW
    python generate_data.py sweep  --l2sea-home DIR --raw RAW [--max-minutes 17] [--check K]
    python generate_data.py shape  --l2sea-home DIR --raw RAW [--max-minutes 17] [--check K]
    python generate_data.py g6     --l2sea-home DIR --raw RAW [--max-minutes 17] [--check K]
    python generate_data.py diag   --l2sea-home DIR --raw RAW [--max-minutes 17] [--check K]
    python generate_data.py assemble --raw RAW --n 384 [--out .]

`design` runs points of the design at fidelity 7 and appends one JSON line per run to
RAW/design_f7.jsonl. Points already in that file are skipped, so the step can be run in
chunks (`--max-minutes` stops it cleanly between runs). The design is the first N points
of a 16-dimensional scrambled Sobol' sequence (scipy.stats.qmc.Sobol, scramble=True,
seed=SOBOL_SEED), mapped linearly to the box LOWER..UPPER. Each run is deterministic, so
the order and the chunking do not change the numbers.

`check` reruns CHECK_ROWS of the design at the finer fidelities in CHECK_FIDELITIES (and the
(row, fidelity) pairs in EXTRA_CHECKS), and
`reference` runs the paper's reference points (the parent hull at Fr = 0.28, T = -6.16
at several fidelities, and the paper's optimized hull, App. A.4.1). Both append to
RAW/check.jsonl and RAW/reference.jsonl.

`quadrature` runs the parent hull on a tensor Gauss-Legendre grid in (Fr, T) (GL_NODES
nodes in each) at fidelity 6, for the paper's second reference value of App. A.3.3 (E[R]
with Fr and T uniform, fidelity 6). It appends to RAW/quadrature.jsonl.

Four further steps, each resumable and stopped cleanly by --max-minutes:
`sweep` runs the parent hull at closely spaced drafts (DRAFT_SWEEPS; RAW/draft_sweep.jsonl),
`shape` the shape-only study (SHAPE_N Sobol' points in s at Fr = 0.28, T = -6.16, fidelity 7;
RAW/shape_f7.jsonl), `g6` reruns rows G6_ROWS of the design on G6 (RAW/check.jsonl), and `diag`
reruns DIAG_ROWS on G7 to record the wave-cut and ITTC diagnostics (RAW/design_diag_f7.jsonl).
With `--check K` each of them instead reruns K evenly spaced stored runs and prints the largest
relative difference from the stored outputs (0 when reproduced exactly).

`assemble` writes inputs.csv, outputs.csv, fidelity_check.csv, reference_runs.csv and
uq_uniform_quadrature.csv from the JSON lines, and, when their raw files exist, draft_sweep.csv,
shape_inputs.csv, shape_outputs.csv and design_diagnostics.csv. A run that raised is written as a
row of NaN outputs; runs that returned numbers are written as printed, also those that fail the
per-run rule of l2sea_driver.failure_reason (DATA.md, Accuracy).

DIR is a built copy of github.com/MAORG-CNR-INM/NATO-AVT-331-L2-Sea-Benchmark (GPL-3.0) at
commit 100da62, with bin/L2-Sea compiled by `make` (DATA.md, Regenerating); it is given by
--l2sea-home or the environment variable L2SEA_HOME. Without it every step but `assemble`
stops with an error that says what to build (exit status 1). `assemble` needs the RAW files
that the other steps write. The shipped files are the record: rerun L2-Sea only if a problem is
found in them. Imports: numpy, scipy and the standard library, plus l2sea_driver.py beside this
file.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2sea_driver  # noqa: E402

NAMES = ["Fr", "T"] + ["s%d" % k for k in range(1, 15)]
#: Bounds from the paper, App. A.1.2 (pp. 18-19), and the UM-Bridge model page.
LOWER = np.array([0.25, -6.776] + [-1.0] * 14)
UPPER = np.array([0.41, -5.544] + [1.0] * 14)
NOMINAL = np.array([0.28, -6.16] + [0.0] * 14)

SOBOL_SEED = 5415
N_SOBOL = 512          # generated; the shipped data use the first N of them
FIDELITY = 7           # UM-Bridge default (models/l2-sea/umbridge-server.py)
CHECK_ROWS = list(range(6))
CHECK_FIDELITIES = [3]
#: Extra (row, fidelity) reruns. Row 63 has an anomalous frictional resistance at fidelity 7
#: (R_f = 354 N, about 14 times the flat-plate value); it is rerun on G7 (determinism), G6 and G5.
#: Rows 3, 71, 189, 207, 305, 323, 343 have friction coefficients that stand out from the
#: others (DATA.md, Accuracy) and rows 0, 1, 2, 4, 5 are ordinary rows; all are rerun on G6.
EXTRA_CHECKS = ([(63, 7), (63, 6), (63, 5)]
                + [(i, 6) for i in (3, 71, 189, 207, 305, 323, 343)]
                + [(i, 6) for i in (0, 1, 2, 4, 5)])

#: The paper's optimized hull (App. A.4.1, p. 33; R = 36.3116 N at fidelity 1).
S_OPT = [-1.000000, -0.906250, -0.781250, -0.242188, 0.533203, -0.906250, 0.128418,
         0.812500, -0.070313, -0.878296, 0.422363, 0.031250, 0.365234, -0.175659]
REFERENCE = ([("parent", f, [0.28, -6.16] + [0.0] * 14) for f in (7, 6, 5, 4, 3, 1)]
             + [("optimum", f, [0.28, -6.16] + S_OPT) for f in (7, 1)])

#: Reruns of a full 128-point Sobol' net of the design (rows 0-127) on G6, for the grid check of
#: the subspace. Pairs already in RAW/check.jsonl are not rerun.
G6_ROWS = list(range(128))

#: Rows of the design rerun on G7 to record the two diagnostics (wave-cut wave resistance, ITTC
#: friction) that outputs.csv lacks: every 12th row, the 7 flagged rows and rows 71 and 101 (in the
#: draft band but valid). Each rerun must reproduce outputs.csv exactly (`assemble` checks it).
DIAG_ROWS = sorted(set(range(0, 384, 12)) | {3, 63, 189, 207, 305, 323, 343, 71, 101})

#: Draft sweeps at the parent hull (s = 0) mapping the band of silent G7 failures:
#: (fidelity, Fr, T values).
DRAFT_SWEEPS = ([(7, 0.28, T) for T in np.round(np.linspace(-5.70, -5.58, 49), 6).tolist()]
                + [(7, 0.36, T) for T in np.round(np.linspace(-5.70, -5.58, 25), 6).tolist()]
                + [(6, 0.28, T) for T in np.round(np.linspace(-5.70, -5.58, 25), 6).tolist()])

#: Shape-only study: the 14 shape coefficients on [-1, 1]^14 at the
#: optimization benchmark's Fr = 0.28 and T = -6.16 (App. A.4.1), fidelity 7; the first
#: SHAPE_N points of a 14-dimensional scrambled Sobol' sequence (seed SHAPE_SEED).
SHAPE_SEED = 14
SHAPE_N = 256
SHAPE_FR, SHAPE_T = 0.28, -6.16

GL_NODES = (6, 3)      # Gauss-Legendre nodes in Fr and in T
GL_FIDELITY = 6

OUT_NAMES = [n for n in l2sea_driver.OUTPUT_NAMES if n != "cpu_s"]
OUT_UNITS = {"R_T": "N", "g1": "-", "g2": "-", "g3": "-", "g4": "-", "R_w": "N",
             "R_f": "N", "S_wet": "m^2", "displacement": "m^3"}


def design_points(n=N_SOBOL):
    u = qmc.Sobol(d=16, scramble=True, seed=SOBOL_SEED).random(n)
    return LOWER + (UPPER - LOWER) * u


def shape_points(n=SHAPE_N):
    u = qmc.Sobol(d=14, scramble=True, seed=SHAPE_SEED).random(n)
    S = -1.0 + 2.0 * u
    return np.hstack([np.full((n, 1), SHAPE_FR), np.full((n, 1), SHAPE_T), S])


def _check_subset(path, k, home):
    """Rerun k evenly spaced stored records of RAW file `path`; print the largest relative
    difference over the nine outputs (0 means the stored values are reproduced exactly)."""
    recs = [r for r in _read(path) if r["out"]]
    pick = [recs[j] for j in np.unique(np.linspace(0, len(recs) - 1, k).round().astype(int))]
    worst = 0.0
    for r in pick:
        new = l2sea_driver.run(r["x"], home, fidelity=r["fidelity"])
        d = max(abs(new[n] - r["out"][n]) / max(abs(r["out"][n]), 1e-12) for n in OUT_NAMES)
        worst = max(worst, d)
        print("check", os.path.basename(path), r.get("i", r.get("T")), r["fidelity"],
              "max rel diff %.3g" % d, flush=True)
    print("check subset of %d records: max relative difference %.3g" % (len(pick), worst))


def _loop(path, todo, key, home, max_minutes):
    """Run (key, x, fidelity) items not yet in `path`, stopping cleanly at the time budget."""
    done = _done(path, key)
    t0 = time.time()
    for kv, x, f, extra in todo:
        if kv in done:
            continue
        if (time.time() - t0) / 60 > max_minutes:
            print("time budget reached before %r" % (kv,))
            return
        rec = dict(extra, **_one(x, home, f))
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(kv, rec["out"]["R_T"] if rec["out"] else rec["error"], rec["wall_s"], flush=True)


def cmd_g6(a):
    path = os.path.join(a.raw, "check.jsonl")
    if a.check:
        recs = [r for r in _read(path) if r["fidelity"] == 6 and r["i"] in G6_ROWS]
        tmp = os.path.join(a.raw, "_g6_subset.jsonl")
        with open(tmp, "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
        return _check_subset(tmp, a.check, a.l2sea_home)
    X = design_points()
    todo = [((i, 6), X[i], 6, {"i": i}) for i in G6_ROWS]
    _loop(path, todo, lambda r: (r["i"], r["fidelity"]), a.l2sea_home, a.max_minutes)


def cmd_diag(a):
    path = os.path.join(a.raw, "design_diag_f7.jsonl")
    if a.check:
        return _check_subset(path, a.check, a.l2sea_home)
    X = design_points()
    todo = [(i, X[i], FIDELITY, {"i": i}) for i in DIAG_ROWS]
    _loop(path, todo, lambda r: r["i"], a.l2sea_home, a.max_minutes)


def cmd_sweep(a):
    path = os.path.join(a.raw, "draft_sweep.jsonl")
    if a.check:
        return _check_subset(path, a.check, a.l2sea_home)
    todo = [((f, fr, t), [fr, t] + [0.0] * 14, f, {"Fr": fr, "T": t}) for f, fr, t in DRAFT_SWEEPS]
    _loop(path, todo, lambda r: (r["fidelity"], r["Fr"], r["T"]), a.l2sea_home, a.max_minutes)


def cmd_shape(a):
    path = os.path.join(a.raw, "shape_f7.jsonl")
    if a.check:
        return _check_subset(path, a.check, a.l2sea_home)
    P = shape_points()
    todo = [(i, P[i], FIDELITY, {"i": i}) for i in range(a.start, min(a.stop, SHAPE_N))]
    _loop(path, todo, lambda r: r["i"], a.l2sea_home, a.max_minutes)


def _done(path, key):
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {key(json.loads(line)) for line in fh if line.strip()}


def _one(x, home, fidelity):
    t = time.time()
    try:
        out = l2sea_driver.run(x, home, fidelity=fidelity)
        err = None
    except Exception as e:  # recorded, never silently dropped
        out, err = None, "%s: %s" % (type(e).__name__, e)
    return {"x": [float(v) for v in x], "fidelity": fidelity, "out": out, "error": err,
            "wall_s": round(time.time() - t, 2), "date": time.strftime("%Y-%m-%dT%H:%M:%S")}


def cmd_design(a):
    X = design_points()
    path = os.path.join(a.raw, "design_f%d.jsonl" % FIDELITY)
    done = _done(path, lambda r: r["i"])
    t0 = time.time()
    for i in range(a.start, a.stop):
        if i in done:
            continue
        if (time.time() - t0) / 60 > a.max_minutes:
            print("time budget reached before row %d" % i)
            break
        rec = dict(i=i, **_one(X[i], a.l2sea_home, FIDELITY))
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(i, rec["out"]["R_T"] if rec["out"] else rec["error"], rec["wall_s"], flush=True)


def cmd_check(a):
    X = design_points()
    path = os.path.join(a.raw, "check.jsonl")
    done = _done(path, lambda r: (r["i"], r["fidelity"]))
    todo = [(i, f) for f in CHECK_FIDELITIES for i in CHECK_ROWS] + EXTRA_CHECKS
    for i, f in todo:
        if (i, f) in done:
            continue
        rec = dict(i=i, **_one(X[i], a.l2sea_home, f))
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(i, f, rec["out"]["R_T"] if rec["out"] else rec["error"], rec["wall_s"],
              flush=True)


def cmd_reference(a):
    path = os.path.join(a.raw, "reference.jsonl")
    done = _done(path, lambda r: (r["name"], r["fidelity"]))
    for name, f, x in REFERENCE:
        if (name, f) in done or (a.only and name not in a.only):
            continue
        if a.fidelities and f not in a.fidelities:
            continue
        rec = dict(name=name, **_one(x, a.l2sea_home, f))
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(name, f, rec["out"]["R_T"] if rec["out"] else rec["error"], rec["wall_s"],
              flush=True)


def gl_grid():
    """Tensor Gauss-Legendre nodes and weights (summing to 1) on the (Fr, T) box."""
    pts, wts = [], []
    for j, n in enumerate(GL_NODES):
        z, w = np.polynomial.legendre.leggauss(n)
        pts.append(LOWER[j] + (UPPER[j] - LOWER[j]) * (z + 1) / 2)
        wts.append(w / 2)
    P = np.array([[a, b] for a in pts[0] for b in pts[1]])
    W = np.array([a * b for a in wts[0] for b in wts[1]])
    return P, W


def cmd_quadrature(a):
    P, _ = gl_grid()
    path = os.path.join(a.raw, "quadrature.jsonl")
    done = _done(path, lambda r: r["i"])
    for i, (fr, t) in enumerate(P):
        if i in done:
            continue
        rec = dict(i=i, **_one([fr, t] + [0.0] * 14, a.l2sea_home, GL_FIDELITY))
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(i, fr, t, rec["out"]["R_T"] if rec["out"] else rec["error"], rec["wall_s"],
              flush=True)


def _read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _outrow(r, names=None):
    names = OUT_NAMES if names is None else names
    return [r["out"][k] if r["out"] else np.nan for k in names]


DIAG = list(l2sea_driver.DIAG_NAMES)


def _save(path, A, desc, cols, fmt):
    header = desc + "\n" + ",".join(cols)
    np.savetxt(path, A, delimiter=",", header=header, comments="# ", fmt=fmt)


def cmd_assemble(a):
    X = design_points()
    recs = {r["i"]: r for r in _read(os.path.join(a.raw, "design_f%d.jsonl" % FIDELITY))}
    missing = [i for i in range(a.n) if i not in recs]
    if missing:
        sys.exit("design rows not run yet: %d (first %d)" % (len(missing), missing[0]))
    for i in range(a.n):
        if np.max(np.abs(np.array(recs[i]["x"]) - X[i])) != 0.0:
            sys.exit("row %d: stored input differs from the design" % i)
    src = ("L2-Sea (CNR-INM, GPL-3.0) built from MAORG-CNR-INM/NATO-AVT-331-L2-Sea-Benchmark"
           " @100da62; fidelity %d, sinkoff=trimoff='y'" % FIDELITY)
    _save(os.path.join(a.out, "inputs.csv"), X[:a.n],
          "DTMB 5415 L2-Sea design: first %d points of a scrambled Sobol' sequence (scipy "
          "qmc.Sobol, seed %d) on the box of DATA.md; physical units" % (a.n, SOBOL_SEED),
          ["Fr (-)", "T (m full scale; ztrasla)"] + ["s%d (-)" % k for k in range(1, 15)],
          "%.17g")
    _save(os.path.join(a.out, "outputs.csv"), np.array([_outrow(recs[i]) for i in range(a.n)]),
          "DTMB 5415 calm-water resistance at the rows of inputs.csv; " + src,
          ["%s (%s)" % (k, OUT_UNITS[k]) for k in OUT_NAMES], "%.9g")
    chk = _read(os.path.join(a.raw, "check.jsonl"))
    rows = [[r["i"], r["fidelity"]] + _outrow(r) + [recs[r["i"]]["out"]["R_T"]] for r in chk]
    _save(os.path.join(a.out, "fidelity_check.csv"), np.array(rows),
          "Rows of inputs.csv rerun at the L2-Sea fidelity in column 2 (the first six on G3, then EXTRA_CHECKS, then the rest of rows 0-127 on G6); last column is R_T at "
          "fidelity %d from outputs.csv" % FIDELITY,
          ["row (-)", "fidelity (-)"] + ["%s (%s)" % (k, OUT_UNITS[k]) for k in OUT_NAMES]
          + ["R_T_f%d (N)" % FIDELITY], "%.9g")
    ref = _read(os.path.join(a.raw, "reference.jsonl"))
    order = {(n, f): j for j, (n, f, _) in enumerate(REFERENCE)}
    ref.sort(key=lambda r: order[(r["name"], r["fidelity"])])
    rows = [[0 if r["name"] == "parent" else 1, r["fidelity"]] + r["x"] + _outrow(r)
            + [r["out"]["cpu_s"] if r["out"] else np.nan, r["wall_s"]] for r in ref]
    _save(os.path.join(a.out, "reference_runs.csv"), np.array(rows),
          "L2-Sea reference runs: hull 0 = parent DTMB 5415, 1 = optimized hull of the paper"
          " (App. A.4.1); cpu_s = the code's own time report (a tenth of the wall time under gfortran;"
          " DATA.md); wall_s on a shared 2-CPU container",
          ["hull (-)", "fidelity (-)"] + NAMES + ["%s (%s)" % (k, OUT_UNITS[k]) for k in OUT_NAMES]
          + ["cpu_s (s)", "wall_s (s)"], "%.9g")
    P, W = gl_grid()
    qd = {r["i"]: r for r in _read(os.path.join(a.raw, "quadrature.jsonl"))}
    rows = [[P[i, 0], P[i, 1], W[i], qd[i]["out"]["R_T"] if qd[i]["out"] else np.nan]
            for i in range(len(P))]
    _save(os.path.join(a.out, "uq_uniform_quadrature.csv"), np.array(rows),
          "Parent hull (s = 0) on a %dx%d tensor Gauss-Legendre grid in (Fr, T), L2-Sea fidelity "
          "%d; weights sum to 1 (uniform density)" % (GL_NODES + (GL_FIDELITY,)),
          ["Fr (-)", "T (m)", "weight (-)", "R_T (N)"], "%.17g")
    _assemble_new(a)
    walls = [recs[i]["wall_s"] for i in range(a.n)]
    cpus = [recs[i]["out"]["cpu_s"] for i in range(a.n) if recs[i]["out"]]
    nfail = sum(recs[i]["out"] is None for i in range(a.n))
    print("assembled %d design rows (%d failed); wall %.0f s total (median %.1f s), code CPU "
          "median %.2f s; %d check runs, %d reference runs"
          % (a.n, nfail, sum(walls), np.median(walls), np.median(cpus), len(chk), len(ref)))


def _assemble_new(a):
    """design_diagnostics.csv, draft_sweep.csv, shape_inputs.csv and shape_outputs.csv."""
    cols = ["%s (%s)" % (k, OUT_UNITS[k]) for k in OUT_NAMES] + ["Rw_wc (N)", "Rf_ittc (N)"]
    p = os.path.join(a.raw, "design_diag_f7.jsonl")
    if os.path.exists(p):
        recs = {r["i"]: r for r in _read(os.path.join(a.raw, "design_f%d.jsonl" % FIDELITY))}
        dg = sorted(_read(p), key=lambda r: r["i"])
        for r in dg:
            if _outrow(r) != _outrow(recs[r["i"]]):
                sys.exit("design_diag row %d does not reproduce outputs.csv" % r["i"])
        _save(os.path.join(a.out, "design_diagnostics.csv"),
              np.array([[r["i"]] + _outrow(r, DIAG) for r in dg]),
              "Rows of inputs.csv rerun on G%d to record two diagnostics the code prints "
              "(force-dyn.out); the nine outputs of each rerun equal outputs.csv exactly" % FIDELITY,
              ["row (-)", "Rw_wc (N)", "Rf_ittc (N)"], "%.9g")
    p = os.path.join(a.raw, "draft_sweep.jsonl")
    if os.path.exists(p):
        sw = {(r["fidelity"], r["Fr"], r["T"]): r for r in _read(p)}
        rows = [[f, fr, t] + _outrow(sw[(f, fr, t)], OUT_NAMES + DIAG)
                for f, fr, t in DRAFT_SWEEPS if (f, fr, t) in sw]
        _save(os.path.join(a.out, "draft_sweep.csv"), np.array(rows),
              "Parent hull (s = 0) draft sweeps at fidelity 7 (Fr 0.28 and 0.36) and 6 (Fr 0.28); "
              "L2-Sea @100da62, sinkoff=trimoff='y'; outputs as the code printed them (no row "
              "removed; see DATA.md for the failure rule)",
              ["fidelity (-)", "Fr (-)", "T (m full scale; ztrasla)"] + cols, "%.9g")
    p = os.path.join(a.raw, "shape_f7.jsonl")
    if os.path.exists(p):
        sh = {r["i"]: r for r in _read(p)}
        n = len(sh)
        if sorted(sh) != list(range(n)):
            sys.exit("shape rows missing")
        P = shape_points()
        for i in range(n):
            if np.max(np.abs(np.array(sh[i]["x"]) - P[i])) != 0.0:
                sys.exit("shape row %d: stored input differs from the design" % i)
        _save(os.path.join(a.out, "shape_inputs.csv"), P[:n, 2:],
              "DTMB 5415 shape-only study: first %d points of a 14-D scrambled Sobol' sequence "
              "(scipy qmc.Sobol, seed %d) on [-1, 1]^14; Fr = %.2f, T = %.2f m, fidelity %d held"
              % (n, SHAPE_SEED, SHAPE_FR, SHAPE_T, FIDELITY),
              ["s%d (-)" % k for k in range(1, 15)], "%.17g")
        _save(os.path.join(a.out, "shape_outputs.csv"),
              np.array([_outrow(sh[i], OUT_NAMES + DIAG) for i in range(n)]),
              "DTMB 5415 shape-only study outputs at the rows of shape_inputs.csv; L2-Sea "
              "@100da62, fidelity %d, sinkoff=trimoff='y'" % FIDELITY, cols, "%.9g")


INSTALL_MSG = (
    "L2-Sea not found%s. Build it (gfortran, LAPACK and BLAS): git clone "
    "https://github.com/MAORG-CNR-INM/NATO-AVT-331-L2-Sea-Benchmark.git L2SEA; cd L2SEA; "
    "git checkout 100da62; mkdir -p lib; make (DATA.md, Regenerating, gives the details), then rerun "
    "generate_data.py with --l2sea-home L2SEA (or set L2SEA_HOME).")


def require_l2sea(home):
    """Exit with status 1 and an install message unless HOME holds a built L2-Sea."""
    if not home:
        sys.exit(INSTALL_MSG % " (no --l2sea-home given and L2SEA_HOME not set)")
    exe = os.path.join(home, "bin", "L2-Sea")
    case = os.path.join(home, "examples", "DTMB-5415")
    if not (os.path.isfile(exe) and os.access(exe, os.X_OK) and os.path.isdir(case)):
        sys.exit(INSTALL_MSG % (" in %s (no executable bin/L2-Sea or no examples/DTMB-5415)" % home))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    for name in ("design", "check", "reference", "quadrature", "g6", "sweep", "shape", "diag"):
        q = sub.add_parser(name)
        q.add_argument("--l2sea-home", default=os.environ.get("L2SEA_HOME"))
        q.add_argument("--raw", required=True)
        if name in ("g6", "sweep", "shape", "diag"):
            q.add_argument("--max-minutes", type=float, default=1e9)
            q.add_argument("--check", type=int, default=0,
                           help="rerun this many stored runs and compare, instead of running new ones")
        if name == "shape":
            q.add_argument("--start", type=int, default=0)
            q.add_argument("--stop", type=int, default=SHAPE_N)
        if name == "design":
            q.add_argument("--start", type=int, default=0)
            q.add_argument("--stop", type=int, default=384)
            q.add_argument("--max-minutes", type=float, default=1e9)
        if name == "reference":
            q.add_argument("--only", nargs="*")
            q.add_argument("--fidelities", nargs="*", type=int)
    q = sub.add_parser("assemble")
    q.add_argument("--raw", required=True)
    q.add_argument("--n", type=int, default=384)
    q.add_argument("--out", default=HERE)
    a = p.parse_args()
    if a.cmd is None:
        require_l2sea(os.environ.get("L2SEA_HOME"))
        p.print_help()
        sys.exit(2)
    if a.cmd != "assemble":
        require_l2sea(a.l2sea_home)
        os.makedirs(a.raw, exist_ok=True)
    else:
        need = [os.path.join(a.raw, f) for f in ("design_f%d.jsonl" % FIDELITY, "check.jsonl",
                                                  "reference.jsonl", "quadrature.jsonl")]
        if not all(os.path.exists(f) for f in need):
            sys.exit("assemble needs the RAW files (%s) that the other steps write with L2-Sea; run "
                     "them first (DATA.md, Regenerating)." % ", ".join(os.path.basename(f) for f in need))
    {"design": cmd_design, "check": cmd_check, "reference": cmd_reference,
     "quadrature": cmd_quadrature, "g6": cmd_g6, "sweep": cmd_sweep, "diag": cmd_diag, "shape": cmd_shape,
     "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
