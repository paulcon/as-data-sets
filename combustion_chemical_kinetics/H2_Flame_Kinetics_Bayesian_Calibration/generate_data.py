"""Regenerate the data files of H2_Flame_Kinetics_Bayesian_Calibration (needs Cantera 3.2.0).

    python generate_data.py --out DIR                        # all 500 rows, then the CSVs in DIR
    python generate_data.py --out DIR --rows 0:40            # rows 0..39 only (resumable chunks)
    python generate_data.py --out DIR --procs 16             # rows in 16 worker processes
    python generate_data.py --out DIR --qoi Su1              # one QoI only (tau, Su1, Su20 or a list)
    python generate_data.py --out DIR --rows 0:3 --check     # recompute 3 rows, compare with shipped
    python generate_data.py --out DIR --nominal              # nominal.csv, nominal_gradients.csv

Design: M = 500 Monte Carlo draws (seed 2019452) from the prior of Bell et al. (2019), Table 1:
31 independent Gaussians, mean = sd = the Table 1 value, each truncated to its bounds L:U
(inverse-CDF transform of uniforms; h2_flame_driver.sample_prior). Inputs are rounded to the
digits written (%.15e) before any run, so every output belongs to its stored input row.

Per row: the ignition delay tau with central-difference gradients (step 1e-3 of each prior sd);
the flame speeds Su1 and Su20, each with central finite differences on the fixed grid of its
flame (FlameSpeed.fd_grad: step 1e-3 of each prior sd, at most 2 % of an A factor). Cantera's
discrete adjoint is not used for the stored gradients: for these flames its scale is wrong (too
small by a factor of about 2 to 8 for Su20, a few per cent off for Su1; DATA.md, "Accuracy"). A flame that fails twice is stored as NaN with
status 2, never dropped; an ignition that fails is stored as NaN with status 2.

Each row and QoI is saved as DIR/rows/<qoi>/row_NNNN.npz as soon as it is done, and files already
there are skipped, so an interrupted run resumes where it stopped. --procs N runs the rows in N
worker processes. CPU per row: tau a few seconds, Su1 about 5 s, Su20 about a minute on average
(up to several minutes). When every row of a QoI exists, its gradients_<qoi>.csv is written to
DIR, with a summary of the finite-difference checks for Su1 and Su20; when every row of all three
QoIs exists, inputs.csv and outputs.csv as well. Progress is appended to DIR/generate.log.

Guard: --out DIR must be given and must lie outside this directory, so the shipped files are
never overwritten (copy the new files in by hand after comparing them). --check compares with a
stated tolerance (relative 1e-6 for values, 3e-4 of the largest sd-scaled stored entry for
gradients) and writes nothing but DIR/rows and DIR/generate.log.
"""
import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h2_flame_driver as D  # noqa: E402

SEED = 2019452
M = 500
QOIS = ["tau", "Su1", "Su20"]
UNITS = {"tau": "s", "Su1": "m/s", "Su20": "m/s"}
TOL_VALUE, TOL_GRAD = 1e-6, 3e-4
HOW = {"tau": "central differences, step 1e-3 of each prior sd",
       "Su1": "central finite differences on the fixed grid of each flame, step 1e-3 of each "
              "prior sd and at most 2 % of an A factor (FlameSpeed.fd_grad)",
       "Su20": "central finite differences on the fixed grid of each flame, step 1e-3 of each "
               "prior sd and at most 2 % of an A factor (FlameSpeed.fd_grad)"}
FD_KEYS = ("su_base", "grad_central10", "grad_forward", "flag", "drift", "h", "base_mode", "grid_match")


def design():
    X = D.sample_prior(M, np.random.default_rng(SEED))
    return np.array([[float("%.15e" % v) for v in row] for row in X])


class Runner:
    def __init__(self):
        self.ign = D.IgnitionDelay()
        self.fl = {c: D.FlameSpeed(c) for c in ("Su1", "Su20")}

    def qoi(self, q, x):
        """One QoI at x: a dict with the value, its status, its gradient and, for a flame, the
        number of grid points and fd_grad's diagnostics."""
        if q == "tau":
            try:
                return {"value": self.ign.tau(x), "grad": self.ign.grad(x), "status": D.STATUS_OK}
            except RuntimeError:
                return {"value": np.nan, "grad": np.full(D.M_INPUTS, np.nan), "status": D.STATUS_FAILED}
        r = self.fl[q].fd_grad(x)
        out = {"value": r["su_auto"], "status": r["status"], "grad": r["grad"],
               "n_points": r.get("n_points", 0)}
        out.update({"fd_" + k: r[k] for k in FD_KEYS if k in r})
        return out


def log(out, msg):
    with open(os.path.join(out, "generate.log"), "a") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


def row_file(out, q, i):
    return os.path.join(out, "rows", q, "row_%04d.npz" % i)


_RUNNER = None


def _one(args):
    """Compute one row of one QoI and save it (in a worker process or in the main one)."""
    global _RUNNER
    i, q, x, fn = args
    if _RUNNER is None:
        _RUNNER = Runner()
    t0 = time.process_time()
    r = _RUNNER.qoi(q, x)
    cpu = time.process_time() - t0
    tmp = fn + ".tmp.npz"
    np.savez(tmp, x=x, cpu=cpu, **{k: np.asarray(v) for k, v in r.items()})
    os.replace(tmp, fn)
    extra = ", %d points" % r["n_points"] if "n_points" in r else ""
    return "row %d %s: %.6g %s (status %d%s), %.1f s CPU" % (i, q, r["value"], UNITS[q], r["status"],
                                                            extra, cpu)


def run_rows(rows, qois, out, X, procs=1):
    for q in qois:
        os.makedirs(os.path.join(out, "rows", q), exist_ok=True)
    todo = [(i, q, X[i], row_file(out, q, i)) for i in rows for q in qois
            if not os.path.exists(row_file(out, q, i))]
    if procs <= 1:
        for t in todo:
            log(out, _one(t))
        return
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs, maxtasksperchild=25) as pool:
        for msg in pool.imap_unordered(_one, todo):
            log(out, msg)


def collect(q, rows, out):
    return [np.load(row_file(out, q, i)) for i in rows]


def rel2(a, b):
    """Relative 2-norm difference of two gradients in sd-scaled coordinates."""
    s = D.PRIOR_SD
    return np.linalg.norm((a - b) * s) / np.linalg.norm(b * s)


def fd_summary(q, R):
    """The finite-difference checks of fd_grad over the rows of a flame QoI (R: row files 0..M-1)."""
    idx = [i for i, r in enumerate(R) if "fd_grad_central10" in r.files and np.all(np.isfinite(r["grad"]))]
    ok = [R[i] for i in idx]
    if not ok:
        return "%s: no finite-difference diagnostics" % q
    step = np.array([rel2(r["fd_grad_central10"], r["grad"]) for r in ok])
    fwd = np.array([rel2(r["fd_grad_forward"], r["grad"]) for r in ok])
    drift = max(float(np.max(np.abs(r["fd_drift"]))) for r in ok)
    grid = sum(int(r["fd_grid_match"]) for r in ok)
    fl = np.concatenate([r["fd_flag"] for r in ok])
    base = np.array([r["fd_su_base"] / r["value"] - 1 for r in ok])
    return ("%s finite differences over %d rows: step check (central with a 10x step vs central) median "
            "%.1e, max %.1e (row %d); forward vs central median %.1e, max %.1e; largest drift back to the "
            "base state %.1e; base solve on the identical grid in %d of %d rows; tightened base vs value "
            "largest relative difference %.1e; flags: %d forward-only, %d failed"
            % (q, len(ok), np.median(step), step.max(), idx[int(np.argmax(step))], np.median(fwd), fwd.max(),
               drift, grid, len(ok), np.max(np.abs(base)), int(np.sum(fl == 1)), int(np.sum(fl == 3))))


def write_csvs(qois, out, X):
    cols = [f"{n} [{u}]" for n, u in zip(D.NAMES, D.UNITS)]
    for q in qois:
        R = collect(q, range(M), out)
        if not np.array_equal(np.array([r["x"] for r in R]), X):
            sys.exit("stored row inputs differ from the design; delete DIR/rows and rerun")
        G = np.array([r["grad"] for r in R])
        np.savetxt(os.path.join(out, f"gradients_{q}.csv"), G, fmt="%.10e", delimiter=",",
                   header=f"d{q}/dx_j in physical units ({HOW[q]}); NaN where the solve failed\n"
                   + ", ".join(f"d{q}/d{n} [{UNITS[q]} per {u}]" for n, u in zip(D.NAMES, D.UNITS)))
        log(out, "wrote gradients_%s.csv; %.0f s CPU for %s" % (q, sum(float(r["cpu"]) for r in R), q))
        if q != "tau":
            log(out, fd_summary(q, R))
    if not all(os.path.exists(row_file(out, q, i)) for q in QOIS for i in range(M)):
        return
    R = {q: collect(q, range(M), out) for q in QOIS}
    h = "Bell et al. (2019) H2 kinetics prior, Monte Carlo, seed %d, rows 0:%d; physical units" % (SEED, M)
    np.savetxt(os.path.join(out, "inputs.csv"), X, fmt="%.15e", delimiter=",", header=h + "\n" + ", ".join(cols))
    F = np.array([[R[q][i]["value"] for q in QOIS] + [R[q][i]["status"] for q in QOIS] for i in range(M)])
    np.savetxt(os.path.join(out, "outputs.csv"), F, fmt="%.15e", delimiter=",", header=(
        "tau: constant-volume ignition delay (T0 1100 K, 2 atm, H2:O2:Ar 2:1:97); Su1: H2/air "
        "flame speed (phi 1, 1 atm, 298 K); Su20: H2/air flame speed (phi 0.4, 20 atm, 298 K); "
        "status 0 ok, 1 ok on retry, 2 failed (value NaN); Cantera 3.2.0, burke2012_x1x6.yaml\n"
        "tau [s], Su1 [m/s], Su20 [m/s], status_tau [-], status_Su1 [-], status_Su20 [-]"))
    log(out, "wrote inputs.csv and outputs.csv")


def write_nominal(out):
    runner = Runner()
    r = {q: runner.qoi(q, D.NOMINAL) for q in QOIS}
    vals = np.array([[r[q]["value"] for q in QOIS]])
    G = np.array([r[q]["grad"] for q in QOIS])
    np.savetxt(os.path.join(out, "nominal.csv"), vals, fmt="%.15e", delimiter=",", header=(
                   "the three QoIs at the Table 1 values (Burke et al. and You et al.); Cantera 3.2.0\n"
                   "tau [s], Su1 [m/s], Su20 [m/s]"))
    np.savetxt(os.path.join(out, "nominal_gradients.csv"), G, fmt="%.10e", delimiter=",", header=(
        "rows tau, Su1, Su20: gradients at the Table 1 values, as in gradients_*.csv\n"
        + ", ".join(f"d/d{n} [per {u}]" for n, u in zip(D.NAMES, D.UNITS))))
    log(out, "wrote nominal.csv, nominal_gradients.csv")


def check(rows, qois, out):
    ld = lambda n: np.loadtxt(os.path.join(HERE, n), delimiter=",", ndmin=2)  # noqa: E731
    Xs, Fs = ld("inputs.csv")[rows], ld("outputs.csv")[rows]
    for q in qois:
        j = QOIS.index(q)
        R = collect(q, range(rows.start, rows.stop), out)
        dx = np.max(np.abs(np.array([r["x"] for r in R]) / Xs - 1)[Xs != 0], initial=0.0)
        new = np.array([float(r["value"]) for r in R])
        st = np.array([int(r["status"]) for r in R])
        rel = np.nanmax(np.abs(new / Fs[:, j] - 1))
        Gs = ld(f"gradients_{q}.csv")[rows]
        Gn = np.array([r["grad"] for r in R])
        sc = D.PRIOR_SD
        grel = np.nanmax(np.abs((Gn - Gs) * sc)) / np.nanmax(np.abs(Gs * sc))
        same = np.array_equal(st, Fs[:, 3 + j])
        ok = dx <= 1e-12 and rel <= TOL_VALUE and grel <= TOL_GRAD and same
        print(f"{q}: inputs max relative difference {dx:.1e}; max relative value difference {rel:.1e} (tol {TOL_VALUE:g}); "
              f"gradient difference {grel:.1e} of the largest sd-scaled entry (tol {TOL_GRAD:g}); status "
              f"equal {same}: {'PASS' if ok else 'FAIL'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory, outside this directory")
    p.add_argument("--rows", default=None, help="a:b, a subset of the rows")
    p.add_argument("--qoi", default="tau,Su1,Su20", help="comma-separated subset of tau, Su1, Su20")
    p.add_argument("--check", action="store_true", help="compare the rows with the shipped files")
    p.add_argument("--nominal", action="store_true", help="write nominal.csv and nominal_gradients.csv")
    p.add_argument("--procs", type=int, default=1, help="worker processes for the rows (default 1)")
    a = p.parse_args()
    out = os.path.abspath(a.out)
    here = os.path.realpath(HERE)
    if os.path.commonpath([os.path.realpath(out), here]) == here:
        sys.exit("--out DIR must lie outside this directory, so that the shipped files are not "
                 "overwritten; compare the new files with the shipped ones and copy them in by hand")
    qois = [q.strip() for q in a.qoi.split(",") if q.strip()]
    if not qois or any(q not in QOIS for q in qois):
        sys.exit("--qoi takes a comma-separated subset of tau, Su1, Su20")
    if D.ct is None:
        print("Cantera is not installed. Install it (pip install cantera==3.2.0, the version that "
              "wrote the shipped files) and rerun generate_data.py.", file=sys.stderr)
        sys.exit(3)
    os.makedirs(out, exist_ok=True)
    if a.nominal:
        write_nominal(out)
        return
    X = design()
    if a.rows:
        i, j = (int(s) for s in a.rows.split(":"))
        rows = slice(i, min(j, M))
    else:
        rows = slice(0, M)
    t0 = time.time()
    run_rows(range(rows.start, rows.stop), qois, out, X, a.procs)
    print("rows %d:%d (%s) done in %.0f s wall" % (rows.start, rows.stop, ",".join(qois), time.time() - t0))
    if a.check:
        check(rows, qois, out)
        return
    done = [q for q in qois if all(os.path.exists(row_file(out, q, i)) for i in range(M))]
    if done:
        write_csvs(done, out, X)


if __name__ == "__main__":
    main()
