"""Generate cipaord_virtual_drugs.csv: the CiPAORdv1.0 endocardial myocyte (Myokit), paced
1000 beats at 2000 ms from the control steady state, at the first 224 points of one scrambled
Sobol' sequence on the nine-input box of Parikh, Di Achille, Kozloski & Gurev (2019,
Front. Pharmacol. 10:1054, Table 2).

Needs Myokit (pip install myokit; the shipped file used 1.39.2) with SUNDIALS CVODES (e.g. apt install
libsundials-dev) and a C compiler; without them the script stops with an error that says what to install.
One process; about 24 s of CPU per point (1000 beats), about 91 min of CPU in all.

    nice -n 10 python generate_data.py            # all chunks (finished ones skipped), then merge
    python generate_data.py --chunk 3             # one chunk of 16 points -> parts/chunk_3.npz
    python generate_data.py --merge               # parts -> cipaord_virtual_drugs.csv
    python generate_data.py --check 3             # rerun 3 points; compare with the CSV and with
                                                  # a tighter tolerance; control steady state

A chunk saves its progress after every point (parts/chunk_<k>.partial.npz), so an interrupted
run resumes where it stopped. Runs with EADs or without full repolarisation are kept and
flagged in the `response` column (never dropped).
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

import cipaord_driver as cm

SEED = 1054
N = 224              # first N points of a scrambled Sobol' sequence of 2^8 = 256 points
CHUNK = 16
N_CHUNKS = N // CHUNK
ABS_TOL, REL_TOL = 1e-6, 1e-4
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cipaord_virtual_drugs.csv")
EXTRA = ["max_dv", "dVdt_max", "n_EAD", "n_norepol", "qNet_spread", "APD90_spread",
         "qNet_b500", "APD90_b500"]
EXTRA_UNITS = ["mV/ms", "mV/ms", "count", "count", "nC/uF", "ms", "nC/uF", "ms"]
COLS = cm.NAMES + cm.QOI_NAMES + EXTRA + ["response", "cpu_s"]
UNITS = cm.UNITS + cm.QOI_UNITS + EXTRA_UNITS + ["code", "s"]


def need_myokit():
    """Stop with an install message unless Myokit can compile and run the model."""
    try:
        import myokit  # noqa: F401
    except ImportError:
        print("Myokit is not installed. Install it (pip install myokit==1.39.2, the version that wrote the "
              "shipped file) together with SUNDIALS (e.g. apt install libsundials-dev or brew install sundials) "
              "and a C compiler, then rerun generate_data.py.", file=sys.stderr)
        sys.exit(3)
    try:
        cm._sim(ABS_TOL, REL_TOL)
    except Exception as e:  # noqa: BLE001
        print(f"Myokit cannot compile the model ({type(e).__name__}: {str(e).splitlines()[0]}). Install SUNDIALS "
              "(e.g. apt install libsundials-dev or brew install sundials) and a C compiler, then rerun "
              "generate_data.py.", file=sys.stderr)
        sys.exit(3)


def design():
    u = qmc.Sobol(d=len(cm.NAMES), scramble=True, seed=SEED).random_base2(8)[:N]
    return cm.LOWER + (cm.UPPER - cm.LOWER) * u


def evaluate(x, abs_tol=ABS_TOL, rel_tol=REL_TOL):
    r = cm.run(x, abs_tol=abs_tol, rel_tol=rel_tol)
    return [r[q] for q in cm.QOI_NAMES] + [r[e] for e in EXTRA] + [cm.response_class(r), r["cpu_s"]]


def run_chunk(k, parts):
    os.makedirs(parts, exist_ok=True)
    final = os.path.join(parts, f"chunk_{k}.npz")
    partial = os.path.join(parts, f"chunk_{k}.partial.npz")
    if os.path.exists(final):
        print(f"chunk {k}: done already, skipped", flush=True)
        return
    start, stop = k * CHUNK, (k + 1) * CHUNK
    X = design()[start:stop]
    rows = []
    if os.path.exists(partial):
        rows = [list(r) for r in np.load(partial)["Y"]]
        print(f"chunk {k}: resuming after {len(rows)} points", flush=True)
    for i in range(len(rows), len(X)):
        row = evaluate(X[i])
        rows.append(row)
        print(f"{start + i:4d} resp={int(row[-2])} qNet={row[0]:7.2f} APD90={row[1]:7.1f} "
              f"(b500 {row[14]:7.2f} {row[15]:7.1f}) cpu={row[-1]:5.1f}s", flush=True)
        np.savez(partial, Y=np.array(rows))
    np.savez(final, X=X, Y=np.array(rows), start=start, stop=stop)
    os.remove(partial)
    print(f"chunk {k}: done", flush=True)


def merge(parts):
    X, Y = [], []
    for k in range(N_CHUNKS):
        d = np.load(os.path.join(parts, f"chunk_{k}.npz"))
        X.append(d["X"]); Y.append(d["Y"])
    X, Y = np.vstack(X), np.vstack(Y)
    assert np.array_equal(X, design())
    header = ("CiPAORdv1.0 endo cell (Myokit), 1000 beats at CL 2000 ms from control steady state; "
              f"first {N} points of a scrambled Sobol' sequence (scipy, seed {SEED}) on Table 2 of "
              "Parikh et al. 2019 (doi:10.3389/fphar.2019.01054); features of beat 1000; "
              "response 0 normal, 1 EAD in beats 996-1000, 2 no full repolarisation\n"
              + ",".join(f"{c} [{u}]" for c, u in zip(COLS, UNITS)))
    np.savetxt(OUT, np.hstack([X, Y]), delimiter=",", header=header, fmt="%.10g")
    print(f"wrote {OUT}: {X.shape[0]} rows; responses", np.bincount(Y[:, -2].astype(int), minlength=3),
          f"; CPU {Y[:, -1].sum() / 60:.1f} min")


def check(n):
    """Rerun n points spread over the design; compare with the CSV (same tolerance) and with
    rel_tol = 1e-6 (the paper's lsoda tolerances were rtol = atol = 1e-6); then pace the
    control model 1000 beats and compare the final state with the initial state."""
    D = np.loadtxt(OUT, delimiter=",")
    idx = np.linspace(0, N - 1, n).round().astype(int)
    m = len(cm.NAMES)
    nq = len(cm.QOI_NAMES)
    X = design()                       # exact inputs (the CSV holds them to 10 digits)
    for i in idx:
        x = X[i]
        a = np.array(evaluate(x))
        b = np.array(evaluate(x, ABS_TOL, 1e-6))
        f = D[i, m:m + nq]
        d_same = np.nanmax(np.abs(a[:nq] - f) / np.abs(f))
        d_tight = np.nanmax(np.abs(b[:nq] - f) / np.abs(f))
        print(f"row {i}: max rel. difference of the 8 features: rerun {d_same:.1e}, "
              f"rel_tol 1e-6 {d_tight:.1e} (qNet {b[0]:.4f} vs {f[0]:.4f}, APD90 {b[1]:.3f} vs {f[1]:.3f}); "
              f"cpu {a[-1]:.1f}s / {b[-1]:.1f}s", flush=True)
    import myokit
    sim, s0, iq = cm._sim(ABS_TOL, REL_TOL)
    sim.reset(); sim.set_time(0); sim.set_state(s0)
    for (name, var, lo, hi, u) in cm.INPUTS:
        sim.set_constant(var, 0.0)
    sim.run(cm.N_BEATS * cm.CL, log=myokit.LOG_NONE)
    s1 = np.array(sim.state()); s0 = np.array(s0)
    keep = [k for k in range(len(s0)) if k != iq and abs(s0[k]) > 1e-12]
    rel = np.abs(s1[keep] - s0[keep]) / np.abs(s0[keep])
    names = [v.qname() for v in myokit.load_model(cm.MMT).states()]
    j = keep[int(np.argmax(rel))]
    print(f"control, 1000 beats from the distributed CL2000 steady state: max rel. state change "
          f"{rel.max():.1e} ({names[j]}), median {np.median(rel):.1e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--check", type=int)
    p.add_argument("--parts", default=os.path.join(HERE, "parts"))
    a = p.parse_args()
    t0 = time.time()
    if not a.merge:
        need_myokit()
    if a.check:
        check(a.check)
    elif a.merge:
        merge(a.parts)
    elif a.chunk is not None:
        run_chunk(a.chunk, a.parts)
    else:
        for k in range(N_CHUNKS):
            run_chunk(k, a.parts)
        merge(a.parts)
    print(f"wall time {time.time() - t0:.0f} s")
