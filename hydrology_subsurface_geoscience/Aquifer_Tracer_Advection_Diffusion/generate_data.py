"""Regenerate the inputs, outputs and gradients of Aquifer_Tracer_Advection_Diffusion.

    python generate_data.py --fd                                 # writes all files here
    python generate_data.py --out DIR --check --fd               # writes to DIR, compares with the files here
    python generate_data.py --out DIR --nmain 20 --nconv 5 --check   # a subset

--check, --nmain and --nconv need --out DIR outside this directory, so that the shipped files are
never overwritten. Run times are in DATA.md.

Inputs: M = 500 Monte Carlo draws of the 10 KLE coefficients z ~ N(0, I),
numpy default_rng(SEED).standard_normal((500, 10)).

Three discretisations of the same model (tracer_model.py), all with 64
Crank-Nicolson steps:

  * main set, Chebyshev order 21 per direction (converged for this purpose):
    outputs.csv and gradients.csv, all 500 rows;
  * benchmark variant, order 11 = PyApprox's highest-fidelity member f0:
    outputs_f0.csv and gradients_f0.csv, all 500 rows;
  * convergence check, order 27: outputs_order27.csv and gradients_order27.csv,
    the first NCONV = 50 rows.

Gradients are exact derivatives of the discrete model (tangent-linear mode in
tracer_model.py). --check compares the new values with the files next to this
script (row for row, on the rows computed; relative differences); --fd also prints a central-difference
check of the gradients at five rows on orders 21 and 11. Only numpy, scipy and
the standard library are imported.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tracer_model as tm  # noqa: E402

SEED = 20231825   # inputs
M = 500
NCONV = 50
NAMES = [f"z_{d + 1}" for d in range(tm.M_KLE)]
# suffix: (order, description)
SETS = {
    "": (21, "order 21, 64 steps (main set), all rows of inputs.csv"),
    "_f0": (11, "order 11, 64 steps (PyApprox benchmark mesh f0), all rows of inputs.csv"),
    "_order27": (27, "order 27, 64 steps (convergence check), first {} rows of inputs.csv"),
}


def read_csv(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def run(model, X, label):
    t0 = time.perf_counter()
    f = np.empty(len(X))
    G = np.empty((len(X), X.shape[1]))
    neg0 = negT = 0
    for i, x in enumerate(X):
        q, g, (u0, uT) = model.solve(x, grad=True, return_states=True)
        f[i], G[i] = q, g
        neg0 += u0.min() < 0
        negT += uT.min() < 0
        if (i + 1) % 100 == 0 or i + 1 == len(X):
            print(f"{label}: {i + 1}/{len(X)}, {time.perf_counter() - t0:.0f} s", flush=True)
    dt = time.perf_counter() - t0
    print(f"{label}: {dt:.1f} s ({dt / max(len(X), 1):.2f} s per value and gradient); "
          f"q min {f.min():.4f}, max {f.max():.4f}; samples with a negative nodal "
          f"concentration: {neg0} at t = 0, {negT} at t = T", flush=True)
    return f, G


def write(out, X, results):
    desc = ("Aquifer tracer transport (PyApprox advection-diffusion benchmark, Jakeman 2023); "
            "{} written by generate_data.py, SEED = %d" % SEED)
    np.savetxt(os.path.join(out, "inputs.csv"), X, fmt="%.17g", delimiter=",",
               header=desc.format(f"{len(X)} Monte Carlo draws of the KLE coefficients z ~ N(0, I_10)")
               + "\n" + ",".join(f"{n} (-)" for n in NAMES))
    qname = "q = integral of c(x, T) over [0.75,1]x[0,0.25] (concentration x area)"
    for suffix, (f, G) in results.items():
        what = SETS[suffix][1].format(len(f))
        np.savetxt(os.path.join(out, f"outputs{suffix}.csv"), f[:, None], fmt="%.17g",
                   delimiter=",", header=desc.format(f"QoI, {what}") + "\n" + qname)
        np.savetxt(os.path.join(out, f"gradients{suffix}.csv"), G, fmt="%.17g",
                   delimiter=",", header=desc.format(
                       f"exact discrete gradients dq/dz_d (tangent-linear), {what}")
                   + "\n" + ",".join(f"dq/d{n} (-)" for n in NAMES))


def fd_check(model, X, rows, h=1e-3):
    errs = []
    for r in rows:
        q, g = model.qoi_and_grad(X[r])
        fd = np.array([(model.qoi(X[r] + h * e) - model.qoi(X[r] - h * e)) / (2 * h)
                       for e in np.eye(X.shape[1])])
        errs.append(np.linalg.norm(g - fd) / np.linalg.norm(g))
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--nmain", type=int, default=M, help="rows for order 21 and f0")
    ap.add_argument("--nconv", type=int, default=NCONV, help="rows for order 27")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fd", action="store_true")
    args = ap.parse_args()
    here = os.path.abspath(args.out) == HERE
    if here and (args.check or args.nmain < M or args.nconv < NCONV):
        ap.error("--check, --nmain and --nconv need --out DIR outside this directory, "
                 "so that the shipped files are not overwritten")
    os.makedirs(args.out, exist_ok=True)

    X = np.random.default_rng(SEED).standard_normal((M, tm.M_KLE))
    nmain = min(args.nmain, M)
    nconv = min(args.nconv, nmain)
    nrows = {"": nmain, "_f0": nmain, "_order27": nconv}
    models = {s: tm.TracerModel(order=SETS[s][0], nsteps=64) for s in SETS}
    t0 = time.perf_counter()
    results = {s: run(models[s], X[:nrows[s]], f"order {SETS[s][0]}") for s in SETS}
    write(args.out, X[:nmain], results)
    print(f"total {time.perf_counter() - t0:.0f} s", flush=True)

    if args.fd:
        rows = [0, 1, 2, 3, 4]
        print("FD check (central, h = 1e-3), relative 2-norm error per row:")
        for s in ("", "_f0"):
            print(f"  order {SETS[s][0]}:", np.array2string(fd_check(models[s], X, rows),
                                                          precision=2), flush=True)

    if args.check:
        X0 = read_csv(os.path.join(HERE, "inputs.csv"))
        print("inputs identical:", np.array_equal(X0[:nmain], X[:nmain]))
        for s, (fa, Ga) in results.items():
            f0 = read_csv(os.path.join(HERE, f"outputs{s}.csv"))[:len(fa), 0]
            G0 = read_csv(os.path.join(HERE, f"gradients{s}.csv"))[:len(fa)]
            print(f"outputs{s}: max rel diff {np.max(np.abs(fa - f0) / np.abs(f0)):.1e}; "
                  f"gradients{s}: max rel row diff "
                  f"{np.max(np.linalg.norm(Ga - G0, axis=1) / np.linalg.norm(G0, axis=1)):.1e}"
                  f" ({len(fa)} rows)")


if __name__ == "__main__":
    main()
