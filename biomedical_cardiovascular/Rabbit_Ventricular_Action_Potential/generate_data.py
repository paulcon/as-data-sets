"""Generate rabbit_ap_samples.csv: the Shannon et al. (2004) rabbit ventricular myocyte,
run with Myokit, at the first 320 points of one scrambled Sobol' sequence on the Table 4 box
of Yang, Gao, Smith & Simitev (2025, PLOS ONE 20(7):e0328261).

Needs Myokit (pip install myokit; the shipped file used 1.39.2) with SUNDIALS CVODES (e.g. apt install
libsundials-dev) and a C compiler; without them the script stops with an error that says what to install.
One process; about 10 s of CPU per point (1000 beats), about 55 min of CPU in all.

    nice -n 10 python generate_data.py            # all 4 chunks (finished ones skipped), then merge
    python generate_data.py --chunk 2             # one chunk of 80 points -> parts/chunk_2.npz
    python generate_data.py --merge               # parts -> rabbit_ap_samples.csv
    python generate_data.py --check 3             # rerun the first 3 points, compare with the CSV

A chunk saves its progress every 5 points (parts/chunk_<k>.partial.npz), so an interrupted
run resumes where it stopped. Runs whose response is not a normal 1:1 AP train are kept and
flagged in the `response` column (never dropped).
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

import shannon_driver as sm

SEED = 20251031
N = 320              # first N points of a scrambled Sobol' sequence of 2^9 = 512 points
N_CHUNKS = 4
N_BEATS = 1000       # paper p. 10: trains of 1000 APs at BCL 500 ms, biomarkers of the last AP
CHECK_BEAT = 500     # biomarkers also recorded at beat 500 (convergence check)
HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "parts")
OUT = os.path.join(HERE, "rabbit_ap_samples.csv")
QOIS = ["Vmax", "Vrest", "A90", "A30", "Vplt", "B", "Cai_ED", "Cai_PSV"]
QUNITS = ["mV", "mV", "ms", "ms", "mV", "mV ms", "mM", "mM"]
CLASSES = {"normal": 0, "no_AP": 1, "alternans_or_2to1": 2}
COLS = (sm.NAMES + QOIS + ["A90_prev"] + ["b500_" + q for q in QOIS] + ["response", "cpu_s"])
UNITS = ["-"] * 15 + QUNITS + ["ms"] + QUNITS + ["code", "s"]


def need_myokit():
    """Stop with an install message unless Myokit can compile and run the model."""
    try:
        import myokit  # noqa: F401
    except ImportError:
        sys.exit("Myokit is not installed. Install it (pip install myokit==1.39.2, the version that wrote the "
                 "shipped file) together with SUNDIALS (e.g. apt install libsundials-dev or brew install sundials) "
                 "and a C compiler, then rerun generate_data.py.")
    try:
        sm._sim()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Myokit cannot compile the model ({type(e).__name__}: {str(e).splitlines()[0]}). Install SUNDIALS "
                 "(e.g. apt install libsundials-dev or brew install sundials) and a C compiler, then rerun "
                 "generate_data.py.")


def design():
    u = qmc.Sobol(d=len(sm.NAMES), scramble=True, seed=SEED).random_base2(9)[:N]
    return sm.LOWER + (sm.UPPER - sm.LOWER) * u


def chunk_range(k):
    size = N // N_CHUNKS
    return k * size, (k + 1) * size


def evaluate(x):
    c0 = time.process_time()
    r = sm.run(x, n_beats=N_BEATS, check_beat=CHECK_BEAT)
    cpu = time.process_time() - c0
    row = ([r[q] for q in QOIS] + [r["prev_A90"]] + [r["b500_" + q] for q in QOIS]
           + [CLASSES[r["cls"]], cpu])
    return row, r["cls"]


def run_chunk(k):
    os.makedirs(PARTS, exist_ok=True)
    final = os.path.join(PARTS, f"chunk_{k}.npz")
    partial = os.path.join(PARTS, f"chunk_{k}.partial.npz")
    if os.path.exists(final):
        print(f"chunk {k}: done already, skipped", flush=True)
        return
    start, stop = chunk_range(k)
    X = design()[start:stop]
    rows = []
    if os.path.exists(partial):
        rows = [list(r) for r in np.load(partial)["Y"]]
        print(f"chunk {k}: resuming after {len(rows)} points", flush=True)
    for i in range(len(rows), len(X)):
        row, cls = evaluate(X[i])
        rows.append(row)
        print(f"{start + i:4d} {cls:18s} A90={row[2]:8.2f} b500_A90={row[11]:8.2f} "
              f"cpu={row[-1]:5.1f}s", flush=True)
        if (i + 1) % 5 == 0:
            np.savez(partial, Y=np.array(rows))
    np.savez(final, X=X, Y=np.array(rows), start=start, stop=stop)
    if os.path.exists(partial):
        os.remove(partial)


def merge():
    parts = [np.load(os.path.join(PARTS, f"chunk_{k}.npz")) for k in range(N_CHUNKS)]
    X = np.vstack([p["X"] for p in parts])
    Y = np.vstack([p["Y"] for p in parts])
    assert np.array_equal(X, design()), "chunk inputs differ from the design"
    hdr = ("Shannon 2004 rabbit ventricular myocyte (Myokit 1.39.2, CVODES tol 1e-6/1e-4); 15 "
           "current scaling factors on the Table 4 box of Yang et al. 2025 (PLOS ONE "
           f"e0328261); first {N} points of a scrambled Sobol' sequence (scipy, seed {SEED}); "
           f"{N_BEATS} beats at BCL 500 ms from the model file's initial state; biomarkers of "
           "the last beat, A90 of beat 999, all biomarkers of beat 500; response 0 = normal 1:1, "
           "1 = no AP, 2 = alternans or 2:1\n"
           + ",".join(f"{c} [{u}]" for c, u in zip(COLS, UNITS)))
    np.savetxt(OUT, np.hstack([X, Y]), delimiter=",", header=hdr, fmt="%.12g")
    resp = Y[:, COLS.index("response") - 15]
    cpu = Y[:, -1]
    print(f"wrote {OUT}: {X.shape[0]} rows; responses "
          + ", ".join(f"{c}={int(np.sum(resp == v))}" for c, v in CLASSES.items())
          + f"; CPU per run {cpu.mean():.1f} s (min {cpu.min():.1f}, max {cpu.max():.1f}), "
          f"total {cpu.sum() / 60:.1f} min")


def check(n):
    D = np.loadtxt(OUT, delimiter=",")
    X = design()
    assert np.allclose(D[:, :15], X, rtol=1e-11, atol=0), "CSV inputs differ from the design"
    worst = 0.0
    for i in range(n):
        row, cls = evaluate(X[i])
        new = np.array(row[:-1])                 # all but cpu_s
        old = D[i, 15:-1]
        rel = np.max(np.abs(new - old) / np.maximum(np.abs(old), 1e-300))
        worst = max(worst, rel)
        print(f"point {i}: {cls}; max relative difference from the CSV {rel:.2e}", flush=True)
    print(f"check: worst relative difference over {n} points {worst:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--check", type=int, metavar="N_POINTS")
    a = ap.parse_args()
    t0 = time.time()
    if not a.merge:
        need_myokit()
    if a.merge:
        merge()
    elif a.check:
        check(a.check)
    elif a.chunk is not None:
        run_chunk(a.chunk)
    else:
        for k in range(N_CHUNKS):
            run_chunk(k)
        merge()
    print(f"wall {time.time() - t0:.0f} s")
