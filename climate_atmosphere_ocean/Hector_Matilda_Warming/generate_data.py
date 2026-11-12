"""Regenerate the data files of Hector_Matilda_Warming (needs R with hector 3.1.1 and matilda v1.0.0).

    python generate_data.py --out DIR                       # all files (about 45 min on an idle core; 2 h shared)
    python generate_data.py --out DIR --rows 0:5 --check    # a subset, compared with the shipped files

Design: 1000 Monte Carlo samples z ~ N(0, I_6) (numpy default_rng, seed 20240502), mapped to
Hector's six parameters with Matilda's priors (hector_driver.z_to_x). Every sample is run under
SSP2-4.5 to 2100. Gradients of both outputs with respect to z are central differences with step
H = 0.25 (in standard-deviation units of z) at the first 400 samples. Weights are Matilda's
score_runs(criterion_gmst_obs(), score_bayesian) over all 1000 samples.

Files written to --out DIR:
  inputs.csv           z_1..z_6 (scaled) and the six physical parameters
  outputs.csv          warming [degC], co2_2100 [ppmv], gmst_1850_1900 [degC], rmse [1], weight [1]
  gmst_1950_2023.csv   raw Hector gmst for 1950-2023 [degC] (the scored series)
  gradients.csv        dwarming/dz_j [degC], dco2_2100/dz_j [ppmv] for rows 0..399

Runs are done in chunks whose raw results are cached in DIR/_chunks (the script resumes from
them; the folder is removed when the files are written). --out is required and must be a
directory outside this data set's directory (not it, not a folder inside it), so the shipped
files are never overwritten. --check compares
with a relative tolerance of 1e-8 (same build); weights are compared only for a full run.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hector_driver as D  # noqa: E402

SEED = 20240502
M = 1000
M_GRAD = 400
H = 0.25
SCEN = "ssp245"
CHUNK_BASE = 250
CHUNK_GRAD = 40
TOL = 1e-8
NEED = ("R with the packages hector 3.1.1 (git tag v3.1.1 of github.com/JGCRI/hector) and "
        "matilda v1.0.0 (git tag v1.0.0 of github.com/JGCRI/matilda); see DATA.md, Regenerating. "
        "Install them and rerun generate_data.py.")


def draw():
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((M, D.M_IN))
    return np.array([[float("%.15e" % v) for v in row] for row in Z])


def log(msg, out):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(out, "_chunks", "log.txt"), "a") as f:
        f.write(line + "\n")


def cached(out, name, fn):
    path = os.path.join(out, "_chunks", name + ".npy")
    if os.path.exists(path):
        return np.load(path)
    t0 = time.time()
    r = fn()
    np.save(path, r)
    log(f"{name}: {time.time() - t0:.1f} s", out)
    return r


def score(G):
    with tempfile.TemporaryDirectory() as d:
        pin, pout = os.path.join(d, "g.csv"), os.path.join(d, "s.csv")
        with open(pin, "w") as f:
            f.write(",".join(f"gmst_{y}" for y in D.HIST_YEARS) + "\n")
            for row in G:
                f.write(",".join(repr(float(v)) for v in row) + "\n")
        r = subprocess.run([D.rscript(), os.path.join(HERE, "matilda_score.R"), pin, pout],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("matilda_score.R failed: " + r.stderr.strip()[-500:])
        return np.atleast_2d(np.genfromtxt(pout, delimiter=",", skip_header=1))


def hdr(desc, cols):
    return desc + "\n" + ", ".join(cols)


def generate(out, rows):
    Z = draw()
    X = D.z_to_x(Z)
    os.makedirs(os.path.join(out, "_chunks"), exist_ok=True)
    t0 = time.time()
    base = []
    for a in range(rows.start, rows.stop, CHUNK_BASE):
        b = min(a + CHUNK_BASE, rows.stop)
        base.append(cached(out, f"base_{a}_{b}", lambda a=a, b=b: D.run_x(X[a:b], SCEN, hist=True)))
    base = np.vstack(base)
    grows = range(rows.start, min(rows.stop, M_GRAD))
    grads = []
    for a in range(grows.start, grows.stop, CHUNK_GRAD):
        b = min(a + CHUNK_GRAD, grows.stop)
        F = cached(out, f"grad_{a}_{b}", lambda a=a, b=b: D.run_z(D.fd_stencil(Z[a:b], H), SCEN))
        grads.append(D.fd_gradients(F, H))
    full = rows.start == 0 and rows.stop == M
    if full:
        sc = score(base[:, 3:])
    else:  # weights need the whole ensemble; rmse alone is per run
        sc = np.full((len(base), 2), np.nan)
        sc[:, 0] = score(np.vstack([base[:, 3:]] * 4))[: len(base), 0]
    dt = time.time() - t0

    names = [f"{p[0]} [{p[2]}]" for p in D.PARAMS]
    np.savetxt(os.path.join(out, "inputs.csv"), np.hstack([Z[rows], X[rows]]), fmt="%.17e",
               delimiter=",", header=hdr(
                   f"Hector/Matilda SSP2-4.5: {M} Monte Carlo samples of z ~ N(0,I_6), seed {SEED}, "
                   f"rows {rows.start}:{rows.stop}; physical inputs from Matilda priors (Brown et al. 2024 Table 1)",
                   [f"z_{p[0]} [1]" for p in D.PARAMS] + names))
    np.savetxt(os.path.join(out, "outputs.csv"), np.hstack([base[:, :3], sc]), fmt="%.17e",
               delimiter=",", header=hdr(
                   "Hector 3.1.1 SSP2-4.5: warming = mean gmst 2081-2100 minus mean gmst 1850-1900; "
                   "CO2 concentration 2100; mean gmst 1850-1900; Matilda score_bayesian rmse and normalised weight (gmst 1950-2023)",
                   ["warming [degC]", "co2_2100 [ppmv]", "gmst_1850_1900 [degC]", "rmse [1]", "weight [1]"]))
    np.savetxt(os.path.join(out, "gmst_1950_2023.csv"), base[:, 3:], fmt="%.17e", delimiter=",",
               header=hdr("raw Hector 3.1.1 gmst (global mean surface temperature anomaly) under SSP2-4.5, "
                          "one row per sample, years 1950-2023", [f"gmst_{y} [degC]" for y in D.HIST_YEARS]))
    if grads:
        Gw = np.vstack([g[0] for g in grads])
        Gc = np.vstack([g[1] for g in grads])
        np.savetxt(os.path.join(out, "gradients.csv"), np.hstack([Gw, Gc]), fmt="%.17e", delimiter=",",
                   header=hdr(f"gradients wrt scaled inputs z, central differences step {H} in z, "
                              f"rows {grows.start}:{grows.stop} of inputs.csv",
                              [f"dwarming/dz_{p[0]} [degC]" for p in D.PARAMS]
                              + [f"dco2_2100/dz_{p[0]} [ppmv]" for p in D.PARAMS]))
    shutil.rmtree(os.path.join(out, "_chunks"))
    print(f"generated rows {rows.start}:{rows.stop} in {dt:.1f} s")
    return grows


def check(out, rows, grows):
    full = rows.start == 0 and rows.stop == M
    ok = True
    for fn, sl in [("inputs.csv", rows), ("outputs.csv", rows), ("gmst_1950_2023.csv", rows),
                   ("gradients.csv", slice(grows.start, grows.stop))]:
        if not os.path.exists(os.path.join(out, fn)):
            continue
        new = np.loadtxt(os.path.join(out, fn), delimiter=",", ndmin=2)
        old = np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)[sl]
        if fn == "outputs.csv" and not full:
            new, old = new[:, :4], old[:, :4]
        # a failed run is nan in both files; the nan entries must be the same ones
        same_nan = new.shape == old.shape and np.array_equal(np.isnan(new), np.isnan(old))
        rel = np.nanmax(np.abs(new - old)) / np.nanmax(np.abs(old)) if same_nan else np.inf
        ok &= bool(rel <= TOL)
        print(f"  {fn}: max |new - shipped| / max |shipped| = {rel:.2e} "
              f"({'within' if rel <= TOL else 'OUTSIDE'} tolerance {TOL:.0e})"
              + ("" if same_nan else "; shape or nan entries differ"))
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--rows", default=None, help="a:b, a subset of the 1000 samples")
    p.add_argument("--check", action="store_true", help="compare with the shipped files")
    a = p.parse_args()
    out_real, here_real = os.path.realpath(a.out), os.path.realpath(HERE)
    if os.path.commonpath([out_real, here_real]) == here_real:
        sys.exit("--out must be a directory outside this data set's directory (not it or a folder "
                 "inside it), so that the shipped files are never overwritten; copy the files over "
                 "by hand after a full run if that is intended")
    try:
        ver = D.tool_versions()
    except RuntimeError as e:
        print(f"{e}\nThis script needs {NEED}", file=sys.stderr)
        sys.exit(3)
    if ver[1] != "3.1.1":
        print(f"hector {ver[1]} found; the shipped files were written with hector 3.1.1. {NEED}", file=sys.stderr)
        sys.exit(3)
    print("R", ver[0], "hector", ver[1], "matilda (DESCRIPTION version)", ver[2])
    rows = slice(0, M)
    if a.rows:
        i, j = (int(s) for s in a.rows.split(":"))
        rows = slice(i, j)
    os.makedirs(a.out, exist_ok=True)
    grows = generate(a.out, rows)
    if a.check and not check(a.out, rows, grows):
        sys.exit(1)


if __name__ == "__main__":
    main()
