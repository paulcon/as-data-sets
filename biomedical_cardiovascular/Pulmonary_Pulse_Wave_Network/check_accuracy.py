"""Accuracy checks for Pulmonary_Pulse_Wave_Network (needs git and g++; reruns the authors' solver).

    python check_accuracy.py --work /tmp/pulmwk                 # rows 0:5, about 8 CPU-minutes
    python check_accuracy.py --work /tmp/pulmwk --rows 0:10

Prints, for each checked row of the shipped files:
  1. the base run again: max |p - shipped| over (p_mean, p_sys, p_dia), in mmHg;
  2. central differences with steps 1e-3 and 1e-4 of each half-range (18 runs each) against the
     shipped forward differences (step 1e-3): ||g_fwd - g_c|| / ||g_c|| per QoI, with g in
     normalized coordinates (each input scaled by its half-range), and the same between the two
     central steps; the same for the pulse pressure pp = p_sys - p_dia (gradient dp_sys - dp_dia);
and, with no solver runs, the mean-pressure check over all rows: p_mean against the Windkessel
estimate Qbar * (R_LPA || R_RPA), R = (Rp + Rd) * R_SCALE, Qbar = mean inflow of Qin.dat.
The solver is cloned into --work and compiled there (it is not part of this data set).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import pulmonary_wk_driver as D  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--source", default=None, help="an existing clone of the authors' repository")
    p.add_argument("--rows", default="0:5")
    a = p.parse_args()
    if os.path.commonpath([os.path.realpath(a.work), HERE]) == HERE:
        sys.exit("--work must be outside this directory")
    try:
        exe = D.build(a.work, a.source)
    except D.SolverMissing as e:
        sys.exit(f"{e}. Install git and g++ and rerun check_accuracy.py.")
    X = np.loadtxt(os.path.join(HERE, "inputs.csv"), delimiter=",")
    F = np.loadtxt(os.path.join(HERE, "outputs.csv"), delimiter=",")
    G = np.loadtxt(os.path.join(HERE, "gradients.csv"), delimiter=",")
    hw = 0.5 * (D.UPPER - D.LOWER)
    QN = D.QOI_NAMES + ["pp"]

    qbar = np.loadtxt(os.path.join(exe, "Qin.dat")).mean()
    ok = F[:, 3] <= 1
    R1 = (X[ok, 3] + X[ok, 5]) * D.R_SCALE
    R2 = (X[ok, 4] + X[ok, 6]) * D.R_SCALE
    est = qbar * R1 * R2 / (R1 + R2)
    rel = (F[ok, 0] - est) / F[ok, 0]
    print(f"mean inflow Qbar = {qbar:.4f} mL/s")
    print(f"p_mean vs Qbar*(R_LPA||R_RPA), {ok.sum()} rows: relative difference "
          f"min {rel.min():.4f}, median {np.median(rel):.4f}, max {rel.max():.4f}")

    i, j = (int(s) for s in a.rows.split(":"))
    worst = {"base": 0.0, "fwd_vs_c3": np.zeros(4), "c4_vs_c3": np.zeros(4)}
    for r in range(i, j):
        if F[r, 3] > 1:
            print(f"row {r}: status {int(F[r, 3])}, skipped")
            continue
        f0 = D.qois(X[r], exe)
        db = np.max(np.abs(f0 - F[r, :3]))
        g3, fl3 = D.fd_gradient(X[r], exe, 1e-3, f0)
        g4, fl4 = D.fd_gradient(X[r], exe, 1e-4, f0)
        gs = G[r].reshape(3, 9) * hw
        g3 = g3 * hw
        g4 = g4 * hw
        gs, g3, g4 = (np.vstack([g, g[1] - g[2]]) for g in (gs, g3, g4))   # add pp = p_sys - p_dia
        e_f = np.linalg.norm(gs - g3, axis=1) / np.linalg.norm(g3, axis=1)
        e_c = np.linalg.norm(g4 - g3, axis=1) / np.linalg.norm(g3, axis=1)
        worst["base"] = max(worst["base"], db)
        worst["fwd_vs_c3"] = np.maximum(worst["fwd_vs_c3"], e_f)
        worst["c4_vs_c3"] = np.maximum(worst["c4_vs_c3"], e_c)
        print(f"row {r}: base run max |diff| {db:.1e} mmHg; forward vs central(1e-3) "
              + ", ".join(f"{q} {e:.1e}" for q, e in zip(QN, e_f))
              + "; central(1e-4) vs central(1e-3) "
              + ", ".join(f"{q} {e:.1e}" for q, e in zip(QN, e_c))
              + f" (flags {fl3}, {fl4})", flush=True)
    print(f"worst over rows {i}:{j}: base {worst['base']:.1e} mmHg; forward vs central "
          + ", ".join(f"{q} {e:.1e}" for q, e in zip(QN, worst["fwd_vs_c3"]))
          + "; central 1e-4 vs 1e-3 "
          + ", ".join(f"{q} {e:.1e}" for q, e in zip(QN, worst["c4_vs_c3"])))


if __name__ == "__main__":
    main()
