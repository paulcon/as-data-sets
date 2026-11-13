"""Recompute the accuracy numbers quoted in DATA.md for Lithium_Ion_Impedance_Calibration.

    python check_accuracy.py               # all checks (checks 5-10 need PyBaMM and PyBOP)
    python check_accuracy.py --skip-slow   # without checks 9-10 (finite differences, the slowest part)
    python check_accuracy.py --data-only   # checks 1-4 only: numpy, from the shipped files

Prints, beside the values DATA.md quotes:

1. Design: inputs.csv against numpy default_rng(20261101) uniform on the Table 3 box (largest
   relative difference), the number of distinct rows, and the kurtosis of each column (uniform:
   1.8).
2. Gradient coordinates: the least-squares slopes of log10(Phi) on the normalized inputs against
   the mean normalized gradients of log10(Phi) for the three largest, and the correlation of the
   two over all 18 inputs.
3. True parameters: true_parameters.csv against the values printed in Table 3, the relative
   difference per input (largest, and the inputs above the 4-digit print rounding).
4. Per-SOC files: the nine Phi_m and their gradients sum to Phi and its gradient.
5. Box centre: Phi at the centre of the Table 3 box (reference value 0.29486627966877590 Ohm^2
   from the original run).
6. Reproduction: rows 0, 517 and 999 recomputed (generate_data.check): Phi, the nine Phi_m, the
   gradient and the per-SOC gradients against the files.
7. Impedance input ordering: Z from the build with the 18 parameters and the SOC as PyBaMM
   inputs against a separate pybamm.EISSimulation with numeric parameter values, at two random
   theta (numpy default_rng(1)) and SOC 30 % and 70 %, on a coarse mesh (10 points per domain).
8. Mesh (generate_data.meshcheck): rows 3, 400 and 871 with every mesh doubled (845 states).
9. Gradients against central finite differences (generate_data.gradcheck): rows 3, 400 and 871,
   steps 1e-4 and 1e-6 of each range (about 4 minutes).
10. Impedance Jacobian at the true parameters: central differences with steps 1e-5 and 1e-3 of
    each range against each other, and the 1e-5 one against impedance_jacobian.csv (about 1-2
    minutes).

Checks 1-4 need numpy only. Checks 5-10 need PyBaMM 26.10.1.1, pybammsolvers 0.11.1, PyBOP 26.3
and CasADi 3.8.1 (the versions that wrote the shipped files; recipe in DATA.md); without them
the script runs checks 1-4 and then stops with an error that says what to install (exit status
3; use --data-only to run checks 1-4 alone). Deterministic; writes nothing.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spme_eis_driver as se  # noqa: E402
import generate_data as gd  # noqa: E402

LABELS = [p[2] for p in se.PARAMETERS]
HALF = (se.UPPER - se.LOWER) / 2
# Table 3, p. 12: the true theta as printed
TABLE3_THETA = np.array([6812, 1041, 409.2, 634.7, 246.2, 0.7128, 0.5319, 804.8, 4657, 27592, 0.5935,
                         0.6719, 0.8540, 0.02635, 0.2638, 0.9106, 0.2594, 0.01])


def data_checks():
    X = gd.read_csv("inputs.csv")
    F = gd.read_csv("outputs.csv")
    G = gd.read_csv("gradients.csv")
    GS = gd.read_csv("gradients_soc.csv").reshape(len(X), 9, 18)
    th = gd.read_csv("true_parameters.csv")[0]

    print("1. Design")
    D = gd.design()
    xn = (X - se.LOWER) / (se.UPPER - se.LOWER)
    kurt = np.mean((xn - xn.mean(0)) ** 4, 0) / xn.var(0) ** 2
    print(f"   inputs.csv vs default_rng({gd.SEED}): largest relative difference "
          f"{np.max(np.abs(X - D) / np.abs(D)):.1e}; {len(np.unique(X, axis=0))} distinct rows of {len(X)}; "
          f"kurtosis per column {kurt.min():.2f} to {kurt.max():.2f}")

    print("2. Gradient coordinates (log10 Phi, normalized inputs)")
    f = np.log10(F[:, 0])
    gn = G / (F[:, :1] * np.log(10)) * HALF
    XN = 2 * xn - 1
    slope = np.linalg.lstsq(np.column_stack([np.ones(len(f)), XN]), f, rcond=None)[0][1:]
    mg = gn.mean(0)
    top = np.argsort(-np.abs(slope))[:3]
    print("   regression slope / mean normalized gradient: "
          + ", ".join(f"{LABELS[j]} {slope[j]:.3f} / {mg[j]:.3f}" for j in top)
          + f"; correlation over the 18 inputs {np.corrcoef(slope, mg)[0, 1]:.3f}")

    print("3. True parameters against Table 3 as printed")
    rel = np.abs(th - TABLE3_THETA) / TABLE3_THETA
    rounding = 0.5 * 10.0 ** (np.floor(np.log10(np.abs(TABLE3_THETA))) - 3) / TABLE3_THETA   # half a unit in the 4th digit
    big = np.flatnonzero(rel > rounding)
    print(f"   largest relative difference {rel.max():.1e} ({LABELS[np.argmax(rel)]}); beyond the 4-digit print "
          "rounding: " + ", ".join(f"{LABELS[j]} {th[j]:.5g} vs {TABLE3_THETA[j]:g} ({rel[j]:.1e})" for j in big))

    print("4. Per-SOC files")
    print(f"   sum of Phi_m vs Phi: {np.max(np.abs(F[:, 1:].sum(1) - F[:, 0]) / F[:, 0]):.1e}; "
          f"sum of the per-SOC gradients vs the gradient: {np.max(np.abs(GS.sum(1) - G)) / np.max(np.abs(G)):.1e} "
          "(relative to the largest entry)")


def model_checks(skip_slow):
    import pybamm
    import pybop

    print(f"   PyBaMM {pybamm.__version__}, PyBOP {pybop.__version__}, CasADi {__import__('casadi').__version__}, "
          f"numpy {np.__version__}, scipy {__import__('scipy').__version__}")
    t = time.time()
    mf = se.ImpedanceMisfit()
    print(f"   set-up {time.time() - t:.1f} s, {mf.eis.n_states} states")

    print("5. Box centre")
    phi = mf.misfit((se.LOWER + se.UPPER) / 2)[0]
    ref = 0.29486627966877590
    print(f"   Phi = {phi:.17g} Ohm^2 (original run {ref:.17g}; relative difference {abs(phi - ref) / ref:.1e})")

    print("6. Reproduction of rows 0, 517, 999")
    gd.check([0, 517, 999])

    print("7. Impedance input ordering (coarse mesh)")
    coarse = {k: 10 for k in se.MESH}
    eis = se.SPMeImpedance(mesh=coarse)
    rng = np.random.default_rng(1)
    worst = 0.0
    for theta in se.LOWER + (se.UPPER - se.LOWER) * rng.random((2, se.M_INPUTS)):
        for soc in (0.3, 0.7):
            z = eis.impedance(theta, soc)
            model = pybop.lithium_ion.GroupedSPMe(options={"surface form": "differential"})
            pv = se.chen2020_grouped()
            for n, v in zip(se.PYBOP_NAMES, theta):
                pv[n] = float(v)
            pv["Initial SoC"] = soc
            var_pts = {k: coarse[k.name] for k in model.default_var_pts}
            z_ref = pybamm.EISSimulation(model, parameter_values=pv, var_pts=var_pts).solve(se.FREQUENCIES).impedance
            worst = max(worst, np.max(np.abs(z - z_ref) / np.abs(z_ref)))
    print(f"   largest relative difference over 2 theta x 2 SOCs x 60 frequencies: {worst:.1e}")

    print("8. Mesh doubled")
    gd.meshcheck([3, 400, 871])

    if skip_slow:
        print("9-10. skipped (--skip-slow)")
        return
    print("9. Adjoint gradients against central differences")
    gd.gradcheck([3, 400, 871])

    print("10. Impedance Jacobian at the true parameters")
    Js = gd.jacobian_fd((1e-5, 1e-3))
    a, b = Js[0] * HALF, Js[1] * HALF
    JZ = gd.read_csv("impedance_jacobian.csv")
    Jfile = (JZ[:, 2:20] + 1j * JZ[:, 20:38]).reshape(Js[0].shape) * HALF
    print(f"   steps 1e-5 vs 1e-3: max |difference| / max |entry| (normalized) = {np.max(np.abs(a - b)) / np.max(np.abs(a)):.1e}; "
          f"step 1e-5 vs impedance_jacobian.csv: {np.max(np.abs(a - Jfile)) / np.max(np.abs(Jfile)):.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-only", action="store_true", help="checks 1-4 only (numpy, no PyBaMM/PyBOP)")
    ap.add_argument("--skip-slow", action="store_true", help="leave out checks 9-10")
    ap.add_argument("--any-version", action="store_true", help="run with other PyBaMM/PyBOP/CasADi versions")
    args = ap.parse_args()
    t0 = time.time()
    data_checks()
    if not args.data_only:
        gd.require_packages(args.any_version)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_checks(args.skip_slow)
    print(f"done, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
