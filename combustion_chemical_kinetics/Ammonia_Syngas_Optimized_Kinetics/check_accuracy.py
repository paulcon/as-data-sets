"""Recompute the accuracy numbers quoted in DATA.md for Ammonia_Syngas_Optimized_Kinetics (needs Cantera).

    python check_accuracy.py            # all checks, about 8 min of CPU
    python check_accuracy.py --quick    # checks 1-3 only (under a minute)

Prints, beside each value DATA.md quotes:

1. The optimized rate parameters in ammonia_syngas_optimized.yaml against the paper's Table 2
   (p. 2076; A*, b*, E* in mol, cm, s, cal; low-pressure limit for the falloff reactions):
   largest relative difference of A* and E*, largest absolute difference of b*, and every entry
   that differs by more than the 4 printed digits allow.
2. tau at the optimized model (x = 0) at conditions A, B, C.
3. The ignition criterion at x = 0: tau by the criterion used for the data against max dT/dt,
   max dP/dt (condition A) and the time at which T reaches T0 + (T_eq - T0)/2.
4. tau re-solved at rtol 1e-11, atol 1e-24 at three shipped rows, against the stored tau.
5. Gradient at shipped row 0: central differences at rtol 1e-11, atol 1e-24 (same step and a
   step 10 times smaller) against the stored forward differences (relative 2-norm error in
   normalized coordinates, i.e. each input scaled by its half-range).
6. Directional derivatives at three shipped rows: central differences of log10 tau along four
   random unit directions (normalized coordinates, step 1e-3, rtol 1e-11) against the stored
   gradient dotted with the direction (relative error, largest over directions).
7. Reproducibility at two shipped rows: tau and the forward-difference gradients recomputed with
   the shipped settings at x and at x (1 +- 1e-13), a rounding-level change of the inputs that
   stands in for a change of platform, against the stored values (the measures of
   generate_data.py --check).

Fixed rows and a fixed seed for the directions, so the output is deterministic up to the
solver. Needs numpy, Cantera (the shipped data were written with Cantera 3.2.0) and the mechanism
file, which is not part of this data set (convert_mechanism.py makes it from the article's
supplementary zip; ammonia_idt_driver.MECH says where it is read from); without either the script
stops with an error that says what to do. Reads the shipped files next to it; writes nothing.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ammonia_idt_driver as M  # noqa: E402

# Table 2 of the paper (p. 2076): A* [mol, cm, s], b*, E* [cal/mol], as printed. N2O + H = N2 + OH
# has two rows (two duplicate Arrhenius entries).
TABLE2 = [
    [(1.381e14, -0.139, 5.459e2)], [(8.540e13, -0.163, 1.642e4)], [(1.454e13, 0.516, -2.255e3)],
    [(1.056e15, -0.098, 5.672e4)], [(2.716e12, -0.515, 5.09e3), (3.621e16, -0.515, 1.962e4)],
    [(6.593e23, -2.183, 7.228e2)], [(1.448e5, 2.577, 9.691e3)], [(4.321e11, -0.029, 2.211e4)],
    [(1.588e8, 1.653, 3.564e4)], [(2.518e16, -0.797, 1.064e3)], [(2.802e16, -1.045, 7.948e2)],
    [(8.361e10, 0.618, 2.881e4)], [(6.474e14, -0.437, -2.662e2)], [(2.499e19, -2.325, 7.549e2)],
    [(7.226e10, 0.254, -8.208e2)], [(1.042e2, 2.972, -4.796e3)], [(2.869e15, -1.19, -1.848e1)],
    [(6.453e16, -1.385, 7.261e2)], [(1.064e6, 1.897, 6.106e3)], [(4.266e14, -0.495, 5.918e1)],
    [(4.994e12, 0.241, 3.634e1)], [(6.667e32, -5.053, 1.943e3)], [(1.656e14, -0.478, 2.309e3)],
    [(1.319e17, -0.213, 3.744e4)], [(2.274e4, 2.435, 1.504e3)], [(5.428e13, -0.049, -7.205e1)],
    [(7.502e26, -3.046, 6.546e4)], [(9.994e5, 2.346, 8.755e2)], [(9.4e12, -0.089, 1.193e4)],
    [(3.145e8, 0.969, 1.206e4)], [(8.722e13, 0.096, 5.259e4)], [(8.595e13, -0.535, 6.417e3)],
    [(1.997e4, 2.755, -1.000e3)], [(1.116e3, 2.723, 1.696e4)], [(1.448e17, -1.196, 1.532e3)],
    [(3.936e12, 0.00, 1.00e3)], [(3.147e7, 2.00, 2.00e3)], [(5.526e13, 0.00, 0.00)],
    [(3.733e7, 2.00, 2.000e3)], [(1.298e14, 0.00, 0.00)], [(5.50e11, 0.00, 2.000e3)],
]
CAL = 4.184e3            # J/kmol per cal/mol
ROWS_TAU = [0, 1, 2]
ROW_GRAD = 0
ROWS_DIR = [0, 1, 2]
ROWS_REPRO = [0, 14]


def cgs_params(gas, j):
    """(A, b, E) of Cantera reaction j in mol-cm-s and cal/mol (low-pressure limit for falloff)."""
    rxn = gas.reaction(j)
    rate = rxn.rate
    order = sum(rxn.reactants.values())
    if hasattr(rate, "low_rate"):
        rate, order = rate.low_rate, order + 1
    elif rxn.reaction_type.startswith("three-body"):
        order += 1
    A = rate.pre_exponential_factor * 1e3 ** (order - 1)       # (m3/kmol)^(n-1) -> (cm3/mol)^(n-1)
    return A, rate.temperature_exponent, rate.activation_energy / CAL


def load():
    ld = lambda n: np.loadtxt(os.path.join(HERE, n), delimiter=",", ndmin=2)  # noqa: E731
    X, tau = ld("inputs.csv"), ld("outputs.csv")
    G = np.stack([ld(f"gradients_{k}.csv") for k in M.QOIS], axis=1)       # (M, 3, 123)
    return X, tau, G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="only checks 1-3")
    a = ap.parse_args()
    if M.ct is None:
        print("Cantera is not installed. Install it (pip install cantera==3.2.0, the version that "
              "wrote the shipped files) and rerun check_accuracy.py.", file=sys.stderr)
        sys.exit(3)
    if not os.path.exists(M.MECH):
        sys.exit(M.MECH_HELP + " check_accuracy.py.")
    t0 = time.process_time()
    print(f"Cantera {M.ct.__version__}, numpy {np.__version__}")
    model = M.IgnitionModel()
    gas = model.gas

    print("\n1. mechanism file against Table 2 (p. 2076)")
    dA, db, dE, flagged = 0.0, 0.0, 0.0, []
    for r, idx in enumerate(M.CANTERA_INDEX):
        for j, (A2, b2, E2) in zip(idx, TABLE2[r]):
            A, b, E = cgs_params(gas, j)
            ra, ab = abs(A / A2 - 1), abs(b - b2)
            re = abs(E - E2) / max(abs(E2), 1.0)
            dA, db = max(dA, ra), max(db, ab)
            if ra > 1e-3 or ab > 1e-3 or re > 1e-3:
                kr = {T: (A / A2) * T ** (b - b2) * np.exp(-(E - E2) * CAL / (M.ct.gas_constant * T))
                      for T in (1000.0, 2000.0)}
                flagged.append(f"   R{r + 1:02d} {M.REACTIONS[r]}: file A {A:.5g}, b {b:.4g}, E {E:.5g}; "
                               f"Table 2 A {A2:.4g}, b {b2:.4g}, E {E2:.4g}; k(file)/k(Table 2) "
                               f"{kr[1000.0]:.1e} at 1000 K, {kr[2000.0]:.1e} at 2000 K")
            else:
                dE = max(dE, re)
    print(f"   largest relative difference of A* {dA:.1e}; largest |b difference| {db:.1e}; "
          f"largest relative difference of E* among the unflagged entries {dE:.1e}")
    print(f"   entries that differ beyond the printed digits: {len(flagged)}")
    for line in flagged:
        print(line)

    print("\n2. tau at the optimized model (x = 0)")
    tau0 = {k: model.ignition_delay(k) for k in M.QOIS}
    for k in M.QOIS:
        print(f"   {k}: tau = {tau0[k] * 1e3:.4f} ms ({M.CONDITIONS[k][4]})")

    print("\n3. ignition criterion at x = 0: relative difference from the criterion used for the data")
    for k in M.QOIS:
        tau, h = model.ignition_delay(k, return_history=True)
        T0 = M.CONDITIONS[k][0]
        t, T = h[:, 0], h[:, 2]
        dTdt = np.gradient(T, t)
        t_dT = t[np.argmax(dTdt)]
        Thalf = T0 + 0.5 * (model.T_eq[k] - T0)
        i = np.argmax(T >= Thalf)
        t_half = t[i - 1] + (Thalf - T[i - 1]) * (t[i] - t[i - 1]) / (T[i] - T[i - 1])
        msg = f"   {k}: max dT/dt {t_dT / tau - 1:+.1e}; T = T0 + (T_eq - T0)/2 {t_half / tau - 1:+.1e}"
        if k == "A":
            saved = M.CONDITIONS["A"]
            M.CONDITIONS["A"] = saved[:3] + ("P",) + saved[4:]
            try:
                msg += f"; max dP/dt {model.ignition_delay('A') / tau - 1:+.1e}"
            finally:
                M.CONDITIONS["A"] = saved
        print(msg)
    if a.quick:
        print(f"\nCPU time {time.process_time() - t0:.0f} s")
        return

    X, tau_s, G = load()
    tight = M.IgnitionModel(rtol=1e-11, atol=1e-24)
    print("\n4. tau at rtol 1e-11, atol 1e-24 against the stored tau (relative difference)")
    for i in ROWS_TAU:
        d = tight.taus(X[i]) / tau_s[i] - 1
        print(f"   row {i}: " + ", ".join(f"{k} {abs(v):.1e}" for k, v in zip(M.QOIS, d)))

    print(f"\n5. gradient at row {ROW_GRAD}: central differences at rtol 1e-11 against the stored "
          "forward differences (relative 2-norm error, normalized coordinates)")
    x = X[ROW_GRAD]
    Gc1 = tight.central_grad(x, M.STEPS)
    Gc2 = tight.central_grad(x, M.STEPS / 10)
    for q, k in enumerate(M.QOIS):
        ref = G[ROW_GRAD, q] * M.UPPER
        e1 = np.linalg.norm(Gc1[q] * M.UPPER - ref) / np.linalg.norm(Gc1[q] * M.UPPER)
        e2 = np.linalg.norm(Gc2[q] * M.UPPER - ref) / np.linalg.norm(Gc2[q] * M.UPPER)
        e12 = np.linalg.norm(Gc1[q] - Gc2[q]) / np.linalg.norm(Gc2[q])
        print(f"   {k}: same step {e1:.1e}, step / 10 {e2:.1e} (the two central gradients differ by {e12:.1e})")

    print("\n6. directional derivatives (4 random unit directions, normalized coordinates, step 1e-3)")
    rng = np.random.default_rng(462)
    V = rng.standard_normal((4, x.size))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    h = 1e-3
    for i in ROWS_DIR:
        errs = np.zeros(len(M.QOIS))
        for v in V:
            dx = h * v * M.UPPER                           # step h along v in normalized coordinates
            fd = (np.log10(tight.taus(X[i] + dx)) - np.log10(tight.taus(X[i] - dx))) / (2 * h)
            st = (G[i] * M.UPPER) @ v
            scale = np.linalg.norm(G[i] * M.UPPER, axis=1)
            errs = np.maximum(errs, np.abs(fd - st) / scale)
        print(f"   row {i}: max |difference| / |gradient| " + ", ".join(f"{k} {e:.1e}" for k, e in zip(M.QOIS, errs)))

    print("\n7. reproducibility: the stored rows recomputed with the shipped settings (rtol 1e-8, forward "
          "differences) at x and at x (1 +- 1e-13), a rounding-level change of the inputs; largest relative "
          "difference from the stored tau and relative 2-norm difference of the gradients (normalized "
          "coordinates), as generate_data.py --check measures them")
    for i in ROWS_REPRO:
        for lab, fac in (("x", 1.0), ("x (1 + 1e-13)", 1.0 + 1e-13), ("x (1 - 1e-13)", 1.0 - 1e-13)):
            t, g = model.log10_tau_and_grad(X[i] * fac)
            dt = np.abs(t / tau_s[i] - 1)
            eg = [np.linalg.norm((g[q] - G[i, q]) * M.UPPER) / np.linalg.norm(G[i, q] * M.UPPER)
                  for q in range(len(M.QOIS))]
            print(f"   row {i}, {lab}: tau " + ", ".join(f"{k} {v:.1e}" for k, v in zip(M.QOIS, dt)) +
                  "; gradients " + ", ".join(f"{k} {v:.1e}" for k, v in zip(M.QOIS, eg)))
    print(f"\nCPU time {time.process_time() - t0:.0f} s")


if __name__ == "__main__":
    main()
