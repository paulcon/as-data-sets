"""Recompute the accuracy numbers quoted in DATA.md for Parcel_Droplet_Activation (needs pyrcel 2.0.0).

    python check_accuracy.py            # all checks, about 15-30 min on one core
    python check_accuracy.py --quick    # checks 1 and 2 on the nominal point and three rows

Prints, beside each value DATA.md quotes:

1. log10 S_max and its gradient at the nominal point of Fig. 3 of the paper (p. 1265:
   N = 1000 cm-3, mu_g = 0.05 um, sigma_g = 2.0, kappa = 0.54, V = 0.5 m/s, T = 283 K,
   P = 850 hPa, alpha_c = 1.0).
2. Forward-mode AD gradients (the shipped gradients.csv) against central finite differences
   of the model with steps 1e-3 and 1e-4 of each half-range, at fixed rows: the first eight
   rows with an interior peak and the first two rows without one (both with grad_ok = 1).
   Errors are relative 2-norms of the gradient scaled to [-1, 1].
3. Solver tolerance: log10 S_max and its gradient re-solved at rtol = 1e-9 at the same rows,
   against the shipped values (pyrcel's default rtol = 1e-7 wrote the files).
4. The maximum located independently by pyrcel's event solve (dS/dt = 0 root-find,
   integrator.find_smax) at the interior-peak rows, against the shipped log10 S_max.
5. The 400 m ascent cap: at the rows without an interior peak, log10 S at 800 m against the
   shipped value at 400 m.
6. Fig. 3 of the paper (p. 1265): S_max and the activated fraction at the ends of its
   one-at-a-time curves (N = 1e2, 1e4 cm-3; mu_g = 5 nm, 0.1 um; kappa = 0.13, 1.2; V = 1,
   10 m/s; other inputs at the nominal point), and kappa = 0.01, 0.001 for its kappa = 0 point.
   The paper's values are read from the black parcel-model curves.

    python check_accuracy.py --fig3     # check 6 only (a few minutes)

The rows are fixed by these rules and the computation is deterministic. Needs numpy and
pyrcel 2.0.0; without pyrcel the script stops with an error. Run it from any directory; it reads
the shipped files next to it and writes nothing.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pyrcel_parcel_driver as D  # noqa: E402

HR = (D.UPPER - D.LOWER) / 2


def ld(n):
    return np.loadtxt(os.path.join(HERE, n), delimiter=",", ndmin=2)


def rel(a, b):
    return np.linalg.norm((a - b) * HR) / np.linalg.norm(b * HR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="checks 1 and 2 on three rows only")
    ap.add_argument("--fig3", action="store_true", help="check 6 (Fig. 3) only")
    a = ap.parse_args()
    if D.pm is None:
        sys.exit("pyrcel is not installed. Install it (pip install pyrcel==2.0.0, the version that wrote "
                 "the shipped files) and rerun check_accuracy.py.")
    X, Y, G = ld("inputs.csv"), ld("outputs.csv"), ld("gradients.csv")
    pk, gok = Y[:, 2] == 1, Y[:, 3] == 1
    rows_pk = list(np.flatnonzero(pk & gok)[:8])
    rows_np = list(np.flatnonzero(~pk & gok)[:2])
    rows = rows_pk + rows_np
    if a.quick:
        rows = rows_pk[:2] + rows_np[:1]

    if a.fig3:
        fig3()
        return
    print("1. nominal point (Fig. 3)")
    r = D.run_with_grad(D.NOMINAL)
    print(f"   log10 S_max = {r['log10_smax']:.6f} (S_max = {100 * 10 ** r['log10_smax']:.4f} %), "
          f"peak at z = {r['z_smax']:.2f} m, act_frac = {r['act_frac']:.4f}")
    print("   gradient scaled to [-1, 1]:", np.array2string(r["grad_log10_smax"] * HR, precision=4))
    fd, _ = D.fd_gradient(D.NOMINAL, 1e-3)
    print(f"   AD vs central FD (step 1e-3): {rel(fd['log10_smax'], r['grad_log10_smax']):.2e}")

    print("2. shipped AD gradients vs central finite differences")
    e3, e4 = [], []
    for k in rows:
        f3, _ = D.fd_gradient(X[k], 1e-3)
        f4, _ = D.fd_gradient(X[k], 1e-4)
        e3.append(rel(f3["log10_smax"], G[k]))
        e4.append(rel(f4["log10_smax"], G[k]))
        print(f"   row {k} (peaked {int(Y[k, 2])}): step 1e-3 {e3[-1]:.2e}, step 1e-4 {e4[-1]:.2e}")
    print(f"   max over {len(rows)} rows: step 1e-3 {max(e3):.2e}, step 1e-4 {max(e4):.2e}; "
          f"median {np.median(e3):.2e}, {np.median(e4):.2e}")
    if a.quick:
        return

    print("3. re-solve at rtol = 1e-9")
    dv, dg = [], []
    for k in rows:
        q = D.run_with_grad(X[k], rtol=1e-9)
        dv.append(abs(q["log10_smax"] - Y[k, 0]))
        dg.append(rel(q["grad_log10_smax"], G[k]))
        print(f"   row {k}: |d log10 S_max| = {dv[-1]:.2e}, gradient {dg[-1]:.2e}")
    print(f"   max: |d log10 S_max| = {max(dv):.2e}, gradient {max(dg):.2e}")

    print("4. pyrcel event solve (find_smax) at the interior-peak rows")
    import jax.numpy as jnp
    from pyrcel.equilibrate import equilibrate_initial_state
    from pyrcel.integrator import find_smax
    from pyrcel.updraft import ConstantV
    de = []
    for k in rows_pk:
        x = X[k]
        r_d, Ni = (np.asarray(v) for v in D._bins(*(jnp.asarray(v) for v in x[:3])))
        kap = np.full(D.NBINS, x[3])
        y0 = equilibrate_initial_state(x[5], D.S0, x[6], r_d, kap, Ni)
        V = 10.0 ** x[4]
        _, smax, _, act = find_smax(y0, (r_d, Ni, kap, x[7], ConstantV(V)), D.Z_CAP / V)
        de.append(abs(np.log10(float(smax)) - Y[k, 0]))
        print(f"   row {k}: event found {bool(act)}, |d log10 S_max| = {de[-1]:.2e}")
    print(f"   max |d log10 S_max| = {max(de):.2e}")

    print("5. ascent cap: rows without an interior peak, S at 800 m vs the shipped 400 m value")
    for k in np.flatnonzero(~pk)[:3]:
        zc = D.Z_CAP
        try:
            D.Z_CAP = 800.0
            D._jit.clear()
            q = D.run(X[k])
        finally:
            D.Z_CAP = zc
            D._jit.clear()
        print(f"   row {k}: log10 S_max(400 m) = {Y[k, 0]:.4f}, max over 800 m = {q['log10_smax']:.4f} "
              f"(peaked within 800 m: {q['peaked']}), difference {q['log10_smax'] - Y[k, 0]:.4f}")
    fig3()


# (label, input index, value in paper coordinates, paper S_max %, paper act. frac.; read from Fig. 3)
FIG3 = [("N = 1e2 cm-3", 0, 2.0, "~0.40", "~0.77"), ("N = 1e4 cm-3", 0, 4.0, "~0.06", "~0.13"),
        ("mu_g = 5 nm", 1, np.log10(0.005), "~0.75", "~0.03"), ("mu_g = 0.1 um", 1, -1.0, "~0.09", "~0.60"),
        ("kappa = 0.13", 3, 0.13, "~0.25", "~0.37"), ("kappa = 1.2", 3, 1.2, "~0.15", "~0.56"),
        ("kappa = 0.01", 3, 0.01, "~0.40 (plotted at 0)", "~0.12 (plotted at 0)"),
        ("kappa = 0.001", 3, 0.001, "~0.40 (plotted at 0)", "~0.12 (plotted at 0)"),
        ("V = 1 m/s", 4, 0.0, "~0.27", "~0.64"), ("V = 10 m/s", 4, 1.0, "~0.98", "~0.95")]


def fig3():
    print("6. Fig. 3 one-at-a-time points (others at the nominal point): ours vs read from the paper")
    r = D.run(D.NOMINAL)
    print(f"   nominal: S_max {100 * 10 ** r['log10_smax']:.4f} %, act_frac {r['act_frac']:.4f}")
    for lab, j, v, ps, pa in FIG3:
        x = D.NOMINAL.copy()
        x[j] = v
        r = D.run(x)
        print(f"   {lab:14s}: S_max {100 * 10 ** r['log10_smax']:.3f} % (paper {ps}), act_frac "
              f"{r['act_frac']:.3f} (paper {pa}), peaked {r['peaked']}")


if __name__ == "__main__":
    main()
