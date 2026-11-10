"""Accuracy checks for Shared_Subspace_H2_Ignition_Flame (needs Cantera 3.2.0 and
konnov2008_h2.yaml from build_mechanism.py). Prints every number of DATA.md "Accuracy".

    python check_accuracy.py            # about 10 min on one core
    python check_accuracy.py --quick    # nominal values and one row only

1. Ignition delay: re-solve at rtol 1e-13, atol 1e-24 (shipped: 1e-10, 1e-20).
2. Ignition-delay gradients: shipped forward differences vs central differences (same step).
3. Flame speed: nominal S_L on a finer grid (ratio 2, slope 0.02, curve 0.04; shipped 2.5,
   0.04, 0.08) and with Newton tolerances 1e-11/1e-16 (shipped 1e-9/1e-14).
4. Flame-speed gradients: shipped forward differences vs central differences, vs Cantera's
   adjoint with the sample's multipliers kept, and vs Cantera's built-in
   FreeFlame.get_flame_speed_reaction_sensitivities(). The built-in routine returns the
   derivative with respect to the rate multiplier, (1/S_L) dS_L/dm_j; times m_j (chain rule)
   it is d ln S_L / d ln k_j, the derivative stored here (4c).

    python check_accuracy.py --adjoint  # section 4c only (rows 328, 218, 239; a few seconds)
Errors are relative 2-norm errors of gradient vectors in the normalized inputs x.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h2_konnov_driver as D  # noqa: E402

ROWS_IDT = [0, 117, 233, 349]
ROWS_SL_CENTRAL = [0, 175, 349]
ROWS_SL_ADJ = list(range(0, 350, 29))          # 13 rows
ROWS_CHAIN = [328, 218, 239]                    # 4c: built-in adjoint times m_j, at 1 atm


def relerr(a, b):
    a = np.asarray(a) * D.SIGMA
    b = np.asarray(b) * D.SIGMA
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--adjoint", action="store_true", help="run section 4c only")
    p.add_argument("--data", default=HERE, help="directory with the CSVs (default: this one)")
    a = p.parse_args()
    if D.ct is None:
        sys.exit("Cantera is not installed: pip install cantera==3.2.0, then rerun check_accuracy.py")
    if not os.path.exists(D.MECH):
        sys.exit("konnov2008_h2.yaml is missing: run build_mechanism.py first")
    import cantera as ct
    ct.suppress_thermo_warnings()
    L = np.loadtxt(os.path.join(a.data, "inputs.csv"), delimiter=",", ndmin=2)
    F = np.loadtxt(os.path.join(a.data, "outputs.csv"), delimiter=",", ndmin=2)
    G = np.loadtxt(os.path.join(a.data, "gradients.csv"), delimiter=",", ndmin=2)
    G = G.reshape(G.shape[0], D.N_OUT, D.M_IN)
    n = L.shape[0]
    rows_idt = [r for r in ROWS_IDT if r < n][:1 if a.quick else None]
    rows_slc = [r for r in ROWS_SL_CENTRAL if r < n][:1 if a.quick else None]
    rows_adj = [r for r in ROWS_SL_ADJ if r < n][:1 if a.quick else None]
    m = D.H2Model()
    x0 = np.zeros(D.M_IN)
    if a.adjoint:
        chain_rule(m, L, G, [r for r in ROWS_CHAIN if r < n])
        return

    print("Nominal values (x = 0):")
    for p_, t_ in D.IDT_CONDITIONS:
        print(f"  tau(P={p_:.4g} atm, T0={t_:.0f} K) = {m.ignition_delay(p_, t_, x0):.6e} s")
    for p_ in D.SL_P:
        print(f"  S_L(P={p_:g} atm) = {m.flame_speed(p_, x0):.6f} m/s, "
              f"{len(m.flames[p_].grid)} grid points")

    print("\n1. tau, tight solver: max relative change over the 16 conditions")
    for r in rows_idt:
        x = L[r] / D.SIGMA
        e = [abs(m.ignition_delay(p_, t_, x, rtol=1e-13, atol=1e-24) / F[r, i] - 1)
             for i, (p_, t_) in enumerate(D.IDT_CONDITIONS)]
        print(f"  row {r}: {max(e):.1e}")

    print("\n2. tau gradients: forward (shipped) vs central differences, worst of 16 conditions")
    for r in rows_idt:
        x = L[r] / D.SIGMA
        e = []
        for i, (p_, t_) in enumerate(D.IDT_CONDITIONS):
            _, gc = m.idt_grad(p_, t_, x, central=True)
            e.append(relerr(G[r, i], gc))
        k = int(np.argmax(e))
        print(f"  row {r}: max {max(e):.1e} (at P={D.IDT_CONDITIONS[k][0]:.4g} atm, "
              f"T0={D.IDT_CONDITIONS[k][1]:.0f} K), median {np.median(e):.1e}")

    print("\n3. S_L at x = 0: finer grid and tighter Newton tolerances (relative change)")
    for p_ in D.SL_P:
        s0 = m.flame_speed(p_, x0)
        g = m.gas
        for j in range(D.M_IN):
            g.set_multiplier(1.0, j)
        g.TPX = 300.0, p_ * D.ATM, "H2:2, O2:1, N2:3.76"
        f = ct.FreeFlame(g, width=0.03 / p_ ** 0.7)
        f.transport_model = "mixture-averaged"
        f.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.04, prune=0.005)
        f.set_max_grid_points(f.flame, 5000)
        f.flame.set_steady_tolerances(default=(1e-11, 1e-16))
        f.flame.set_transient_tolerances(default=(1e-11, 1e-16))
        f.solve(loglevel=0, auto=True)
        print(f"  P={p_:g} atm: {float(f.velocity[0]) / s0 - 1:+.1e} ({len(f.grid)} points)")

    i_sl = list(range(D.N_IDT, D.N_OUT))
    print("\n4a. S_L gradients: forward (shipped) vs central differences")
    for r in rows_slc:
        x = L[r] / D.SIGMA
        e = []
        for k, p_ in enumerate(D.SL_P):
            _, gc = m.sl_grad(p_, x, central=True)
            e.append(relerr(G[r, i_sl[k]], gc))
        print(f"  row {r}: " + ", ".join(f"{p_:g} atm {v:.1e}" for p_, v in zip(D.SL_P, e)))

    print(f"\n4b. S_L gradients on {len(rows_adj)} rows: adjoint vs shipped forward differences "
          "(median and max over rows)")
    for k, p_ in enumerate(D.SL_P):
        ea, eb = [], []
        for r in rows_adj:
            x = L[r] / D.SIGMA
            _, ga = m.sl_adjoint(p_, x)
            _, gb = m.sl_adjoint(p_, x, builtin=True)
            ea.append(relerr(ga, G[r, i_sl[k]]))
            eb.append(relerr(gb, G[r, i_sl[k]]))
        print(f"  P={p_:g} atm: multipliers kept {np.median(ea):.1e} / {max(ea):.1e}; "
              f"built-in {np.median(eb):.1e} / {max(eb):.1e}")
    _, ga0 = m.sl_adjoint(D.SL_P[-1], x0)
    _, gb0 = m.sl_adjoint(D.SL_P[-1], x0, builtin=True)
    print(f"  at x = 0, P=20 atm, the two adjoints differ by {relerr(gb0, ga0):.1e}")
    chain_rule(m, L, G, [r for r in ROWS_CHAIN if r < n][:1 if a.quick else None])


def chain_rule(m, L, G, rows):
    """4c. Cantera's built-in result is (1/S_L) dS_L/dm_j; times m_j = exp(sigma_j x_j) it is
    d ln S_L/d ln k_j. Compared at 1 atm with the adjoint that keeps the multipliers and with
    the shipped forward differences."""
    i_sl = D.N_IDT
    p_ = D.SL_P[0]
    print(f"\n4c. S_L at {p_:g} atm: Cantera's built-in adjoint (derivative w.r.t. the multiplier m_j) "
          "with and without the factor m_j")
    for r in rows:
        x = L[r] / D.SIGMA
        _, ga = m.sl_adjoint(p_, x)
        _, gb = m.sl_adjoint(p_, x, builtin=True)
        gbm = gb * np.exp(D.SIGMA * x)
        print(f"  row {r}: built-in vs shipped FD {relerr(gb, G[r, i_sl]):.2e}; built-in x m_j vs "
              f"multipliers-kept adjoint {relerr(gbm, ga):.1e}, vs shipped FD {relerr(gbm, G[r, i_sl]):.1e}; "
              f"multipliers-kept adjoint vs shipped FD {relerr(ga, G[r, i_sl]):.1e}")


if __name__ == "__main__":
    main()
