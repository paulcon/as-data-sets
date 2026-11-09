"""Recompute the accuracy numbers quoted in DATA.md for H2_Flame_Kinetics_Bayesian_Calibration (needs Cantera).

    python check_accuracy.py                 # checks 1-7 (about 2-3 h on one core)
    python check_accuracy.py --quick         # checks 1-3 only (about 1 min)
    python check_accuracy.py --checks 5,8    # the checks named (8, 9 and 10 run only when named)

Prints, beside each value DATA.md quotes:

1. The mechanism file against Bell et al. (2019) Table 1 (p. 6): A, beta, Ea of every
   Arrhenius expression (cm-mol-s-cal units), the Troe Fc values and every printed third-body
   efficiency, as the largest relative difference, with the entries that differ.
2. The three QoIs at the Table 1 values, against nominal.csv.
3. Spatial resolution of the flame speeds: Su at the Table 1 values and at the check rows
   with the grid criteria of the data (ratio 2, slope 0.02, curve 0.04) and with slope 0.01,
   curve 0.02.
4. Ignition delay at the check rows: tau re-solved at rtol 1e-12, atol 1e-22, and the
   gradient with a step ten times smaller, against the stored values; tau with the ignition
   threshold at a quarter and three quarters of the temperature rise instead of one half.
5. Su1 gradients and 6. Su20 gradients (central finite differences, FlameSpeed.fd_grad) at the
   check rows: the stored gradient against fd_grad recomputed here; fd_grad's own step check
   (central with a step ten times larger), forward against central, and the largest drift back
   to the base state; the stored gradient against independent central differences (flame_fd:
   step 1e-2 of each prior sd, Su re-solved on the converged grid with tight steady-state
   tolerances, relative 1e-9 and absolute 1e-14; A factors in place, efficiencies through a new
   gas with the converged solution restored, whose shift of Su cancels in central differences);
   Cantera's discrete adjoint (FlameSpeed.adjoint_grad) against the stored gradient: the ratio
   of norms, the cosine and the relative error; and the driver's adjoint against Cantera's own
   get_flame_speed_reaction_sensitivities for the 13 A factors of non-falloff reactions.
7. Row 412: Su20 re-solved from scratch, against outputs.csv (see DATA.md, "Row 412").
8. Cantera's adjoint against the stored finite-difference gradients of Su1 and Su20 over all
   500 rows (or --rows a:b): ratio of norms, cosine and relative error, as median, 5-95 %
   range and extremes (about 40 min on one core).
9. Grid dependence of the adjoint defect without the driver: a plain Cantera FreeFlame with the
   unmodified mechanism at the Table 1 values, at the Su20 and Su1 conditions, with coarse grid
   criteria (ratio 3, slope 0.06, curve 0.12) and with those of the data; the elasticity of Su
   to the R9 rate multiplier from get_flame_speed_reaction_sensitivities against central
   differences of the multiplier (+-1e-3) on the same fixed grid (a few minutes).
10. Truncation of fd_grad's efficiency scheme for the falloff reactions R9 and R15: the derivative
   of the forward rate constant in an efficiency from the multipliers (0, 1) and (2, -1), i.e. a
   forward difference with step 1e-3 of the prior sd, against a central difference with the same
   step, at the Table 1 values, 900 K and 20 atm and 1500 K and 1 atm (relative error).
Errors are relative 2-norm errors of the gradient in sd-scaled coordinates unless stated.

Check rows (ROWS): 0, 1, 2 (the first three of the design) and 105 (the smallest Su20 of the
500). The rows are fixed and the computation is deterministic. Needs numpy, scipy and Cantera (the
shipped data were written with Cantera 3.2.0); without Cantera the script stops with an error.
Run it from any directory; it reads the shipped files next to it and writes nothing (except, with
--save DIR outside this directory, check 8's per-row results, so that an interrupted run resumes).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h2_flame_driver as D  # noqa: E402

ROWS = [0, 1, 2, 105]     # check rows: the first three of the design and the slowest Su20 flame
TIGHT = (1e-9, 1e-14)

# Bell et al. (2019) Table 1, p. 6: (label, Cantera equation, occurrence, part, A, beta, Ea [cal/mol])
TABLE1 = [
    ("R1", "H + O2 <=> O + OH", 0, "", 1.04e14, 0, 15286),
    ("R2", "H2 + O <=> H + OH", 0, "", 3.818e12, 0, 7948),
    ("R2'", "H2 + O <=> H + OH", 1, "", 8.792e14, 0, 19170),
    ("R3", "H2 + OH <=> H + H2O", 0, "", 2.16e8, 1.51, 3430),
    ("R4", "2 OH <=> H2O + O", 0, "", 3.34e4, 2.42, -1930),
    ("R5", "H2 + M <=> 2 H + M", 0, "", 4.577e19, -1.40, 104380),
    ("R5 Ar", "H2 + AR <=> 2 H + AR", 0, "", 5.84e18, -1.10, 104380),
    ("R5 He", "H2 + HE <=> 2 H + HE", 0, "", 5.84e18, -1.10, 104380),
    ("R6", "2 O + M <=> O2 + M", 0, "", 6.165e15, -0.5, 0),
    ("R6 Ar", "2 O + AR <=> O2 + AR", 0, "", 1.886e13, 0, -1788),
    ("R6 He", "2 O + HE <=> O2 + HE", 0, "", 1.886e13, 0, -1788),
    ("R7", "H + O + M <=> OH + M", 0, "", 4.714e18, -1, 0),
    ("R8", "H2O + M <=> H + OH + M", 0, "", 6.064e27, -3.322, 120790),
    ("R8 H2O", "H2O + H2O <=> H + OH + H2O", 0, "", 1.006e26, -2.44, 120180),
    ("R9 inf", "H + O2 (+M) <=> HO2 (+M)", 0, "high", 4.65084e12, 0.44, 0),
    ("R9 0", "H + O2 (+M) <=> HO2 (+M)", 0, "low", 6.366e20, -1.72, 524.8),
    ("R10", "H + HO2 <=> H2 + O2", 0, "", 2.750e6, 2.09, -1451),
    ("R11", "H + HO2 <=> 2 OH", 0, "", 7.079e13, 0, 295),
    ("R12", "HO2 + O <=> O2 + OH", 0, "", 2.850e10, 1, -723.93),
    ("R13", "HO2 + OH <=> H2O + O2", 0, "", 2.890e13, 0, -497),
    ("R14", "2 HO2 <=> H2O2 + O2", 0, "", 4.200e14, 0, 11982),
    ("R14'", "2 HO2 <=> H2O2 + O2", 1, "", 1.300e11, 0, -1630),
    ("R15 inf", "H2O2 (+M) <=> 2 OH (+M)", 0, "high", 2.00e12, 0.9, 48749),
    ("R15 0", "H2O2 (+M) <=> 2 OH (+M)", 0, "low", 2.49e24, -2.3, 48749),
    ("R16", "H + H2O2 <=> H2O + OH", 0, "", 2.410e13, 0, 3970),
    ("R17", "H + H2O2 <=> H2 + HO2", 0, "", 4.820e13, 0, 7950),
    ("R18", "H2O2 + O <=> HO2 + OH", 0, "", 9.550e6, 2, 3970),
    ("R19", "H2O2 + OH <=> H2O + HO2", 0, "", 1.740e12, 0, 318),
    ("R19'", "H2O2 + OH <=> H2O + HO2", 1, "", 7.590e13, 0, 7270),
    ("X1", "H + HO2 <=> H2O + O", 0, "", 3.97e12, 0, 671),
    ("X6", "O + OH + M <=> HO2 + M", 0, "", 8.000e15, 0, 0),
]
TABLE1_EFF = {
    "H2 + M <=> 2 H + M": {"H2": 2.5, "H2O": 12, "AR": 0, "HE": 0},
    "2 O + M <=> O2 + M": {"H2": 2.5, "H2O": 12, "AR": 0, "HE": 0},
    "H + O + M <=> OH + M": {"H2": 2.5, "H2O": 12, "AR": 0.75, "HE": 0.75},
    "H2O + M <=> H + OH + M": {"H2": 3, "H2O": 0, "HE": 1.1, "N2": 2, "O2": 1.5},
    "H + O2 (+M) <=> HO2 (+M)": {"H2": 2.0, "H2O": 14, "O2": 0.78, "AR": 0.67, "HE": 0.8},
    "H2O2 (+M) <=> 2 OH (+M)": {"H2": 3.7, "H2O": 7.5, "H2O2": 7.7, "O2": 1.2, "N2": 1.5, "HE": 0.65},
    "O + OH + M <=> HO2 + M": {"H2": 2, "H2O": 12, "AR": 0.7, "HE": 0.7},
}
TABLE1_FC = {"H + O2 (+M) <=> HO2 (+M)": 0.5, "H2O2 (+M) <=> 2 OH (+M)": 0.43}
CAL = 4184.0          # J/kmol per cal/mol


def check_mechanism():
    ct = D.ct
    g = ct.Solution(D.MECH)
    eqs = [g.reaction(j).equation for j in range(g.n_reactions)]
    worst, diffs = 0.0, []
    for lab, eq, occ, part, A, b, Ea in TABLE1:
        r = g.reaction([j for j, e in enumerate(eqs) if e == eq][occ])
        u = r.rate_coeff_units
        L, Q = u.dimension("length"), u.dimension("quantity")
        rate = r.rate
        if part == "low":
            rate, L, Q = rate.low_rate, L + 3, Q - 1
        elif part == "high":
            rate = rate.high_rate
        A_cgs = rate.pre_exponential_factor * 100.0**L * 1000.0**Q
        vals = [(A_cgs, A), (rate.temperature_exponent, b), (rate.activation_energy / CAL, Ea)]
        for (v, ref), what in zip(vals, ("A", "beta", "Ea")):
            d = abs(v - ref) / max(abs(ref), 1e-300) if ref != 0 else abs(v)
            if d > 1e-9:
                diffs.append(f"{lab} {what}: file {v:.6g}, Table 1 {ref:.6g} (relative {d:.1e})")
            worst = max(worst, d) if ref != 0 else worst
    for eq, eff in TABLE1_EFF.items():
        r = g.reaction(eqs.index(eq))
        fe = r.third_body.efficiencies
        for sp, v in eff.items():
            if abs(fe.get(sp, r.third_body.default_efficiency) - v) > 1e-12:
                diffs.append(f"{eq} efficiency {sp}: file {fe.get(sp)}, Table 1 {v}")
        extra = sorted(set(fe) - set(eff))
        if extra:
            diffs.append(f"{eq}: efficiencies in the file but not printed in Table 1: {extra}")
    for eq, fc in TABLE1_FC.items():
        r = g.reaction(eqs.index(eq))
        if abs(r.rate.falloff_coeffs[0] - fc) > 1e-12:
            diffs.append(f"{eq} Troe A: file {r.rate.falloff_coeffs[0]}, Table 1 {fc}")
    print(f"   {len(TABLE1)} Arrhenius expressions, {sum(len(e) for e in TABLE1_EFF.values())} "
          f"efficiencies, 2 Troe Fc compared; largest relative difference {worst:.1e}")
    for d in diffs:
        print("   differs: " + d)
    # the 31 inputs: Table 1 value in the driver = value in the file
    m = D.H2Kinetics()
    print("   31 inputs: every nominal value in h2_flame_driver.PARAMS reproduces the file's rate "
          "(rebuild of each reaction at the nominal point changes no rate): "
          + str(_nominal_rebuild_ok(m)))


def _nominal_rebuild_ok(m):
    ct = D.ct
    g0 = ct.Solution(D.MECH)
    g1 = m.make_gas(D.NOMINAL)
    X = "H:0.1, O:0.1, OH:0.1, HO2:0.1, H2:1, O2:1, H2O:1, AR:1, HE:1, N2:1, H2O2:0.1"
    for T, P in ((800.0, 1e5), (1500.0, 2e6)):
        g0.TPX = T, P, X
        g1.TPX = T, P, X
        if np.max(np.abs(g1.forward_rates_of_progress / g0.forward_rates_of_progress - 1)) > 1e-12:
            return False
    return True


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=",", ndmin=2)


def rel2(a, b):
    s = D.PRIOR_SD
    return np.linalg.norm((a - b) * s) / np.linalg.norm(b * s)


def flame_fd(F, x, rel=1e-2):
    """Adjoint (tight) and central finite differences of Su on the converged grid."""
    su, st = F.solve(x)
    base = F.flame
    base.flame.set_steady_tolerances(default=TIGHT)
    base.solve(loglevel=0, refine_grid=False, auto=False)
    g_adj = F.adjoint_grad()
    # Cantera's own adjoint for reaction multipliers, against ours for the plain A factors
    s_ct = np.asarray(base.get_flame_speed_reaction_sensitivities())
    su_t = float(base.velocity[0])
    kA = [k for k in range(D.M_INPUTS) if D.PARAMS[k][2] == "A"]
    ours = np.array([g_adj[k] * x[k] / su_t for k in kA])
    theirs = np.array([s_ct[F.index[D.PARAMS[k][1]]] for k in kA])
    ct_diff = np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs))
    arr, mdot = base.to_array(), base.inlet.mdot
    gas = base.gas
    fd = np.empty(D.M_INPUTS)
    for k in range(D.M_INPUTS):
        h = rel * D.PRIOR_SD[k]
        vals = []
        for s in (+1, -1):
            xp = x.copy()
            xp[k] = x[k] + s * h
            if D.PARAMS[k][2] == "eff":      # efficiencies: a new gas, restored solution
                f = F.new_flame(F.make_gas(xp))
                f.from_array(arr)
                f.inlet.mdot = mdot
                f.flame.set_steady_tolerances(default=TIGHT)
                f.solve(loglevel=0, refine_grid=False, auto=False)
                vals.append(float(f.velocity[0]))
            else:                            # A factors: in place on the converged flame
                F.set_A(gas, x, k, xp[k])
                base.solve(loglevel=0, refine_grid=False, auto=False)
                vals.append(float(base.velocity[0]))
                F.set_A(gas, x, k, x[k])
        fd[k] = (vals[0] - vals[1]) / (2 * h)
    base.solve(loglevel=0, refine_grid=False, auto=False)
    return su, g_adj, fd, ct_diff


def check_nominal(ign, fl, nom):
    print("\n2. QoIs at the Table 1 values (nominal.csv in parentheses)")
    tau0 = ign.tau(D.NOMINAL)
    print(f"   tau  = {tau0 * 1e3:.6f} ms ({nom[0] * 1e3:.6f}); threshold T = {ign.T_c:.2f} K, "
          f"T_eq = {ign.T_eq:.2f} K")
    for j, c in enumerate(("Su1", "Su20")):
        su, st = fl[c].solve(D.NOMINAL)
        print(f"   {c} = {su:.6f} m/s ({nom[1 + j]:.6f}), status {st}, {len(fl[c].flame.grid)} points")


def check_grid(fl, X):
    print("\n3. flame speed with finer grid criteria (slope 0.01, curve 0.02): relative change")
    fine = {"ratio": 2.0, "slope": 0.01, "curve": 0.02, "prune": 0.0}
    for c in ("Su1", "Su20"):
        Ff = D.FlameSpeed(c, refine=fine)
        out = []
        for lab, x in [("nominal", D.NOMINAL)] + [(f"row {i}", X[i]) for i in ROWS]:
            s0, _ = fl[c].solve(x)
            s1, _ = Ff.solve(x)
            out.append(f"{lab} {s1 / s0 - 1:+.1e} ({len(Ff.flame.grid)} points)")
        print(f"   {c}: " + ", ".join(out))


def check_ignition(ign, X, F_, G):
    print("\n4. ignition delay at the check rows")
    tight = D.IgnitionDelay(rtol=1e-12, atol=1e-22)
    for i in ROWS:
        x = X[i]
        t_t = tight.tau(x)
        g_s = ign.grad(x, rel=D.FD_REL / 10)
        base = ign.tau(x)
        quarter = D.IgnitionDelay()
        quarter.T_c = D.IGNITION["T0"] + 0.25 * (quarter.T_eq - D.IGNITION["T0"])
        three = D.IgnitionDelay()
        three.T_c = D.IGNITION["T0"] + 0.75 * (three.T_eq - D.IGNITION["T0"])
        print(f"   row {i} (tau {F_[i, 0] * 1e3:.4f} ms): tight solve {abs(t_t / F_[i, 0] - 1):.1e}; "
              f"gradient with step/10 {rel2(g_s, G['tau'][i]):.1e}; threshold 1/4 "
              f"{quarter.tau(x) / base - 1:+.2e}, 3/4 {three.tau(x) / base - 1:+.2e}")


def compare_adjoint(adj, stored):
    """|stored| / |adjoint|, cosine and relative error of the adjoint, sd-scaled."""
    a_s, s_s = adj * D.PRIOR_SD, stored * D.PRIOR_SD
    return (np.linalg.norm(s_s) / np.linalg.norm(a_s),
            a_s @ s_s / np.linalg.norm(a_s) / np.linalg.norm(s_s), rel2(adj, stored))


def fd_checks(F, c, X, F_, stored_all):
    j = 1 if c == "Su1" else 2
    for i in ROWS:
        x = X[i]
        stored = stored_all[i]
        r = F.fd_grad(x)
        g = r["grad"]
        _, adj_t, fd2, ct_diff = flame_fd(F, x)
        F.solve(x)
        ratio, cos, err = compare_adjoint(F.adjoint_grad(), stored)
        print(f"   row {i} ({c} {F_[i, j]:.4g} m/s): recomputed vs stored {rel2(g, stored):.1e}; step check "
              f"(10x step) {rel2(r['grad_central10'], g):.1e}; forward vs central {rel2(r['grad_forward'], g):.1e}; "
              f"max drift {np.max(np.abs(r['drift'])):.1e}; independent FD (flame_fd, step 1e-2 sd) vs stored "
              f"{rel2(fd2, stored):.1e}; adjoint: |stored| / |adjoint| {ratio:.3f}, cosine {cos:.5f}, relative "
              f"error {err:.1e}; adjoint at tight tolerances vs stored {rel2(adj_t, stored):.1e}; driver's adjoint "
              f"vs Cantera's reaction sensitivities (13 plain A factors, relative to the largest) {ct_diff:.1e}")


def adjoint_rows(fl, X, F_, G, rows, flames=("Su1", "Su20"), save=None):
    for c in flames:
        F = fl[c]
        done = {}
        fn = None if save is None else os.path.join(save, f"check8_{c}.txt")
        if fn is not None and os.path.exists(fn):
            for line in open(fn):
                v = line.split()
                done[int(v[0])] = tuple(float(s) for s in v[1:])
        for i in rows:
            if i in done or not np.all(np.isfinite(G[c][i])):
                continue
            F.solve(X[i])
            done[i] = compare_adjoint(F.adjoint_grad(), G[c][i])
            if fn is not None:
                with open(fn, "a") as fh:
                    fh.write("%d %.17g %.17g %.17g\n" % ((i,) + done[i]))
        res = [(i,) + done[i] for i in rows if i in done]
        R = np.array(res)
        rows_, ratio, cos, err = R[:, 0].astype(int), R[:, 1], R[:, 2], R[:, 3]
        p = lambda v, f="%.3g": ("median {0}, 5-95 %% {0} to {0}, min {0} (row %d), max {0} (row %d)".format(f)  # noqa: E731
                                % (np.median(v), *np.percentile(v, [5, 95]), v.min(), rows_[np.argmin(v)], v.max(),
                                   rows_[np.argmax(v)]))
        print(f"   {c} ({len(R)} rows): |stored FD| / |adjoint| {p(ratio)}")
        print(f"   {c}: cosine {p(cos, '%.5f')}")
        print(f"   {c}: relative 2-norm error of the adjoint {p(err)}")


def plain_flames():
    """Cantera alone: R9 elasticity of Su, adjoint against fixed-grid central differences."""
    ct = D.ct
    eq = "H + O2 (+M) <=> HO2 (+M)"
    coarse = {"ratio": 3.0, "slope": 0.06, "curve": 0.12}
    data = {k: D.REFINE[k] for k in ("ratio", "slope", "curve")}
    for c in ("Su20", "Su1"):
        cond = D.FLAMES[c]
        for lab, ref in (("coarse", coarse), ("data", data)):
            g = ct.Solution(D.MECH)
            j = [r.equation for r in g.reactions()].index(eq)
            g.TP = cond["T"], cond["P_atm"] * ct.one_atm
            g.set_equivalence_ratio(cond["phi"], "H2", cond["oxidizer"])
            f = ct.FreeFlame(g, width=cond["width"])
            f.transport_model = D.TRANSPORT
            f.set_refine_criteria(**ref)
            f.solve(loglevel=0, auto=True)
            f.flame.set_steady_tolerances(default=TIGHT)
            f.solve(loglevel=0, refine_grid=False, auto=False)
            su = float(f.velocity[0])
            adj = float(np.asarray(f.get_flame_speed_reaction_sensitivities())[j])
            fd = []
            for e in (1e-3, 1e-2):
                v = []
                for m in (1 + e, 1 - e):
                    g.set_multiplier(m, j)
                    f.solve(loglevel=0, refine_grid=False, auto=False)
                    v.append(float(f.velocity[0]))
                g.set_multiplier(1.0, j)
                fd.append((np.log(v[0]) - np.log(v[1])) / (np.log(1 + e) - np.log(1 - e)))
            print(f"   {c} ({cond['P_atm']:g} atm, phi {cond['phi']:g}), {lab} grid criteria {ref}: Su {su:.6g} m/s, "
                  f"{len(f.grid)} points; d ln Su / d ln k(R9): adjoint {adj:+.4f}, fixed-grid central "
                  f"differences {fd[0]:+.4f} (multiplier step 1e-3; {fd[1]:+.4f} with 1e-2), ratio {fd[0] / adj:.2f}")


def eff_truncation():
    """Relative error of fd_grad's efficiency difference for falloff reactions (rate constants only)."""
    K = D.H2Kinetics()
    states = ((900.0, 20 * 101325.0, "H2:0.1, O2:0.15, N2:0.65, H2O:0.1, H:1e-4, HO2:1e-4, OH:1e-4, H2O2:1e-4"),
              (1500.0, 101325.0, "H2:0.05, O2:0.02, N2:0.65, H2O:0.28, H:1e-3, HO2:1e-5, OH:1e-3, H2O2:1e-6"))
    for name in ("R9_eff_H2", "R9_eff_H2O", "R15_eff_H2", "R15_eff_H2O"):
        k = D.NAMES.index(name)
        lab = D.PARAMS[k][1]
        d = D.SHADOW_REL * D.PRIOR_SD[k]
        out = []
        for T, P, Xs in states:
            q = {}
            for s in (-1, 0, 1):
                x = D.NOMINAL.copy()
                x[k] += s * d
                g = K.make_gas(x)
                g.TPX = T, P, Xs
                q[s] = g.forward_rate_constants[K.index[lab]]
            out.append(f"{T:.0f} K, {P / 101325:.0f} atm {((q[1] - q[0]) / d) / ((q[1] - q[-1]) / (2 * d)) - 1:+.1e}")
        print(f"   {name}: " + "; ".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="checks 1-3 only")
    ap.add_argument("--checks", default=None, help="comma-separated check numbers (1-10)")
    ap.add_argument("--rows", default="0:500", help="rows a:b for check 8 (default all)")
    ap.add_argument("--flames", default="Su1,Su20", help="flame QoIs for check 8 (default Su1,Su20)")
    ap.add_argument("--check-rows", default=None, help="comma-separated rows for checks 3-6 (default 0,1,2,105)")
    ap.add_argument("--save", default=None, help="directory outside this one: check 8 keeps its per-row "
                    "results there and resumes from them")
    a = ap.parse_args()
    global ROWS
    if a.check_rows:
        ROWS = [int(s) for s in a.check_rows.split(",")]
    if a.save is not None:
        sv = os.path.abspath(a.save)
        here = os.path.realpath(HERE)
        if os.path.commonpath([os.path.realpath(sv), here]) == here:
            sys.exit("--save DIR must lie outside this directory")
        os.makedirs(sv, exist_ok=True)
    if a.checks:
        todo = {int(s) for s in a.checks.split(",")}
    else:
        todo = {1, 2, 3} if a.quick else set(range(1, 8))
    if D.ct is None:
        sys.exit("Cantera is not installed. Install it (pip install cantera==3.2.0, the version that "
                 "wrote the shipped files) and rerun check_accuracy.py.")
    t0 = time.perf_counter()
    print(f"Cantera {D.ct.__version__}, numpy {np.__version__}")
    X, F_ = load("inputs.csv"), load("outputs.csv")
    G = {q: load(f"gradients_{q}.csv") for q in ("tau", "Su1", "Su20")}
    nom = load("nominal.csv")[0]
    if 1 in todo:
        print("\n1. mechanism file against Table 1")
        check_mechanism()
    fl = {c: D.FlameSpeed(c) for c in ("Su1", "Su20")}
    ign = D.IgnitionDelay()
    if 2 in todo:
        check_nominal(ign, fl, nom)
    if 3 in todo:
        check_grid(fl, X)
    if 4 in todo:
        check_ignition(ign, X, F_, G)
    for num, c in ((5, "Su1"), (6, "Su20")):
        if num in todo:
            print(f"\n{num}. {c} gradients (central finite differences) at the check rows (sd-scaled)")
            fd_checks(fl[c], c, X, F_, G[c])
    if 7 in todo:
        print("\n7. row 412: Su20 re-solved from scratch")
        F20 = fl["Su20"]
        su, st = F20.solve(X[412])
        print(f"   Su20 {su:.10g} m/s (status {st}, {len(F20.flame.grid)} points); outputs.csv {F_[412, 2]:.10g}; "
              f"relative difference {abs(su / F_[412, 2] - 1):.1e}")
    if 8 in todo:
        i, j = (int(s) for s in a.rows.split(":"))
        print(f"\n8. Cantera's adjoint against the stored finite-difference gradients, rows {i}:{min(j, len(X))}")
        adjoint_rows(fl, X, F_, G, range(i, min(j, len(X))), flames=[s.strip() for s in a.flames.split(",")],
                     save=a.save and os.path.abspath(a.save))
    if 9 in todo:
        print("\n9. adjoint defect against the grid, plain Cantera FreeFlame at the Table 1 values")
        plain_flames()
    if 10 in todo:
        print("\n10. efficiency scheme of fd_grad for falloff reactions: relative error of the rate derivative")
        eff_truncation()
    print(f"\nrun time {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
