"""Recompute the numbers of the Accuracy section of DATA.md (needs OpenDSSDirect.py 0.9.4).

    python check_accuracy.py            # all checks (about 1 min on one core)
    python check_accuracy.py --quick    # the nominal point and the solver-tolerance check only

Checks
  1. Nominal point: the three QoIs and the three phase voltages at bus 899, with the solution
     tolerance used for the data (1e-12) and with OpenDSS's default (1e-4); a cold start
     (fresh engine) against a warm start.
  2. Gradients at three shipped rows of each case: relative 2-norm difference between the shipped
     central differences (step 0.01 sd) and (a) steps of 0.001 sd and 0.1 sd, (b) a fourth-order
     central difference with step 0.1 sd.
  3. Load-model switch: OpenDSS's constant-P load becomes a constant impedance above 1.05 p.u.;
     for the 1000 shipped 55-input rows, the number of loads above 1.05 p.u. and the smallest
     distance of any load voltage from 1.05 p.u.; a finite-difference pair can straddle the switch
     only where that distance is small, so the rows closer than 1e-5 p.u. are rechecked with a
     step of 0.001 sd.
  4. The four readings of the paper among which Table VIII selects, by 2000 Monte Carlo samples (seed 1):
     sd of P = 0.1 kW or 0.1 P_Li; loads as distributed (Vmaxpu 1.05) or constant P throughout
     (Vminpu 0.8, Vmaxpu 1.2). Printed: mean and sd of V899A and the three probabilities of the
     paper's Table VIII, for all four combinations.
  5. Negative load powers: N(P_Li, 0.1^2) gives negative powers (injections). The 10 000 shipped
     Latin hypercube rows are rerun (a) with every negative power set to 0 and (b) with the same
     stratified uniforms mapped through the normal truncated at 0; printed with the shipped values.
  6. Snapshot set-up: a yearly-mode solve of the unchanged Master.dss (1-min steps, 566 steps,
     ending at point 566, 09:26) against the driver's nominal bus-899 voltages.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import opendss_driver as M  # noqa: E402

ROWS = [0, 1, 2]


def load(fn):
    return np.loadtxt(os.path.join(HERE, fn), delimiter=",", ndmin=2)


def bus899(dss):
    dss.Circuit.SetActiveBus("899")
    vm = dss.Bus.VMagAngle()[0::2]
    nodes = dss.Bus.Nodes()
    return [vm[nodes.index(k)] for k in (1, 2, 3)]


def nominal_checks():
    dss = M.dss
    m = M.FeederModel()
    f = m.solve(M.P_REF)
    va, vb, vc = bus899(dss)
    print(f"1. nominal point (tol 1e-12): V899A = {f[0]:.6f} V, I78A = {f[1]:.6f} A, "
          f"PLOSS = {f[2]:.6f} kW; bus 899 phases A, B, C = {va:.3f}, {vb:.3f}, {vc:.3f} V")
    m.solve(M.P_REF * 1.2)                     # move away, then come back (warm start)
    fw = m.solve(M.P_REF)
    m2 = M.FeederModel()                       # fresh engine (cold start)
    fc = m2.solve(M.P_REF)
    print(f"   warm vs cold start: max relative difference {np.max(np.abs(fw - fc) / np.abs(fc)):.1e}")
    m3 = M.FeederModel(tol=1e-4, maxiter=15)
    m3.solve(M.P_REF * 1.2)
    fd = m3.solve(M.P_REF)
    print(f"   OpenDSS default tolerance 1e-4 (after a warm start from 1.2 P_L): V899A differs by "
          f"{fd[0] - f[0]:+.2e} V, I78A by {fd[1] - f[1]:+.2e} A, PLOSS by {fd[2] - f[2]:+.2e} kW")
    return M.FeederModel()                     # OpenDSSDirect has one engine: rebuild at tol 1e-12


def gradient_checks(m):
    for case, fun, steps0, sd in [("55", m.qoi55, M.STEPS55, M.SD55), ("110", m.qoi110, M.STEPS110, M.SD110)]:
        X = load(f"inputs_{case}.csv")
        G = np.stack([load(f"grad_{q}_{case}.csv") for q in M.QOI_NAMES])   # (3, M, m)
        for r in ROWS:
            x = X[r]
            g0 = G[:, r, :] * sd                                             # normalized by sd
            out = []
            for fac in (0.1, 10.0):
                g = m.grad(fun, x, steps0 * fac) * sd
                out.append(np.linalg.norm(g - g0, axis=1) / np.linalg.norm(g0, axis=1))
            h = 0.1 * sd
            g4 = np.empty((3, len(x)))
            for j in range(len(x)):
                e = np.zeros(len(x))
                e[j] = h[j]
                g4[:, j] = (8 * (fun(x + e) - fun(x - e)) - (fun(x + 2 * e) - fun(x - 2 * e))) / (12 * h[j])
            out.append(np.linalg.norm(g4 * sd - g0, axis=1) / np.linalg.norm(g0, axis=1))
            labels = ["step 0.001 sd", "step 0.1 sd", "4th order, step 0.1 sd"]
            print(f"2. case {case}, row {r}: relative difference (V899A / I78A / PLOSS): "
                  + "; ".join(f"{lab} " + "/".join(f"{v:.1e}" for v in o) for lab, o in zip(labels, out)))


def switch_checks(m):
    X = load("inputs_55.csv")
    dist = np.empty(len(X))
    above = np.empty(len(X), int)
    for k, x in enumerate(X):
        m.solve(x)
        v = m.load_voltages_pu()
        dist[k] = np.min(np.abs(v - 1.05))
        above[k] = np.sum(v > 1.05)
    near = np.where(dist < 1e-5)[0]
    G = np.stack([load(f"grad_{q}_55.csv") for q in M.QOI_NAMES])
    worst = np.zeros(3)
    for r in near:
        g = m.grad(m.qoi55, X[r], M.STEPS55 * 0.1)
        worst = np.maximum(worst, np.linalg.norm(g - G[:, r, :], axis=1) / np.linalg.norm(G[:, r, :], axis=1))
    m.solve(M.P_REF)
    v0 = m.load_voltages_pu()
    print(f"3. load voltages at the nominal point: {np.sum(v0 > 1.05)} of {M.NLOAD} above 1.05 p.u. "
          f"(range {v0.min():.4f}-{v0.max():.4f} p.u.)")
    print(f"   1000 shipped 55-input rows: {above.min()}-{above.max()} loads above 1.05 p.u.; smallest "
          f"distance of a load voltage from 1.05 p.u. {dist.min():.1e} p.u.; rows with a distance below "
          f"1e-5 p.u.: {np.sum(dist < 1e-5)}, below 1e-4 p.u.: {np.sum(dist < 1e-4)}")
    print(f"   at the {len(near)} rows closer than 1e-5 p.u., shipped gradients against step 0.001 sd: largest "
          f"relative difference (V899A / I78A / PLOSS) " + "/".join(f"{v:.1e}" for v in worst))


def alternative_checks():
    dss = M.dss
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((2000, M.NLOAD))
    for label, vmax in [("loads as distributed (Vmaxpu 1.05)", None), ("constant P (Vmaxpu 1.2)", 1.2)]:
        m = M.FeederModel()
        if vmax is not None:
            for n in M.LOAD_NAMES:
                dss.Loads.Name(n)
                dss.Loads.Vmaxpu(vmax)
                dss.Loads.Vminpu(0.8)
        v0 = m.solve(M.P_REF)[0]
        for sl, S in [("sd 0.1 kW", M.SD55), ("sd 0.1 P_Li", 0.1 * M.P_REF)]:
            V = np.array([m.solve(M.P_REF + S * z)[0] for z in Z])
            print(f"4. {label}, {sl}: nominal V899A {v0:.3f} V; mean {V.mean():.3f} V, sd {V.std(ddof=1):.3f} V; "
                  f"P(V <= 250) {100 * np.mean(V <= 250):.2f} %, P(250.4 < V <= 250.8) "
                  f"{100 * np.mean((V > 250.4) & (V <= 250.8)):.2f} %, P(V > 250.8) {100 * np.mean(V > 250.8):.2f} %")


def table8(V):
    return (100 * np.mean(V <= 250), 100 * np.mean((V > 250.4) & (V <= 250.8)), 100 * np.mean(V > 250.8))


def negative_load_checks():
    from scipy.stats import norm
    m = M.FeederModel()
    XL = load("lhs_inputs_55.csv")
    FL = load("lhs_outputs_55.csv")
    neg = XL < 0
    print(f"5. Latin hypercube rows: {100 * neg.mean():.3f} % of the P_i negative, {neg.sum(1).mean():.2f} "
          f"negative loads per row on average, smallest P_i {XL.min():.3f} kW")
    U = norm.cdf((XL - M.MEAN55) / M.SD55)
    a = norm.cdf(-M.P_REF / M.SD_P)
    cases = [("as shipped (negative powers kept)", FL[:, 0], XL),
             ("negative powers set to 0", None, np.maximum(XL, 0.0)),
             ("normal truncated at 0", None, M.P_REF + M.SD_P * norm.ppf(a + U * (1 - a)))]
    v_ship = FL[:, 0].mean()
    for label, V, X in cases:
        if V is None:
            V = np.array([m.solve(x)[0] for x in X])
        print(f"   {label}: mean V899A {V.mean():.4f} V (shift {V.mean() - v_ship:+.4f} V); "
              "P(V <= 250) {:.2f} %, P(250.4 < V <= 250.8) {:.2f} %, P(V > 250.8) {:.2f} %".format(*table8(V)))


def yearly_check():
    dss = M.dss
    m = M.FeederModel()
    m.solve(M.P_REF)
    v_drv = bus899(dss)
    dss.Text.Command("clear")
    cwd = os.getcwd()
    try:
        dss.Text.Command(f'redirect "{M.MASTER}"')
    finally:
        os.chdir(cwd)
    dss.Text.Command("set mode=yearly number=566 stepsize=1m")
    dss.Solution.Solve()
    hour = dss.Solution.Hour() + dss.Solution.Seconds() / 3600.0
    v_yr = bus899(dss)
    print(f"6. yearly-mode solve of the unchanged Master.dss, 566 one-minute steps (ends at hour {hour:.4f}): "
          f"bus 899 phases A, B, C = {v_yr[0]:.3f}, {v_yr[1]:.3f}, {v_yr[2]:.3f} V; driver at the nominal point "
          f"{v_drv[0]:.3f}, {v_drv[1]:.3f}, {v_drv[2]:.3f} V; largest difference "
          f"{max(abs(x - y) for x, y in zip(v_yr, v_drv)):.1e} V")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if M.dss is None:
        sys.exit(f"OpenDSSDirect.py is not installed. Install it ({M.INSTALL}) and rerun check_accuracy.py.")
    m = nominal_checks()
    if a.quick:
        return
    gradient_checks(m)
    switch_checks(m)
    alternative_checks()
    negative_load_checks()
    yearly_check()


if __name__ == "__main__":
    main()
