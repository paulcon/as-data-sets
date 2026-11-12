"""Recompute the accuracy numbers quoted in DATA.md for NH3_H2_Kinetics_Uncertainty (needs Cantera).

    python check_accuracy.py            # all checks, about 37 min on one core
    python check_accuracy.py --quick    # checks 1-3 only (about 2 min)
    python check_accuracy.py --flames   # checks 4-5 only
    python check_accuracy.py --adjoint  # check 6 only (about 8 min)
    python check_accuracy.py --path     # check 7 only (about 3.5 min)

Prints, beside each value DATA.md quotes:

1. tau and Su of the nominal model (x = 0) against the stored nominal.csv.
2. The ignition criterion: relative change of tau for a temperature rise of 200 and 800 K instead
   of the 400 K used for the data, at the nominal point and at the shortest and longest stored
   tau of the ignition design, and the constant-volume equilibrium temperature rise. Then the
   time of the maximum of dT/dt (the inflection of T) as an alternative criterion, at the nominal
   point and the shortest, median and longest tau: the change of tau, and the cosine and norm
   ratio of its log10-tau gradient against the gradient for the 400 K rise (both by central
   differences, step 1e-2, normalized coordinates); and the same cosine for the 800 K rise at the
   shortest tau.
3. Ignition delay: tau re-solved at rtol 1e-12, atol 1e-20 at three stored rows (the shortest,
   median and longest tau), and the stored forward-difference gradients against central
   differences (step 1e-3) at rtol 1e-12: relative 2-norm error of the gradient in normalized
   coordinates (each component times f_i).
4. Flame speed, gradient: the stored adjoint gradients at the nominal point and at two stored rows
   (the slowest and the fastest flame) against central differences on the same grid (step 1e-2,
   each perturbed flame solved twice from the unperturbed one with rtol 1e-9): relative 2-norm
   error in normalized coordinates, the cosine between the two vectors and the ratio of their norms.
   Each point uses a fresh model instance, so that the flame restarts from the nominal flame only.
5. Flame speed, grid: Su at the nominal point and at the two rows of check 4 with the paper's
   refinement criteria (ratio 2, slope 0.01, curve 0.02; paper p. 4) against the stored values
   (ratio 3, slope 0.1, curve 0.2), and, at the nominal point, the cosine between the adjoint
   gradients on the two grids. A flame that needs more than Cantera's default limit of 1000 grid
   points on the finer grid is reported as such.
6. Flame speed, size of the adjoint gradient across the design: at 21 flame rows (ranks 0, 10, ...,
   190 and 199 of the stored Su, each in a fresh model instance), the adjoint directional
   derivative against central differences on the same grid (steps 1e-2 and 3e-3 in normalized
   coordinates, perturbed flames solved twice at rtol 1e-9) along the adjoint direction and along
   one random direction (seed 6). A fit 1 - ratio = a Su^b over these rows, the share of the 200
   design rows it puts more than 1, 3 and 5 % short, and the subspace of Su with every stored
   gradient divided by the fitted ratio, against the stored one (eigenvalue ratios, |w_j . w_j'|,
   distance of the 2-dimensional subspaces).
7. Flame speed, dependence on the solver history: rows 24, 26 and 187 solved in a fresh model
   instance and, for row 24, solved a second time in the same instance, against the stored values;
   and the derivative along the normalized adjoint direction from flames refined on their own
   grids (central differences, step 2e-2, each flame in a fresh instance) against the fixed-grid
   adjoint derivative, at the nominal point and at row 192.

Rows are fixed by rules on the stored files, and the computation is deterministic. Needs numpy,
scipy, Cantera (the data were written with Cantera 3.2.0) and the mechanism file NH3_H2.yaml from
the paper's supplement (see nh3h2_kinetics_driver.py); without them the script stops with an
error. Run it from any directory; it reads the shipped files next to it and writes nothing.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nh3h2_kinetics_driver as D  # noqa: E402

PAPER_REFINE = {"ratio": 2.0, "slope": 0.01, "curve": 0.02}


def load(case):
    ld = lambda n: np.loadtxt(os.path.join(HERE, n), delimiter=",", ndmin=2)  # noqa: E731
    return ld(f"{case}_inputs.csv"), ld(f"{case}_outputs.csv")[:, 0], ld(f"{case}_gradients.csv")


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def cos(a, b):
    return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -- check 2: the time of the maximum of dT/dt ------------------------------------------------
def tau_max_dtdt(model, x, npts=800):
    """Time of the maximum of dT/dt in the constant-volume reactor of the driver.

    A first pass with the solver's own steps brackets the peak (the interval where dT/dt exceeds
    half its maximum); a second pass reads the state at npts equally spaced times over three
    half-widths around the peak, and a parabola through the largest value and its neighbours
    gives the time of the maximum."""
    ct = D.ct
    model.set_x(x)
    gas = model.gas
    saved = gas.state

    def start():
        gas.TPX = D.IDT_COND["T"], D.IDT_COND["P"], D.IDT_COND["X"]
        r = ct.IdealGasReactor(gas, clone=False)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = model.rtol, model.atol
        net.max_steps = 500000
        return r, net

    def tdot(r):
        g = r.phase
        return -np.dot(g.partial_molar_int_energies, g.net_production_rates) / (g.density * g.cv_mass)

    try:
        r, net = start()
        ts, ds = [], []
        dmax = 0.0
        while True:
            t = net.step()
            d = tdot(r)
            ts.append(t)
            ds.append(d)
            dmax = max(dmax, d)
            if r.T > D.IDT_COND["T"] + 400.0 and d < 0.05 * dmax:
                break
            if t > D.IDT_TMAX:
                raise RuntimeError("no ignition")
        ts, ds = np.array(ts), np.array(ds)
        k = int(np.argmax(ds))
        above = np.nonzero(ds >= 0.5 * ds[k])[0]
        t_lo, t_hi = ts[max(above[0] - 1, 0)], ts[min(above[-1] + 1, len(ts) - 1)]
        tk = ts[k]
        grid = np.linspace(tk - 1.5 * (tk - t_lo), tk + 1.5 * (t_hi - tk), npts)
        r, net = start()
        dg = np.empty(npts)
        for j, t in enumerate(grid):
            net.advance(t)
            dg[j] = tdot(r)
        j = int(np.clip(np.argmax(dg), 1, npts - 2))
        y0, y1, y2 = dg[j - 1], dg[j], dg[j + 1]
        h = grid[1] - grid[0]
        return grid[j] + 0.5 * h * (y0 - y2) / (y0 - 2 * y1 + y2)
    finally:
        gas.state = saved


def grad_log10(fun, x, step=1e-2):
    """Central differences of log10 fun(x) in x, normalized coordinates (times f_i)."""
    g = np.empty(D.M_INPUTS)
    for i in range(D.M_INPUTS):
        e = np.zeros(D.M_INPUTS)
        e[i] = step
        g[i] = (np.log10(fun(x + e)) - np.log10(fun(x - e))) / (2 * step)
    return g * D.F_INI


# -- checks 6 and 7: flames on a fixed grid ---------------------------------------------------
def fixed_grid_value(model, f, state, x):
    """Su (m/s) at x on the grid of the flame state, solved twice at rtol 1e-9 (as lfs_grad_fd)."""
    model.set_x(x)
    f.from_array(state)
    f.solve(loglevel=0, refine_grid=False)
    f.solve(loglevel=0, refine_grid=False)
    return f.velocity[0]


def directional(model, x, dirs, steps=(1e-2, 3e-3)):
    """Fresh-instance flame at x: Su (cm/s), the adjoint gradient (cm/s per normalized unit) and,
    for each unit direction d (normalized coordinates) and step, the central difference of Su."""
    Su, g = model.lfs_grad(x)
    g = 100.0 * g * D.F_INI
    f = model._flame
    state = f.to_array()
    f.flame.set_steady_tolerances(default=(1e-9, 1e-15))
    out = []
    for d in dirs:
        d = g / np.linalg.norm(g) if d is None else d
        fd = []
        for h in steps:
            vp = fixed_grid_value(model, f, state, x + h * d * D.F_INI)
            vm = fixed_grid_value(model, f, state, x - h * d * D.F_INI)
            fd.append(100.0 * (vp - vm) / (2 * h))
        out.append((g @ d, fd))
    f.flame.set_steady_tolerances(default=(1e-4, 1e-9))
    model.set_x(x)
    f.from_array(state)
    return 100.0 * Su, g, out


def eig(G):
    lam, W = np.linalg.eigh(G.T @ G / len(G))
    return lam[::-1], W[:, ::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="only checks 1-3")
    ap.add_argument("--flames", action="store_true", help="only checks 4-5")
    ap.add_argument("--adjoint", action="store_true", help="only check 6")
    ap.add_argument("--path", action="store_true", help="only check 7")
    ap.add_argument("--mech", default=None, help="path of NH3_H2.yaml (default: see the driver)")
    a = ap.parse_args()
    if D.ct is None:
        print("Cantera is not installed. Install it (pip install cantera==3.2.0, the version that wrote "
              "the shipped files) and rerun check_accuracy.py.", file=sys.stderr)
        sys.exit(3)
    try:
        D.mech_path(a.mech)
    except FileNotFoundError as e:
        sys.exit(str(e))
    some = a.quick or a.flames or a.adjoint or a.path
    do = {k: not some for k in (1, 2, 3, 4, 5, 6, 7)}
    if a.quick:
        do.update({1: True, 2: True, 3: True})
    if a.flames:
        do.update({4: True, 5: True})
    if a.adjoint:
        do[6] = True
    if a.path:
        do[7] = True
    t0 = time.perf_counter()
    print(f"Cantera {D.ct.__version__}, numpy {np.__version__}")
    model = D.KineticsModel(mech=a.mech)
    tight = D.KineticsModel(mech=a.mech, rtol=1e-12, atol=1e-20)
    nom = np.loadtxt(os.path.join(HERE, "nominal.csv"), delimiter=",")
    Xi, ti, Gi = load("idt")
    ok = np.isfinite(ti)
    Xl, sl, Gl = load("lfs")
    okl = np.isfinite(sl)
    lo, hi = np.nanargmin(ti), np.nanargmax(ti)
    med = int(np.argsort(np.where(ok, ti, np.inf))[ok.sum() // 2])

    if do[1]:
        tau0 = model.idt()
        Su0, f0 = model.lfs()
        print(f"\n1. nominal model: tau = {tau0 * 1e3:.5f} ms (stored {nom[0, 0] * 1e3:.5f} ms), "
              f"Su = {100 * Su0:.4f} cm/s (stored {nom[1, 0]:.4f} cm/s), {len(f0.grid)} grid points")

    if do[2]:
        g = model.gas
        saved = g.state            # restored below: the flames of this instance take their pressure from g
        g.TPX = D.IDT_COND["T"], D.IDT_COND["P"], D.IDT_COND["X"]
        g.equilibrate("UV")
        dT_eq = g.T - D.IDT_COND["T"]
        g.state = saved
        print("\n2. ignition criterion: relative change of tau for a rise of 200 and 800 K instead of 400 K "
              f"(the constant-volume equilibrium rise at the nominal point is {dT_eq:.0f} K)")
        for label, x in [("nominal", D.NOMINAL), (f"shortest tau, row {lo}", Xi[lo]), (f"longest tau, row {hi}", Xi[hi])]:
            base = model.idt(x)
            d = [model.idt(x, t_rise=dt) / base - 1 for dt in (200.0, 800.0)]
            print(f"   {label} (tau = {base:.4g} s): " + ", ".join(f"{v:+.1e}" for v in d))
        print("   maximum of dT/dt instead of the 400 K rise (gradients of log10 tau, central differences, "
              "step 1e-2, normalized coordinates)", flush=True)
        for label, x in [("nominal", D.NOMINAL), (f"shortest tau, row {lo}", Xi[lo]),
                         (f"median tau, row {med}", Xi[med]), (f"longest tau, row {hi}", Xi[hi])]:
            base = model.idt(x)
            tm = tau_max_dtdt(model, x)
            g400 = grad_log10(model.idt, x)
            gmx = grad_log10(lambda z: tau_max_dtdt(model, z), x)
            line = (f"   {label}: tau(max dT/dt) / tau(400 K) - 1 = {tm / base - 1:+.2e}; gradient cosine "
                    f"{cos(gmx, g400):.5f}, norm ratio {np.linalg.norm(gmx) / np.linalg.norm(g400):.4f}")
            if label.startswith("shortest"):
                g800 = grad_log10(lambda z: model.idt(z, t_rise=800.0), x)
                line += (f"; 800 K rise: cosine {cos(g800, g400):.4f}, norm ratio "
                         f"{np.linalg.norm(g800) / np.linalg.norm(g400):.3f}")
            print(line, flush=True)

    if do[3]:
        print("\n3. ignition delay at rtol 1e-12, atol 1e-20 (rows: shortest, median, longest tau)")
        for i in (lo, med, hi):
            tt = tight.idt(Xi[i])
            gc = tight.idt_grad_central(Xi[i])
            print(f"   row {i}: tau {ti[i]:.4g} s, relative difference {abs(tt / ti[i] - 1):.1e}; "
                  f"gradient relative error {rel(Gi[i] * D.F_INI, gc * D.F_INI):.1e}")

    ls, lf = int(np.nanargmin(np.where(okl, sl, np.nan))), int(np.nanargmax(np.where(okl, sl, np.nan)))
    pts = [("nominal", D.NOMINAL, nom[1, 0], nom[1, 1:]), (f"slowest flame, row {ls}", Xl[ls], sl[ls], Gl[ls]),
           (f"fastest flame, row {lf}", Xl[lf], sl[lf], Gl[lf])]
    if do[4]:
        print("\n4. flame-speed gradients: adjoint (stored) against central differences on the same grid")
        for label, x, s, g in pts:
            fm = D.KineticsModel(mech=a.mech)   # a fresh instance per point: a flame restart depends on earlier solves
            Su, gf = fm.lfs_grad_fd(x)
            gf = 100.0 * gf
            print(f"   {label} (Su = {100 * Su:.4f} cm/s, stored {s:.4f}): relative error "
                  f"{rel(g * D.F_INI, gf * D.F_INI):.1e}, cosine {cos(g * D.F_INI, gf * D.F_INI):.5f}, "
                  f"norm ratio adjoint/differences {np.linalg.norm(g * D.F_INI) / np.linalg.norm(gf * D.F_INI):.3f}",
                  flush=True)

    if do[5]:
        print("\n5. flame speed with the paper's refinement criteria (ratio 2, slope 0.01, curve 0.02)")
        for label, x, s, g in pts:
            fm = D.KineticsModel(mech=a.mech)
            try:
                Su_f, f = fm.lfs(x, refine=PAPER_REFINE)
            except D.ct.CanteraError:
                print(f"   {label}: no solution on the paper's grid within Cantera's limit of 1000 grid points", flush=True)
                continue
            line = (f"   {label}: Su {100 * Su_f:.4f} cm/s ({len(f.grid)} points) against the stored {s:.4f} cm/s, "
                    f"relative difference {100 * Su_f / s - 1:+.2e}")
            if label == "nominal":
                _, gfine = fm.lfs_grad(x, refine=PAPER_REFINE)
                line += f"; cosine of the adjoint gradients on the two grids {cos(g * D.F_INI, 100 * gfine * D.F_INI):.5f}"
            print(line, flush=True)

    if do[6]:
        print("\n6. size of the adjoint gradient across the flame design: ratio of the adjoint directional "
              "derivative to central differences on the same grid (steps 1e-2, 3e-3)")
        order = np.argsort(np.where(okl, sl, np.inf))[:okl.sum()]
        ranks = list(range(0, len(order), 10)) + [len(order) - 1]
        rnd = np.random.default_rng(6).standard_normal(D.M_INPUTS)
        rnd /= np.linalg.norm(rnd)
        S, Rg = [], []
        for rk in ranks:
            i = int(order[rk])
            fm = D.KineticsModel(mech=a.mech)
            Su, g, res = directional(fm, Xl[i], [None, rnd])
            (ag, fg), (ar, fr) = res
            Gs = Gl[i] * D.F_INI
            S.append(Su)
            Rg.append(ag / fg[0])
            print(f"   rank {rk:3d}, row {i:3d}: Su {Su:8.4f} cm/s (stored {sl[i]:8.4f}); adjoint/differences along "
                  f"the gradient {ag / fg[0]:.4f} (step 3e-3: {ag / fg[1]:.4f}), along the random direction "
                  f"{ar / fr[0]:.4f}; cosine of this adjoint with the stored one {cos(g, Gs):.6f}", flush=True)
        S, Rg = np.array(S), np.array(Rg)
        short = 1.0 - Rg
        use = short > 0
        b, loga = np.polyfit(np.log(S[use]), np.log(short[use]), 1)
        print(f"   shortfall 1 - ratio: mean {short[S < 15].mean():.3f} (Su < 15 cm/s, {np.sum(S < 15)} rows), "
              f"{short[(S >= 15) & (S < 30)].mean():.3f} (15-30), {short[(S >= 30) & (S < 50)].mean():.3f} (30-50), "
              f"{short[S >= 50].mean():.3f} (> 50); largest {short.max():.3f} at Su {S[np.argmax(short)]:.2f} cm/s")
        print(f"   fit over {use.sum()} rows: 1 - ratio = {np.exp(loga):.3f} Su^{b:.3f} (Su in cm/s)")
        fit = 1.0 - np.exp(loga) * sl[okl] ** b
        sh = 1.0 - fit
        print(f"   applied to the {okl.sum()} design rows: share more than 1, 3, 5 % short: "
              f"{np.mean(sh > 0.01):.2f}, {np.mean(sh > 0.03):.2f}, {np.mean(sh > 0.05):.2f}")
        G0 = Gl[okl] * D.F_INI
        G1 = G0 / fit[:, None]
        l0, W0 = eig(G0)
        l1, W1 = eig(G1)
        d2 = np.linalg.norm(W0[:, :2] @ W0[:, :2].T - W1[:, :2] @ W1[:, :2].T, 2)
        print(f"   subspace of Su with the gradients divided by the fitted ratio: lambda1/lambda2 {l0[0] / l0[1]:.3f} -> "
              f"{l1[0] / l1[1]:.3f}, lambda2/lambda3 {l0[1] / l0[2]:.3f} -> {l1[1] / l1[2]:.3f}, lambda1 x "
              f"{l1[0] / l0[0]:.4f}; |w1.w1'| {abs(W0[:, 0] @ W1[:, 0]):.5f}, |w2.w2'| {abs(W0[:, 1] @ W1[:, 1]):.5f}, "
              f"distance of the 2-d subspaces {d2:.4f}", flush=True)

    if do[7]:
        print("\n7. flame speed and the solver history")
        for i in (24, 26, 187):
            fm = D.KineticsModel(mech=a.mech)
            s1, f = fm.lfs(Xl[i])
            line = (f"   row {i}: stored {sl[i]:.4f} cm/s; fresh instance {100 * s1:.4f} cm/s ({len(f.grid)} points, "
                    f"{100 * s1 / sl[i] - 1:+.2%} vs stored)").replace("%", " %")
            if i == 24:
                s2, f = fm.lfs(Xl[i])
                line += f"; solved again in the same instance {100 * s2:.4f} cm/s ({len(f.grid)} points)"
            print(line, flush=True)
        for label, x in [("nominal", D.NOMINAL), ("row 192", Xl[192])]:
            fm = D.KineticsModel(mech=a.mech)
            Su, g = fm.lfs_grad(x)
            g = 100.0 * g * D.F_INI
            d = g / np.linalg.norm(g)
            h = 2e-2
            v = []
            for s in (h, -h):
                v.append(100.0 * D.KineticsModel(mech=a.mech).lfs(x + s * d * D.F_INI)[0])
            fd = (v[0] - v[1]) / (2 * h)
            print(f"   {label}: Su {100 * Su:.4f} cm/s; derivative along the adjoint direction: adjoint "
                  f"{np.linalg.norm(g):.4f}, refined-grid differences {fd:.4f} cm/s per normalized unit "
                  f"(Su at -h, +h: {v[1]:.4f}, {v[0]:.4f})", flush=True)
    print(f"\nrun time {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
