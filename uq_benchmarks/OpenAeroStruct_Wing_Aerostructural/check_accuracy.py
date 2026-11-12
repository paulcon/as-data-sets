"""Recompute the numbers of DATA.md that the notebook does not print (needs OpenAeroStruct 2.12.0).

    python check_accuracy.py                       # checks 0-6 (about a minute)
    python check_accuracy.py --psdr DIR            # also check 7 (DIR holds psdr's tests/oas.input, oas.output)
    python check_accuracy.py --slow                # also checks 8 and 9 (about 9 minutes more)

Prints, beside the values DATA.md quotes:

0. The stored inputs against the design drawn from the seed (numpy only; runs without OpenAeroStruct):
   largest difference relative to the half-range, the share of entries stored exactly, duplicate
   rows, and whether every row is inside the box.
1. Reproduction and gradients at the shipped rows 0, 1, 2, 100, 250 and 499: outputs and gradients
   recomputed with oas_wing_driver.py against the files, and the gradients against independent central
   differences of the whole model (step 1e-5 x half-range in every input), as the relative 2-norm
   difference of each QoI's normalized gradient vector.
2. The E and G columns with relative steps 1e-3 and 1e-4 at the same rows, relative to the largest
   entry of each column.
3. Fourth-order central differences (steps 1e-3 and 1e-4 x half-range) at rows 23, 241, 249 and 56.
4. Solver tolerance: rows 0, 1, 2 re-solved with atol 1e-11 instead of 1e-10.
5. Material constants written in place (as oas_wing_driver.py does) against a fresh build of the problem,
   at rows 0, 1, 2: outputs and gradients.
6. Scale of the inputs at the nominal point: the change of fuel burn over the whole E range and over
   the whole alpha range; and the near-divergence corner of the box (every thickness 5 mm,
   root chord 1.3, taper 1.25, alpha 5 deg, twist +1 deg, E and G at their minimum, density nominal).
7. (--psdr DIR) psdr's 10 stored test points: stock OAS 2.12 and OAS 2.12 with OAS v1's load
   transfer against psdr's outputs (psdr's weights in N: the structural weight is 9.81 x spar mass,
   v1's g, and the total weight 9.81 x spar mass + 9.80665 x (W0 + fuel burn)); and the fuel-burn error with the thickness control points
   reversed or passed without the v1-to-v2 map.
8. (--slow) All 2000 shipped inputs re-run with v1's load transfer (outputs only): the differences
   from the shipped outputs, the number of samples with failure > 0 under each transfer, and the
   same numbers on the rows the notebook keeps (failure <= 0 in the shipped file).
9. (--slow) An independent 400-sample design, numpy default_rng(99), with gradients: the fuel-burn
   eigenvalue ratios with all samples and with the notebook's rule (failure <= 0).

"v1's load transfer" is OAS v1's TransferLoads (transfer.py of J. Hokanson's OpenAeroStruct fork,
commit 8ee82d3), re-typed here as a replacement for the compute method of OAS 2.12's
LoadTransfer component: each panel's moment about the element midpoint at fem_origin, split equally
between the element's two nodes. It is used for function values only (no derivatives).

Deterministic (fixed rows and seeds). Reads the CSVs next to it and writes nothing; runs from any
directory. Needs numpy and OpenAeroStruct 2.12.0 (pip install openaerostruct==2.12.0, which pulls
OpenMDAO); without it the script stops with an error that says what to install.
"""
import argparse
import contextlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oas_wing_driver as ow  # noqa: E402

HALF = (ow.UPPER - ow.LOWER) / 2
OAS_VERSION = "2.12.0"


def load():
    ld = lambda n: np.loadtxt(os.path.join(HERE, n), delimiter=",")  # noqa: E731
    return ld("inputs.csv"), ld("outputs.csv"), ld("gradients.csv").reshape(-1, 5, 12)


def keep_rule(F):
    """The notebook's rule: keep the wings whose spar is within the allowable stress (failure <= 0)."""
    return F[:, 4] <= 0


def rel_norm(a, b):
    """Relative 2-norm difference of the normalized gradient vectors, per QoI (rows)."""
    a, b = a * HALF, b * HALF
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def fd_gradient(model, x, steps, order=2):
    D = np.zeros((5, 12))
    for j in range(12):
        h = steps[j]
        e = np.zeros(12); e[j] = h
        if order == 2:
            D[:, j] = (model.evaluate(x + e) - model.evaluate(x - e)) / (2 * h)
        else:
            D[:, j] = (-model.evaluate(x + 2 * e) + 8 * model.evaluate(x + e)
                       - 8 * model.evaluate(x - e) + model.evaluate(x - 2 * e)) / (12 * h)
    return D


def _v1_compute(self, inputs, outputs):
    """OAS v1 TransferLoads: moment arm from each panel's aerodynamic centre to the element midpoint
    at fem_origin; forces and moments split equally between the element's two nodes."""
    mesh = inputs["def_mesh"]
    sf = inputs["sec_forces"]
    w = 0.25
    a = (0.5 * (1 - w) * mesh[:-1, :-1] + 0.5 * w * mesh[1:, :-1]
         + 0.5 * (1 - w) * mesh[:-1, 1:] + 0.5 * w * mesh[1:, 1:])
    w = self.fem_origin
    s = (0.5 * (1 - w) * mesh[0, :-1] + 0.5 * w * mesh[-1, :-1]
         + 0.5 * (1 - w) * mesh[0, 1:] + 0.5 * w * mesh[-1, 1:])
    moment = np.sum(np.cross(a - s[None], sf), axis=0)
    force = np.sum(sf, axis=0)
    loads = np.zeros((self.ny, 6))
    loads[:-1, :3] += 0.5 * force
    loads[1:, :3] += 0.5 * force
    loads[:-1, 3:] += 0.5 * moment
    loads[1:, 3:] += 0.5 * moment
    outputs["loads"] = loads


@contextlib.contextmanager
def v1_transfer():
    from openaerostruct.transfer.load_transfer import LoadTransfer
    stock = LoadTransfer.compute
    LoadTransfer.compute = _v1_compute
    try:
        yield
    finally:
        LoadTransfer.compute = stock


def eig_fuelburn(G):
    Gn = G[:, 0] * HALF
    lam = np.linalg.eigvalsh(Gn.T @ Gn / len(Gn))[::-1]
    return lam


def check_inputs(X):
    print("0. Stored inputs against the design drawn from the seed (default_rng(2018))")
    D = ow.LOWER + (ow.UPPER - ow.LOWER) * np.random.default_rng(2018).random((len(X), 12))
    d = np.abs(X - D) / HALF
    print(f"   largest difference {d.max():.1e} of the half-range; entries stored exactly: {np.mean(X == D):.0%}; "
          f"duplicate rows: {len(X) - len(np.unique(X, axis=0))}; all inside the box: "
          f"{bool(np.all((X >= ow.LOWER) & (X <= ow.UPPER)))}")


def check_rows(X, F, G):
    print("1. Reproduction and central differences (step 1e-5 x half-range) at rows 0, 1, 2, 100, 250, 499")
    model = ow.OASWing()
    worst_y, worst_g, worst_fd = 0.0, 0.0, np.zeros(5)
    for r in (0, 1, 2, 100, 250, 499):
        y, D = model.evaluate_with_gradient(X[r])
        worst_y = max(worst_y, np.max(np.abs(y - F[r]) / np.abs(F[r])))
        worst_g = max(worst_g, np.max(rel_norm(D, G[r])))
        fd = fd_gradient(model, X[r], 1e-5 * HALF)
        e = rel_norm(D, fd)
        worst_fd = np.maximum(worst_fd, e)
        print(f"   row {r:4d} (thickness_3 = {1e3 * X[r, 5]:4.1f} mm): FD error per QoI "
              + ", ".join(f"{q} {v:.1e}" for q, v in zip(ow.QOIS, e)))
    print(f"   recomputed vs shipped: outputs max relative difference {worst_y:.1e}; "
          f"gradients max relative 2-norm difference {worst_g:.1e}")
    print("   largest FD error per QoI: " + ", ".join(f"{q} {v:.1e}" for q, v in zip(ow.QOIS, worst_fd)))

    print("2. E and G columns, relative step 1e-3 against 1e-4 (rows as in 1)")
    worst = 0.0
    for r in (0, 1, 2, 100, 250, 499):
        _, D3 = model.evaluate_with_gradient(X[r], rel_step=1e-3)
        _, D4 = model.evaluate_with_gradient(X[r], rel_step=1e-4)
        for k in (9, 10):
            worst = max(worst, np.max(np.abs(D3[:, k] - D4[:, k])) / np.max(np.abs(D4[:, k])))
    print(f"   largest difference relative to the largest entry of the column: {worst:.1e}")

    print("3. Fourth-order central differences at rows 23, 241, 249, 56")
    for r in (23, 241, 249, 56):
        e = [np.max(rel_norm(G[r], fd_gradient(model, X[r], h * HALF, order=4))) for h in (1e-3, 1e-4)]
        print(f"   row {r:4d}: largest relative difference over the QoIs {e[0]:.1e} (step 1e-3 x half-range), "
              f"{e[1]:.1e} (step 1e-4 x half-range)")

    print("4. Solver tolerance: atol 1e-11 against the shipped 1e-10 (rows 0, 1, 2)")
    tight = ow.OASWing(tol=1e-11)
    d = max(np.max(np.abs(tight.evaluate(X[r]) - F[r]) / np.abs(F[r])) for r in (0, 1, 2))
    print(f"   largest relative change of the outputs {d:.1e}")

    print("5. Material constants written in place against a fresh build (rows 0, 1, 2)")
    dy = dg = 0.0
    for r in (0, 1, 2):
        y1, D1 = model.evaluate_with_gradient(X[r])
        y2, D2 = ow.OASWing().evaluate_with_gradient(X[r])
        dy = max(dy, np.max(np.abs(y1 - y2) / np.abs(y2)))
        dg = max(dg, np.max(rel_norm(D1, D2)))
    print(f"   outputs {dy:.1e}, gradients {dg:.1e} (largest relative differences)")
    return model


def check_scale(model):
    print("6. Scale of the inputs at the nominal point, and the near-divergence corner")
    x = ow.NOMINAL.copy()
    for k, name in ((9, "E"), (8, "alpha")):
        lo, up = x.copy(), x.copy()
        lo[k], up[k] = ow.LOWER[k], ow.UPPER[k]
        print(f"   fuel burn change over the whole {name} range: {model.evaluate(up)[0] - model.evaluate(lo)[0]:+.2f} kg")
    xc = np.array([1, 1, 1, 0.005, 0.005, 0.005, 1.3, 1.25, 5.0, ow.LOWER[9], ow.LOWER[10], ow.NOMINAL[11]])
    y = model.evaluate(xc)
    print(f"   corner: fuel burn {y[0]:.1f} kg, lift {y[2] / 1e3:.1f} kN, failure {y[4]:.2f}, "
          f"lift / weight {y[2] / (ow.G0 * y[3]):.2f}")


def check_psdr(psdr_dir):
    print("7. psdr's 10 stored test points (tests/oas.input, tests/oas.output)")
    P = np.loadtxt(os.path.join(psdr_dir, "oas.input"))
    Q = np.loadtxt(os.path.join(psdr_dir, "oas.output"))
    X = ow.LOWER + (P + 1) / 2 * (ow.UPPER - ow.LOWER)

    def compare(Y, label):
        r = lambda a, b: np.max(np.abs(a - b) / np.abs(b))  # noqa: E731
        df = Y[:, 4] - Q[:, 4]
        print(f"   {label}: fuel burn {r(Y[:, 0], Q[:, 0]):.1e}, structural weight (9.81 x mass) "
              f"{r(9.81 * Y[:, 1], Q[:, 1]):.1e}, lift {r(Y[:, 2], Q[:, 2]):.1e}, total weight "
              f"{r(9.81 * Y[:, 1] + ow.G0 * (Y[:, 3] - Y[:, 1]), Q[:, 3]):.1e} (max relative differences); failure difference "
              f"{df.min():+.3f} to {df.max():+.3f} (largest magnitude {np.abs(df).max():.1e}), same sign at {np.sum(np.sign(Y[:, 4]) == np.sign(Q[:, 4]))} of {len(Q)}")

    model = ow.OASWing()
    compare(np.array([model.evaluate(x) for x in X]), "stock OAS 2.12      ")
    with v1_transfer():
        v1 = ow.OASWing()
        compare(np.array([v1.evaluate(x) for x in X]), "with v1 load transfer")
    T = model._T
    for label, Tv in (("thickness control points reversed", T[:, ::-1]), ("no v1-to-v2 map", np.eye(3))):
        model._T = Tv
        Y = np.array([model.evaluate(x) for x in X])
        print(f"   stock, {label}: fuel burn max relative difference from psdr "
              f"{np.max(np.abs(Y[:, 0] - Q[:, 0]) / Q[:, 0]):.1e}")
    model._T = T


def check_v1_all(X, F):
    print("8. All 2000 shipped inputs with v1's load transfer (outputs only)")
    t0 = time.time()
    with v1_transfer():
        v1 = ow.OASWing()
        Y = np.array([v1.evaluate(x) for x in X])
    print(f"   {time.time() - t0:.0f} s")
    keep = keep_rule(F)
    for label, rows in (("all 2000 rows", np.ones(len(F), bool)), (f"the {keep.sum()} rows kept by the notebook", keep)):
        dfb = np.abs(F[rows, 0] - Y[rows, 0]) / Y[rows, 0]
        dl = np.abs(F[rows, 2] - Y[rows, 2]) / Y[rows, 2]
        dfail = Y[rows, 4] - F[rows, 4]
        print(f"   {label}: fuel burn differs by a median {100 * np.median(dfb):.2f} %, 90th percentile "
              f"{100 * np.percentile(dfb, 90):.2f} %, max {100 * dfb.max():.2f} % (row {np.flatnonzero(rows)[np.argmax(dfb)]}); "
              f"lift by at most {100 * dl.max():.1f} %; v1 failure minus shipped failure {dfail.min():.3f} to "
              f"{dfail.max():.3f} (median {np.median(dfail):.3f}); failure > 0: {np.sum(F[rows, 4] > 0)} shipped, "
              f"{np.sum(Y[rows, 4] > 0)} with v1; sign differs at {np.sum(np.sign(F[rows, 4]) != np.sign(Y[rows, 4]))}")
    print(f"   rows the rule would keep with v1's failure: {np.sum(Y[:, 4] <= 0)}")


def check_draw99():
    print("9. Independent design: 400 samples, numpy default_rng(99), with gradients")
    X = ow.LOWER + (ow.UPPER - ow.LOWER) * np.random.default_rng(99).random((400, 12))
    model = ow.OASWing()
    t0 = time.time()
    F = np.zeros((400, 5))
    G = np.zeros((400, 5, 12))
    for i, x in enumerate(X):
        F[i], G[i] = model.evaluate_with_gradient(x)
    print(f"   {time.time() - t0:.0f} s")
    keep = keep_rule(F)
    for label, rows in (("all 400", np.ones(400, bool)), (f"failure <= 0 ({keep.sum()} of 400)", keep)):
        lam = eig_fuelburn(G[rows])
        print(f"   {label}: lambda_1/lambda_2 = {lam[0] / lam[1]:.2f}, lambda_2/lambda_3 = {lam[1] / lam[2]:.2f}; "
              f"fuel burn up to {F[rows, 0].max():.1f} kg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psdr", default=None, help="directory with psdr's tests/oas.input and tests/oas.output")
    ap.add_argument("--slow", action="store_true", help="also run checks 8 and 9")
    a = ap.parse_args()
    t0 = time.time()
    X, F, G = load()
    check_inputs(X)
    try:
        import openaerostruct
    except ImportError:
        print(f"OpenAeroStruct is not installed. Install it (pip install openaerostruct=={OAS_VERSION}, the "
              "version that wrote the shipped files; it pulls OpenMDAO) and rerun this script.", file=sys.stderr)
        sys.exit(3)
    import openmdao
    print(f"OpenAeroStruct {openaerostruct.__version__}, OpenMDAO {openmdao.__version__}, numpy {np.__version__}, "
          f"Python {sys.version.split()[0]}")
    if openaerostruct.__version__ != OAS_VERSION:
        print(f"warning: the shipped files were written with OpenAeroStruct {OAS_VERSION}")
    model = check_rows(X, F, G)
    check_scale(model)
    if a.psdr:
        check_psdr(a.psdr)
    if a.slow:
        check_v1_all(X, F)
        check_draw99()
    print(f"run time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
