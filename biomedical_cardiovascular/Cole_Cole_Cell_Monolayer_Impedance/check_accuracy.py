"""Recompute the numbers of the Accuracy section of DATA.md (Cole_Cole_Cell_Monolayer_Impedance).

    python check_accuracy.py

Run it from any directory; it reads the shipped CSVs next to this file. It does not use
cole_model.py: the model is written out again below with numpy's complex power, so that the
check is independent of the polar form the module uses. Each line prints the recomputed value;
DATA.md quotes these values (as orders of magnitude where they are at the rounding level).

Checks (all 1100 rows, both parts Re Z and Im Z, all 31 frequencies):
 1. Model values: R_inf + (R_0 - R_inf) / (1 + (1j*omega*tau)**alpha) against outputs.csv, as the
    largest of |model - file| / |Z| over rows, frequencies and the two parts.
 2. Gradients: central differences of that implementation in the log inputs z = ln x (relative
    step h in each x_j), against the stored gradients converted to x * grad f; the relative error
    of the 4-vector per row, part and frequency, for h = 1e-4, 1e-5 and 1e-6 (maximum, and the
    median for h = 1e-6).

Deterministic; takes about a second. Imports numpy and the standard library.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return np.loadtxt(os.path.join(HERE, name), delimiter=',', comments='#', ndmin=2)


def omega():
    """The 31 angular frequencies (rad/s), as generate_data.OMEGA (the header of outputs.csv rounds them)."""
    return np.sort(np.r_[10.0 ** (np.arange(29) / 4), 2e5, 2e6])


def impedance(X, omega):
    """Cole term, written independently of cole_model.py: shape (N, K) complex."""
    r_inf, r_0, alpha, tau = (X[:, j:j + 1] for j in range(4))
    return r_inf + (r_0 - r_inf) / (1 + (1j * omega[None, :] * tau) ** alpha)


def main():
    X = load('inputs.csv')
    F = load('outputs.csv')
    G = load('gradients.csv')
    w = omega()
    N, K = len(X), len(w)
    F = F.reshape(N, 2, K)
    G = G.reshape(N, 2, K, 4)

    Z = impedance(X, w)
    err = max(np.max(np.abs(Z.real - F[:, 0]) / np.abs(Z)), np.max(np.abs(Z.imag - F[:, 1]) / np.abs(Z)))
    print(f'1. independent model vs outputs.csv ({N} rows, {K} frequencies, Re and Im): '
          f'max |model - file| / |Z| = {err:.1e}')

    Gl = G * X[:, None, None, :]                          # x * df/dx = df/d(ln x)
    zl = np.log(X)
    for h in (1e-4, 1e-5, 1e-6):
        D = np.stack([(impedance(np.exp(zl + h * e), w) - impedance(np.exp(zl - h * e), w)) / (2 * h)
                      for e in np.eye(4)], axis=-1)       # (N, K, 4) complex
        D = np.stack([D.real, D.imag], axis=1)
        rel = np.linalg.norm(D - Gl, axis=-1) / np.linalg.norm(Gl, axis=-1)
        extra = f', median {np.median(rel):.1e}' if h == 1e-6 else ''
        print(f'2. central differences in ln x, h = {h:.0e}: max relative error of x * grad f '
              f'{rel.max():.1e}{extra}')


if __name__ == '__main__':
    main()
