"""In-plane elastic buckling load of shallow circular steel I-section arches.

Closed-form model used by T.-H. Nguyen, "Global sensitivity analysis of in-plane
elastic buckling of steel arches", Eng. Technol. Appl. Sci. Res. 10(5):6476-6480,
2020 (doi:10.48084/etasr.3833), Eqs. (1)-(3) and (6), which the paper takes from
Y.-L. Pi, M. A. Bradford and B. Uy, "In-plane stability of arches", Int. J. Solids
Struct. 39(1):105-125, 2002 (doi:10.1016/S0020-7683(01)00209-8).

    lambda = S Phi / (4 r_x),   r_x = sqrt(I / A)                          (1)
    N_cr   = pi^2 E I / (k S)^2,  k = 0.35 fixed, 0.5 pin-ended             (6)
    pin-ended (2):  N_acr = (0.15 + 0.006 lambda^2) N_cr,       3.88 < lambda < 9.38
                    N_acr = (0.26 + 0.74 sqrt(1 - 0.63 pi^4/lambda^2)) N_cr, lambda > 9.38
    fixed (3):      N_acr = (0.36 + 0.0011 lambda^2) N_cr,      9.87 < lambda < 18.6
                    N_acr = (0.6 + 0.4 sqrt(1 - 3.109 pi^4/lambda^2)) N_cr,  lambda > 18.6

The second-branch expressions are read from the rendered equations on p. 6477 of
the paper (the PDF text layer drops the square root). The fixed-arch branches
meet continuously at lambda = 18.6; the pin-ended ones jump by about 1.6 % at
lambda = 9.38 (the constants are rounded in the paper).

Choices made here (see DATA.md):

- Inputs are x = (S, Phi, E, D, B, tw, tf). The paper also samples the span L
  independently, but L does not enter Eqs. (1)-(6), and for a circular arch it is
  fixed by S and Phi, L = 2 (S/Phi) sin(Phi/2); so it is not an input here.
- E is in MPa on [1.8e5, 2.2e5] (the paper prints 18E5-22E5 MPa, ten times
  steel's modulus). Lengths in m. Loads in kN.
- I and A are those of a doubly symmetric I-section without root fillets:
  A = 2 B tf + (D - 2 tf) tw,  I = [B D^3 - (B - tw)(D - 2 tf)^3] / 12.
- Below the lower validity limit of Eq. (3) (lambda <= 9.87, fixed arches) the
  first-branch polynomial is used unchanged; `branch` reports such samples as 0.

Only numpy is imported.
"""
import numpy as np

#: Input names, in column order.
NAMES = ('S', 'Phi', 'E', 'D', 'B', 'tw', 'tf')
UNITS = ('m', 'rad', 'MPa', 'm', 'm', 'm', 'm')
DESCRIPTIONS = ('arch length (along the axis)', 'included angle', "Young's modulus",
                'section depth', 'flange width', 'web thickness', 'flange thickness')
#: Bounds of the independent uniform inputs, Table II of the paper (E corrected, see above).
LOWER = np.array([6.283, 0.628, 1.8e5, 0.235, 0.136, 0.007, 0.011])
UPPER = np.array([7.679, 0.768, 2.2e5, 0.287, 0.166, 0.008, 0.014])

#: The paper's Table II range for the span L (m), not used (see module docstring).
L_TABLE = (9.00, 10.00)

#: Coefficients of Eqs. (2)-(3): (a, b, p, q, alpha, lambda_low, lambda_switch, k).
#: c(lambda) = a + b lambda^2 below lambda_switch, p + q sqrt(1 - alpha pi^4/lambda^2) above.
CASES = {
    'fixed':  dict(a=0.36, b=0.0011, p=0.6, q=0.4, alpha=3.109, low=9.87, switch=18.6, k=0.35),
    'pinned': dict(a=0.15, b=0.006, p=0.26, q=0.74, alpha=0.63, low=3.88, switch=9.38, k=0.5),
}

KN_PER_MPA_M2 = 1e3  # E in MPa (1e6 N/m^2) times m^4 / m^2 = 1e6 N = 1e3 kN


def section(D, B, tw, tf):
    """Area A (m^2) and major-axis second moment I (m^4) of the I-section, and their
    partial derivatives with respect to (D, B, tw, tf), shapes (..., 4)."""
    h = D - 2*tf                       # clear web depth
    A = 2*B*tf + h*tw
    I = (B*D**3 - (B - tw)*h**3)/12
    dA = np.stack([tw + 0*D, 2*tf + 0*D, h, 2*B - 2*tw], axis=-1)
    dI = np.stack([(3*B*D**2 - 3*(B - tw)*h**2)/12,
                   (D**3 - h**3)/12,
                   h**3/12 + 0*D,
                   (B - tw)*h**2/2], axis=-1)
    return A, I, dA, dI


def slenderness(X):
    """Modified slenderness lambda = S Phi / (4 r_x), Eq. (1), for each row of X."""
    X = np.atleast_2d(X)
    A, I, _, _ = section(X[:, 3], X[:, 4], X[:, 5], X[:, 6])
    return X[:, 0]*X[:, 1]/4*np.sqrt(A/I)


def branch(X, case):
    """1 or 2 for the branch of Eq. (2)/(3) used; 0 below its lower validity limit."""
    c = CASES[case]
    lam = slenderness(X)
    return np.where(lam <= c['low'], 0, np.where(lam < c['switch'], 1, 2))


def coefficient(lam, case):
    """The factor c(lambda) = N_acr/N_cr of Eq. (2) or (3), and dc/dlambda."""
    c = CASES[case]
    lam = np.asarray(lam, dtype=float)
    upper = lam >= c['switch']
    s = np.sqrt(np.where(upper, 1 - c['alpha']*np.pi**4/lam**2, 1.0))
    coef = np.where(upper, c['p'] + c['q']*s, c['a'] + c['b']*lam**2)
    dcoef = np.where(upper, c['q']*c['alpha']*np.pi**4/(lam**3*s), 2*c['b']*lam)
    return coef, dcoef


def buckling_load(X, case='fixed', grad=False):
    """In-plane elastic buckling load N_acr in kN, Eq. (2) ('pinned') or (3) ('fixed').

    X has rows (S, Phi, E, D, B, tw, tf) in physical units. With grad=True, also
    returns dN_acr/dx, shape (M, 7), in kN per unit of each input (exact).
    """
    c = CASES[case]
    X = np.atleast_2d(np.asarray(X, dtype=float))
    S, Phi, E, D, B, tw, tf = X.T
    A, I, dA, dI = section(D, B, tw, tf)
    lam = S*Phi/4*np.sqrt(A/I)
    Ncr = KN_PER_MPA_M2*np.pi**2*E*I/(c['k']*S)**2

    coef, dcoef = coefficient(lam, case)
    N = coef*Ncr
    if not grad:
        return N

    # d lambda / dx and d N_cr / dx
    dlam = np.zeros_like(X)
    dNcr = np.zeros_like(X)
    dlam[:, 0], dlam[:, 1] = lam/S, lam/Phi
    dNcr[:, 0], dNcr[:, 2] = -2*Ncr/S, Ncr/E
    dlam[:, 3:] = lam[:, None]/2*(dA/A[:, None] - dI/I[:, None])
    dNcr[:, 3:] = Ncr[:, None]*dI/I[:, None]
    G = dcoef[:, None]*dlam*Ncr[:, None] + coef[:, None]*dNcr
    return N, G


def implied_span(X):
    """Span (chord) L = 2 R sin(Phi/2) of the circular arch with R = S/Phi, in m."""
    X = np.atleast_2d(X)
    return 2*X[:, 0]/X[:, 1]*np.sin(X[:, 1]/2)


def sample(M, rng):
    """M independent uniform samples of the inputs, physical units, shape (M, 7)."""
    rng = np.random.default_rng(rng)
    return LOWER + (UPPER - LOWER)*rng.random((M, len(NAMES)))
