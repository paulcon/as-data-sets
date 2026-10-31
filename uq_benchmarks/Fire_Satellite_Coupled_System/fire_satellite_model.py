"""Three-discipline fire detection satellite (Sankararaman and Mahadevan, J. Mech. Des. 134(3), 2012, Sec. 6).

Orbit, attitude control and power subsystems of a forest-fire-monitoring
satellite (Wertz and Larson, Space Mission Analysis and Design), with feedback
coupling between attitude control and power: attitude control needs the
maximum and minimum moments of inertia (I_max, I_min) from the power
subsystem, and the power subsystem needs the attitude-control power P_ACS.
The coupled system is solved by fixed-point iteration on P_ACS.

Equations (11)-(33) and Tables 1-2 of the paper, reimplemented from the
paper (no code was copied). Choices where the printed paper is ambiguous:

- theta (deg) and the sun incidence angle i (deg) are converted to radians.
- omega_max = 6000 rpm is converted to rad/s (6000 * 2 pi / 60). This gives
  P_ACS of about 67 W at the mean inputs, the peak of the paper's Fig. 11;
  using 6000 as rad/s (the OpenTURNS use case does this) gives about 130 W.
- eps_deg = 0.0375 ("% per year" in Table 1) is used as the fraction 0.0375
  per year in P_EOL = P_BOL (1 - eps_deg)^LT.
- Eq. (27) prints L = sqrt(A_sa r_lw / m_sa) and W = sqrt(A_sa / (r_lw m_sa)),
  with m_sa the array mass, which is itself defined from L and W. We read m_sa
  there as n_sa, the number of solar arrays (Table 1), as in the OpenTURNS use
  case.

`qoi(X)` returns the three outputs of Sec. 6.3 (total torque tau_tot in N m,
total power P_tot in W, solar array area A_sa in m^2) for each row of physical
inputs X (columns in the order of NAMES). `grad(X)` returns their gradients
with respect to the physical inputs by the complex-step method carried
through the fixed-point iteration. `solve(X)` returns all intermediate
quantities.

Only numpy and scipy are imported.
"""
import numpy as np
from scipy import stats

#: Input names, in column order (Table 2 of the paper).
NAMES = ('H', 'P_other', 'F_s', 'theta', 'L_sp', 'q', 'R_D', 'L_a', 'C_d')
#: Units of the inputs.
UNITS = ('m', 'W', 'W/m^2', 'deg', 'm', '-', 'A m^2', 'm', '-')
#: Means (Table 2).
MEAN = np.array([18e6, 1000., 1400., 15., 2., .5, 5., 2., 1.])
#: Standard deviations (Table 2). The paper prints 1 for q; the OpenTURNS use case has 0.1.
SD = np.array([1e6, 50., 20., 1., .4, 1., 1., .4, .3])
#: Truncation bounds of the OpenTURNS use case (mean -+ 3 sd for the eight
#: inputs other than q; q in [0.2, 0.8]). The paper's inputs are untruncated normals.
LOWER = np.array([15e6, 850., 1340., 12., .8, .2, 2., .8, .1])
UPPER = np.array([21e6, 1150., 1460., 18., 3.2, .8, 8., 3.2, 1.9])

#: Output names and units.
QOI_NAMES = ('tau_tot', 'P_tot', 'A_sa')
QOI_UNITS = ('N m', 'W', 'm^2')

# Deterministic quantities (Table 1).
RE = 6378140.            # Earth's radius, m
MU = 3.986e14            # gravitational parameter, m^3 s^-2
PHI_TARGET = 235000.     # target diameter, m
C_LIGHT = 2.9979e8       # light speed, m/s
A_S = 13.85              # area reflecting radiation, m^2
I_SUN = 0.               # sun incidence angle, deg
DT_SLEW = 760.           # slewing time period, s
M_EARTH = 7.96e15        # magnetic moment of earth, A m^2
RHO = 5.1480e-11         # atmospheric density, kg/m^3
A_DRAG = 13.85           # cross-sectional area in flight direction, m^2
N_WHEELS = 3             # number of reaction wheels
OMEGA_MAX = 6000*2*np.pi/60   # maximum wheel velocity: 6000 rpm, in rad/s
P_HOLD = 20.             # holding power, W
I_D = 0.77               # inherent degradation of array
ETA = 0.22               # power efficiency
LT = 15.                 # lifetime, years
EPS_DEG = 0.0375         # degradation in power production capability, per year
R_LW = 3.                # length to width ratio of solar array
N_SA = 3.                # number of solar arrays
RHO_SA = 700.            # average mass density of arrays, kg/m^3
T_PANEL = 0.005          # thickness of solar panels, m
D_PANEL = 2.             # distance between panels, m
I_BODY = (6200., 6200., 4700.)   # moments of inertia of the body (X, Y, Z), kg m^2

#: Default relative tolerance of the fixed-point iteration on P_ACS.
TOL = 1e-13
MAXIT = 100


def _max(a, b):
    """max(a, b), selecting by real part (complex-step safe)."""
    return np.where(a.real >= b.real, a, b)


def _min(a, b):
    return np.where(a.real <= b.real, a, b)


def orbit(H, mu=MU):
    """Eqs. (11)-(14): velocity, orbit period, eclipse period, maximum slewing angle (rad)."""
    R = RE + H
    v = np.sqrt(mu/R)
    dt_orbit = 2*np.pi*R/v
    dt_eclipse = dt_orbit/np.pi*np.arcsin(RE/R)
    a = PHI_TARGET/RE
    theta_slew = np.arctan(np.sin(a)/(1 - np.cos(a) + H/RE))
    return v, dt_orbit, dt_eclipse, theta_slew


def attitude(x, I_max, I_min, orb, mu=MU, omega_max=OMEGA_MAX):
    """Eqs. (15)-(22): torques and the attitude-control power P_ACS; `x` maps NAMES to arrays."""
    H, F_s, theta, L_sp, q, R_D, L_a, C_d = (x[k] for k in ('H', 'F_s', 'theta', 'L_sp', 'q',
                                                         'R_D', 'L_a', 'C_d'))
    v, _, _, theta_slew = orb
    R = RE + H
    tau_slew = 4*theta_slew/DT_SLEW**2*I_max
    dI = I_max - I_min
    dI = dI*np.sign(dI.real)                                    # |I_max - I_min|
    tau_g = 3*mu/(2*R**3)*dI*np.sin(2*(np.pi/180*theta))
    tau_sp = L_sp*F_s/C_LIGHT*A_S*(1 + q)*np.cos(np.pi/180*I_SUN)
    tau_m = 2*M_EARTH*R_D/R**3
    tau_a = 0.5*L_a*RHO*C_d*A_DRAG*v**2
    tau_dist = np.sqrt(tau_g**2 + tau_sp**2 + tau_m**2 + tau_a**2)
    tau_tot = _max(tau_slew, tau_dist)
    P_ACS = tau_tot*omega_max + N_WHEELS*P_HOLD
    return dict(tau_slew=tau_slew, tau_g=tau_g, tau_sp=tau_sp, tau_m=tau_m, tau_a=tau_a,
                tau_dist=tau_dist, tau_tot=tau_tot, P_ACS=P_ACS)


def power(x, P_ACS, orb):
    """Eqs. (23)-(33): total power, solar array area, and the moments of inertia."""
    P_other, F_s = x['P_other'], x['F_s']
    _, dt_orbit, dt_eclipse, _ = orb
    P_tot = P_ACS + P_other
    Te = dt_eclipse
    Td = dt_orbit - Te
    P_sa = (P_tot*Te/0.6 + P_tot*Td/0.8)/Td
    P_BOL = ETA*F_s*I_D*np.cos(np.pi/180*I_SUN)
    P_EOL = P_BOL*(1 - EPS_DEG)**LT
    A_sa = P_sa/P_EOL
    L = np.sqrt(A_sa*R_LW/N_SA)
    W = np.sqrt(A_sa/(R_LW*N_SA))
    m_sa = 2*RHO_SA*L*W*T_PANEL
    I_saX = m_sa*((L**2 + T_PANEL**2)/12 + (D_PANEL + L/2)**2)
    I_saY = m_sa/12*(T_PANEL**2 + W**2)
    I_saZ = m_sa*((L**2 + W**2)/12 + (D_PANEL + L/2)**2)
    Ix, Iy, Iz = I_saX + I_BODY[0], I_saY + I_BODY[1], I_saZ + I_BODY[2]
    I_max = _max(_max(Ix, Iy), Iz)
    I_min = _min(_min(Ix, Iy), Iz)
    return dict(P_tot=P_tot, P_sa=P_sa, P_EOL=P_EOL, A_sa=A_sa, L=L, W=W, m_sa=m_sa,
                I_max=I_max, I_min=I_min)


def solve(X, tol=TOL, maxit=MAXIT, mu=MU, omega_max=OMEGA_MAX, full=False):
    """Solve the coupled system by fixed-point iteration on P_ACS.

    Parameters
    ----------
    X : array, shape (M, 9) or (9,), physical inputs (complex allowed, for the complex step).
    tol : stop when |P_ACS^(k+1) - P_ACS^(k)| <= tol |P_ACS^(k+1)| for every row, separately
        for the real part and (complex X) the imaginary part.
    full : also return the intermediate quantities and the iteration count.

    Returns
    -------
    F : array (M, 3): tau_tot (N m), P_tot (W), A_sa (m^2); and, if `full`, a dict.
    """
    X = np.atleast_2d(X)
    x = {name: X[:, j] for j, name in enumerate(NAMES)}
    orb = orbit(x['H'], mu)
    P = np.full(len(X), N_WHEELS*P_HOLD, dtype=X.dtype)      # start: no torque
    for k in range(1, maxit + 1):
        pw = power(x, P, orb)
        at = attitude(x, pw['I_max'], pw['I_min'], orb, mu, omega_max)
        P_new = at['P_ACS']
        d = P_new - P
        P = P_new
        done = np.all(np.abs(d.real) <= tol*np.abs(P.real))
        if np.iscomplexobj(P):
            done = done and np.all(np.abs(d.imag) <= tol*np.abs(P.imag) + 1e-300)
        if done:
            break
    else:
        raise RuntimeError(f'fixed point not converged in {maxit} iterations')
    pw = power(x, P, orb)
    F = np.column_stack([at['tau_tot'], pw['P_tot'], pw['A_sa']])
    if full:
        return F, dict(iterations=k, last_change=d, **at,
                       **{key: val for key, val in pw.items() if key != 'P_tot'})
    return F


def qoi(X, **kw):
    """tau_tot (N m), P_tot (W), A_sa (m^2) at each row of physical inputs X, shape (M, 3)."""
    return solve(np.asarray(X, dtype=float), **kw)


def grad(X, h=1e-30, **kw):
    """Gradients of the three outputs w.r.t. the physical inputs, shape (M, 3, 9).

    Complex step: input j is perturbed by i h SD[j] and the fixed-point
    iteration is run in complex arithmetic until both the real and imaginary
    parts of P_ACS have converged, so the result is the derivative of the
    converged fixed point (to rounding error; no subtractive cancellation).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    G = np.empty((len(X), 3, 9))
    for j in range(9):
        Xc = X.astype(complex)
        Xc[:, j] += 1j*h*SD[j]
        G[:, :, j] = solve(Xc, **kw).imag/(h*SD[j])
    return G


def sample(N, rng, sd=SD):
    """N independent draws from the truncated normals TN(MEAN, sd, LOWER, UPPER), shape (N, 9)."""
    rng = np.random.default_rng(rng)
    a, b = (LOWER - MEAN)/sd, (UPPER - MEAN)/sd
    return np.column_stack([stats.truncnorm.rvs(a[j], b[j], loc=MEAN[j], scale=sd[j], size=N,
                                                random_state=rng) for j in range(9)])
