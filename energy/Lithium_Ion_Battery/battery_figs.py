"""Loading, Algorithm 1 and figures for the lithium-ion battery data.

Constantine & Doostan, Stat. Anal. Data Min. 10(5):243-262, 2017, doi:10.1002/sam.11347.
The data are the Newman-model runs of Hadigol, Maute & Doostan, J. Power Sources
300:507-524, 2015 (see DATA.md). Matplotlib settings are left to the notebook.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import active_subspaces as ac

DATA = Path(__file__).with_name('battery_data.npz')
RATES = {'0.25C': ('025C', 0.25), '1C': ('1C', 1.0), '4C': ('4C', 4.0)}   # npz key, multiple of 1C
I_1C = 30.0             # A m^-2: current density of a 1C discharge. Inferred from the 2016 code, not confirmed
CUTOFF, TOL = 2.8, 1e-3                  # V: every run stops at the cutoff; TOL decides an overshoot
TSTAR = np.linspace(1, 99, 50)[::-1]     # t* of voltage column k is 99 - 2k (2016 code)
# Grid columns analyzed: every voltage of the capacity grid; every t* but 99 and 1 (2016 code)
COLS = {'capacity': np.arange(50), 'voltage': np.arange(1, 49)}
BANDS = {('capacity', '0.25C'): (2, 17, 38), ('capacity', '1C'): (5, 34, 48), ('capacity', '4C'): (5, 34, 48),
         **{('voltage', r): (2, 38, 47) for r in RATES}}    # grid indices of the regions A, B, C
YLABEL = {'capacity': r'Capacity [mAh cm$^{-2}$]', 'voltage': 'Voltage [V]'}
XLABEL = {'capacity': 'Voltage [V]', 'voltage': r'$t^*$'}

# The 19 inputs in data-column order, with the 2016 line styles (style, color, width).
LABELS = [r'$\epsilon_a$', r'$\epsilon_s$', r'$\epsilon_c$', r'brugg$_a$', r'brugg$_s$', r'brugg$_c$',
          r'$t_+^0$', r'$D$', r'$D_a$', r'$D_c$', r'$\sigma_a$', r'$\sigma_c$', r'$k_a$', r'$k_c$',
          r'$r_a$', r'$r_c$', r'$L_a$', r'$L_s$', r'$L_c$']
NAMES = ['eps_a', 'eps_s', 'eps_c', 'brugg_a', 'brugg_s', 'brugg_c', 't+0', 'D', 'D_a', 'D_c', 'sigma_a',
         'sigma_c', 'k_a', 'k_c', 'r_a', 'r_c', 'L_a', 'L_s', 'L_c']     # plain text, for printing
STYLES = [('-', 'b', 3), ('-', 'g', 3), ('-', 'r', 3), ('-', 'b', 1), ('-', 'g', 1), ('-', 'r', 1),
          ('-', 'm', 1), (':', 'm', 1), ('-.', 'b', 1), ('-.', 'r', 1), (':', 'b', 1), (':', 'r', 1),
          ('-.', 'b', 3), ('-.', 'r', 3), ('--', 'b', 1), ('--', 'r', 1), ('--', 'b', 3), ('--', 'g', 3),
          ('--', 'r', 3)]
GROUPS = {'Anode': [0, 3, 16, 8, 10, 12, 14], 'Cathode': [2, 5, 18, 9, 11, 13, 15],
          'Separator': [1, 4, 17], 'Miscellany': [6, 7]}


def load(rate):
    """All 3600 runs at one rate: X (3600, 19) in [-1, 1], voltage [V] (3600, 50) on the t* grid,
    capacity [mAh cm^-2] (3600, 50) on the voltage grid, and that grid vgrid [V] (50,)."""
    key, c = RATES[rate]
    with np.load(DATA) as z:
        seconds = z[f'time_{key}']        # elapsed discharge time [s] (inferred)
        # capacity = current density * time: A m^-2 * s / 3600 = Ah m^-2 = 0.1 mAh cm^-2
        return z['X'], z[f'voltage_{key}'], c * I_1C * seconds / 3600 / 10, z[f'vgrid_{key}']


def overshoot(voltage):
    """True for runs whose last voltage is more than TOL below the cutoff."""
    return voltage[:, -1] < CUTOFF - TOL


def row(qoi, c):
    """Row of a weights array (one row per analyzed column) that holds grid column c."""
    return c - COLS[qoi][0]


def weights(X, F, nboot=0, rng=None):
    """Algorithm 1 (OLS) at each column of F; one SubspaceResult per column.

    v1 returns the signed direction w = u/||u|| of the fitted gradient u, so f increases
    along w. On a uniform space every OLS call warns ValidityWarning (the heuristic has no
    Stein-type guarantee there); the warning is expected and filtered here."""
    space = ac.Uniform.reference(X.shape[1], names=LABELS)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ac.ValidityWarning)
        return [ac.active_subspace(X=X, f=f, space=space, estimator='ols', n=1, nboot=nboot,
                                   bootstrap_vectors=1 if nboot else 0, rng=rng) for f in F.T]


def bootstrap_se(X, F, W, rng, nboot=100):
    """Bootstrap standard error of w (P, 19) at every column of F (M, P).

    Each replicate resamples the M runs with replacement, one resample for all P columns,
    so one least-squares solve gives Algorithm 1 at every column. Each replicate of w is
    signed to agree with W, as v1 aligns its replicates; the SE is their standard deviation."""
    M = X.shape[0]
    A = np.hstack([np.ones((M, 1)), X])
    reps = np.empty((nboot,) + W.shape)
    for b in range(nboot):
        i = rng.integers(0, M, size=M)
        U = np.linalg.lstsq(A[i], F[i], rcond=None)[0][1:].T
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        reps[b] = U * np.where(np.sum(U * W, axis=1, keepdims=True) < 0, -1.0, 1.0)
    return reps.std(axis=0)


def _grid_axis(ax, qoi, grid, centers):
    """Descending grid axis with the shaded regions A, B, C (at 'centers', +-1 grid point)."""
    ticks = grid[[0, 12, 25, 37, 49]]
    ax.set_xticks(ticks, [f'{t:.1f}' for t in ticks])
    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel(XLABEL[qoi])
    ax.grid(True)
    for c, name in zip(centers, 'ABC'):
        ax.axvspan(grid[c - 1], grid[c + 1], color='0.7', zorder=0)
        ax.text(grid[c], 0.93, name, transform=ax.get_xaxis_transform(), ha='center')


def _curves(ax, x, A):
    return [ax.plot(x, A[:, i], ls, color=c, lw=lw)[0] for i, (ls, c, lw) in enumerate(STYLES)]


def plot_weights(qoi, rate, grid, x, W):
    """Components of w against the grid (paper Figs. 2-7, top), legends grouped by component."""
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.7, 1], wspace=0.05)
    ax, lax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    lines = _curves(ax, x, W)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(rf'$\mathbf{{w}}$ components, {qoi.capitalize()}, {rate}')
    _grid_axis(ax, qoi, grid, BANDS[qoi, rate])
    lax.axis('off')
    for (title, idx), (bx, by, loc) in zip(GROUPS.items(), [(0, 1, 'upper left'), (0.52, 1, 'upper left'),
                                                            (0, 0, 'lower left'), (0.52, 0, 'lower left')]):
        lax.add_artist(lax.legend([lines[i] for i in idx], [LABELS[i] for i in idx], title=title, loc=loc,
                                  bbox_to_anchor=(bx, by), borderaxespad=0, handlelength=2.2, fontsize=14))
    return fig


def plot_summaries(qoi, rate, grid, X, F, W_abc):
    """Summary plots of F at A, B, C against w^T x: full range (top) and zoomed (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True)
    lo, hi = (0, 1.05 * F.max()) if qoi == 'capacity' else (0.95 * F.min(), 1.05 * F.max())
    for j, (c, name) in enumerate(zip(BANDS[qoi, rate], 'ABC')):
        for ax in axes[:, j]:
            ax.plot(X @ W_abc[j], F[:, c], 'ko', ms=2)
            ax.grid(True)
            ax.locator_params(axis='y', nbins=4)
        axes[0, j].set_ylim(lo, hi)
        axes[0, j].set_title(f'{name}, V = {grid[c]:.1f} V' if qoi == 'capacity' else
                             rf'{name}, $t^*$ = {grid[c]:.1f}')
        axes[1, j].set_xlabel(r'$\mathbf{w}^T\mathbf{x}$')
    for ax in axes[:, 0]:
        ax.set_ylabel(YLABEL[qoi])
    axes[0, 0].set_xlim(-2.2, 2.2)
    axes[0, 0].set_xticks([-2, 0, 2])
    fig.suptitle(f'{qoi.capitalize()}, I = {rate}')
    fig.tight_layout()
    return fig


def plot_se(ax, qoi, rate, grid, x, SE):
    """Bootstrap standard errors of the components of w against the grid (paper Fig. 9)."""
    _curves(ax, x, SE)
    ax.set_ylim(0, 1.15 * SE.max())
    ax.set_title(f'{qoi.capitalize()}, I = {rate}')
    _grid_axis(ax, qoi, grid, BANDS[qoi, rate])
