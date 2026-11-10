# NACA4412_Shape_Lift_OpenFOAM: data

Kind: **stored data** (converted from outside files). The 170 unsteady RANS runs were made by Tezzele et al.
(2020) with OpenFOAM v6 and ITHACA-FV, and their shape weights and lift-coefficient histories are in the
authors' GitHub repository. They were not regenerated (about 45,000 CPU s per run, paper p. 14) and are not
altered: `convert_data.py` changes only the layout, and `--check` confirms that every value is the stored one.

## Files

| file | contents | role |
|---|---|---|
| `NACA4412_Shape_Lift_OpenFOAM.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 4412; outputs committed |
| `shape_inputs.csv` | 170 × 11: the weights c1..c5, d1..d5 (dimensionless) and `paper_training_run` (1 for the 70 runs the paper trained on, rows 0–69) | from `parameters/parTop.npy` (c) and `parameters/parBot.npy` (= −d) |
| `lift_series.npz` | the 170 × 30,000 lift coefficients, one per time step of 1e-3 s, as `mantissa_diff` (int32) and `scale` (int8) | from `outputs/lift_170par_30000t.npy` |
| `convert_data.py` | reads the four `.npy` files (SHA-256 checked, `allow_pickle=False`), writes the two files into `--out DIR`; `--check` compares | the conversion (numpy only) |
| `check_accuracy.py` | recomputes the numbers of this file marked with its name | check |

`shape_inputs.csv` is comma-separated after two `#` lines (a description; column names with units), so
`np.loadtxt(path, delimiter=',')` reads it. **Sign of d:** the repository stores the lower-surface weights as
negative numbers (`parBot` in [−0.03, 0]), because the solver adds `parBot · r_i` to the lower ordinate; the
CSV stores d = −parBot, the positive weights of the paper's Eq. 5. Negation is exact.

**Lift encoding.** Every stored C_L value has at most 7 significant decimal digits (`check_accuracy.py`:
largest mantissa 9999888 < 10⁷, powers 10⁻¹⁰ to 10⁻⁴). Each value is written as an integer mantissa times
10^−scale, and the mantissas are stored as differences along time, which compresses the 41 MB array to
2.8 MB (`check_accuracy.py`: 2,780,988 bytes). Decoding:

```python
with np.load('lift_series.npz') as z:
    lift = np.cumsum(z['mantissa_diff'], axis=1, dtype=np.int64) / 10.0 ** z['scale']
```

The division of two exactly representable numbers is correctly rounded, so it returns the double nearest to
the 7-digit decimal, which is the stored value; `convert_data.py` refuses to write unless every value
round-trips, and `--check` reports no value differing. Column k (0-based) is time step k + 1, t = (k + 1) ×
1e-3 s, an inference (see Unknowns).

`outputs/lift_170par_20000t.npy` of the repository is the first 20,000 columns of the 30,000-step file
(`convert_data.py` checks this) and is not shipped.

## Model

**Citation.** M. Tezzele, N. Demo, G. Stabile, A. Mola, G. Rozza, "Enhancing CFD predictions in shape design
problems by model and parameter space reduction", *Advanced Modeling and Simulation in Engineering Sciences*
7(1):40 (2020), doi:10.1186/s40323-020-00177-y (publisher PDF read, CC BY 4.0; volume, issue, article number and
authors checked on Crossref; the other references were not). All
authors: Mathematics Area, mathLab, SISSA, Trieste.

**Flow** (pp. 3, 8–9, 14). 2-D unsteady incompressible Navier–Stokes (Eq. 3) with RANS and the
Spalart–Allmaras model, solved by OpenFOAM's PIMPLE algorithm (`pimpleFoam` in the repository's
`runOffline.sh`) around a NACA 4412 at 0° angle of attack; u_in = 1 m/s, ν = 2e−5 m²/s, chord 1 m,
Re = 50,000; reference mesh 46,500 hexahedral cells (blockMesh); Δt = 1e−3 s, second-order backward time
differences, second-order upwind convection, 30 s of flow from an impulsive start. One simulated second cost
about 1508 CPU s serial on an Intel Xeon E5-2640 (p. 14). The case files give span 0.2 m and Aref = 0.2 m²
(`system/controlDict`, `blockMeshDict`), so C_L is the usual 2-D coefficient. Validation in the paper: the
undeformed NACA 4412 gives C_L = 0.355 against 0.1804–0.3708 in a database depending on N_crit (p. 9); the
footnote says the comparison does not fully verify the full-order model.

**Inputs** (m = 10; Eqs. 4–5, p. 4; box p. 9).

| name | surface | change | range | units | source |
|---|---|---|---|---|---|
| c1..c5 | upper | y_u = ȳ_u + Σ c_i r_i | [0, 0.03] | 1 | Eq. 4, p. 4; D = [0, 0.03]^10, p. 9 |
| d1..d5 | lower | y_l = ȳ_l − Σ d_i r_i | [0, 0.03] | 1 | Eq. 5, p. 4; p. 9 |

The shape functions (paper: from Hicks and Henne, ref. 24, plotted in Fig. 1 without formulas) as coded in the
authors' `FOM/MORTECH.C`, with ξ = x/chord from the leading edge and the chord as length scale:
r1 = √ξ (1−ξ) e^(−15ξ), r2 = sin³(π ξ^0.25), r3 = sin³(π ξ^0.757), r4 = sin³(π ξ^1.357),
r5 = √ξ (1−ξ) e^(−10ξ). r2–r4 reach 1; r1 and r5 peak at 0.1072 and 0.1292 (notebook), so c1, c5, d1, d5 move
the surface by at most 0.00322 and 0.00388 chords. The mesh follows by RBF morphing (Wendland second order,
r_RBF = 0.1 m, r_out = 7 m; p. 9). The solver reads `parameters/parTop.npy` and `parBot.npy` directly
(`MORTECH.C`, `main`), so the stored weights are the ones that were run.

**Quantity of interest.** Stored: C_L(t) at every step. The paper analyses C_L at fixed times (6, 10, 14, 18 s
for the active subspaces; 20 s and DMD forecasts to 30 s for the regressions, pp. 11–16). The notebook's
scalar is **the mean C_L over 25 < t ≤ 30 s (5000 steps), this data set's choice**: the first 25 s are
dropped as start-up. The notebook shows the effect of the window start (from 20 s on the 2-D subspace moves by
at most 0.012; whole-run averaging gives a different first direction, |w1·w1| = 0.614).

**Gradients.** None. The subspace is estimated from values with the global quadratic estimator (QPHD;
M = 170, 66 quadratic coefficients), checked against local linear gradients and the linear fit. Coordinates:
inputs mapped from [0, 0.03] to [−1, 1].

**Sampling design.** Not stated in the paper (70 training and 100 test runs, p. 14). In the data the 170
points are consistent with independent uniform draws on D (Kolmogorov–Smirnov p 0.062–0.969; 42–48 of 70 and
62–68 of 100 strata occupied, against 44.4 and 63.4 expected for independent draws and all for a Latin
hypercube; notebook), so the notebook treats them as one Monte Carlo sample. One column is flatter than
usual: c1 has normalized kurtosis 1.59, below the 2.5 % quantile 1.65 for 170 independent uniform draws
(the other nine 1.74–1.92; notebook). No run is outside D, no
input row or history is duplicated.

**Published results.** Dynamic active subspace of C_L at 6, 10, 14, 18 s: first eigenvector and 1-D summary
plot (Figs. 6–9, pp. 12–14), computed with ATHENA; `dyas.py` in the repository uses its local-linear method
on the 70 training runs, 1000 bootstrap replicates. No eigenvalues are printed. c1, c5, d1, d5 "almost zero"
(p. 13); GPR on the 6 remaining weights halves the error (Figs. 10–11; 0.042 vs 0.087 at regime, p. 16).

## Regenerating

The runs cannot be regenerated from the repository (about 45,000 CPU s each; an install-and-run recipe for
the authors' case is given below as an external run). The two files are regenerated from the authors'
repository:

```
git clone https://github.com/mtezzele/MORtech-2019-Special-Issue.git
git -C MORtech-2019-Special-Issue checkout ec05216d2c8e10c437503eaf2ff71a8f43d50e79
python convert_data.py MORtech-2019-Special-Issue --out <DIR outside this folder> --check
```

`--check` writes into DIR and compares with the shipped files; it passes when no value differs by more than
1e-12 and prints the largest difference (0 for both files, no value differing, in the Linux container
where the shipped files were written, on 2026-11-09; under 1 s; not rerun in the repository's `.venv`, which
has no copy of the authors' files). Source files (commit ec05216, 2020-05-04):

| file | bytes | SHA-256 |
|---|---|---|
| `parameters/parTop.npy` | 6,928 | `177563dbba96736efd535fd9693ef1d5dac616bbb6eb75bd1d5e9872b0cac0cf` |
| `parameters/parBot.npy` | 6,928 | `3e7c0e90f4ac52b51ac06554ecc7a8aae4fc3db028301cc786854187c938721c` |
| `outputs/lift_170par_30000t.npy` | 40,800,128 | `24ca1ddfc49016455d5ff4edd262931bc227ea6b83d91ca8c2e3604798b96c3b` |
| `outputs/lift_170par_20000t.npy` | 27,200,128 | `0c52794f03ee7881e11fe14ac3064bb51fe4144994b2b2b9910142fcaa9f6e9a` |

In the repository's `.venv` the notebook runs in about 10 s (it prints 7 s in the committed run; 9 and 15 s
of wall time under nbconvert in two runs on 2026-11-10, with other notebooks running at the same time), and
`check_accuracy.py` in under 1 s.

**Rerunning the CFD (external run, not done here).** Install OpenFOAM v6 (openfoam.org; the README says v5,
v1812 and v1906 also work) and ITHACA-FV (github.com/mathLab/ITHACA-FV), source both `etc/bashrc` files, run
`wmake` in the repository's `FOM/`, then `./runOffline.sh`: it runs `MORTECH` (writes the 170 morphed cases to
`defGeom/<i>`) and `pimpleFoam` in each. The lift is written by the `forceCoeffs1` function object of
`system/controlDict`. Expect about 12.6 CPU-hours per case serially (p. 14). Agreement with the stored values
across OpenFOAM builds is not known.

## Accuracy

- Conversion: lossless (`convert_data.py --check`, above).
- Stored precision: at most 7 significant digits per C_L value (`check_accuracy.py`); the within-window
  standard deviation of C_L is a median 2.2e-04 (notebook), far above that precision. Shape weights are full
  doubles (12–17 significant digits in shortest round-trip form; `check_accuracy.py`).
- Start-up: 2395 values with |C_L| > 1, all within the first 34 steps (`check_accuracy.py`); the smallest
  |C_L| stored, 0.000334, is also in those spikes. The QoI window excludes them.
- Settling inside the window (notebook): the two halves of 25–30 s differ by at most 0.0071 (run 61), the
  trend is at most 2.86e-03 per s in magnitude, and the QoI differs from C_L(30 s) by at most 0.0061 against a
  QoI sd of 0.0592.
- Solver accuracy: no mesh or time-step convergence study is reported; the only check is the reference C_L
  (p. 9, above). No gradients, so no gradient check.

## How these files relate to the paper

They are the paper's 170 runs (70 training, 100 test; p. 14), the files its scripts read. The Fig. 4 caption
sample is row 1 (a training run), with the d weights listed first and the values truncated to four decimals
(notebook). The notebook (i) reproduces Figs. 6–9 with local linear gradients on the 70 training runs
(|w·w_fig| 0.996–1.000) and with QPHD on all 170 (0.948–1.000; at 18 s QPHD gives c4 = −0.78 against −0.55),
and (ii) analyses the late-time mean, for which the paper has no active subspace (it notes that a
higher-dimensional subspace is needed near steady state, p. 13): λ1/λ2 = 4.12, λ2/λ3 = 11.14, n = 2.
**Sign of d in the figures:** `dyas.py` normalizes `parBot` (= −d), so the figures' d axis is −d. The paper's
reading of the positive c3, c4, d3, d4 components at 10 s as a thickness increase (p. 12) holds for −d; in
Eq. 5's d, C_L(10 s) correlates positively with c3, c4 and negatively with d3, d4 (notebook), i.e. with
camber. The GPR and DMD results (Figs. 10–11) are not reproduced.

## License and credit

Data: © Tezzele, Demo, Stabile, Mola, Rozza; from github.com/mtezzele/MORtech-2019-Special-Issue (commit
ec05216). **No licence is stated** for the repository; the files are hosted with the source credited and are
listed in the exceptions of `LICENSE-DATA` (decision of 2026-10-30). Article CC BY 4.0. Cite the paper when
using these files. `convert_data.py`, `check_accuracy.py` and the notebook: MIT (this repository). No code of
the authors' repository is included (the shape-function formulas are quoted from `MORTECH.C` as facts of the
model).

## Unknowns and open questions

- Time stamp of column k: assumed t = (k + 1) × 1e-3 s (OpenFOAM's forceCoeffs writes from the first step;
  30,000 columns for 30 s). The authors' `dyas.py` labels column k as k/1000 s. The 1 ms difference does not
  affect the QoI.
- Sampler: not stated; the data look like independent uniform draws (seed unknown).
- Solver version for the runs: OpenFOAM v6 per the README; ITHACA-FV commit not stated.
- Whether all 170 runs converged without restarts is not stated; the histories show no discontinuities
  (largest second difference after 1 s 1.46e-05 per step; notebook).
- The averaging window (25–30 s) is this data set's choice; the paper uses instantaneous values.
- Question for the authors (not contacted): were the Figs. 6–9 d components intended in the coordinates of
  Eq. 5? The plotted axis is the stored −d.
