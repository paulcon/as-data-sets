# RAE2822_SHAP_Design_Exploration: data

Kind: **stored data**, converted from outside files. The values were computed by Palar et al. (2024) with
SU2 (RANS, Spalart–Allmaras) and posted in the authors' repository
[optimuspram/SHAPMODE](https://github.com/optimuspram/SHAPMODE) as `Airfoil_Viscous_data.mat`. They were not
rerun (about 20 min of wall-clock time per run, paper p. 17) and are not altered: `convert_data.py` writes
every double of the file as the shortest decimal that reads back as the same double.

## Files

| file | contents | role |
|---|---|---|
| `RAE2822_SHAP_Design_Exploration.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 275; outputs committed |
| `rae2822_hh_inputs.csv` | 98 × 20: Hicks–Henne bump amplitudes `c1` … `c20` (fraction of chord), each in [−0.003, 0.003]; `c1`–`c10` upper surface, `c11`–`c20` lower surface | array `X` of `Airfoil_Viscous_data.mat` |
| `rae2822_hh_outputs.csv` | 98 × 2: `Cd`, `abs_Cm` (dimensionless); row i matches row i of the inputs | array `ff` of `Airfoil_Viscous_data.mat` |
| `convert_data.py` | checks the `.mat` file's SHA-256, reads it with `scipy.io.loadmat`, writes the two CSVs; `--check` compares | the conversion (numpy, scipy) |
| `check_accuracy.py` | prints the stored precision of the outputs (the one Accuracy number the notebook and the converter do not print) | check (numpy only) |

The CSVs are plain comma-separated numbers after two `#` lines (a description, then the column names with
units), so `np.loadtxt(path, delimiter=',')` reads them. Column `c<i>` is the paper's design variable
$x_i$ (Eq. 19 calls the amplitudes $c_i$; the text and figures call them $x_i$); the notebook uses `x1` … `x20`.
`abs_Cm` is the paper's $|C_m|$.

The authors' repository also holds `Airfoil_Euler_data.mat` (Case 1 of the paper: 315 inviscid Euler runs of
a 9-variable PARSEC airfoil, `X` 315 × 9 and `ff` 315 × 3). It is not used or shipped here: that case
overlaps `aerospace_cfd/Transonic_Airfoil_Shape_Parameterizations`, and its third output column is not
identified in the paper or the scripts.

## Model

**Citation.** P. S. Palar, Y. B. Dwianto, L. R. Zuhal, J. Morlier, K. Shimoyama, S. Obayashi,
"Multi-objective design space exploration using explainable surrogate models", *Structural and
Multidisciplinary Optimization* 67(3):38 (2024), doi:10.1007/s00158-024-03769-z (checked on Crossref). The
version read is the HAL deposit hal-04512428 (submitted 2024-03-20) in the publisher's layout; page numbers
below are its pages. Code: MATLAB, needs UQLab and the MATLAB Statistics and Machine Learning Toolbox; no
licence file.

**Flow and geometry** (Sec. 4.2, pp. 16–17). RAE 2822 airfoil, SU2 RANS with the Spalart–Allmaras model
(Economon et al. 2016), Mach 0.729, fixed lift coefficient $C_l$ = 0.723 (the solver adjusts the angle of
attack). About 20 min of wall-clock time per run (p. 17). Reynolds number, mesh, convergence criteria and
SU2 version are not given. The problem was first studied by Palar et al. (2018), as cited on p. 17. Shape
(Eqs. 19–20, p. 16):

$$\bar y_c = y_{c,\mathrm{base}} + \sum_{i=1}^{k} c_i \sin^{w_i}\!\left(\pi x_c^{v_i}\right), \qquad v_i = \ln(0.5)/\ln(x_{J_i}),$$

with $x_{J_i}$ the location of the maximum of bump $i$ and $w_i = 3$. The maxima are set "from $x_{J1}$ to
$x_{J10}$ in a step of 0.05" (p. 16); the first location is not stated. Ten bumps on the upper surface
($x_1$–$x_{10}$), mirrored on the lower surface ($x_{11}$–$x_{20}$); $x_1$ and $x_{11}$ at the leading edge,
$x_{20}$ the lower-surface bump closest to the trailing edge (pp. 17–18). Positive amplitudes thicken the
airfoil on either surface (p. 17).

**Inputs** (m = 20).

| name | surface | units | range | source |
|---|---|---|---|---|
| $x_1$ … $x_{10}$ (`c1`–`c10`) | upper | fraction of chord (this data set's reading) | [−0.003, 0.003] | Sec. 4.2, p. 17; Fig. 16 |
| $x_{11}$ … $x_{20}$ (`c11`–`c20`) | lower, mirror of the upper bumps | fraction of chord | [−0.003, 0.003] | Sec. 4.2, p. 17; Fig. 16 |

The bounds are also those of the authors' `SHAP_Airfoil_viscous.m` (`lb = -0.003`, `ub = 0.003`), which
normalizes to [−1, 1]^20 before fitting. The notebook uses the same normalization (each amplitude divided by
0.003) and a uniform input space on the box.

**Quantities of interest** (p. 17). $C_d$ and $|C_m|$ about the quarter chord, both to be minimized;
dimensionless. $C_l$ is held fixed and is not an output.

**Gradients.** None. The paper computed activity scores from gradients of PC-Kriging surrogates
(Appendix 1, p. 23); no SU2 adjoint gradients were posted. The notebook uses OLS (one direction) and the
gradients of a ridge-penalized quadratic fitted to the 98 runs, both in normalized coordinates.

**Sampling design.** Not stated in the paper. In the file every normalized coordinate lies on a lattice of
spacing 1/128 of the range (to within about 1e-14, notebook), each input has 98 distinct levels, no row
repeats, and row 1 is the centre of the box (the baseline RAE 2822). The first input runs 0.5, 0.75, 0.25,
0.375, 0.875, 0.625, 0.125, 0.1875 of its range in rows 1-8 (notebook): an unscrambled Sobol'-type sequence
on 128 levels that starts at its second point. It is not scipy's unscrambled Sobol' sequence (the first
points agree, later points do not); MATLAB's `sobolset` (first point skipped) is a plausible source, not
verified. Whether runs failed or were dropped is not known. M = 98 is the number of rows; the paper does not state it.

**Published results.** No eigenvalues. One-dimensional active-subspace summary plots: Fig. 17a ($C_d$, a
non-monotonic trend with a "valley", p. 17) and Fig. 18a ($|C_m|$, monotonic, "extremely close to linear",
p. 17); activity scores, total Sobol' indices and averaged SHAP values with 50-sample bootstrap error bars
(Figs. 17b–d, 18b–d). PC-Kriging (order 2) LOOCV errors 0.0631 ($C_d$) and 9.6 × 10⁻³ ($|C_m|$) (p. 17).
Leading drag inputs $x_3$ and $x_4$, then $x_{20}$ or $x_5$; $x_{20}$ drives the nonlinearity of $C_d$
(pp. 17–18).

## Regenerating

The runs cannot be regenerated from the repository (SU2 RANS, about 20 min per run; mesh and solver
settings not published). The CSVs are regenerated from the authors' file:

```
git clone https://github.com/optimuspram/SHAPMODE.git
python convert_data.py SHAPMODE/SHAP_MODE/test_problems_and_data --out DIR           # writes the two CSVs into DIR
python convert_data.py SHAPMODE/SHAP_MODE/test_problems_and_data --out DIR --check   # DIR outside this directory
```

`--out` is required; `--check` refuses a DIR inside this directory, builds the files there and compares them
with the shipped ones entry by entry with relative tolerance 1e-15 (it prints the largest difference). On
2026-11-09, in the Linux container where the shipped files were written, it reported agreement for both
files in under 1 s. It was not rerun in the repository's `.venv`, which has no copy of the authors' file.

**Source.** github.com/optimuspram/SHAPMODE, commit `159f915d57f50d6a70d1e3147e140ddbdcd107e6`
(2023-12-13, branch `main`, the only branch), fetched 2026-11-09 by `git clone`. File
`SHAP_MODE/test_problems_and_data/Airfoil_Viscous_data.mat`: 4,674 bytes, SHA-256
`984026f1f7f5a7c24fa8df169181f752bc04081bd0691e611e18d2d6dd8ae56a`, MATLAB 5.0 MAT-file created
2023-12-12 on PCWIN64; arrays `X` (98 × 20) and `ff` (98 × 2), float64. The file is under
`SHAP_MODE/test_problems_and_data/`, not at the repository root.

In the repository's `.venv` the notebook runs in about 12 s under nbconvert (11 and 13 s of wall time in two
runs on 2026-11-10, with other notebooks running at the same time), and `check_accuracy.py` in under 1 s.

## Accuracy

- Conversion: lossless (`convert_data.py` checks that every written value reads back as the same double;
  `--check`, above).
- Stored precision (`check_accuracy.py`): the outputs carry at most 6 decimals (resolution 1e-6), against
  sample standard deviations 2.688e-03 ($C_d$) and 1.223e-02 ($|C_m|$).
- No gradients, so no gradient check. Fits printed by the notebook: leave-one-out $R^2$ of the linear fit
  0.670 ($C_d$) and 0.994 ($|C_m|$); of the ridge quadratic (penalty 0.1) 0.858 and 0.994; a quadratic in
  the OLS active variable with the direction refitted in each fold 0.726 and 0.994; for $C_d$, the linear
  fit plus the single term $z_{20}^2$ 0.752. A permutation null (500 shuffles) gives leave-one-out $R^2$ of
  at most −0.043 and −0.040 at its 99th percentile; it tests "no signal" only (at the residual variance of
  the linear fit, the expected noise part of $\lambda_1$ is 7.0 % for $C_d$ and 0.1 % for $|C_m|$).
- Dimension (notebook): a null of a one-dimensional truth (quadratic in the OLS active variable) plus
  Gaussian noise at its residual sd, 200 draws, analysed with the same ridge quadratic, gives
  $\lambda_2/\lambda_3$ a 95th percentile of 1.88 ($C_d$) and 1.83 ($|C_m|$); observed 2.74 ($C_d$, above
  all 200 draws) and 1.17 ($|C_m|$). A bootstrap that refits the quadratic (200 replicates) gives $C_d$
  $\lambda_1/\lambda_2$ in [8.8, 40.3] and $\lambda_2/\lambda_3$ in [1.16, 4.22], with $x_{20}$ leading
  $\mathbf{w}_2$ in 141 of 200. The notebook takes $n = 2$ for $C_d$ and $n = 1$ for $|C_m|$.
- The CFD accuracy (mesh and iterative convergence) is not reported in the paper.

## How these files relate to the paper

The 98 runs are the training data of the paper's PC-Kriging surrogates for Case 2 (the authors' script
`SHAP_Airfoil_viscous.m` loads this file and fits them); the paper's activity scores, Sobol' indices and
SHAP values are computed on those surrogates at 1,000–10,000 surrogate samples, not on the runs. The
notebook works on the runs directly. With M = 98 and m = 20 a full quadratic has 231 coefficients, so QPHD
is not possible; OLS gives one direction per output and a ridge-penalized quadratic gives the eigenvalue
ratios, which depend on the penalty (for $C_d$ $\lambda_1/\lambda_2$ 11.2–23.4 over penalties $10^{-3}$–10).
Agreement: the $|C_m|$ direction and trend (Fig. 18a, the largest activity score at $x_{20}$, about 1.0 × 10⁻⁴
read from Fig. 18b, notebook 1.02e-04); $x_3$, $x_4$ as the main drag inputs; and $x_{20}$ as the source of
the $C_d$ nonlinearity ("primarily due to $x_{20}$", p. 18): in the runs $z_{20}^2$ is the largest of the 20
squared terms ($|t|$ 3.2), adding it to the linear fit raises the leave-one-out $R^2$ from 0.670 to 0.752,
and $x_{20}$ leads the second $C_d$ eigenvector, which passes a one-dimensional null. Activity-score ranks
depend on the surrogate: the paper's Fig. 17b (p. 18) reads about 0.72, 0.59, 0.42, 0.30 × 10⁻⁵ for $x_4$,
$x_3$, $x_{20}$, $x_5$; the ridge quadratic ranks $x_{20}$ 6th (1.03e-06), because its single penalty on 210
second-order terms spreads the $x_{20}$ curvature over cross terms, while the linear fit plus $z_{20}^2$
ranks it first (7.19e-06, then $x_4$ 5.55e-06, $x_3$ 5.12e-06). Difference: the notebook's $C_d$ summary plot
along $\mathbf{w}_1$ is monotonic with wide scatter; the paper's Fig. 17a plots the same runs against its
PC-Kriging $\mathbf{w}_1$ and describes a "valley of minimum" (p. 17), which is weak in the figure. The paper's Appendix 1 formula for the
activity scores sums over all $m$ eigenpairs (p. 23), which makes them the diagonal of $\mathbf{C}$; the
notebook computes the same quantity.

## License and credit

Data: © Palar et al., from github.com/optimuspram/SHAPMODE, which states no licence; hosted here with the
source credited and listed in `LICENSE-DATA` as an exception. Cite the paper when using these files. No
code of the authors' repository is included. `convert_data.py`, `check_accuracy.py` and the notebook: MIT
(this repository).

## Unknowns

- The sampling design (the lattice structure suggests a Sobol'-type sequence) and whether failed runs were
  dropped.
- The first bump location $x_{J1}$ (only the 0.05 step is given), and so the exact bump positions.
- Reynolds number, mesh, turbulence-model settings, convergence criteria and SU2 version.
- Whether $|C_m|$ is taken of the moment at the converged fixed-$C_l$ angle of attack (assumed).
- Which $\mathbf{w}_1$ Fig. 17a uses (PC-Kriging gradients at 10,000 points, presumably) and whether its
  valley persists once the $x_{20}$ curvature is accounted for.

## Open questions

- For the authors: the Reynolds number and the SU2 configuration of the viscous case (mesh, SU2 version), and
  the location of the first bump maximum $x_{J1}$.
- Was the 98-run design MATLAB `sobolset` with the first point skipped? Were any runs dropped (98 is not a
  power of two)?
- Fig. 17a: which surrogate $\mathbf{w}_1$ is plotted, and does the valley persist if the $x_{20}$ curvature
  is removed?
