# Speed_Reducer_Shaft_Limit_State: data

Kind: **Python model**. The model is `shaft_model.py` (numpy and scipy), a closed-form limit state written from Eq. (34) and Table 8 of Du and Sudjianto (2004), p. 1205, and cross-checked against the UQTestFuns package. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs; a rerun reproduces them to the tolerance under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `Speed_Reducer_Shaft_Limit_State.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 2004`; outputs committed |
| `shaft_model.py` | `limit_state(X, grad=False)`, `to_normal(X)`, `from_normal(Z)`, `dx_dxi(Z)`, `sample(M, rng)`, `form(start=None, threshold=0.0)`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `MEAN`, `SD`, `MU_F`, `BETA_F`, `S_LO`, `S_HI` | the model |
| `generate_data.py` | draws the samples and writes the three CSVs; `--check` compares a rerun with the shipped files; checks the gradients by finite differences | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below that the notebook and `generate_data.py` do not print | check |
| `inputs.csv` | 1000 × 5 physical inputs D (mm), L (mm), F (N), T (N m), S (MPa) | model input |
| `outputs.csv` | 1000 × 1: g (Pa) | model output |
| `gradients.csv` | 1000 × 5: ∂g/∂(D, L, F, T, S) in Pa/mm, Pa/mm, Pa/N, Pa/(N m), Pa/MPa, with respect to the **physical** inputs | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits.

## Model

**Source.** X. Du (University of Missouri–Rolla, Rolla, Missouri) and A. Sudjianto (Ford Motor Company, Dearborn, Michigan), "First-Order Saddlepoint Approximation for Reliability Analysis", *AIAA Journal* 42(6):1199–1207, 2004, [doi:10.2514/1.3877](https://doi.org/10.2514/1.3877) (authors and affiliations as printed on p. 1199; venue checked on Crossref, which spells the title "First Order"). The problem is the article's Example 2, "Nonlinear Limit-State Function" (Sec. IV.B, p. 1205). The function `RSSpeedReducerShaft` of UQTestFuns 0.7.0 (D. Wicaksono and M. Hecht, MIT licence; *J. Open Source Softw.* 8(90):5671, 2023, doi:10.21105/joss.05671; input model `Du2004` in `uqtestfuns/test_functions/rs_speed_reducer_shaft.yaml`, function `evaluate` in `rs_speed_reducer_shaft.py`) implements the same problem and is used here as an implementation cross-check (`check_accuracy.py`, checks 3–5).

**Limit state** (Eq. 34, p. 1205, "the difference between the strength and the maximum stress"):

g = S − (32 / (π D³)) · √(F² L² / 16 + T²),

evaluated as g = S · 10⁶ − 32 / (π (D/10³)³) · √(F² (L/10³)² / 16 + T²)   [Pa],

the equivalent stress of a solid circular shaft under the bending moment FL/4 and the torque T. The article states no units for g, but its thresholds 4.5 × 10⁷ (Table 10) and 2.48 × 10⁷ (Table 11) are in Pa with D and L in m; the Monte Carlo values below reproduce all three of its tables with this conversion (UQTestFuns converts the same way). The failure event is g < 0 (Table 9).

**Inputs** (m = 5), from Table 8, p. 1205:

| name | quantity | units | distribution | mean | sd | source |
|---|---|---|---|---|---|---|
| `D` | shaft diameter | mm | Normal(39, 0.1) | 39 | 0.1 | Table 8 (footnote a: parameters are mean and sd) |
| `L` | shaft span | mm | Normal(400, 0.1) | 400 | 0.1 | Table 8 ("Span L") |
| `F` | external force | N | Gumbel (max), location μ = 1342.481377, scale β = 272.893880 | 1500 | 350 | Table 8 (footnote b: parameters are mean and sd) |
| `T` | torque | N m | Normal(250, 35) | 250 | 35 | Table 8 |
| `S` | strength | MPa | Uniform[70, 80] | 75 | 10/√12 = 2.887 | Table 8 (footnote c: parameters are the bounds) |

The Gumbel parameters follow from the mean and sd by β = sd·√6/π and μ = mean − γβ, as in UQTestFuns (`gumbel_max_beta`, `gumbel_max_mu` in `uqtestfuns/core/registry/parser/builtin_functions.py`). Table 8 gives marginals only; the text says that "all of the random variables are tractable" (case 1, p. 1205) but not whether they are independent. This data set takes them as **independent** (its reading; UQTestFuns does the same, "Copulas: Independence"). `check_accuracy.py` (checks 3–4) confirms that `shaft_model` reproduces UQTestFuns' g and marginal CDFs at the shipped inputs.

**Gradients.** Exact, closed form, with respect to the physical inputs: with d = D/10³, l = L/10³, k = 32/(πd³) and R = √(F²l²/16 + T²), ∂g/∂D = 3kR/d · 10⁻³, ∂g/∂L = −kF²l/(16R) · 10⁻³, ∂g/∂F = −kFl²/(16R), ∂g/∂T = −kT/R, ∂g/∂S = 10⁶.

**Normalized coordinates (in the notebook).** Each input is scaled consistently in two ways: by its sd, z = (x − mean)/sd, and through its CDF to a standard normal variable, ξ = Φ⁻¹(F_i(x)), with ∂g/∂ξ = (dx/dξ) ∂g/∂x and dx/dξ = φ(ξ)/ρ_i(x) (`shaft_model.dx_dxi`). The two agree for the normal inputs D, L, T.

**Sampling design.** M = 1000 independent Monte Carlo samples: ξ = `numpy.random.default_rng(2004).standard_normal((1000, 5))`, x_i = F_i⁻¹(Φ(ξ_i)). Ranges: D 38.74–39.32 mm, L 399.7–400.3 mm, F 825.4–2945 N, T 139.3–373.1 N m, S 70.01–80 MPa; all F > 0. g lies in [4.5173e5, 4.7494e7] Pa; no sample fails (printed by `generate_data.py`).

**Published results** (p. 1205). P{g(X) < c}, with the number of function evaluations in parentheses and Monte Carlo (MCS) with N = 10⁶:

| table | c (Pa) | FORM | SORM | FOSPA | MCS |
|---|---|---|---|---|---|
| Table 9 | 0 | 7.007 × 10⁻⁷ (1472) | 4.3581 × 10⁻⁷ (1514) | 6.1754 × 10⁻⁴ (102) | 7.850 × 10⁻⁴ |
| Table 10 | 4.5 × 10⁷ | 0.96798 (212) | 0.97406 (254) | 0.99927 (55) | 0.99938 |
| Table 11 | 2.48 × 10⁷ | 0.1538 (93) | 0.1406 (135) | 0.4825 (43) | 0.5038 |

Li et al., *Structural Safety* 73:42–53, 2018 (doi:10.1016/j.strusafe.2018.02.005), Table 9: sequential surrogate (44 evaluations) 7.52 × 10⁻³, as listed by UQTestFuns (not checked). No active subspace is published for this problem.

**What the notebook finds.** n = 1 in both scalings. Scaled by sd: λ₁/λ₂ = 94, λ₂/λ₃ = 200.5, w₁ = (−0.0582, 0.0005, 0.4639, 0.7698, −0.4345) for (D, L, F, T, S), a quadratic in y₁ with LOO R² = 0.99354. Through the CDF: λ₁/λ₂ = 15.71, λ₂/λ₃ = 3.077 (comparable), w₁ = (−0.0575, 0.0005, 0.4914, 0.7504, −0.4383), LOO R² = 0.96335. D and L (sd 0.1 mm) carry about 3 × 10⁻³ and 3 × 10⁻⁷ of the trace; L is the last eigenvector (λ₅ nine to ten orders of magnitude below λ₁). Monte Carlo with 2 × 10⁷ samples gives P{g < 0} = 7.7205e-04 (95 % interval [7.5988e-04, 7.8422e-04]), P{g < 4.5 × 10⁷} = 0.99936 [0.99935, 0.99937] and P{g < 2.48 × 10⁷} = 0.5039 [0.50368, 0.50412]; FORM gives β = 3.19455, Φ(−β) = 7.0025e-04 at c = 0, and 0.99882 (β = 3.04145) and 0.46424 (β = 0.08976) at the other thresholds; SORM at c = 0 gives 6.9886e-04 (Breitung) and 7.0473e-04 (Hohenbichler–Rackwitz).

## Regenerating

```bash
python generate_data.py --check                 # writes to a temporary directory and compares
python generate_data.py --out DIR --check       # the same, keeping the files in DIR (not this directory)
python generate_data.py --out .                 # from this directory: rewrites the shipped files
python check_accuracy.py                        # the numbers marked below
```

`--out` is required without `--check` (with `--check` it defaults to a temporary directory, removed afterwards), and `--check` refuses this directory and any directory inside it; it exits with code 1 only on a structural mismatch: a missing file, a different shape or NaN pattern, or inputs that differ from the shipped ones by more than 1e-8 (relative to the largest value per column). One process. The shipped files were written on 2026-11-09 in a Linux container (Python 3.11.18, numpy 2.4.6, scipy 1.17.1). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-11-10) `python generate_data.py --check` takes under 1 s (the script reports 0.04 s for the draws, the model, the writing and the comparison) and prints, per file, the maximum |difference| / max |shipped value| per column: at the rounding level, of order 1e-16, for all three files (largest 3.7e-16, `gradients.csv`, in `.venv` on macOS, Python 3.11, 2026-11-12). The inputs pass through `scipy.special` (`ndtr`, `log_ndtr`), whose last bits can differ between platforms and library versions.

## Accuracy

- **Gradients** against central differences (step 10⁻⁶ × the input's mean), each component scaled by the input's sd: largest relative error (row 2-norm) 3.6 × 10⁻¹⁰ over all 1000 samples (`generate_data.py`); 2.3 × 10⁻¹⁰ on the first 20 (notebook).
- **Gradients in the standard normal coordinates**, ∂g/∂ξ, against central differences in ξ (step 10⁻⁶): 3.2 × 10⁻⁹ (`check_accuracy.py`, check 2). dx/dξ against central differences of the map: 2.2 × 10⁻⁷, the differences' truncation error in the Gumbel tail (check 1). Round trip ξ → x → ξ on the shipped samples: at the rounding level, about 10⁻¹⁵ (check 1).
- **Against UQTestFuns 0.7.0** at the shipped inputs (run on 2026-11-09 in the Linux container; `.venv` does not have the package): g agrees to about 10⁻¹⁵ of max |g| (check 3); Φ⁻¹ of UQTestFuns' marginal CDFs agrees with `to_normal` to 3 × 10⁻¹⁴ or less per input, and its Gumbel location and scale equal `MU_F`, `BETA_F` to the six printed decimals (check 4).
- **Failure probability with UQTestFuns' own sampler** (same run in the Linux container), 10⁷ samples (seed 2026): 7.7250e-04, 95 % interval [7.5528e-04, 7.8972e-04] (check 5), consistent with the notebook's 7.7205e-04 from `shaft_model`'s sampler.

Checks 3–5 need `pip install uqtestfuns==0.7.0`; without it the script runs checks 1–2 and exits with an error naming the package. In `.venv` (no uqtestfuns; 2026-11-10) it prints the check 1–2 values above and exits with that error in under 1 s. With the package, in the Linux container, the whole script took about 3 s.

## How these files relate to the paper

- **The model is the article's.** Eq. (34) and Table 8 (p. 1205) define it; UQTestFuns' implementation agrees with `shaft_model` (check_accuracy.py). The 1000 Monte Carlo samples, their gradients, the probability estimates and the active subspace are new computations here; the article has no active subspace.
- **Monte Carlo agrees at all three thresholds** (notebook): P{g < 0} 7.7205e-04 [7.5988e-04, 7.8422e-04] against 7.850 × 10⁻⁴ (Table 9; its own 95 % binomial interval [7.3011e-04, 8.3989e-04]; 0.45 combined standard errors); P{g < 4.5 × 10⁷} 0.99936 against 0.99938 (Table 10; 0.82); P{g < 2.48 × 10⁷} 0.5039 against 0.5038 (Table 11; −0.20). Together these check the formula, the units and the five marginals.
- **FORM and SORM.** The article prints FORM 7.007 × 10⁻⁷ and SORM 4.3581 × 10⁻⁷ for P{g < 0} (Table 9). This data set's FORM gives Φ(−β) = 7.0025e-04 (β = 3.19455; every converged start of the search reaches the same design point) and its SORM 6.9886e-04 (Breitung) and 7.0473e-04 (Hohenbichler–Rackwitz), close to the Monte Carlo value. The article's FORM values at the other thresholds also differ from this data set's: 0.96798 against 0.99882 (Table 10) and 0.1538 against 0.46424 (Table 11). The article's FORM and SORM values are therefore not reproduced at any of the three thresholds. The article gives no design points or reliability indices, so the source of the difference cannot be determined from it.
- **Li et al. (2018).** The listed 7.52 × 10⁻³ is about ten times the Monte Carlo value; it was not checked.

## Open questions

1. For the authors: how were the FORM and SORM values of Tables 9–11 obtained (transformation to standard normal variables, search, design point)? A standard FORM in the independent standard normal coordinates gives β = 3.19455 (Φ(−β) = 7.00 × 10⁻⁴) at c = 0, and 0.99882 and 0.46424 at the Table 10 and 11 thresholds.
2. For the authors: were the five inputs taken as independent?

## License and credit

Code (`shaft_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`); generated here. No UQTestFuns code is copied: `shaft_model.py` is written from the formula and distributions, which UQTestFuns (MIT, © 2022 Damar Wicaksono) documents. Cite X. Du and A. Sudjianto, *AIAA Journal* 42(6):1199–1207, 2004, doi:10.2514/1.3877, for the problem, and D. Wicaksono and M. Hecht, *J. Open Source Softw.* 8(90):5671, 2023, doi:10.21105/joss.05671, for its definition in UQTestFuns.

## Unknowns

- The dependence of the inputs (taken as independent here; see Open question 2).
- How the article's FORM and SORM values were computed (Open question 1).
