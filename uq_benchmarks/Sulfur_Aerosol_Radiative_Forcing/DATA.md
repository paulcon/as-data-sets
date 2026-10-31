# Sulfur_Aerosol_Radiative_Forcing: data

Kind: **Python model**. The model is `sulfur_model.py`, a closed form. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs, and a rerun on the same platform reproduces them bit for bit (on another, to rounding error; see below).

## Files

| file | contents | role |
|---|---|---|
| `Sulfur_Aerosol_Radiative_Forcing.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 1992; outputs committed |
| `sulfur_model.py` | `forcing`, `grad_log`, `grad_physical`, `log_forcing`, `burden`, `moments`, `constant`; `NAMES`, `UNITS`, `MU`, `SIGMA`, `S0`, `AREA`, and Charlson's `NOMINAL_CHARLSON`, `REL_UNC_CHARLSON`, `S0_CHARLSON`, `AREA_CHARLSON` | the model (numpy only) |
| `generate_data.py` | draws the samples and writes the three CSVs; `--check` compares a rerun with the shipped files | regenerates the data |
| `inputs.csv` | 1000 × 9 physical inputs, columns in the order of `NAMES` (units in the header) | model input |
| `outputs.csv` | 1000 × 1: ΔF_R in W m⁻² | model output |
| `gradients.csv` | 1000 × 9: ∂ΔF_R/∂(ln x_j) in W m⁻², the gradient with respect to the **log** inputs | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits, so they round-trip exactly.

## Model

**Source.** R. J. Charlson, S. E. Schwartz, J. M. Hales, R. D. Cess, J. A. Coakley, Jr., J. E. Hansen, D. J. Hofmann, "Climate forcing by anthropogenic aerosols", *Science* 255(5043):423–430, 1992, doi:10.1126/science.255.5043.423. The PDF read is the publisher's scan (file named `_oa`, but it is the science.org PDF with an OCR text layer); Eqs. 1–5 and Table 1 were read from the page image of p. 426.

**Equations** (Charlson Eqs. 1–5, p. 426): ΔF_R = −½ F_T T² (1 − A_c)(1 − R_s)² β̄ δ, δ = α f(RH) B, B = Q Y τ / A. The benchmark version used here (UQTestFuns, "sulfur"; after Penner et al. 1994 and Tatang et al. 1997, Eq. 12, with UQTestFuns' corrections: S₀ not squared, factors 10¹² and 365 added) is

  ΔF_R = −½ S₀ (1 − A_c) T² (1 − R_s)² β̄ Ψ_e f_Ψ · 3·10¹² Q Y L / (365 A),

with S₀ = 1361 W m⁻² (Kopp and Lean 2011, via UQTestFuns) and A = 5.1 × 10¹⁴ m² (UQTestFuns labels its unit "days", a typo). The factor 3 is the sulfate-to-sulfur mass ratio 96/32. Charlson uses F_T = 1370 W m⁻² and A = 5 × 10¹⁴ m²; `sulfur_model` takes both as optional arguments.

**Inputs** (m = 9), independent lognormal, ln x_i ~ N(μ_i, σ_i²):

| name | symbol | units | μ | σ | median e^μ | Charlson nominal (Table 1, p. 426) | Charlson rel. uncertainty | sources |
|---|---|---|---|---|---|---|---|---|
| `Q` | Q_SO₂ | Tg S yr⁻¹ | 4.26268 | 0.139762 | 71 | 90 | 15 % | T1; UQTestFuns |
| `Y` | Y_SO₄²⁻ | – | −0.693147 | 0.405465 | 0.5 | 0.4 | 50 % | T1; UQTestFuns |
| `L` | τ_SO₄²⁻ | days | 1.70475 | 0.405465 | 5.5 | 0.02 yr = 7.3 d | 50 % | T1; UQTestFuns |
| `Psi_e` | α_SO₄²⁻ | m² g⁻¹ | 1.60944 | 0.336472 | 5 | 5 | 40 % | T1; UQTestFuns |
| `beta` | β̄ | – | −1.20397 | 0.262364 | 0.3 | 0.29 | 25 % | T1; UQTestFuns |
| `f_Psi` | f(RH) | – | 0.530628 | 0.182322 | 1.7 | 1.7 | 20 % | T1; UQTestFuns |
| `T2` | T² | – | −0.544727 | 0.336472 | 0.58 | T = 0.76 (T² = 0.5776) | 20 % (of T) | T1; UQTestFuns |
| `one_minus_Ac` | 1 − A_c | – | −0.941609 | 0.0953102 | 0.39 | 0.4 | 10 % | T1; UQTestFuns |
| `one_minus_Rs_sq` | (1 − R_s)² | – | −0.328504 | 0.182322 | 0.72 | 1 − R_s = 0.85 (0.7225) | 10 % (of 1 − R_s) | T1; UQTestFuns |

Charlson gives nominal values and relative uncertainties, not distributions. The lognormals are those of UQTestFuns, which attributes them to Penner et al. (1994); we did not read Penner et al., so the attribution is UQTestFuns'. Observations: σ_i = ln(1 + u_i) with u_i Charlson's relative uncertainty, doubled for the squared factors T² and (1 − R_s)², except β̄ (ln 1.3, not ln 1.25). The medians match Charlson's nominals except Q (71 vs 90), Y (0.5 vs 0.4), L (5.5 vs 7.3 days), β̄ (0.30 vs 0.29) and 1 − A_c (0.39 vs 0.4).

**Transform.** active_subspaces v1 has uniform and Gaussian input spaces only, so the analysis is in the log inputs z = ln x, which are exactly Gaussian: `ac.Gaussian(MU, SIGMA**2)`, normalized coordinates (z − μ)/σ. `inputs.csv` holds the physical x; the notebook passes Z = ln X with the gradients with respect to z.

**Quantity of interest.** ΔF_R in W m⁻² (negative, a cooling). The notebook also uses ln(−ΔF_R) = ln(−K) + Σ z_i, linear in z.

**Gradients.** Exact: ∂ΔF_R/∂z_i = x_i ∂ΔF_R/∂x_i = ΔF_R for every i, because every input enters with power 1. So the active subspace in log coordinates is exactly one-dimensional, w₁ = σ/‖σ‖, λ₁ = E[ΔF_R²] ‖σ‖², λ₂ = … = λ₉ = 0. Gradients with respect to the physical inputs are ΔF_R/x_i (`grad_physical`); the notebook uses them for a comparison in physical coordinates.

**Sampling design.** M = 1000 independent draws, x = exp(μ + σ ⊙ ξ), ξ ~ N(0, I₉), `numpy.random.default_rng(1992).standard_normal((1000, 9))`. M ≥ 10 m and ≥ 300.

**Published results.** Charlson et al. report a point estimate: Table 1 gives B = 4.6 × 10⁻³ g SO₄²⁻ m⁻² (4.8 × 10⁻⁵ mol m⁻²), δ = 0.04 and ΔF_R = −1.3 W m⁻²; the text (p. 426) gives −1 W m⁻² "uncertain to a factor of 2". No sampling study and no active subspace. UQTestFuns gives the analytic mean −1.2741 and variance 1.7089 of the benchmark (Monte Carlo with 10⁶ samples: −1.2741 and 1.7007).

## Regenerating

```bash
python generate_data.py                          # writes the three CSVs here
python generate_data.py --out /tmp/sulfur --check   # writes elsewhere and compares
```

The run takes 0.02–0.04 s (0.2 s with Python start-up) and also prints the gradient check. `--check` on 2026-10-30 (numpy 2.5.4, Python 3.13) reported the three files identical to the shipped ones. In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-10-31) they differ by rounding error: at most a relative 2.2e-16 (inputs) and 8.9e-16 (outputs and gradients). The notebook runs in about 3 s.

## Accuracy

- **Values.** Closed form, evaluated in double precision; `sulfur_model.forcing` reproduces `outputs.csv` exactly (relative difference 0, printed in the notebook).
- **Gradients.** Exact. Checked against central finite differences in z with step 10⁻⁶ at all 1000 samples: largest relative error (row 2-norm) 2.3 × 10⁻¹⁰, the size expected from rounding in the differences.
- **Moments.** `sulfur_model.moments()` gives mean −1.2741 and variance 1.70897 W² m⁻⁴, matching UQTestFuns' analytic −1.2741 and 1.7089 (which confirms the constants S₀ = 1361, A = 5.1 × 10¹⁴, 3, 10¹² and 365). The 1000-sample mean and variance are −1.2586 and 1.4944.
- **Rounding in the eigenvalues.** λ₂, …, λ₉ come out at about 10⁻³¹ or smaller instead of 0; the printed values change with the machine and the linear-algebra library. Three pairs of inputs share a σ (Y and L; Ψ_e and T²; f_Ψ and (1 − R_s)²), so their normalized gradient columns are identical and three directions are exactly null, which is why λ₇–λ₉ are near 10⁻⁶⁴ or smaller.

## How these files relate to the paper

- **Not the paper's computation.** Charlson et al. evaluate the model once at nominal values. The distributions, the solar constant and the area of the benchmark come from UQTestFuns (after Penner et al. 1994), not from this paper; the samples are new.
- **Charlson's Table 1 is not reproduced exactly by its own inputs.** 3 Q Y τ / A with Q = 90 Tg S yr⁻¹, Y = 0.4, τ = 0.02 yr, A = 5 × 10¹⁴ m² gives B = 4.32 × 10⁻³ g m⁻² (4.5 × 10⁻⁵ mol m⁻²), against the table's 4.6 × 10⁻³ (4.8 × 10⁻⁵); carried through, δ = 0.0367 and ΔF_R = −1.218 W m⁻², against the table's rounded 0.04 and −1.3. We could not tell how the table's burden was computed.
- **Two sets of nominals.** At the benchmark medians, ΔF_R = −0.889 W m⁻²; at Charlson's nominal values with the benchmark constants, −1.186; with Charlson's constants, −1.218.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations; no data were copied. The input distributions are parameter values tabulated by UQTestFuns (D. Wicaksono and M. Hecht, *J. Open Source Softw.* 8(90):5671, 2023, doi:10.21105/joss.05671; page https://uqtestfuns.readthedocs.io/en/latest/test-functions/sulfur.html, no license stated on the page). Cite:

- R. J. Charlson et al., "Climate forcing by anthropogenic aerosols", *Science* 255(5043):423–430, 1992, doi:10.1126/science.255.5043.423 (model).
- J. E. Penner et al., "Quantifying and minimizing uncertainty of climate forcing by anthropogenic aerosols", *Bull. Amer. Meteor. Soc.* 75(3):375–400, 1994, doi:10.1175/1520-0477(1994)075<0375:QAMUOC>2.0.CO;2 (input distributions, as attributed by UQTestFuns).
- M. A. Tatang, W. Pan, R. G. Prinn, G. J. McRae, "An efficient method for parametric uncertainty analysis of numerical geophysical models", *J. Geophys. Res.* 102(D18):21925–21932, 1997, doi:10.1029/97JD01654 (benchmark use of the model).
