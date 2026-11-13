# Cole_Cole_Cell_Monolayer_Impedance: data

Kind: **Python model**. The model is `cole_model.py` (closed form), and `generate_data.py` writes `inputs.csv`, `outputs.csv` and `gradients.csv` from it with a saved seed.

## Files

| file | contents | role |
|---|---|---|
| `Cole_Cole_Cell_Monolayer_Impedance.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 5861; outputs committed |
| `cole_model.py` | `impedance(X, omega)`, `impedance_and_grad(X, omega)`, `is_physical(X)`; `NAMES`, `UNITS`, `MEAN`, `SD` (Table 1), `LOG_MU`, `LOG_SIGMA` | the model |
| `generate_data.py` | draws the inputs and writes the three CSVs; `OMEGA`, `SEED`, `N`; `--check` | regenerates the data |
| `check_accuracy.py` | recomputes the numbers of the Accuracy section below, with an implementation of the model independent of `cole_model.py` | check |
| `inputs.csv` | 1100 × 4: R_inf (Ω), R_0 (Ω), alpha (–), tau (s), physical units, all draws | model input |
| `outputs.csv` | 1100 × 62: Re Z at the 31 frequencies, then Im Z at the 31 frequencies (Ω) | model output |
| `gradients.csv` | 1100 × 248: exact ∂Re Z/∂x_j and ∂Im Z/∂x_j w.r.t. the **physical** inputs; column 124 p + 4 k + j for part p (0 Re, 1 Im), frequency k, input j, so `G.reshape(N, 2, 31, 4)` | model output |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column names), so `np.loadtxt(path, delimiter=',')` reads them. Sizes: 0.07, 1.3 and 5.3 MB. The 31 angular frequencies (rad/s) are ω = 10^(k/4), k = 0, …, 28 (1 to 10⁷), plus 2·10⁵ and 2·10⁶, sorted (`generate_data.OMEGA`); the header of `outputs.csv` lists them.

## Model

**Paper.** J. Zhuang, C. Zhu, R. Han, A. Steuer, J. F. Kolb, F. Shi, "Uncertainty quantification and sensitivity analysis for the electrical impedance spectroscopy of changes to intercellular junctions induced by cold atmospheric plasma", *Molecules* 27(18):5861, 2022, doi:10.3390/molecules27185861 (open access, CC BY 4.0). Read from the publisher's PDF.

**Equation.** Eq. (1), Sec. 3.3 (p. 11): two Cole terms in series, Z = Z_ep + Z_cells, for electrode polarization and the cell monolayer. The paper's UQ and SA (Sec. 2.1, Figs. 1–3, Table 1) are on the four cell-monolayer parameters, and it gives no values for the electrode term, so the model is the cell-monolayer term

Z(ω; x) = R_inf + (R_0 − R_inf) / (1 + (iωτ)^α),  x = (R_inf, R_0, α, τ),

with (iωτ)^α = (ωτ)^α e^{iαπ/2} (principal branch, τ > 0). `cole_model.py` evaluates it in that polar form.

**Frequency axis.** The paper's figures label their axis "ω Hz" and its text speaks of "200 kHz" (peak of the imaginary part, p. 3) and "2 MHz" (p. 5). The imaginary part peaks at ωτ = 1, i.e. ω = 2·10⁵ rad/s for τ = 5 µs, which is where Fig. 1b puts it; with ω = 2πf the peak would be at f = 3.2·10⁴ on that axis. So the paper's axis value is ω in rad/s, the variable of Eq. (1), and we evaluate the model at the paper's axis values taken as ω. The notebook checks this (Monte Carlo, comparison section).

**Inputs** (m = 4; Table 1, p. 4: Gaussian, mean and "Sigma", the standard deviation; no truncation stated).

| name | symbol | units | Table 1 mean | Table 1 sd | physical range (source) | used here |
|---|---|---|---|---|---|---|
| `R_inf` | R_inf | Ω | 200 | 60 | > 0, < R_0 | lognormal, mean 200, sd 60 |
| `R_0` | R_0 | Ω | 1000 | 300 | > R_inf | lognormal, mean 1000, sd 300 |
| `alpha` | α | – | 0.7 | 0.21 | 0 < α < 1 (Sec. 3.3, p. 11) | lognormal with mean 0.7, sd 0.21 before the restriction to α < 1; **after it, mean 0.657, sd 0.158** |
| `tau` | τ | s | 5·10⁻⁶ | 1.5·10⁻⁶ | > 0 | lognormal, mean 5·10⁻⁶, sd 1.5·10⁻⁶ |

**The analysed α does not have the Table 1 moments.** Restricting the lognormal (mean 0.7, sd 0.21) to the physical range α < 1 gives a distribution with mean 0.657 and sd 0.158 (exact; the notebook prints them, and the 1022 physical rows have sample values 0.6534 and 0.157): 25 % less spread in α than Table 1. A Gaussian truncated to (0, 1), the other natural reading of Table 1, would have mean 0.668 and sd 0.182. The other three inputs keep their Table 1 moments up to sampling error.

**Input distribution: our choice, not the paper's.** Every Table 1 sd is 30 % of its mean. The lognormals have the Table 1 means and sds: ln x_j ~ N(μ_j, σ²) with σ² = ln(1 + 0.3²) (σ = 0.2936 for all four) and μ_j = ln(mean_j) − σ²/2 (`cole_model.LOG_MU`, `LOG_SIGMA`). We did not use the Gaussians because
- untruncated, they give α ≥ 1 with probability 7.7 %, and τ ≤ 0 (where the model is undefined) and α ≤ 0 each with probability 4.3·10⁻⁴;
- a Gaussian τ has positive density at τ = 0⁺, where ∂Z/∂τ ∝ τ^(α−1), so E[(∂Z/∂τ)²] is infinite whenever α ≤ 1/2 (probability 0.17 under Table 1): the gradient-based matrix C does not exist under the paper's distribution, and its sample estimate grows with the sample size (notebook: C_ττ for Re Z(2·10⁶) is 1423, 2577, 3496, 4893 for M = 10³ … 10⁶ Gaussian samples, against 444.5–450.6 for the lognormals).
- These arguments call for a non-Gaussian **τ** (and an α that stays positive). **R_inf and R_0** do not need it: they are lognormal too only so that one `ac.Gaussian` space in ln x covers all four inputs with the same scaling. That is a convenience, not a necessity.

Lognormal α exceeds 1 with probability 8.7 %; such samples are outside the physical range and are removed in the notebook by the stated rule `cole_model.is_physical` (R_inf > 0, R_0 > R_inf, 0 < α < 1, τ > 0; only α < 1 and R_0 > R_inf can fail for lognormals, the latter with probability 5·10⁻⁵).

**Sampling design.** Monte Carlo: N = 1100 independent draws from the four lognormals, `numpy.random.default_rng(5861).standard_normal((1100, 4))`, rounded to 12 significant digits, all written to `inputs.csv`. 78 rows have α ≥ 1 (largest 1.617) and none has R_0 ≤ R_inf, so **1022 rows are physical**; these are a Monte Carlo sample of the lognormals conditioned on α < 1, and they are the rows the notebook analyses. The model is still defined for 1 ≤ α < 2 (the denominator cannot vanish), so the 78 rows have finite outputs; they are kept in the files and not analysed.

**Sensitivity to the α cut-off.** The size of the headline gap is set by the upper tail of α near its physical limit. Recomputing the gradient matrix from the files with different cuts on α (same normalization; notebook, "Sensitivity to the α cut-off"):

| α cut-off | rows | Im Z(2·10⁵): λ1/λ2 | Re Z(2·10⁶): λ1/λ2 |
|---|---|---|---|
| α < 0.8 | 820 | 155.1 | 21.7 |
| α < 0.9 | 948 | 151.0 | 18.8 |
| α < 1 (headline) | 1022 | 138.3 | 16.1 |
| α < 1.1 | 1054 | 126.3 | 15.2 |
| α < 1.2 | 1064 | 111.5 | 14.8 |
| no cut | 1100 | 17.1 | 13.4 |

The 74 physical rows with α > 0.9 (7.2 %) contribute 25.9 % of λ2 of Im Z(2·10⁵). The cut at α < 1 is the paper's physical range (p. 11), and rows beyond it move towards the singularity of 1 + (iωτ)^α at ωτ = 1 as α → 2 and bring in large gradients, so we keep it as the headline; a cut by α is exact rejection sampling, so each row of the table is a Monte Carlo estimate for the lognormals restricted to that cut (the design is not biased). Normalizing ln α by its sample sd on the physical rows (0.2496) instead of σ = 0.2936 gives λ1/λ2 = 133.3 and 13.8 and moves each component of w1 by at most 0.063.

**Quantities of interest.** Re Z(ω) and Im Z(ω) in Ω at each of the 31 frequencies (Im Z is negative). The notebook's main sections use Im Z at ω = 2·10⁵ (the paper's imaginary peak) and Re Z at ω = 2·10⁶ (the paper's low/high boundary), with a sweep over all 31 frequencies.

**Gradients.** Exact, from the closed form: with L = ln(ωτ) + iπ/2, s = e^{αL}, D = 1 + s, ∂Z/∂R_inf = 1 − 1/D, ∂Z/∂R_0 = 1/D, ∂Z/∂α = −(R_0 − R_inf) s L / D², ∂Z/∂τ = −(R_0 − R_inf) α s / (τ D²); the real and imaginary parts are the gradients of Re Z and Im Z. Stored w.r.t. the physical inputs. The notebook converts once to the log inputs, ∂f/∂ln x_j = x_j ∂f/∂x_j, and passes them with `ac.Gaussian(LOG_MU, LOG_SIGMA**2)`, which normalizes each log input by its σ (the same σ for all four). This σ is that of the untruncated lognormal, so after the cut at α < 1 the normalized ln α has sample mean −0.19 and sd 0.85; normalizing it by its restricted sample sd instead changes little (see "Sensitivity to the α cut-off").

**Published results.** No active subspace. Fig. 1: mean, sd and 90 % band of Re Z, Im Z, |Z| and phase from 1 to 10⁷ on the ω axis; Fig. 2: first-order Sobol' indices of Re Z and Im Z over frequency; Fig. 3: frequency-averaged first-order and total indices; text values on pp. 3–5 (e.g. at 2 MHz, S_Rinf = 0.2 and S_R0 = 0.19 for Re Z). All by polynomial chaos under the Table 1 Gaussians; order, sample size and the averaging grid are not reported.

## Regenerating

```bash
python generate_data.py                       # writes the three CSVs here
python generate_data.py --check               # writes to a temporary directory, compares, and removes it
python generate_data.py --out DIR --check     # the same, keeping the files in DIR (outside this directory)
```

The shipped files were written on 2026-10-31 in a Linux container. In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, macOS; 2026-11-02) `generate_data.py` takes under 1 s (0.3–0.5 s for the draws, the model and the writing; under 1 s with the start of Python). `--check` writes to a temporary directory unless `--out DIR` is given, and refuses `--out` set to this directory, so the shipped files are never overwritten. It prints, for each file, the largest relative difference of an entry from the shipped file and the number of entries that differ. In `.venv` (macOS, Python 3.11.16, numpy 2.4.6; 2026-11-12) it took under 0.4 s with the start of Python: `inputs.csv` is reproduced exactly (the design comes from the seed, rounded to 12 significant digits), and `outputs.csv` and `gradients.csv` differ in the last of their 13 digits in a few entries (20 of 68,200 and 218 of 272,800), with largest relative differences of 3.9e-13 and 3.9e-11 (the larger value is a small gradient entry that loses digits to cancellation). The text of these two files is therefore not identical to the shipped text on every platform. `--check` exits with status 1 only on a structural mismatch: a file of a different shape, a different NaN pattern, or `inputs.csv` differing by more than 1e-8 (relative).

## Accuracy

`check_accuracy.py` recomputes these numbers in `.venv` (about 1 s). They are at the rounding level, so they are given as orders of magnitude.

- **Model values** (`check_accuracy.py`). An implementation independent of `cole_model.py`, with numpy's complex power, `R_inf + (R_0 − R_inf)/(1 + (1j*ω*τ)**α)`, agrees with `outputs.csv` to a relative error of order 1e-13 (|model − file| / |Z|, all 1100 rows, 31 frequencies, both parts; the files hold 13 significant digits). The notebook compares `cole_model.py` with the file and finds the same order.
- **Gradients** (`check_accuracy.py`). Against central differences of that implementation in the log inputs (step h in ln x_j), the relative error of the 4-vector x ⊙ ∇f over all 1100 rows, both parts and 31 frequencies is at most of order 1e-6 for h = 10⁻⁴, 1e-8 for h = 10⁻⁵ and 1e-9 for h = 10⁻⁶ (median of order 1e-10): truncation error falling as h² down to round-off. The notebook repeats the check on the 1022 physical rows with h = 10⁻⁶ and finds the same orders.

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from polynomial chaos under Gaussian inputs, with unreported order, samples and code. The data here are an independent Monte Carlo computation of the same formula under lognormal inputs with the same means and sds, restricted to α < 1. The notebook compares with the paper under the paper's Gaussians (Monte Carlo, marginals truncated only to α > 0 and τ > 0): the imaginary peak location, its mean (−247 Ω), the low-frequency mean and sd of Re Z (996/299 Ω) and the Re Z band at 10⁷ (126–386 Ω, matching Fig. 1a; the text says 200–400) agree. Not reproduced: **the paper's finding that τ is negligible** (the first-order index of τ reaches 0.20 for Re Z and 0.19 for Im Z, where the paper reports values near 0 and below 0.1); the sd of Re Z near 10⁵ (221 Ω against about 270 in Fig. 1a); the Re Z indices at 2 MHz (S_Rinf 0.24 ± 0.01, S_R0 0.13 ± 0.01 against 0.2 and 0.19; our crossover is between 5.6·10⁵ and 10⁶); and the paper's large total-minus-first-order gaps (Fig. 3b: R_inf for Im Z 0.21 vs 0.03; ours 0.02 vs 0.01). The active-subspace numbers (λ1/λ2 = 138.3 and 16.1) are for the lognormals restricted to α < 1, whose α has mean 0.657 and sd 0.158, not the Table 1 values.
- **Two readings of the paper, kept.** The inputs are lognormal for all four parameters, with Table 1's means and standard deviations, and the rows with α ≥ 1 are not analysed, so the analysed α has standard deviation 0.158 against Table 1's 0.21 (see "Model"). The paper's axis label "ω Hz" is read as ω in rad/s, so its "200 kHz" and "2 MHz" are ω = 2·10⁵ and 2·10⁶ rad/s (see "Frequency axis"). Both are choices of this data set; the open questions below ask the authors about them.
- **Electrode term.** Left out (no values in the paper). With it, the impedance at low frequency would be dominated by electrode polarization and the low-frequency results would change.
- **Measured data.** The paper's spectra and fitted Cole parameters (Figs. 4–6) are "available on request from the corresponding author" and are not used.

## Open questions for the authors

- Is the frequency axis of Figs. 1–3 ω in rad/s (as the peak location implies), with "200 kHz" and "2 MHz" meaning ω = 2·10⁵ and 2·10⁶ rad/s?
- Were the Table 1 Gaussians truncated (α ≥ 1 has probability 7.7 %, τ ≤ 0 has 4.3·10⁻⁴), and how were samples with α ≥ 1 handled?
- Polynomial-chaos order, number of samples or quadrature nodes, and the frequency grid used for the averages in Fig. 3; the code used (ref. [29] is an application paper).
- Why the total indices in Fig. 3 exceed the first-order ones by about 0.2 (R_0 and R_inf, Im Z), when Monte Carlo under the same Gaussians gives gaps of at most 0.09; and why τ's first-order index stays near 0 in Fig. 2 when it reaches about 0.2 near ω = 3·10⁵ (Re Z) and 3·10⁶ (Im Z) in our computation.
- Values of the electrode-polarization parameters of Eq. (1).
- Are the Table 1 standard deviations measured, or a chosen 30 % coefficient of variation? Every sd is exactly 0.3 × mean, while p. 12 says they were "determined" from the fits.

## License and credit

Code (the `.py` files) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`); they are computed here from the published formula and Table 1, and no outside data are included. Model and parameter values: J. Zhuang, C. Zhu, R. Han, A. Steuer, J. F. Kolb and F. Shi (article CC BY 4.0). Cite Zhuang et al., *Molecules* 27(18):5861, 2022, doi:10.3390/molecules27185861.
