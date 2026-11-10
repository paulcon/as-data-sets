# Cavity_Shielding_Effectiveness: data

Kind: **Python model**. The model is `cavity_se_model.py` (numpy only), written from the paper's printed equations. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs; a rerun reproduces them to the tolerances under "Regenerating".

## Files

| file | contents | role |
|---|---|---|
| `Cavity_Shielding_Effectiveness.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2021; outputs committed |
| `cavity_se_model.py` | `se_db(X, f_hz, grad=False)`, `band_mean(X, grad=False)`, `qois(X, grad=False)`, `parts(X, f_hz)`, `kl_over_pi(X, f_hz)`; `NAMES`, `UNITS`, `DESCRIPTIONS`, `LOWER`, `UPPER`, `NOMINAL`, `NORMAL_SD`, `FREQS_GHZ`, `BAND_GHZ`, `QOI_NAMES` | the model |
| `generate_data.py` | draws the samples and writes the three CSVs; `--out DIR --check` compares a rerun with the shipped files; checks gradients by finite differences | regenerates the data |
| `check_accuracy.py` | recomputes the numbers below that the notebook and `generate_data.py` do not print | check |
| `inputs.csv` | 1000 × 6 physical inputs d (in), l_p (in), w (mil), h (in), a (in), sigma (10⁷ S/m) | model input |
| `outputs.csv` | 1000 × 4: SE bound (dB) at 1, 2, 3 GHz and its mean over 51 frequencies 1–3 GHz | model output |
| `gradients.csv` | 1000 × 24: ∂(output)/∂(d, l_p, w, h, a, sigma), six columns per output in the order of `outputs.csv`, with respect to the **physical** inputs in the units above (dB/in, dB/in, dB/mil, dB/in, dB/in, dB per 10⁷ S/m) | model output |

The CSVs have two `#` lines (a description and the column names with units), are read by `np.loadtxt(path, delimiter=',')`, and hold 17 significant digits.

## Model

**Source.** S. Campione, J. A. Stephens, N. Martin, A. Eckert, L. K. Warne, G. Huerta, R. A. Pfeiffer, A. Jones, "Developing uncertainty quantification strategies in electromagnetic problems involving highly resonant cavities", *J. Verif. Valid. Uncert. Quantif.* 6(4):041003, 2021, doi:10.1115/1.4051906. Read from the accepted manuscript (Sandia report SAND2021-8020J, 37 pages); page numbers below are those of the manuscript and should be checked against the published version. The model is the matched power-balance bound of Appendix A (pp. 20–21), which the paper takes from its refs. [5] (Warne et al., SAND2016-9029) and [8] (Campione et al., IEEE Trans. EMC 62(5):1980–1988, 2020, doi:10.1109/TEMC.2020.2977600). The unmatched modal model (Appendix B) and the full-wave EIGER runs of Sec. 4 are not built.

**Geometry** (Sec. 2, p. 4). Aluminium-alloy cylinder, interior height h and radius a, wall thickness d, wall conductivity σ, with an azimuthal slot of width w and length ℓ midway up its side; plane-wave illumination along −x with E along z.

**Equations** (as printed, pp. 20–21). Eq. (5): ℓ|H₀|²(8/π²) / (R_int π²/(ωLℓ)² + G_rad) = P_rec = P_wall = 2AR_S⟨|H_j|²⟩_A, with A = 2πa² + 2πah, R_S = √(ωμ₀/(2σ)), L = μ₀π/Ω_e, Ω_e = Ω + C_e, Ω = 2 ln(ℓ/a_eq), C_e = 2(ln 2 − 7/3), a_eq = 2w/(π exp^{1+πd/(2w)}), R_int = Re(Z_int), Z_int = (L/L_int^r)² Z_int^r, Z_int^r = 2Z_S/d, Z_S = (1+i)R_S, L_int^r = μ₀w/d, and πη₀ℓG_rad = −[1+cos kℓ] + (kℓ/(2π) + π/(2kℓ))[Cin(kℓ+π) − Cin(π−kℓ)] + (kℓ/2)[1 − (π/kℓ)²][Si(kℓ+π) − Si(π−kℓ)]. Eq. (6): SE_max ≈ 4⟨SE⟩ = 4⟨|H_j|²⟩_A/|H₀|². Since L and L_int^r are real, R_int = (L/L_int^r)² 2R_S/d (coded so). Constants: μ₀ = 4π × 10⁻⁷ H/m, c = 299 792 458 m/s, η₀ = μ₀c.

**Choices made here** (not stated in the paper):
1. **dB conversion.** Eq. (6) is a ratio of squared H-fields, so the output is 10 log₁₀ SE_max, i.e. 20 log₁₀ of the H-field ratio (Eq. (1), p. 4, is written for the E-field ratio). The paper is not consistent here: Eq. (1) converts a field ratio by 20 log₁₀, while Eq. (6) (p. 21) calls the squared ratio "SE". Read literally, 20 log₁₀ of Eq. (6) gives 47.85 dB at the nominal point at 2 GHz (notebook), outside the 0–35 dB axes of Fig. 3 (p. 28), whose curves lie near 23–24 dB; 10 log₁₀ gives 23.925 dB, and with it the one-at-a-time sweeps match Fig. 3 (notebook, comparison table).
2. **Slot length.** Tables 1–2 (pp. 33–34) and the Fig. 1 caption call the input the projected length ℓ_p; Table 3 (p. 35) and Figs. 2–3 label the same column ℓ. The p. 4 sentence points the other way from the tables: "the projected length used to setup the geometry in the simulation is computed from the formula ℓ = 2a sin⁻¹[ℓ_p/(2a)]", i.e. the paper derives ℓ_p *from* ℓ to build its full-wave geometry, which suggests that the slot (arc) length ℓ is the primary parameter. Two readings are equally plausible: (i) the Table 1 values are ℓ_p and Eqs. (5)–(6) use the arc length computed from them; (ii) the Table 1 values are the arc length ℓ itself, which in the code is the same as setting ℓ = ℓ_p. This data set uses (i), so that the input keeps Table 1's label ℓ_p; that is this data set's choice. With reading (ii), the nominal SE is 23.156, 23.848, 21.928 dB at 1, 2, 3 GHz instead of 23.285, 23.925, 21.967, and w₁ of SE_2GHz changes by at most 0.0056 per entry, with λ₁/λ₂ = 43.9 instead of 43.1 (`check_accuracy.py`, check 5). The paper's figures cannot separate the two readings.
3. **Distribution.** The sensitivity analysis states ranges only (p. 5); the inputs are independent and uniform on the Table 1 ranges. The notebook repeats the analysis at 2 GHz with Table 3's normal alternative (mean = midpoint, sd = range/6, p. 13), which the paper uses only for its unmatched model.
4. **Outputs.** The paper computes SE at 51 equally spaced frequencies 1–3 GHz (p. 8). The files hold SE at 1, 2 and 3 GHz and the mean in dB over those 51 frequencies; the notebook's main output is 2 GHz, the centre of the band, where every sample is below the slot resonance.

**Inputs** (m = 6):

| name | symbol | units | lower | upper | nominal (Table 3 mean) | Table 3 normal sd | source |
|---|---|---|---|---|---|---|---|
| `d` | d, slot depth (wall thickness) | in | 0.2 | 0.3 | 0.25 | 0.01667 | Table 1, p. 33; Table 3, p. 35 |
| `l_p` | ℓ_p, projected slot length | in | 1.5 | 2.5 | 2.0 | 0.1667 | Table 1, p. 33; Table 3, p. 35 |
| `w` | w, slot width | mil | 5 | 25 | 15 | 3.333 | Table 1, p. 33; Table 3, p. 35 |
| `h` | h, cavity interior height | in | 21.6 | 26.4 | 24.0 | 0.8 | Table 1, p. 33; Table 3, p. 35 |
| `a` | a, cavity interior radius | in | 3.6 | 4.4 | 4.0 | 0.1333 | Table 1, p. 33; Table 3, p. 35 |
| `sigma` | σ, wall conductivity | 10⁷ S/m | 2.2 | 3.0 | 2.6 | 0.1333 | Table 1, p. 33; Table 3, p. 35 |

The paper says these ranges "are not meant to capture physical uncertainties" but show how the parameters affect SE (p. 6).

**Quantities of interest.** SE_1GHz, SE_2GHz, SE_3GHz: 10 log₁₀ SE_max (dB) at those frequencies; SE_band: the mean of 10 log₁₀ SE_max over 1, 1.04, …, 3 GHz. Positive values mean the interior field bound exceeds the incident field. At the nominal point: 23.285, 23.925, 21.967 dB (notebook). The matched model has no cavity resonances, so SE is smooth in frequency; the TM₀₁₀ resonance near 1.129 GHz (p. 5) belongs to the unmatched model. The paper states the model for frequencies below the slot resonance (p. 20). At 1 and 2 GHz every sample is below it (kℓ/π ≤ 0.862); at 3 GHz 569 of the 1000 samples are above it (kℓ/π up to 1.293), and these same samples exceed it somewhere in the band (notebook, `generate_data.py`). They are kept, as the paper's own 1–3 GHz study keeps them. The formula is smooth through kℓ = π (`check_accuracy.py`, check 4).

**Gradients.** Complex-step derivatives, ∂f/∂x_j = Im f(x + i t e_j)/t with t = 10⁻³⁰ in the input's paper units: accurate to rounding (no subtractive cancellation). Every operation is analytic in the inputs. Si and Cin are evaluated by their power series (45 terms, complex-safe) for |z| ≤ 12; the model raises beyond (frequencies above about 4.8 GHz for the longest slot). Coordinates: physical, paper units.

**Sampling design.** M = 1000 independent Monte Carlo draws, uniform on the Table 1 box: `LOWER + (UPPER − LOWER) * numpy.random.default_rng(2021).random((1000, 6))`. Every sample is in the model's domain (positive sizes, ℓ_p < 2a).

**Published results** (no active subspace in the paper). Fig. 2 (p. 27): first-order and total Sobol indices of the matched bound vs frequency from fourth-order PCE (Dakota, orthogonal matching pursuit, 10-fold CV, MSE 3 × 10⁻⁴ to 7 × 10⁻⁴) on 448 Latin hypercube runs (pp. 8–9); read by eye, S_w ≈ 0.77 and T_w ≈ 0.77 at 1 GHz, ≈ 0.69 and 0.73 at 3 GHz; S_ℓ ≈ T_ℓ ≈ 0.20 at 1 GHz, ≈ 0.18 and 0.22 at 3 GHz; the other four below about 0.04. Text, p. 9: slot width "70%- 80% of the variation in SE". Fig. 3 (p. 28): one-at-a-time sweeps. Sec. 4, Tables 4–5 and Figs. 6–7: peak TM₀₁₀ SE of the unmatched model (normal inputs mean 11.3, sd 3.1 dB; uniform 10.7, 5.5 dB; Table 5, p. 37), not comparable with this data set.

**What the notebook finds.** A dominant direction at every frequency, with a clear second one. λ₁/λ₂ = 34.4, 43.1, 36.6 and 40.2 and λ₂/λ₃ = 55.0, 15.9, 10.1 and 20.5 for SE_1GHz, SE_2GHz, SE_3GHz and SE_band. The notebook chooses n per output from these ratios, their bootstrap intervals, the share of the trace left out and leave-one-out R²: n = 1 at 2 GHz (λ₁/λ₂ 39.9–47.5 over the bootstrap against λ₂/λ₃ 14.4–17.3), at 3 GHz and for the band mean; n = 2 at 1 GHz, where the larger gap is after λ₂ (λ₂/λ₃ 49.8–60.5 against λ₁/λ₂ 32.2–37.2), only 0.00064 of the trace lies outside λ₁, λ₂, and the quadratic leave-one-out R² rises from 0.97384 in y₁ to 0.99710 in (y₁, y₂). (The library's default linear-gap rule, λ_k − λ_{k+1}, gives n = 1 for all four because λ₁ dominates; its log-gap rule gives 2, 1, 1, 1. Neither is the basis of the choice.) At 3 GHz λ₃/λ₄ = 28.5 is a further clear gap. At 2 GHz w₁ = (−0.0789, 0.3206, 0.9375, −0.0479, −0.0663, 0.0737) for (d, ℓ_p, w, h, a, σ), the same direction at 1 and 3 GHz to |w₁·w₁′| ≥ 0.99905. A cubic in y₁ has leave-one-out R² = 0.97182 at 2 GHz; a quadratic in (y₁, y₂) 0.99134. Sobol indices computed from the model (Jansen estimators, 2¹⁵ base samples) agree with Fig. 2 to the precision of reading it (S_w = 0.768 at 1 GHz, 0.687 at 3 GHz); the sweeps match Fig. 3.

## Regenerating

```bash
python generate_data.py --out DIR --check     # writes to DIR (not this directory) and compares
python generate_data.py --here                # maintainers only: rewrites the shipped files in this directory
python check_accuracy.py                      # the numbers marked below
```

`generate_data.py` refuses to write into this directory unless given `--here` (for maintainers only; users should use `--out`), and `--out` must be outside it (symbolic links resolved); `--here` cannot be combined with `--out` or `--check`. One process. The shipped files were written on 2026-11-09 in a Linux container (Python 3.11, numpy 2.4.6). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-11-10), `python generate_data.py --out DIR --check` took about 1.6 s (1.0 s for the 1000 samples × 54 frequencies × 7 evaluations) and passed: the inputs matched the shipped file exactly, and the outputs and gradients to rounding level (largest differences of order 10⁻¹⁶ and 10⁻¹⁵, relative to each column's largest value). The stated tolerance is a maximum difference of 10⁻¹⁵ relative for the inputs and 10⁻¹² relative to each column's largest value for the outputs and gradients (differences in libm rounding of exp, log, sin, cos amplified by the series).

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name; in `.venv` it took about 4 s (2026-11-10), and the values below are from that run.

- **Special functions** (`check_accuracy.py`, check 1). The Si and Cin series agree with `scipy.special.sici` to 3.7 × 10⁻¹³ (absolute) on [−12, 12] and to rounding level (about 10⁻¹⁴) on the arguments the design uses (|z| ≤ 7.20).
- **Outputs** (check 2). An independent implementation of Eqs. (5)–(6) with `scipy.special.sici`, written in `check_accuracy.py`, reproduces all 4000 shipped values to rounding level (largest difference of order 10⁻¹⁴ dB).
- **Gradients.** Against central differences of `cavity_se_model` (step 10⁻⁶ of each range): largest relative error (row 2-norm) 1.2 × 10⁻⁸ on the first 100 samples (`generate_data.py`) and 8.3 × 10⁻⁹ on five samples (notebook), the round-off level of the differences. Against Richardson-extrapolated central differences (steps 10⁻³ and 5 × 10⁻⁴ of each range) of the independent scipy implementation, all 1000 rows: largest 4.9 × 10⁻¹¹, median 2.8 × 10⁻¹² (check 3a); of `cavity_se_model` itself, rows 0–19: 2.4 × 10⁻¹¹ (check 3b). (scipy's `sici` does not carry a 10⁻³⁰ imaginary part for |z| > 1, so the scipy implementation cannot be complex-stepped.)
- **Slot resonance** (check 4). For the longest slot (ℓ_p = 2.5 in, a = 3.6 in, ℓ = 2.5532 in), kℓ = π at 2.3114 GHz; over ±0.1 GHz SE varies within 24.803–25.343 dB and ∂SE/∂w within 0.1390–0.1717 dB/mil, and the second difference in frequency at the resonance equals the median one (ratio 1.00 for both): no kink.
- **Monte Carlo error.** The Sobol estimates in the notebook carry 95 % bootstrap intervals (printed there).

## How these files relate to the paper

- **Not the paper's computation.** The paper's results come from Sandia's own implementation of the bound driven by Dakota (448 LHS runs, PCE surrogates). The data here are an independent numpy computation of the printed equations with the choices above, on a new Monte Carlo design; no code or data from the authors were used (none are published).
- **Agreement.** One-at-a-time sweeps (Fig. 3): w at 1 GHz 14.03–27.19 dB against about 14.5–27.3 read from the figure; ℓ_p at 1 GHz 19.66–25.97 against about 20–26; at 3 GHz 17.46–22.65 and 20.66–22.68 against about 17.5–22.7 and 21–23 (notebook). Sobol indices (Fig. 2) within about 0.01 of the values read from the figure at 1 and 3 GHz (notebook).
- **Not reproduced.** Table 5, Figs. 6–7 and Table 4 (unmatched model, EIGER, PCE of peak SE). This is not a choice of convenience: the unmatched model of Appendix B cannot be rebuilt from the paper. Its Eq. (8) (p. 22) needs the slot voltage drive V(0), which is given only as "proportional to the exterior short circuit current density at the center of the slot" K_z^sc, with no formula or constant; the symbols b and ε_m in Eq. (8) are not defined (only ε_n is); and the observation point r at which the peak SE is taken is not stated. In addition, Table 5 comes from PCE surrogates fitted to 112 Latin hypercube runs of that model (p. 13), not from the model itself, and Table 4 and Figs. 6–7 also use EIGER, a full-wave code not available here.

## Open questions

These need the authors (not contacted).

1. Whether the Table 1 slot lengths are projected lengths ℓ_p (as Tables 1–2 say) or arc lengths ℓ (as Table 3, Figs. 2–3 and the p. 4 sentence suggest), and which one the matched model takes (worth < 0.13 dB at the nominal point).
2. The dB conversion of Eq. (6) (10 log₁₀ of the squared-field ratio is assumed; the Fig. 3 match supports it, and the literal 20 log₁₀ would give 47.85 dB at the nominal point at 2 GHz, off Fig. 3's axes).
3. The input distribution of the Sec. 3 sensitivity analysis (uniform is assumed; Dakota's generalized PCE would use Legendre polynomials for it).
4. Whether frequencies above the slot resonance (kℓ > π, reached at 3 GHz for long slots) were meant to be in the Sec. 3 study, given the validity statement on p. 20 (below the slot resonance, electrically short slot); at 2 GHz kℓ/π already reaches 0.862.
5. Appendix B: what are the voltage drive V(0) (its constant of proportionality to K_z^sc), b and ε_m in Eq. (8), and at which point r is the peak SE of Tables 4–5 evaluated?

## License and credit

Code (`cavity_se_model.py`, `generate_data.py`, `check_accuracy.py`) and the notebook: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations; no data or code were copied. Cite S. Campione et al., *J. Verif. Valid. Uncert. Quantif.* 6(4):041003, 2021, doi:10.1115/1.4051906, and for the bound S. Campione et al., *IEEE Trans. Electromagn. Compat.* 62(5):1980–1988, 2020, doi:10.1109/TEMC.2020.2977600. The paper's work was partially supported by the Laboratory Directed Research and Development program at Sandia National Laboratories (contract DE-NA-0003525).

## Unknowns

- The five open questions above.
- The page numbers are those of the accepted manuscript; the published article's pagination was not checked.
- Fig. 2 and Fig. 3 values are read by eye from the figures (no tables of them are printed).
