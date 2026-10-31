# Geological_CO2_Storage_Leakage: data

Kind: **Python model**. The model is `co2_leakage_model.py`, a closed form reimplemented in numpy from the paper. `generate_data.py` draws the inputs with saved seeds and writes all six CSVs, and a rerun reproduces them bit for bit.

## Files

| file | contents | role |
|---|---|---|
| `Geological_CO2_Storage_Leakage.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2139; outputs committed |
| `co2_leakage_model.py` | `layers`, `leakage_time`, `min_flow`, `soft_min_flow`, `grad_log10_leakage_time`, `grad_log10_soft_min_flow`, `grad_log10_min_flow`, `argmin_layer`, `sample`, `bounds`, `nominal`, `names`, `units`, `descriptions`, `distributions`; `SITES` (Tables 3 and 4), constants | the model and its input distributions (numpy only) |
| `generate_data.py` | draws the samples and writes the six CSVs; `--check` compares a rerun with the shipped files | regenerates the data |
| `ketzin_inputs.csv` | 2000 × 22 physical inputs, columns in the order of `names('ketzin')` (units in the header) | model input |
| `ketzin_outputs.csv` | 2000 × 3: t_total in years, V_min and V_soft in m³/year (per m² of cross-section) | model output |
| `ketzin_gradients.csv` | 2000 × 44: columns 0–21 ∂log₁₀ t_total/∂x_j, columns 22–43 ∂log₁₀ V_soft/∂x_j, with respect to the **physical** inputs | model output |
| `quest_inputs.csv`, `quest_outputs.csv`, `quest_gradients.csv` | the same for Quest: 2000 × 30, 2000 × 3, 2000 × 60 | model input and output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits, so they round-trip exactly.

## Model

**Source.** S. K. T. Amrhein, T. Most, C. Walther, "An Uncertainty Quantification Approach for Long-Term Risk Assessment of Geological CO2 Storage Sites Using a One-Dimensional Leakage Model", Preprints.org, 2026, doi:10.20944/preprints202609.2139.v1 (version 1, posted 24 September 2026, **not peer reviewed**; CC BY 4.0). Institute of Structural Mechanics, Bauhaus-Universität Weimar. No published version exists yet; page, table and equation numbers below refer to the preprint and may change. The authors evaluated the model in MS Excel and did the sampling and sensitivity analysis in Ansys optiSLang 2025 R1 (Sec. 3.1, p. 11); a MATLAB listing for the median Ketzin inputs is in Table A1 (p. 19).

**Equations** (Sec. 2.2–2.3, pp. 5–8). Steady, single-phase, laminar 1-D Darcy flow of CO₂ up through n horizontal layers above the reservoir; layer 1 lies on the reservoir, layer n reaches the surface; h_i is the depth of the lower edge of layer i, h_{n+1} = 0; cross-section A = 1 m².

- L_i = h_i − h_{i+1} (Eq. 8)
- p_total,i = h_i · 0.01 MPa/m + p_pump · h_i / h_1 (Eqs. 11–12); Δp_i = p_total,i − p_total,i+1 (Eq. 13)
- η_i = (η_U,i − η_L,i) u_η + η_L,i (Eq. 14)
- V̇_i = K_i A Δp_i / (η_i L_i) (Eq. 3); t_leak,i = η_i L_i² / (K_i Δp_i) (Eq. 5)
- V̇_min = min_i V̇_i (Eq. 6); t_leak,total = Σ_i t_leak,i (Eq. 7)

Flows in m³/year and times in years, with a 365-day year (Table A1). K_i = 10^(log K_i) in m². Because both pressure terms are linear in depth, p_total,i = c h_i with c = 10⁴ Pa/m + 10⁶ p_pump/h_1, so Δp_i = c L_i and **the thickness cancels from the flow rate**: V̇_i = K_i c / η_i, t_leak,i = η_i L_i / (K_i c). The paper does not state this. The code computes the pressures as Table A1 does; the cancellation holds to 3 × 10⁻¹⁵ (notebook).

**Inputs** (independent; Sec. 3.1, p. 11 for Ketzin, Sec. 3.2, p. 14 for Quest; Table 2, p. 8 lists them):

| name | symbol | units | nominal (median) | distribution | source |
|---|---|---|---|---|---|
| `logK1`…`logKn` | log₁₀ K_i | log₁₀ m² | midpoint of the bounds below | symmetric triangular on the Table 3/4 permeability bounds | Table 3, p. 10; Table 4, p. 11; Sec. 2.3, p. 8 |
| `h1`…`hn` | h_i | m | Table 3/4 depth | uniform on [h_i − 5, h_i + 5] | Sec. 3.1, 3.2 |
| `p_pump` | p_pump | MPa | 12.5 (Ketzin), 8.34 (Quest) | symmetric triangular, median ± 4.5 | Sec. 3.1, 3.2 |
| `u_eta` | u_η | 1 | 0.5 | uniform on [0, 1], one number for all layers | Eq. (14), Sec. 3.1, 3.2 |

Ketzin (n = 10, m = 22; Table 3, p. 10). Reservoir at 702 m.

| layer | rock (Table 3) | h_i (m) | K_i bounds (m²) | η bounds (10⁻³ Pa s) |
|---|---|---|---|---|
| 1 | clay stone and clay marl stone | 629 | 10⁻¹⁸–10⁻¹⁶ | 0.014–0.070 |
| 2 | gypsum/anhydrite | 572 | 10⁻¹⁴–10⁻¹² | 0.014–0.070 |
| 3 | clay stone | 557 | 10⁻¹⁸–10⁻¹⁶ | 0.014–0.070 |
| 4 | clay marl stone, sandstone and clay stone | 465 | 10⁻¹⁵–10⁻¹³ | 0.014–0.070 |
| 5 | predominantly sandstone and siltstone | 381 | 10⁻¹³–10⁻¹¹ | 0.014–0.017 |
| 6 | clay stone, siltstone to sandstone | 310 | 10⁻¹⁴–10⁻¹² | 0.014–0.017 |
| 7 | sandstone, siltstone and clay stone | 264 | 10⁻¹⁴–10⁻¹² | 0.014–0.017 |
| 8 | clay | 151 | 10⁻¹⁷–10⁻¹⁵ | 0.014–0.017 |
| 9 | silt | 55 | 10⁻¹⁴–10⁻¹² | 0.014–0.017 |
| 10 | sand, gravel marl | 45 | 10⁻¹¹–10⁻⁹ | 0.014–0.017 |

These agree with the MATLAB listing (Table A1: log K = −17, −13, −17, −14, −12, −13, −13, −16, −13, −10; η bounds 1.4–7.0 × 10⁻⁵ Pa s for layers 1–4, 1.4–1.7 × 10⁻⁵ for 5–10).

Quest (n = 14, m = 30; Table 4, p. 11). Reservoir (Basal Cambrian Sand) at 2206 m.

| layer | rock (Table 4) | h_i (m) | K_i bounds (m²) | η bounds (10⁻³ Pa s) |
|---|---|---|---|---|
| 1 | Middle Cambrian Shale, primary seal | 2166 | 10⁻¹⁸–10⁻¹⁶ | 0.040–0.070 |
| 2 | Basal Red Beds | 2121 | 10⁻¹⁴–10⁻¹² | 0.040–0.070 |
| 3 | Lower Lotsberg Salt, secondary seal | 1987 | 10⁻¹⁹–10⁻¹⁷ | 0.040–0.070 |
| 4 | Devonian Mudstones | 1952 | 10⁻¹⁴–10⁻¹² | 0.040–0.070 |
| 5 | Upper Lotsberg Salt, ultimate seal | 1895 | 10⁻¹⁹–10⁻¹⁷ | 0.040–0.070 |
| 6 | Winnipegosis/Contact Rapids | 1810 | 10⁻¹⁴–10⁻¹² | 0.040–0.070 |
| 7 | Prairie Evaporite, additional seal | 1708 | 10⁻¹⁸–10⁻¹⁶ | 0.040–0.070 |
| 8–11 | not available | 1623, 1360, 1268, 1038 | 10⁻¹⁰–10⁻⁸ | 0.040–0.070 |
| 12 | not available | 983 | 10⁻¹⁰–10⁻⁸ | 0.025–0.070 |
| 13 | not available | 683 | 10⁻¹⁰–10⁻⁸ | 0.014–0.070 |
| 14 | not available | 138 | 10⁻¹⁰–10⁻⁸ | 0.014–0.017 |

Readings of the paper that it leaves implicit (our choices):

- "Triangular distribution with a median value of 12.5 MPa and a scatter of ± 4.5 MPa" is read as the symmetric triangular law on [8, 17] MPa (Quest: [3.84, 12.84] MPa), and "symmetric triangular distribution with the lower and upper bounds according to Table 3" as the symmetric triangular law on those bounds in log₁₀ K.
- "Uniform distribution with mean values according to Table 3 and a scatter range of ± 5.0 m" is read as uniform on [h_i − 5, h_i + 5].
- The "nominal" values (medians) are those of Table A1; they are a reference point for EMA, not a result of the paper. At the median Ketzin point layers 1 and 3 tie (same K, same η), so it lies on the kink of V̇_min.

**Quantities of interest.** Both paper outputs are heavy-tailed (Table 5; Fig. 7 fits log-normal laws), so the notebook uses base-10 logarithms.

1. log₁₀ t_leak,total (years): smooth (a log-sum-exp of the layer times).
2. log₁₀ V̇_soft (m³/year), the p-norm soft minimum V̇_soft = (Σ_i V̇_i^(−q))^(−1/q) with **q = 10 (our choice)**, in place of the paper's V̇_min, which has a kink wherever two layers tie. 0 ≤ log₁₀ V̇_min − log₁₀ V̇_soft ≤ log₁₀(n)/q (0.10 for Ketzin); on the shipped samples the difference is at most 0.0300 (Ketzin) and 0.0341 (Quest). `outputs.csv` also stores the exact V̇_min; `grad_log10_min_flow` gives its gradient where one layer attains the minimum, and the notebook shows its two-dimensional active subspace is 0.0007 from that of V̇_soft.

**Gradients.** Exact, with respect to the physical inputs. With t_i the layer times and T their sum: ∂T/∂log K_i = −ln 10 · t_i; ∂T/∂h_j = t_j/L_j − t_{j−1}/L_{j−1} (plus T · 10⁶ p_pump/(c h_1²) for j = 1, through c); ∂T/∂p_pump = −T (10⁶/h_1)/c; ∂T/∂u_η = Σ_i t_i (η_U,i − η_L,i)/η_i; then ∂log₁₀T = ∂T/(T ln 10). For log₁₀ V̇_soft, with softmax weights w_i = V̇_i^(−q)/Σ_j V̇_j^(−q): ∂/∂log K_i = w_i; ∂/∂p_pump = (10⁶/h_1)/(c ln 10); ∂/∂h_1 = −(10⁶ p_pump/h_1²)/(c ln 10); ∂/∂u_η = −Σ_i w_i (η_U,i − η_L,i)/(η_i ln 10); other depths 0.

**Sampling design.** Per site, M = 2000 independent draws by the inverse CDF of `numpy.random.default_rng(seed).random((2000, m))`, seed 2139 (Ketzin) and 2140 (Quest) (`co2_leakage_model.sample`). M ≥ 10 m and ≥ 300. The paper used 500 improved Latin hypercube samples per site with maximum input-correlation error about 0.02 (Sec. 3.1, Fig. 6), seed not given.

**Published results.** No active subspace. Table 5 (p. 14, Ketzin) and Table 6 (p. 16, Quest): minimum, maximum, mean, standard deviation, 5 % and 95 % quantiles of V̇_min, t_leak,total and selected layer outputs (e.g. Ketzin t_leak,total mean 1063.0 years, sd 962.0, 5 % 209.3, 95 % 2248.8; V̇_min mean 0.214 m³/year, 95 % 0.681). MOP "Coefficients of Prognosis" for the raw outputs (Fig. 7g–h, Ketzin: V̇_min log K₁ 53 %, log K₃ 52 %, u_η 30 %, p_pump 3 %; t_leak log K₃ 59 %, u_η 33 %, log K₁ 19 %, p_pump 3 %, log K₈ 1 %, h₃ 0 %. Fig. 9g–h, Quest: V̇_min log K₅ 60 %, log K₃ 57 %, u_η 3 %, p_pump 1 %; t_leak log K₅ 79 %, log K₃ 15 %, u_η 6 %, log K₇ 1 %, p_pump 1 %, h₄ 1 %), read from the bar labels; layer-wise MOP matrices (Figs. 8, 10).

**What the notebook finds** (seed 2139). Ketzin, log₁₀ t_leak,total: λ₁/λ₂ = 5.49, λ₂/λ₃ = 5.83; λ₁ + λ₂ hold 97.0 % of the trace; w₁ ≈ (log K₃ −0.748, log K₁ −0.483, u_η +0.431, p_pump −0.134), w₂ the contrast log K₁ − log K₃; cubic in (y₁, y₂) leave-one-out R² = 0.9859 (line in y₁: 0.9167). Ketzin, log₁₀ V̇_soft: λ₁/λ₂ = 1.44, λ₂/λ₃ = 21.31, a two-dimensional subspace whose second direction is the fold log K₃ − log K₁ of the minimum. Quest: log₁₀ t_leak,total λ₁/λ₂ = 6.17, λ₂/λ₃ = 7.07 (w₁ led by log K₅, then log K₃); log₁₀ V̇_soft 1.19, 54.07.

## Regenerating

```bash
python generate_data.py                           # writes the six CSVs here
python generate_data.py --out /tmp/co2 --check    # writes elsewhere and compares
```

The run takes 0.3 s (0.7 s with Python start-up) and also prints the gradient check. `--check` on 2026-10-30 (numpy 2.5.4, Python 3.13.18) reported the six files identical to the shipped ones. The notebook runs in about 9 s (13 s with kernel start-up under nbconvert).

## Accuracy

- **Values.** Closed form in double precision; the model reproduces the CSVs exactly (difference 0, printed in the notebook).
- **Gradients.** Exact. Checked against central finite differences with step 10⁻⁶ of each input's range, at all 2000 samples per site: largest relative error (row 2-norm) 8.2 × 10⁻¹⁰ (log₁₀ t_total) and 8.0 × 10⁻¹⁰ (log₁₀ V̇_soft) for Ketzin, 1.4 × 10⁻⁹ and 1.0 × 10⁻⁹ for Quest, the size expected from rounding in the differences.
- **Against the paper.** The sample statistics agree with Tables 5 and 6 within the sampling error of 500 samples (one exception, below), and total Sobol' indices of the raw outputs (Jansen's estimator, 20 000 base samples) agree with the MOP indices of Figs. 7 and 9 within 10 percentage points (largest difference: log K₃ for the Quest V̇_min, 57 % against 67.3 %).

## How these files relate to the paper

- **Not the paper's computation.** The paper's samples (improved LHS in optiSLang, Excel model) are not reproduced; its calculation files are said to be on https://refodat.de, but the record was not located and was not used. The samples here are new, from the published equations and distributions.
- **The model is the paper's.** Statistics and sensitivity indices agree (above). The notebook prints the comparison.
- **Table 5's 95 % quantile of the Ketzin total leakage time (2248.8 years) is inconsistent** with its own 95 % quantile of the layer-3 time (2283.0 years), since t_leak,total ≥ t_leak,3 at every sample. Ours is 2882.6 years; a log-normal law with Table 5's mean and standard deviation gives about 2800 years. We do not copy the table's value.
- **The minimum is smoothed.** The notebook's second QoI is V̇_soft, not the paper's V̇_min (see above); the exact V̇_min is stored too.
- **Preprint.** The paper is version 1 of a preprint; inputs, tables or the model may change in a later version.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). The CSVs were generated here from the published equations and distributions; no data or code were copied (the paper itself is CC BY 4.0). Model and analysis: S. K. T. Amrhein, T. Most, C. Walther (Bauhaus-Universität Weimar). The work was partially funded by the Free State of Thuringia under grant "2024 VFE 0077" with funds from the European Regional Development Fund. Cite:

- S. K. T. Amrhein, T. Most, C. Walther, "An Uncertainty Quantification Approach for Long-Term Risk Assessment of Geological CO2 Storage Sites Using a One-Dimensional Leakage Model", Preprints.org, 2026, doi:10.20944/preprints202609.2139.v1.
