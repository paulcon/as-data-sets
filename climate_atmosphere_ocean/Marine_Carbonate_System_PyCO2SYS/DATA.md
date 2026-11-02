# Marine_Carbonate_System_PyCO2SYS: data

Kind: **stored data**. The CSVs were written by PyCO2SYS v1.8.3 (Humphreys et al. 2022; GPLv3), an outside code the notebook does not need. `carbonate_driver.py` (our wrapper, MIT) and `generate_data.py` call PyCO2SYS, which must be installed separately; no PyCO2SYS code is copied here. A rerun with the same PyCO2SYS, numpy and Python versions reproduces the files bit for bit (checked; see Regenerating).

## Files

| file | contents | role |
|---|---|---|
| `Marine_Carbonate_System_PyCO2SYS.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2022; outputs committed |
| `carbonate_driver.py` | `evaluate(X, outputs, full, held)`, `gradients(X, outputs, steps, held)`; `NAMES`, `UNITS`, `OUTPUTS`, `HELD`, `HELD_OCEANWIDE`, `NOMINAL`, `SD_MEASUREMENT`, `SD_OCEANWIDE`, `STEPS` | wrapper that runs PyCO2SYS (needs PyCO2SYS 1.8.3). The `_model.py` name follows the repo's file naming; it does not make this a Python-model data set (it needs PyCO2SYS, not only numpy/scipy) |
| `generate_data.py` | draws both designs, runs the wrapper, writes the seven CSVs; `--check`, `--checks-only`, `--reference-only` | regenerates the data (needs PyCO2SYS 1.8.3) |
| `inputs_measurement.csv` | 1000 × 14 physical inputs, columns in the order of `NAMES` (units in the header) | model input, measurement study |
| `outputs_measurement.csv` | 1000 × 4: pH (total scale), pCO₂ (µatm), Ω_aragonite, Ω_calcite | model output |
| `gradients_measurement.csv` | 1000 × 56: column 14 q + j is ∂(output q)/∂(input j), physical units | model output |
| `inputs_oceanwide.csv`, `outputs_oceanwide.csv`, `gradients_oceanwide.csv` | the same for the ocean-wide (surface-ocean, 0 dbar) study | model input and output |
| `nominal_reference.csv` | 5 × 9, for pH, pCO₂ (µatm), Ω_ar, Ω_ca, fCO₂ (µatm), xCO₂ (µmol/mol), [CO₃²⁻], [HCO₃⁻], [CO₂(aq)] (µmol/kg): PyCO2SYS at the nominal point at 1234 dbar (row 0) and its own linear propagation (paper Eq. 24) of the measurement-study sds there (row 1); the nominal point at 0 dbar (row 2) and Eq. 24 of the ocean-wide sds there (row 3); row 0 plus the paper's round-robin script's two hypothetical acid–base systems (row 4, reproduces Table 4) | reference values for the notebook's comparison |

The CSVs have two `#` lines (a description with the PyCO2SYS version and seed, and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits.

## Model

**Source.** M. P. Humphreys, E. R. Lewis, J. D. Sharp, D. Pierrot, "PyCO2SYS v1.8: marine carbonate system calculations in Python", *Geosci. Model Dev.* 15(1):15–43, 2022, doi:10.5194/gmd-15-15-2022 (publisher PDF, CC BY 4.0). Code: https://github.com/mvdh7/PyCO2SYS, GPLv3; v1.8.0 archived at doi:10.5281/zenodo.5602840 (paper, Code availability). We used v1.8.3 from PyPI, the version the online documentation describes (v2 is in beta).

**Equations.** From a measured pair, here total alkalinity A_T and dissolved inorganic carbon T_C (`par1_type=1`, `par2_type=2`), PyCO2SYS solves the alkalinity equation for [H⁺] by Newton–Raphson with a tolerance of 10⁻⁸ in pH (paper Sect. 3.1 and Table 4 caption), with the carbonate, borate, water, phosphate, silicate, ammonia, sulfide, sulfate and fluoride equilibria (Appendix B). Then fCO₂ = [CO₂(aq)]/K₀*, pCO₂ = fCO₂/G (fugacity factor), and Ω = [Ca²⁺][CO₃²⁻]/K_sp* (Appendix D). The stoichiometric constants K* are parameterisations in temperature, salinity and pressure (Tables 1–3). User-given constants and totals are used "as-is", with no pH-scale or pressure corrections (p. 19).

**Held conditions and options.** The centre of both studies is the paper's round-robin test point, Table 4 (p. 25): A_T 2300, T_C 2100 µmol/kg, salinity 33, temperature 22 °C, total silicate 10, total phosphate 1 µmol/kg, with total ammonia 2 and total sulfide 3 µmol/kg held. Results are at the input conditions (no `temperature_out`/`pressure_out`).

- **Pressure.** The measurement study holds the table's 1234 dbar. That is a synthetic test point (22 °C water does not occur at 1234 dbar in the ocean), acceptable for an uncertainty study at one water sample. The ocean-wide study spreads temperature over 6.0–36.8 °C, which only the upper ocean reaches, so it is a **surface-ocean study at 0 dbar** (`carbonate_driver.HELD_OCEANWIDE`). Pressure matters: at the nominal point Ω_ar is 1.9409 at 1234 dbar and 2.3391 at 0 dbar (+20.5 %), pH 7.8759 and 7.9187, pCO₂ 565.35 and 573.66 µatm (+1.5 %) (`nominal_reference.csv` rows 0 and 2).
- **Options.** The paper says Table 4 used "default parameterisation options", i.e. the defaults of PyCO2SYS v1.8.0, the paper's version: `opt_k_carbonic=16` (Sulpis et al. 2020), `opt_total_borate=1` (Uppström 1974), `opt_k_fluoride=1` (Dickson and Riley 1979), `opt_k_bisulfate=1` (Dickson 1990), `opt_gas_constant=3` (2018 CODATA); `opt_pressured_kCO2` (added after v1.8.0) is 0, the v1.8.0 behaviour; total pH scale. We read these from the v1.8.0 source (git tag `v1.8.0` of https://github.com/mvdh7/PyCO2SYS, the release archived at doi:10.5281/zenodo.5602840; `PyCO2SYS/engine/nd.py`). v1.8.3 changed some defaults (its defaults give pH 7.875 and pCO₂ 566.2 µatm at Table 4's conditions), so `carbonate_driver.HELD` sets them explicitly.
- **Table 4 reproduced.** The paper's script for Table 4 (`manuscript/round_robin.py` in that release) adds two hypothetical acid–base systems that the Table 4 caption does not list: `total_alpha=5`, `k_alpha=1e-4`, `total_beta=6`, `k_beta=1e-8` (µmol/kg, mol/kg). With them, the v1.8.0 defaults in PyCO2SYS v1.8.3 give pH 7.8706, pCO₂ 572.6166, fCO₂ 570.7219, xCO₂ 587.6667, [CO₃²⁻] 143.7652, [HCO₃⁻] 1938.5305, [CO₂(aq)] 17.7043, which round to all seven printed values (`nominal_reference.csv` row 4; the script prints pH to three decimals and the rest to one, by rounding). These two systems are not seawater constituents, so the studies leave them out; without them the nominal point has pH 7.8759 and pCO₂ 565.35 µatm.

**Inputs** (m = 14), independent Gaussian, x = mean + sd · ξ:

| name | symbol | units | mean | sd, measurement study | sd, ocean-wide study | source |
|---|---|---|---|---|---|---|
| `alkalinity` | A_T | µmol/kg | 2300 | 2 | 50 | mean: Table 4, p. 25; sd: our choice |
| `dic` | T_C | µmol/kg | 2100 | 2 | 50 | mean: Table 4; sd: our choice |
| `temperature` | T | °C | 22 | 0.05 | 5 | mean: Table 4; sd 0.05: PyCO2SYS docs example (`uncertainty_from={"par1": 2.0, "temperature": 0.05}`), 5 × seacarb's `errors()` default `eT = 0.01`; 5: our choice |
| `salinity` | S | – | 33 | 0.01 | 1 | mean: Table 4; sd 0.01: default `eS` of seacarb's `errors()` (seacarb 3.4.1); 1: our choice |
| `total_silicate` | Si_T | µmol/kg | 10 | 0.4 (4 %) | 2 | mean: Table 4; sd: our choice |
| `total_phosphate` | P_T | µmol/kg | 1 | 0.04 (4 %) | 0.2 | mean: Table 4; sd: our choice |
| `dpk_CO2` | δpK₀* | pK | 0 | 0.002 | 0.002 | Orr et al. (2018) via `pyco2.uncertainty_OEDG18` (v1.8.3 `uncertainty.py`); also seacarb's `errors()` default `epK` |
| `dpk_carbonic_1` | δpK₁* | pK | 0 | 0.0075 | 0.0075 | same; applied to Sulpis et al. (2020), which Orr et al. (2018) predate: our assumption |
| `dpk_carbonic_2` | δpK₂* | pK | 0 | 0.015 | 0.015 | same, same assumption |
| `dpk_borate` | δpK_B* | pK | 0 | 0.01 | 0.01 | same |
| `dpk_water` | δpK_w* | pK | 0 | 0.01 | 0.01 | same |
| `dpk_aragonite` | δpK_sp,ar* | pK | 0 | 0.02 | 0.02 | same |
| `dpk_calcite` | δpK_sp,ca* | pK | 0 | 0.02 | 0.02 | same |
| `f_total_borate` | f_B | – | 1 | 0.02 | 0.02 | same (`total_borate__f`: 2 % fractional; seacarb `eBt = 0.02`); multiplies Uppström (1974) borate |

- **Constants as offsets.** The paper does not tabulate input uncertainties (Sect. 3.6 describes the method only). An uncertain constant enters as K* = K*_param(T, S, P) · 10^(−δpK), with K*_param PyCO2SYS's own parameterisation at the sample's T, S and P, so δpK is the error of pK* and carries the Orr et al. standard uncertainty; total borate is B_T = f_B · B_T,param(S). Every other constant and total follows its parameterisation. The wrapper calls PyCO2SYS twice per batch: once to get K*_param and B_T,param, once with the override arguments (`k_CO2`, `k_carbonic_1`, …, `total_borate`). With all offsets 0 and f_B = 1 the second call reproduces the first exactly (relative difference 0).
- **Measurement uncertainties are our choice.** Neither the paper nor the PyCO2SYS documentation gives them, apart from the documentation's example of 2 µmol/kg and 0.05 °C. seacarb's `errors()` (seacarb 3.4.1, the Orr/Epitalon uncertainty tool) has defaults `eS = 0.01` (our salinity sd) and `eT = 0.01` °C (our 0.05 °C is 5 times that), and no default for the measured pair (`evar1 = evar2 = 0`). We read Orr et al. (2018) only through PyCO2SYS's and seacarb's defaults; their recommended measurement uncertainties were not checked.
- **Constant uncertainties and Sulpis et al. (2020).** The Orr et al. (2018) pK₁* and pK₂* uncertainties (0.0075, 0.015) were assessed before the Sulpis et al. (2020) K₁*, K₂* existed, so they were not assessed for that parameterisation. Applying them to it is our assumption (also an open question for the authors).
- **Scaling.** Every input is scaled by its own standard deviation (the active_subspaces Gaussian normalization (x − mean)/sd), measured inputs and constants alike. The pK offsets are already on a log scale.
- **Ocean-wide study (surface ocean).** Same centre and constant uncertainties, measured inputs spread widely and independently, at 0 dbar. It is a wide spread of surface-water states around the paper's test point, a parameter study labelled as such, not a sample of the real ocean: in the ocean A_T and T_C are correlated, which this design ignores.
- **Samples outside physical meaning or rare in the surface ocean** (counts printed by `generate_data.py` and the notebook). None in the measurement study. In the ocean-wide study: 54 samples have T > 30 °C, rare in open-ocean surface water, and 2 of them T > 35 °C (max 36.8 °C); 15 have A_T − T_C < 50 µmol/kg (minimum about 10), which give pCO₂ from 1262.8 to 2271.7 µatm; in all, 111 samples have pCO₂ > 1000 µatm and 18 have Ω_ar < 1, values rare at the sea surface. None has T < −1.9 °C, a negative salinity or nutrient, f_B ≤ 0, or A_T < T_C. All are kept: PyCO2SYS evaluates its parameterisations there without error. The calibration ranges of the parameterisations are not given in the paper, so we do not count samples outside them. **Pressure:** the ocean-wide study was first generated at the Table 4 pressure of 1234 dbar, where these temperatures do not occur; it is now at 0 dbar (see Held conditions). The pressure effect at the nominal point is +20.5 % in Ω_ar, +0.043 in pH and +1.5 % in pCO₂ from 1234 to 0 dbar.

**Quantities of interest.** pH on the total scale, pCO₂ (µatm), aragonite and calcite saturation states, all at the input conditions. The notebook's main QoI is pCO₂.

**Gradients.** Central differences in the physical inputs, through the wrapper, with steps h_j = 10⁻³ × (measurement-study sd of input j): 2·10⁻³ µmol/kg for A_T and T_C, 5·10⁻⁵ °C, 10⁻⁵ in salinity, 4·10⁻⁴ and 4·10⁻⁵ µmol/kg for the nutrients, 2·10⁻⁶ to 2·10⁻⁵ pK for the offsets, 2·10⁻⁵ for f_B. The same steps are used in both studies. A temperature or salinity step also moves every parameterised constant and total, so ∂/∂T and ∂/∂S are total derivatives, as in PyCO2SYS's own `grads_wrt`. PyCO2SYS has automatic differentiation only inside its solver and buffer factors; it propagates uncertainty with forward differences (paper Eq. 23, steps in Appendix H), so we use our own central differences and check them against those (Accuracy).

**Sampling design.** Two independent Monte Carlo designs, M = 1000 each: ξ = `numpy.random.default_rng(seed).standard_normal((1000, 14))` with seed 20220104 (measurement) and 20220105 (ocean-wide). M ≥ 10 m = 140 and ≥ 300.

**Published results.** The paper has no sampling study and no active subspace. It reports the Table 4 round-robin values (pH_T 7.871, pCO₂ 572.6 µatm, fCO₂ 570.7 µatm, xCO₂ 587.7 µmol/mol, [CO₃²⁻] 143.8, [HCO₃⁻] 1938.5, [CO₂(aq)] 17.7 µmol/kg; Table 4 prints the xCO₂ unit as µatm, but xCO₂ is a dry-air mole fraction, so we use µmol/mol), that its linear propagation agrees with Monte Carlo (n = 10⁴) to within 3 % (Sect. 4.1.3), and that it reproduces the derivatives and propagated uncertainties of Orr et al. (2018) Tables 2–4 to within 10⁻³ % and 10⁻⁴ % (Sect. 4.2.3).

## Regenerating

```bash
python3.13 -m venv pyco2-venv && pyco2-venv/bin/pip install PyCO2SYS==1.8.3 pandas   # pandas: an undeclared import of PyCO2SYS 1.8.3
pyco2-venv/bin/python generate_data.py                       # writes the seven CSVs here, then runs the checks
pyco2-venv/bin/python generate_data.py --out /tmp/c --check  # writes elsewhere and compares with these files
```

Shipped files written 2026-11-01 (after the review fixes) with PyCO2SYS 1.8.3, autograd 1.9.1, numpy 2.5.4, Python 3.13.18 (Linux container, 2 CPUs shared with other jobs). Writing both studies took 62 s (30.0 s and 31.8 s; each study is 57 vectorized evaluations of 1000 samples, two PyCO2SYS calls each); with the checks that follow (step changes and PyCO2SYS's own derivatives) the run took 305 s. A `--check` rerun (same versions, written to a scratch directory: 26.0 s and 29.4 s, 228 s in all) reported all seven files identical bit for bit. The input files are identical in value to the first version of this data set (same seeds and distributions); the outputs and gradients changed with the options and the ocean-wide pressure. Other numpy or PyCO2SYS versions may change the last digits.

## Accuracy

From `generate_data.py`'s checks (run 2026-10-31; the log is printed by the script):

- **Step size.** Gradients with steps 10× larger and 10× smaller agree with the shipped ones to a relative 2.4·10⁻⁸ (pH), 1.2·10⁻⁸ (pCO₂) and 3.2·10⁻⁸ (Ω) in the measurement study, and 1.9·10⁻⁸, 9.0·10⁻⁹ and 3.3·10⁻⁸ in the ocean-wide study (worst row 2-norm over 20 samples per study and per output).
- **Against PyCO2SYS's built-in derivatives** (`grads_of`/`grads_wrt`: forward differences with the paper's Appendix H steps, an independent code path) at 20 samples per study, at each study's pressure, with the constants at their parameterised values (offsets 0, f_B = 1, where the two definitions coincide): largest relative difference of the gradient (row 2-norm) 3.0·10⁻⁵ for pH, 1.0·10⁻⁴ for pCO₂ and 1.1·10⁻⁴ for Ω (measurement), and 3.4·10⁻⁵ to 1.1·10⁻⁴ (ocean-wide). The differences sit in the pK components and match the forward-difference truncation error expected for a step of 10⁻⁴ in pK, ln(10) · 10⁻⁴ / 2 ≈ 1.2·10⁻⁴ relative; for A_T, T_C and the nutrients they are at most 3.1·10⁻⁷ (sd-scaled, relative to the gradient norm), for salinity 1.3·10⁻⁷ and 1.1·10⁻⁶, for temperature 1.3·10⁻⁶ and 1.1·10⁻⁵ (measurement, ocean-wide). So the central differences are the more accurate of the two.
- **Values.** The pH solver's tolerance is 10⁻⁸ in pH, but Newton's method converges quadratically and the step checks above show no solver noise at the 10⁻⁸ level in the gradients.
- **Linear propagation.** `nominal_reference.csv` row 1 is PyCO2SYS's own Eq. 24 at the nominal point: σ(pCO₂) = 18.08 µatm, σ(pH) = 0.01192, σ(Ω_ar) = 0.0963. The notebook's √trace C is 18.10 µatm, 0.01192 and 0.0966.

## How these files relate to the paper

- **Not the paper's computation.** The paper defines no parameter study. The two designs, the measurement uncertainties, the offset formulation of the constants' uncertainty, the ocean-wide spread and its surface pressure are ours. The centre is the paper's Table 4, and the constants' standard uncertainties are the Orr et al. (2018) values PyCO2SYS ships (applied to Sulpis et al. 2020 by assumption).
- **Options from the paper's code.** The options are PyCO2SYS v1.8.0's defaults, read from its source, and with the paper's round-robin script's two hypothetical acid–base systems they reproduce all seven Table 4 values (above). Every option is therefore identified, `opt_k_bisulfate` and `opt_gas_constant` included; none was fitted.
- **The notebook's comparison.** Over measurement uncertainty, the Monte Carlo standard deviations of the 1000 samples agree with PyCO2SYS's Eq. 24 within the 95 % sampling interval for pH and pCO₂ (ratios 0.971, 1.002); for Ω the ratios are 1.044 and 1.049, at the edge of the interval, and the same samples through the linearized model give ratios 0.9987–1.0001, so the excess is sampling, not nonlinearity.

## License and credit

Code (`carbonate_driver.py`, `generate_data.py`, the notebook): MIT (`LICENSE`); they import PyCO2SYS (GPLv3), which is not distributed here. Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`); they are outputs of PyCO2SYS runs, not copies of any published data. The paper is CC BY 4.0. Cite:

- M. P. Humphreys, E. R. Lewis, J. D. Sharp, D. Pierrot, "PyCO2SYS v1.8: marine carbonate system calculations in Python", *Geosci. Model Dev.* 15(1):15–43, 2022, doi:10.5194/gmd-15-15-2022 (model and code).
- M. P. Humphreys et al., PyCO2SYS v1.8.0, Zenodo, 2021, doi:10.5281/zenodo.5602840 (code).
- J. C. Orr, J.-M. Epitalon, A. G. Dickson, J.-P. Gattuso, "Routine uncertainty propagation for the marine carbon dioxide system", *Mar. Chem.* 207:84–107, 2018, doi:10.1016/j.marchem.2018.10.006 (standard uncertainties of the constants; not read, used through PyCO2SYS).
- O. Sulpis, S. K. Lauvset, M. Hagens, *Ocean Sci.* 16(4):847–862, 2020, doi:10.5194/os-16-847-2020; L. R. Uppström, *Deep Sea Res. Oceanogr. Abstr.* 21(2):161–162, 1974, doi:10.1016/0011-7471(74)90074-6; A. G. Dickson and J. P. Riley, *Mar. Chem.* 7(2):101–109, 1979, doi:10.1016/0304-4203(79)90002-1 (cited by the paper and PyCO2SYS for K_HF*); A. G. Dickson, *J. Chem. Thermodyn.* 22(2):113–127, 1990, doi:10.1016/0021-9614(90)90074-Z (parameterisations used).
- J. Orr, J.-P. Gattuso, J.-M. Epitalon, H. Lavigne, seacarb: Seawater Carbonate Chemistry, R package version 3.4.1, 2026, https://CRAN.R-project.org/package=seacarb (`errors()` default uncertainties).
