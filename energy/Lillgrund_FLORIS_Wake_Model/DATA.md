# Lillgrund_FLORIS_Wake_Model: data

Kind: **Python model**. The model is `lillgrund_model.py`, a numpy port of the FLORIS 2.0.1 Gauss-Curl Hybrid wake model (zero yaw) on the 48-turbine Lillgrund layout, which reproduces FLORIS 2.0.1 itself to rounding error. `generate_data.py` draws the inputs with a saved seed and writes all three CSVs; a rerun reproduces them.

## Files

| file | contents | role |
|---|---|---|
| `Lillgrund_FLORIS_Wake_Model.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2021; outputs committed |
| `lillgrund_model.py` | `turbine_powers`, `farm_power`, `farm_power_and_grad`, `free_power`, `turbine_table`; `NAMES`, `UNITS`, `LOWER`, `UPPER`, `NOMINAL`, `DIRECTIONS`, `SPACINGS`, the layout and the held settings | the model (numpy only; derived from FLORIS 2.0.1, Apache-2.0) |
| `generate_data.py` | draws the samples and writes the three CSVs; `--rows`, `--check` | regenerates the data |
| `inputs.csv` | 500 × 6 physical wake parameters α, β, k_a, k_b, a_d (m), b_d, in the order of `NAMES` | model input |
| `outputs.csv` | 500 × 6: farm power in MW at 8 m/s for wind from 120°, 42°, 180°, 75°, 156°, 90° (spacings 3.3, 4.3, 4.8, 5.9, 7.1, 8.6 D) | model output |
| `gradients.csv` | 500 × 36: column 6 i + j is ∂P(direction i)/∂x_j in MW per unit of the **physical** input j | model output |

The CSVs have two `#` lines (a description and the column names with units) and are read by `np.loadtxt(path, delimiter=',')`. Inputs are written with 17 significant digits (exact round trip), outputs with 16, gradients with 13.

## Model

**Source.** M. T. van Beek, A. Viré, S. J. Andersen, "Sensitivity and Uncertainty of the FLORIS Model Applied on the Lillgrund Wind Farm", *Energies* 14(5):1293, 2021, doi:10.3390/en14051293 (publisher's open-access PDF, CC BY 4.0). FLORIS 2.0.1 (NREL 2020, Apache-2.0), Gauss-Curl Hybrid configuration (Sec. 2.2): `gauss_legacy` velocity deficit with near wake (Eqs. 1–2, 7–9; Bastankhah and Porté-Agel 2016), `gauss` deflection with natural deflection a_d + b_d x (Eqs. 3–6, 10), secondary steering and yaw-added recovery from the wake-rotation vortices (Sec. 2.2.3, Eq. 11; King et al. 2021), Crespo-Hernández added turbulence, sum-of-squares wake combination, FLORIS 2.0.1's turbine grid (5 × 5 points per rotor; the 9 inside the disk give the rotor-averaged speed, the cube root of the mean cube). The farm is the Lillgrund offshore farm: 48 Siemens SWT-2.3-93 turbines, D = 92.6 m, hub height 65 m, 2.3 MW (p. 2).

**Inputs** (m = 6), independent uniform on the bounds of Table 2 (p. 7), which the paper uses for its calibration and, as uniform distributions, for its Sobol analysis (Sec. 3.1, p. 9):

| name | symbol | units | lower | upper | FLORIS default (nominal) | role | source |
|---|---|---|---|---|---|---|---|
| `alpha` | α | – | 0.125 | 2.5 | 0.58 | near-wake length, turbulence weight (Eq. 1) | Table 2, p. 7 |
| `beta` | β | – | 0.015 | 0.3 | 0.077 | near-wake length, thrust weight (Eq. 1) | Table 2, p. 7 |
| `ka` | k_a | – | 0.05 | 1.5 | 0.38 | wake expansion k = k_a I + k_b (Eq. 9) | Table 2, p. 7 |
| `kb` | k_b | – | 0.0 | 0.02 | 0.004 | wake expansion, base rate (Eq. 9) | Table 2, p. 7 |
| `ad` | a_d | m | −1.0 | 1.0 | 0.0 | natural deflection offset (Eq. 10) | Table 2, p. 7 |
| `bd` | b_d | – | −0.1 | 0.1 | 0.0 | natural deflection per metre downstream (Eq. 10) | Table 2, p. 7 |

The defaults are FLORIS 2.0.1's (`default_parameters` of `gauss_legacy` and `gauss`), the paper's Ψ_org; the paper does not list them. The six values are set in both the velocity and the deflection model, which in FLORIS 2.0.1 keep separate copies of α, β, k_a, k_b.

**Units of a_d.** The paper's Eq. 10 writes δ_r = a_d D + b_d x. FLORIS 2.0.1 adds `ad + bd*(x - x_turbine)` (in `wake_deflection/gauss.py`), so a_d is in metres and over [−1, 1] it moves the wakes by at most 1 m. This is why the paper finds a_d negligible and drops it after the Sobol study (p. 10). The port keeps FLORIS's definition.

**Held settings** (not varied):

| setting | value | source |
|---|---|---|
| layout | 48 positions, UTM (m) shifted by 358805 E, 6152606 N | PyWake 2.6.20 Lillgrund example (MIT); the paper shows the layout in Fig. 1 and cites Jeppsson et al. 2008 |
| power and thrust curves | generic SWT-2.3-93, 3–25 m/s in 1 m/s steps (PyWake); C_p = P/(½ρπR²U³) with D = 92.6 m | PyWake 2.6.20. The paper's empirical SCADA curve is under NDA (p. 18) |
| D, hub height | 92.6 m, 65 m | paper p. 2 |
| wind speed, direction | 8 m/s; from 180° (main case) and from 120°, 42°, 75°, 156°, 90° | paper Fig. 6 case (p. 11); Table 1 (p. 3) |
| ambient TI, shear exponent, veer, air density | 0.06, 0.12, 0, 1.225 kg/m³ | FLORIS 2.0.1 `examples/example_input.json`; the paper uses a TI map from met-mast data (p. 3–4, not given) and does not state the shear |
| TSR, pP = pT | 8, 1.88 | FLORIS 2.0.1 example input |
| vortex core e, α_r | 0.2 D, 0.03 | paper p. 6 (FLORIS 2.0.1's default e is 0.3 D) |
| Crespo-Hernández (initial, constant, ai, downstream) | (0.5, 0.9, 0.75, −0.325) | FLORIS 2.0.1 example input and defaults |
| power-yaw loss, yaw | none; all yaw angles 0 | baseline case (paper Sec. 2.4) |

**Quantity of interest.** Total electrical power of the 48 turbines, in MW, at 8 m/s, for wind from 180° (the main case: the case of the paper's Fig. 6, turbine rows 4.8 D apart along the wind) and, as extra columns, one alignment direction of each other spacing in Table 1. The paper's Sobol output is the annual energy production over its met-mast wind rose (Eq. 14, p. 7; Sec. 3.1), reported per spacing in Figs. 4–5; the wind rose is confidential, so we use single flow cases instead.

**Smoothness.** Two choices in FLORIS 2.0.1 are discrete. (1) Secondary steering gives each turbine an effective yaw angle, chosen as the integer in [−30°, 30°] that best matches the mean circulation seen at its rotor (`calculate_effective_yaw_angle`). Even at zero yaw, the wake-rotation vortices of the turbines upstream give waked turbines effective yaws of several degrees (0–30° over the samples at 180°), which deflect their wakes. (2) The added turbulence from an upstream turbine is weighted by an "area overlap" from the count of rotor points where that turbine's deficit is at most 0.05 m/s. So the farm power is piecewise smooth, with jumps where either choice changes. Along each input's range through the FLORIS defaults (101 points, 180°), the jumps, measured as the difference between the power and the previous piece continued, are at most 0.047 MW each and add up to at most 0.64 MW (b_d), against ranges of 3–14 MW for the five inputs other than a_d (a_d: no jumps, range 0.15 MW). FLORIS also drops rotor points where the field is NaN from the rotor average (a NaN arises where a wake's near-wake formula is undefined); the port does the same, and this happens in 1 of the 3000 sample-direction pairs (sample 448 at 120°, three points).

**Gradients.** Central differences in each physical input with step 10⁻⁴ of its half-range, with the discrete choices (effective yaw angles and overlap counts) frozen at the sample, so the stencil never straddles a jump: the gradient of the smooth piece the sample lies on (`farm_power_and_grad`, 13 farm solves per sample and direction).

**Sampling design.** M = 500 independent draws, `numpy.random.default_rng(2021).uniform(LOWER, UPPER, size=(500, 6))`. M ≥ 10 m and ≥ 300.

**Published results.** No active subspace. Sobol first- and total-order indices from 12,000 evaluations with 100 bootstrap resamples (p. 9–10): at 4.8 D, k_a, b_d and, to a lesser degree, α carry about 79 % of the variance of AEP; a_d is negligible (and set to 0 afterwards); total-order indices are well above first-order ones, so the parameters interact (Fig. 4). Across spacings (Fig. 5), the sensitivity to k_a grows with spacing and that to α and b_d falls. Only these statements are in the text; the bar heights of Figs. 4–5 are not given as numbers.

## Regenerating

```bash
python generate_data.py                                  # 500 samples, 2 processes, writes here
python generate_data.py --rows 0:6 --out /tmp/lg --check     # a subset, compared with the files here
```

The inputs come from the seed and the outputs from the model, so a rerun reproduces the files; `--workers` changes only the run time. The shipped files were written on 2026-10-30 in a Linux container (Python 3.13.18, numpy 2.5.4). In the repo's `.venv` (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, macOS; 2026-10-31) a full rerun with `--check` took 987 s (16.5 min) on two processes, with other jobs running: 78 farm solves per sample (6 directions × 13), about 45 ms of CPU each. It reported the inputs identical, outputs to a relative 1.0 × 10⁻¹⁵ and gradients to 1.0 × 10⁻¹⁰ (2-norm of each row). The notebook runs in about 10 s under nbconvert.

## Accuracy

- **Port against FLORIS 2.0.1.** FLORIS 2.0.1 was installed from PyPI in a separate Python 3.11 venv (numpy 1.26.4, scipy 1.11.4; steps in the EMA notes, `Lillgrund_FLORIS_Wake_Model.ema/MODEL_NOTES.md`) and run with the same layout, turbine table, settings and parameters through its own input dictionary. Over 29 flow cases (the defaults at 180° and 8 and 11 m/s; TI 0.10 with no shear; TI 0.08 with shear 0.2 at 270°; three random parameter sets at each of the six directions and at 213° and 270°; sample 448 at 120°, the one with NaN rotor points), the largest difference in any turbine's power is 6.5 × 10⁻¹⁶ of the largest turbine power, and the turbines' final turbulence intensities agree exactly. A FLORIS solve takes about 0.4–0.5 s, a port solve 0.03–0.06 s.
- **Gradients.** Against the same frozen differences with steps 10⁻³, 10⁻⁵ and 10⁻⁶ of the half-range (12 samples at 180°, 120° and 90°): relative differences (2-norm) of 10⁻⁸–8 × 10⁻⁶ at 10⁻³ (truncation, falling as h²), and 3 × 10⁻¹⁰–8 × 10⁻⁸ at 10⁻⁵ and 10⁻⁶. Against plain central differences (nothing frozen) with step 10⁻⁶: 3 × 10⁻¹⁰–8 × 10⁻⁸ at the same samples. Plain central differences with the shipped step 10⁻⁴ differ from the frozen ones by more than 10⁻⁴ at none of 100 random samples at 180°, so jumps are rarely inside a stencil; freezing guards against the rare case. The notebook's check cell re-solves three samples (relative difference from `outputs.csv` printed there) and repeats the plain-difference check at two.

## How these files relate to the paper

- **Not the paper's computation.** The paper's runs used FLORIS 2.0.1 with an empirical SCADA power curve, its met-mast TI map and wind rose, and AEP as the output; those data are under a non-disclosure agreement (p. 18). The data here use the same model code and parameter ranges, with a public layout, a generic power curve, FLORIS's example settings, and single flow cases. So the numbers are not the paper's, and the sensitivity comparison is qualitative.
- **Model, not data, from FLORIS.** No FLORIS output is shipped; FLORIS 2.0.1 was run only to check the port.
- **Unknowns.** The paper does not give its ambient TI and shear for the Fig. 6 case, whether it set the four shared parameters in both FLORIS sub-models, its Crespo-Hernández constants (we assume FLORIS's example values), or the numeric Sobol indices.

## License and credit

Code: MIT (`LICENSE`), except `lillgrund_model.py`, which is derived from FLORIS 2.0.1 (Copyright 2020 NREL, Apache License 2.0) and carries that notice. Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`), generated here; no data were copied from the paper. The layout and the turbine curves are from PyWake 2.6.20 (DTU Wind Energy, MIT). Cite:

- M. T. van Beek, A. Viré, S. J. Andersen, *Energies* 14(5):1293, 2021, doi:10.3390/en14051293 (study and parameter ranges).
- NREL, FLORIS, version 2.0.1, 2020, https://github.com/NREL/floris (model).
- J. King et al., "Control-oriented model for secondary effects of wake steering", *Wind Energy Science* 6(3):701–714, 2021, doi:10.5194/wes-6-701-2021 (GCH).
- M. Bastankhah, F. Porté-Agel, *J. Fluid Mech.* 806:506–541, 2016, doi:10.1017/jfm.2016.595 (Gaussian wake and deflection).
