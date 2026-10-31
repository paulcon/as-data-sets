# Atrial_Cell_Models_Courtemanche_Maleckar: data

Kind: **stored data**. The values come from MATLAB runs of the Courtemanche (1998) and Maleckar (2009) human atrial myocyte models by Coveney and Clayton for "Sensitivity and Uncertainty Analysis of Two Human Atrial Cardiac Cell Models Using Gaussian Process Emulators" (Front. Physiol. 11:364, 2020). The authors published the runs on figshare as eight text files. `convert_data.py` copied them into CSV files without changing a digit. Nothing was rerun, refitted or corrected. The original text files are not shipped; their DOI, MD5 and SHA-256 are below, and `convert_data.py --check` rebuilds the CSVs from them.

## Files

| file | contents | role |
|---|---|---|
| `Atrial_Cell_Models_Courtemanche_Maleckar.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2020; outputs committed |
| `convert_data.py` | writes the eight CSVs from the figshare files and checks them bit for bit; holds the input names, units, Table 1 values, the design bounds (`inputs`, `bounds`), the output names and units, and the paper's Sobol' input variances | lossless conversion and metadata |
| `courtemanche_stage1_inputs.csv` | 295 × 20: Stage 1 inputs of the Courtemanche model, physical units (`CourtemancheStage1Inputs.txt`) | stored inputs |
| `courtemanche_stage1_outputs.csv` | 295 × 11: the 11 biomarkers of the same runs (`CourtemancheStage1Outputs.txt`) | stored outputs |
| `courtemanche_stage2_inputs.csv` | 200 × 10: Stage 2 inputs (9 model inputs + DI) | stored inputs |
| `courtemanche_stage2_outputs.csv` | 200 × 11 | stored outputs |
| `maleckar_stage1_inputs.csv` | 286 × 22: Stage 1 inputs of the Maleckar model | stored inputs |
| `maleckar_stage1_outputs.csv` | 286 × 11 | stored outputs |
| `maleckar_stage2_inputs.csv` | 195 × 8: Stage 2 inputs (7 model inputs + DI) | stored inputs |
| `maleckar_stage2_outputs.csv` | 195 × 11 | stored outputs |

The CSVs are comma-separated numbers after two `#` lines (a description with the credit, and the column names with units), so `np.loadtxt(path, delimiter=',')` reads them. Row *i* of an outputs file belongs to row *i* of the inputs file of the same model and stage. Every number is the source's fixed-point text with six decimals.

Output columns (all files; Sec. 2.2, p. 4, Fig. 1): `dVdt_max` maximum upstroke slope; `Vmax` peak voltage; `V20`, `V40`, `V60`, `V80` membrane voltage at 20, 40, 60 and 80 % of APD90; `APD50`, `APD90` action potential duration at 50 and 90 % repolarization; `RestVm` resting potential (mean over 10 ms, 100 ms before the upstroke); `Ca_min`, `Ca_max` minimum and maximum intracellular Ca²⁺. **Units are not stated** in the paper or the files. From the values they are V/s (= mV/ms) for `dVdt_max` (55–344), mV for the voltages, ms for the durations (APD90 75–509), and mM for calcium (`Ca_min` 7e-6 to 2.4e-4, `Ca_max` 1.1e-4 to 3.1e-3, i.e. 7 nM to 3 µM).

**Source.** R. Clayton and S. Coveney, "Design data for sensitivity analysis of human atrial myocyte models", The University of Sheffield, Dataset, figshare article 11309756, version 1, published 2019-12-03, doi:10.15131/shef.data.11309756.v1, licence CC BY 4.0 (figshare API record read 2026-10-31). The files were downloaded by P. Constantine's browser (figshare is not reachable from the build container) and their MD5s match the figshare listing:

| file | bytes | MD5 (figshare) | SHA-256 |
|---|---|---|---|
| `CourtemancheStage1Inputs.txt` | 56208 | `23fdfad590e19d3d22472901a6959be9` | `5e6cf19e79058dfb39cc85eba29235c8ebff6d5e06cca71ac65a47a76c570817` |
| `CourtemancheStage1Outputs.txt` | 35990 | `33cd8f4207961b479f3d1e22e4e3f4e1` | `56650670cd14ac37fcd78c3cf1beafddb1af0dbf6bbff62a33b0c715d1fe032e` |
| `CourtemancheStage2Inputs.txt` | 19150 | `565aafb2516bc1ad45f5b8ae7ced61d7` | `4d5bd8170fb87cbd890b88b2b77fc657c2dcfccdfeb52929633622d4abceef83` |
| `CourtemancheStage2Outputs.txt` | 24400 | `7117ccbb1cb186f0bb9cbac22571aa76` | `955246cf525ad5bf3b48a3346f4b8de49230b60ae9b03c7290bce1744ddf6dae` |
| `MaleckarStage1Inputs.txt` | 60446 | `7b2a906afb0315db9c9ad73d3fd58f79` | `2430c8bd0600915cfecb6809be3642b2c7094c33c48d299bacd4e11f7e4a6c16` |
| `MaleckarStage1Outputs.txt` | 34892 | `10bd9cf818fc40190e6b468518547779` | `55e063b45ed11fb23459f7144f0119f30c4dc3328e294b3b613a9be437252cc1` |
| `MaleckarStage2Inputs.txt` | 15057 | `3654270661c44a78cd9330864fab72f0` | `2ef7808c9c71cfa565089fe15d0574f997448a884a4b99695f0dbf08f3e64d42` |
| `MaleckarStage2Outputs.txt` | 23790 | `2fd078545853b05095876de5d7ccdb7f` | `43c61e14217bfe18f3a77c691bf822210ad91e46fe6bb5e30a9cb394df9b9421` |

The source files are space-separated, no header, LF line ends, six decimals. The figshare description says "Input files contain samples of inputs, normalised to vary between 0 and 1, with default values at 0.5"; **the files actually hold physical values** (e.g. Courtemanche G_Na 3.906–11.695 nS/pF), not normalized ones.

## Model

Content of the paper card. Page, table and figure numbers refer to the published article: S. Coveney and R. H. Clayton, "Sensitivity and Uncertainty Analysis of Two Human Atrial Cardiac Cell Models Using Gaussian Process Emulators", Front. Physiol. 11:364 (2020), doi:10.3389/fphys.2020.00364 (CC BY 4.0; preprint bioRxiv doi:10.1101/818047).

- **Simulators.** The Courtemanche, Ramirez and Nattel (1998) model (Am. J. Physiol. Heart Circ. Physiol. 275(1):H301–H321, doi:10.1152/ajpheart.1998.275.1.H301) and the Maleckar, Greenstein, Giles and Trayanova (2009) model (Am. J. Physiol. Heart Circ. Physiol. 297(4):H1398–H1410, doi:10.1152/ajpheart.00411.2009), an extension of Nygren et al. (1998) with modified I_Kur, I_to and Na⁺ movement (Sec. 2.1, p. 3). MATLAB code generated from the CellML repository (http://cellml.org), solved with `ode15s`, relative and absolute tolerance 1e-6, maximum step 0.5 ms (Sec. 2.3, p. 4).
- **Modifications** (Sec. 2.3, p. 4). Courtemanche: [Na⁺]_i and [K⁺]_i fixed at their initial values 11.17 and 139.00 mM (to remove drift). Maleckar: I_K,ACh fixed at zero.
- **Stimulus** (Sec. 2.3, p. 4). Courtemanche: 2000 pA for 2 ms. Maleckar: "750/C_m pA/pF for 6 ms" as printed.
- **Protocol.** Stage 1: 40 beats at a cycle length of 1000 ms, "with a check to ensure that APD90 had reached steady state"; biomarkers from the last beat. Stage 2: 39 S1 beats at 1000 ms, then an S2 stimulus at APD90 of the last S1 beat + 10 ms + DI, DI on 50–450 ms (Latin hypercube with the other Stage 2 inputs); biomarkers from the S2 beat (Secs. 2.3, 3.2, pp. 4 and 8).
- **Inputs** (m = 20 Courtemanche, m = 22 Maleckar; Table 1, p. 3). Each input was varied uniformly (Latin hypercube) over ±50 % of its model default, except G_K1 and C_m (±25 %) and [Na⁺]_o, [K⁺]_o, [Ca²⁺]_o (±10 %) (Sec. 2.5, p. 4–5). Table 1 has misprints. The column "design range used" is what the notebook uses (`convert_data.inputs`): the stated percentage about Table 1's central value where the data agree, otherwise the central value (or, for Maleckar G_bCa, the bounds) that put the Stage 1 points one per stratum of a 300-point Latin hypercube. Every stored row lies inside these ranges.

  Courtemanche (all "Table 1, p. 3"):

  | name | units | Table 1 central | Table 1 printed range | design range used | note |
  |---|---|---|---|---|---|
  | `GNa` | nS/pF | 7.8 | 5.85–11.70 (±50%) | 3.9–11.7 | printed lower end is 0.75 × centre |
  | `GK1` | nS/pF | 0.09 | 0.045–0.018 (±25%) | 0.0675–0.1125 | printed range misprinted |
  | `Gto` | nS/pF | 0.165 | 0.0826–0.0330 (±50%) | 0.0826–0.2478 (centre 0.1652) | data reach 0.24764 > 1.5 × 0.165 |
  | `fGKur` | 1 | 1.0 | 0.50–1.50 | 0.5–1.5 | multiplier of G_Kur |
  | `GKr` | nS/pF | 0.0294 | 0.0147–0.0441 | 0.014706–0.044118 (centre 0.029412) | centre refined from the strata |
  | `GKs` | nS/pF | 0.1294 | 0.0647–0.1941 | 0.064705–0.194115 (centre 0.12941) | centre refined from the strata |
  | `GCaL` | nS/pF | 0.1237 | 0.0619–0.1856 | 0.061875–0.185625 (centre 0.12375) | data reach 0.185619 > 1.5 × 0.1237 |
  | `GbNa` | nS/pF | 0.0006 | 0.0003–0.0010 | 0.00033722–0.0010117 (centre 0.00067444) | data 0.000338–0.001011 |
  | `GbCa` | nS/pF | 0.0011 | 0.0005–0.0017 | 0.0005655–0.0016965 (centre 0.001131) | data reach 0.001695 > 1.5 × 0.0011 |
  | `INaKmax` | pA/pF | 0.5993 | 0.2997–0.8990 | 0.29967–0.89899 (centre 0.59933) | centre refined from the strata |
  | `INaCamax` | pA/pF | 1600.0 | 800.00–2400.0 | 800–2400 | |
  | `IpCamax` | pA/pF | 0.275 | 0.1375–0.4125 | 0.1375–0.4125 | |
  | `Krel` | 1/ms | 0.30 | 0.15–0.45 | **15–45** (centre 30) | data 15.04–44.91, 100 × the printed values |
  | `tau_tr` | ms | 180.0 | 90.0–270.0 | 90–270 | |
  | `Iupmax` | mM/ms | 0.005 | 0.0025–0.0075 | 0.0025–0.0075 | |
  | `Kup` | mM | 0.00092 | 0.00046–0.0014 | 0.00046–0.00138 | |
  | `Cm` | pF | 100.0 | 75.0–125.0 (±25%) | 75–125 | |
  | `Nao` | mM | 140.0 | 126.0–154.0 (±10%) | 126–154 | |
  | `Ko` | mM | 5.4 | 4.86–5.94 (±10%) | 4.86–5.94 | |
  | `Cao` | mM | 1.8 | 1.62–1.98 (±10%) | 1.62–1.98 | |

  Maleckar (all "Table 1, p. 3"):

  | name | units | Table 1 central | Table 1 printed range | design range used | note |
  |---|---|---|---|---|---|
  | `PNa` | nL/s | 0.0018 | 0.0009–0.0027 | 0.0009–0.0027 | |
  | `GK1` | nS | 3.1 | 2.325–3.875 (±25%) | 2.325–3.875 | |
  | `Gt` | nS | 8.25 | 4.125–12.375 | 4.125–12.375 | |
  | `GKur` | nS | 2.25 | 1.125–3.375 | 1.125–3.375 | |
  | `GKr` | nS | 0.5 | 0.250–0.750 | 0.25–0.75 | |
  | `GKs` | nS | 1.0 | 0.50–1.50 | 0.5–1.5 | |
  | `GCaL` | nS | 6.75 | 3.375–10.125 | 3.375–10.125 | |
  | `GbNa` | nS | 0.0605 | 0.303–0.909 | 0.0303–0.0909 (centre 0.0606) | printed range 10 × too large |
  | `GbCa` | nS | 0.0590 | 0.0295–0.0885 | **0.0078656–0.086548** | data 0.00802–0.08621 (Stage 1), 0.00788–0.08540 (Stage 2); see below |
  | `INaKmax` | pA | 68.55 | 34.27–102.82 | 34.275–102.825 | |
  | `KNaCa` | pA/mM⁴ | 0.0750 | 0.0375–0.1125 | **0.018742–0.056226** (centre 0.037484) | data 0.01877–0.05619: centre is half the printed value |
  | `IpCamax` | pA | 4.0 | 2.0–6.0 | 2–6 | |
  | `alpha_rel` | pA/mM | 200,000 | 100,000–300,000 | 1e5–3e5 | |
  | `tau_tr` | s | 0.01 | 0.005–0.015 | 0.005–0.015 | |
  | `Iupmax` | pA | "28,00" | 1,400–4,300 | 1400–4200 (centre 2800) | data 1402.0–4196.5 |
  | `Kcyca` | mM | 0.0003 | 0.00015–0.00045 | 0.00015–0.00045 | |
  | `Ksrca` | mM | 0.5 | 0.25–0.75 | 0.25–0.75 | |
  | `Kxcs` | 1 | 0.4 | 0.2–0.6 | 0.2–0.6 | |
  | `Cm` | pF | 50 | 37.5–62.5 (±25%) | 37.5–62.5 | |
  | `Nao`, `Ko`, `Cao` | mM | 140.0, 5.4, 1.8 | ±10% | 126–154, 4.86–5.94, 1.62–1.98 | |

  Stage 2 inputs (Sec. 3.2, p. 8; Figs. 7–8): Courtemanche `GK1, Gto, fGKur, GKr, GCaL, GbCa, INaKmax, INaCamax, IpCamax, DI`; Maleckar `GK1, Gt, GKur, GCaL, GbCa, INaKmax, Cm, DI`; DI in ms on 50–450; all other inputs at their Table 1 central values. Same ranges as Stage 1.

  **How the ranges were inferred.** A Latin hypercube of N points on [a, b] has exactly one point in each interval [a + k(b − a)/N, a + (k+1)(b − a)/N). For each column we searched (a, b) near the data's extremes for the bounds that maximize the number of distinct strata (N = 300 for Stage 1, 200 for Stage 2). With the ranges above, 11 of 20 Courtemanche and 17 of 22 Maleckar Stage 1 columns, and 7 of 10 and 7 of 8 Stage 2 columns, have every point in its own stratum; the others have small values that the six printed decimals cannot place in a stratum (e.g. Maleckar K_cyca, stratum width 1e-6 mM) or bounds that the stored points leave loose (all within the stated percentage). The refined centres come from the strata. For Maleckar, two of them agree with a CellML encoding of the model (see "CellML check" below): K_NaCa 0.037484 (CellML 0.0374842 pA/mM⁴) and G_bNa 0.0606 (CellML 0.060599 nS). The refined Courtemanche centres (G_to 0.1652, G_Kr 0.029412, G_Ks 0.12941, G_CaL 0.12375, G_bNa 0.00067444, I_NaK,max 0.59933, K_rel 30 ms⁻¹) look like the Courtemanche (1998) defaults that Table 1 rounds; this is to be checked against a Courtemanche CellML file, which we did not have.

  **Maleckar G_bCa.** The data span 0.0080–0.0862 nS in Stage 1 (0.0079–0.0854 in Stage 2): not ±50 % of 0.0590. All 286 Stage 1 points fall in distinct strata of [0.0078656, 0.086548], a range of 0.0787 that runs from 0.1 × to 1.1 × 0.0787 nS and is centred at 0.0472 = 0.6 × 0.0787. Table 1's central value 0.0590 is 0.75 × 0.0787. **Lead:** the Maleckar CellML encoding we checked has g_B,Ca = 0.078681 nS, the Nygren et al. (1998) value, and 0.1 × and 1.1 × 0.078681 are 0.0078681 and 0.086549 nS, which agree with the inferred bounds (0.0078656 and 0.086548) to a small fraction of one 1/300 stratum. So the authors most likely sampled 0.1–1.1 × the model default (−90 %/+10 %), and Table 1's 0.0590 is wrong. The centre of the sampled range (0.0472) is therefore not a model default. We use the inferred bounds and list the question for the authors. The paper's Sobol' indices are on normalized units, so for this input they refer to the range actually sampled.

  **CellML check** (2026-10-31, one fetch): `heart/src/odes/cellml/Maleckar2008.cellml` from a shallow clone of github.com/Chaste/Chaste (commit 357af77f3d8b47aa179dbebbdde04c89ce3f2aca, SHA-256 of the file e80c74055f246856d5be5c41ada196fde7a5ce029b4b999d4e5ed9aeaab7caf0). Its `xml:base` names the Physiome Model Repository workspace `maleckar_greenstein_trayanova_giles_2009`, revision c446ef4d188574654356f9edbfb8f10711209357. The paper does not say which revision it used, so this is a lead, not proof. Defaults read: g_B_Ca 0.078681 nS, g_B_Na 0.060599 nS, K_NaCa 0.0374842 pA/mM⁴, P_Na 0.0018 nL/s, g_Ca_L 6.75, g_t 8.25, g_kur 2.25, g_K1 3.1, g_Ks 1, g_Kr 0.5 nS, i_NaK_max 68.55 pA, i_CaP_max 4 pA, I_up_max 2800 pA, alpha_rel 200000 pA/mM, k_cyca 0.0003 mM, k_srca 0.5 mM, k_xcs 0.4, tau_tr 0.01 s, Cm 0.05 nF; all equal the design centres except g_B_Ca. Nothing was copied from the file (Chaste is BSD-licensed).
- **Quantities of interest.** The 11 biomarkers above, from the last (Stage 1) or S2 (Stage 2) beat. The notebook's headline QoI is APD90 (ms), the one the paper analyzes most (Figs. 6, 9, 11, 12), and summarizes the other ten. No transform.
- **Gradients.** None in the paper or the data. Stage 1 (295 and 286 runs, fewer than the 462 and 552 that a full quadratic in 20 and 22 inputs needs at twice its 231 and 276 coefficients): a global linear model (OLS), whose direction is a one-variable summary (n = 1 by construction) that cannot determine the subspace dimension, plus a partial quadratic (quadratic in the Stage 2 inputs, linear in the rest; 66 and 51 coefficients) as a diagnostic of what the summary misses. Local linear gradients are shown only to document that their eigenvalue ratio is set by the neighbour count (λ1/λ2 for Courtemanche APD90 from 9.0 at p = 25 to 241.0 at p = 200; 42.8 at the default p for a synthetic function whose exact ratio is 7.86); they are not used as evidence. Stage 2: the global quadratic estimator (QPHD; 200 ≥ 132 and 195 ≥ 90), with the dimension chosen as the smallest n ≤ 3 whose LOO R² for a cubic in n active variables is within 0.03 of the best (not the library's linear eigenvalue-gap rule, which returns n = 1 almost by construction).
- **Sampling design.** Stage 1: "optimized Latin hypercube", 300 runs per model (criterion and seed not given), plus 150 validation runs (not published). Stage 2: Latin hypercube of 200 runs over the reduced inputs, plus 100 validation runs (not published) (Sec. 2.5, pp. 4–5). The strata test above confirms Latin hypercubes of 300 and 200 points.
- **Removed runs** (Sec. 2.5, p. 5): pacemaking, resting potential > −60 mV, APD90 > 600 ms, or APD90 of beats 39 and 40 differing by more than 5 % (alternans). Stage 1: 5 Courtemanche runs (long APD or alternans) and 14 Maleckar runs (pacemaking or failure to repolarize); Stage 2: 0 and 5. The files hold only the kept runs (295, 286, 200, 195 rows); the removed runs' inputs are not published. No kept run violates the criteria; no input or calcium value is non-positive.
- **Cost per run.** Not reported.
- **Published results** (no active subspace). Signed first-order Sobol' indices from GP emulators (Fig. 3, Stage 1; Fig. 7, Stage 2), total indices (Figs. 4, 8), PLS regression indices (Fig. 5), main effects (Figs. 6, 9), APD90 restitution surfaces (Fig. 11), output coefficient of variation versus input standard deviation (Fig. 12), and 6 (Courtemanche) and 5 (Maleckar) output principal components for 95 % of the variance (Sec. 4.4.3, p. 19). The Sobol' indices treat the inputs as **Gaussian** in normalized units u ∈ [0, 1] with mean 0.5 and variance 0.02 (±50 % inputs), 0.04 (G_K1, C_m) or 0.1 (extracellular concentrations) (Sec. 2.7, p. 6), not as the uniform design distribution. The notebook reads the printed cells of Figs. 3, 4, 7 and 8 (rasterized at 200 dpi); Figs. 3 and 7 omit first-order indices below 0.05, Figs. 4 and 8 total indices below 0.01.

## Regenerating

```bash
python convert_data.py --src /path/to/figshare_11309756            # writes the eight CSVs (0.15 s)
python convert_data.py --src /path/to/figshare_11309756 --check    # rebuilds them in memory and compares
```

The script refuses a source file whose MD5 or SHA-256 differs from the table above, checks the column counts, writes each source token unchanged with commas between, and checks that the CSV read with `np.loadtxt(..., delimiter=',')` equals the source read with `np.loadtxt` bit for bit (as uint64). The shipped files were written on 2026-10-31 with Python 3.13.18 and numpy 2.5.4, and `--check` reports all eight identical. There is no randomness in the conversion. The notebook's bootstrap uses `SEED = 2020`; the notebook runs in 25.6 s (its last cell; about 29 s wall-clock with `nbconvert`). Rerunning the cell models is a different matter: see the EMA notes kept outside the repository (`Atrial_Cell_Models_Courtemanche_Maleckar.ema/MODEL_NOTES.md`).

## Accuracy

- **Conversion:** exact (the same decimal text; bit-for-bit check).
- **Precision of the stored values:** six decimals, fixed point. This limits small quantities: Ca_min has only 72–115 distinct values per file (1–3 significant digits; the Maleckar Stage 1 minimum is 0.000007 mM), Ca_max 173–269; inputs such as Maleckar K_cyca (0.00015–0.00045 mM) or Courtemanche G_bNa have 3–4 significant digits.
- **Solver accuracy:** `ode15s` with tolerances 1e-6 and a maximum step of 0.5 ms (paper). No convergence study is reported. There are no gradients to check.
- **Emulator accuracy (paper, p. 7):** for most Stage 1 and Stage 2 emulators the mean average predicted error was below 10 % and the median individual standard error below 1.0 on 150 and 100 validation runs; the table is in the paper's Supplementary Material (not read).

## How these files relate to the paper

- **The same runs.** These are the paper's design runs as the authors published them, not a regeneration. The row counts equal the paper's runs minus its removals. The output principal components (6 and 5 for 95 % of the standardized output variance) reproduce Sec. 4.4.3.
- **Notebook results (seed 2020).** Stage 2, all 11 biomarkers, QPHD with the dimension rule above: n = 1 in 7 of the 22 biomarker × model cases (Courtemanche RestVm; Maleckar V20, V80, APD90, RestVm, Ca_min, Ca_max), n = 2 in 13, n = 3 in 2 (Courtemanche V20, APD50); with a tolerance of 0.02 the counts are 4/12/6, with 0.05 they are 12/9/1. A second active variable raises the cubic LOO R² by more than 0.05 in 9 of 22 cases (e.g. Courtemanche V80 0.567 → 0.822). APD90, Stage 2: Courtemanche λ1/λ2 = 7.5 [5.0, 8.8], LOO R² 0.820 / 0.915 / 0.937 for n = 1 / 2 / 3, chosen n = 2; Maleckar λ1/λ2 = 31.2 [21.0, 44.4], LOO R² 0.959 / 0.977 / 0.982, chosen n = 1. APD90, Stage 1 (one-variable summary only; the dimension is not determined): OLS direction led by I_NaK,max, G_bCa, G_CaL, G_Kr, I_pCa,max, G_K1 (Courtemanche) and G_bCa, G_K1, G_Kur, G_CaL, C_m (Maleckar); LOO R² of a cubic in it 0.834 and 0.920; partial quadratic 0.879 and 0.867.
- **Sobol' comparison.** With the paper's Gaussian input variances, the Sobol' indices of a linear (Stage 1) or quadratic (Stage 2) fit (not the active-subspace weights) reproduce Figs. 3 and 7 closely (top input per biomarker as in Fig. 3 for 8/11 Courtemanche and 10/11 Maleckar biomarkers; Stage 2 APD90 indices within 0.06 of Fig. 7 except Maleckar C_m, +0.249 against +0.14). The uniform active-subspace weights differ from the paper's indices mainly where the paper's distribution weights an input more (G_K1, C_m, extracellular concentrations).
- **Discrepancies between paper and data** (not corrected; documented): Table 1 ranges (above); the figshare description ("normalised") versus physical values in the files; the Courtemanche Stage 2 input list (text: 8 inputs + DI; files and Figs. 7A/8A: 9 + DI, with G_Kr).

## Open questions for the authors

1. Maleckar G_bCa: the sampled range is 0.0079–0.0865 nS, not Table 1's 0.0295–0.0885 (±50 % of 0.0590). The CellML encoding we checked has g_B,Ca = 0.078681 nS (the Nygren et al. value), and the sampled range is 0.1–1.1 × that default. Was G_bCa sampled −90 %/+10 % about its default (and is Table 1's 0.0590 a misprint), and was that intended?
2. Courtemanche K_rel: data 15–45, Table 1 0.15–0.45 ms⁻¹; Maleckar K_NaCa: data centre 0.0375, Table 1 0.0750. Are these misprints in Table 1?
3. Stage 2 Courtemanche inputs: the text lists eight; the file and Figs. 7A/8A include G_Kr. Was G_Kr included?
4. Sobol' input variances: what variance did DI have in Stage 2, and was C_m given 0.04 in the Maleckar Stage 2 analysis (a quadratic fit gives S = 0.25 with 0.04, against 0.14 in Fig. 7B)?
5. Maleckar Stage 1 total indices (Fig. 4B): every input has ST ≥ 0.06, including inputs with no visible effect in the runs; is this an emulator artefact?
6. Units of the stored outputs (V/s, mV, ms, mM inferred) and the stimulus of the Maleckar model ("750/C_m pA/pF").
7. Which space-filling criterion did the "optimized" Latin hypercube use (it affects only the stratum test above, not the analysis)?
8. The inputs of the removed runs (5 + 14 in Stage 1, 5 in Stage 2) and the 150 + 100 validation runs: could they be published, so that the failure regions can be studied?

## License and credit

The data are © R. Clayton and S. Coveney, published under **CC BY 4.0** (figshare record 11309756). The eight CSV files are a format conversion of their files, with unchanged values, redistributed under the same licence with credit: R. Clayton and S. Coveney (2019), "Design data for sensitivity analysis of human atrial myocyte models", The University of Sheffield, Dataset, doi:10.15131/shef.data.11309756.v1; runs for S. Coveney and R. H. Clayton, Front. Physiol. 11:364 (2020), doi:10.3389/fphys.2020.00364. The credit is also in the first line of every CSV. Code here (`convert_data.py`, the notebook): MIT (`LICENSE`). No code of the authors (MATLAB models, the maGPy emulator) is copied.

Cite: S. Coveney and R. H. Clayton, "Sensitivity and Uncertainty Analysis of Two Human Atrial Cardiac Cell Models Using Gaussian Process Emulators", Front. Physiol. 11:364, 2020, doi:10.3389/fphys.2020.00364; and the dataset above.
