# SIPNET_Ecosystem_Carbon_Flux: data

> **Substitute set-up (author decision, P. Constantine, 2026-10-31: acceptable when documented).** These data are **not** the paper's runs. They use the Niwot Ridge meteorological driver that ships with SIPNET instead of the paper's Bartlett Experimental Forest driver, and PEcAn's template values (`template.param_v2`) for every SIPNET parameter the paper did not calibrate instead of the paper's Bartlett trait meta-analysis values. The eight calibrated parameters, their priors (Table A2) and the four initial pools (Table A1) are the paper's. In this set-up 902 of the 1000 samples are net carbon sources over the two years. Details under "Configuration" and "How these files relate to the paper".

Kind: **stored data from an outside code** (decision 1 of 2026-10-30). The model is SIPNET, a C program; it was compiled and run in the build container and the outputs are shipped here with their provenance. `generate_data.py` reproduces the files when given a compiled SIPNET binary. The notebook needs only the CSV files and the driver.

## Files

| file | contents | role |
|---|---|---|
| `SIPNET_Ecosystem_Carbon_Flux.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 514`; outputs committed |
| `generate_data.py` | runs SIPNET at the Latin hypercube samples and writes the three CSVs; `NAMES`, `LOWER`, `UPPER`, `NOMINAL`, `SEED`, `H` | provenance and regeneration (needs a SIPNET binary) |
| `inputs.csv` | 1000 × 8 physical parameters, columns in the order of `NAMES` | model input |
| `outputs.csv` | 1000 × 3: `cum_nee` [g C m⁻²], `total_et` [mm], `total_gpp` [g C m⁻²], totals over 2004–2005 | model output |
| `gradients.csv` | 1000 × 24: column 8 i + j is ∂(output i)/∂(input j) in **physical** units | model output |
| `fd_check_soilWHC.csv` | 100 × 8: for the samples with normalized `soilWHC` < −0.8 (`soilWHC` < 3.69 cm): row index, `soilWHC` [cm], and ∂(output)/∂`soilWHC` of the 3 outputs [per cm] with steps 3e-5 and 1e-5 | accuracy check (`generate_data.py --fd-check`), used in the notebook |
| `sipnet_template.param` | SIPNET parameter file: PEcAn's `template.param_v2` with the paper's initial pools | model input file |
| `niwot_2004_2005.clim` | SIPNET driver, 1462 half-day steps, 2004-01-01 to 2005-12-31 (old 14-column format) | model input file |
| `LICENSE-SIPNET-PEcAn.txt` | BSD 3-clause licences of SIPNET and PEcAn, and what was taken from each | licence |

The CSVs are plain comma-separated numbers after two `#` lines (a description and the column names with units), read by `np.loadtxt(path, delimiter=',')`.

## Model

**Paper.** I. Fer, R. Kelly, P. R. Moorcroft, A. D. Richardson, E. M. Cowdery, M. C. Dietze, "Linking big models to big data: efficient ecosystem model calibration through Bayesian model emulation", *Biogeosciences* 15(19):5801–5830, 2018, doi:10.5194/bg-15-5801-2018 (publisher PDF, CC BY 4.0). The paper calibrates SIPNET (and ED2) in PEcAn v1.4.10 against NEE and LE from the Bartlett Experimental Forest eddy-covariance tower and soil respiration, 2005–2006, half-hourly, with a Gaussian-process emulator of likelihood sufficient statistics (Sec. 2.1–2.2). SIPNET took "∼ 5.5 s per execution" (Sec. 2.3), stated in App. A2 as ∼ 5.5 s "per MCMC iteration in PEcAn including model execution and writing and reading model outputs". Initial pools: Table A1. Emulator design: three rounds, cumulatively 243, 486 and 729 knots (Table 1 caption); the first 243 from a Latin hypercube, each later 243 drawn 90 % from the previous emulator posterior and 10 % from the prior (Sec. 2.1).

**Code.** SIPNET v2.2.0-11, github.com/PecanProject/sipnet, commit `c3479bd681ad092e6be33fcf14272e23e0b3fcbf` (2026-10-28), BSD 3-clause. Built with `make sipnet` (gcc 13.3.0, Ubuntu 24.04, flags `-Wall -Werror -g`, no optimization); the build passes SIPNET's Niwot smoke test byte for byte. One run of the two-year driver takes about 0.02 s (0.05 s including file I/O and process start).

**Configuration** (author's choice where the paper's set-up is not available; listed under unknowns in the report):

- Parameter file: PEcAn `models/sipnet/inst/template.param_v2` (github.com/PecanProject/pecan, commit `22ab2838`, BSD 3-clause), the template PEcAn writes SIPNET runs from, with the initial pools of Table A1 (p. 5818): wood 9600 g C m⁻², LAI 0, litter 200 g C m⁻², soil 1600 g C m⁻². It is a deciduous set-up (leaf-on by growing degree days, `fracLeafFall` = 1). The paper's other parameter values came from PEcAn's trait meta-analysis for the Bartlett PFT, which is in the BETYdb database, not in a repository; template values are used instead. So only the eight priors and the four initial pools are the paper's; every other parameter is PEcAn's generic deciduous template, not a Bartlett value.
- Driver: the meteorological file of SIPNET's Niwot smoke test (`tests/smoke/niwot/sipnet.clim`, 1998-11 to 2005-12, two steps per day), lines for 2004 and 2005, unchanged. SIPNET's smoke-test README calls it "a simple run at Niwot Ridge(?)"; it matches the twice-daily Niwot Ridge driver of Sacks et al. (2006). The paper's Bartlett driver (tower data via AmeriFlux, processed by PEcAn; App. A1, p. 5817) is not packaged as a SIPNET driver in either repository, and AmeriFlux is not reachable from the build container. Totals over 2004–2005: precipitation 1494 mm (747 mm per year, against about 1300 mm per year at Bartlett, App. A1), mean air temperature 2.25 °C.
- Run options: SIPNET defaults (GDD phenology, snow, moisture-dependent heterotrophic respiration; no litter pool, nitrogen or anaerobic modules); events on with an empty `events.in`. (With `--no-events`, this SIPNET version segfaults when it logs a computed leaf-on event, because `events.out` is never opened.)

- Cold-season soil respiration: none. SIPNET v2 has a single soil-respiration law all year. The paper's 2018 PEcAn set-up tied SIPNET's cold-season soil respiration to the two calibrated soil parameters: PEcAn `write.configs.SIPNET.R` (commit `22ab2838`, lines 576–585) sets `soilRespQ10Cold = soilRespQ10` and `baseSoilRespCold = 0.25 × baseSoilResp`, with the comment "these were the two assumptions used in the emulator paper in order to reduce dimensionality" and the note that they "don't affect anything when the seasonal soil respiration functionality in SIPNET is turned-off". The paper does not say whether that switch was on. SIPNET v2.0 removed both parameters (same file, 2025 comment), so this build cannot reproduce the coupling.

So the data describe a **hybrid test case**: the paper's model family, priors and initial pools under a subalpine climate, with generic template values elsewhere. They are not a reproduction of the paper's runs. **Regime:** 902 of the 1000 samples (90.2 %) are net carbon sources over 2004–2005 (median cumulative NEE +674.75 g C m⁻²), whereas the paper's site is a moist northern hardwood forest; the subspaces describe this regime and should not be transferred to Bartlett.

**Inputs** (m = 8): the eight SIPNET parameters the paper calibrates (Sec. 2.3, Fig. 2a), independent uniform on the priors of Table A2 (p. 5819); definitions and units from Table A3. PEcAn passes each trait to the SIPNET parameter of the second column unchanged (`write.configs.SIPNET.R`).

| SIPNET name | PEcAn trait (paper) | units (SIPNET) | prior | template value | source |
|---|---|---|---|---|---|
| `baseSoilResp` | som_respiration_rate | yr⁻¹ | U(0.001, 0.3) | 0.06 | Table A2, p. 5819 |
| `soilRespQ10` | soil_respiration_Q10 | – | U(1.4, 3.0) | 2.9 | Table A2 |
| `soilWHC` | soilWHC | cm | U(0.1, 36.0) | 12 | Table A2 |
| `halfSatPar` | half_saturation_PAR | Einsteins m⁻² day⁻¹ | U(4.0, 27.0) | 17 | Table A2 |
| `dVpdSlope` | dVPDSlope | kPa⁻¹ | U(0.01, 0.25) | 0.05 | Table A2 |
| `leafGrowth` | leafGrowth | g C m⁻² | U(0.0, 252.0) | 126 | Table A2 |
| `psnTOpt` | psnTOpt | °C | U(5.0, 40.0) | 24 | Table A2 |
| `leafTurnoverRate` | leaf_turnover_rate | yr⁻¹ | U(0.03, 6.0) | 0.13 | Table A2 |

Table A3 gives the SOM respiration rate in day⁻¹; we treat this as **an error in the paper**. Evidence: PEcAn `write.configs.SIPNET.R` (commit `22ab2838`, line 547) passes `som_respiration_rate` unchanged into `baseSoilResp` with the comment "units = 1/year"; SIPNET divides `baseSoilResp` by 365 (`src/sipnet/sipnet.c`, line 1865, "change from per-year to per-day rate"); and the paper's brute-force posterior N(0.1, 0.009) (Table A2) would mean a soil-carbon turnover time of 10 days if read per day. The prior is applied in yr⁻¹ here. The paper's ninth calibrated quantity, the multiplicative soil-respiration bias k (standard lognormal prior, Sec. 2.2), enters only the likelihood and is not a model input. Template values are in the last column but are not used as a centre: the study samples the whole prior box.

**QoIs**: totals over 2004-01-01 to 2005-12-31: cumulative NEE (`sipnet.NEE_cum` at the last step, printed to 6 decimals; positive = carbon source), evapotranspiration (sum of the `evapotranspiration` column of `sipnet.out`, 8 decimals, converted from cm to mm), and GPP (`sipnet.GPP_cum`, 6 decimals). Total ET is approximately a linear function of the mean latent heat flux LE, the paper's second data stream: SIPNET's ET column also counts snow sublimation (SIPNET `docs/user-guide/model-outputs.md`, column 24), which has a different latent heat. The paper's QoIs were likelihood statistics against observations and are not reproduced; soil respiration totals were not kept because SIPNET prints them to 3 decimals per step, too coarse for finite differences.

**Gradients**: central differences in the normalized inputs x ∈ [−1, 1]⁸ with step h = 1e-3, divided by (upper − lower)/2 to give physical gradients; within h of a bound, the second-order one-sided formula (0, ±h, ±2h), so no run leaves the prior box. 17 runs per sample (more at the few samples near a bound).

**Sampling design**: a Latin hypercube of 1000 points (`scipy.stats.qmc.LatinHypercube(d=8, seed=514)`, scipy 1.18.1), mapped linearly to the prior box.

**Physical-meaning checks** (counts printed in the notebook): no input outside its prior; 144 samples have total ET above the 1494 mm of precipitation, none above precipitation plus the initial soil water and snow; 26 samples have two-year GPP below 10 g C m⁻² (the canopy never develops), and 98 samples are net carbon sinks, so 902 are net carbon sources. All are kept.

**Published results to reproduce**: none of this kind. The paper reports no active subspace, gradients or Sobol' indices; its Fig. 2a shows PEcAn partial variances of NEE, LE and soil respiration (bars, read to about ±0.03: NEE psnTOpt 0.47, SOM respiration 0.30, Q10 0.12, half-saturation PAR 0.05; LE soilWHC 0.24, leaf growth 0.20, psnTOpt 0.14, leaf turnover 0.13, VPD slope 0.11, half-saturation PAR 0.10). The notebook compares their ranking with activity scores.

## Regenerating

```bash
git clone https://github.com/PecanProject/sipnet && cd sipnet && git checkout c3479bd681ad092e6be33fcf14272e23e0b3fcbf && make sipnet
cd <this directory>
OPENBLAS_NUM_THREADS=1 python generate_data.py --sipnet /path/to/sipnet/sipnet                  # all 1000 samples
python generate_data.py --sipnet /path/to/sipnet/sipnet --rows 0:20 --out /tmp/sipnet --check   # subset, compared with these files
python generate_data.py --sipnet /path/to/sipnet/sipnet --fd-check                                 # fd_check_soilWHC.csv (add --out DIR --check to compare)
```

There is no randomness beyond the saved Latin hypercube seed. The shipped files were written on 2026-10-31 (Python 3.13.18, numpy 2.5.4, scipy 1.18.1) on 2 worker processes in about 4 minutes (about 17 000 SIPNET runs). A 20-sample `--check` run took 8.2 s and matched the shipped files to 2.4e-10 relative (the 10 significant digits written). `fd_check_soilWHC.csv` was added on 2026-10-31 after the review (same binary and versions, 2 workers, 8.8 s, about 500 SIPNET runs); a `--fd-check --check` rerun matched it to 5.0e-11 relative, and a 20-sample `--check` rerun with the edited script matched the three main files to 2.4e-10.

## Accuracy

- **Smoothness.** SIPNET is piecewise smooth: `max(0, ·)` clipping of the temperature and VPD factors, the soil-water bucket, snow and the leaf-on trigger make kinks. One-dimensional scans of 401 points along each input at one sample showed no jumps; the steepest regions are small `soilWHC` and small `psnTOpt`.
- **Step study** at the first 6 samples, h = 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4 (relative 2-norm difference of the normalized 8-vector from the h = 1e-3 gradient). NEE: against h = 3e-4, 2.0e-6 to 1.2e-5 at four samples, and 1.2e-3 and 7.9e-3 at two samples whose stencils meet kinks (the difference does not shrink at h = 1e-4); against h = 3e-3, 7e-7 to 1.0e-2; h = 1e-2 to 1e-1 is truncation-limited (up to 0.11). GPP and ET behave the same way (ET up to 3.9e-3 against h = 3e-4). h = 1e-3 was chosen.
- **Independent check** at 40 random samples, against fourth-order central differences with h = 2e-3 and against central differences with h = 3e-4 (relative 2-norm error of the gradient): NEE median 2.0e-5, 90th percentile 3.7e-3, maximum 9.9e-3 at these 40 samples; ET median 2.2e-5, 90th percentile 2.6e-3, maximum 4.1e-2 (1.1e-1 against h = 3e-4 at one sample near a kink); GPP median 7.1e-6, 90th percentile 4.6e-3, maximum 1.8e-2. Scratch scripts, not committed. **These 40 samples miss the worst region, small `soilWHC`** (next item); the maxima above are not maxima over the data set.
- **Small `soilWHC`** (found by the independent review, which measured errors of 3–9 % in the NEE gradient at `soilWHC` ≈ 0.5–0.6 cm). The soil bucket empties and refills at many time steps when it is small, so the NEE and ET responses have densely packed kinks and the h = 1e-3 stencil straddles some of them. At all 100 samples with normalized `soilWHC` < −0.8 (`soilWHC` < 3.69 cm; 14 below 0.6 cm), `fd_check_soilWHC.csv` holds ∂/∂`soilWHC` with h = 3e-5 and 1e-5. Replacing only the `soilWHC` component of the stored gradient by the h = 3e-5 value (in the review's check at a sample with `soilWHC` = 0.57 cm the other components agreed to 4+ digits), the relative 2-norm error of the stored normalized gradient is: **NEE median 8.7e-4, above 1 % at 12 samples, maximum 0.137 (13.7 %) at `soilWHC` = 0.48 cm, at most 2.0e-2 for `soilWHC` ≥ 0.6 cm**; ET median 2.2e-4, maximum 0.028; GPP median 2.3e-4, maximum 0.035. The h = 3e-5 reference is itself uncertain where the kinks are densest (h = 1e-5 differs from it by up to 8.1e-3 for NEE, 1.9e-3 for ET, 6.8e-3 for GPP), so a smaller step does not give a converged value everywhere and the shipped gradients keep h = 1e-3.
- **Effect on the subspaces.** With the h = 3e-5 `soilWHC` derivatives substituted at those 100 samples, the eigenvalue ratios λ1/λ2, λ2/λ3, λ3/λ4 become 3.675, 1.161, 3.088 for NEE (stored 3.666, 1.162, 3.089), 52.007, 2.833, 7.641 for ET (stored 51.941, 2.833, 7.641) and 2.666, 1.315, 6.937 for GPP (stored 2.661, 1.316, 6.942), and |w1ᵀw1_new| = 1.000000 for all three. The notebook prints this check. Dropping the 14 samples with `soilWHC` < 0.6 cm instead is not a test of the error: it truncates the input distribution and removes the steepest real part of the ET response (λ1/λ2 for ET falls to 17.9), so we report the substitution.

## How these files relate to the paper

- **Not the paper's computation.** Same model family, priors, parameters and initial pools; different site driver (Niwot instead of Bartlett), period (2004–2005 instead of 2005–2006), time step (half-daily instead of half-hourly), non-calibrated parameter values (PEcAn template instead of the Bartlett trait meta-analysis), SIPNET version (2026 instead of the 2018 PEcAn build), soil-respiration structure (no cold-season soil respiration; in the paper's PEcAn set-up `baseSoilRespCold` = 0.25 × `baseSoilResp` and `soilRespQ10Cold` = `soilRespQ10`, see Configuration), and QoIs (forward totals instead of likelihood statistics). The climate puts the model in a net-source regime (902 of 1000 samples).
- **Ranking differences.** The top parameter agrees with Fig. 2a for NEE (`psnTOpt`) and LE/ET (`soilWHC`). The largest discrepancy, `baseSoilResp` (2nd for NEE in Fig. 2a, about 0.30; 4th here, activity-score fraction 0.081), has candidate causes in the cold, small-soil-pool set-up and in the missing cold-season coupling, which in the paper's build would let `baseSoilResp` and `soilRespQ10` also set winter soil respiration.
- **What would make it the paper's set-up**: the PEcAn workflow files of the paper (Table A7 lists workflow IDs on pecan2.bu.edu, not checked), or the Bartlett met driver in SIPNET format and the PFT trait values. With those, `generate_data.py` needs only a different `.clim` and `.param`. A practical route without the paper's files: download the AmeriFlux US-Bar half-hourly data for 2005–2006 and convert them with PEcAn's `met2model.SIPNET`; the cold-season coupling would still need SIPNET v1.

## License and credit

Code: MIT (`LICENSE`). Data (the CSVs): CC BY 4.0 (`LICENSE-DATA`). `niwot_2004_2005.clim` and `sipnet_template.param` come from SIPNET and PEcAn under BSD 3-clause (`LICENSE-SIPNET-PEcAn.txt`); the Niwot Ridge meteorology was collected by the Niwot Ridge AmeriFlux site (US-NR1) team (Sacks et al. 2006). Model: SIPNET, B. H. Braswell, W. J. Sacks and contributors; PEcAn project. Study design and priors: Fer et al. (2018), cited above, CC BY 4.0. Cite also B. H. Braswell, W. J. Sacks, E. Linder, D. S. Schimel, *Global Change Biology* 11(2):335–355, 2005, doi:10.1111/j.1365-2486.2005.00897.x, and W. J. Sacks, D. S. Schimel, R. K. Monson, B. H. Braswell, *Global Change Biology* 12(2):240–259, 2006, doi:10.1111/j.1365-2486.2005.01059.x.
