# Photosynthesis_Stomatal_Resistance: data

Kind: **stored data**. ParFlow-CLM (PF-CLM) produced the files, and they are the only record of these runs. The notebook reads them as shipped and changes no value.

## Files

| file | contents | role |
|---|---|---|
| `Photosynthesis_Stomatal_Resistance.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom; outputs committed; no random numbers |
| `stomata_realization_inputs.txt` | 301 × 20 inputs | stored |
| `param_ranges.txt` | 20 × 2 bounds (min, max), one row per input in the column order of the inputs file | stored |
| `Tonzi_Stomata_SC.mo_avg_tran_veg.txt` | 301 × 12 monthly transpiration | stored |
| `Tonzi_Stomata_SC.mo_avg_rssun.txt` | 301 × 12 monthly sunlit stomatal resistance | stored |
| `Tonzi_Stomata_SC.mo_avg_rssha.txt` | 301 × 12 monthly shaded stomatal resistance | stored |
| `Tonzi_Stomata_SC.JJA_hr_avg_tran_veg.txt` | 301 × 14 hourly transpiration, June–August | stored |
| `Tonzi_Stomata_SC.JJA_hr_avg_rssun.txt` | 301 × 14 hourly sunlit stomatal resistance, June–August | stored |
| `Tonzi_Stomata_SC.JJA_hr_avg_rssha.txt` | 301 × 14 hourly shaded stomatal resistance, June–August | stored |

Row i of every output file is the run with the inputs of row i of `stomata_realization_inputs.txt`.

| file | bytes | md5 |
|---|---|---|
| `param_ranges.txt` | 192 | 536eb928c70909544cfabe371dd1d388 |
| `stomata_realization_inputs.txt` | 32072 | 21b71cb595a56aee40576f2dc769524e |
| `Tonzi_Stomata_SC.mo_avg_tran_veg.txt` | 75519 | 37770c56595ef034b59f5c6b666b12ba |
| `Tonzi_Stomata_SC.mo_avg_rssun.txt` | 61156 | 2216e93e028b112bebec4095c141f707 |
| `Tonzi_Stomata_SC.mo_avg_rssha.txt` | 61133 | 30b2fe2220ea41fa849a7d7422c49616 |
| `Tonzi_Stomata_SC.JJA_hr_avg_tran_veg.txt` | 88158 | 07542375ea953657c13cad07eb1cdba7 |
| `Tonzi_Stomata_SC.JJA_hr_avg_rssun.txt` | 71280 | 7078458a78b18ad92214156f0839b07c |
| `Tonzi_Stomata_SC.JJA_hr_avg_rssha.txt` | 71134 | 3e14b17dbe843498081af489e3801f54 |

All files are space-separated text with one header row of quoted names and no index column. All values are positive and finite.

## Inputs (`stomata_realization_inputs.txt`, `param_ranges.txt`)

- Columns, as stored: `qe25 ko25 kc25 vcmx25 ako akc avcmx bp mp drb drs ecp hv ocr sv wj1 wj2 rsmax0 ppcd we1`. The paper's Table 1 and the notebook use another order (vcmx25 first); the notebook reorders by name. `param_ranges.txt` has no name column; its rows follow the stored column order.
- The values are physical, not normalized, and rounded (3 to 235 distinct values per column).
- **Row 0 is the default run.** All 20 values equal the Default column of Table 1 of the paper. The paper keeps it apart from "the 300 realizations" (p. 909). Rows 1–300 are the realizations. The notebook fits rows 1–300 only; the file is unchanged.
- Ranges, defaults and units are those of Table 1 of the paper; the notebook's table lists them all. Units: vcmx25 µmol CO2 m⁻² s⁻¹; hv J mol⁻¹; sv J mol⁻¹ K⁻¹; kc25, ko25 Pa; ecp µmol J⁻¹; qe25 µmol CO2 µmol photon⁻¹; bp µmol m⁻² s⁻¹; rsmax0 s m⁻¹; ppcd see below; the other inputs are dimensionless.
- **ppcd** is stored as a mole fraction, mol mol⁻¹ (bounds 3.55e-4 to 4e-4). The paper gives 355–400 ppm.
- **wj1 and wj2 are discrete.** wj1 ∈ {1, 4, 4.5} and wj2 ∈ {2, 8, 10.5}, the sets of Table 1 of the paper (hard-coded constants in CLM). In rows 1–300 the levels have 93, 98, 109 runs (wj1) and 87, 104, 109 runs (wj2), and all 9 combinations occur (28 to 41 runs each). The 2016 notebook described them as U(1, 4.5) and U(2, 10.5). The other 18 inputs lie inside their bounds; how they were sampled (step 1 of the paper says uniform draws) is not recorded beyond that.

## Outputs (`Tonzi_Stomata_SC.*.txt`)

Forcing: Tonzi Ranch, California, September 2002 to August 2003, the average year of the three Tonzi years in the paper (§3b).

| file prefix | columns | values |
|---|---|---|
| `mo_avg` | `September` … `August` (water-year order) | mean of the daytime hourly values in each month |
| `JJA_hr_avg` | `GMT_0`, `GMT_1`, `GMT_12`, …, `GMT_23` (stored in this order) | mean of the daytime values at each UTC hour over June–August 2003 |

| quantity | file suffix | unit (inferred) | range, rows 1–300 |
|---|---|---|---|
| transpiration | `tran_veg` | mm s⁻¹ (= kg m⁻² s⁻¹) | monthly 5.13e-7 to 8.38e-5; hourly 3.57e-7 to 1.2e-4 |
| sunlit stomatal resistance | `rssun` | s m⁻¹ | monthly 241 to 7.89e3; hourly 380 to 9.69e3 |
| shaded stomatal resistance | `rssha` | s m⁻¹ | monthly 1.01e3 to 2.32e4; hourly 1.2e3 to 3.2e4 |

- **Units are not stated in the files and are not confirmed** (author, 2026-10-30). They are inferred from the paper: transpiration × 86400 gives 0.04 to 7.2 mm d⁻¹ (monthly) and 0.03 to 10.4 mm d⁻¹ (hourly), on the scales of Figs. 4f and 9d, and the resistances match the s m⁻¹ axes of Figs. 7 and 9.
- **Local time is not settled.** The hourly columns are in UTC. Fig. 8 of the paper labels UTC 12 as 6 am, i.e. UTC−6; Pacific daylight time, in effect at Tonzi Ranch in summer, is UTC−7. Which is right is unknown (author, 2026-10-30).

## Provenance

- Generating code: ParFlow coupled to the Common Land Model (PF-CLM), single-column domain, deciduous tree cover, loam soil (paper §2b). The PF-CLM version, the run date and the machine are **not recorded** (author, 2026-10-30). The runs were made for the study of Jefferson, Maxwell and Constantine at the Colorado School of Mines.
- The paper made 2,100 runs: these 300 realizations with 5 forcing sets plus 2 extra scenarios. Only the Tonzi average-year runs (and the default run) are in this repository.
- The processing from hourly PF-CLM output to monthly and hourly means is not in the repository.

## Regenerating

The files cannot be regenerated from this repository. The notebook reads them and runs in about 10 s on an idle machine (25 s under load). It uses no random numbers, and two runs give identical printed output. With all 301 rows, the v1 directions equal those of the 2016 notebook up to sign to 4.4e-16; with row 0 dropped, |cos| ≥ 0.9997.

## License and credit

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`). Credit the PF-CLM study of J. L. Jefferson, R. M. Maxwell and P. G. Constantine (Colorado School of Mines). The 2016 notebook was written by Ryan Howard. Cite J. L. Jefferson, R. M. Maxwell and P. G. Constantine, "Exploring the sensitivity of photosynthesis and stomatal resistance parameters in a land surface model", J. Hydrometeorol. 18(3):897–915, 2017, doi:10.1175/JHM-D-16-0053.1; the study is also in J. L. Jefferson, PhD thesis, Colorado School of Mines, 2016, hdl:11124/170080.
