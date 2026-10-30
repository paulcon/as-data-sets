# Land_Surface_Energy_Fluxes: data

Kind: **stored data**. ParFlow-CLM (PF-CLM) produced the files, and they are the only record of these runs. The notebook reads them as shipped and changes no value.

## Files

| file | contents | role |
|---|---|---|
| `Land_Surface_Energy_Fluxes.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom; outputs committed; no random numbers |
| `SC_ScatterParameters_grass.txt` | 300 × 19 inputs, grass cover | stored |
| `SC_ScatterParameters_bare.txt` | 300 × 3 inputs, bare soil | stored |
| `ScatterSummary_grass.txt` | 300 × 5 outputs, grass cover | stored |
| `ScatterSummary_bare.txt` | 300 × 5 outputs, bare soil | stored |

Row i of a `SC_ScatterParameters_*` file is the input of row i of the `ScatterSummary_*` file with the same land cover. All runs use the single column domain of Jefferson et al. (2015).

| file | bytes | md5 |
|---|---|---|
| `SC_ScatterParameters_bare.txt` | 4588 | 8c3e83a294fc525cb59324af94276b7c |
| `SC_ScatterParameters_grass.txt` | 26666 | 19e7c4cd25877e27f1d93fe7f201875f |
| `ScatterSummary_bare.txt` | 25345 | 1e2ba7c3334da2f5b72d2ff52597b939 |
| `ScatterSummary_grass.txt` | 25310 | 4989dab98b8d6e6d18aa0a3fbbf544f8 |

## Inputs (`SC_ScatterParameters_*.txt`)

Space-separated text, one header row of quoted parameter names, then 300 rows. No index column. The values are physical, not normalized. The grass file's column order differs from the paper's Table 2; the notebook reorders the columns by name.

- Grass columns, as stored: `lai lai0 sai z0m dleaf rhol_vis rhol_nir rhos_vis rhos_nir taul_vis taul_nir taus_vis taus_nir xl fieldcap dewmx watertable soilcol wiltpt`.
- Bare soil columns, as stored: `soilcol z0m watertable`.

Distributions and units are those of Table 2 of the paper. The notebook's table lists them all. The units are mm for dewmx, m for dleaf, z0m and watertable; the other inputs are dimensionless.

- 18 inputs are uniform on the ranges in the table (grass). Bare soil uses only soilcol and watertable of these.
- z0m (m) is lognormal: log z0m ~ N(−5.11, 0.52²) for grass and N(−8.83, 1.47²) for bare soil. The sample mean and standard deviation of log z0m are −5.10 / 0.50 (grass) and −8.94 / 1.53 (bare soil). The paper's step 1 (§3) says all inputs are uniform; its eq. (14) and App. A.1, and these data, have log z0m normal.
- soilcol is an integer class, 1 to 8, normalized like a uniform input on [1, 8].
- The values are rounded to one or two significant figures, so each column has few distinct values: 8 to 89 for grass (z0m has 21), and 8, 219 and 87 for bare soil (soilcol, z0m, watertable).
- Every uniform input lies inside its range.
- The grass default point of the paper's Table 2 lies outside the sampled ranges. Its rhol_nir = 0.58 is above U(0.35, 0.55), and Table 2 gives both values; which one is in error is unknown. Its z0m = 0.03 m is above every sampled grass value (largest 0.029).

## Outputs (`ScatterSummary_*.txt`)

Space-separated text, no header, 300 rows × 5 columns. Columns are numbered from 0.

| column | quantity | range, bare soil | range, grass | used |
|---|---|---|---|---|
| 0 | total latent heat | 1.16e6 – 1.74e6 | 5.17e5 – 1.54e6 | yes |
| 1 | total sensible heat | 5.63e5 – 9.21e5 | 6.34e5 – 1.80e6 | yes |
| 2 | total ground heat | 1.68e5 – 4.12e5 | 1.16e5 – 3.52e5 | yes |
| 3 | unknown | 16.6 – 25.0 | 7.4 – 22.1 | no |
| 4 | unknown (values 301.6 – 309.2 suggest a temperature in K; not confirmed) | 302.6 – 309.2 | 301.6 – 308.3 | no |

- The quantities in columns 0–2 are the total latent, sensible and ground heat over the 144-hour forcing period (paper §4.3). Their units and normalization are **unknown** (author, 2026-10-30). The paper labels its outputs W m⁻² (p. 17). The constant terms of the quadratic fits to these files are 103 to 617 times those of the paper's Table 4, with a different factor for each flux (printed by the notebook). So these files are not the paper's outputs in other units.
- The meaning of columns 3 and 4 is **unknown** (author, 2026-10-30).

## Provenance

- Generating code: ParFlow coupled to the Common Land Model (ParFlow-CLM). The PF-CLM version, the run date and who ran the simulations are **not recorded**.
- The 2016 notebook calls these "updated model evaluations" compared with the paper. They are later runs than those behind the paper's figures and tables, and they do not reproduce them: R², fitted coefficients, the ranking of the grass latent heat parameters and the default-point example all differ (the notebook prints each difference). The paper's tilted-v runs are not in the repo.
- The simulation setup (domain, subsurface parameters, the 144-hour forcing from the Little Washita watershed, Oklahoma) is in Tables 2–3 and §2 of the paper.

## Regenerating

The files cannot be regenerated from this repository. The notebook reads them and runs in a few seconds. It uses no random numbers, so two runs give identical output. The 2016 notebook's least-squares eigenvalues ‖â‖² are reproduced to a relative 1e-14, and its in-sample R² to two decimals.

## License and credit

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`). Credit the PF-CLM study of Jefferson, Gilbert, Constantine and Maxwell (Colorado School of Mines); who ran these later runs is not recorded. The 2016 notebook was written by Ryan Howard. Cite J. L. Jefferson, J. M. Gilbert, P. G. Constantine and R. M. Maxwell, "Active subspaces for sensitivity analysis and dimension reduction of an integrated hydrologic model", Comput. Geosci. 83:127–138, 2015, doi:10.1016/j.cageo.2015.07.001 (reprinted in Comput. Geosci. 90:78–89, 2016, doi:10.1016/j.cageo.2015.11.002).
