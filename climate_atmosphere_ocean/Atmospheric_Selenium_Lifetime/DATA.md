# Atmospheric_Selenium_Lifetime: data

Kind: **stored data**. The values were computed by Feinberg et al. (2020) with the aerosol-chemistry-climate
model SOCOL-AER (with their selenium chemistry) and deposited at the ETH Research Collection,
[doi:10.3929/ethz-b-000357105](https://doi.org/10.3929/ethz-b-000357105) (CC BY 4.0). They cannot be
regenerated here (one run took about 14 h on 48 cores, p. 1373; the code is available on request under the
ECHAM5 licence). `convert_data.py` copies the input files character for character and reduces the gridded
annual-mean output fields to global totals by a stated formula; nothing is refitted or corrected.

## Files

| file | contents | role |
|---|---|---|
| `Atmospheric_Selenium_Lifetime.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 425; outputs committed |
| `se_training_inputs.csv` | 400 × 36: run (1–400), experiment number (20001–20516), the 34 inputs in physical units (names and units in the second `#` line) | deposit `Parameter_values_training_runs_original_units.csv`, values unchanged (the header labels emiss_vol as the volcanic map scaling) |
| `se_training_inputs_transformed.csv` | 400 × 36: the same runs in the authors' transformed units (log10 for the 20 log-uniform inputs; the conditional CDF value of emiss_aer; the latent value of lat_aer, band = ceil) | deposit `Parameter_values_training_runs_transform_units.csv`, unchanged |
| `se_training_outputs.csv` | 400 × 22: run; global totals of the 9 annual-mean Se burdens (kg Se), 4 wet and 4 dry deposition rates (kg Se/s); total burden, wet, dry; tau (days) | area integrals of the deposit's `sensitivity_runs_data.nc`; row i matches row i of the inputs |
| `se_validation_outputs.csv` | 50 × 22: the same for the 50 validation runs | area integrals of `validation_runs_data.nc`; no inputs (see Unknowns) |
| `convert_data.py` | checks the SHA-256 of the four deposited files, copies the inputs, integrates the fields; `--check --out DIR` compares | the conversion (numpy, scipy) |
| `check_accuracy.py` | recomputes the Accuracy numbers that the notebook and `convert_data.py` do not print | check |

The CSVs are plain comma-separated numbers after two `#` lines (a description, then the column names with
units), so `np.loadtxt(path, delimiter=',')` reads them. Deposited input labels → column names: `Emiss_mar` →
`emiss_mar` (likewise terr, ant, vol), `%OCSe_ant` → `pOCSe_ant` (likewise CSe2, CSSe, H2Se, and `%H2Se_vol` →
`pH2Se_vol`); the others keep their names (`k1` … `k17`, `fk18` … `fk21`, `acc_coeff`, `r_aer`, `emiss_aer`,
`lat_aer`). Output columns keep the deposit's variable names (`bOCSe`, `waer_se`, `dOX_Se_IN`, …).

The gridded fields themselves (64 × 128 per run and variable; 222,831,808 and 27,860,808 bytes) are not
shipped; they are the record's files, with the checksums below.

## Model

**Citation.** A. Feinberg, M. Maliki, A. Stenke, B. Sudret, T. Peter, L. H. E. Winkel, "Mapping the drivers of
uncertainty in atmospheric selenium deposition with global sensitivity analysis", *Atmospheric Chemistry and
Physics* 20(3):1363–1390 (2020), doi:10.5194/acp-20-1363-2020 (publisher PDF read; CC BY 4.0; checked on
Crossref). The second author is Moustapha Maliki (the data record lists him as "Moustapha, M.").

**Simulator** (pp. 1365–1368, 1373). SOCOL-AERv2: ECHAM5 general circulation model + MEZON chemistry + AER
sulfate aerosol microphysics (40 bins), with interactive wet and dry deposition (Sheng et al. 2015). The paper
adds 7 gas-phase Se tracers, Se in the 40 sulfate bins, and a monodisperse "dummy" aerosol tracer emitted in
one latitude band. Runs: year-2000 boundary conditions, T42 (64 × 128 Gaussian grid), 39 levels to 0.01 hPa,
chemistry decoupled from radiation, 18 months from zero Se, the last 12 analysed; about 14 h on 48 cores,
about 1e9 core-seconds for the 400 runs (p. 1373).

**Inputs** (m = 34; Table 3, p. 1370; distributions independent except emiss_aer).

| name(s) | description | units | distribution | source |
|---|---|---|---|---|
| k1–k12 | measured rate constants (OCSe+O, CSe2+O ×2, CSSe+O ×2, DMSe+OH/NO3/O3/Cl, H2Se+O/Cl/O3) | cm3 molec-1 s-1 | log-uniform, bounds k^(1±0.12), Eq. 1, cap 1e-9 | Table 3; Sect. 3.1.1, p. 1369 |
| k13–k17 | estimated rate constants (OCSe+OH, CSe2+OH, CSSe+OH ×2, H2Se+OH) | cm3 molec-1 s-1 | log-uniform, bounds from Eq. 2 | Table 3; Sect. 3.1.2 |
| fk18–fk21 | photolysis scale factors (OCSe, CSe2, CSSe, H2Se) | 1 | uniform [0, 2] | Table 3; Sect. 3.1.3 |
| emiss_mar | marine biogenic emissions | Gg Se/yr | log-uniform [0.4, 35] | Table 3; Sect. 3.1.4 |
| emiss_terr | terrestrial biogenic emissions | Gg Se/yr | uniform [0.15, 5.25] | Table 3 |
| emiss_ant | anthropogenic emissions | Gg Se/yr | uniform [3, 9.6] | Table 3 |
| emiss_vol | volcanic emissions | deposited column: map scaling, 1 unit ≈ 1.27 Gg Se/yr emitted (see below; the CSV header says so) | Table 3: log-uniform [0.076, 49.1]; deposited design: [0.06, 39] | Table 3; this data set |
| pOCSe_ant, pCSe2_ant, pCSSe_ant, pH2Se_ant | species fractions of anthropogenic emissions | % | uniform [0, 6] | Table 3; Sect. 3.1.5 |
| pH2Se_vol | H2Se fraction of volcanic emissions | % | uniform [0, 13] | Table 3 |
| acc_coeff | accommodation coefficient of oxidized Se | 1 | uniform [0.02, 1] | Table 3; Sect. 3.1.6 |
| r_aer | dummy aerosol radius | µm | log-uniform [0.01, 3] | Table 3; Sect. 3.1.7 |
| emiss_aer | dummy aerosol emission | kg/yr | lognormal given r_aer: ln e ~ N(19.0 + 2.54 r, 2.88²) | Sect. 3.1.7, p. 1372 |
| lat_aer | dummy aerosol latitude band, 1 = 90–80° S … 18 = 80–90° N | band | discrete uniform on 18 bands | Table 3 |

**The law of emiss_aer.** The paper gives "mean µ = 2.54r + 19.0 and standard deviation σ = 2.88" without
units or the base of the logarithm. The deposited files fix it: the transformed file stores
u = Φ((ln e − 19.0 − 2.54 r)/2.88) with e in kg/yr and r in µm (to 1.1e-03; a least-squares fit of the 400 runs
gives 18.9923 + 2.5446 r + 2.8816 z, so the printed constants are rounded). This u is the authors' independent
coordinate (a Rosenblatt transform).

**emiss_vol.** The deposited column spans 0.06055–38.55. The log-uniform ranges on which it is a Latin hypercube
(one run in each of 400 strata) have lower ends 0.059997–0.060011 and upper ends 38.9990–39.0052, so the design
range is [0.06, 39] in the column's units (notebook, four linear programs over the strata constraints); on Table
3's [0.076, 49.1] the column occupies only 293 strata and 15 runs are below 0.076. Table 3's bounds exceed the
design bounds by factors 1.267 and 1.259. In a regression of total deposition on the four emission columns the
volcanic coefficient is 1.2693 (the others 0.9839, 1.0253, 1.0219). This data set's reading: the column scales a
volcanic map that emits about 1.27 Gg Se/yr per unit, and Table 3's range is the emitted Se (the paper says every
map was normalized to 1 Gg Se/yr, p. 1371). A candidate cause, also this data set's inference: 0.06 and 39 are
the paper's volcanic Se/S ratio bounds, 6e-6 and 3.9e-3 (p. 1371), times 10 Tg S/yr, while Table 3's
0.076–49.1 are the same ratios times the 12.6 Tg S/yr the paper uses (p. 1371); 12.6/10 = 1.26. The column is
shipped as deposited; its CSV header names it a map scaling with 1 unit ≈ 1.27 Gg Se/yr emitted. For the
active-subspace analysis the factor does not matter: the normalized coordinate on [0.06, 39] is, up to the
rounding of Table 3's bounds, the one on [0.076, 49.1] of emitted Se.

**Quantity of interest.** The global atmospheric Se lifetime, "the total atmospheric Se burden divided by
total deposition" (p. 1373):
tau = Σ_s Σ_cells b_s A_j / Σ_d Σ_cells F_d A_j / 86400 s/d, over the 9 burden and 8 deposition variables of
the deposit, with A_j = a² w_j 2π/128 from the 64 Gauss–Legendre weights w_j (a = 6.371e6 m cancels). The
paper also maps deposition in 8192 grid boxes and averages Sobol' indices over cropland and pasture (Fig. 11,
p. 1383, with the maps of Ramankutty et al. 2008); it reports no scalar agricultural deposition value, and the
maps are not in the deposit, so that quantity is not derived here. The shipped totals of each deposition
variable allow other global quantities (total deposition, wet share).

**Gradients.** None: SOCOL-AER computes none, and finite differences would cost 35 runs of about 14 h on 48
cores per gradient. The notebook works in normalized coordinates z = 2F(x) − 1 (each input through its own
CDF: Table 3's laws, emiss_vol on its design range [0.06, 39], emiss_aer through its conditional CDF, lat_aer at the band
midpoints (2 band − 19)/18); any gradient there is per unit of probability.

**Sampling design** (pp. 1372–1374). A Latin hypercube of 400 points ("each parameter range is divided into N
equally probable subintervals"); the 50 validation runs enrich it to "a pseudo-Latin hypercube of 450 runs"
with maximal distance to the training runs (p. 1374), a distance-maximizing enrichment, not an independent
draw from the input law. In the deposited transformed file 28 inputs (emiss_aer's
u and the latent lat_aer included) occupy each of the 400 strata exactly once; k13–k17 lie inside their Table 3
ranges but occupy only 363–373 strata (not stratified on these ranges; the paper does not say why), and
emiss_vol is stratified on [0.06, 39], not Table 3's range (notebook). lat_aer has 21–23 runs per band.

**Published results.** No active subspace. The lifetime surrogate is a burden PCE divided by a deposition PCE
(Sect. 3.3, pp. 1373–1374: hyperbolic truncation q = 0.75, p. 1373; least-angle regression and a degree-adaptive
scheme up to degree 13, p. 1374; computed with UQLab, "Code and data availability", p. 1385). Fig. 4 (p. 1377):
resampling it gives median 4.4 d, 2nd–98th percentiles 2.9–6.4 d. Fig. 5 (p. 1378): total Sobol' indices > 0.05 for k13 (largest), the aggregated dummy
aerosol, emiss_vol, emiss_mar, k1, %H2Se_vol, %OCSe_ant, k15, emiss_terr; first-order indices sum to 53 %
(p. 1377). LOO errors 0.02 (burden PCE), ~1e-6 (deposition PCE), 0.35 for the lifetime surrogate (p. 1376).
Mass conservation 98–102 % (p. 1376).

## Regenerating

The model runs cannot be regenerated from the repository. The CSV files are regenerated from the deposited
files, with a SHA-256 check of each:

```
python convert_data.py <folder with the deposited files> --out <dir>             # writes the four CSVs
python convert_data.py <folder with the deposited files> --check --out <dir>     # also compares
```

`<dir>` must be outside this directory (not the directory or a folder in it); the script refuses it otherwise.

`--check` writes the CSVs to `<dir>` and compares every number with the shipped files, relative tolerance
1e-12; in the build environment (Python 3.13.18, numpy 2.5.4, scipy 1.18.1, active_subspaces 1.0.0.dev0; 2026-11-03) it reports a largest
relative difference of 0.0e+00 for all four files and identical text for the two input files, in about 4 s
(again on 2026-11-08, after the emiss_vol header was changed).
It also prints the largest difference between the file's latitudes and the Gauss–Legendre latitudes
(7.1e-15 degrees). The notebook takes about 1 min (46 s printed in the committed `.venv` run), most of it the 400
leave-one-out refits (of tau, 1/tau and log tau) and 300 bootstrap refits of the penalized quadratic.

**Source.** ETH Research Collection, "Simulation data used for sensitivity analysis of atmospheric selenium in
SOCOL-AER" (A. Feinberg et al., 2019), doi:10.3929/ethz-b-000357105, CC BY 4.0; the DOI the paper's "Code and
data availability" gives (p. 1384). Files used (downloaded 2026-11-03):

| file | bytes | SHA-256 |
|---|---|---|
| `sensitivity_runs_data.nc` (NetCDF classic) | 222,831,808 | `99dd85465c11bd7afa5687ca37f6166018c94526bbc2cfb0d53a18cc0bbc40a2` |
| `validation_runs_data.nc` | 27,860,808 | `e483d33c28c7d2635710e188ad7274efcfe91e0aa4893b4bd69ab30d8ae3ed0a` |
| `Parameter_values_training_runs_original_units.csv` | 177,088 | `e2b2033bcc2ef90b74a482db17c8f3503afedf114907916dc54edfa9077f94be` |
| `Parameter_values_training_runs_transform_units.csv` | 244,511 | `42cbf7093a50d350e04d85cad901f551723c9bc2cb77079fff45a3ff11940650` |

The record's README lists two more files, `Parameter_values_validation_runs_original_units.csv` and
`..._transform_units.csv`, which the record page did not offer for download.

**External run** (not reproducible from the repository). To rerun the model: obtain SOCOL-AER with the Se scheme
from the authors (ECHAM5 licence agreement with MPI-M; the address http://www.mpimet.mpg.de/en/science/models/license/
returned 404 on 2026-11-09, and MPI-M's current licence page is the MPI-ESM licence project,
https://code.mpimet.mpg.de/projects/mpi-esm-license), build it for a
cluster, use the authors' 10-year spin-up initial file and year-2000 boundary conditions, set the 34 inputs of
one row of `se_training_inputs.csv` (emiss_vol in the column's units, emiss_aer in kg/yr, lat_aer as the band),
run 18 months at T42L39 with chemistry-radiation decoupling, average months 7–18 of the 17 Se fields, and
integrate them as `convert_data.py` does. Agreement to within about 1 % of the deposited tau would be the
check; bit-identity is not expected on another platform.

## Accuracy

`check_accuracy.py` prints the numbers marked with its name (checks 1–2 from the shipped CSVs, under 1 s;
check 3 needs the deposited `sensitivity_runs_data.nc`, about 3 s).

- Conversion: lossless for the inputs, and the output totals reproduce to 0.0e+00 relative (`convert_data.py --check`).
- Stored precision (`check_accuracy.py`): the physical file writes k1–k17 with 3 significant digits (relative
  rounding up to 4.9e-03 against the transformed file's full-precision values, median 6.4e-04) and the other
  inputs with 15. The notebook builds its coordinates from the physical file; from the transformed file they
  would differ by at most 2.9e-03 and the OLS direction would not change (notebook).
- Quadrature of the area integral (`check_accuracy.py`): with cos(latitude) cell weights instead of the
  Gauss–Legendre weights tau changes by at most 5.1e-05 (relative); with float32 instead of float64
  accumulation by at most 1.3e-07. Both are far below the run-to-run spread of tau (3.11–10.07 d).
- Mass balance (notebook): total deposition is 0.987–1.023 of emissions with the volcanic factor 1.269, and
  0.988–1.233 with the deposited column taken as Gg Se/yr (paper: 98–102 %).
- No gradients, so no gradient check. Fits (notebook, everything refitted in each fold): LOO R² 0.360 (linear),
  0.492 (ridge-penalized quadratic in all 34 inputs, penalty 42.2 chosen by LOO), 0.428 / 0.501 / 0.500 for a
  quadratic in 1 / 2 / 3 active variables (gain from 1 to 2: paired-bootstrap 95 % interval [0.016, 0.133]). On
  1/tau and log tau the same pipeline gives 0.488 / 0.597 / 0.607 and 0.471 / 0.571 / 0.581 (notebook): the 8 runs
  with tau > 6.4 d hold 0.435 of the sum of squares of tau, so part of the unexplained variance is the tail.
- The eigenvalue ratios belong to the ridge-penalized surrogate and depend on its penalty (notebook): λ1/λ2 =
  10.88 at the chosen penalty, 9.29–13.09 over penalties with LOO R² within 0.005 of the best, 6.03–38.56 within
  0.05. The dimension is chosen from refitted LOO R², not from the gap.

## How these files relate to the paper

The inputs are exactly the 400 training runs of the paper's surrogates, and the outputs are reductions of the
deposited annual-mean fields from which the paper's lifetime (p. 1373) is computed; the notebook computes tau
directly for each run instead of dividing two surrogates. The median and 98th percentile of tau over the runs
(4.36 d, 6.38 d) agree with Fig. 4 (4.4 d, 6.4 d); the 2nd percentile does not (3.52 d against 2.9 d, and no run
is below 3.11 d), so the paper's lower tail comes from its surrogate. The notebook's leading direction ranks
k13 first, as Fig. 5 does, and its signs agree with the effects described in Sect. 5.1; the order of the other
inputs differs from Fig. 5's total Sobol' indices (an entry of w1 and a total index measure different things:
interaction-dominated inputs such as emiss_mar, k15 and %H2Se_vol rank low in w1). Two inconsistencies
between the paper and the deposit are reported, not fixed: the emiss_vol range (Table 3 vs the design column)
and the mass balance (98–102 % in the paper; 0.988–1.233 with the column as Gg Se/yr). The paper's
deposition-map analyses (Figs. 8–11) and its comparison with rainwater measurements (Fig. 12) are not
reproduced.

## License and credit

Data: © Feinberg et al., ETH Research Collection doi:10.3929/ethz-b-000357105, CC BY 4.0 (as stated for the
record); article CC BY 4.0. Cite the paper and the data record when using these files. The CSVs here are CC BY
4.0. `convert_data.py`, `check_accuracy.py` and the notebook: MIT (this repository). No SOCOL-AER code is included.

## Unknowns

- **Validation inputs.** The input values of the 50 validation runs (`Parameter_values_validation_runs_*.csv`,
  listed in the record's README) were not offered on the record page, so the validation runs are used only as a
  second sample of tau. With them, the notebook's fits could be tested on 50 held-out runs (chosen by a
  distance-maximizing enrichment of the design, p. 1374, so not an independent sample of the input law). Action:
  ask the authors or the ETH Research Collection to add the two files to the record.
- **emiss_vol units.** The factor of about 1.27 between the design column ([0.06, 39]) and Table 3 is inferred from
  the strata and the mass balance; the paper does not state it. Was the design drawn on the Se/S ratios times
  10 Tg S/yr instead of 12.6 Tg S/yr, and which range (emitted Se or map scaling) do Fig. 5 and Fig. 6f refer to?
- **k13–k17 design.** Why the five estimated rate constants are not stratified on their Table 3 ranges is not
  known; their bounds may have been computed differently when the design was drawn.
- **The lifetime formula.** The paper gives it in words; the ratio of annual means of burden and deposition, and
  the Gaussian-grid cell areas, are this data set's choices (cos-latitude weights change tau by at most 5.1e-05).
- **Agricultural-region deposition** (named in the paper's motivation, Fig. 11) is not derived: the
  cropland/pasture maps are not in the deposit and the paper reports no value to check against.
- **Grid orientation** of the deposited fields (north to south) is taken from the file's latitude variable.
- **Long upper tail of tau.** 8 runs (2 %) hold 0.435 of its sum of squares; the R² values of the fits depend on
  this scale (higher on 1/tau and log tau). Whether these runs are physical extremes or numerical artefacts of
  SOCOL-AER is not known.
- **Latitude band of the dummy aerosol.** The residuals of the quadratic in two active variables differ by band
  (Kruskal–Wallis p = 0.005, non-monotone); a quadratic in the coded band cannot represent this, so the subspace
  says nothing about the band's effect.
- **Lifetime surrogate error.** The paper's 0.35 (p. 1376) is not defined beyond Eq. 7, which applies to a single
  PCE; how it was computed for a ratio of two PCEs is not known.
