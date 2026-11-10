# BISON_Fuel_Performance_Surrogates: data

Kind: **stored data**. The values come from 400 runs of the BISON fuel-performance code (Idaho National
Laboratory) by Radaideh and Kozlowski (2020), as distributed with the pyMAISE package (Apache License 2.0).
They were not regenerated: BISON is licensed by INL and is not available here, and the paper's BISON input
file (made with the "PinToPerform" templates, Acknowledgment, p. 30) is not published. `convert_data.py`
copies every number character for character into the CSV layout of this repository.

## Files

| file | contents | role |
|---|---|---|
| `BISON_Fuel_Performance_Surrogates.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2020; outputs committed |
| `bison_fp_inputs.csv` | 400 × 13: the 13 BISON inputs, physical units (column names and units in the second `#` line) | pyMAISE `fp_inp.csv` |
| `bison_fp_outputs.csv` | 400 × 4: fission gas produced (mol), maximum fuel centreline and surface temperatures (K), radial cladding diameter displacement (m); row i matches row i of the inputs | pyMAISE `fp_out.csv` |
| `pyMAISE_LICENSE.txt` | the Apache License 2.0 text distributed in the pyMAISE 0.0.2 wheel | licence of the source |
| `convert_data.py` | reads the two files from the pyMAISE 0.0.2 wheel (or a folder with the Zenodo files), checks SHA-256, writes the CSVs; `--check` compares | the conversion (standard library and numpy) |
| `check_accuracy.py` | recomputes the stored-precision numbers of this file | check |

The CSVs are plain comma-separated numbers after two `#` lines, so `np.loadtxt(path, delimiter=',')` reads
them. Column names are those of the source files.

## Model

**Citation.** M. I. Radaideh, T. Kozlowski, "Surrogate modeling of advanced computer simulations using deep
Gaussian processes", *Reliability Engineering & System Safety* 195:106731 (March 2020),
[doi:10.1016/j.ress.2019.106731](https://doi.org/10.1016/j.ress.2019.106731) (version of record; checked on Crossref). The paper was read in the authors'
accepted manuscript (33 pages); **page, table and equation numbers in this file and the notebook are those of
the accepted manuscript**. The pyMAISE documentation states that the data "originates from" this article.

**Simulator and problem** (Sect. 3.4, pp. 15–16). BISON, "a finite element-based fuel performance code"
(Williamson et al., J. Nucl. Mater. 423:149–163, 2012, doi:10.1016/j.jnucmat.2012.01.012); the paper gives no
BISON equations. Case 1 of the OECD/NEA pellet-cladding mechanical interaction (PCMI) benchmark (Rossiter,
Massara, Amaya, AECL technical report, 2016): a beginning-of-life ramp of a 10-pellet PWR rod to an average
linear heat rate of 40 kW/m, with reduced inner and outer cladding diameters so that contact occurs during the
ramp; uniform axial power and rod surface temperature profiles, the surface at 330 °C (p. 15). The BISON input
file was generated with the "PinToPerform" templates (Acknowledgment, p. 30).

**Inputs** (m = 13; Sect. 3.4, p. 15, and Table 2, p. 16). "The selected uncertain parameters are all sampled
independently from normal distribution with uncertainty values shown in Table 2. The nominal values are taken
from the benchmark" (p. 15); Table 2 gives the relative standard deviation (sd / nominal) of each input. The
nominal values are not printed. Names and units below are those of the pyMAISE files, descriptions those of
Table 2. `ax_pow` is the paper's linear heat rate (LHR, $q'$), stored as a factor of its nominal: it is the only
2.0 % input of Table 2, and the paper's inputs include "inlet temperature, system pressure, and linear heat rate
(LHR)" (p. 15). Sample mean and standard deviation (ddof = 1), printed by the notebook:

| name | description (Table 2) | units | Table 2 sd/nominal | mean | sd (sd/mean) | sample min–max |
|---|---|---|---|---|---|---|
| fuel_dens | fuel density | kg/m³ | 0.2 % | 10456 | 20.1 (0.19 %) | 10396–10522 |
| porosity | porosity | 1 | 15.0 % | 0.046896 | 0.007236 (15.43 %) | 0.021141–0.068638 |
| clad_thick | clad thickness | m | 0.5 % | 5.6976e-4 | 2.978e-6 (0.52 %) | 5.6167e-4–5.7653e-4 |
| pellet_OD | pellet outer diameter | m | 0.1 % | 4.0961e-3 | 4.211e-6 (0.10 %) | 4.0824e-3–4.1081e-3 |
| pellet_h | pellet height | m | 4.4 % | 0.013762 | 5.999e-4 (4.36 %) | 0.012056–0.015709 |
| gap_thick | gap thickness | m | 0.5 % | 1.9008e-5 | 9.633e-8 (0.51 %) | 1.88e-5–1.93e-5 (6 stored values) |
| inlet_T | inlet temperature | K | 0.5 % | 292.31 | 1.433 (0.49 %) | 287.52–296.2 |
| enrich | U-235 enrichment | 1 | 0.1 % | 0.044872 | 4.527e-5 (0.10 %) | 0.044749–0.04501 |
| rough_fuel | fuel roughness | m | 10.0 % | 1.7887e-6 | 1.817e-7 (10.16 %) | 1.26e-6–2.31e-6 |
| rough_clad | clad roughness | m | 10.0 % | 3.9874e-7 | 3.759e-8 (9.43 %) | 2.89e-7–5.19e-7 |
| ax_pow | linear heat rate (factor of nominal) | 1 | 2.0 % | 1.0014 | 0.02132 (2.13 %) | 0.92857–1.0563 |
| clad_T | clad surface temperature | K | 1.0 % | 603.62 | 6.196 (1.03 %) | 585.43–620.33 |
| pressure | pressure | Pa | 1.0 % | 1.5498e7 | 1.478e5 (0.95 %) | 1.5022e7–1.5979e7 |

Every sample relative sd agrees with Table 2 to within 1.82 sampling standard errors (notebook). **The notebook
uses the sample mean and sd as the normal law of each input** (`ac.Gaussian(mean, sd**2)`), a stand-in for the
declared law (Table 2 sd × benchmark nominal), whose nominal is not printed.

**Quantities of interest** (Sect. 3.4, p. 16: "over the last time step only"). `fis_gas_produced` (mol),
`max_fuel_centerline_temp` (K), `max_fuel_surface_temp` (K), `radial_clad_dia` (m, the paper's "radial clad
diameter", p. 16, or "radial clad displacement", p. 26); units from the pyMAISE documentation.

**Gradients.** None: BISON was run as a black box and cannot be rerun here. The notebook uses QPHD (M = 400;
a full quadratic in 13 inputs has 105 coefficients). Inputs are scaled by their sample standard deviations.

**Sampling design.** Independent normal laws (p. 15). The stored samples agree: kurtosis 2.55–3.41, 0.65–0.70
within one sd, Kolmogorov–Smirnov p ≤ 2.9e-13 against a uniform law on the sample range, Shapiro–Wilk
p ≥ 0.073 for 12 of 13 inputs, largest |correlation| 0.143, within what independent samples give (notebook).
The pyMAISE documentation's statement that the inputs "were uniformly randomly sampled independently within
their uncertainty bounds" is an error of that documentation. The sampler (Monte Carlo or a stratified normal
design) and its seed are not stated. The paper says ±3σ "is the support of the samples used to train the
DeepGP models" (p. 27); the stored samples are not truncated there: about the sample mean, in Table 2 sd, they
reach 3.66σ (porosity), 3.64σ (`ax_pow`) and 3.34σ (`pellet_OD`) (notebook), as expected of 400 untruncated
normal values. `gap_thick`, `rough_fuel` and `rough_clad` are stored to 3 significant digits, the others to 5
(`check_accuracy.py`); `gap_thick` has 6 distinct values, a spacing of 1e-7 m = 1.04 sample sd = 1.05 Table 2
sd (notebook), so its stored values are a coarse rounding of the values the runs used (not known).

**Number of runs.** The paper's DeepGP for this case used 200 training and 100 validation samples (Table 3,
p. 18), i.e. 300; pyMAISE distributes 400. Which 300 the paper used is not stated, and subsets do not settle it
(the last 300 rows give 1591.6 K and 701.82 K, the first 300 give 1591.4 K and 701.95 K; notebook).

**Published results** (pp. 26–27). Output mean (relative sd): fission gas 3.1e-5 mol (4.8 %), centreline
1591.6 K (2.4 %), surface 701.8 K (1.1 %), displacement 1.92e-5 m (4.2 %) (p. 26). DeepGP validation $Q^2$ on
100 runs: 0.987, 0.986, 0.933, 0.978 (Table 8, p. 26). Morris screening (800 trajectories, ±3σ bounds) and
Sobol' indices from the DeepGP (Figs. 9–10, p. 27): pellet height and LHR "contribute to more than 98 %" of the
fission gas variance; the centreline temperature is sensitive to porosity, roughness, LHR and clad surface
temperature; the surface temperature variance "is controlled by the clad surface temperature, fuel roughness,
and LHR, with insignificant effect from the pellet height", while its Morris pellet-height effect has a large
σ; the displacement follows the centreline temperature. No active subspace is reported.

## Regenerating

The runs cannot be regenerated here (BISON licensed by INL; input file not published). The CSV files are
rebuilt from the source with a SHA-256 check:

```
pip download pyMAISE==0.0.2 --no-deps -d SRC          # pyMAISE-0.0.2-py3-none-any.whl
python convert_data.py SRC/pyMAISE-0.0.2-py3-none-any.whl --out DIR --check
```

`--out` is required and must lie outside this directory, subdirectories included (the script exits
otherwise, with or without `--check`); it never writes into the data set. `--check` compares the values with a
relative tolerance of 1e-12 and also reports whether the text is identical; run on 2026-11-09 in the Linux
container where the shipped files were written, it printed "True" for both files, both tests (under 1 s). The
wheel is not in this repository, so the check was not repeated in `.venv`.

**Source.** PyPI `pyMAISE-0.0.2-py3-none-any.whl` (402,517 bytes, SHA-256
`feca174a5e0ed1a29a8903f07610b99b39c4d52b7b20d070b2977b78d8286f81`), members `pyMAISE/data/fp_inp.csv`
(SHA-256 `cf1e4c80685f540b73b3133fe43fcf373fdb744a22cf4fe4e4948852aa3c7525`) and `pyMAISE/data/fp_out.csv`
(`72811c5a55c497248a4675db64cd05009172ba4fb029dbc878567348a8333037`). These two hashes equal the ones in the
download registry of current pyMAISE (github.com/aims-umich/pyMAISE, commit 810853b, `pyMAISE/datasets/_handler.py`),
which fetches the files from Zenodo record https://zenodo.org/records/20140559; so the Zenodo files are the
same bytes, and `convert_data.py` also accepts a folder with them. The original files have CRLF line endings;
the CSVs here use LF.

In the repository's `.venv` the notebook runs in about 30 s under nbconvert (22 s by its own timer in the
committed run, with other jobs running), and `check_accuracy.py` in under 1 s.

## Accuracy

- Conversion: lossless (`convert_data.py --check`, above).
- Stored precision (`check_accuracy.py`): inputs 5 significant digits, except `gap_thick`, `rough_fuel`,
  `rough_clad` (3); `fis_gas_produced` and `radial_clad_dia` 3 significant digits (rounding step 1e-7), which
  caps the R² any fit can reach at 0.9996 and 0.9987; the temperatures are stored with 10 significant digits.
- No gradients, so no gradient check. Fit quality is printed by the notebook: linear LOO R² 0.9992 / 0.9967 /
  0.9181 / 0.9966, a quadratic in the active variables 0.9994 / 0.9973 / 0.9401 (n = 2) / 0.9978.
- Non-smooth dependence on the pellet height (notebook): the residuals of both temperatures rise and fall once
  in `pellet_h`. For the surface temperature, one step at 13.256 mm gives LOO R² 0.9628, two steps (13.252 and
  13.923 mm, a raised window about 4 K high) 0.9783, and an additive fit (linear in 12 inputs, piecewise linear
  in `pellet_h`) 0.9839, against 0.9181 linear. The lower edge is not set by the pellet height alone: in the
  band 13.25–13.45 mm, 35 of 53 rows are in the high state (notebook rule), and they have a higher linear heat
  rate, porosity, fuel roughness and cladding temperature (Mann–Whitney p from 0.000653 to 0.0236). This points
  to a temperature-dependent switch in the runs rather than an axial-mesh effect; the cause is not known.

## How these files relate to the paper

These are the BISON runs of the paper's fuel-performance case: the 13 inputs and 4 outputs match Sect. 3.4,
the input spreads match Table 2, and the means and relative sds of the four outputs over all 400 rows equal the
paper's p. 26 values at the printed precision. The paper trained and validated its DeepGP on 300 runs
(Table 3); pyMAISE distributes 400, and which 300 were used is not known. The paper fits DeepGP surrogates and
computes Morris and Sobol' sensitivities; the notebook estimates active subspaces from the runs and compares
the leading inputs with the paper's sensitivity statements (they agree for all four outputs), and its surrogate
fits with Table 8. One disagreement: the paper's Sobol' indices find the pellet height insignificant for the
surface temperature, while the stored runs carry a non-smooth pellet-height term that adds 0.0659 to the LOO R²
(notebook).

## License and credit

Data: from pyMAISE (P. Myers, C. Craig, V. Joynt, M. I. Radaideh et al.; University of Michigan), distributed
under the Apache License 2.0 (`pyMAISE_LICENSE.txt`, the text in the wheel; the wheel's metadata says "License:
UNKNOWN" and the licence file names no copyright holder; the package does not say whether the licence is meant
to cover the data). The Zenodo record that current pyMAISE downloads the same files from
(https://zenodo.org/records/20140559, "pyMAISE Benchmark Data", P. Myers, N. Panczyk, M. Radaideh, 2026) lists
`fp_inp.csv` and `fp_out.csv` and gives its licence as CC BY 4.0 (record metadata read on 2026-11-10; the
files were not downloaded). Runs by Radaideh and Kozlowski (2020), supported by the U.S. Department of Energy (award
DE-NE0008573, NEUP), with the BISON input file generated from the "PinToPerform" templates (p. 30). Cite the
paper and pyMAISE when using these files. `convert_data.py`, `check_accuracy.py` and the notebook: MIT (this
repository).

## Unknowns

- The PCMI benchmark nominal values (Table 2 gives only relative sds), the sampler and its seed; the notebook
  uses the sample moments.
- Which 300 of the 400 runs the paper used (Table 3), and whether the 400 were generated as one design.
- Whether any truncation at ±3σ was intended (p. 27); the stored samples reach 3.66σ.
- The BISON version, mesh and input file (PinToPerform template); the cause of the pellet-height jump of both
  temperatures, whose position also depends on power, porosity, roughness and cladding temperature.
- The values of `gap_thick` the runs used (stored rounded to 3 significant digits).
- Why the paper's Sobol' indices give the pellet height an insignificant effect on the surface temperature
  while the stored runs carry a pellet-height term worth 0.0659 of LOO R².

## Open questions

For the authors (not contacted): which 300 of the 400 pyMAISE runs were the 200 training + 100 validation
samples (Table 3, p. 18), and were the 400 generated in one design; the benchmark nominal values used and
whether any ±3σ truncation was applied; the BISON version and the PinToPerform template; and whether the
≈4 K jump of both fuel temperatures near a pellet height of 13.25–13.45 mm, whose position shifts with power,
porosity and cladding temperature, is a BISON model threshold or a switch of the location of the maximum.
