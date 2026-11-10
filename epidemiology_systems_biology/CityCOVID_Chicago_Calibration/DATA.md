# CityCOVID_Chicago_Calibration: data

Kind: **stored data**, converted from outside files. The values were computed by Robertson et al. (2025)
with CityCOVID, an agent-based COVID-19 model of Chicago developed at Argonne National Laboratory (not
public), and published in the authors' code repository
[sandialabs/Bayesian-calibration-of-stochastic-agent-based-model-via-random-forest](https://github.com/sandialabs/Bayesian-calibration-of-stochastic-agent-based-model-via-random-forest)
(MIT licence). They were not regenerated (the model is not available, and the design took 35,000 runs) and
are not altered: `convert_data.py` copies every number character for character into the CSV layout of this
repository.

## Files

| file | contents | role |
|---|---|---|
| `CityCOVID_Chicago_Calibration.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 562; outputs committed |
| `citycovid_inputs.csv` | 700 × 9: the nine stored parameter columns (four vary, five held); units in the second `#` line | authors' `data/citycovid_parameters.csv` |
| `citycovid_hospitalizations.csv` | 700 × 95: seed-averaged daily hospital census, persons, 2020-03-02 to 2020-06-04; row i matches row i of the inputs | authors' `data/citycovid_results.csv`, first 95 columns |
| `citycovid_deaths.csv` | 700 × 95: seed-averaged cumulative deaths, persons, same days | authors' `data/citycovid_results.csv`, last 95 columns |
| `observed_chicago.csv` | 70 × 3: day index (days since 2020-03-02), observed cumulative deaths, observed hospital census (`nan` = not recorded) | authors' `data/observed_chicago.csv` |
| `convert_data.py` | reads the three source files, checks their SHA-256, writes the four CSVs into `--out DIR`; `--check` compares with the shipped files | the conversion (standard library; numpy for `--check`) |
| `check_accuracy.py` | recomputes the numbers of this file that neither the notebook nor `convert_data.py` prints | check |
| `CityCOVID_calibration_LICENSE.txt` | the MIT licence of the authors' repository, kept beside the files taken from it | licence |

The CSVs are plain comma-separated numbers after two `#` lines (a description, then the column names with
units), so `np.loadtxt(path, delimiter=',')` reads them. Stored label → column name:
`Number of initial infected` → `n_initial_infected`, `Rate of exposure to infected` → `theta1_exposure_rate`,
`Seasonality multiplier` → `seasonality_multiplier`, `Shielding by susceptible` → `shielding_by_susceptible`,
`Proportion of isolated in homes` → `isolation_in_homes`, `Proportion of isolated in nursing homes` →
`isolation_in_nursing_homes`, `Time of initial seeding of infection` → `theta2_seeding_time`,
`Probability of stay at home` → `theta3_stay_at_home`, `Probability of protective behaviors` →
`theta4_protective_behaviors`; result columns `hospitalizations`/`deaths` with date `YYYY-MM-DD` →
`H_YYYY-MM-DD`/`D_YYYY-MM-DD`. The leading `instance` column (1–700) is dropped after `convert_data.py`
checks that it numbers the rows 1–700 in the same order in both source files.

The authors' `data/` folder also holds the IMABC runs (`abc_parameters.csv`, `abc_results.csv`) and the
pushforward runs of their calibrations (Bayesian, GP and Stein); they are not used here and not shipped.

## Model

**Citation.** C. Robertson, C. Safta, N. Collier, J. Ozik, J. Ray, "Bayesian calibration of stochastic agent
based model via random forest", *Statistics in Medicine* 44(6):e70029, 2025,
[doi:10.1002/sim.70029](https://doi.org/10.1002/sim.70029) (volume, issue and article number checked on
Crossref). Read in the authors' accepted manuscript (SAND2025-03333J, 25 pages); page numbers here are those
of the manuscript.

**Simulator** (p. 3). CityCOVID, built on ChiSIM and Repast HPC: 2.7 million agents of an age-stratified
Chicago population in 1.2 million locations, moving on hourly schedules; states susceptible, exposed,
presymptomatic, infected (asymptomatic and symptomatic), hospitalized, ICU, recovered and deceased;
probabilities of exposure, hospitalization and death; Gamma-distributed durations. One CityCOVID run in the
earlier IMABC calibration cost 12.8 core-hours (Table 5, p. 22).

**Inputs** (m = 4; Table 2, p. 19; nine Morris-screened parameters in Table 1, p. 18, reduced to four by a
random-forest sensitivity analysis, Appendix p. 24).

| symbol | column | description | units | Table 2 prior | stored range | source |
|---|---|---|---|---|---|---|
| θ1 | `theta1_exposure_rate` | rate of exposure to infected | hourly probability (p. 5) | U(0.046, 0.069) | 0.04601–0.06894 | Table 2, p. 19 |
| θ2 | `theta2_seeding_time` | time of initial seeding of infections | not stated (see below) | U(31, 59) | **31.07–132.71** | Table 2, p. 19 |
| θ3 | `theta3_stay_at_home` | probability of stay at home | probability | U(0.939, 0.981) | 0.93901–0.98093 | Table 2, p. 19 |
| θ4 | `theta4_protective_behaviors` | probability of protective behaviors | probability | U(0.407, 0.492) | 0.40712–0.49177 | Table 2, p. 19 |

Held in every row (values from the data; the paper names these parameters in Table 1 but gives no values):
number of initially infected 443.6, seasonality multiplier 0.211, shielding by susceptible 0.967, proportion
isolated in homes 0.471, proportion isolated in nursing homes 0.549.

**Seeding-time discrepancy.** Table 2 gives θ2 ~ U(31, 59), but 500 of the 700 stored rows (exactly rows
1–500) lie in (59, 133]; rows 501–700 lie in [31, 59). The notebook uses the **stored design's bounds
[31, 133]** for its main analysis (all 700 rows; the design is close to uniform on that interval, 7.14 rows
per unit below 59 and 6.76 above, printed by the notebook) and repeats the analysis on the 200 rows inside
Table 2's range with the Table 2 bounds. The paper does not mention the wider range.

**Seeding-time units.** The paper states none and says θ2 "is aligned on 24 hour boundaries" (p. 8). In the
data the deaths depend on θ2 through the nearest multiple of 24 (a fit with θ2 replaced by
floor((θ2 + 12)/24) has leave-one-out RMS residual 51.8 deaths against 92.2 for a smooth quadratic; notebook),
consistent with θ2 in hours rounded to whole days (1 to 6 days over the stored range). This is an inference.

**Quantity of interest** (this data set's choice; the paper calibrates the whole series). Seed-averaged
cumulative deaths in Chicago on 2020-06-04, the last stored day (`citycovid_deaths.csv`, last column); range
745.54–5665.00 over the design, observed value 2331.

**Gradients.** None. CityCOVID computes none, it is not public, and finite differences would need new
cluster runs; the subspace is estimated from values (QPHD, with an OLS check).

**Sampling design** (p. 4). 700 quasi-random Halton points, each simulated with 50 random seeds (35,000 runs)
and averaged over the seeds. Reconstructed from the data (notebook and `check_accuracy.py`): θ1, θ3, θ4 are
Halton coordinates in bases 7, 5 and 3, points 1–700, scaled to the Table 2 bounds (largest deviation of
order 1e-16); θ2 is the base-2 coordinate of points 1–500 scaled to [59, 133] for rows 1–500 and
31 + 28 h(j) for 200 distinct integers j between 1003 and 1400 (base-2 radical inverse h) for rows 501–700
(`check_accuracy.py`). So the 700 rows are a union of two θ2 blocks, not one Halton sequence over the Table 2
box. In rows 501–700 the θ2 values are therefore not the base-2 Halton points of the indices that give
θ1, θ3 and θ4; the largest absolute correlation between two of the four inputs in those rows is 0.016
(`check_accuracy.py`), so the mismatch induces no visible dependence. Every stored output is a multiple of 1/50 (notebook), as a mean of 50 integer counts is.

**Published results** (no active subspace). Random-forest importances and Sobol' indices of the surrogate
(Table 3, p. 20; total Sobol' θ1 0.56, θ2 0.33, θ3 0.09, θ4 0.01); 4 PCA components capture over 95 % of the
variance of the series (Fig. 3, p. 7); random-forest median absolute relative error < 5 % in 5-fold cross
validation (p. 7); a posterior with a strong positive θ1–θ3 correlation and an almost uniform θ4 marginal
(Fig. 4, p. 8); pushforward scores against IMABC (Table 6, p. 23).

## Regenerating

The runs cannot be regenerated: CityCOVID is not public, and each run is a cluster job. The CSVs are rebuilt
from the authors' repository, with a SHA-256 check of each source file:

```
git clone https://github.com/sandialabs/Bayesian-calibration-of-stochastic-agent-based-model-via-random-forest src
git -C src checkout e4c1e0e30e9e99381d3904fc238bcbf9444cce86
python convert_data.py src --out /tmp/citycovid            # writes the four CSVs there
python convert_data.py src --out /tmp/citycovid --check    # prints "same within tolerance" for the four files
```

`--out` is required and may not be this directory. `--check` compares every value with the shipped file
(absolute tolerance 1e-12, NaN equal to NaN); it ran in about 0.2 s on 2026-11-09 and printed "same within tolerance" for
all four files.

**Source.** Repository commit `e4c1e0e30e9e99381d3904fc238bcbf9444cce86` (2025-03-27). Files and SHA-256:
`data/citycovid_parameters.csv` (64,949 bytes,
`1d65d3f26674c3a46471e5f8e735d96541ae80a079a5845f76a2a7fa5af7f80d`), `data/citycovid_results.csv`
(1,087,033 bytes, `821c3acd0bfa749230933001de410a10f214122c96964abc91e6ec9f12ecf981`),
`data/observed_chicago.csv` (1,549 bytes,
`6b22b399db50262cab71c6cc885404548695e6e2edf4d1b909c5fa42cce0866e`), `LICENSE` (1,234 bytes,
`d338b72b1fa14b2e03ac6a1cc2c13faee7105cb0755fc21a76f0809ef5bf1a88`). The repository's README states that it
"contains hospitalization and death data produced by the CityCOVID agent based model".

The notebook runs in 5–8 s (nbconvert, 2026-11-09, shared machine).

## Accuracy

- Conversion: lossless (`convert_data.py --check`, above). Two stated layout rules: dates become day indices
  since 2020-03-02, and empty observed census entries become `nan`.
- Stored precision (`check_accuracy.py`): deaths written with at most 2 decimals; the census file holds the
  floating-point text of the authors' means (up to 16 decimals, e.g. `7.380000000000001`), all multiples of
  1/50 to within 1e-6 (notebook). θ2 takes 700 distinct values (`check_accuracy.py`), although the model
  appears to use only its nearest whole day.
- Seed noise: the outputs are means over 50 seeds; per-seed values are not stored, so the noise level of the
  QoI is not known. The smallest leave-one-out RMS residual among the notebook's fits is 51.8 deaths (the fit with θ2
  as a whole-day factor), so the seed-mean noise of the QoI is at most of that order; a quadratic in the one
  active variable leaves 123.2 deaths (notebook).
- No gradients, so no gradient check. The notebook's checks of the estimate: OLS and QPHD first directions
  agree to |w1·w1'| = 0.99999; 200 bootstrap replicates give a mean subspace distance of 0.006 for n = 1.
- Observed data (`check_accuracy.py`): deaths on 70 days (day 25 to 94), census on 64 days (day 31 to 94).

## How these files relate to the paper

The 700 input rows and seed-averaged series are, by the repository's description and the paper's p. 4, the
training data of the paper's random-forest surrogate; the stored θ2 range does not match Table 2 (above), so
either the surrogate was trained on a wider θ2 range than Table 2 states, or the paper's surrogate used a
subset; the paper does not say. The paper's analysis is a PCA plus random forest of the whole series and an
MCMC calibration; the notebook analyses one scalar of the stored runs directly with QPHD and compares its
direction with Table 3's importances and with the posterior correlation of Fig. 4. The notebook's squared
w1 components for 2020-06-04 deaths are 0.695 / 0.069 / 0.195 / 0.041 for θ1–θ4, which ranks θ2 third while
Table 3 ranks it second; the two are different quantities (the table does not say which output or PCA
coefficient its indices refer to).

**Overlap with other data sets of this repository.** `COVID19_SEIR_Epidemic_Peak` (an SEIR ODE) and
`COVID19_Exit_Strategies_virsim` (the virsim individual-based model) are other COVID-19 models with their own
runs; no runs are shared. This is the only city-scale agent-based model among them, and the only one whose
runs come with observed data.

## License and credit

Data: © the authors (Robertson, Safta, Collier, Ozik, Ray; National Technology & Engineering Solutions of
Sandia, LLC), from their repository under its MIT licence (kept beside the files as
`CityCOVID_calibration_LICENSE.txt`); the licence does not say whether it covers the data, so the four CSVs are
listed in the exceptions of `LICENSE-DATA`. The observed Chicago counts were collected by the Illinois National
Electronic Disease Surveillance System (p. 3) and are taken as stored by the authors. Cite the paper when using
these files. `convert_data.py`, `check_accuracy.py` and the notebook: MIT (this repository). No code of the
authors' repository is included.

## Unknowns

- The unit of θ2 (inferred as hours rounded to whole days; not stated in the paper).
- Why the stored θ2 design spans [31, 133] while Table 2 gives U(31, 59), and whether the paper's surrogate
  and calibration used all 700 rows.
- The origin of θ2 (from what time the seeding time is counted), and the CityCOVID version.
- The seed-to-seed spread of the outputs (only means are stored).
- Why the stored θ4 (probability of protective behaviours) raises deaths in the fitted direction; the paper
  does not define the parameter's direction of effect.

## Open questions

For the authors (not contacted): the θ2 range and units above; whether rows 1–500 and 501–700 were run as two
campaigns; and whether the paper's Table 3 indices were computed on all 700 rows.
