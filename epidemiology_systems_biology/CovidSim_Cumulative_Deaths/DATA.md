# CovidSim_Cumulative_Deaths: data

Kind: **stored data** (converted from outside files). The 3000 CovidSim runs were made by W. Edeling on the
Altair supercomputer (Poznan Supercomputing and Networking Center) and published in the MIT-licensed
repository [wedeling/deep_active_subspace_data](https://github.com/wedeling/deep_active_subspace_data)
(archived as [doi:10.5281/zenodo.7009160](https://doi.org/10.5281/zenodo.7009160), the paper's ref. [15]).
They were not regenerated (about 14,000 core-hours, p. 81) and are not altered: `convert_data.py` writes the
authors' HDF5 values into CSV/npz files without changing any value.

## Files

| file | contents | role |
|---|---|---|
| `CovidSim_Cumulative_Deaths.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with `SEED = 132`; outputs committed |
| `covidsim_inputs.csv` | 3000 × 51: the uniform inputs of every run, physical units (names and units in the second `#` line), column order of `vary.py` | HDF5 `params` |
| `covidsim_input_ranges.csv` | 2 × 51: lower (row 1) and upper (row 2) bound of each uniform input | `cp.Uniform` entries of `COVID19/vary.py` |
| `covidsim_deaths_day801.csv` | 3000 × 1: cumulative deaths after 801 simulated days (persons), row i matches row i of the inputs | HDF5 `samples[:, -1]` |
| `covidsim_cumulative_deaths_daily.npz` | `deaths` (3000 × 801, float64, cumulative deaths per daily output; integer-valued) and `day` (1..801, the output column number) | HDF5 `samples`, unchanged |
| `convert_data.py` | reads the HDF5 file and `vary.py` (as text, never imported), checks their SHA-256, writes the four data files; `--check` compares | the conversion (numpy, h5py) |
| `check_accuracy.py` | prints the numbers of the Accuracy section that the notebook and `convert_data.py` do not print | check (numpy; h5py for `--src`) |
| `LICENSE-deep_active_subspace_data.txt` | the MIT licence of the source repository (Copyright (c) 2021 Wouter Edeling) | licence |

The CSVs are plain comma-separated numbers after two `#` lines (a description, then the column names with
units), so `np.loadtxt(path, delimiter=',')` reads them. Input values are written with Python's shortest
round-trip representation, so they read back to the same float64 values.

The source repository also holds the HIV data of the paper's Sec. 8.1 (not used here: the HIV model is
already `epidemiology_systems_biology/HIV_Long_Term_Dynamics`), the authors' notebooks, and the replica
error file `COVID19/errors_n_neurons10.hdf5` behind Fig. 15 (not shipped; `check_accuracy.py --src` reads it).

## Model

**Citation.** W. Edeling, "On the deep active-subspace method", *SIAM/ASA Journal on Uncertainty
Quantification* 11(1):62–90, 2023, doi:10.1137/21M1463240 (publisher PDF read). Single author: Centrum
Wiskunde & Informatica, Scientific Computing Group, Amsterdam. The paper applies the deep active-subspace
(DAS) method of Tripathy and Bilionis (2019) to an HIV model (Sec. 8.1) and to CovidSim (Sec. 8.2, pp. 80–82);
only the CovidSim case is used here.

**Simulator** (p. 80). CovidSim, the individual-based COVID-19 code of the MRC Centre for Global Infectious
Disease Analysis, Imperial College London (paper ref. [18], Ferguson et al. 2020). Households, schools,
universities and workplaces (place types 0–3) and a spatial contact kernel; no equations are given in the
article. Code version not stated.

**The population modelled: what is and is not known.** Known: the runs are individual-based, with people in
households and in four place types (elementary school, high school, university, workplaces: place types 0–3,
named in the comments of the authors' `vary.py`) linked by a spatial contact kernel (p. 80); the 51 inputs
include closure, quarantine, case-isolation and social-distancing parameters (inputs 1–10), so an intervention
scenario was active; and the simulated epidemic produces between 27,659 and 142,155 cumulative deaths after 801
days (median 69,999; notebook). Not known: the country or region, the population and density files, the
population size and age structure, the intervention scenario and its timing, and the values of the inputs not
varied. The article (pp. 80–82) does not state them, and the authors' repository (commit 880745c: README,
`vary.py`, the notebook, `recompute_COVID_errors.py` and the HDF5 attributes, which are empty) names no country
or population file. The article cites an earlier CovidSim study of the same group (ref. [16], Edeling et al.,
*Nature Computational Science* 1:128–135, 2021, 19 inputs) as a previous study of "this model"; that study's
setting is not evidence for the setting of these runs, so this data set assumes none.

**Inputs** (m = 51, uniform; p. 80: "D = 51 continuous input parameters, for which we prescribe uniform input
distributions"). The article's table of inputs, defaults and ranges is in its SIAM supplement
(`supplement.pdf`, not read). The ranges here are the `cp.Uniform` entries of the authors' `COVID19/vary.py`,
the EasyVVUQ sampling dictionary; its 51 active entries match the 51 HDF5 columns in order (every stored value
inside its range; the samples span 99.80–99.99 % of each range, `check_accuracy.py`).

| no. | name(s) | group (`vary.py` comment) | units | range |
|---|---|---|---|---|
| 1 | Relative_household_contact_rate_after_closure | intervention | 1 | [1.2, 1.8] |
| 2 | Relative_spatial_contact_rate_after_closure | intervention | 1 | [1.0, 1.5] |
| 3 | Relative_household_contact_rate_after_quarantine | intervention | 1 | [1.6, 2.4] |
| 4 | Residual_spatial_contacts_after_household_quarantine | intervention | 1 | [0.2, 0.3] |
| 5 | Household_level_compliance_with_quarantine | intervention | 1 | [0.5, 0.9] |
| 6 | Individual_level_compliance_with_quarantine | intervention | 1 | [0.9, 1.0] |
| 7 | Proportion_of_detected_cases_isolated | intervention | 1 | [0.6, 0.8] |
| 8 | Residual_contacts_after_case_isolation | intervention | 1 | [0.2, 0.3] |
| 9 | Relative_household_contact_rate_given_social_distancing | intervention | 1 | [1.1, 1.5] |
| 10 | Relative_spatial_contact_rate_given_social_distancing | intervention | 1 | [0.15, 0.35] |
| 11 | Symptomatic_infectiousness_relative_to_asymptomatic | disease | 1 | [1, 2] |
| 12 | Proportion_symptomatic | disease | 1 | [0.4, 0.8] |
| 13 | Latent_period | disease | day (inferred) | [3, 6] |
| 14 | Mortality_factor | disease | 1 | [0.8, 1.2] |
| 15 | Household_attack_rate | disease | 1 | [0.1, 0.19] |
| 16 | Household_transmission_denominator_power | disease | 1 | [0.7, 0.9] |
| 17 | Delay_from_end_of_latent_period_to_start_of_symptoms | disease | day (inferred) | [0, 1.5] |
| 18–21 | Relative_transmission_rates_for_place_types0–3 | disease | 1 | [0.08, 0.15], [0.08, 0.15], [0.05, 0.1], [0.05, 0.07] |
| 22 | Relative_spatial_contact_rates_by_age_power | disease | 1 | [0.25, 4] |
| 23–24 | Proportion_of_places_remaining_open_after_closure_by_place_type2, 3 | spatial | 1 | [0.2, 0.3], [0.8, 1.0] |
| 25–28 | Residual_place_contacts_after_household_quarantine_by_place_type0–3 | spatial | 1 | [0.2, 0.3] each |
| 29–32 | Relative_place_contact_rate_given_social_distancing_by_place_type0–3 | spatial | 1 | [0.8, 1.0], [0.8, 1.0], [0.6, 0.9], [0.6, 0.9] |
| 33 | Relative_rate_of_random_contacts_if_symptomatic | spatial | 1 | [0.4, 0.6] |
| 34–37 | Relative_level_of_place_attendance_if_symptomatic0–3 | spatial | 1 | [0.2, 0.3], [0.2, 0.3], [0.4, 0.6], [0.4, 0.6] |
| 38 | Kernel_scale | leftovers | m (inferred) | [3600, 4400] |
| 39 | Kernel_Shape | leftovers | 1 | [2.4, 3.6] |
| 40–43 | Kernel_shape_params_for_place_types0–3 | leftovers | 1 | [2.4, 3.6] each |
| 44–47 | Kernel_scale_params_for_place_types0–3 | leftovers | m (inferred) | [3600, 4400] each |
| 48–51 | Proportion_of_between_group_place_links0–3 | leftovers | 1 | [0.2, 0.3] each |

The notebook's group labels ("disease" for 11–22, "spatial / place" for 23–37, "kernel / place links" for
38–51) follow the `vary.py` comments, which place input 22 under "Disease parameters". Several bounds are
written in `vary.py` as nominal × 0.8 and × 1.2 (or × 0.9 and × 1.1), e.g. `cp.Uniform(1.5*0.8, 1.5*1.2)`,
so they carry the floating-point products (1.2000000000000002 for 1.5 × 0.8); the stored bounds are those
products. Units: neither the article nor `vary.py` gives units; days and metres follow the CovidSim
parameter-file conventions (inference). Eleven entries of `vary.py` are commented out (four discrete
quarantine and isolation delays and durations, `Reproduction_number`, `Infectious_period`, four place group
sizes, `CLP1`): not varied, values not recorded.

**Quantity of interest** (p. 81). "the predicted cumulative death count after a simulated time of 801 days";
Fig. 14 caption: "the final cumulative death count after 801 days". Persons. The HDF5 file holds 801 daily
outputs per run; the QoI is the last (the authors' notebook and `recompute_COVID_errors.py` use
`samples[:, -1]`). The first column is 0 in every run; whether column k is simulated day k or day k − 1 is not
documented (the `day` array of the npz is the column number 1..801). All runs are non-decreasing in time and
integer-valued; at day 801 the count ranges from 27,659 to 142,155 (median 69,999).

**Gradients.** None. CovidSim computes none; DAS is gradient-free; finite differences would cost 52 runs
(about 240 core-hours at 28 cores × 10 min per run) per gradient.

**Sampling design** (p. 81). 3000 Monte Carlo samples from an EasyVVUQ design (independent uniform draws; seed
not reported), run with FabSim3 and QCG-PilotJob, 28 cores and about 10 minutes per run, about 14,000
core-hours. In the data every input passes a Kolmogorov–Smirnov test against its uniform distribution at 5 %
(largest statistic 0.0244 against the critical value 0.0248) and the largest sample correlation between two
inputs is 0.0583 (`check_accuracy.py`). No run failed or was removed (3000 distinct rows).

**Published results.** DAS eigenvalues of $\overline{C}_{ANN}$ (eq. (4.5), p. 70) for a $d = 5$ network,
Fig. 13 (p. 81): a log plot with no printed values, read here as about 2.95, 0.219, 0.052, 0.041, 0.020
(external, see Accuracy); text: "a small gap after $\lambda_1$ … there is also a smaller gap between
$\lambda_2$ and $\lambda_3$"; $d = 2$ chosen because "$d = 1$ yielded relatively high training and test
errors" (p. 81). Fig. 14 (p. 82): contours of the DAS prediction and of the 1500 validation runs in the
DAS coordinates $y = W_1^\top x$. Fig. 15 (p. 82): relative error $e$ of eq. (8.2) (p. 75) of DAS, constrained
and unconstrained ANNs over 100 replica networks for training sizes 1500–2700. No eigenvectors, input
rankings or subspace angles are published for CovidSim ("We do not have a reference active-subspace solution
for CovidSim", p. 81). The authors' repository notebook adds a derivative-based input ranking of the DAS
network, marked "not in article".

## Regenerating

The runs cannot be regenerated from the repository (a supercomputer campaign of about 14,000 core-hours).
The data files are regenerated from the authors' files:

```
git clone https://github.com/wedeling/deep_active_subspace_data     # commit 880745c6d5a3d1e7cd727ad635af91f84b441f14 (2024-06-05)
pip install h5py                                                      # tested with h5py 3.16.0, numpy 2.x
python convert_data.py deep_active_subspace_data/COVID19 --out DIR           # writes the four files into DIR
python convert_data.py deep_active_subspace_data/COVID19 --check --out DIR   # DIR outside this directory
```

`--out` is required (the script never writes into this directory unless it is named explicitly), and
`--check` refuses an `--out` inside this directory. `--check` compares every value with the shipped files at
a relative tolerance of 1e-15; on 2026-11-09 it printed a maximum relative difference of 0.0e+00 for all
five arrays, in under 2 s. Source files and checksums:

| file (folder `COVID19/`) | size | SHA-256 |
|---|---|---|
| `covidsim_samples_only_cont.hdf5` | 20,449,672 bytes | `09182744645145a4c1769b2ae92c4d12a474f12d8a89052d7c8210d69226935f` |
| `vary.py` | 5,244 bytes | `5b3611b7ef7f6f2cefeff0af071b786d84abb7409d54a70d17e854db8c9f785d` |
| `errors_n_neurons10.hdf5` (not converted; read by `check_accuracy.py --src`) | 49,944 bytes | `26d0e1b4e4e7f0d9d5181a947e50238112b678734fc6b20d11bffe7f91de180f` |

The HDF5 file is kept out of this directory; the npz holds all of its values (5.6 MB compressed). The
Zenodo archive 10.5281/zenodo.7009160 ("Final Revision of DAS Article") was not compared with the GitHub
commit used here.

The notebook runs in about 4–17 min (264–1019 s in a Linux container with 2 CPUs shared with other jobs; 1019 s for the committed run under heavy load); most
of it is the 200 QPHD bootstrap replicates (2–6 min) and the 30 polynomial-ridge fits (2–10 min).

## Accuracy

`check_accuracy.py` prints the numbers marked with its name, in under 1 s.

- Conversion: lossless (`convert_data.py --check`, above).
- Stored precision (`check_accuracy.py`): inputs float64 as stored by the authors; death counts integer-valued
  in all 2,403,000 daily values, so the QoI has a resolution of one death against a standard deviation of
  19,848.9 (notebook).
- Simulator noise: CovidSim is stochastic, and whether the 3000 runs used one random seed or different seeds is
  not reported; there are no replicate runs, so the run-to-run noise cannot be measured. A cubic in the two
  QPHD active variables leaves 0.0437 of var(f) (notebook), an upper bound on the noise share.
- No gradients, so no gradient check.
- Published DAS errors (`check_accuracy.py --src`, from the authors' `errors_n_neurons10.hdf5`): mean training
  / test error $e$ over 100 replicas 5.42 % / 5.84 % at 1500 training runs and 5.57 % / 4.95 % at 2700 (DAS);
  constrained ANN 5.48 % / 5.91 % and 5.60 % / 4.96 %; unconstrained ANN 3.12 % / 7.34 % and 3.75 % / 5.26 %.
  These are the values plotted in Fig. 15.
- Fig. 13 eigenvalues (2.95, 0.219, 0.052, 0.041, 0.020): read from the published PDF rendered at 400 dpi,
  marker centres against the log-axis ticks (decade = 359 px); external run, not reproducible from the
  repository (the article PDF is not shipped). The reading error is about ±2 px, i.e. ±1.3 % in each value.
- Mortality_factor: correlation with the deaths 0.0316, linear coefficient 135.4 deaths per half-range
  against the largest coefficient 26,870.5 (`check_accuracy.py`); see Unknowns.

## How these files relate to the paper

The files are exactly the 3000 runs of Sec. 8.2 (Figs. 13–15), as published by the author for reproducing
those figures. The paper trains DAS networks; the notebook estimates the subspace from the same values with
QPHD (checked with OLS, OPG, two permutation nulls, and a polynomial ridge function fitted and tested on
random halves), chooses $n = 2$ as the paper does, and compares eigenvalue ratios (Fig. 13), errors (Fig. 15,
by eq. (8.2)), the death range (Fig. 14) and the input ranking of the authors' notebook. The notebook
agrees with the paper on the gap after $\lambda_1$, on $\lambda_2/\lambda_1$, on the chosen dimension and on the
input ranking; the later ratios differ. Its $\lambda_2/\lambda_1$ 0.064 and $\lambda_1/\lambda_2$ 15.7 are close to
Fig. 13 (about 0.074 and 13.5); its $\lambda_2/\lambda_3$ (3.0) is smaller than in Fig. 13 (about 4.2), and its
$\lambda_3/\lambda_4$ (2.5) about twice Fig. 13's (about 1.3). DAS eigenvalues come from a network on standardized outputs and the QPHD
eigenvalues from a quadratic on deaths, so only ratios are compared. Fig. 15's legend labels the constrained
network "ANN (d=1)" although the text and the authors' `recompute_COVID_errors.py` use $d = 2$.

## License and credit

The data are W. Edeling's, from the MIT-licensed repository wedeling/deep_active_subspace_data (Copyright (c)
2021 Wouter Edeling; licence text in `LICENSE-deep_active_subspace_data.txt`), and are hosted here under that
licence, not CC BY 4.0 (listed in the exceptions of `LICENSE-DATA`). Credit: W. Edeling, SIAM/ASA JUQ 11:62–90
(2023), doi:10.1137/21M1463240; data doi:10.5281/zenodo.7009160. CovidSim is the code of the MRC Centre for
Global Infectious Disease Analysis, Imperial College London (https://github.com/mrc-ide/covid-sim); none of
it is in this directory. The notebook, `convert_data.py` and `check_accuracy.py` are MIT (repository licence).

## Unknowns

- Simulated country or region, population files, population size and intervention scenario: not stated in
  the article or the authors' repository (see Model); this data set assumes none.
- CovidSim version, parameter files and held values of the 11 commented-out `vary.py` entries; the defaults
  of the 51 inputs (SIAM supplement, not read).
- Random seeds of the runs (fixed or varied), hence the simulator noise level.
- Units of the inputs (days and metres inferred).
- Whether output column k is simulated day k or k − 1.
- Why Mortality_factor (input 14, U[0.8, 1.2]) has almost no effect on cumulative deaths (correlation 0.0316):
  it may scale a quantity other than the death output, or apply to a subset of cases.
- The training/validation split of the paper's Fig. 14 and Fig. 15 runs is partly knowable: the authors'
  `COVID19.ipynb` trains with `test_frac = 0.5` and its contour cells plot `samples[n_train:]` as the
  validation data, i.e. the second half (the last 1500) of the stored rows, provided EasySurrogate does not
  shuffle the rows before splitting (not checked). The article does not state the split.
- Where the runs were made: the article (p. 81) says the Altair supercomputer (PSNC); the authors' repository
  notebook (`COVID19.ipynb`, markdown of cell 6) says the "Eagle" supercomputer. This data set follows the
  article.

## Open questions

For the author (not contacted): which country, population file and intervention scenario the 3000 runs
simulate; whether a fixed random seed was used; what CovidSim's `Mortality_factor` scales in this setup;
whether the supplement's defaults and ranges equal `vary.py`; whether the 50 % validation set of Figs. 14–15
was the last 1500 stored rows; whether the runs were made on Altair (article) or Eagle (repository notebook).
