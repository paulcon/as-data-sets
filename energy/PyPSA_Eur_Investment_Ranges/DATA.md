# PyPSA_Eur_Investment_Ranges: data

Kind: **stored data**, converted from outside files. The values are the authors' runs of the PyPSA-Eur
capacity-expansion model (linear programs solved with Gurobi on a high-performance cluster), committed by
F. Neumann and T. Brown to https://github.com/fneum/broad-ranges (`results/`, CC BY 4.0). They were not
regenerated (the 1,515 high-fidelity runs took about 5 h and 20 GB each, p. 20) and are not altered:
`convert_data.py` copies the numbers into this repository's layout.

## Files

| file | contents | role |
|---|---|---|
| `PyPSA_Eur_Investment_Ranges.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2023106702; outputs committed |
| `least_cost_low.csv` | 500 × 14: the 5 cost multipliers and 9 outputs of the 500 least-cost runs of the low-fidelity model | `results/dataset_low.csv`, runs with objective `cost` |
| `least_cost_high.csv` | 15 × 14: the same for the 15 least-cost runs of the high-fidelity model | `results/dataset_high.csv`, runs with objective `cost` |
| `all_runs.npz` | all 50,500 low-fidelity and 1,515 high-fidelity runs (least-cost and near-optimal): per fidelity `*_inputs` (N × 5), `*_outputs` (N × 9), `*_epsilon`, `*_position` (NaN = none), `*_sense`, `*_objective`, `*_fixed` (int8 codes; labels in `sense_labels`, `objective_labels`, `fixed_labels`), plus `input_names`, `output_names`, `output_units` | both `results/dataset_*.csv` files |
| `pce_least_cost.csv` | 31 × 21: the authors' least-cost polynomial-chaos surrogates; columns: the exponents of each monomial $\prod_i f_i^{e_i}$ in the raw multipliers, then the coefficients of the low-fidelity fit and of the multi-fidelity surrogate for 8 outputs | `results/pce/polynomial-low-min-cost-0.0.txt`, `polynomial-high-min-cost-0.0.txt` |
| `convert_data.py` | reads the four source files of a clone of the authors' repository, checks their SHA-256 and writes the four files; `--check` compares | the conversion (numpy and the standard library) |
| `check_accuracy.py` | consistency checks of the stored runs, the annuity ratios of the cost ranges and the hydrogen CAPEX sums (see Accuracy) | check |

The CSV files are plain comma-separated numbers after two `#` lines (a description, then the column names with
units), so `np.loadtxt(path, delimiter=',')` reads them. Column names: `f_onwind, f_offwind, f_solar, f_H2,
f_battery` (source `onwind-cost`, …), then `onwind, offwind, wind, solar, H2, battery, transmission, tsc, gini`
(source names unchanged; `wind` = onwind + offwind). The source column `carrier` is empty in every run and is
not stored.

## Model

**Citation.** F. Neumann and T. Brown, "Broad ranges of investment configurations for renewable power systems,
robust to cost uncertainty and near-optimality", *iScience* 26(5):106702 (2023), [doi:10.1016/j.isci.2023.106702](https://doi.org/10.1016/j.isci.2023.106702)
(publisher PDF read, open access, CC BY; page numbers below are the journal's printed ones). Volume, issue,
article number and authors checked on Crossref. Of the other references, Hörsch et al. (2018) was checked on
Crossref; the rest were not, and their details are those printed in the paper's reference list and the DOIs
given in the notebook. Code: https://github.com/fneum/broad-ranges
(Snakemake workflow on PyPSA-Eur and PyPSA; MIT code, CC BY 4.0 results and data, CC0 for other files, by its
`.reuse/dep5`).

**Model** (pp. 13–17). PyPSA-Eur long-term investment planning: minimize the total annual system cost, annualized
capital costs of generators, storage and lines plus variable costs (Eq. 1, p. 13), subject to capacity potentials,
renewable availability, storage consistency, nodal balance, Kirchhoff's voltage law and line limits
(Eqs. 2–13, pp. 13–14); $C = \min\{c^\top x \mid Ax \le b\}$ (Eq. 14, p. 15). Fully renewable greenfield system
(onshore and offshore wind, solar, existing hydro not extendable, battery storage with 6 h and hydrogen storage
with 168 h at full power, pp. 4, 16), weather year 2013, perfect foresight for one year. Low-fidelity model:
37 nodes, 4-hourly, transport model; high-fidelity model: 128 nodes, 2-hourly, linearized load flow (p. 16).
Solver settings of the authors' `config.yaml`: Gurobi barrier, `BarConvTol` 1e-3, no crossover,
`noisy_costs: true` (a PyPSA-Eur option that perturbs cost coefficients slightly). Near-optimal problems
(Eqs. 15–17, p. 15): minimize or maximize one technology's total capacity subject to
$c^\top x \le (1+\epsilon) C$ for $\epsilon \in \{1, 2, 4, 6, 8\}\,\%$, and three technology pairs with the
second technology fixed at positions $\alpha \in \{0, 25, 50, 75, 100\}\,\%$ of its range at $\epsilon = 6\,\%$
(p. 15); with the least-cost problem, 101 problems per cost sample (p. 20).

**Inputs** (m = 5). Paper: uncertainty in the annuities of onshore wind, offshore wind, solar, hydrogen storage
and battery storage, independent and uniform between the optimistic and pessimistic DEA 2050 projections (Table 1,
p. 5; pp. 16–17). Stored: capital-cost multipliers $f_i$, uniform on $[\ell_i, 1]$ by `config.yaml`
(`uncertainties`); the workflow passes them to PyPSA-Eur as `onwind+c0.732` etc. That $f_i = 1$ is the
pessimistic annuity is an inference: the stored lower bounds equal the optimistic/pessimistic annuity ratios
rebuilt from the authors' `data/costs_optimist.csv` and `data/costs_pessimist.csv` within 0.002 for four
technologies (`check_accuracy.py`; hydrogen differs, see Unknowns).

| name | technology | units | stored range ($\ell_i$, 1) | Table 1 CAPEX, EUR/kW (p. 5) | CAPEX ratio |
|---|---|---|---|---|---|
| f_onwind | onshore wind | 1 | [0.670, 1] | 800–1190 | 0.672 |
| f_offwind | offshore wind | 1 | [0.726, 1] | 1420–1950 | 0.728 |
| f_solar | solar PV (half utility, half rooftop, p. 17) | 1 | [0.679, 1] | 420–620 | 0.677 |
| f_H2 | hydrogen storage (electrolysis, cavern, fuel cell) | 1 | [0.429, 1] | 668–2002 | 0.334 |
| f_battery | battery storage (with 6 h of energy) | 1 | [0.240, 1] | 316–1306 | 0.242 |

The ratios are printed by the notebook. For hydrogen the stored lower bound differs from the CAPEX ratio; see
Unknowns.

**Quantities of interest** (units from the source's `scripts/collect.py`). `tsc`: total annual system cost,
bn EUR/a (capital and marginal costs of all components). Capacities, GW: `onwind`, `offwind` (AC and DC
connected), `wind`, `solar`, `H2` and `battery` (power capacity of the storage units). `transmission`:
$\sum$ length × capacity of AC lines and DC links / $10^6$, TWkm. `gini`: a Gini coefficient of the spatial
distribution (dimensionless; not used in the paper or the notebook; its levels differ between the two models,
0.04–0.36 against 0.61–0.64). The paper's surrogates use the first eight (p. 17). The notebook's main QoI is
`tsc` of the least-cost runs.

**Gradients.** None are stored. The notebook estimates them from values (local linear fits on 8 nearest
neighbours; a global quadratic), derives them from the envelope theorem of the linear program
($\partial C/\partial f_i$ = capital cost of technology $i$ at $f_i = 1$, with one average annuity per GW
estimated by a linear fit; an inference of this data set, not from the paper), and differentiates the authors'
stored PCEs exactly. Coordinates: all gradients are passed to `active_subspaces` in the physical multipliers
and normalized by the input space ($z_i = (2f_i - \ell_i - 1)/(1 - \ell_i)$).

**Sampling design.** A Halton sequence (chaospy rule `halton`, p. 19 and `config.yaml`) of 500 points in the
box, rounded to three decimals (`config.yaml`; all stored multipliers lie on a 0.001 grid); the high-fidelity
runs use the first 15 points (rows 0–14 of `least_cost_low.csv`; Eq. 29, p. 19). It is a deterministic
low-discrepancy design, not a random sample. Run counts in `all_runs.npz`, low fidelity: 500 least-cost,
5,000 each for onwind, solar, battery and transmission, 10,000 each for offwind, wind and H2 (these include the
fixed-pair runs); high fidelity: 15 and 1,500 near-optimal.

**Published results.** No active subspace. Least-cost distributions (Fig. 3, p. 6; cost 160–220 bn EUR/a, p. 2),
own-cost sweeps (Fig. 4, p. 7), first-order and total Sobol' indices from the multi-fidelity PCE (Fig. 5, p. 8;
system cost first-order 71, 2, 4, 16, 5 % for onwind, offwind, solar, H2, battery), surrogate cross-validation
(Fig. 2, p. 4), near-optimal corridors (Figs. 6–7). The PCEs: total degree 3, Lasso penalty 0.005 (Eq. 25,
p. 18), 400 training and 100 validation runs (p. 20), additive linear high-fidelity correction (Eqs. 27–30,
p. 19).

## Regenerating

The runs cannot be regenerated from this repository: they need PyPSA-Eur (the version pinned as a submodule of
the authors' repository), Gurobi, the PyPSA-Eur input data and a cluster (about 5 min and 3 GB per
low-fidelity run, 5 h and 20 GB per high-fidelity run, p. 20; 50,500 + 1,515 runs). The external run, for the
record: `git clone --recursive https://github.com/fneum/broad-ranges`, `mamba env create -f environment.yaml`,
set `enable: collect_samples: true` in `config.yaml`, and `snakemake -call build_all_surrogates` (the
repository README); not tried here.

The files of this data set are regenerated from the authors' repository:

```
git clone https://github.com/fneum/broad-ranges && git -C broad-ranges checkout b95bbc98
python convert_data.py broad-ranges --out /tmp/pypsa                # writes the four files to /tmp/pypsa
python convert_data.py broad-ranges --out /tmp/pypsa --check        # ... and compares with the shipped files
```

`--out` is required and must be a folder outside this directory (the script refuses this directory), so the shipped files are never overwritten.
`--check` compares every value with a relative tolerance of 1e-15 and says whether the CSV text is identical;
run on 2026-11-09 in the Linux container where the shipped files were written, on a clone at commit b95bbc98,
it printed "same within tolerance" for all four files and "text identical: True" for the three CSVs, in about
1.4 s. The authors' files are not in this repository, so the check was not repeated in `.venv`;
`check_accuracy.py` runs there in under 1 s and prints the numbers quoted under Accuracy.

**Source.** Commit b95bbc98 (2023-04-14) of https://github.com/fneum/broad-ranges; the four files are unchanged
since tag v0.1.0 (2022-06-14), the version archived at Zenodo as
[doi:10.5281/zenodo.6642651](https://doi.org/10.5281/zenodo.6642651) (the repository's `CITATION.cff`); the
paper cites [doi:10.5281/zenodo.6641551](https://doi.org/10.5281/zenodo.6641551). The Zenodo archives were not
opened. The data files are plain CSV in git (not git-lfs):

| source file | bytes | SHA-256 |
|---|---|---|
| `results/dataset_low.csv` | 11,301,332 | `a1087ace71e403bb7e67b8c3c45217def3ad00585f695b98ac07a013f2640f30` |
| `results/dataset_high.csv` | 336,730 | `d1147a893048f4ae6877b99aebcc7d8dc99431b3727c73c881851a93b7ad7ca5` |
| `results/pce/polynomial-low-min-cost-0.0.txt` | 2,597 | `0cb4c12ae5f1edd5a34ec30219f72cc6aebc3237d7ca35ab460d0395c84ff7fe` |
| `results/pce/polynomial-high-min-cost-0.0.txt` | 2,598 | `5d6a6f7082c80c60521116743916c5b81f16e6495153aa380e2fd139b9f7066d` |

**What the conversion does.** The source CSVs are transposed (one row per field, one column per run). The least-cost
CSVs copy each value's text unchanged. In `all_runs.npz` each numeric field is float64; `convert_data.py`
stops unless every source string is the shortest repr of its float64, so the original text is recovered by
`repr`. Text fields become int8 codes with their labels. The polynomial files (numpoly text format) are
decoded: each 5-character key gives the exponents of variables q0..q4, which are the multipliers in the order of
`config.yaml` (onwind, offwind, solar, H2, battery); the coefficient text is copied unchanged. The notebook
confirms the order: the stored low-fidelity polynomial predicts the 500 runs with $R^2 = 0.9872$.

## Accuracy

- **Noise in the stored cost.** The runs are solved to a barrier tolerance of $10^{-3}$ without crossover
  (`config.yaml`). A linear fit of `tsc` on the envelope-theorem terms leaves residuals with sd
  0.308 bn EUR/a (max 1.180; $R^2$ = 0.99925) against a cost sd of 11.219 bn EUR/a, and a cubic in all five
  inputs has LOO $R^2$ = 0.9998 (both printed by the notebook).
- **Cost constraint of the near-optimal runs** (`check_accuracy.py`). $\mathrm{tsc}/((1+\epsilon)\,\mathrm{tsc}_{lc}) - 1$
  has median 1.84e-05 and maximum 2.50e-03 over the 50,000 low-fidelity near-optimal runs (4 above 1e-3), and
  median 1.84e-05, maximum 1.00e-02 over the 1,500 high-fidelity ones (24 above 1e-3): a few runs exceed the
  cost bound, by up to 1 % in the high-fidelity model.
- **Ordering** (`check_accuracy.py`). No minimized capacity exceeds, and no maximized capacity falls below, its
  least-cost value (largest relative violation 0 in both models).
- **Consistency of the files** (`check_accuracy.py`): both least-cost CSVs equal the least-cost rows of
  `all_runs.npz`.
- **Annuity ratios and hydrogen CAPEX sums** (`check_accuracy.py`): see Inputs and Unknowns.
- **PCE gradients** (notebook): the analytic gradient of the stored multi-fidelity polynomial matches central
  differences to a relative error of order 1e-10.

## How these files relate to the paper

They are the paper's own runs (the training data of its surrogates), not a regeneration. The paper's figures
come from the multi-fidelity surrogates; the stored multi-fidelity polynomial reproduces the first-order
Sobol' indices of Fig. 5A to within 1–8 percentage points over the eight outputs (3 points for system cost;
notebook). The differences are larger than rounding: the workflow splits training and test runs with a random
shuffle and no stored seed (`config.yaml`, `scripts/surrogate.py`), so the stored polynomials may be a refit.

## License and credit

Data: CC BY 4.0, from the authors' repository (`results/*` is CC-BY-4.0 by its `.reuse/dep5`); credit
F. Neumann and T. Brown, https://github.com/fneum/broad-ranges, and the paper. No code of the repository is
copied (`convert_data.py` and the notebook are new, MIT). No file of this data set needs an entry in the
exceptions list of `LICENSE-DATA`.

## Unknowns

- **Hydrogen multiplier range.** The stored lower bound 0.429 differs from the Table 1 CAPEX ratio 668/2002 =
  0.334. The other four bounds are within 0.003 of their CAPEX ratios. The paper says the ranges are of annuities
  (p. 17) but does not give them. Annuities rebuilt from the authors' cost files (`check_accuracy.py`, this data
  set's reconstruction) give optimistic/pessimistic ratios 0.672, 0.728, 0.679, 0.481 and 0.240 for onwind,
  offwind, solar, H2 and battery: they match the stored bounds within 0.002 except for hydrogen (0.481 against
  0.429), so how 0.429 was obtained is not known. A possible lead (`check_accuracy.py`): Table 1's pessimistic
  hydrogen CAPEX 2002 EUR/kW equals electrolysis 800 + fuel cell 900 + 168 h of cavern at 1.8 EUR/kWh
  (2002.4), but 1.8 EUR/kWh is the cavern cost in `data/costs_optimist.csv`; the pessimistic file has
  1.0 EUR/kWh (sum 1868.0). The cavern costs may be swapped between the two files. Table 1's optimistic 668
  matches neither optimistic sum (1002.4 with the file's 1.8, 868.0 with 1.0). This does not change the
  stored data.
- **What $f_i$ multiplies.** By inference, the whole PyPSA-Eur capital cost of the technology, including the
  grid connection of offshore wind; not stated in the paper.
- **Stored polynomials vs. Fig. 5.** Whether the stored PCEs are the fit behind the figures (see above).
- **Gini coefficient.** The definition (`scripts/collect.py`) is not described in the paper; not used.
- **Zenodo archive.** Whether doi:10.5281/zenodo.6641551 (paper) and 10.5281/zenodo.6642651 (repository) hold
  byte-identical copies of the four files was not checked.

## Open questions

For the authors (not contacted): how the hydrogen-storage multiplier range [0.429, 1] was derived from Table 1,
and whether the cavern costs (1.8 and 1.0 EUR/kWh) are swapped between `data/costs_optimist.csv` and
`data/costs_pessimist.csv`;
whether the stored `polynomial-*-min-cost-0.0.txt` are the surrogates behind Fig. 5.
