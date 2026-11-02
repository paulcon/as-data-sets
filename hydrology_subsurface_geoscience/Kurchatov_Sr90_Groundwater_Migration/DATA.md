# Kurchatov_Sr90_Groundwater_Migration: data

Kind: **stored data**. The 300 runs were made by the paper's authors with the groundwater code
MARTHE (BRGM, France). The code and the site model are not available ("The computer code is no
longer accessible", VLSE page), so no new runs and no gradients can be computed. The runs are
those that the Virtual Library of Simulation Experiments (VLSE) hosts as its "MARTHE" data set.
`marthedata.txt` is the VLSE file, byte for byte. `inputs.csv` and `outputs.csv` are a lossless,
scripted conversion of it. No value has been changed, and no row has been dropped.

## Files

| file | contents | role |
|---|---|---|
| `Kurchatov_Sr90_Groundwater_Migration.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 85; outputs committed |
| `marthedata.txt` | 301 lines: a tab-separated header line, then 300 runs × 30 columns (20 inputs, then 10 outputs); 81,934 bytes, SHA-256 `2c0981a73bf4b2641db19d1b6727113918569f90d5ac51488e869d5a263a39c7` | stored data, as downloaded |
| `inputs.csv` | 300 × 20: per1, per2, per3, perz1–perz4, d1–d3, dt1–dt3, kd1–kd3, poros, i1, i2, i3, physical units, the number strings of `marthedata.txt` unchanged | stored data (converted) |
| `outputs.csv` | 300 × 10: Sr-90 concentration (Bq/l) at the end of 2010 at wells p102K, p104, p106, p2.76, p29K, p31K, p35K, p37K, p38, p4b, same rows | stored data (converted) |
| `convert_data.py` | writes the two CSVs from `marthedata.txt` (checks its SHA-256 first); `--check` verifies that the shipped CSVs are byte-identical to a fresh conversion and that every value and number string equals the source | conversion and check |

The CSVs are plain comma-separated numbers after two `#` lines (a description, and the column
names with units), so `np.loadtxt(path, delimiter=',')` reads them. `[n.r.]` marks a unit the paper
and the VLSE page do not report.

## Model

Citation: E. Volkova, B. Iooss, F. Van Dorpe, "Global sensitivity analysis for a numerical model of
radionuclide migration from the RRC 'Kurchatov Institute' radwaste disposal site", *Stochastic
Environmental Research and Risk Assessment* 22(1):17–31, 2008 (online 23 November 2006),
doi:10.1007/s00477-006-0093-y (publisher PDF read). Page numbers below are the journal's.

- **Site** (Sec. 3.1, pp. 22–23): about 2 ha on the territory of the RRC "Kurchatov Institute",
  Moscow, with eleven temporary radioactive-waste repositories used from 1943 to 1974. Sr-90 is the
  main groundwater contaminant. The upper moraine aquifer (unconfined) has 20 observation wells.
- **Model** (Secs. 3.2–3.3, p. 23): MARTHE three-dimensional transient groundwater flow and
  convection–dispersion transport of Sr-90 with sorption and radioactive decay in the upper moraine
  aquifer; three layers; uniform 5 m × 5 m planar mesh oriented south-west along the flow; domain
  250 m × 350 m (Fig. 2); four low-conductivity zones in the second layer and zones of high
  infiltration from pipe leaks (Fig. 4). Initial plume: the one observed in 2002; prediction to the
  end of 2010 (Fig. 3). Parameters calibrated against observations, field and laboratory data and
  geostatistics. The paper prints no equations; the notebook gives the generic form.
- **Cost per run**: not reported. A "special program" wrote the MARTHE input files for each of the 300
  input sets, ran MARTHE and saved the 20 outputs (Sec. 4.2).

**Inputs** (Table 1, p. 26; "model value" = nominal). The units are reported only for kd (m³/m³,
Fig. 5 caption, p. 25); the others are not reported (n.r.).

| # | name | description | units | nominal | distribution | source |
|---|---|---|---|---|---|---|
| 1 | per1 | hydraulic conductivity, layer 1 | n.r. | 8 | uniform [1, 15] | Table 1, p. 26 |
| 2 | per2 | hydraulic conductivity, layer 2 | n.r. | 15 | uniform [5, 20] | Table 1, p. 26 |
| 3 | per3 | hydraulic conductivity, layer 3 | n.r. | 8 | uniform [1, 15] | Table 1, p. 26 |
| 4–7 | perz1–perz4 | hydraulic conductivity, zones 1–4 | n.r. | 8 each | uniform [1, 15] each | Table 1, p. 26 |
| 8–10 | d1–d3 | longitudinal dispersivity, layers 1–3 | n.r. | 0.8 each | uniform [0.05, 2] each | Table 1, p. 26 |
| 11–13 | dt1–dt3 | transversal dispersivity, layers 1–3 | n.r. | 0.08 each | uniform [0.01 d_k, 0.1 d_k], given d_k | Table 1, p. 26 |
| 14 | kd1 | volumetric distribution coefficient, layer 1 | m³/m³ | 5.1 | Weibull (1.1597, 19.9875) | Table 1, p. 26 |
| 15 | kd2 | volumetric distribution coefficient, layer 2 | m³/m³ | 0.34 | Weibull (0.891597, 24.4455) | Table 1, p. 26 |
| 16 | kd3 | volumetric distribution coefficient, layer 3 | m³/m³ | 5.1 | Weibull (1.27363, 22.4986) | Table 1, p. 26 |
| 17 | poros | porosity | – | 0.3 | uniform [0.3, 0.37] | Table 1, p. 26 |
| 18 | i1 | infiltration, type 1 | n.r. | 0.0001 | uniform [0, 0.0001] | Table 1, p. 26 |
| 19 | i2 | infiltration, type 2 | n.r. | 0.004 | uniform [i1, 0.01] | Table 1, p. 26 |
| 20 | i3 | infiltration, type 3 | n.r. | 0.02 | uniform [i2, 0.1] | Table 1, p. 26 |

The paper does not label the two Weibull parameters. The VLSE page lists them as α, β and states
"where α is the shape parameter, and β is the scale parameter" (page fetched 2026-11-01), so the first
number is the shape and the second the scale. The data agree: Kolmogorov–Smirnov p-values are 0.32,
1.00 and 0.71, and both Latin-hypercube blocks (below) are stratified in those CDFs; the reverse
reading (shape ≈ 20, scale ≈ 1) is incompatible with values up to 148. The kd distributions were fitted to measured histograms (Fig. 5); the other
inputs are uniform for lack of data (Sec. 4.1).

**Quantity of interest**: the Sr-90 concentration (Bq/l, Fig. 6 caption, p. 27) predicted by the
model at an observation well at the end of 2010 (Sec. 4.1). The paper used 20 wells, dropped two with
negligible concentrations in all runs, and analysed 18 (Sec. 4.2). The VLSE file has 10:
p102K, p104, p106, p2.76 (the paper's "p2–76"), p29K, p31K, p35K, p37K, p38, p4b. The VLSE page
gives no unit or year for them; the year and unit above come from the paper, and the file's
outputs reproduce the paper's Table 2 (below), so they are the paper's 2010 concentrations.

**Gradients**: none, and none can be computed (the code is not accessible).

**Sampling design**: the paper says "300 variants of model input values are generated using the
Latin Hypercube Sampling method" with dt_k generated given d_k and i2, i3 given i1, i2 (Sec. 4.1,
pp. 25–26); seed not reported. The file shows a different structure. With every input mapped to
[0, 1] through its own (conditional) distribution function (the notebook's coordinates), rows
1–100 and rows 101–200 are each a Latin hypercube of 100: each input hits all 100 equal-probability
strata, except i1 (92 and 96; i1 is rounded to 3 significant digits in the file) and single misses
in perz1, poros, perz4 and d2. Rows 201–300 hit 57 to 69 of 100 strata per input, as independent
sampling would (about 63), and are written with more digits (mean line length 299 characters
against 258). So the file is two Latin hypercubes of 100 plus 100 runs with no Latin structure. Why,
and whether the paper's analysis used the same 300 rows in this order, is not stated (see Unknowns).

**Published results** (comparison targets; no active subspace is reported):
- Rank-regression determination coefficients R²* and SRRC (×100, |SRRC| ≥ 0.10 only), Table 2,
  p. 28, for all 20 wells.
- Boosting-tree response surfaces (10,000 trees, interaction order 3), 10-fold cross-validation R²,
  and total Sobol' indices (%) from them, Table 3, p. 28. The three most influential inputs overall:
  kd2, kd1 and i3 (Sec. 4.5); dispersivities and porosity have "practically no influence" (Sec. 5);
  first-order and total indices "quite close" (Sec. 4.5).
- Histograms of p1–76, p102K and p103 (Fig. 6) and moment convergence for p1–76 (Fig. 7).

**Data source**: https://www.sfu.ca/~ssurjano/marthe.html (S. Surjanovic and D. Bingham, Simon
Fraser University), data file https://www.sfu.ca/~ssurjano/Code/marthedata.txt, downloaded
2026-11-01 by P. Constantine's browser (the container cannot reach sfu.ca). The page says the data
were obtained from the "Benchmark Proposals of GdR MASCOT-NUM" page
(http://www.gdr-mascotnum.fr/benchmarks.html), retrieved April 2014, and cites the paper above and
A. Marrel, B. Iooss, F. Van Dorpe, E. Volkova, "An efficient methodology for modeling complex
computer codes with Gaussian processes", *Computational Statistics & Data Analysis*
52(10):4731–4744, 2008, doi:10.1016/j.csda.2008.03.026, a Gaussian-process study of the same runs.

## Regenerating

The runs cannot be regenerated: MARTHE's site model is the authors' and the VLSE page says the code
is no longer accessible. The CSVs are regenerated from the shipped source file with

```bash
python convert_data.py           # writes inputs.csv and outputs.csv (< 1 s)
python convert_data.py --check   # verifies them; prints OK
```

`--check` on the shipped files: both CSVs byte-identical to a fresh conversion, parsed values
identical to `np.loadtxt('marthedata.txt', skiprows=1)` (`np.array_equal`), every number string
identical, SHA-256 of `marthedata.txt` as above.

## Accuracy

- **Conversion**: exact (above). The file's own precision is as written by the source: median
  5 significant digits for most inputs in rows 1–200 and 7–10 in rows 201–300 (trailing zeros
  dropped, so some values show fewer), 7–8 for the conditional inputs dt1–dt3, i2 and i3 in all
  rows, 3 for every i1 (267 distinct values in 300 rows), and
  7 for most concentrations (very small values with fewer).
- **Identity with the paper**: the notebook repeats the paper's rank regression. R²* equals Table 2
  to the printed two decimals at all 10 wells (max difference 0.00), and all 54 SRRC entries of
  Table 2 for these wells equal the notebook's to the integer (checked in the notebook). So the file's
  columns are labelled as in the paper and its outputs are the paper's runs. One entry is missing
  from Table 2: the rank regression gives SRRC +0.134 for dt3 at p38, but Table 2 has no
  dispersivity columns. On the concentration scale the effect is small (OLS component −0.042,
  bootstrap 95 % interval [−0.174, +0.117]).
- **Support**: no value outside the supports of Table 1 (conditional ranges included); all
  concentrations positive (no run outside physical meaning). Very small concentrations occur and
  are kept: values below 1e-3 Bq/l in 10 runs at p104 (minimum 7.85e-13), 4 at p37K (6.26e-16), 1 at
  p2.76 and 1 at p38.
- **Gradients**: none, so no gradient check. The notebook's estimators use values only.

## How these files relate to the paper

The file has 10 of the paper's 18 analysed wells and, as far as Table 2 can show, exactly the paper's
300 runs. The notebook analyses all 10 wells (two in detail: p104, and p29K on a log10 scale) in the
inputs' own distribution coordinates (a Rosenblatt map to independent uniforms). The paper's rank
regression is a close relative of that scale, and the notebook's OLS directions agree with the
paper's SRRC vectors (|cos| 0.986 and 0.997 at the headline wells) and with its most influential
input by total Sobol' index at 9 of the 10 wells. Beyond the two headline wells one direction
fits the concentration poorly (nested LOO R² of a cubic in the active variable 0.134–0.623), but
on the rank scale, the scale of the paper's R²*, nearly the same direction does much better
(linear LOO R² 0.522–0.882 at six of the eight), so the poor fits are largely an effect of the
output scale; p2.76 and p38 are poor on both scales. The paper's boosting-tree surfaces and Sobol'
indices are not recomputed. The paper's description of the design (one Latin hypercube of 300) does
not match the file (above).

## License and credit

The VLSE page and the GdR MASCOT-NUM source state no licence for the data. Per the repository
author's decision of 2026-10-30, data with no stated licence are hosted with credit: please cite
Volkova, Iooss and Van Dorpe (2008), doi:10.1007/s00477-006-0093-y, for the model and runs, and the
Virtual Library of Simulation Experiments (S. Surjanovic and D. Bingham,
https://www.sfu.ca/~ssurjano/marthe.html) for the hosted file. `convert_data.py` and the notebook
are MIT; the notebook's text and figures are CC BY 4.0 with the repository.

## Unknowns (questions for the authors)

- Units of the hydraulic conductivities, dispersivities and infiltration rates (m/day, m, m/day?), and
  the exact MARTHE sorption form for the "volumetric distribution coefficient".
- Did Table 2 deliberately leave out the dispersivity SRRCs (e.g. p38 dt3 ≈ +0.13)?
- The design: rows 1–200 of the file are two Latin hypercubes of 100 and rows 201–300 are not Latin;
  the paper says one Latin hypercube of 300. Were the last 100 runs drawn separately (e.g. for
  validation), and in which order did the paper use them?
- Seed of the design, cost per MARTHE run, and the MARTHE version.
- Which 10 of the 18 analysed wells were released, and why these.
