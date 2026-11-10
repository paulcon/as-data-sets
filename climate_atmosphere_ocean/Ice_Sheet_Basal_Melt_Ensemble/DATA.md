# Ice_Sheet_Basal_Melt_Ensemble: data

Kind: **stored data** (converted from outside files). The values were computed by Berdahl et al. (2021) with
the Community Ice Sheet Model (CISM v2.1) on an Antarctic 4 km grid and published in the authors' code and
data repository, https://github.com/mberdahl-uw/EmulatorPaper (no licence stated; archived at Zenodo,
[doi:10.5281/zenodo.4902149](https://doi.org/10.5281/zenodo.4902149)). They were not regenerated (500
continental ice-sheet runs of 200 years on supercomputers) and are not altered: `convert_data.py` copies
every number character for character into the CSV layout of this repository.

## Files

| file | contents | role |
|---|---|---|
| `Ice_Sheet_Basal_Melt_Ensemble.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2683; outputs committed |
| `basal_melt_design.csv` | 500 × 32: `ens`, `flag`, then for each basin (amundsen, eais, peninsula, ross, weddell) `M200frac`, `Mmaxfrac` (fractions of the basin cap), `t0`, `tau` (yr), `M200`, `Mmax` (m/yr) | the five `design_metadata_<basin>.csv` of the authors' repository |
| `ice_volume_above_flotation.csv` | 500 × 201: `ens`, then `V1`…`V200`, the annual ice volume above flotation in mm sea-level equivalent (SLE); row i matches row i of the design | `SLR.csv` of the authors' repository |
| `convert_data.py` | checks the SHA-256 of the six source files, joins the design files, copies the volume file; `--check` compares | the conversion (standard library; numpy for `--check`) |
| `check_accuracy.py` | recomputes the numbers of this file that neither the notebook nor `convert_data.py` prints | check (numpy, scipy) |

The CSVs are plain comma-separated numbers after two `#` lines (a description, then the column names with
units), so `np.loadtxt(path, delimiter=',')` reads them. Source label → column: `m2200` → `M200frac_<basin>`,
`mmax` → `Mmaxfrac_<basin>`, `m2200 scaled` → `M200_<basin>`, `mmax scaled` → `Mmax_<basin>`, `t0`, `tau`
unchanged. The authors' "m2200" is the paper's $M_{200}$, the anomaly at simulation year 200 (their scripts
label simulation years 100 and 200 as 2100 and 2200). `ens` and `flag` are identical in the five source
files (checked by `convert_data.py`) and are written once. The first line of `SLR.csv`, `# ensemble member`,
is replaced by the two header lines. The repository's three R scripts (emulator fitting, validation and
prior propagation) are not shipped.

## Model

**Citation.** M. Berdahl, G. Leguy, W. H. Lipscomb, N. M. Urban, "Statistical emulation of a perturbed basal
melt ensemble of an ice sheet model to better quantify Antarctic sea level rise uncertainties", *The
Cryosphere* 15(6), 2683–2699 (2021), doi:10.5194/tc-15-2683-2021 (publisher PDF read, CC BY 4.0; checked on
Crossref). CISM v2.1: Lipscomb et al., *Geosci. Model Dev.* 12, 387–424 (2019), doi:10.5194/gmd-12-387-2019.

**Simulator** (Sect. 2.1, pp. 2685–2687). CISM v2.1 on a uniform 4 km Antarctic grid: depth-integrated
higher-order solver; Schoof (2005) sliding in the power-law limit; grounding-line parameterizations of basal
shear stress and melt; melt applied to partially floating cells in proportion to their floating fraction;
no-advance calving; constant RACMO2 1976–2016 surface mass balance; Shapiro–Ritzwoller geothermal flux. Spin-up
of 40 000 years with nudging of basal friction and basal melt toward observed thickness; a 1000-year control
run drifts by < 1 Gt/yr (p. 2687). Each member runs 200 years from the end of spin-up with a spatially
uniform basal-melt anomaly per LARMIP basin (Peninsula, Weddell, East Antarctica, Ross, Amundsen; Fig. 4,
after Levermann et al. 2020), applied also to newly ungrounded cells (p. 2689). Years are relative to the end
of spin-up, not calendar years (p. 2687). Cost per run: not reported (computing at LANL and on NCAR's
Cheyenne, p. 2696).

**Melt-rate anomaly** (Eqs. 1–5, p. 2687): $M(t) = A/(1+e^{x}) - B$, $x = -(t-t_0)/\tau$,
$A = K M_{\max}/(K-1)$, $K = 1+e^{t_0/\tau}$, $B = A/K$, so $M(0) = 0$. Eq. (1) is printed with an unmatched
parenthesis, "$A/((1+e^x)$". The ensemble is parameterized by $M_{200} = M(200)$ instead of $M_{\max}$
(p. 2688); the stored `Mmax` columns satisfy $M_{200} = M_{\max}\,g(t_0,\tau)$,
$g = \frac{K}{K-1}\big(\frac{1}{1+e^{-(200-t_0)/\tau}} - \frac{1}{K}\big)$, to a relative error of at most
4.8e-10 (`check_accuracy.py`).

**Inputs** (m = 15; Sect. 2.2, p. 2688; no table): per basin $t_0 \in [100, 225]$ yr, $\tau \in [10, 75]$ yr
and $M_{200} \in [0, 1]$ scaled by a basin cap, "at least twice the maximum value found in the literature"
(p. 2688).

| inputs | units | range | source |
|---|---|---|---|
| `M200_amundsen`, `t0_amundsen`, `tau_amundsen` | m/yr, yr, yr | [0, 50], [100, 225], [10, 75] | Sect. 2.2, p. 2688 |
| `M200_eais`, `t0_eais`, `tau_eais` | m/yr, yr, yr | [0, 36], [100, 225], [10, 75] | p. 2688 |
| `M200_peninsula`, `t0_peninsula`, `tau_peninsula` | m/yr, yr, yr | [0, 12], [100, 225], [10, 75] | p. 2688 |
| `M200_ross`, `t0_ross`, `tau_ross` | m/yr, yr, yr | [0, 20], [100, 225], [10, 75] | p. 2688 |
| `M200_weddell`, `t0_weddell`, `tau_weddell` | m/yr, yr, yr | [0, 16], [100, 225], [10, 75] | p. 2688 |

The scaled columns equal the unscaled fractions times these caps (largest relative deviation of order 1e-16,
`check_accuracy.py`); all 500 runs lie inside the bounds (`check_accuracy.py`).

**Quantity of interest.** Antarctic sea-level contribution $f = -(V_{200} - V_1)$ in mm at simulation year
200, the authors' definition in `cism_slr_emulator.R` (`slr = -(V_t - V_1)`, column `yr - 2000`); year 100,
$-(V_{100}-V_1)$, is analysed briefly. The unit (mm SLE) is inferred: the file has no unit, and the
contributions reproduce the paper's mm ranges (below). Whether `V1` is the state at year 0 or after year 1 is
not stated (see Unknowns).

**Gradients.** None: CISM's sea-level output was not differentiated, and finite differences would need 16
continental runs per gradient. The notebook estimates the subspace with QPHD (a global quadratic, 136
coefficients from 491 runs, in normalized coordinates $[-1,1]^{15}$, each input scaled by its p. 2688 bounds)
and cross-checks it with local linear gradients (fits on the 25 nearest runs). The local linear fits are not
local here: the median 25-run neighbourhood radius is 2.30 in normalized units, more than the cube side 2, and
the mean local gradient is within 0.044 (relative) of one global linear slope (notebook). Such smoothed
gradients concentrate $\mathbf{C}$ on one direction, so the local linear $\lambda_1/\lambda_2$ = 19.57 is
inflated and is not used to choose the dimension. QPHD gives $\lambda_1/\lambda_2$ = 6.97 and
$\lambda_2/\lambda_3$ = 3.74 (then 1.27), and the notebook takes $n = 2$ at year 200 (one direction at year
100).

**Sampling design.** The paper: a quasi-random Sobol' sequence sampling bounded uniform distributions, 500
members (pp. 2688–2689); seed and generator not reported. On the assignment across basins it says (p. 2689):
"For each CISM run, a random basal melt rate curve generated with the Sobol' sequence is chosen for each
basin. We have chosen not to assume any correlation between basins." A random choice of curve per basin
would leave the basins nearly uncorrelated, so this description does not explain the correlations below. In the
stored data each basin's 3-D design (M200, t0, tau) has a centred L2 discrepancy of 8.5e-05 to 1.2e-04 on the
unit cube, against 1.5e-03 for uniform random designs of the same size (`check_accuracy.py`), consistent with
a low-discrepancy sequence per basin. **Across basins the stored inputs are correlated**: largest
|r| = 0.63 ($\tau$ Peninsula–Weddell), 8 of 105 pairs with |r| > 0.2 (also $\tau$ EAIS–Peninsula +0.35,
$\tau$ Amundsen–Peninsula +0.35, $M_{200}$ Peninsula–Weddell −0.33), against a sampling scale
1/√500 = 0.045 (`check_accuracy.py`). The stored design is therefore not the uncorrelated design the quoted
text describes. The correlation is not a property of an unscrambled 15-D Sobol' sequence either (scipy's first
500 points have largest |r| 0.023, `check_accuracy.py`). The notebook checks that the
subspace does not depend on this: a quadratic fit differentiated at 20 000 independent uniform points gives
the same first direction as at the stored points (1 − |cos| of order 1e-6) and a 2-D subspace 6.8e-02 away
(notebook).

**Failed runs.** `flag = 1` marks 9 runs (ens 70, 92, 96, 97, 224, 246, 249, 257, 483); their volume rows are
0 from stored year 135–191 on, and no succeeded run holds a 0 (`check_accuracy.py`). The paper does not
mention them ("the full 500-member CISM ensemble", p. 2689); the authors' R scripts drop them for both years,
and so does the notebook. **Do they bias the design?** In input space hardly: no input differs clearly
between failed and succeeded runs (smallest two-sided Mann–Whitney p = 0.029, for `t0_peninsula`; 0.44 after a
Bonferroni adjustment over 15 inputs), and 9 of 500 points leave the coverage of the space-filling design
intact. In output they are not random: their year-100 contributions (20.6–65.5 mm) lie above the median of
the succeeded runs in 8 of 9 cases (Mann–Whitney p = 0.0095; `check_accuracy.py`), so their early
contributions are high. On the year-200 active variable they are not in the upper tail. Their positions on
$\mathbf{w}_1^\top\mathbf{x}$ (QPHD) among the 500 runs are at fractions 0.17–0.92 (median 0.66). 1 of the 50
runs in the top 10 % failed (2.0 %), against 1.8 % of the rest (notebook). The loss is mild and spread over the
middle and upper half. The analysis describes the model conditional on a run completing.

**Published results.** No active subspace. Ensemble ranges at years 100 and 200, 0.5–96 mm and 33–543 mm
(p. 2691); GP emulator validation on a 20 % hold-out, correlations 0.98 and 0.99 (Fig. 8, p. 2690); 500-member
time series and histograms (Fig. 9); emulated statistics for four priors (Table 2, p. 2693; e.g. Method 1
year-200 2.5–97.5 % range 40.3–170.8 mm, mode 82 mm), which the notebook does not recompute.

## Regenerating

The runs cannot be regenerated from the repository (CISM v2.1 at 4 km on a supercomputer; the authors'
spin-up state and run configurations are not in their repository). The CSV files are regenerated from the
authors' files:

```
git clone https://github.com/mberdahl-uw/EmulatorPaper      # commit dd8643c5962f045cf905f3ba50f95ce3ffd37431
python convert_data.py EmulatorPaper --out DIR               # writes the two CSVs into DIR
python convert_data.py EmulatorPaper --out DIR --check       # also compares with the shipped files
```

`DIR` must lie outside this directory (the script refuses otherwise). `--check` passes when the largest
relative difference is at most 1e-12 and also reports whether the text is identical; on 2026-11-09, in the
Linux container where the shipped files were written, it reported a largest relative difference of 0.0e+00
and identical text for both files, in under 1 s. It was not rerun in the repository's `.venv`, which has no
copy of the authors' files.
In the repository's `.venv` the notebook runs in about 20–30 s under nbconvert (18 and 29 s of wall time in
two runs on 2026-11-10, with other notebooks running at the same time), most of it the two 500-replicate
bootstraps.

**Source.** GitHub repository mberdahl-uw/EmulatorPaper, commit dd8643c (2020-06-19, "Add files via upload";
cloned 2026-11-09). Files and SHA-256:

| file | bytes | SHA-256 |
|---|---|---|
| `SLR.csv` | 2,512,518 | `d774ce52efedc259634bb4cab5c839c9da7a71387294cc979c7f1d1f337d5b46` |
| `design_metadata_amundsen.csv` | 57,362 | `766f4214327bed8dc8e4ec133590e0975a01d9b27f7752e1270047b18b527742` |
| `design_metadata_eais.csv` | 57,809 | `da5fa5c3c4273b59e4f7022d30f95f6812b8913ca7a6fb3c98b06b96b2379fe7` |
| `design_metadata_peninsula.csv` | 57,718 | `9b4811186229e5a402aff3688d8adbf34ac98354a1841b2bf951c33e6d15b34f` |
| `design_metadata_ross.csv` | 57,537 | `8ac5d29a27d6af0105747717642717a0cd8684415d1092088be702517b81dcba` |
| `design_metadata_weddell.csv` | 57,761 | `2f5a6dabbfd05c47e2b6ed9abe90963cf4f34217d36b4f315fed4a9d3a6f9c9a` |

The paper's Code and data availability statement (p. 2696) gives this repository and the Zenodo record
10.5281/zenodo.4902149 ("CISM Emulator Code and data (Version 1.0.0)", 2021). The Zenodo copy was not opened
here; whether its files are byte-identical to the commit used is not checked.

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name, in about 1 s.

- Conversion: lossless (`convert_data.py --check`, above).
- Are these genuine simulation outputs? Checks of internal consistency (`check_accuracy.py` unless noted):
  the stored `Mmax` and `M200` satisfy Eqs. 1–5 to 4.8e-10 relative error; scaled = fraction × cap; all 500
  runs start from one volume, 55404.91363231 mm SLE (one spin-up); every year-to-year volume change of the 491
  succeeded runs is an integer multiple of 0.006078 mm (within 1.1e-06 of the quantum over 97,709 changes),
  as written by a model diagnostic of fixed precision; the failed runs carry the flag in all five design
  files and zeros only after they stopped; and the ranges of the contribution, 0.50–95.81 mm at year 100 and
  33.03–542.77 mm at year 200 (notebook), round to the paper's 0.5–96 mm and 33–543 mm (p. 2691). No values
  were found that look inserted.
- CISM's own numerical accuracy is not reported in the paper beyond the control-run drift (< 1 Gt/yr,
  p. 2687); external run, not reproducible from the repository.
- No gradients, so no gradient check. Fit quality (notebook; LOO given the $\mathbf{W}$ estimated from all
  491 runs): leave-one-out R² 0.900 for a quadratic in the first active variable, 0.951 in the two QPHD
  variables (0.943 with the local linear pair), 0.972 for a full quadratic in all 15 inputs, 0.884 for a linear
  fit in all 15.

## How these files relate to the paper

The files are the paper's 500-member CISM ensemble (Fig. 9) with its design, as the authors deposited them
for their GP emulator; the 491 completed runs are the authors' training set (their R scripts drop the 9
flagged runs; the paper speaks of 500). The paper builds and validates a GP emulator and propagates prior
distributions through it; the notebook instead estimates an active subspace of the year-200 (and year-100)
contribution directly from the runs. The paper reports no active subspace, eigenvalues or sensitivity
ranking, so the notebook compares the ensemble ranges and the fit quality (a quadratic on the active
variables against the GP's Fig. 8 correlations; a different validation). Two discrepancies with the paper:
the stored design is correlated across basins although the paper describes a random choice of curve per basin
with no correlation assumed (p. 2689), and the paper
does not mention the 9 failed runs. Eq. (1) is printed with an unmatched parenthesis.

## License and credit

Data: © M. Berdahl, G. Leguy, W. H. Lipscomb, N. M. Urban; the repository states no licence, so the two CSVs
are hosted with the source credited (repository decision of 2026-10-30) and are listed in the exceptions of
`LICENSE-DATA`. The article is CC BY 4.0. Cite the paper and the Zenodo record when using these files.
`convert_data.py`, `check_accuracy.py` and the notebook: MIT (this repository). No code of the authors'
repository is included.

## Unknowns

- Whether `V1` is the volume at year 0 (end of spin-up) or after the first year; the contribution "at year
  200" is $-(V_{200}-V_1)$ as in the authors' script, and may span 199 years.
- The unit of `SLR.csv` (mm SLE) is inferred from the paper's mm ranges; the file states none.
- How the Sobol' points were generated and assigned to basins (generator, scrambling, seed), and why the
  basins are correlated in the stored design although the paper says a random curve is chosen for each basin
  with no correlation assumed (p. 2689).
- Why the 9 runs failed (numerical failure, time limit or other); the paper does not mention them.
- Whether the Zenodo record 10.5281/zenodo.4902149 holds byte-identical files (not opened).
- Cost per CISM run (not reported).

## Open questions

- For the authors: how were the "random curves generated with the Sobol' sequence" (p. 2689) assigned to the
  basins of each run (one 15-D sequence, or a 3-D sequence per basin with a random choice), and why does the
  stored design have a Peninsula–Weddell $\tau$ correlation of +0.63?
- For the authors: what made the 9 flagged runs stop (stored years 135–191), and why does the paper speak of
  "the full 500-member CISM ensemble" (p. 2689) when the R scripts train on 491?
- For the authors: is `V1` the end-of-spin-up state or the volume after year 1?
