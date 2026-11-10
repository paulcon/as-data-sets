# JAG_ICF_Implosion_Scalars: data

Kind: **stored data** (converted from outside files). The values are runs of JAG, the semi-analytic inertial
confinement fusion (ICF) implosion simulator of Lawrence Livermore National Laboratory, released by Anirudh
et al. in the GitHub repository [llnl/macc](https://github.com/llnl/macc) as `data/icf-jag-10k.tar.gz` (MIT
licence). JAG is not public, so the runs cannot be regenerated. They are not altered: `convert_data.py` writes
every stored binary value as decimal text that converts back to it. **The data set is anonymous**: the release
names neither the five inputs nor the fifteen scalars and gives no units; inputs are `x1`–`x5`, scalars
`s00`–`s14`, in the stored column order.

## Files

| file | contents | role |
|---|---|---|
| `JAG_ICF_Implosion_Scalars.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 646; outputs committed |
| `jag_icf_inputs.csv` | 10000 × 5: inputs `x1`–`x5` as released (float64, 17 significant digits) | tarball member `jag10K_params.npy` |
| `jag_icf_scalars.csv` | 10000 × 15: scalars `s00`–`s14` as released (float32 values, 9 significant digits); row i matches row i of the inputs | tarball member `jag10K_0_scalars.npy` |
| `convert_data.py` | reads the two `.npy` members from the tarball (SHA-256 checked, nothing extracted, no pickles), writes the CSVs; `--check` compares | the conversion (numpy) |
| `check_accuracy.py` | prints the numbers of this file that neither the notebook nor `convert_data.py` prints (stored precision, the README example, the tarball members) | check |
| `LICENSE-macc.txt` | the MIT licence and NOTICE of llnl/macc, verbatim | licence of the two CSVs |

The CSVs are plain comma-separated numbers after two `#` lines (a description, then the column names with
"units"), so `np.loadtxt(path, delimiter=',')` reads them. The unit field says `units not given`: neither
the paper nor the repository gives units or says whether the values are physical or scaled (see Unknowns). The tarball's third member, `jag10K_images.npy` (10000 × 16384
float32: four 64 × 64 X-ray images per run), is not used and not shipped.

## Model

**Citation.** R. Anirudh, J. J. Thiagarajan, P.-T. Bremer, B. K. Spears, "Improved surrogates in inertial
confinement fusion with manifold and cycle consistencies", *Proc. Natl. Acad. Sci. USA* 117(18):9741–9746
(2020), doi:10.1073/pnas.1916634117 (checked on Crossref). Read: the arXiv preprint 1912.08113v1 (a
preprint; page numbers below are its pages). The published article and its supplement were not read; they
might name the inputs.

**Simulator.** "the semi-analytical ICF simulator described in [1]" (p. 4; [1] is an APS meeting abstract of
Gaffney, Springer and Collins). The paper gives no equations. The README of llnl/macc describes JAG as an
analytic hotspot profile (Betti et al. 2002; Springer et al. 2013) plus coupled ODEs of the energy balance
(Betti et al. 2015; radiation, electron conduction, alpha heating, ...) and a thin-shell model of the 3D
hotspot evolution, post-processed into X-ray images and scalars, and says that "JAG inputs do not relate to
actual experimental observables, rather the state of the implosion once the laser drive has switched off".

**Inputs** (m = 5). The paper: a "5-dimensional input parameter space" (p. 4), process inputs "say target and
laser settings for ICF" (p. 2, general), scans between "the min and max ranges of that parameter" (p. 5). No
names, units or ranges are given in the paper or the repository.

| input | range | units | source |
|---|---|---|---|
| `x1`, `x4`, `x5` | [−0.3, 0.3] each | not given | read from the data (sample min/max within 1e-5 of the ends; notebook) |
| `x2`, `x3` | [0, 1] each | not given | read from the data |

The notebook declares `ac.Uniform` on this box (this data set's choice) and scales each input to [−1, 1]
through it.

**Quantities of interest.** "15 diagnostic scalar quantities such as yield, ion temperature, pressure, etc."
(p. 4); which column is which is not released. Headline output: `s00`, by the fixed rule "first stored
column" (no column can be chosen by meaning); the other fourteen are secondary. Fifteen columns form eight
groups with |r| ≥ 0.99 (`s07` and `s08` differ by at most 1.2e-6), printed by the notebook.

**Gradients.** None released (JAG is not available). The notebook estimates them by local linear fits to the
8 nearest rows (inputs scaled to [−1, 1]; neighbourhood radius median 0.338), and cross-checks with 12, 20 and
40 neighbours, with the exact gradients of a degree-6 polynomial fitted to all rows, and with QPHD.

**Sampling design.** Paper: the training data are "a random subset (100K samples) of a Latin Hypercube
experiment design containing 1 million samples" (p. 4), evaluated on "a held-out 10K validation set"
(p. 5). Which 10,000 runs were released is not stated (the README says 9,984 rows; the file has 10,000). The
released design is close to uniform (900–1109 rows per tenth of each range) but is not a plain random subset of
an LHS: about 0.12 of the values of each input lie within 1e-4 of the ten cell boundaries of its range, 20
values per input (just inside and outside each boundary) are shared by 34–79 rows each, and 3528 rows (1567 groups)
have the same five within-cell positions as another row to within 6e-7 of a cell width (5.9e-8 of the range),
not exactly (no match at 9 decimals), as if copied and shifted by whole cells with a perturbation of the order of
single-precision rounding (all numbers printed by the notebook). The notebook keeps all rows with
equal weight and shows that the subspace of `s00` is the same without the 5251 rows that have a boundary value.

**Published results.** Surrogate accuracy only: mean R² over the 15 scalars 0.9990 (baseline network) and
0.9974 (MaCC) on the 10K validation set (Table 1, p. 6), image MSEs, and a cycle-consistency test. No
sensitivity or subspace results.

## Regenerating

The runs cannot be regenerated (the simulator is not released). The CSVs are regenerated from the tarball:

```
git clone https://github.com/llnl/macc.git          # data/icf-jag-10k.tar.gz is a plain git file
python convert_data.py macc/data/icf-jag-10k.tar.gz --out DIR            # writes the two CSVs into DIR
python convert_data.py macc/data/icf-jag-10k.tar.gz --out DIR --check    # ... and compares with the shipped CSVs
```

`--out` is required and may not be this directory. `--check` passes when every written value is within 1e-12
(inputs) or 1e-8 (scalars) of the shipped one and of the `.npy` value, and every value converts back to the
stored binary value (float64 inputs, float32 scalars). On 2026-11-09, in the Linux container where the
shipped files were written, it printed `OK` for both files (no difference from the shipped files above the
tolerance; scalars within 5.0e-10 of the float32 values; no value failing the round trip) in about 2 s. It
was not rerun in the repository's `.venv`, which has no copy of the tarball.

**Source.** github.com/llnl/macc, commit a0318c2216d3e09a9c4c59f61066043e6e8a8799 (2022-08-11),
`data/icf-jag-10k.tar.gz`, 66,417,328 bytes, SHA-256
`0bdfc984fa5c10c45014bed1cdd3fae4980b5de9f39344105a052cfaad6bd14f` (members dated 2019-04-11). The README
gives the dataset citation Gaffney et al. 2020, doi:10.6075/J0RV0M27 (UC San Diego Library; not read). The
tarball is not shipped.

In the repository's `.venv` the notebook takes 3 to 4 minutes (it prints 185 s in the committed run, and
printed 210 s and 220 s in two earlier runs with other notebooks running at the same time; all under
nbconvert on 2026-11-10); most of it is the 60 polynomial fits of the smoothness check, the neighbour-count comparison and
the per-column fits of the last section. `check_accuracy.py` (without the tarball) takes under 1 s.

## Accuracy

- Conversion: lossless (`convert_data.py --check`, above).
- Stored precision (`check_accuracy.py`): inputs float64; scalars float32 (spacing 6.0e-08 at the largest
  value), against column standard deviations of 0.0955–0.1964 (notebook).
- No simulator gradients, so no gradient check against finite differences. In their place the notebook
  compares the local linear estimate with the degree-6 polynomial's gradients (LOO R² 0.9970 for `s00`):
  first directions agree to subspace distance 0.0255 (QPHD 0.0169); the local-linear eigenvalues 2–5 are
  1.09e-01–5.26e-02 of λ1 against 5.53e-02–7.47e-04 from the polynomial, so they are dominated by the
  local fits' error (per-row gradient error against the polynomial's: median 0.35, 90th percentile 0.90 with 8
  neighbours; 0.13 and 0.27 with 40). The bootstrap of the notebook resamples these fixed gradients and does
  not include that error (its 95th percentile for n = 1, 0.0146, is below the 0.0255 distance).
- Smoothness (notebook): degree-6 polynomials in the five inputs reach LOO R² 0.9933–1.0000 for the 15
  columns; the same fit to permuted `s00` gives −0.0535. The data behave like deterministic simulation output.
- README example (`check_accuracy.py`): the "sample 0" printed in the llnl/macc README does not match the
  file. Its inputs differ from row 0 by up to 0.345 (nearest row 282, at 0.041 in max-norm); it lists 22
  scalars, not 15; 1 of its 14 distinct values occurs anywhere in the file (within 1e-7), and no row holds 3
  of them. The README example therefore comes from a different file or version; the tarball members are as
  listed by `check_accuracy.py <tarball>` (shapes (10000, 5), (10000, 15), (10000, 16384)).

## How these files relate to the paper

The files are the authors' release of 10,000 JAG runs, not the paper's 100K training set. Whether they are its
10K validation set, part of the training set or another draw is not stated. The paper's networks predict
these scalars with mean R² ≈ 0.99; the notebook finds degree-6 polynomials do as well (mean LOO R² 0.9983) and
that every scalar's leading active direction lies in the plane of `x2` and `x3` (weights of `x1`, `x4`, `x5`
at most 0.0150), which the paper does not report; several scalars also depend on `x5` (LOO R² in `(x2, x3)`
alone 0.8658–0.9269 for `s03`, `s07`, `s09`; 0.9874–0.9974 with `x5`). For `s00`, n = 1: λ1/λ2 depends on the
neighbour count of the local linear fits (9.19 with 8, 15.99 with 12, 20.72 with 20, 24.41 with 40) and is 18.1
from the polynomial's gradients, while w1 stays within subspace distance 0.0255 of the polynomial's; LOO R²
0.9758 on one active variable, 0.9874 on two. The polynomial spectrum has later gaps too (λ2/λ3 = 4.65,
λ3/λ4 = 6.09), so n = 1 rests on the LOO R² and the stable w1, not on a single gap.

## License and credit

Data: © 2018 Lawrence Livermore National Security, LLC, from github.com/llnl/macc under the MIT licence
(`LICENSE-macc.txt`, with the repository's NOTICE, kept beside the files); listed in the exceptions of
`LICENSE-DATA`. Cite the paper and the dataset (Gaffney et al. 2020, doi:10.6075/J0RV0M27). The article is
CC BY-NC-ND 4.0 (Crossref) and is not included. `convert_data.py`, `check_accuracy.py` and the notebook: MIT
(this repository). Work performed under the auspices of the U.S. DOE by LLNL under Contract DE-AC52-07NA27344
(p. 7); LLNL LDRD award 18-SI-002 (Crossref).

## Unknowns

- Names, units and physical meaning of the five inputs; whether the released values are physical or scaled.
  (Other LLNL publications on the JAG ensemble may name them; not checked, and the column order could not be
  matched.)
- Names of the fifteen scalars; whether they are scaled, and how (values in (0, 1), minima 0.0005–0.0159 and
  maxima 0.5070–0.9914, so not the min–max of these rows; linear or log not known); why `s07` and `s08` nearly
  coincide.
- Which runs of the 1M-point design the 10,000 are; how the design was built (cell-boundary clusters and
  cell-translated rows, which agree to about 1e-7 rather than exactly, perhaps through a single-precision step, are
  not explained by "a random subset of a Latin hypercube"); why the README says 9,984
  rows and prints a 22-scalar example that is not in the file.
- JAG's version, settings and run cost.
- The UC San Diego dataset record doi:10.6075/J0RV0M27 itself was not read (its page refuses scripted
  requests). Its DataCite metadata (read on 2026-11-10) gives the licence as CC BY 4.0 and describes the
  tarball of llnl/macc; the files are hosted here under the MIT licence of llnl/macc, from which they were
  taken.

## Open questions

For the authors (not contacted): the input and scalar names and units in column order; whether either set is
scaled, and how; the design construction (cell-boundary clusters; rows repeating another row's within-cell
position to about 1e-7); whether the release is the paper's validation set; why the README says 9,984 rows and
prints a 22-scalar example.
