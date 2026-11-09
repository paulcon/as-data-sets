# Subglacial_Drainage_GlaDS: data

Kind: **stored data**. The values were computed by Hill et al. (2025) with the GlaDS subglacial drainage model
(ISSM v4.24) and deposited with their code in the Zenodo record "timghill/GladsGP: v1.0.0",
[doi:10.5281/zenodo.15053542](https://doi.org/10.5281/zenodo.15053542) (MIT licence). They were not
regenerated (one GlaDS run took 25 min, the 512 training runs 216 h; Table 4, p. 4058) and are not altered:
`convert_data.py` copies the design values character for character and writes every stored output so that
it reloads bit for bit.

## Files

| file | contents | source in the deposit |
|---|---|---|
| `Subglacial_Drainage_GlaDS.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 637; outputs committed |
| `glads_train_inputs.csv` | 512 × 9: `run_id`, then `k_s, k_c, h_b, r_b, l_c, A, omega, e_v` in physical units (units in the second `#` line) | `experiments/synthetic/expdesign/synthetic_train_physical.csv` |
| `glads_test_inputs.csv` | 100 × 9, same columns | `experiments/synthetic/expdesign/synthetic_test_physical.csv` |
| `glads_train_qoi.csv` | 512 × 17: `run_id`, `fQ_x05km` … `fQ_x30km`, `fQ_mean` (-), `log10Ts_x05km` … `log10Ts_x30km`, `log10Ts_mean` (log10 of years), `Lc_R0.5m`, `Lc_R1.0m` (km); row i is the run of row i of the inputs | `experiments/synthetic/issm/train/synthetic_{channel_frac,log_transit_time,channel_length}.npy` |
| `glads_test_qoi.csv` | 100 × 17, same columns | `experiments/synthetic/issm/test/synthetic_*.npy` (same three names) |
| `convert_data.py` | reads the eight source files of the extracted deposit, checks their SHA-256 and writes the four CSVs; `--check` compares with the shipped files | the conversion (numpy only; no pickle is loaded) |
| `GladsGP_LICENSE.txt` | the MIT licence of the deposit (identical in size, 1065 bytes, to the deposit's `LICENSE`; text from github.com/timghill/GladsGP at commit e7f143a) | licence of the four CSV files |

The CSVs are plain comma-separated numbers after two `#` lines (a description; column names with units), so
`np.loadtxt(path, delimiter=",")` reads them. `run_id` is the authors' job number, counted from 1; the paper's
Table B1 (p. 4067) counts test runs from 0: its simulations 66, 63 and 71 are `run_id` 67, 64 and 72 (notebook).
23 of the 24 values it prints agree with the file to two digits, and in the first five inputs each run lies within
0.011 decades of its column while the next closest test run is at least 0.27 decades away. The exception is
$\omega$ of simulation 63: Table B1 prints 1.3 × 10⁻⁴, outside the Table 2 range [2 × 10⁻⁴, 2 × 10⁻³] (p. 4047),
and the file has 1.3 × 10⁻³; the printed value is presumably a misprint. The design files are printed with 7
significant digits. The paper's three scalar outputs are `fQ_mean`,
`log10Ts_mean` and `Lc_R0.5m`; the other columns (each fluxgate, $L_c$ for $R \ge 1$ m) are the rest of what
the deposit stores for each run. The deposit's flotation-fraction fields (`synthetic_ff.npy`, 2.8 GB for the
training runs) and per-run outputs are not used and not shipped.

## Model

**Citation.** T. Hill, D. Bingham, G. E. Flowers, M. J. Hoffman, "Computationally efficient subglacial
drainage modelling using Gaussian process emulators: GlaDS-GP v1.0", *Geoscientific Model Development* 18,
4045–4074 (2025), doi:10.5194/gmd-18-4045-2025 (publisher PDF read, CC BY 4.0). Page numbers below are the
journal's (4045–4074). Code and runs: Zenodo 10.5281/zenodo.15053542 (also github.com/timghill/GladsGP).

**Simulator** (pp. 4046, 4051–4052, 4065–4066). GlaDS (Werder et al. 2013) in ISSM v4.24, with the
laminar–turbulent sheet flow of Hill et al. (2024b): a distributed water sheet (Eqs. A1–A3) coupled to
ice-walled channels on mesh edges (Eqs. A4–A8). Synthetic land-terminating margin of 100 km × 25 km modified
from SHMIP (flat bed at 350 m, surface 390–1910 m), mesh of 3693 nodes (mean edge about 900 m), 60 moulins,
basal melt 0.05 m w.e. a⁻¹, surface melt from a positive degree-day model driven by 2014 KAN_L daily air
temperatures, atmospheric pressure at the 25 km terminus and no flux elsewhere; two years with identical
forcing, the first discarded as spin-up. Fixed constants and parameters: Table 2 (top and middle groups).

**Inputs** (m = 8; Table 2, bottom group, p. 4047). All log-transformed in the paper (p. 4052).

| column | symbol | description | range | units as printed | source |
|---|---|---|---|---|---|
| `k_s` | $k_s$ | sheet conductivity | [0.01, 1] | Pa s⁻¹ | Table 2, p. 4047 |
| `k_c` | $k_c$ | channel conductivity | [0.05, 0.5] | m^3/2 s⁻¹ | Table 2 |
| `h_b` | $h_b$ | bed bump height | [0.05, 1] | m | Table 2 |
| `r_b` | $r_b$ | bed bump aspect ratio | [10, 100] | – | Table 2 |
| `l_c` | $l_c$ | width of sheet beneath channels | [1, 100] | m | Table 2 |
| `A` | $A$ | ice flow law coefficient | [1e-24, 1e-23] | s⁻¹ Pa⁻³ | Table 2 |
| `omega` | $\omega$ | laminar–turbulent transition parameter | [1/5000, 1/500] | – | Table 2 |
| `e_v` | $e_v$ | englacial void fraction | [5e-5, 5e-4] | – | Table 2 |

The column order is that of the authors' design files ($l_c$ before $A$; Table 2 lists $A$ first). No nominal
values are given (the study varies all eight together).

**Quantities of interest** (p. 4053). For fluxgates 5, 10, …, 30 km from the terminus: $f_Q$, channel
discharge divided by total discharge across the gate ("melt-season-integrated" in the text); $T_s$, the
width-averaged time for a water parcel to reach the terminus through the sheet, in years; and $L_c$, the
total length of channel segments with radius $\ge R$, with $R$ = 0.5 m. The paper's $f_Q$ and $\log T_s$
are means over the six gates. For $T_s$ the text says it "is computed as the average time starting from each of
the same six fluxgate positions" (p. 4053), but the stored `log10Ts_mean` is the mean of the six $\log_{10}$ gate
values (to a relative 1.5e-07; notebook), i.e. $\log_{10}$ of the geometric mean time; the log of the average time
would differ by up to 0.121. Table 5's test range matches the stored column.
The authors' aggregation script (`src/aggregate_outputs.py`, read at
github.com/timghill/GladsGP commit e7f143a) defines the stored values: discharge summed over all stored
daily outputs of a run, $\log_{10}$ of the yearly mean of $T_s$ at each gate, the gate means appended as a
seventh row, and $L_c$ as the maximum over the year divided by 1000 (km), for $R$ = 0.5 and 1.0 m.

**Gradients.** None: GlaDS computes none, and finite differences would cost m + 1 = 9 runs of 25 min per
gradient. The notebook estimates the subspaces from values (QPHD; checks with local linear gradients and OLS).

**Sampling design** (p. 4052). Training: 512 points of a Sobol' sequence (p. 4052), uniform in the log of each
input over the Table 2 box. The paper does not state the generator; the deposit's
`experiments/synthetic/train_config.py` does: the first 512 points of a scrambled Sobol' sequence,
`scipy.stats.qmc.Sobol(8, seed=20240318, optimization=None)` (scrambled by scipy's default), mapped affinely
onto the $\log_{10}$ bounds of Table 2 ($\omega$'s bounds are listed high to low, which reverses that coordinate
but not its range). With scipy 1.18.1 this reproduces the $\log_{10}$ training design to 2.1e-07, the 7-digit
print of the design files (notebook); each log10 input of the first 256 and of all 512 points fills all 256 and
512 equal strata. The paper's scalar emulators use $m$ = 256 training simulations (p. 4058); that they are the
first 256 rows, and that the Fig. 12 Sobol' indices come from these emulators, is from the authors' code
(`train_config.py`: `m = 256`; `experiments/synthetic/analysis/sensitivity_indices.py` reads rows
`[:config.m]`). Test: a 100-point centred-discrepancy-minimised Latin hypercube (p. 4052), also uniform in log
space (every value at a stratum centre; notebook). No run was filtered out (p. 4052).

**Published results** (no active subspace). Test-set errors of GP emulators fitted to 256 runs (Table 5,
p. 4059: RMSE 0.034, 0.097 and 90 km; MAPE 5.2 %, 8.8 %, 9.0 % for $f_Q$, $\log T_s$, $L_c$); first-order
and total Sobol' indices from 5120 emulator evaluations (Fig. 12, p. 4061; bars only). Text (p. 4060): $f_Q$
"controlled by the rate of cavity opening" ($r_b$), $T_s$ most sensitive to $k_s$, $L_c$ controlled by $l_c$,
very weak sensitivity to $\omega$.

## Regenerating

The runs cannot be regenerated here (GlaDS/ISSM is not installed; 216 h for the training runs). The CSV files
are regenerated from the deposit with `convert_data.py`. Because the archive is 5.2 GB, this was done as an
**external run, not reproducible from the repository without the download**:

- **Source.** Zenodo record 15053542, "timghill/GladsGP: v1.0.0" (published 2025-03-19, MIT),
  doi:10.5281/zenodo.15053542, the DOI of the paper's Code and data availability statement (p. 4070).
  File `GladsGP.tar.gz`: 5,157,897,717 bytes, MD5 `69f34e04de98634a5a46787f9c2dc8cd` (as listed by the
  record), SHA-256 `f17b96da1a469cb5f2be2663ed8174f6aef8f426585a5098e0b05c5d3561c450`; 539 members, 6.3 GB
  extracted. The SHA-256 of the eight members read are in `convert_data.py`.
- **Where.** An AWS EC2 `t3.large` (2 vCPU, 8 GB) in us-east-1, Ubuntu Server 24.04 LTS, 40 GB gp3 root
  volume, Python 3.12.3 with numpy 2.1.3, on 2026-11-09 (UTC). The download took about 6 min, checksums and
  extraction about 1 min, the conversion seconds; with set-up, about 15–20 min of instance time. At on-demand
  list prices (t3.large about $0.08 per hour plus the volume) the run costs a few cents.
- **Recipe.**

  ```
  python3 -m venv venv && venv/bin/pip install numpy
  curl -L -o GladsGP.tar.gz https://zenodo.org/api/records/15053542/files/GladsGP.tar.gz/content
  md5sum GladsGP.tar.gz            # 69f34e04de98634a5a46787f9c2dc8cd
  sha256sum GladsGP.tar.gz         # f17b96da1a469cb5f2be2663ed8174f6aef8f426585a5098e0b05c5d3561c450
  mkdir extracted && tar -xzf GladsGP.tar.gz -C extracted
  venv/bin/python -I convert_data.py extracted --out converted --check
  ```

  `--out` must be a directory outside this data set's directory; `--check` compares every value with the
  shipped files to a relative tolerance of 1e-12 and prints whether the text is identical. If the archive
  extracts into a subfolder, pass the folder that contains `experiments/`.
- **Result.** The shipped CSVs were written on that instance by a reduction script whose conversion is the
  one in `convert_data.py`; it also kept verbatim copies of the eight source files (about 138 KB). Those copies
  are not shipped: the CSVs are lossless re-serialisations of them, and the SHA-256 of each source file is
  pinned in `convert_data.py`, so a copy obtained from the deposit can be verified before it is converted.
  With Python 3.13.18 and numpy 2.5.4
  (2026-11-08), `convert_data.py <copies> --out <dir> --check` on those copies
  passes the SHA-256 checks and prints "text identical" for all four files (max relative difference
  0.0e+00, tolerance 1e-12) in under 1 s. The design CSVs of the deposit are byte-identical to those at
  github.com/timghill/GladsGP commit e7f143a.

The notebook runs in 20–40 s (22 s in two runs on 2026-11-08, up to 41 s with other jobs sharing two CPUs), most of it the 500
bootstrap replicates and the radial-basis surrogate fits.

## Accuracy

- Conversion: lossless (`convert_data.py --check`: text identical; float32 values written with 9 significant
  digits and float64 values with `repr`, each checked to reload bit for bit).
- Stored precision: the authors' arrays of $f_Q$ and $\log_{10} T_s$ are float64 and those of $L_c$ float32,
  but every stored value, the gate means included, is exactly representable in float32 (`convert_data.py`
  prints this check): the values, and the means, were rounded to float32. The mean columns equal the mean
  of the six gates to a relative 1.8e-07 or better (notebook). `convert_data.py` writes each value so that it
  reloads bit for bit in its stored type, so the precision is unaffected.
- Consistency (notebook): $f_Q$ in [2.27e-05, 0.9989] (train); $L_c(R \ge 1\,\mathrm{m}) \le L_c(R \ge
  0.5\,\mathrm{m})$ in every run; one training run has $L_c$ = 0. The test ranges match Table 5.
- Simulator accuracy: not quantified by the paper for these outputs. It notes (p. 4051) that for some
  parameter combinations the channel network may not reach a steady state within the two simulated melt
  seasons; no run is flagged, and the notebook keeps all runs, as the paper does.
- No gradients, so no gradient check. Fit quality in the notebook: the QPHD quadratic has leave-one-out
  $R^2$ 0.920, 0.928, 0.908 (test 0.945, 0.938, 0.914) for $f_Q$, $\log_{10} T_s$, $L_c$; radial-basis
  surrogates on the chosen active variables (n chosen by leave-one-out $R^2$) have leave-one-out $R^2$ 0.961,
  0.894, 0.926 and test $R^2$ 0.959, 0.911 (n = 1) and 0.919 (n = 2).

## How these files relate to the paper

The files are the paper's 512 training and 100 test runs, all of them, with every scalar output the deposit
stores. The paper fits GP emulators to 256 training runs (p. 4058; the first 256, per the authors' code) and computes
Sobol' indices from them; the notebook estimates active subspaces from all 512 training runs, chooses their
dimension by leave-one-out $R^2$ (n = 1, 1, 2), and uses the test runs only to check. It compares the order of
inputs by activity score with the bars of Fig. 12 (read from the figure, about ±0.03; Spearman rank correlation
0.97–0.99 over all eight inputs, 0.90–1.00 over the five largest bars), the sizes for $f_Q$ with n = 1 and n = 2
($h_b$: 0.112 of the summed bars, activity share 0.067 with n = 1 and 0.125 with n = 2), and the surrogate test
RMSE with Table 5 (whose 256-run setting the notebook repeats). Points the notebook notes:
the authors' Fig. 12 script (`experiments/synthetic/analysis/sensitivity_indices.py`, `cols = [5, 5, 0]`)
reads the 30 km gate of $f_Q$ and $\log_{10} T_s$, while the text and Table 5 use the gate means (the
notebook shows both); Table B1 prints $\omega$ = 1.3 × 10⁻⁴ for simulation 63, outside the Table 2 range, where
the file has 1.3 × 10⁻³; the $L_c$ unit is m in Table 1 and km in Table 5 and Fig. 12; the threshold text "R =
h_b = 0.5 m, corresponding to the maximum bed bump height" (p. 4053) disagrees with the $h_b$ range [0.05, 1]
m of Table 2.

## License and credit

Data: © Tim Hill and co-authors, Zenodo 10.5281/zenodo.15053542, **MIT licence** (the record's licence; the
paper says the deposit's "simulation outputs, trained emulators, example notebooks, and code" are available
under it, p. 4070). The four CSV files are therefore an exception to this repository's CC BY 4.0 data licence
and are listed in `LICENSE-DATA`; `GladsGP_LICENSE.txt` keeps the copyright and permission notice beside them.
Cite the paper and the Zenodo record when using these files. `convert_data.py` and the notebook: MIT (this
repository). No code of the authors is included; their aggregation and sensitivity scripts were read to
establish the meaning of the stored columns. Article: CC BY 4.0.

## Unknowns

- **Units of $k_s$ and $k_c$.** Table 2 prints Pa s⁻¹ and m^3/2 s⁻¹. Dimensional analysis of Eqs. (A1) and
  (A4), with the sheet flux in m² s⁻¹ and $\alpha_c = 5/4$, $\beta_c = 3/2$, gives Pa⁻¹ s⁻¹ and
  m^3/2 kg^-1/2. The CSV headers keep the printed units and say so; the values are unaffected.
- **Integration window of $f_Q$ and $T_s$.** The text says "melt-season-integrated"; the authors' script sums
  (and averages) over all stored daily outputs of the year. The stored values are used as they are.
- **Columns behind Fig. 12.** The authors' script (`sensitivity_indices.py`, `cols = [5, 5, 0]`; GitHub copy at
  commit e7f143a, the same size as the deposited file) reads the 30 km gate of $f_Q$ and $\log_{10} T_s$, while the
  figure's labels and the text (p. 4053) define gate means. Which was intended is a question for the authors; the
  notebook shows both.
- **Error bars of Fig. 12** (an inference from the authors' script, not verified by running it). The script passes
  the lower and upper bootstrap confidence limits as matplotlib `xerr`, which draws them as offsets from the bar
  (from S − low to S + high) rather than as the interval itself. If so, the true intervals are narrower than drawn.
  The notebook does not use the whiskers.
- **$\omega$ of Table B1, simulation 63.** Printed as 1.3 × 10⁻⁴, outside the Table 2 range; the file has
  1.3 × 10⁻³, which is presumably the intended value.
- **Gate mean of $\log T_s$.** The text describes an average time; the stored column is the mean of the six
  $\log_{10}$ values (the log of the geometric mean time). Table 5's range matches the stored column.
- **The $R$ threshold of $L_c$** (0.5 m, called the maximum bed bump height, while $h_b$ reaches 1 m).
- **Convergence of individual runs.** The paper notes that some runs may not reach a periodic channel network
  in two years (p. 4051) and stores no flag; all runs are kept.
- **Base of "log $T_s$"** is not stated in the paper; the stored values are log10 (the authors' script) and
  Table 5's test range matches them.
- **References** in the notebook are taken from the paper's reference list and the DOIs given there; they were
  not checked on Crossref (Crossref was not reachable when this data set was written).
