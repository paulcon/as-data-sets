# Lithium_Ion_Battery: data

Kind: **stored data**. The runs come from Hadigol et al.'s simulations of a Newman-model battery, and the text files are the only record of them. Their values are never regenerated or corrected. `battery_data.npz` is a lossless, scripted conversion of the columns the notebook uses.

## Files

| file | contents | role |
|---|---|---|
| `Lithium_Ion_Battery.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2016; outputs committed |
| `battery_figs.py` | `load`, `overshoot`, `weights` (Algorithm 1 via v1 OLS), `bootstrap_se`, the figure functions and the line-style tables | helper module |
| `battery_data.npz` | the used columns of the six text files below (8.6 MB) | what the notebook reads |
| `make_battery_npz.py` | writes `battery_data.npz` and checks it against the text files, value by value | the conversion |
| `0.25C.txt`, `1C.txt`, `4C.txt` | 3600 runs × 270 columns each, CSV with one header row (22.4 MB each) | originals (see "Originals and Zenodo") |
| `V_025C.txt`, `V_1C.txt`, `V_4C.txt` | 50 voltages [V], one per line, descending to 2.8 | originals: the voltage grid of the capacity columns |
| `README` | one-line pointer to the paper | |

md5 of the originals: `0.25C.txt` 6b79c1937889d1ec6eee09a97f5e6c84, `1C.txt` 7503e2027909253248df565778e76d5b, `4C.txt` 2c9ba7fcbcadd97aaf4d84cac5b8fec5, `V_025C.txt` 9d61d12b2a087a163cdec7cb775ca91c, `V_1C.txt` 92d665207ba6ab473c1639cef0d342e5, `V_4C.txt` c6f9009eb54a28b709c9165003297f20. The npz stores the same list as `source_md5`.

## Provenance

- **Generating code**: a finite-difference solver for Newman's porous-electrode model of a LiC₆/LiCoO₂ cell, set up by M. Hadigol, K. Maute and A. Doostan for their UQ study (J. Power Sources 300:507–524, 2015, doi:10.1016/j.jpowsour.2015.09.060). The code, its version and its settings are not recorded here.
- **Who ran it**: Hadigol et al., on the University of Colorado Boulder's JANUS supercomputer: about 20 min per run on one core, 3600 CPU-hours for the 10,800 runs (Constantine & Doostan 2017, §4). Dr. Hadigol provided the data set (the paper's Acknowledgments). Ryan Howard added the files to this repository on 2016-07-15 (commit 0604f0d); they have not changed since.
- **Sampling**: each run draws the 19 inputs independently from the distributions in the paper's Table 1, and the same 3600 input vectors are used at the three discharge rates (0.25C, 1C, 4C). The files store the inputs only after a shift and scale to [−1, 1].
- The paper (§4) also points to `bitbucket.org/paulcon/time-dependent-gsa-for-batteries` for the data and scripts; that link was not checked for this port.

## Columns of the text files

| columns | header | contents | units |
|---|---|---|---|
| 0 | `Run` | 1, …, 3600 | — |
| 1–19 | `$\epsilon_a$`, …, `$L_c$` | the inputs, normalized to [−1, 1], in the order ε_a, ε_s, ε_c, brugg_a, brugg_s, brugg_c, t₊⁰, D, D_a, D_c, σ_a, σ_c, k_a, k_c, r_a, r_c, L_a, L_s, L_c. Identical in the three files | — |
| 20–69 | `Vol01`–`Vol50` | cell voltage on the t* grid | V |
| 70–119 | `Cap01`–`Cap50` | elapsed discharge time when the voltage reaches each value of `V_<rate>.txt` (inferred) | s (inferred) |
| 120–269 | `ConA01`–`ConC50` | 50 columns each of anode, separator and cathode concentration (`ConA`, `ConS`, `ConC`) | unknown |

Values are written as `%.16e`, which float64 reads back exactly. No value is missing.

## battery_data.npz

| key | shape | dtype | contents |
|---|---|---|---|
| `X` | (3600, 19) | float64 | text columns 1–19 (stored once) |
| `voltage_025C`, `voltage_1C`, `voltage_4C` | (3600, 50) | float64 | `Vol01`–`Vol50` [V] |
| `time_025C`, `time_1C`, `time_4C` | (3600, 50) | float64 | `Cap01`–`Cap50`, unchanged [s, inferred] |
| `vgrid_025C`, `vgrid_1C`, `vgrid_4C` | (50,) | float64 | `V_<rate>.txt` [V] |
| `run` | (3600,) | int64 | `Run` |
| `input_names` | (19,) | str | the input headers, verbatim |
| `dropped_runs_025C`, `dropped_runs_1C`, `dropped_runs_4C` | (k,) | int64 | Run numbers flagged by the overshoot rule below: none, none, [1538, 2969] |
| `source_md5` | (6, 2) | str | file name and md5 of each original |

`python make_battery_npz.py` (from this directory, about 5 s) rewrites the file and checks that every array equals the text values exactly. The npz bytes differ between runs (zip timestamps); the arrays do not.

The concentration columns are not in the npz: nothing uses them, and their units and spatial locations are not documented.

## Units, grids and conversions

- **Inputs**: normalized coordinates in [−1, 1]. The notebook's table gives the physical units, nominal values and ranges from the paper's Table 1. For the 17 uniform inputs the physical value is presumably lo + (hi − lo)(x + 1)/2; the paper says only "shift and scale". There are no gradients: the analysis uses values only (Algorithms 1 and 2), in normalized coordinates.
- **t\* grid**: t* = 100 at the start of the discharge and 0 at the 2.8 V cutoff. The column-to-t* map is not recorded. The notebook keeps the 2016 code's labels, column k at t* = 99 − 2k (author, 2026-10-30), and analyzes columns 1–48. `Vol50` is within 0.5 mV of 2.8 V in every run that does not overshoot, which suggests the last column is at the cutoff itself.
- **Capacity**: the notebook converts `Cap*` to capacity in mAh cm⁻² as rate × I_1C × t / 36000 with I_1C = 30 A m⁻² (the 2016 code's factors 7.5/36000, 30/36000, 120/36000). That `Cap*` is time in seconds and that I_1C = 30 A m⁻² are **inferred, not confirmed** (author, 2026-10-30). They are consistent with the data: the largest 0.25C value is 13,906 s ≈ 3.9 h and the largest 1C value 3,442 s ≈ 1 h, and the resulting capacities match the paper's axes (up to 2.9 mAh cm⁻² at 0.25C).

## Dropped runs

Every run stops at the 2.8 V cutoff. Two 4C runs end below it: Run 1538 at 2.746 V (|z| = 59 among the final voltages) and Run 2969 at 2.791 V (|z| = 9.5). The rule "final voltage more than 1 mV below 2.8 V" flags exactly these two, and no 0.25C or 1C run; |z| > 5 flags the same two. The rows stay in the text files and in the npz arrays. The notebook applies the rule and checks it against `dropped_runs_<rate>`, so N = 3598 at 4C. The 2016 code dropped only Run 1538, without comment. The paper says N = 3600 for every fit. Dropping Run 2969 as well changes the 4C weights by at most 1.2e-4.

## Unknowns

- The simulation code, its version and its settings.
- The particle sizes r_a and r_c: the paper's Table 1 says N(2, 0.1354) and N(2, 0.3896), but the stored values are uniform on [−1, 1] like the other inputs (excess kurtosis −1.22 and −1.20). Unknown which is right, and how a Normal sample would have been mapped to [−1, 1] (author, 2026-10-30).
- That `Cap*` is elapsed time in seconds, and I_1C = 30 A m⁻² (see above).
- The t* value of each voltage column (see above).
- The units and positions of the concentration columns.

## Originals and Zenodo

The author decided (2026-10-30) that full originals of large data sets go to Zenodo with a DOI, and the repository ships the used columns. The deposit of the three text files (with the concentration columns) and the voltage grids, with Hadigol et al. as data authors, has not been made yet. Until it has a DOI, the original text files stay in this directory. The notebook reads only `battery_data.npz`; `make_battery_npz.py` is the only reader of the text files. When the DOI exists, the text files can be removed from HEAD and the DOI added here; their md5s above identify them.

## License and credit

Code (`*.py`, the notebook): MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`). The data are the work of M. Hadigol, K. Maute and A. Doostan: cite their J. Power Sources paper (above) when you use them. Cite the analysis as P. G. Constantine and A. Doostan, "Time-dependent global sensitivity analysis with active subspaces for a lithium ion battery model", Stat. Anal. Data Min. 10(5):243–262, 2017, doi:10.1002/sam.11347.
