# CO2_Injection_Benchmark_Saturation: data

Kind: **stored data**. The saturations are outputs of the authors' finite-volume code, the Linux executable `prog` of the benchmark's Zenodo archive (no source), run for this repository on the archive's 10,000 input samples (2026-11-02). The files are the record of those runs; `generate_data.py` records how they were made and can remake them on Linux x86-64 (elsewhere it exits with a message saying what is needed). The input samples and the authors' Monte Carlo reference curves are stored data from the same archive, converted losslessly.

## Files

| file | contents | role |
|---|---|---|
| `CO2_Injection_Benchmark_Saturation.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom in `.venv` with `SEED = 933827` (bootstrap only); outputs committed; about 10 s |
| `benchmark_input_samples.csv` | 10,000 × 3: the coded samples `ir`, `kd`, `po` (injection rate, relative permeability degree, porosity), `InputParameters.csv` of the archive | stored inputs |
| `saturation_field.npz` | `S` (10,000 × 250): the CO₂ saturation at t = 100 days at the 250 cell positions for every sample, as `prog` printed it (17 significant digits, stored as float64); `r` (250): the cell positions in m, 0 to 250 in steps of 250/249 | stored outputs |
| `outputs.csv` | 10,000 × 5: `S` at cells 50, 75, 100, 125, 150 (r = 50.2, 75.3, 100.4, 125.5, 150.6 m), copied from `saturation_field.npz` | the same outputs as CSV |
| `mc_reference_mean_std.csv` | 250 × 9: the authors' Monte Carlo mean and standard deviation of S(r) for the joint case and the three single-input cases, `MCrefs_MeanStd.csv` of the archive | stored reference (paper Fig. 2) |
| `generate_data.py` | `run` (prog on the samples, in parallel chunks), `assemble` (the files above from the chunk outputs; `--check` compares with the shipped files), `archive` (the two stored CSVs from the archive) | provenance |
| `LICENSE-uqBenchmarkData.txt` | the archive's BSD 3-clause licence, unchanged | licence of the archive's files |

The CSVs have two `#` lines (a description, then the column names) and are read by `np.loadtxt(path, delimiter=',')`. Values are written with 17 significant digits; the two archive files round-trip the archive's values exactly (checked when written).

## Model

**Source.** M. Köppel, F. Franzelin, I. Kröker, S. Oladyshkin, G. Santin, D. Wittwar, A. Barth, B. Haasdonk, W. Nowak, D. Pflüger, C. Rohde, "Comparison of data-driven uncertainty quantification methods for a carbon dioxide storage benchmark scenario", *Computational Geosciences* 23(2):339–354, 2019, doi:10.1007/s10596-018-9785-x. Read from the open-access preprint arXiv:1802.03064v1 (8 Feb 2018); page and equation numbers below are the preprint's. Data and executables: Zenodo, doi:10.5281/zenodo.933827 (paper ref. [28]), file `uqBenchmarkData.zip`, 2,691,045 bytes, SHA-256 fa88bc303c8a58ce9e0155fa86e88897ad1c6806d0be9bc4972094e4da87001c, fetched 2026-11-01.

**Equations** (Sec. 2, pp. 3–5; Table 1). Radially symmetric, capillarity-free fractional-flow model of CO₂ injected into a homogeneous brine aquifer (a reduction of the Class et al. 2009 benchmark): incompressible immiscible phases, isothermal, constant fluid properties, analytic pressure (Eqs. 10, 13), so the total radial flux is Q/r with Q = C_p q_CO2 (1 + θ₁). The saturation obeys φ ∂S/∂t + (1/r) ∂(Q f(S; n))/∂r − q_CO2 = 0 with f = λ_g/(λ_g + λ_w), λ_g = Sⁿ/μ_g, λ_w = (1 − S)ⁿ/μ_w (Eqs. 5–6, 14–15), μ_g = 3.950·10⁻⁵ Pa s, μ_w = 2.535·10⁻⁴ Pa s, S(left) = 0.8, t = 100 days. Eq. (11) as printed (p. 4) has a minus sign before the flux term; its semi-discrete form Eq. (12) and the physics require a plus sign (a misprint). The authors solve it with a semi-discrete central-upwind finite-volume scheme (Kurganov and Petrova 2005) and second-order Runge–Kutta time stepping (Sec. 2.2).

**The executable.** `prog` (3,108,832 bytes, SHA-256 ca3773a458c753e408cb765beaebd369546e1551182cbad9c13eb564334c51b9; a statically linked x86-64 ELF for GNU/Linux 2.6.32) reads a 4-column parameter file and, with `-allS`, writes S at every grid point for every sample (`*__S_0000.dat`) plus the mean and variance over the samples. Its defaults, used here, are 250 grid points on [0, 250] m (`-N250 -Xmin0 -Xmax250`), t = 8.64·10⁶ s = 100 days, CFL parameter 0.5 and test case 5023 ("case 1+3+4": uncertain injection rate, "model uncertainty", porosity; `prog -h`). It reads columns 1, 3 and 4 of `InputParameters.txt` as (ir, kd, po) and prints them per sample in the order (ir, po, kd); column 2 holds residual-saturation samples that test case 5023 does not use. `InputParameters.txt` (SHA-256 c5b9e358ff7f54eca20b183110183ef534aac37135f2a4c1fed1a061fbf0707e) gives the samples to 8 significant digits; they differ from `benchmark_input_samples.csv` (the archive's `InputParameters.csv`) by at most 5·10⁻⁸. The number of time steps per sample ranged from 2,413 to 617,590 (the CFL step shrinks with small porosity and large injection rate).

**Inputs** (independent, given only as 10,000 coded samples; Sec. 3, Fig. 1). `ir` ∈ [0.047, 0.998] (injection rate θ₁), `kd` ∈ [0.0004, 0.9996] (relative permeability degree θ₂), `po` ∈ [−0.464, 3.398] (porosity θ₃); sample correlations below 0.01 in absolute value. The archive README calls them "the distribution of the uncertain parameters"; the maps from the codes to Q, n and φ are inside the executable and are not stated. The paper's Fig. 1 shows n on [2, 4], the porosity on about [0, 0.73] and a skewed injection-rate distribution; the samples come from data-driven distributions (Sec. 3). Fig. 1's bar heights do not match equal-width bin counts of the archived samples under any increasing affine map (porosity, archive `po`, 10 bins: 1257, 2574, 2504, 1826, 1056, 493, 201, 70, 16, 3; Fig. 1, read from a 600-dpi render of p. 6, ±20: about 1050, 2433, 2469, 1835, 1217, 587, 281, 100, 33, 6). The Monte Carlo reference was nevertheless computed on the archived samples (see Accuracy), so Fig. 1 is probably a histogram of another draw or of other bins (open).

**Quantities of interest.** The paper studies the mean and standard deviation of S(r, t = 100 d) over r (Fig. 2). The notebook analyses two scalars computed from `saturation_field.npz`: the front radius, the largest cell position with S > 0.01 (it moves in steps of the cell size and is capped at 250 m for the 190 samples whose plume reaches the end of the domain), and S at r = 100.4 m (cell 100), near where the reference standard deviation peaks (0.3095 at 98.4 m).

**Gradients.** None: `prog` gives values only. The notebook estimates the subspaces from local linear fits (8 nearest samples), with QPHD and OLS as checks of the direction.

**Sampling design.** The benchmark's 10,000 samples, as given (Monte Carlo samples from the data-driven distributions; Sec. 3, p. 5).

**Published results.** No active subspace. Monte Carlo mean and standard deviation of S(r) at 100 days for the joint and single-input cases (Fig. 2; numbers in `mc_reference_mean_std.csv`): joint mean 0.308 and standard deviation 0.309 at r = 100.4 m. Qualitatively (p. 5), the relative permeability degree does not influence the saturation significantly, and the injection rate and the porosity are the main sources of uncertainty. Four surrogate methods (aPC, adaptive sparse grids, kernel greedy, hybrid stochastic Galerkin) are compared against this reference (Secs. 4–5).

## Regenerating

```bash
# needs Linux x86-64 and the archive (Zenodo 10.5281/zenodo.933827); WORK outside this directory
unzip uqBenchmarkData.zip                                              # -> uqBenchmarkData/
python generate_data.py run --archive uqBenchmarkData --work WORK --jobs 16
python generate_data.py assemble --work WORK --check                   # print the largest differences from the shipped files
python generate_data.py --check --archive uqBenchmarkData --jobs 16    # run in a temporary WORK, then compare
python generate_data.py assemble --work WORK                           # or write them (here, or --out DIR)
python generate_data.py archive --archive uqBenchmarkData --out DIR    # the two stored CSVs
```

`run` checks the SHA-256 of `prog` and `InputParameters.txt` above, splits the 10,000 rows into chunks of 625 and runs `prog -pfile WORK/chunks/in_kk -path WORK/out_kk/ -allS -MC625` for each chunk in parallel. On any other system it exits with status 3 and a message. `--check` without a mode is `assemble --check`; without `--work` it first runs `prog` on all 10,000 samples in a temporary folder (as long as `run`), and without `--archive` as well it exits with status 3. `--check` prints the largest difference of the saturations from the shipped ones, relative to the largest saturation (0 for the shipped run: assembling its chunk outputs reproduces the files bit for bit; see below), and does not judge it; it exits with status 1 only on a structural mismatch: different shapes or NaN pattern, or a grid `r` that differs by more than a relative 1e-8.

**The run that wrote the shipped files** (2026-11-02): an AWS c7a.4xlarge (16 vCPUs, AMD EPYC), Ubuntu 24.04, 16 chunks in parallel (the same split as `run`, with `split -l 625`). The chunks took 7.1–8.0 hours of wall time each, 28,927 s in all, about 44 s of CPU per sample on average (the cost grows with the number of time steps). `prog` keeps every sample's result in memory and writes the files when its chunk ends. `assemble --check` in `.venv` against the chunk outputs prints identical `S`, `r` and `outputs.csv` (under a second). `archive` in `.venv` rewrote the two archive files with the data lines unchanged.

## Accuracy

- **The runs are the paper's reference runs.** The mean and standard deviation of S over the 10,000 samples (with the 1/(N − 1) normalization) agree with the joint columns of `mc_reference_mean_std.csv` to 5·10⁻⁹ at all 250 cell positions (notebook), the precision of the archive file (8 significant digits). With 1/N they differ by up to 1.5·10⁻⁵. So the authors' Monte Carlo reference was computed with this executable on these samples, and these files are, sample by sample, the runs behind the paper's Fig. 2.
- **Discretization.** 250 points on [0, 250] m (spacing 1.004 m) is the code's default and, by the agreement above, the authors' setting. The front is smeared over a few cells by the shock-capturing scheme; the front radius is defined by a threshold (S > 0.01) and is quantized to the cell spacing. No refinement study was made (the source is not available, and the paper's reference uses this grid).
- **Inputs.** `prog` ran on the 8-digit samples of `InputParameters.txt`, which differ from `benchmark_input_samples.csv` by at most 5·10⁻⁸; the notebook uses the CSV. The effect on the outputs is far below the variation over the samples.
- **Reproducibility.** `generate_data.py assemble --check` reproduces the shipped files exactly from the chunk outputs. A second run of `prog` was not made (8 hours); the code is deterministic as far as known.

## How these files relate to the paper

- **Same code, same samples, same grid.** These are runs of the authors' executable on the archive's samples, and their statistics equal the paper's reference. The paper publishes only the mean and standard deviation, not the per-sample outputs; this data set adds the 10,000 saturation fields.
- **Not studied in the paper.** The single-input cases of Fig. 2 (columns `eir`, `sir`, `ekd`, `skd`, `epo`, `spo`) hold the other inputs at values the paper does not state; they were not rerun.
- **Coded inputs.** The physical meaning of the codes is fixed inside the executable. The notebook analyses the coded inputs, so the active subspace is in the codes' coordinates (each scaled to [−1, 1] by its sample range); the ranking of the inputs does not depend on the unknown maps if they are monotone, but the eigenvector weights do.

## Open questions

1. The maps from `ir`, `kd`, `po` to the injection rate, the relative permeability degree and the porosity.
2. Why the bar heights of Fig. 1 differ from the archived samples (above).
3. The source term −q_CO2 in Eqs. (11)–(12): how the executable applies it (in the domain or only at the well).

## License and credit

Code (`generate_data.py`, the notebook): MIT (`LICENSE`). Data computed here (`saturation_field.npz`, `outputs.csv`): CC BY 4.0. The archive's files (`benchmark_input_samples.csv`, `mc_reference_mean_std.csv`) and the executable that computed the saturations: copyright 2017 Fabian Franzelin, Markus Koeppel, Ilja Kroeker, Sergey Oladyshkin, Gabriele Santin, Dominik Wittwar, under the BSD 3-clause licence in `LICENSE-uqBenchmarkData.txt`, which asks users to cite the benchmark paper (cited there under its working title, "Transfer and intercomparison of data-driven uncertainty quantification methods for two-phase flow in carbon dioxide storage"; published as the paper above); they are listed in the exceptions of `LICENSE-DATA`. The executable is not redistributed here. Cite M. Köppel et al., Comput. Geosci. 23(2):339–354, 2019, doi:10.1007/s10596-018-9785-x, and the Zenodo record. The paper's work was funded by the DFG within the Cluster of Excellence in Simulation Technology (EXC 310/2), University of Stuttgart.
