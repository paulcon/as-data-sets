# Epoxy_MD_Force_Field_Sensitivity: data

Kind: **stored data** (converted from outside files). The values were computed by Edeling et al. (2024) with
LAMMPS molecular dynamics of a cross-linked epoxy resin on SuperMUC-NG and published as CSV files in the
authors' repository [wedeling/MD-active-subspace](https://github.com/wedeling/MD-active-subspace), tag 1.1
(MIT licence), archived at Zenodo, [doi:10.5281/zenodo.10817481](https://doi.org/10.5281/zenodo.10817481).
They were not regenerated (each of the 500 input points was run as 20 MD replicas on a supercomputer) and are
not altered: `convert_data.py` copies every number character for character into the CSV layout of this
repository.

## Files

| file | contents | role |
|---|---|---|
| `Epoxy_MD_Force_Field_Sensitivity.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 195; outputs committed |
| `epoxy_inputs.csv` | 500 × 103: the sampled force-field coefficients, physical units (names and units in the second `#` line) | authors' `epoxy/data_epoxy/epoxy_inputs.csv` |
| `epoxy_defaults.csv` | 1 × 103: the default value $\bar x_i$ of each input (same columns) | authors' `epoxy/epoxy.params.json` (the 103 sampled entries) |
| `epoxy_E11_replicas.csv` | 500 × 20: $E_{11}$ [Pa] of the 20 replicas at each input point (the paper's "Young's modulus $E$"; see Model) | `epoxy_E11_output.csv` |
| `epoxy_E22_replicas.csv`, `epoxy_E33_replicas.csv` | 500 × 20: $E_{22}$, $E_{33}$ [Pa] | `epoxy_E22_output.csv`, `epoxy_E33_output.csv` |
| `epoxy_poisson_replicas.csv` | 500 × 20: Poisson ratio [1] | `epoxy_poisson_output.csv` |
| `epoxy_bulk_replicas.csv`, `epoxy_shear2_replicas.csv` | 500 × 20: bulk modulus $K$ and shear modulus $G$ (authors' label `shear2`) [Pa] | `epoxy_bulk_output.csv`, `epoxy_shear2_output.csv` |
| `convert_data.py` | checks the SHA-256 of the eight source files and writes the eight CSVs into `--out DIR`; `--check` compares them with the shipped files | the conversion |
| `check_accuracy.py` | recomputes the numbers of this file that neither the notebook nor `convert_data.py` prints | check |
| `LICENSE-MD-active-subspace.txt` | the MIT licence of the authors' repository, kept beside the files converted from it | licence |

Row $i$ of every output file belongs to row $i$ of `epoxy_inputs.csv`; column $r$ is replica $r$. The CSVs are
plain comma-separated numbers after two `#` lines (a description, then the column names with units), so
`np.loadtxt(path, delimiter=',')` reads them. The notebook uses `epoxy_inputs.csv`, `epoxy_defaults.csv`, the
$E_{11}$ and Poisson files, and (to identify what $E_{11}$ is) the $E_{22}$, $E_{33}$, bulk and shear files.

The authors' repository also holds the ESMACS and TIES binding-free-energy data sets of the same paper
(NAMD; 153–169 inputs, 215–428 points), their KAS-GP results and the MD set-up files. They are not used here
and not shipped.

## Model

**Citation.** W. Edeling, M. Vassaux, Y. Yang, S. Wan, S. Guillas, P. V. Coveney, "Global ranking of the
sensitivity of interaction potential contributions within classical molecular dynamics force fields",
*npj Computational Materials* 10:87 (2024), doi:10.1038/s41524-024-01272-z (publisher PDF read, open access,
CC BY 4.0). Code and data: the repository above, tag 1.1 (commit 76784ca, 2024-03-14), whose Jupyter
notebooks reproduce the paper's DAS (EasySurrogate) and KAS-GP (mogp-emulator) results. The paper's Data and
Code availability statements (p. 11) point to the Zenodo record of the same release (ref. 75).

**System and simulation** (Methods, pp. 9–10). Tetraglycidyl methylene dianiline (TGMDA) cured with
polyetheramine (PEA), 1:1, packed and cross-linked with an in-house code; "an ensemble of 20 replicas for which
91.8% ± 0.5% of the functional groups have reacted". LAMMPS on SuperMUC-NG, "following the ELASTIC procedure
from the LAMMPS examples database". The authors' `molecular_models/epoxy_elasticity/simulation.in` (not
shipped) sets `units real`, `pair_style lj/cut/coul/long 12.0 9.0`, `bond_style harmonic`, `angle_style
harmonic`, `dihedral_style opls`, PPPM (accuracy 1e-4), SHAKE with a 2 fs step and an NVT run of 100,000 steps
at 300 K; `epoxy.data` (36,822 atoms, 26 atom types) holds the structure and the default coefficients.
Replicas are runs "at the same epistemic parameters and a different random seed" (p. 2). Cost per run is not
reported.

**Inputs** ($m = 103$; Table 1, p. 3). Force-field coefficients only, each independent and uniform,
$x_i \sim \mathcal{U}(0.85\,\bar x_i, 1.15\,\bar x_i)$ (p. 3); partial charges are not varied (p. 3). Names and
defaults are the authors' (`epoxy.params.json`); the paper's Supplementary Table 1 (not read) lists them. The
role of each name and its unit are **inferred**: the first letter names the term, the last digit the
coefficient, and each class of defaults equals one coefficient row of the matching section of `epoxy.data`
(`check_accuracy.py`: 9 of 9 pair, 10 of 10 bond, 15 of 15 angle, 17 of 17 dihedral classes); the units are
those of LAMMPS `units real` for these styles.

| names | count | coefficient | units | defaults | source |
|---|---|---|---|---|---|
| p11 … p91 | 9 | Lennard-Jones well depth $\varepsilon$ | kcal/mol | 0.01–0.17 | Table 1, p. 3; `epoxy.params.json` |
| p12 … p92 | 9 | Lennard-Jones $\sigma$ | Å | 2.42–3.55 | same |
| b11 … b101 | 10 | bond stiffness $K$ in $K(r - r_0)^2$ | kcal/mol/Å² | 268–553 | same |
| b12 … b102 | 10 | bond length $r_0$ | Å | 0.945–1.529 | same |
| a11 … a151 | 15 | angle stiffness $K$ in $K(\theta - \theta_0)^2$ | kcal/mol/rad² | 33–70 | same |
| a12 … a152 | 15 | equilibrium angle $\theta_0$ | deg | 106.4–120 | same |
| d11 … d181 | 35 | nonzero OPLS coefficients $K_1$–$K_4$ of 17 dihedral classes | kcal/mol | −7.582 to 8 | same |

The 37 OPLS coefficients whose default is zero (all four of class d7 among them) are not varied. 14 dihedral
coefficients have a negative default (d11, d21, d22, d31, d41, d42, d51, d52, d61, d62, d142, d152, d162,
d172); their interval runs from $1.15\,\bar x_i$ to $0.85\,\bar x_i$. The paper's temperature exception
(±7.5 %, p. 3) concerns the binding-energy studies, which vary simulation parameters; the epoxy study varies
force-field parameters only (Table 1). All sampled values lie between 0.850001 and 1.149988 times their
defaults (notebook).

**Quantities of interest.** Six elastic outputs per replica: `E11`, `E22`, `E33`, `poisson`, `bulk`, `shear2`.
The paper analyses "the Young's modulus" $E$ and the Poisson ratio (pp. 3–6, Figs. 1–5, Table 2), using the
file `E11` for $E$; the authors' notebook calls it "the 11 component of the elasticity tensor (equal to the
Young's modulus E in this isotropic case)". The paper's text does not say how $E$ was computed. **The file the
paper (and the authors' notebook) uses as $E$ is the stiffness component $C_{11}$, not an isotropic Young's
modulus**: for every run, bulk + 4/3 · shear2 equals the mean of `E11`, `E22`, `E33`, and
`poisson` equals $(3K - 2G)/(2(3K + G))$, to relative deviations of order 1e-15 (notebook). These are the
cubic-averaged quantities of the LAMMPS ELASTIC example ($K = (\bar C_{11} + 2\bar C_{12})/3$,
$G = (\bar C_{11} - \bar C_{12})/2$). The isotropic Young's modulus $9KG/(3K + G)$ is a median 0.605 of `E11`
(range 0.344 to 1.202 over the 10,000 runs; notebook). Ratios above 1, impossible for an isotropic $C_{11}$,
come from the anisotropy of single runs ($E_{11}/E_{22}$ from 0.790 to 1.516, 1st to 99th percentile; see
Accuracy). The QoIs here are the replica means of `E11` (called $E_{11}$) and of the Poisson ratio, as in the
paper (p. 10: "ensemble-averaged over the replicas unless otherwise noted"). Units: Pa (the paper's Fig. 3
axis "E [Pa]") and dimensionless.

**Gradients.** None. MD gives no derivatives with respect to the force-field coefficients, and a
finite-difference gradient would need 104 points of 20 replicas each. The paper works without gradients too
(DAS and KAS-GP, pp. 4, 10).

**Sampling design.** Simple Monte Carlo: 500 independent draws from the product of uniforms, generated with
EasyVVUQ (pp. 3, 10), each point run as 20 replicas (Table 1), 10,000 runs in all (Table 2 caption). Every row
is used: the files have no missing values and the paper reports no failed epoxy runs (Table 1: $n_{data}$ = 500). One single-replica
Poisson value is negative (in one row; notebook); it is kept.

**Published results** (all for the replica means). Eigenvalues $\lambda_i/\lambda_1$, $i \le 4$, of the DAS and
KAS-GP gradient matrices for $E$ (Fig. 2a, p. 6; only a plot; the KAS-GP values are in the authors' file
`epoxy/GP_eigvals/scaled_eigen_results_epoxy_E11.csv`: 1, 0.0865, 0.0107, 0.0061); "an eigengap λ1 − λ2 of one
order of magnitude" for DAS and the choice $d = 1$ (p. 4); 1-D DAS and KAS-GP surrogates along $y_1$ (Fig. 3a,
b); DAS Poisson surrogate with and without replica averaging (Fig. 4); the 25 largest derivative-based indices
$\nu_i$ (Fig. 5a, b); moments over all 10,000 runs (Table 2: $E$ 6.54e9, 3.22e9, skew 1.61, excess kurtosis
3.76; Poisson 0.34, 0.053, −1.01, 0.97); a vector-valued DAS for four outputs (Supplementary Section 2, not
read). No eigenvectors are published.

## Regenerating

The MD runs cannot be regenerated from this repository (LAMMPS on a supercomputer, 10,000 runs; the strain
protocol of the ELASTIC procedure and the replica structures are only partly in the authors' files). The CSVs
are regenerated from the authors' files, with a SHA-256 check of each source file:

```
git clone --branch 1.1 https://github.com/wedeling/MD-active-subspace.git
python convert_data.py MD-active-subspace/epoxy --out /tmp/epoxy_csv            # writes the eight CSVs
python convert_data.py MD-active-subspace/epoxy --out /tmp/epoxy_csv --check    # and compares with the shipped files
```

`--out` must be outside this directory (the script refuses otherwise). `--check` compares every value with the
shipped file with a relative tolerance of 1e-15 and reports whether the texts are identical; on 2026-11-09 it
reported a maximum relative difference of 0.0e+00 and identical text for all eight files, in under 1 s. The
Zenodo archive of the same release (doi:10.5281/zenodo.10817481; not opened for this data set) should give the same files; the SHA-256 checks decide.

**Source files** (tag 1.1, commit 76784ca44570380ba9c0edd6180486d9497f2636), SHA-256:

| file | SHA-256 |
|---|---|
| `epoxy/data_epoxy/epoxy_inputs.csv` | `ab1989deb15ef572eb03bc2d38a338daa881e2e12743540d79ab912eb908fab4` |
| `epoxy/data_epoxy/epoxy_E11_output.csv` | `c142ba216c4a42eeafca8f16bbf998fcbc3b9081797bbbec890ce7f1e57ea6e3` |
| `epoxy/data_epoxy/epoxy_E22_output.csv` | `fd943c60baab60a3f29d6da9d74e7925838781efebbabc034feaba7334cc9033` |
| `epoxy/data_epoxy/epoxy_E33_output.csv` | `da6c15697ba228e046ccebb5db517f5a6560bd5bc954f5bda6fbbdb81ca4d522` |
| `epoxy/data_epoxy/epoxy_poisson_output.csv` | `9495fdd7eb052a6fe0e4330ac54aac26d49249eb8e1bfc45ecee6d45ece650b2` |
| `epoxy/data_epoxy/epoxy_bulk_output.csv` | `090c559dcc6b50c73555dbfb248fece9ac466831c5757dbd3fc91715c76c1195` |
| `epoxy/data_epoxy/epoxy_shear2_output.csv` | `90598ade21e4c9be92d229a0302fd4a271981024632f7676e37b5461ba66f44a` |
| `epoxy/epoxy.params.json` | `68feaa1a31389fe75bcfa5a51d8773c1d99d098f426914a73f72bc64e607fc2a` |
| `molecular_models/epoxy_elasticity/epoxy.data` (check 4 of `check_accuracy.py` only) | `7c38ddf03da84829398c969f2a8c8458c8441e7bf2283ef83933be4c0f6f54a3` |

The notebook runs in about 9–19 min in the analysis environment (512 s, 697 s and 1104 s in three runs on 2026-11-09, other jobs sharing two
CPUs; the notebook prints its run time); most of it is the 200-replicate OPG bootstrap for both QoIs (554 s in
the committed run) and the null and neighbourhood-size studies.

## Accuracy

`check_accuracy.py` recomputes the numbers marked with its name in under 1 s (check 4 needs `--source` with a
clone of the authors' repository).

- Conversion: lossless (`convert_data.py --check`, above).
- Stored precision (`check_accuracy.py`): at most 15 significant digits in inputs and outputs, far below any
  difference that matters here.
- Replica noise (notebook): the noise variance of a 20-replica mean is a share 8.67e-04 ($E_{11}$) and 6.76e-03
  (Poisson) of the variance of the means across the 500 points, so no function of the inputs can exceed $R^2$
  = 0.9991 and 0.9932; for a single replica the shares are 0.017 and 0.111. Median replica coefficient of
  variation of $E_{11}$ 0.070, median replica standard deviation of the Poisson ratio 0.0136
  (`check_accuracy.py`).
- Per-run anisotropy (`check_accuracy.py`; also printed by the notebook): $E_{11}/E_{22}$ has median 1.032 and
  1–99 % range 0.790 to 1.516 over the 10,000 runs, and median 1.031 and 1–99 % range 0.960 to 1.202 over the
  500 replica means, so single runs are far from isotropic and `E11` of one run carries direction-dependent
  noise that the cubic average used for bulk, shear and Poisson does not. Averaging the 20 replicas leaves a
  systematic offset: $E_{11}$ is about 3 % above $E_{22}$ in the median replica mean, which suggests that the
  replicas share structures or that the strain protocol is directional; either way the QoI depends on which
  direction is called 1.
- No gradients, so no gradient check. The estimator (OPG, local linear fits on 175 neighbours) is checked
  against nulls in the notebook: replica noise alone gives eigenvalues of 3.5e-04 ($E_{11}$) and 2.5e-03
  (Poisson) of the data's $\lambda_1$ (medians of 30 draws); an exact one-dimensional ridge with replica noise
  gives $\lambda_2/\lambda_1$ = 0.0074 and 0.0026 against the data's 0.0101 and 0.0046; and the data's
  $\lambda_2/\lambda_1$ changes from 0.0233 to 0.0014 ($E_{11}$) as the neighbourhood grows from 130 to 400
  points, i.e. $\lambda_1/\lambda_2$ from 42.9 to 739.1 ($E_{11}$) and from 81.5 to 2144.8 (Poisson), while
  $\lambda_2/\lambda_3$ stays between 1.04 and 1.27 (notebook). **Eigenvalue ratios beyond the first are set largely by the estimator**; the notebook reports them
  with these calibrations and does not treat them as properties of the model.

## How these files relate to the paper

The files are the 500 × 20 epoxy runs behind every epoxy result of the paper (Table 1, Figs. 1–5, Table 2); the
notebook reproduces Table 2's moments to the printed precision. The paper estimates the subspace with a deep
neural network (DAS) and with kernel dimension reduction plus a GP (KAS-GP); the notebook uses OPG, with OLS as
a check of the direction, and compares eigenvalues (Fig. 2a), the dimension (p. 4), the sufficient summary
plots (Fig. 3a, b), the effect of replica averaging (Fig. 4) and the top-10 sensitivity rankings (Fig. 5a, b).
Agreements: one active direction; the same ten leading inputs for $E$ (10 of 10) and 9 of 10 for the Poisson
ratio; 8 inputs shared between the two top-10 lists, as on p. 6; a scatter about the 1-D trend that doubles in
standard deviation for one seed against the replica mean (Poisson). Eigenvalues: the comparison depends on the OPG neighbourhood size $p$. At $p$ = 130, OPG's
$\lambda_2/\lambda_1$ = 0.0233 for $E_{11}$ lies inside the DAS 95 % interval of Fig. 2a (about 0.01 to 0.08 around
about 0.03, read from the plot) and its $\lambda_3/\lambda_1$ = 0.0207 is above DAS's ≈ 0.01; at the default
$p$ = 175 and above, OPG's $\lambda_2/\lambda_1$ is smaller (0.0101 at $p$ = 175). KAS-GP's 0.0865 is larger than
OPG's at every $p$ tried. The gap $\lambda_1/\lambda_2$ is 42.9 to 739.1 over $p$ = 130 to 400 against the paper's
"one order of magnitude" (DAS, p. 4); the notebook shows that this ratio depends on the estimator's smoothing,
and rests the choice $n = 1$ on the flat tail, the nulls and the LOO $R^2$ instead. The Poisson eigenvalues are not in the paper (the authors' file
`scaled_eigen_results_epoxy_poisson.csv` gives KAS-GP 1, 0.0943, 0.0391, 0.0084; OPG gives 0.0046 for
$\lambda_2/\lambda_1$). Two statements of the paper are qualified by the files: the file it uses as "Young's
modulus" is the stiffness component $C_{11}$ (see Model); and p. 6 says b12 and b42 "refer to the equilibrium
distance of C-H and C-C atom types" (without "respectively"); by the defaults, b12 (1.529 Å) is a C–C and b42
(1.09 Å) a C–H bond length.

The 14 negative-default dihedral inputs are normalized here with the interval's lower end at $1.15\,\bar x_i$;
the authors' notebooks use $0.85\,\bar x_i$ as the "lower" end for all inputs, so for these 14 inputs their
normalized coordinate is the negative of this one. Eigenvalues and $\nu_i$ are unaffected; the signs of those 14
eigenvector components are reversed relative to the authors' coordinates.

## License and credit

Data: © Edeling et al., from github.com/wedeling/MD-active-subspace (tag 1.1), MIT licence (Copyright (c) 2023
Wouter Edeling; `LICENSE-MD-active-subspace.txt`, kept beside the files); the Zenodo record of the same
release is listed as CC BY 4.0 (not verified here). Article: CC BY 4.0. Cite the paper and the Zenodo record
when using these files. `convert_data.py`, `check_accuracy.py` and the notebook: MIT (this repository). No code
of the authors' repository is included; the notebook reimplements nothing of theirs (it uses
active_subspaces v1).

## Unknowns

- What the authors mean by "Young's modulus": the files show it is $C_{11}$ of each run; whether the paper's
  conclusions about "$E$" were meant for the isotropic Young's modulus is not stated.
- Which atom, bond, angle and dihedral types each class index (p1–p9, b1–b10, a1–a15, d1–d18) stands for, and
  whether a sampled value scales every LAMMPS type that shares the class's default coefficients (Supplementary
  Table 1, not read; the inference above rests on equal default values).
- What a "replica" varies: p. 2 says only the random seed changes, p. 10 that 20 replica structures were built
  with 91.8 % ± 0.5 % of the functional groups reacted; whether each replica is a different cross-linked
  structure, and whether replica $r$ uses the same structure at every input point, is not stated.
- The strain amplitude, the number of strain steps and the MD lengths of the ELASTIC stage (the paper says "low
  strain amplitudes", p. 6; the shipped `simulation.in` holds only the equilibration).
- Cost per run (core-hours) is not reported.
- Why $E_{11}$ stays about 3 % above $E_{22}$ after averaging the 20 replicas (median 1.031 over the 500 replica
  means; Accuracy): shared replica structures or a directional strain protocol.
- The DAS eigenvalues of Fig. 2a are read from the plot (≈ 0.03, 0.01, 0.004 for $i$ = 2–4); the authors'
  notebook recomputes them by training networks, which was not done here.

### Open questions for the authors

- Is `E11` intended as the Young's modulus, and would the analysis of the isotropic Young's modulus
  $9KG/(3K+G)$ give the same ranking?
- Does each replica index correspond to one fixed cross-linked structure across the 500 input points? The
  persistent 3 % $E_{11}/E_{22}$ offset of the replica means suggests so.
- What were the ELASTIC strain amplitude and run lengths?
