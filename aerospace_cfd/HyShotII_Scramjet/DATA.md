# HyShotII_Scramjet: data

Kind: **stored data**. HyShot II scramjet simulations computed the exit pressures, and the two csv files are the only record of those runs. They must not be regenerated or corrected. The notebook selects rows by the `validation` flag and leaves the files unchanged.

## Files

| file | contents | role |
|---|---|---|
| `HyShotII_Scramjet.ipynb` | the case study (Python 3, active_subspaces v1) | executed top to bottom with seed 2015; outputs committed |
| `HyShotII_phi030.csv` | 52 runs at fuel plenum pressure 4.8 bar (φ = 0.30), Constantine et al., JCP 2015 | stored data |
| `HyShotII_phi035.csv` | 16 runs at φ = 0.30 and φ = 0.35, Constantine, *Active Subspaces*, SIAM 2015, §5.1 | stored data |
| `convert_legacy_data.py` | rewrites the 2016 files `HyShotII30.txt` and `HyShotII35.txt` (tag `legacy-master`, directory `HyShotII/`) as the two csv files, and checks the round trip | conversion script |

Both csv files are plain text: `#` header lines (source, units, normalization, the meaning of `validation`), a `# columns:` line with the column names and units, then one comma-separated row per run. Numbers are written with Python's shortest round-trip representation, and missing values as `nan`. `np.loadtxt(path, delimiter=",")` reads them.

## Columns

| column | meaning | unit | lower bound | upper bound |
|---|---|---|---|---|
| `run` | run number (row order of the 2016 file) | — | | |
| `angle_of_attack` | angle of attack α | deg | 2.6 | 4.6 |
| `turbulence_intensity` | inflow turbulence intensity | % | 0.1 | 1.9 |
| `turbulence_length_scale` | inflow turbulence length scale | mm | 122.5 | 367.5 |
| `stagnation_pressure` | stagnation pressure P₀ | Pa | 1.6448e7 | 1.9012e7 |
| `stagnation_enthalpy` | stagnation enthalpy H₀ | J/kg | 3.0551e6 | 3.4280e6 |
| `cowl_transition_location` | boundary-layer transition location on the cowl | m | 0.03 | 0.07 |
| `ramp_transition_location` | boundary-layer transition location on the intake ramp | m | 0.087 | 0.203 |
| `exit_pressure` (phi030 file); `exit_pressure_phi030`, `exit_pressure_phi035` (phi035 file) | pressure averaged over the last 10 mm of the combustor, 0.64 ≤ x ≤ 0.65 m (JCP 2015 eq. (15)) | bar | | |
| `validation` | 1 for the two runs made after the active subspace was computed, at opposite corners of the input box; 0 for the design runs | — | | |

- **Normalization bounds.** The inputs are independent and uniform between the bounds above. The notebooks normalize them to [−1, 1] with z = (x − (u + ℓ)/2) / ((u − ℓ)/2), the map of `ac.Uniform(lower, upper)`. The bounds are those of the 2016 notebook (cell 5, where the transition bounds were written as 0.6 and 1.4 times the nominal 0.05 m and 0.145 m). They match JCP 2015 Table 2 and book Table 5.1 except in units, and except for the turbulence length scale.
- **Units that differ from the published tables.** JCP 2015 Table 2 and book Table 5.1 give the turbulence intensity as a fraction (0.001 to 0.019) and the length scale in m. The data use percent (as the JCP text, §3.2, does) and mm.
- **Turbulence length scale.** Both tables print 0.1325 to 0.3575 m. The JCP text (§3.2) gives a nominal 245 mm ± 50 %, which is 122.5 to 367.5 mm, and the design runs reach 127.0 mm (book file) and 363.6 mm (JCP file), outside the printed range. The bounds here are 122.5 and 367.5 mm, and the printed range is a typo in both tables.
- **The two φ labels.** JCP 2015 runs at fuel plenum pressures P₀,H₂ = 4.8 bar and 5.6 bar, which give φ = 0.30 and 0.35 at nominal inflow. The book labels its runs by φ. The 5.6 bar runs of JCP 2015 (Table 3, second column) are not in this repository.

## Rows

- **`HyShotII_phi030.csv`** (JCP 2015, 4.8 bar): runs 1–50 are the design runs used to compute w (JCP Table 3 reproduces to 4 digits); runs 51–52 (`validation = 1`) are the corner runs that estimate the range of the exit pressure.
- **`HyShotII_phi035.csv`** (book §5.1): runs 1–14 are the design runs, the same input points for both φ (book Table 5.2 reproduces to 4 digits for both); runs 15–16 are the corner runs. Their φ = 0.30 exit pressures are **unavailable** (author, 2026-10-30) and stored as `nan`. The 2016 file held `0` placeholders there.
- **Corners.** Both files' corner runs are the same two input points, within 4e-6 of ±1 in normalized coordinates. They are not exactly ±sign(w) for the shipped w: the cowl-transition component has the opposite sign (w₆ < 0 for both φ). The notebook explains this and uses −sign(w) for the safe-box problem.
- **Two φ = 0.30 studies.** The book's 14 φ = 0.30 runs are not among the JCP runs, and they lie 0.06 to 0.10 bar above a linear fit to the 50 JCP design runs (printed by the notebook). Why the two φ = 0.30 studies differ is unknown.

## Conversion from the 2016 files

The 2016 files mixed coordinates: `HyShotII30.txt` had no header and stored all inputs normalized; `HyShotII35.txt` had physical inputs in runs 1–14 and normalized inputs in runs 15–16. `convert_legacy_data.py` (author decision, 2026-10-30) maps the normalized inputs to physical units with x = (u + ℓ)/2 + z (u − ℓ)/2 and writes both files. It reads the 2016 files from the tag `legacy-master` (or `--legacy-dir`), checks their SHA-256 hashes, and verifies:

- every exit pressure, and every input that was already physical, is bit-identical;
- the converted inputs, read back from the written text and normalized again, equal the 2016 normalized values to within 2e-15 (measured 1.4e-15). This is the float64 resolution of the physical values: one ulp of a stagnation pressure near 1.9e7 Pa is 1.5e-15 in normalized units, so the conversion cannot be bit-exact in both directions;
- the `nan` entries are exactly runs 15–16 of the φ = 0.30 column, and every converted input lies inside the bounds.

`python convert_legacy_data.py --check` repeats the conversion and compares it with the committed files byte for byte. The notebook's printed w equals the 2016 notebook's printed w and the published tables to 4 digits.

## Provenance and unknowns

- **Code.** Joe, the finite-volume RANS solver of Stanford's Center for Turbulence Research, with the k–ω SST turbulence model and a flamelet/progress-variable combustion model (JCP 2015 §2.2; the book cites that paper for the simulation). JCP 2015 reports about 9500 CPU-hours per run (a 2D intake simulation and a 3D combustor simulation); the book says each run takes "approximately two hours on available resources".
- **Who ran it.** The simulations were done for JCP 2015 at Stanford's Center for Turbulence Research (Michael Emory and Gianluca Iaccarino are the Stanford authors); the paper does not say who ran them. Ryan Howard added the 2016 files and notebook to this repository: the design runs on 2016-06-02 (commit 96221f4), the validation runs on 2016-06-03 (cd66a9e).
- **Unknown:**
  - the Joe version, the meshes, and the convergence criteria;
  - how the design points were drawn (independent uniform, Latin hypercube, …);
  - why the book's φ = 0.30 runs differ from the JCP runs, and where the JCP 5.6 bar runs are;
  - the exit pressures of the book's two φ = 0.30 corner runs (unavailable, author 2026-10-30).

## Regenerating

The data cannot be regenerated from this repository. The notebook reads the two csv files and computes everything from them in under a minute. The only randomness is the bootstrap (10 000 replicates for each φ) and the 100 000 uniform samples for the distribution functions, all from one `np.random.default_rng(2015)`. Two runs give identical printed output.

## License and credit

Code: MIT (`LICENSE`). Data (`HyShotII_phi030.csv`, `HyShotII_phi035.csv`): CC BY 4.0 (`LICENSE-DATA`). Cite P. G. Constantine, M. Emory, J. Larsson and G. Iaccarino, "Exploiting active subspaces to quantify uncertainty in the numerical simulation of the HyShot II scramjet", J. Comput. Phys. 302:1–20, 2015, doi:10.1016/j.jcp.2015.09.001, for the φ = 0.30 (4.8 bar) data, and P. G. Constantine, *Active Subspaces: Emerging Ideas for Dimension Reduction in Parameter Studies*, SIAM, 2015, §5.1, doi:10.1137/1.9781611973860, for the `HyShotII_phi035.csv` data.
