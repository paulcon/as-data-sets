# Porting `check_accuracy.py` to the older data sets

Instructions for a later PR (author, 2026-11-03), written so that Claude, or anyone else, can do it without the history of this repository. Until that PR, the numbers listed below are labelled "external run, not reproducible from the repository" in their DATA.md.

## The rule

`docs/README.md` ("Two kinds of data sets", decision A8 of 2026-11-02, revised): every number in a DATA.md Accuracy section that neither the notebook nor the generating or converting script prints comes from a `check_accuracy.py` in the data-set directory. If the check needs an outside code, the script runs where that code is installed and otherwise exits with status 1 and a message naming what to install and saying to rerun it. A number that no script here can produce (a commercial code, a lost run) keeps the label "external run, not reproducible from the repository".

The data sets shipped in #64 and #68 already follow the rule (32 scripts). The ones below, from #56, #60 and the 2016 ports, do not.

## Where the original checks are

The checks behind these numbers were run while the data sets were built, outside the repository. Their notes are in `~/Documents/Claude/ActiveSubspacesDiscovered/builds/<domain>/`:

- `<Name>.ema/MODEL_NOTES.md`: the build notes. They describe the checks (settings, rows, steps, tolerances) and usually give the numbers. The scratch scripts themselves were mostly not kept. The one exception is Lillgrund's `floris_check.py`, which compares `lillgrund_model.py` with FLORIS 2.0.1.
- `<Name>.ema/*_ema.py`, `model.json`, `space.json`: wrappers for the build tool. **Do not port these**, and do not name them, the `.ema` folders or "EMA" in the repository (A7).
- `_reviews/<Name>.md` and `<Name>.fixes.md`: the build-time review and the fixes made after it.
- `~/Documents/Claude/ActiveSubspacesDiscovered/review-2026-11-01/details/E1.md`–`E6.md`: the review of 2026-11-01. Some of its checks recompute these numbers independently. For example, E5 recomputed the Vlasov continuation check.

The build-folder material is a source, not something to copy (C5): write each script fresh, against the files and modules in the repository.

## How to write one

Follow an existing script. `uq_benchmarks/Hat_Stiffened_Panel_Shear_Buckling/check_accuracy.py` is the numpy-only pattern. `electronics_circuits/Two_Stage_CMOS_Op_Amp/check_accuracy.py` is the pattern for a check that needs an outside code.

1. Read the data set's DATA.md and list every number labelled "external run, not reproducible from the repository".
2. For each number, find in `MODEL_NOTES.md` how it was computed: which rows, which steps, which reference (a tighter solver tolerance, a complex step, finite differences, an outside code).
3. Write `check_accuracy.py` in the data-set directory:
   - The docstring gives the command, the run time, and a numbered list of the checks. Each check states what it computes and the DATA.md value it reproduces. It also names the numbers that the notebook or `generate_data.py` already prints, and leaves them out of the script.
   - It imports numpy, scipy and the data set's own modules only (by path relative to `__file__`). If an outside code is needed, it stops with an install message when the code is missing.
   - It is deterministic (fixed rows or seeds), reads the shipped files, and writes nothing into the directory.
   - It prints each result beside the value DATA.md quotes.
4. Run it in `.venv` from the data-set directory, twice. If a number differs from DATA.md beyond rounding, DATA.md takes the `.venv` value (CLAUDE.md: report `.venv` results only). Quote rounding-level numbers as orders of magnitude. An outside-code check may run in a throwaway environment, never in `.venv`. DATA.md then names the platform and package version (author, 2026-11-02).
5. In DATA.md, replace each "external run" label with the script and check number, e.g. "(`check_accuracy.py`, check 2)". Add the script to the Files table, with its run time.
6. Leave the notebooks unchanged unless a quoted number changed.
7. Record the PR in `docs/TODO.md` (A8).

## The data sets and numbers

Labelled "external run, not reproducible from the repository" in DATA.md as of #70 (2026-11-03). The build-folder notes are in `builds/<domain>/<Name>.ema/MODEL_NOTES.md` unless noted.

| Data set | Numbers to reproduce (DATA.md section) | Outside code | Notes |
|---|---|---|---|
| `biomedical_cardiovascular/Hodgkin_Huxley_Action_Potential` | the I = 130–170 µA/cm² scan of nominal trajectories (choice 3); the printed-form vs H-H-form comparison on 300 samples (choice 5); the first two smoothness checks; the first three accuracy checks (tighter Radau solves) | none | the third smoothness check and the last accuracy check are in the notebook |
| `epidemiology_systems_biology/HIV_Long_Term_Dynamics` | the checks against tighter solves | none | no `.ema` folder; use the 2016 spec `docs/port-2026/HIV.md` and the HIV port PR (#51). Also say how the 2016-file comparison is made (files at `legacy-master`) |
| `epidemiology_systems_biology/STAT5_Dimerization_Signaling` | the wide-bounds stiffness check (study box); the first four accuracy checks | none for the checks; the PEtab benchmark (2.0e-6, NLL 138.222) needs the benchmark repository, which can be fetched read-only | |
| `epidemiology_systems_biology/Zika_SEIR_SEI_Brazil` | DOP853 vs Radau at seven rows (outputs); the [0, 5e-3] range test (problem 3) | none | |
| `plasma_fusion/MHD_Generator` | the 2016-gradient comparison (1e-13) and the 9-median-residual check of label 73 | none | no `.ema` folder; the 2016 `gradients.py` is at `legacy-master`; read `docs/port-2026/MHD.md` |
| `plasma_fusion/Vlasov_Poisson_Instability_Growth_Rate` | the wide-box stress test (failure rates); gradients vs finite differences at 60 rows per file; the continuation check of the stable Bump-on-Tail samples | none | the review (E5) recomputed the continuation check on 97 samples (2.5e-14) |
| `structural_materials_manufacturing/Steel_Arch_Inplane_Buckling` | gradients vs complex step (7.4e-16, 1.2e-15) | none | |
| `uq_benchmarks/Fire_Satellite_Coupled_System` | the 50-sample gradient step study; the OpenTURNS 1.27 comparison | OpenTURNS (throwaway environment) | |
| `uq_benchmarks/Golinski_Speed_Reducer` | g3 and g5 against psdr's printed expressions (4.4e-16); the "other readings tried" figures | psdr expressions can be transcribed; psdr itself is not needed | |
| `uq_benchmarks/River_Flood_Dyke_Overflow` | the 400 000-sample Monte Carlo check of `exact_moments` | none | |
| `combustion_chemical_kinetics/Annular_Combustor_Thermoacoustics` | the L_i = 0.04 m case-A roots (choice 1); the sample-by-sample comparison with the 2016 files ("The 2016 files") | none | the 2016 files are in git history before 9c1b8a7 |
| `energy/Lillgrund_FLORIS_Wake_Model` | the port against FLORIS 2.0.1 (29 flow cases, 6.5e-16); the gradient step study at 12 samples; the smoothness sweeps | FLORIS 2.0.1 (throwaway Python 3.11 venv, numpy 1.26.4, scipy 1.11.4) | the build folder keeps `floris_check.py`, the one original check script, as a starting point |
