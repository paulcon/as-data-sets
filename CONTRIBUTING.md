# Contributing a data set

A contribution is one directory: a case study that applies active subspaces to one model or one set of simulation runs. This guide covers what goes in it and how to submit it. To use the existing notebooks rather than add one, see [docs/using-the-notebooks.md](docs/using-the-notebooks.md).

## 1. Set up

```bash
git clone https://github.com/paulcon/as-data-sets.git
cd as-data-sets
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The notebooks use the v1 API of [active_subspaces](https://github.com/paulcon/active_subspaces), pinned in `requirements.txt`. Its main functions are `ac.active_subspace`, `ac.finite_difference_gradients`, `ac.local_linear_gradients`, `ac.fit_surrogate`, `ac.orient` and the plots in `ac.plotting`. The library's docstrings give the signatures.

## 2. Choose a category and a name

Put the data set in the category directory it belongs to (`aerospace_cfd/`, `energy/`, `uq_benchmarks/`, …; the list is in the [README](README.md#data-sets)). Name the directory in `Upper_Snake_Case` after the application and the quantity studied, e.g. `HyShotII_Scramjet` or `Offshore_Wind_Turbine_Fatigue`, and name the notebook after the directory.

```
energy/My_New_Data_Set/
    My_New_Data_Set.ipynb    the case study, executed, outputs committed
    DATA.md                  what the files are, where they came from, how to check them
    inputs.csv ...           the data
    my_model.py              (Python model) the model
    generate_data.py         (when the files were made by code) writes the data files
    check_accuracy.py        (when needed) reproduces the accuracy numbers in DATA.md
```

Notebooks and scripts load files by paths relative to their own directory, so they run from inside it.

## 3. Decide which kind of data set it is

Every data set is one of two kinds. The kind decides what you must ship and what "reproducible" means. Mark it at the end of the data set's README entry.

### Stored data

An outside code (a CFD solver, a hydrology code, MATLAB, a commercial package, …) produced the runs, and the files are their record.

- **Ship the values as they are.** Never regenerate, refit or "correct" them. A format conversion (whitespace text to CSV with a `#` header, or to `.npz`) must be lossless and done by a committed script (e.g. `convert_data.py`) that checks the values did not change.
- **Drop rows in the notebook, not in the file.** Rows that the analysis leaves out (duplicates, failed runs, validation runs) are removed in the notebook by a stated rule or a flag column.
- **A Python re-implementation may check the data**, but the files stay the record. Report any disagreement; do not fix the data to match.
- **Data you made yourself with an outside code** (e.g. Cantera, ngspice, SMRT) are stored data too. Add a `generate_data.py` that records how they were made and reruns the code where it is installed. When the code is missing, the script exits with an error that names what to install. Name a module that drives an outside code `*_driver.py`; `*_model.py` is reserved for numpy/scipy models.
- **Large originals** (over about 10 MB) belong in an archive such as Zenodo, with the DOI in `DATA.md`. Ship a reduced file here.

### Python model

The model is Python code in the directory, so the data can be regenerated and the notebook can call the model.

- **The model is a module named `*_model.py`** that imports only numpy and scipy and loads its own files relative to `__file__`. A closed-form model of a few lines may stay in a notebook cell, but a module is preferred so the model can be used outside the notebook.
- **Every shipped data file is written by `generate_data.py`**, with fixed seeds and all settings in the script. The script must reproduce the shipped files to a tolerance stated in `DATA.md`. Give it a `--check` option that regenerates into memory or another directory and compares, and never overwrites the shipped files.
- **Inputs the model needs but that cannot be regenerated** (a basis computed by a lost script, measured data the model is fitted to) are treated as stored data.
- **The notebook calls the model directly** when that is cheap, and loads the shipped files when regeneration is expensive.

A data set may have one part of each kind (see `plasma_fusion/MHD_Generator/`).

### Accuracy numbers

Every accuracy number in `DATA.md` (solver tolerances, gradient checks, comparisons with a reference code) must be printed by the notebook, by `generate_data.py`, or by a `check_accuracy.py` in the directory. Like `generate_data.py`, a `check_accuracy.py` that needs an outside code exits with an install message when the code is missing. Label a number that no script here can produce (a commercial code, a run that cannot be repeated) as "external run, not reproducible from the repository".

## 4. Write the notebook

The notebook is the case study: it explains the application, runs the analysis and says what the results mean. `uq_benchmarks/Hat_Stiffened_Panel_Shear_Buckling/` is a compact example to follow.

**Structure.** Most notebooks follow this order:

1. Title, the authors of the model or the runs, and a one-paragraph statement of the question.
2. An import cell that sets the seed (`SEED = ...`, the only seed in the notebook) and starts a timer.
3. **Summary**: the findings in a few sentences, so a reader can stop here.
4. **The model**: the physics, the inputs (names, units, ranges or distributions, sources), the quantity of interest.
5. **Load data**, with checks: shapes, ranges, the files against the model where there is one, gradients against finite differences.
6. **Estimate the subspace**, then the results: eigenvalues with bootstrap bands, subspace errors, eigenvectors, and sufficient summary plots of `f` against the active variables.
7. Optionally, **surrogates** on the active variables (`ac.fit_surrogate`) and other uses (integration, optimization, inversion).
8. **Comparison with the paper**: a table of published values against the notebook's values, with every difference explained or stated as unexplained.
9. **References**, then a final cell that prints the total run time.

**Conventions.**

- **Declare the inputs as a v1 input space in physical units.** Use `ac.Uniform(lower, upper, names=...)`, `ac.Gaussian(mean, cov, names=...)`, or `ac.Product` for mixed inputs. Use `ac.Uniform.reference(m)` only for data already on `[-1, 1]^m`. Pass `X` and gradients in physical units: the library normalizes them. Say in `DATA.md` whether stored gradients are with respect to physical or normalized inputs, and convert normalized ones before passing them.
- **Choose the estimator from what the data have.** Use `estimator='as'` with gradients (exact, adjoint, or finite-difference). Without gradients, use `'ols'` (a global linear model), `'qphd'` (a global quadratic model), or `'opg'` (local linear gradients). Say which one and why.
- **Fix the eigenvector signs** with `ac.orient` (e.g. `by='covariance'` or `by='mean_gradient'`), choosing the convention of the published figures when there is one. State the sign rule in the text.
- **Justify the subspace dimension** with the eigenvalue gaps, the bootstrap subspace distances and the share of the trace left out, not only with a surrogate's fit.
- **Make it deterministic.** All randomness comes from the one seed. Run the notebook twice and check that the printed numbers agree.
- **Write the numbers in the text from the outputs.** Quote values at rounding level (null eigenvalues, model-vs-file differences near 1e-16) as orders of magnitude, since they change between platforms.
- **Write links inline**: `[[1]](https://doi.org/...)`. Jupyter renders each markdown cell on its own, so a reference-style link `[1]: ...` works only in the cell that defines it.
- **Cite every source** (paper, code, data) with a DOI or a stable URL.

**Execute it** headless from its directory and commit the outputs:

```bash
cd energy/My_New_Data_Set
jupyter nbconvert --to notebook --execute --inplace My_New_Data_Set.ipynb
```

## 5. Write DATA.md

Start from [docs/templates/DATA.md](docs/templates/DATA.md). It records what someone needs to trust and reuse the files without the notebook: every file and its columns and units, the model or the generating code (with version), who ran it, the inputs and their distributions, the sampling design, the gradient coordinates, how to regenerate or check the files and how long that takes, the accuracy, how the files relate to the published runs, the license and credit, and **everything that is unknown**. State an unknown as unknown; do not guess.

## 6. License the files

Code (notebooks, `.py` files) is under the MIT License ([LICENSE](LICENSE)), and data are under CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)). Contribute only data you may redistribute. If a source states no license or a different one, add the files and their source to the exceptions list in `LICENSE-DATA`, and credit the source in `DATA.md`.

## 7. List it

- Add a bullet to the category's section of the [README](README.md#data-sets), in alphabetical order. The bullet gives the directory link, one sentence on the model and the quantity studied, a link to the paper, and the kind in italics, e.g. `_Python model (my_model.py)._` or `_Stored data (SU2)._`.
- Add a line to the category's `README.md`.
- Add the notebook's number of inputs and active subspace dimension to `SIZES` in `docs/make_notebook_index.py`, then run `python docs/make_notebook_index.py` to add the notebook to [docs/notebook-index.md](docs/notebook-index.md).

## 8. Open a pull request

`main` is protected, so every change lands through a pull request. Before you open one, check that:

- [ ] the notebook runs top to bottom from its directory in a fresh kernel, and twice gives the same numbers;
- [ ] `generate_data.py --check` (where there is one) reproduces the shipped files;
- [ ] `check_accuracy.py` (where there is one) reproduces the numbers in `DATA.md`;
- [ ] `DATA.md` has every section of the template, including the unknowns;
- [ ] the license is clear, and `LICENSE-DATA` lists any exception;
- [ ] the README, the category README and `docs/notebook-index.md` list the data set.

In the pull request, say which paper the analysis follows, which kind of data set it is, and how long the notebook and the scripts take to run.
