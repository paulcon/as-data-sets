# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of data sets plus Jupyter notebooks that apply **active subspaces** (dimension reduction for simulation models) to science/engineering models. There is no package, build system, test suite, or linter. Each top-level directory (`Ebola/`, `HIV/`, `MCMC/`, `Elliptic_PDE/`, …) is a self-contained case study: one notebook, its data files, and sometimes helper `.py` modules. The README has the per-dataset descriptions and publication links. Add new data sets the same way, and add a bullet for each new one to the README's "Data sets" list.

## Two incompatible environments

This is the most important thing to know before running or editing anything:

- **The notebooks and scripts in the repo are legacy code**: their kernelspec is `python2` (2.7.11), several use Python 2 `print x` statements (`HyShotII`, `MCMC`, `NREL_Wind`, `SingleDiodePV`, `Subsurface_Permeability` notebooks, plus `Elliptic_PDE/pde_study.py` and `HIV/generate_data.py`), and they call the **old** `active_subspaces` API: `ac.subspaces.Subspaces()` with `.compute(df=...)`/`.partition()`/`.W1`/`.eigenvals`, `ac.utils.plotters.*`, `ac.utils.misc.BoundedNormalizer`, `ac.utils.response_surfaces.*`, `ac.utils.simrunners.*`, `ac.domains.*`.
- **The local `.venv` (the PyCharm interpreter) is Python 3.11** with `active_subspaces` 1.0.0.dev0, installed editable from `../ActiveSubspacesRevision/upstream-v1/src`. That is a rewritten API: top-level functions such as `ac.active_subspace`, `ac.finite_difference_gradients`, `ac.local_linear_gradients`, `ac.fit_surrogate`, `ac.PolynomialSurrogate`/`ac.RBFSurrogate`, `ac.active_domain`, and `ac.plotting`. **`ac.utils` and `ac.subspaces.Subspaces` do not exist in it.** Check that source tree for signatures rather than guessing.
- `.venv` also lacks `pandas` (used by most notebooks, imported as `pn`) and `numpad` (the autodiff library used by `Elliptic_PDE/pde_model.py` and `MCMC/pde_model.py`).

So no notebook runs unmodified in `.venv`. If you port a notebook, the job is three things: convert it to Python 3, translate the `ac.*` calls to the v1 API, and install any missing dependencies.

## Commands

```bash
source .venv/bin/activate
jupyter lab                                   # interactive
# Execute a notebook headless. Run it from inside its directory, because notebooks and helper modules load data by relative path:
cd HIV && jupyter nbconvert --to notebook --execute --inplace HIV.ipynb
```

## Update specs

`specs/` holds the October 2026 review: one spec per notebook (known errors, the v1 call map, a pandas-free loader, acceptance criteria), `TODO.md`, `BRANCH_MERGE.md` (master + dev → main), and `numpad_replacement.md`. Read a directory's spec before you change its notebook.

## Conventions in the notebooks

- Inputs are normalized to `[-1, 1]^m` before computing subspaces. The usual pattern is `ac.utils.misc.BoundedNormalizer(xl, xu)`, or the hand-written affine map `p = .5*(diag(xu-xl) @ XX.T + xu + xl).T`. Gradients taken with respect to physical inputs are then rescaled by `.5*(xu - xl)`.
- The typical flow is: load samples (inputs X, outputs f, and gradients df when available) → build the subspace from gradients, or from local-linear/global-linear approximations when gradients are missing → plot eigenvalues and eigenvectors → make a sufficient-summary plot of `f` against `X @ W1` → optionally fit a response surface on the active variables.
- Helper modules are imported from the notebook's own directory. Examples: `HIV_model.Tcells`, `pde_model.PDESolver` (KL-expansion coefficients loaded from `kl-short.npz`/`kl-long.npz`), and `battery_figs`. The `generate_data.py` / `*_study.py` scripts regenerate the shipped data files and can be expensive to run.
- `Elliptic_PDE/pde_model.py` and `MCMC/pde_model.py` share the same `PDESolver` discretization (`set_coeff`/`residual`) but expose different quantities of interest. The Elliptic_PDE version has `qoi`/`grad_qoi`, a boundary average. The MCMC version has `observations`/`misfit`/`misfit_grad`, a data misfit at sensor indices. A fix to the shared solver code has to be made in both files.
