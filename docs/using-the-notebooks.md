# Using the notebooks

Each notebook is a worked example of an active subspace analysis of one science or engineering model. This guide explains how to run the notebooks, how they are laid out, and how to read their results. To find a notebook that uses a particular method, see [notebook-index.md](notebook-index.md). To add a data set of your own, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Reading without running

GitHub renders a notebook, with its committed outputs, when you click the `.ipynb` file. Its rendering of mathematics is not always reliable; [nbviewer](https://nbviewer.org/github/paulcon/as-data-sets/tree/main/) renders the same notebooks more faithfully. Every notebook is committed with its outputs, so you can read the full analysis without installing anything.

## Running

```bash
git clone https://github.com/paulcon/as-data-sets.git
cd as-data-sets
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd energy/Single_Diode_Solar_Cell        # any data set's directory
jupyter lab
```

Start Jupyter from the data set's directory: the notebooks and their helper modules load files by relative path. The notebooks need only numpy, scipy, matplotlib and [active_subspaces](https://github.com/paulcon/active_subspaces) (version 1). Each one sets a single seed at the top, so a rerun prints the same numbers, and its last cell prints the run time. Numbers at the level of rounding error (eigenvalues near 1e-30, differences near 1e-16) can differ on another machine.

## What is in a data set's directory

- **The notebook**, named after the directory.
- **`DATA.md`**: what every file contains, where the data came from, how they were checked, how to regenerate them, the license, and what is unknown about them. Read it before reusing the data elsewhere.
- **The data**: mostly CSV files with `#` header lines, read by `np.loadtxt(path, delimiter=',')`, or NumPy `.npz` files.
- Sometimes **Python modules**: a model (`*_model.py`), a script that writes the data files (`generate_data.py`), a check of the accuracy numbers quoted in `DATA.md` (`check_accuracy.py`), or a module that drives an outside simulation code (`*_driver.py`).

Each data set is one of two kinds:

- **Stored data**: an outside simulation code (a CFD solver, a hydrology code, a battery model, …) produced the runs, and the files are their record. You can rerun the analysis but not the simulations, unless the outside code is installed and the directory has a `generate_data.py` for it.
- **Python model**: the model is Python code in the directory. The notebook can call it, and `generate_data.py` regenerates the data files. These are the data sets to use when you want to try a method that needs new model evaluations.

## How a notebook is laid out

Most notebooks follow the same order, so you can skip to the part you need:

1. **Summary**: the findings in a few sentences.
2. **The model**: the application, the inputs with their ranges or distributions, and the quantity of interest.
3. **Load data**, with checks of the files.
4. **Estimate the subspace**, then the results: eigenvalues, subspace errors, eigenvectors, and sufficient summary plots.
5. Often, **surrogates** on the active variables, or another use of the subspace.
6. **Comparison with the paper**: published values beside the notebook's values.
7. **References**.

## Reading the results

**Inputs and normalization.** The inputs are declared in physical units as an input space: `ac.Uniform` for independent inputs on a box, `ac.Gaussian` for (possibly correlated) normal inputs, `ac.Product` for a mix. The library maps them to normalized coordinates, `[-1, 1]` for uniform inputs and a standard normal for Gaussian ones, and every eigenvector is in those coordinates. That makes the weights of inputs with different units comparable.

**How the subspace is estimated.** The active subspace is spanned by the leading eigenvectors of C, the average of ∇f ∇fᵀ over the input distribution. What the notebook estimates C from depends on what the data provide:

- *gradients* (`estimator='as'`): exact, adjoint or finite-difference gradients of the model. This is the active subspace itself.
- *local linear gradients* (`ac.local_linear_gradients`, or `estimator='opg'`): gradients estimated from values by fitting a linear model near each point. These need many samples per input.
- *linear fit* (`estimator='ols'`): the gradient of one global linear model. It finds at most one direction, the one along which f changes most on average.
- *quadratic fit* (`estimator='qphd'`): the matrix of a global quadratic model, which can find several directions.
- *SIR*, and a matrix or basis computed in the notebook (e.g. by quadrature), appear in a few notebooks.

Estimates from values are approximations to the active subspace. Each notebook says which estimator it uses and why.

**Eigenvalues** (`ac.plotting.plot_eigenvalues`). These are on a log scale, with a band from bootstrap resampling. A large gap after the n-th eigenvalue means that f varies mostly along the first n directions. The eigenvalues have the units of f squared, so read them as ratios, not as absolute sizes.

**Subspace errors** (`ac.plotting.plot_subspace_errors`). These show the bootstrap estimate of the distance between the estimated and the true n-dimensional subspace, for each n. A small distance means the directions are well determined by the samples. The distance is usually smallest where the eigenvalue gap is largest.

**Eigenvectors** (`ac.plotting.plot_eigenvectors`). The weight of each input in a direction. Inputs with large weights in the first eigenvector w₁ are the ones that matter most, and the signs say which way each input pushes f. An eigenvector's overall sign is arbitrary. The notebooks fix it with `ac.orient`, usually so that f increases along w₁, and state the rule. Some notebooks also compute *activity scores*, α_j = Σᵢ λᵢ w_ij² over the first n directions: one importance number per input.

**Sufficient summary plots** (`ac.plotting.plot_sufficient_summary`). f against the active variable y₁ = w₁ᵀx (one dimension), or colored over (y₁, y₂) (two dimensions). If the points fall on a thin curve, f is close to a function of the active variables alone. The spread around the curve shows what the remaining directions contribute.

**Choosing the dimension.** The notebooks justify the dimension n they use from the eigenvalue gaps, the subspace errors and the share of the eigenvalue sum left out. A surrogate's fit is reported as a check, not as the reason.

**Surrogates** (`ac.fit_surrogate`). A polynomial or radial basis function in the active variables, with its leave-one-out R². A high R² with few active variables means that a cheap low-dimensional model can stand in for the simulation. Some notebooks go further: prediction intervals, an *active domain* (the set of values the active variables take over the input box: an interval or a zonotope), an *inverse map* (drawing inputs that map to given values of the active variables), integration, optimization, or MCMC.

## Reusing the data

The data are under CC BY 4.0 (exceptions are listed in `LICENSE-DATA`) and the code under the MIT License. Each `DATA.md` says whom to credit and which papers to cite. Read its sections on units, sampling and unknowns before using the files in another study: they record what is known about the data and what is not.
