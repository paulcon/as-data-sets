# Guides

- **[Using the notebooks](using-the-notebooks.md)**: how to read and run the notebooks, what is in a data set's directory, and how to read the results (eigenvalues, eigenvectors, sufficient summary plots, surrogates).
- **[Notebooks by method](notebook-index.md)**: every notebook, with its number of inputs, the dimension of the active subspace it finds, how it estimates the subspace, how it models the inputs, and what else it does with the subspace. Use it to find an example of a method.
- **[Contributing a data set](../CONTRIBUTING.md)**: what a data set directory contains, the rules for stored data and Python models, the notebook conventions, and how to submit it.
- **[Suggested articles](candidates.md)**: published articles that might support a new data set, from an AI-assisted literature search, with where the list came from and how to use it.
- **[DATA.md template](templates/DATA.md)**: the sections every data set's `DATA.md` has.

For the applications themselves, start from the [list of data sets](../README.md#data-sets) or a category's `README.md` (e.g. [energy/](../energy/README.md)).

`make_notebook_index.py` writes `notebook-index.md`; rerun it after adding or changing a notebook. `make_candidates_md.py` writes `candidates.md` from `candidates.csv`.
