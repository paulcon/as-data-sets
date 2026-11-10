# My_New_Data_Set: data

<!-- Copy this file into the data set's directory as DATA.md and replace every placeholder.
     Keep every section. Where something is not known, say so in Unknowns rather than leaving a section out.
     Examples: uq_benchmarks/Hat_Stiffened_Panel_Shear_Buckling/DATA.md (Python model),
     aerospace_cfd/ONERA_M6_Wing/DATA.md (stored data). -->

Kind: **Python model** | **stored data**. One or two sentences: what produced the files, and whether they can be regenerated here.

## Files

| file | contents | role |
|---|---|---|
| `My_New_Data_Set.ipynb` | the case study | executed top to bottom with seed ...; outputs committed |
| `my_model.py` | the functions it provides | the model |
| `generate_data.py` | writes the data files; `--check` compares a rerun with them | regenerates the data |
| `inputs.csv` | M × m inputs, physical units (column names and units in the header) | model input |
| `outputs.csv` | M × k: the quantities of interest, with units | model output |
| `gradients.csv` | M × m: ∂f/∂x_j with respect to the **physical** or **normalized** inputs | model output |

How the files are read (e.g. `np.loadtxt(path, delimiter=',')`), the number of header lines, and the precision of the values.

## Model

**Source.** The paper, report or code the model or the runs come from, with a DOI or a stable URL, and the version or commit of any code.

**Equations or solver.** The model in enough detail to check it: the equations, or for an outside code its name, version, mesh and solver settings.

**Inputs** (m = ...):

| name | symbol | units | description | range or distribution | source |
|---|---|---|---|---|---|
| `x1` | x₁ | ... | ... | U[a, b] | paper, Table ... |

**Quantities of interest.** What each output is, its units, its range over the samples.

**Gradients.** How they were computed (exact, adjoint, finite differences with step ...), and with respect to which coordinates.

**Sampling design.** Monte Carlo, Latin hypercube, quadrature, …; the number of samples, the seed or the generator.

## Regenerating

```bash
python generate_data.py           # writes the data files here
python generate_data.py --check   # reruns and compares with the shipped files; writes nothing here
```

The run time, the platform, and the largest differences that `--check` reports. For stored data: what cannot be regenerated, or the outside code (and version) that `generate_data.py` needs, and the message it gives when that code is missing.

## Accuracy

Each accuracy number, and the script that prints it: the notebook, `generate_data.py`, or `check_accuracy.py` (check number). Label a number that no script here can produce "external run, not reproducible from the repository".

## How these files relate to the paper

Whether these are the paper's runs or new ones, which of the paper's results the notebook reproduces, and every difference from the paper, explained or stated as unexplained.

## Unknowns

Everything a user might need that is not known: units, versions, who ran it, which rows are which, how the samples were drawn. One bullet each.

## License and credit

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`), or the exception listed there. Who to credit, and what to cite:

- Authors, "Title", *Journal* volume(issue):pages, year, doi:...
