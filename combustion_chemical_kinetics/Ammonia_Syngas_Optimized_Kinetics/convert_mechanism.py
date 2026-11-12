"""Make the mechanism file ammonia_syngas_optimized.yaml from the article's supplementary zip (needs Cantera 3.2.0).

    python convert_mechanism.py --zip d3re00160a1_suppl.zip --out PATH/ammonia_syngas_optimized.yaml

The mechanism is not part of this data set. It is the optimized model of W. Li, C. Zou and H. Yao,
"An optimized model for ammonia/syngas combustion", React. Chem. Eng. 8 (2023) 2071-2085,
doi:10.1039/D3RE00160A, distributed by the publisher (RSC) as electronic supplementary information
(ESI) with no stated licence. Download it from the article page, https://doi.org/10.1039/D3RE00160A,
link "Electronic supplementary information": the file d3re00160a1_suppl.zip (65 231 bytes, sha256
below), which holds optimized.inp, thermo.dat and tran.dat in CHEMKIN format.

This script checks the sha256 of the zip, reads optimized.inp and thermo.dat from it (nothing is
extracted elsewhere), converts them with Cantera's ck2yaml (permissive mode, which skips a
duplicate HOCO entry in thermo.dat; tran.dat is not needed), and checks that the result loads
with 37 species and 246 reaction entries (237 reactions, 9 of them written as DUPLICATE pairs).
It is the same as running

    ck2yaml --permissive --input optimized.inp --thermo thermo.dat --output ammonia_syngas_optimized.yaml

on the unzipped files.

--out is required: a path of your choice outside this data set's directory (the script refuses the
directory and any folder in it; the file is not part of the repository). Then set the environment
variable AMMONIA_SYNGAS_MECH to that path: ammonia_idt_driver.py reads the file from there.
"""
import argparse
import hashlib
import os
import sys
import tempfile
import zipfile

ZIP_NAME = "d3re00160a1_suppl.zip"
ZIP_SHA256 = "ee493371ad68aef892269bc3d19951609caa364d4ccd7d83112c3d0414a86a6b"
ZIP_BYTES = 65231
N_SPECIES, N_REACTIONS = 37, 246


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True, help=f"path of {ZIP_NAME}")
    p.add_argument("--out", required=True, help="output file (.yaml)")
    p.add_argument("--any-zip", action="store_true",
                   help="convert even if the sha256 of the zip differs from the one above")
    a = p.parse_args()
    try:
        import cantera as ct
        from cantera import ck2yaml
    except ImportError:
        print("Cantera is not installed. Install it (pip install cantera==3.2.0, the version used for "
              "the shipped data) and rerun convert_mechanism.py.", file=sys.stderr)
        sys.exit(3)
    if not os.path.isfile(a.zip):
        print(f"{a.zip} not found. Download {ZIP_NAME} from https://doi.org/10.1039/D3RE00160A "
              "(link 'Electronic supplementary information') and rerun.", file=sys.stderr)
        sys.exit(3)
    digest = sha256(a.zip)
    if digest != ZIP_SHA256 and not a.any_zip:
        sys.exit(f"sha256 of {a.zip} is {digest}, not {ZIP_SHA256} ({ZIP_BYTES} bytes), the file the shipped "
                 "data were made with. Rerun with --any-zip to convert it anyway.")
    out = os.path.abspath(a.out)
    if not out.endswith((".yaml", ".yml")):
        sys.exit("--out must end in .yaml")
    here = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    if os.path.commonpath([os.path.realpath(out), here]) == here:
        sys.exit("--out must be outside this data set's directory (not the directory or a folder in it)")
    with zipfile.ZipFile(a.zip) as z, tempfile.TemporaryDirectory() as tmp:
        members = {os.path.basename(n): n for n in z.namelist()}
        paths = {}
        for name in ("optimized.inp", "thermo.dat"):
            if name not in members:
                sys.exit(f"{name} is not in {a.zip}")
            paths[name] = os.path.join(tmp, name)
            with open(paths[name], "wb") as fh:
                fh.write(z.read(members[name]))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ck2yaml.convert(paths["optimized.inp"], thermo_file=paths["thermo.dat"], out_name=out,
                        permissive=True, quiet=True)
    import warnings
    with warnings.catch_warnings():           # NASA-polynomial discontinuity notice for OH*
        warnings.simplefilter("ignore")
        gas = ct.Solution(out)
    if (gas.n_species, gas.n_reactions) != (N_SPECIES, N_REACTIONS):
        sys.exit(f"{out}: {gas.n_species} species and {gas.n_reactions} reaction entries, expected "
                 f"{N_SPECIES} and {N_REACTIONS}")
    print(f"wrote {out}: {gas.n_species} species, {gas.n_reactions} reaction entries (Cantera {ct.__version__})")
    print(f"set AMMONIA_SYNGAS_MECH={out} before running generate_data.py or check_accuracy.py")


if __name__ == "__main__":
    main()
