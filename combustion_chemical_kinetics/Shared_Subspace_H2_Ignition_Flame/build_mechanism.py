"""Build konnov2008_h2.yaml (Cantera) from the supplementary files of Konnov (2008).

    python build_mechanism.py DIR

DIR holds the three supplementary files of A. A. Konnov, "Remaining uncertainties in the
kinetic mechanism of hydrogen combustion", Combust. Flame 152 (2008) 507-528,
doi:10.1016/j.combustflame.2007.10.024 (publisher's "Supplementary data"; not redistributed
here):
    1-s2.0-S0010218007003318-mmc1.txt   mechanism (CHEMKIN)
    1-s2.0-S0010218007003318-mmc2.txt   thermodynamic data
    1-s2.0-S0010218007003318-mmc3.txt   transport data

The script checks their SHA-256, removes the non-standard token "BASE M=N2" from the
REACTIONS line (standard CHEMKIN gives unlisted third bodies an efficiency of 1, as Konnov's
Table 1 footnote a states), runs Cantera's ck2yaml, and writes konnov2008_h2.yaml beside this
script. Nothing else in the files is changed. Needs Cantera (pip install cantera==3.2.0).
"""
import hashlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "mech": ("1-s2.0-S0010218007003318-mmc1.txt",
             "3bcd9bf3d5c34a717c31e90bb779210088df37138fc351e23eb815bd860232dd"),
    "thermo": ("1-s2.0-S0010218007003318-mmc2.txt",
               "2e8f449b9e90067a94674ecec22a8bf6284a6405efbbdb3ed6ae6256b28b6920"),
    "tran": ("1-s2.0-S0010218007003318-mmc3.txt",
             "91e0b98c2129de80fd92cef17b2234d1a14d058d2c601423716d27efd7ff05cb"),
}


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)   # the downloaded files are needed
        sys.exit(3)
    src = sys.argv[1]
    try:
        import cantera  # noqa: F401
    except ImportError:
        print("Cantera is not installed: pip install cantera==3.2.0, then rerun build_mechanism.py", file=sys.stderr)
        sys.exit(3)
    paths = {}
    for key, (name, sha) in FILES.items():
        p = os.path.join(src, name)
        if not os.path.exists(p):
            print(f"missing {name} in {src}", file=sys.stderr)
            sys.exit(3)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != sha:
            sys.exit(f"{name}: sha256 {h} differs from the expected {sha}")
        paths[key] = p
    lines = open(paths["mech"], encoding="latin-1").read().splitlines(True)
    n = 0
    for i, line in enumerate(lines):
        if line.upper().startswith("REACTIONS") and "BASE" in line.upper():
            lines[i] = "REACTIONS\n"
            n += 1
    if n != 1:
        sys.exit("expected one 'REACTIONS BASE M=N2' line")
    out = os.path.join(HERE, "konnov2008_h2.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        mech = os.path.join(tmp, "konnov2008_mech.inp")
        with open(mech, "w", encoding="latin-1") as fh:
            fh.writelines(lines)
        cmd = [sys.executable, "-m", "cantera.ck2yaml", f"--input={mech}",
               f"--thermo={paths['thermo']}", f"--transport={paths['tran']}",
               f"--output={out}", "--quiet"]
        subprocess.run(cmd, check=True, cwd=tmp)
    import cantera as ct
    gas = ct.Solution(out)
    print(f"wrote konnov2008_h2.yaml: {gas.n_species} species, {gas.n_reactions} reactions")
    if (gas.n_species, gas.n_reactions) != (10, 33):
        sys.exit("unexpected mechanism size (expected 10 species, 33 reactions)")


if __name__ == "__main__":
    main()
