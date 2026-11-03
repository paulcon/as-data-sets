"""Build the paper-matched PROCESS input-deck template from PROCESS v3.4.2
examples/data/large_tokamak_IN.DAT (J. Morris, UKAEA, 28/06/23; MIT licence).

Every edit is listed in EDITS and applied by exact line match (fails loudly if a line is
missing). Placeholders {hfact} ... are filled per sample by fill_deck().
"""
import sys

# (old line, new line or None to delete). Order matters only for readability.
EDITS = [
    ("runtitle = Generic large tokamak",
     "runtitle = EU H-mode DEMO-like baseline for the Miralles-Dolz et al. 2022 interval study"),
    # 1) H-factor: input, not iteration variable (stock deck iterates it, ixc=10, ub 1.2)
    ("ixc = 10", None),
    ("boundu(10) = 1.2", None),
    ("hfact = 1.1", "hfact = {hfact}"),
    # 2) divertor limit PsepBt/qAR (constraint 68): stock 10.0, paper baseline 9.2
    ("p_div_bt_q_aspect_rmajor_max_mw = 10.0", "p_div_bt_q_aspect_rmajor_max_mw = {psepbqar_max}"),
    # 3) core radius for radiation correction of confinement scaling (paper 'core radius')
    ("radius_plasma_core_norm = 0.75", "radius_plasma_core_norm = {core_radius}"),
    # 4) W concentration: stock 5e-6, paper baseline printed '50^-5' (read 5e-5)
    ("f_nd_impurity_electrons(14) = 0.000005", "f_nd_impurity_electrons(14) = {f_w}"),
    # 5) elongation at separatrix (i_plasma_geometry = 0: input kappa is the X-point value)
    ("kappa = 1.85", "kappa = {kappa}"),
    # 6) thermal He-4 fraction: input, not iteration variable (stock: ixc=109, ub 0.1)
    ("ixc = 109", None),
    ("boundu(109) = 0.1", "f_nd_alpha_thermal_electron = {f_he}"),
    # major-radius bounds: stock [8, 9] m would clip the paper's [7.93, 10.32] m interval
    ("boundl(3) = 8.0", "boundl(3) = 6.0"),
    ("boundu(3) = 9.0", "boundu(3) = 13.0"),
    ("rmajor = 8.0", "rmajor = {rmajor0}"),
    # cap optimiser iterations (stock 200; PROCESS retries a failed solve up to 3 times)
    ("epsvmc = 1e-7", "epsvmc = 1e-7\n\n* Max VMCON iterations per attempt (added)\nmaxcal = {maxcal}"),
]


def build(src, dst):
    lines = open(src, encoding="utf-8").read().split("\n")
    stripped = [l.strip() for l in lines]
    for old, new in EDITS:
        hits = [i for i, l in enumerate(stripped) if l == old]
        if len(hits) != 1:
            raise SystemExit(f"edit target {old!r} found {len(hits)} times")
        i = hits[0]
        lines[i] = "" if new is None else new
        stripped[i] = "<edited>"
    hdr = ("* TEMPLATE derived from PROCESS v3.4.2 examples/data/large_tokamak_IN.DAT\n"
           "* (Generic large tokamak, James Morris, UKAEA, 28/06/23; PROCESS is MIT-licensed).\n"
           "* Edits for the Miralles-Dolz et al. (2022) interval study are listed in make_template.py.\n")
    open(dst, "w", encoding="utf-8").write(hdr + "\n".join(lines))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
