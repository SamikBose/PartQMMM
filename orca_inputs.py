#!/usr/bin/env python3
"""
orca_inputs.py
==============
Generate ORCA electrostatic-embedding QM/MM single-point inputs from an MD
trajectory, using qmmm_partition as the engine-agnostic backend.

For each selected frame it writes:
    frame_{idx}.inp     - ORCA input with QM coords inline and a %pointcharges
                          directive pointing at the .pc file (electrostatic
                          embedding)
    frame_{idx}.pc      - ORCA point-charge file: first line = count, then
                          'q  x  y  z' per MM atom (charges in e, coords in Å)

ORCA reads external point charges via:
    % pointcharges "frame_{idx}.pc"
and includes them in the SCF as fixed charges -> electrostatic embedding, so
the QM density polarizes in the field of the MM environment (NOT mechanical
embedding).

Later engines (Psi4, Q-Chem) get their own driver modules that import the same
qmmm_partition backend; only the writer differs.

Usage:
    python orca_inputs.py --top system_1264.parm7 --traj prod.dcd \
        --region minimal --output-dir orca_min --frames 0:None:10 \
        --method "wB97M-V" --basis "def2-TZVP" --charge -1 --mult 1

Dependencies: qmmm_partition (+ its deps: mdtraj, parmed, numpy)
"""

import argparse
import sys
from pathlib import Path

import qmmm_partition as qp


# ---------------------------------------------------------------------------
# ORCA writers
# ---------------------------------------------------------------------------
def write_pc_file(path, charges, positions_A):
    """ORCA external point-charge file. First line: count. Then: q x y z."""
    with open(path, "w") as f:
        f.write(f"{len(charges)}\n")
        for q, p in zip(charges, positions_A):
            f.write(f"{q:>12.8f} {p[0]:>14.6f} {p[1]:>14.6f} {p[2]:>14.6f}\n")


def write_orca_input(path, elements, positions_A, pc_filename, *,
                     method, basis, charge, mult, nprocs, maxcore_mb,
                     extra_keywords="", scf_block=""):
    """
    Write an ORCA input doing a QM/MM electrostatic-embedding single point.

    The QM atoms are inline; the MM charges are referenced via %pointcharges.
    `charge`/`mult` are for the QM region only.
    """
    keyword_line = f"! {method} {basis} {extra_keywords}".rstrip()
    lines = []
    lines.append(keyword_line)
    lines.append("")
    if nprocs and nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    if maxcore_mb:
        lines.append(f"%maxcore {maxcore_mb}")
    # Electrostatic embedding: external point charges
    lines.append(f'%pointcharges "{pc_filename}"')
    if scf_block:
        lines.append(scf_block)
    lines.append("")
    lines.append(f"* xyz {charge} {mult}")
    for el, p in zip(elements, positions_A):
        lines.append(f"{el:<3s} {p[0]:>14.6f} {p[1]:>14.6f} {p[2]:>14.6f}")
    lines.append("*")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # partition backend args
    ap.add_argument("--top", required=True, help="AMBER parm7/prmtop")
    ap.add_argument("--traj", required=True,
                    help="trajectory (.dcd/.nc/.xtc/.h5) or restart (.rst7)")
    ap.add_argument("--region", choices=list(qp.QM_REGIONS), default="minimal")
    ap.add_argument("--resnum-offset", type=int, default=qp.DEFAULT_RESNUM_OFFSET,
                    help="PDB->topology residue-number offset (default -4)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames", default="0:None:1",
                    help="start:stop:stride (default all frames)")
    ap.add_argument("--mm-cutoff", type=float, default=0.0,
                    help="keep whole MM residues/molecules with at least one MM "
                         "atom within this Å of any QM atom (0=all)")
    ap.add_argument("--allow-missing-qm", action="store_true",
                    help="warn and continue if requested QM residues/ligand/metal "
                         "are missing; default is fail-fast")
    ap.add_argument("--dry-run", action="store_true",
                    help="process only the first frame and print diagnostics")

    # ORCA QM args
    ap.add_argument("--method", default="wB97M-V",
                    help="DFT functional / method (default wB97M-V)")
    ap.add_argument("--basis", default="def2-TZVP")
    ap.add_argument("--charge", type=int, default=1,
                    help="QM-region NET charge. For the minimal region with Zn "
                         "in QM: Zn(+2) + AZM(-1) + neutral His/Thr/water = +1. "
                         "Recompute if you change the region contents!")
    ap.add_argument("--mult", type=int, default=1, help="QM-region multiplicity")
    ap.add_argument("--extra-keywords", default="TightSCF DEFGRID3",
                    help="extra ! keywords appended to the ORCA keyword line")
    ap.add_argument("--nprocs", type=int, default=1)
    ap.add_argument("--maxcore", type=int, default=3000,
                    help="ORCA %%maxcore in MB per core")
    ap.add_argument("--engrad", action="store_true",
                    help="request gradients (adds EnGrad) — needed for ML force training")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    extra = args.extra_keywords
    if args.engrad and "engrad" not in extra.lower():
        extra = (extra + " EnGrad").strip()

    print(f"[orca_inputs] region={args.region} offset={args.resnum_offset}")
    print(f"[orca_inputs] method={args.method} basis={args.basis} "
          f"charge={args.charge} mult={args.mult}")
    print(f"[orca_inputs] writing to {outdir}/")
    if args.mm_cutoff and args.mm_cutoff > 0:
        print(f"[orca_inputs] MM cutoff mode: whole residues/molecules within "
              f"{args.mm_cutoff:g} Å; QM atoms remain excluded")

    n = 0
    charge_warned = False
    for fp in qp.iter_frame_partitions(
            args.top, args.traj, region_name=args.region,
            resnum_offset=args.resnum_offset, frames=args.frames,
            mm_cutoff_A=args.mm_cutoff, strict=(not args.allow_missing_qm)):

        # Cross-check the detected integer QM charge against the user's --charge.
        # The partition rounds the QM fragment's real partial-charge sum to the
        # nearest integer and makes the MM field consistent with it. If that
        # detected integer disagrees with --charge, the user likely mis-set it.
        if fp.qm_total_charge is not None and not charge_warned:
            if int(fp.qm_total_charge) != args.charge:
                print(f"  WARNING: --charge {args.charge} but the partition "
                      f"detected QM formal charge {int(fp.qm_total_charge)} "
                      f"(from rounding the QM fragment's partial-charge sum). "
                      f"The MM field was made consistent with "
                      f"{int(fp.qm_total_charge)}. If {args.charge} is what you "
                      f"intend, re-check the region contents; otherwise pass "
                      f"--charge {int(fp.qm_total_charge)}.")
                charge_warned = True

        if fp.mm_cutoff_A > 0 and n == 0:
            print(f"  cutoff diagnostics: retained MM charge "
                  f"{sum(fp.mm_charges):+.6f} e from {len(fp.mm_charges)} "
                  f"point charges; full-field MM charge would be "
                  f"{fp.mm_full_charge_sum:+.6f} e")

        tag = f"{fp.frame_index:06d}"
        pc_name = f"frame_{tag}.pc"
        inp_name = f"frame_{tag}.inp"
        write_pc_file(outdir / pc_name, fp.mm_charges, fp.mm_positions_A)
        write_orca_input(
            outdir / inp_name, fp.qm_elements, fp.qm_positions_A, pc_name,
            method=args.method, basis=args.basis,
            charge=args.charge, mult=args.mult,
            nprocs=args.nprocs, maxcore_mb=args.maxcore,
            extra_keywords=extra)

        if args.dry_run:
            s = qp.summarize_partition(fp)
            print("\n--- frame diagnostics ---")
            for k, v in s.items():
                print(f"  {k}: {v}")
            print(f"\nWrote {outdir/inp_name}")
            print(f"Wrote {outdir/pc_name}")
            # show the input head
            print("\n--- ORCA input (head) ---")
            print((outdir / inp_name).read_text().split("\n*\n")[0][:600])
            return

        n += 1
        if n % 200 == 0:
            print(f"  wrote frame {fp.frame_index} (total {n})")

    print(f"\nDone. {n} ORCA input/pc pairs in {outdir}/")
    if n:
        print("\nReminder: --charge is the QM-region NET charge. For the minimal")
        print("region (Zn in QM) it is +2 (Zn) + -1 (deprotonated AZM) = +1.")
        print("If you exclude Zn or change ligand/region contents, recompute it.")


if __name__ == "__main__":
    main()
