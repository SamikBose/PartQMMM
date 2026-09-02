#!/usr/bin/env python3
"""
Generate exactly two files per selected MD snapshot:
  1) frame_XXXXXX_qm.xyz : QM atoms + link H atoms in standard XYZ format
  2) frame_XXXXXX_mm.pc  : MM point charges as: q  x  y  z

No ORCA input is generated in PartQMMM V1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import qmmm_partition as qp


def write_qm_xyz(path: Path, fp: qp.FramePartition):
    comment = (
        f"frame={fp.frame_index} qm_formal_charge={fp.qm_formal_charge:+d} "
        f"n_hbonded_waters={len(fp.selected_water_resids)} "
        f"n_link_atoms={fp.n_link_atoms} "
        f"boundary_charge_method={fp.boundary_charge_method}"
    )
    with path.open("w") as fh:
        fh.write(f"{len(fp.qm_elements)}\n")
        fh.write(comment + "\n")
        for element, pos in zip(fp.qm_elements, fp.qm_positions_A):
            fh.write(
                f"{element:<3s} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f}\n"
            )


def write_mm_pc(path: Path, fp: qp.FramePartition):
    """Simple point-charge file: first line count, then q x y z."""
    with path.open("w") as fh:
        fh.write(f"{len(fp.mm_charges)}\n")
        for charge, pos in zip(fp.mm_charges, fp.mm_positions_A):
            fh.write(
                f"{charge:>14.9f} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f}\n"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", required=True, help="AMBER parm7/prmtop")
    ap.add_argument("--traj", required=True, help="MD trajectory/restart/PDB")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--region", choices=list(qp.QM_REGIONS), default="minimal")
    ap.add_argument("--resnum-offset", type=int, default=qp.DEFAULT_RESNUM_OFFSET)
    ap.add_argument("--frames", default="0:None:1", help="start:stop:stride")
    ap.add_argument(
        "--hbond-distance", type=float, default=3.5,
        help="donor-acceptor H-bond cutoff in Angstrom (default 3.5)",
    )
    ap.add_argument(
        "--hbond-angle", type=float, default=140.0,
        help="minimum D-H...A angle in degrees (default 140)",
    )
    ap.add_argument(
        "--zn-water-cutoff", type=float, default=qp.DEFAULT_ZN_WATER_CUTOFF_A,
        help=(
            "promote a whole water to QM when Zn-O is within this Angstrom "
            "distance (default 2.6; use 0 to disable Zn-coordination selection)"
        ),
    )
    ap.add_argument(
        "--boundary-charge-method",
        choices=("shift", "rcd"),
        default="shift",
        help=(
            "covalent QM/MM boundary electrostatics: 'shift' keeps the original "
            "CA->M2 charge shift; 'rcd' uses PBC-aware Redistributed Charge and "
            "Dipole virtual midpoint sites (default shift)"
        ),
    )
    ap.add_argument(
        "--ion-guard", type=float, default=4.0,
        help="fail if a non-QM free ion is within this distance of any QM real atom; 0 disables",
    )
    ap.add_argument(
        "--allow-close-ions", action="store_true",
        help="report close ions but do not fail (not recommended for production labels)",
    )
    ap.add_argument(
        "--allow-missing-qm", action="store_true",
        help="warn instead of failing if requested fixed-QM residues are missing",
    )
    args = ap.parse_args()
    if args.zn_water_cutoff < 0:
        ap.error("--zn-water-cutoff must be >= 0 (0 disables Zn-water selection)")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    expected_qm_charge = None
    n = 0
    for fp in qp.iter_frame_partitions(
        args.top,
        args.traj,
        region_name=args.region,
        resnum_offset=args.resnum_offset,
        frames=args.frames,
        hbond_distance_A=args.hbond_distance,
        hbond_angle_deg=args.hbond_angle,
        zn_water_cutoff_A=args.zn_water_cutoff,
        boundary_charge_method=args.boundary_charge_method,
        ion_guard_A=args.ion_guard,
        fail_on_close_ion=(not args.allow_close_ions),
        strict=(not args.allow_missing_qm),
    ):
        if expected_qm_charge is None:
            expected_qm_charge = fp.qm_formal_charge
        elif fp.qm_formal_charge != expected_qm_charge:
            raise RuntimeError(
                f"QM formal charge changed from {expected_qm_charge:+d} to "
                f"{fp.qm_formal_charge:+d} at frame {fp.frame_index}. "
                "V1 requires frame-invariant QM charge."
            )

        tag = f"{fp.frame_index:06d}"
        qm_path = outdir / f"frame_{tag}_qm.xyz"
        mm_path = outdir / f"frame_{tag}_mm.pc"
        write_qm_xyz(qm_path, fp)
        write_mm_pc(mm_path, fp)
        n += 1
        
        if n == 1 or n % 100 == 0:
        summary = qp.summarize_partition(fp)
            print(
                f"Processed {n} frames; latest frame={fp.frame_index}; "
                f"waters={len(fp.selected_water_resids)}; "
                f"QM atoms={len(fp.qm_elements)}; "
                f"MM charges={len(fp.mm_charges)}",
                flush=True,
                )
        
        #summary = qp.summarize_partition(fp)
        #print(json.dumps(summary, sort_keys=True))
        #for resid, reasons in fp.selected_water_reasons.items():
        #    for reason in reasons:
        #        print(f"  water_resid={resid}: {reason}")
        #n += 1

    print(f"Done: wrote {n} snapshot pair(s) to {outdir}")
    if expected_qm_charge is not None:
        print(f"Verified invariant QM formal charge: {expected_qm_charge:+d}")


if __name__ == "__main__":
    main()
