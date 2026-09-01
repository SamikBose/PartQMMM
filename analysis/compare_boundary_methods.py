#!/usr/bin/env python3
"""Compare frame-level electrostatic diagnostics from shift and RCD analyses.

Example:
  python compare_boundary_methods.py \
    ensemble_shift/minimal_shift/frame_metrics.csv \
    ensemble_rcd/minimal_rcd/frame_metrics.csv \
    --output boundary_comparison.csv
"""
from __future__ import annotations
import argparse
import pandas as pd

KEYS = ["run", "frame"]
COMPARE = [
    "Emax_V_per_A", "Emax_real_qm_V_per_A", "Emean_V_per_A",
    "E_Zn_V_per_A", "potential_Zn_V", "nearest_mm_charge_to_Zn_A",
    "potential_abs_max_V", "nearest_mm_charge_to_qm_A",
    "nearest_topology_mm_charge_to_qm_A",
    "n_mm_charges_within_3A", "n_mm_charges_within_4A", "n_mm_charges_within_5A",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shift_csv")
    ap.add_argument("rcd_csv")
    ap.add_argument("--output", default="boundary_comparison.csv")
    args = ap.parse_args()
    a = pd.read_csv(args.shift_csv)
    b = pd.read_csv(args.rcd_csv)
    keep_a = KEYS + [c for c in COMPARE if c in a.columns]
    keep_b = KEYS + [c for c in COMPARE if c in b.columns]
    m = a[keep_a].merge(b[keep_b], on=KEYS, suffixes=("_shift", "_rcd"), validate="one_to_one")
    for c in COMPARE:
        cs, cr = f"{c}_shift", f"{c}_rcd"
        if cs in m.columns and cr in m.columns:
            m[f"delta_{c}_rcd_minus_shift"] = m[cr] - m[cs]
    m.to_csv(args.output, index=False)
    print(f"Wrote {len(m)} matched frames to {args.output}")


if __name__ == "__main__":
    main()
