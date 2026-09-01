#!/usr/bin/env python3
"""Tests for PartQMMM ensemble analysis. Run: pytest -q test_analysis.py"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import mdtraj as md

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
import qmmm_partition as qp
from analysis_core import (
    COULOMB_V_ANGSTROM_PER_E,
    structured_hbond_contacts,
    zn_coordinated_water_contacts,
    nearest_zn_water,
    embedding_field_metrics,
    site_specific_field_rows,
)


def test_coulomb_conversion_constant():
    assert abs(COULOMB_V_ANGSTROM_PER_E - 14.39964548) < 1e-7


def real_system_smoke(parm7: str, pdb: str):
    top, charges = qp.load_topology_and_charges(parm7)
    traj = md.load_pdb(pdb)
    xyz = traj.xyz[0] * 10.0
    box = traj.unitcell_vectors[0] * 10.0 if traj.unitcell_vectors is not None else None
    spec = qp.build_qm_region_spec(top, charges, qp.QM_REGIONS["minimal"], -4)
    results = {}
    for method in ("shift", "rcd"):
        fp = qp.partition_frame(
            top, charges, spec, xyz, 0, box,
            boundary_charge_method=method,
            fail_on_close_ion=False,
        )
        contacts = structured_hbond_contacts(top, spec, xyz, box, -4)
        zn_contacts = zn_coordinated_water_contacts(top, spec, xyz, box, 2.6)
        expected = {c["water_residue_index"] for c in contacts} | {c["water_residue_index"] for c in zn_contacts}
        assert expected == set(fp.selected_water_resids)
        assert len(fp.selected_water_resids) == 4
        assert len(zn_contacts) == 0
        zn = nearest_zn_water(spec, xyz, box)
        assert 5.0 < zn["nearest_zn_water_A"] < 5.3
        emb, fields = embedding_field_metrics(top, spec, fp, -4, chunk_size=10000)
        assert np.isfinite(emb["Emax_V_per_A"])
        assert emb["Emax_atom_label"]
        assert "Emax_is_link_atom" in emb
        assert "Emax_topology_atom_index" in emb
        assert emb["nearest_mm_charge_to_qm_A"] > 0
        sites = site_specific_field_rows(fields)
        assert any(x["site_class"] == "ZN" for x in sites)
        assert any(x["site_class"] == "HIS_N" for x in sites)
        assert abs(fp.charge_error) < 1e-5
        results[method] = (fp, emb, fields)
    assert results["shift"][0].n_boundary_virtual_sites == 0
    assert results["rcd"][0].n_boundary_virtual_sites == 12
    assert results["shift"][0].qm_elements == results["rcd"][0].qm_elements
    assert np.allclose(results["shift"][0].qm_positions_A, results["rcd"][0].qm_positions_A)

    # Synthetic direct Zn coordination must become an adaptive QM water and
    # remain charge-neutral.
    xyz2 = xyz.copy()
    water = spec.water_records[0]
    zn_idx = spec.metal_indices[0]
    o, h1, h2 = water.oxygen_index, *water.hydrogen_indices
    xyz2[o] = xyz2[zn_idx] + np.array([2.10, 0.0, 0.0])
    xyz2[h1] = xyz2[o] + np.array([0.96, 0.0, 0.0])
    xyz2[h2] = xyz2[o] + np.array([-0.24, 0.93, 0.0])
    zc = zn_coordinated_water_contacts(top, spec, xyz2, box, 2.6)
    assert water.residue_index in {c["water_residue_index"] for c in zc}
    fp2 = qp.partition_frame(
        top, charges, spec, xyz2, 1, box,
        zn_water_cutoff_A=2.6, boundary_charge_method="rcd", fail_on_close_ion=False,
    )
    assert water.residue_index in fp2.selected_zn_coordinated_water_resids
    assert fp2.qm_formal_charge == +1
    assert abs(fp2.charge_error) < 1e-5
    emb2, fields2 = embedding_field_metrics(top, spec, fp2, -4, chunk_size=10000)
    sites2 = site_specific_field_rows(fields2)
    assert any(
        x["site_class"] == "QM_WATER_O"
        and x["residue_index"] == water.residue_index
        for x in sites2
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parm7", required=True)
    ap.add_argument("--pdb", required=True)
    args = ap.parse_args()
    test_coulomb_conversion_constant()
    real_system_smoke(args.parm7, args.pdb)
    print("PASS")
