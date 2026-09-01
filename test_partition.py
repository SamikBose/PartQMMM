#!/usr/bin/env python3
"""Focused PartQMMM V1 regression tests."""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import qmmm_partition as qp


HERE = os.path.dirname(__file__)


def test_triclinic_minimum_image():
    box = np.array([
        [10.0, 0.0, 0.0],
        [2.0, 9.0, 0.0],
        [1.0, 1.0, 8.0],
    ])
    # Construct points differing by one lattice vector plus a short displacement.
    short = np.array([0.3, -0.2, 0.1])
    delta = box[1] + short
    got = qp.minimum_image_displacement(delta, box)
    assert np.allclose(got, short, atol=1e-10), (got, short)


def test_pbc_hbond_geometry():
    box = np.diag([10.0, 10.0, 10.0])
    # D-H...A crosses x boundary and should still be linear/short.
    d = np.array([9.8, 5.0, 5.0])
    h = np.array([9.9, 5.0, 5.0])
    a = np.array([0.2, 5.0, 5.0])
    assert abs(qp.pbc_distance(d, a, box) - 0.4) < 1e-10
    assert qp.pbc_angle_dha(d, h, a, box) > 179.9


def test_rcd_pbc_midpoint_charge_and_dipole():
    """RCD midpoint must follow the minimum-image CA--M2 bond, not box center."""
    box = np.diag([10.0, 10.0, 10.0])
    xyz = np.array([
        [9.0, 5.0, 5.0],
        [9.8, 5.0, 5.0],
        [0.2, 5.0, 5.0],
    ])
    charges = np.array([0.0, 0.12, -0.40])
    q_real, qvirt, pvirt, labels = qp._rcd_boundary_charges(
        charges, xyz, [(0, 1)], [[2]], anchor_index=1, box_vectors_A=box
    )
    assert len(qvirt) == 1 and len(labels) == 1
    assert abs(qvirt[0] - 0.24) < 1e-12
    assert abs(q_real[1]) < 1e-12
    assert abs(q_real[2] - (-0.52)) < 1e-12
    assert abs(q_real.sum() + qvirt.sum() - charges.sum()) < 1e-12
    assert abs(qp.pbc_distance(xyz[1], pvirt[0], box) - 0.2) < 1e-12
    assert abs(qp.pbc_distance(pvirt[0], xyz[2], box) - 0.2) < 1e-12
    d_m2 = qp.minimum_image_displacement(xyz[2] - xyz[1], box)
    d_v = qp.minimum_image_displacement(pvirt[0] - xyz[1], box)
    delta_mu = (-0.12) * d_m2 + 0.24 * d_v
    assert np.allclose(delta_mu, 0.0, atol=1e-12), delta_mu


def test_uploaded_example(top_path, pdb_path):
    mdtop, charges = qp.load_topology_and_charges(top_path)
    spec = qp.build_qm_region_spec(
        mdtop, charges, qp.QM_REGIONS["minimal"], qp.DEFAULT_RESNUM_OFFSET
    )
    import mdtraj as md
    traj = md.load_pdb(pdb_path)
    box = traj.unitcell_vectors[0] * 10.0 if traj.unitcell_vectors is not None else None
    fp = qp.partition_frame(
        mdtop, charges, spec, traj.xyz[0] * 10.0, 0, box,
        hbond_distance_A=3.5, hbond_angle_deg=140.0,
        boundary_charge_method="shift",
    )
    assert fp.qm_formal_charge == +1
    assert abs(fp.qm_formal_charge + fp.mm_charge_sum - fp.system_charge) < 1e-5
    assert set(fp.qm_all_topology_indices).isdisjoint(fp.mm_atom_indices)
    assert fp.n_link_atoms == 4
    assert fp.boundary_charge_method == "shift"
    assert fp.n_boundary_virtual_sites == 0
    assert fp.selected_zn_coordinated_water_resids == []

    fp_rcd = qp.partition_frame(
        mdtop, charges, spec, traj.xyz[0] * 10.0, 0, box,
        hbond_distance_A=3.5, hbond_angle_deg=140.0,
        boundary_charge_method="rcd",
    )
    assert fp_rcd.qm_atom_indices == fp.qm_atom_indices
    assert fp_rcd.qm_elements == fp.qm_elements
    assert np.allclose(fp_rcd.qm_positions_A, fp.qm_positions_A)
    assert fp_rcd.qm_formal_charge == fp.qm_formal_charge
    assert fp_rcd.selected_water_resids == fp.selected_water_resids
    assert fp_rcd.boundary_charge_method == "rcd"
    assert fp_rcd.n_boundary_virtual_sites == sum(map(len, spec.m2_per_cb))
    assert len(fp_rcd.mm_charges) == (
        len(fp_rcd.mm_atom_indices) + fp_rcd.n_boundary_virtual_sites
    )
    assert fp_rcd.mm_site_types.count("rcd_virtual") == fp_rcd.n_boundary_virtual_sites
    assert abs(fp_rcd.qm_formal_charge + fp_rcd.mm_charge_sum - fp_rcd.system_charge) < 1e-5
    assert abs(fp_rcd.boundary_charge_residual - fp.boundary_charge_residual) < 1e-12
    # Every selected adaptive water must be neutral in the topology.
    selected = set(fp.selected_water_resids)
    for water in spec.water_records:
        if water.residue_index in selected:
            assert abs(charges[list(water.all_atom_indices)].sum()) < 1e-4

    # Force one real OPC water into a clean O-H...A_core hydrogen bond and
    # verify variable-size QM selection without charge change or double counting.
    xyz2 = traj.xyz[0].copy() * 10.0
    water = spec.water_records[0]
    acc = spec.core_hbond_acceptors[0]
    a_pos = xyz2[acc].copy()
    o, h1, h2 = water.oxygen_index, *water.hydrogen_indices
    xyz2[o] = a_pos + np.array([2.80, 0.00, 0.00])
    xyz2[h1] = a_pos + np.array([1.84, 0.00, 0.00])  # linear O-H...A
    xyz2[h2] = xyz2[o] + np.array([0.00, 0.96, 0.00])
    fp2 = qp.partition_frame(
        mdtop, charges, spec, xyz2, 1, box,
        hbond_distance_A=3.5, hbond_angle_deg=140.0,
        boundary_charge_method="rcd",
    )
    assert water.residue_index in fp2.selected_water_resids
    assert set(water.all_atom_indices).issubset(fp2.qm_all_topology_indices)
    assert set(water.all_atom_indices).isdisjoint(fp2.mm_atom_indices)
    # OPC contributes four topology sites but only three real QM XYZ atoms.
    assert len(fp2.qm_all_topology_indices) >= len(fp.qm_all_topology_indices) + 4
    assert len(fp2.qm_elements) >= len(fp.qm_elements) + 3
    assert fp2.qm_formal_charge == fp.qm_formal_charge == +1
    assert abs(fp2.qm_formal_charge + fp2.mm_charge_sum - fp2.system_charge) < 1e-5

    # Force a neutral OPC water into the Zn first coordination shell. It must be
    # selected even if it has no H-bond reason, and its full topology residue
    # (including virtual site) must disappear from MM while formal charge stays +1.
    xyz3 = traj.xyz[0].copy() * 10.0
    zn_idx = spec.metal_indices[0]
    water3 = spec.water_records[1]
    o3, h31, h32 = water3.oxygen_index, *water3.hydrogen_indices
    zn_pos = xyz3[zn_idx].copy()
    xyz3[o3] = zn_pos + np.array([2.10, 0.00, 0.00])
    xyz3[h31] = xyz3[o3] + np.array([0.96, 0.00, 0.00])
    xyz3[h32] = xyz3[o3] + np.array([-0.24, 0.93, 0.00])
    zn_selected, zn_reasons = qp._select_zn_coordinated_waters(
        mdtop, spec, xyz3, box, zn_water_cutoff_A=2.6
    )
    assert water3.residue_index in {w.residue_index for w in zn_selected}
    assert any("Zn-coordination" in x for x in zn_reasons[water3.residue_index])
    fp3 = qp.partition_frame(
        mdtop, charges, spec, xyz3, 2, box,
        hbond_distance_A=3.5, hbond_angle_deg=140.0,
        zn_water_cutoff_A=2.6, boundary_charge_method="rcd",
    )
    assert water3.residue_index in fp3.selected_water_resids
    assert water3.residue_index in fp3.selected_zn_coordinated_water_resids
    assert set(water3.all_atom_indices).issubset(fp3.qm_all_topology_indices)
    assert set(water3.all_atom_indices).isdisjoint(fp3.mm_atom_indices)
    assert fp3.qm_formal_charge == +1
    assert abs(fp3.qm_formal_charge + fp3.mm_charge_sum - fp3.system_charge) < 1e-5


def main():
    test_triclinic_minimum_image()
    test_pbc_hbond_geometry()
    test_rcd_pbc_midpoint_charge_and_dipole()
    print("[PASS] PBC minimum-image + RCD midpoint/dipole tests")

    if len(sys.argv) == 3:
        test_uploaded_example(sys.argv[1], sys.argv[2])
        print("[PASS] uploaded-system partition/charge/link tests")
    else:
        print("[INFO] system test skipped; pass parm7 and PDB paths to enable")


if __name__ == "__main__":
    main()
