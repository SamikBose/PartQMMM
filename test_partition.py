#!/usr/bin/env python3
"""
test_partition.py
=================
Structural tests for the qmmm_partition library, run against a PDB of the
built hCA II + AZM + Zn system (system_1264.pdb).

NOTE: a PDB carries no MM charges, so these tests validate the STRUCTURAL
parts only — QM-region selection, link-atom placement, Zn coordination, region
sizes. For charge-dependent behavior, run against the parm7 (system_1264.parm7).

Run:  python test_partition.py        (plain asserts, no pytest needed)
  or: pytest test_partition.py
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import qmmm_partition as qp

PDB = os.path.join(os.path.dirname(__file__), "system_1264.pdb")
OFFSET = -4   # this system: residues renumbered from 0; PDB His94 -> resSeq 90


def _partition(region):
    fps = list(qp.iter_frame_partitions(
        PDB, PDB, region_name=region, resnum_offset=OFFSET, frames="0:1:1"))
    assert len(fps) == 1, f"expected 1 frame, got {len(fps)}"
    return fps[0]


def test_minimal_region_sizes():
    fp = _partition("minimal")
    s = qp.summarize_partition(fp)
    assert s["n_qm_heavy"] == 36, f"minimal heavy atoms {s['n_qm_heavy']} != 36"
    assert s["n_link"] == 4, f"minimal link atoms {s['n_link']} != 4"
    assert s["elements"].get("Zn", 0) == 1, "exactly one Zn expected in QM"
    assert s["elements"].get("S", 0) == 2, "two S (AZM ring + sulfonyl) expected"
    print("[PASS] minimal region sizes (36 heavy, 4 links, 1 Zn, 2 S)")


def test_larger_region_sizes():
    fp = _partition("larger")
    s = qp.summarize_partition(fp)
    assert s["n_qm_heavy"] == 49, f"larger heavy atoms {s['n_qm_heavy']} != 49"
    assert s["n_link"] == 6, f"larger link atoms {s['n_link']} != 6"
    print("[PASS] larger region sizes (49 heavy, 6 links)")


def test_link_atoms_are_hydrogens():
    fp = _partition("minimal")
    link_els = fp.qm_elements[-fp.n_link_atoms:]
    assert all(e == "H" for e in link_els), f"link atoms not all H: {link_els}"
    print("[PASS] all link atoms are hydrogens")


def test_zn_tetrahedral_coordination():
    """Zn should be coordinated by exactly 4 N atoms (3 His + AZM N1) at ~2 A."""
    fp = _partition("minimal")
    zn_i = [i for i, e in enumerate(fp.qm_elements) if e == "Zn"][0]
    zn = fp.qm_positions_A[zn_i]
    coord_n = []
    for i, (e, p) in enumerate(zip(fp.qm_elements, fp.qm_positions_A)):
        if e == "N" and np.linalg.norm(p - zn) < 2.6:
            coord_n.append(np.linalg.norm(p - zn))
    assert len(coord_n) == 4, f"expected 4 coordinating N, got {len(coord_n)}: {coord_n}"
    assert all(1.8 < d < 2.2 for d in coord_n), f"Zn-N distances off: {coord_n}"
    print(f"[PASS] Zn 4-coordinate; Zn-N = {[f'{d:.2f}' for d in sorted(coord_n)]} A")


def test_link_atom_bond_length():
    """Each link H sits ~1.09 A from its Cbeta (we re-derive from the spec)."""
    mdtop, charges = qp.load_topology_and_charges(PDB)
    spec = qp.build_qm_region_spec(mdtop, qp.QM_REGIONS["minimal"], OFFSET)
    import mdtraj as md
    t = md.load_pdb(PDB)
    xyz_A = t.xyz[0] * 10.0
    for (cb, ca) in spec.boundary_pairs:
        link = qp._place_link_atom(xyz_A[cb], xyz_A[ca])
        d = np.linalg.norm(link - xyz_A[cb])
        assert abs(d - 1.09) < 1e-3, f"link C-H {d:.3f} != 1.09"
    print("[PASS] all link H placed at 1.09 A from Cbeta")


def test_qm_atoms_excluded_from_mm():
    """No QM atom index should appear in the MM field (by construction)."""
    fp = _partition("minimal")
    # MM charges are zero for a PDB, but the COUNT must be total - qm_real.
    # qm real atoms = heavy + H (not link atoms, which are synthetic).
    # We just assert the MM field is smaller than the full system by at least
    # the number of selected real QM atoms.
    import mdtraj as md
    total = md.load_pdb(PDB).topology.n_atoms
    n_real_qm = len(fp.qm_elements) - fp.n_link_atoms
    assert len(fp.mm_charges) <= total - n_real_qm, "QM atoms not excluded from MM"
    print(f"[PASS] QM atoms excluded from MM field "
          f"({len(fp.mm_charges)} MM atoms, {n_real_qm} real QM removed)")


def test_charge_conservation_and_integrality():
    """parm7-only: with integer-charge enforcement, MM field = system - QM_formal,
    and QM_formal + MM = system net charge. Skips if the parm7 isn't present."""
    parm = os.path.join(os.path.dirname(__file__), "system_1264.parm7")
    rst = os.path.join(os.path.dirname(__file__), "system_1264.rst7")
    if not (os.path.exists(parm) and os.path.exists(rst)):
        print("[SKIP] charge test (system_1264.parm7/.rst7 not present)")
        return
    mdtop, charges = qp.load_topology_and_charges(parm)
    total = float(charges.sum())
    fp = list(qp.iter_frame_partitions(
        parm, rst, region_name="minimal", resnum_offset=OFFSET,
        frames="0:1:1", enforce_integer_qm_charge=True))[0]
    mm_sum = float(np.sum(fp.mm_charges))
    qm_formal = fp.qm_total_charge
    assert qm_formal == 1, f"expected QM formal +1, got {qm_formal}"
    assert abs((qm_formal + mm_sum) - total) < 1e-3, "QM+MM != system charge"
    assert abs(mm_sum - (total - qm_formal)) < 1e-3, "MM not integer-consistent"
    print(f"[PASS] charge conserved & integer-consistent "
          f"(QM={qm_formal:+d}, MM={mm_sum:+.4f}, sys={total:+.4f})")


def test_whole_residue_cutoff_keeps_complete_mm_residues():
    """parm7-only: cutoff must not split waters/residues into partial charges."""
    parm = os.path.join(os.path.dirname(__file__), "system_1264.parm7")
    rst = os.path.join(os.path.dirname(__file__), "system_1264.rst7")
    if not (os.path.exists(parm) and os.path.exists(rst)):
        print("[SKIP] whole-residue cutoff test (parm7/rst7 not present)")
        return
    mdtop, _charges = qp.load_topology_and_charges(parm)
    fp = list(qp.iter_frame_partitions(
        parm, rst, region_name="minimal", resnum_offset=OFFSET,
        frames="0:1:1", mm_cutoff_A=15.0))[0]
    keep = set(fp.mm_atom_indices)
    qm = set(fp.qm_atom_indices)
    for res in mdtop.residues:
        mm_atoms = {a.index for a in res.atoms if a.index not in qm}
        if keep & mm_atoms:
            assert mm_atoms <= keep, (
                f"residue {res.name}{res.resSeq} was split by MM cutoff")
    # Whole-residue cutoff can retain a charged shell, but it should be very
    # close to an integer, not an arbitrary atom-wise fractional value.
    mm_sum = float(np.sum(fp.mm_charges))
    assert abs(mm_sum - round(mm_sum)) < 1e-3, (
        f"whole-residue cutoff MM charge is not integer-like: {mm_sum}")
    print(f"[PASS] whole-residue cutoff kept complete residues "
          f"({len(fp.mm_charges)} point charges, MM={mm_sum:+.4f})")


def test_dcd_frame_start_stride_uses_physical_frames():
    """DCD start/stride must process the requested physical frames, not relabel frame 0."""
    import mdtraj as md
    t0 = md.load_pdb(PDB)
    xyz = np.repeat(t0.xyz, 5, axis=0)
    for i in range(5):
        xyz[i, :, 0] += 0.1 * i   # nm = 1 Å per frame in x
    traj = md.Trajectory(xyz=xyz, topology=t0.topology)
    with tempfile.TemporaryDirectory() as td:
        dcd = os.path.join(td, "shifted.dcd")
        traj.save_dcd(dcd)
        fps = list(qp.iter_frame_partitions(
            PDB, dcd, region_name="minimal", resnum_offset=OFFSET,
            frames="2:4:1"))
    assert [fp.frame_index for fp in fps] == [2, 3]
    # The Zn x-coordinate in processed physical frame 2 should be shifted by 2 Å.
    fp0 = list(qp.iter_frame_partitions(
        PDB, PDB, region_name="minimal", resnum_offset=OFFSET,
        frames="0:1:1"))[0]
    zn0_i = [i for i, e in enumerate(fp0.qm_elements) if e == "Zn"][0]
    zn2_i = [i for i, e in enumerate(fps[0].qm_elements) if e == "Zn"][0]
    dx = fps[0].qm_positions_A[zn2_i, 0] - fp0.qm_positions_A[zn0_i, 0]
    assert abs(dx - 2.0) < 1e-3, f"expected physical frame 2 shift of 2 Å, got {dx}"
    print("[PASS] DCD frames start:stop:stride use physical frame indices")


def test_wrong_residue_offset_fails_fast():
    """A bad offset should not silently skip required QM residues."""
    try:
        list(qp.iter_frame_partitions(
            PDB, PDB, region_name="minimal", resnum_offset=0, frames="0:1:1"))
    except ValueError as e:
        assert "required sidechain" in str(e) or "required" in str(e)
        print("[PASS] wrong residue offset fails fast")
        return
    raise AssertionError("bad residue offset did not fail")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} structural tests on {os.path.basename(PDB)}\n")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
