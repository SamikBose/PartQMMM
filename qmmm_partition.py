#!/usr/bin/env python3
"""
qmmm_partition.py
=================
Library (helper) for partitioning an MD trajectory into QM and MM subsystems
for electrostatic-embedding QM/MM single points.

This module is ENGINE-AGNOSTIC: it produces, per frame, the data structures a
QM/MM single point needs:
    - QM atom elements + coordinates (heavy atoms + ligand + metal + link Hs)
    - MM point charges + coordinates (with boundary-charge redistribution)
QM-engine-specific input writers (ORCA, Psi4, Q-Chem, ...) live in separate
driver scripts that import this module and consume `FramePartition` objects.

Boundary scheme (link-atom method):
    - QM/MM covalent cuts are made at Cα-Cβ bonds for sidechains.
    - A link H is placed along Cβ -> Cα at 1.09 Å from Cβ (capping the QM side).
    - The MM boundary atom (Cα) charge is ZEROED and redistributed equally over
      its M2 neighbors (backbone N, C, HA) — the Walker-Crowley-Case scheme
      (Walker, Crowley & Case, J. Comput. Chem. 2008, 29, 1019). This keeps the
      total system charge integral and avoids over-polarizing the QM region with
      a fractional point charge sitting on the cut bond.

Notes baked in from a real hCA II + AZM build:
    - RESNUM_OFFSET handles topologies renumbered from the original PDB
      (tleap/CHARMM-GUI AMBER often renumber starting at the first resolved
      residue; for cleaned 3HS4 with residues 1-3 missing the offset is -4).
    - Virtual sites (OPC EPW, element symbol 'VS'/atomic number 0) are excluded
      from the QM region (QM engines can't take them) but the whole water is
      still removed from the MM field when that water is selected into QM.
    - Topology is read with mdtraj.load_prmtop (no OpenMM dependency) and
      charges with parmed.

Dependencies: mdtraj, parmed, numpy (scipy only if MM cutoff is used)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import mdtraj as md
except ImportError as e:  # pragma: no cover
    raise ImportError("qmmm_partition requires mdtraj "
                      "(conda install -c conda-forge mdtraj)") from e
try:
    import parmed as pmd
except ImportError as e:  # pragma: no cover
    raise ImportError("qmmm_partition requires parmed "
                      "(conda install -c conda-forge parmed)") from e


# ===========================================================================
# QM region presets
# ===========================================================================
# Residue numbers are given in ORIGINAL PDB numbering; RESNUM_OFFSET maps them
# to the topology's numbering. For cleaned 3HS4 (residues 1-3 unresolved), the
# tleap/CHARMM-GUI AMBER topology starts at residue index such that the offset
# is -4. Set RESNUM_OFFSET=0 if your topology preserves original PDB numbers.
DEFAULT_RESNUM_OFFSET = -4

QM_REGIONS = {
    "minimal": {
        "sidechains": {
            "HIS": [94, 96, 119],   # Zn-coordinating histidines (PDB numbering)
            "THR": [199],           # H-bonds to AZM N1
        },
        "ligand_resnames": ["AZM"],
        "metal_resnames":  ["ZN", "ZN2", "Zn2+"],
        "n_waters_near_metal": 1,
    },
    "larger": {
        "sidechains": {
            "HIS": [64, 94, 96, 119],   # adds His64 proton shuttle
            "THR": [199],
            "GLU": [106],               # H-bonds to Thr199 backbone
        },
        "ligand_resnames": ["AZM"],
        "metal_resnames":  ["ZN", "ZN2", "Zn2+"],
        "n_waters_near_metal": 3,
    },
}

# Sidechain atoms (AMBER ff19SB naming). Only Cβ and beyond go into QM; Cα stays
# MM and is the boundary atom. HID/HIE/HIP variants list their own protons.
SIDECHAIN_ATOMS = {
    "HIS": {"CB", "HB2", "HB3", "CG", "ND1", "HD1", "CE1", "HE1",
            "NE2", "HE2", "CD2", "HD2"},
    "HID": {"CB", "HB2", "HB3", "CG", "ND1", "HD1", "CE1", "HE1",
            "NE2", "CD2", "HD2"},                    # no HE2
    "HIE": {"CB", "HB2", "HB3", "CG", "ND1", "CE1", "HE1",
            "NE2", "HE2", "CD2", "HD2"},             # no HD1
    "HIP": {"CB", "HB2", "HB3", "CG", "ND1", "HD1", "CE1", "HE1",
            "NE2", "HE2", "CD2", "HD2"},
    "THR": {"CB", "HB", "OG1", "HG1", "CG2", "HG21", "HG22", "HG23"},
    "GLU": {"CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "OE1", "OE2"},
    "GLH": {"CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "OE1", "OE2", "HE2"},
}

M2_ATOMS = {"N", "C", "HA"}             # MM neighbors of Cα that absorb its charge
LINK_H_BOND_A = 1.09                    # link C-H bond length (Å)
WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL", "T3P", "OPC"}
WATER_O_NAMES = {"O", "OW", "OH2"}
HID_HIE_VARIANTS = ("HID", "HIE", "HIP", "HSD", "HSE", "HSP")


# ===========================================================================
# Data structures
# ===========================================================================
@dataclass
class QMRegionSpec:
    """Frame-independent description of the QM region in a given topology."""
    qm_static: list           # atom indices always in QM (sidechain+ligand+metal)
    boundary_pairs: list      # list of (cb_idx, ca_idx) for link atoms
    m2_per_cb: list           # list of [N,C,HA] indices per boundary, for charge redist.
    metal_indices: list       # metal atom indices
    water_o_indices: np.ndarray  # O atom index of every water
    n_waters_near_metal: int  # how many nearest waters to pull into QM per frame


@dataclass
class FramePartition:
    """Per-frame QM/MM partition, ready for any QM-engine writer."""
    frame_index: int
    qm_elements: list                 # element symbols, incl. link Hs (as 'H')
    qm_positions_A: np.ndarray        # (n_qm, 3) in Å
    mm_charges: np.ndarray            # (n_mm,) point charges (e)
    mm_positions_A: np.ndarray        # (n_mm, 3) in Å
    n_link_atoms: int = 0
    box_vectors_A: Optional[np.ndarray] = None  # (3,3) if periodic
    qm_total_charge: Optional[float] = None     # set by caller if known
    qm_atom_indices: list = field(default_factory=list)  # real topology atom indices in QM; link Hs excluded
    selected_water_resids: list = field(default_factory=list)
    mm_atom_indices: list = field(default_factory=list)
    mm_cutoff_A: float = 0.0
    mm_cutoff_mode: str = "all"
    mm_full_charge_sum: Optional[float] = None


# ===========================================================================
# Topology loading
# ===========================================================================
def load_topology_and_charges(top_path):
    """Return (mdtraj.Topology, charges ndarray). No OpenMM dependency.

    Format-aware:
      - .parm7/.prmtop : charges from parmed, topology from mdtraj.load_prmtop
                         (the normal production path — real MM charges).
      - .pdb           : topology from mdtraj.load_pdb; charges are ZERO with a
                         warning. Useful for STRUCTURAL testing of the QM-region
                         selection and link-atom geometry only — the MM
                         point-charge field will be all zeros, so do NOT use a
                         PDB for real QM/MM single points.
    """
    lower = str(top_path).lower()
    if lower.endswith((".parm7", ".prmtop", ".top")):
        parm = pmd.load_file(top_path)
        charges = np.array([a.charge for a in parm.atoms], dtype=float)
        mdtop = md.load_prmtop(top_path)
        return mdtop, charges
    if lower.endswith(".pdb"):
        print("  WARNING: loading a PDB — MM charges will be ZERO. This is for "
              "structural testing of QM-region selection only; use a parm7 for "
              "real QM/MM inputs.")
        t = md.load_pdb(top_path)
        charges = np.zeros(t.topology.n_atoms, dtype=float)
        return t.topology, charges
    # Fallback: try parmed for charges, mdtraj prmtop for topology
    parm = pmd.load_file(top_path)
    charges = np.array([a.charge for a in parm.atoms], dtype=float)
    mdtop = md.load_prmtop(top_path)
    return mdtop, charges


# ===========================================================================
# QM region resolution (frame-independent)
# ===========================================================================
def _find_residue(mdtop, name, resnum):
    for res in mdtop.residues:
        if res.name == name and res.resSeq == resnum:
            return res
    return None


def _fail_or_warn(message, strict=True):
    if strict:
        raise ValueError(message)
    print(f"  WARNING: {message}")


def build_qm_region_spec(mdtop, region, resnum_offset=DEFAULT_RESNUM_OFFSET,
                         strict=True):
    """Resolve the QM region into a QMRegionSpec for this topology.

    In production mode (strict=True, default), every requested sidechain,
    ligand, metal, and Cα-Cβ boundary must be present. This avoids silently
    generating a plausible-looking ORCA input for the wrong QM region when the
    residue offset, residue names, protonation labels, or ligand names are off.
    """
    qm_static, boundary_pairs, m2_per_cb = [], [], []

    for resname, resnums in region["sidechains"].items():
        for pdb_resnum in resnums:
            resnum = pdb_resnum + resnum_offset
            res = _find_residue(mdtop, resname, resnum)
            if res is None and resname == "HIS":
                for alt in HID_HIE_VARIANTS:
                    res = _find_residue(mdtop, alt, resnum)
                    if res is not None:
                        break
            if res is None:
                _fail_or_warn(
                    f"required sidechain {resname}{pdb_resnum} "
                    f"(topology resSeq {resnum}) was not found; check "
                    f"--resnum-offset and protonation/residue names.",
                    strict=strict)
                continue

            allowed = SIDECHAIN_ATOMS.get(res.name,
                                          SIDECHAIN_ATOMS.get(resname, set()))
            ca_idx = cb_idx = None
            backbone_m2 = []
            found_m2_names = set()
            found_qm_atoms = []
            for atom in res.atoms:
                if atom.name == "CA":
                    ca_idx = atom.index
                elif atom.name == "CB":
                    cb_idx = atom.index
                    qm_static.append(atom.index)
                    found_qm_atoms.append(atom.name)
                elif atom.name in allowed:
                    qm_static.append(atom.index)
                    found_qm_atoms.append(atom.name)
                elif atom.name in M2_ATOMS:
                    backbone_m2.append(atom.index)
                    found_m2_names.add(atom.name)

            if ca_idx is None or cb_idx is None:
                _fail_or_warn(
                    f"required sidechain boundary for {res.name}{resnum} is "
                    f"missing CA or CB; cannot place a link atom safely.",
                    strict=strict)
                continue
            missing_m2 = sorted(M2_ATOMS - found_m2_names)
            if missing_m2:
                _fail_or_warn(
                    f"required M2 atom(s) {missing_m2} missing in "
                    f"{res.name}{resnum}; cannot redistribute boundary charge "
                    f"safely.",
                    strict=strict)
            if not found_qm_atoms:
                _fail_or_warn(
                    f"no sidechain QM atoms selected for {res.name}{resnum}; "
                    f"check atom naming against SIDECHAIN_ATOMS.",
                    strict=strict)

            boundary_pairs.append((cb_idx, ca_idx))
            m2_per_cb.append(backbone_m2)

    ligand_hits = 0
    for ligname in region["ligand_resnames"]:
        for res in mdtop.residues:
            if res.name == ligname:
                ligand_hits += 1
                qm_static.extend(a.index for a in res.atoms)
    if ligand_hits == 0 and region.get("ligand_resnames"):
        _fail_or_warn(
            f"none of the required ligand residue names "
            f"{region['ligand_resnames']} were found.", strict=strict)

    metal_indices = []
    metal_hits = 0
    for metalname in region["metal_resnames"]:
        for res in mdtop.residues:
            if res.name == metalname:
                metal_hits += 1
                idxs = [a.index for a in res.atoms]
                qm_static.extend(idxs)
                metal_indices.extend(idxs)
    if metal_hits == 0 and region.get("metal_resnames"):
        _fail_or_warn(
            f"none of the required metal residue names "
            f"{region['metal_resnames']} were found.", strict=strict)

    water_o = _find_water_oxygens(mdtop)
    if len(water_o) < int(region.get("n_waters_near_metal", 0)):
        _fail_or_warn(
            f"requested {region.get('n_waters_near_metal', 0)} nearest water(s), "
            f"but only found {len(water_o)} water oxygen(s).", strict=strict)

    return QMRegionSpec(
        qm_static=sorted(set(qm_static)),
        boundary_pairs=boundary_pairs,
        m2_per_cb=m2_per_cb,
        metal_indices=metal_indices,
        water_o_indices=water_o,
        n_waters_near_metal=region["n_waters_near_metal"],
    )

def _find_water_oxygens(mdtop):
    out = []
    for res in mdtop.residues:
        if res.name in WATER_RESNAMES:
            for atom in res.atoms:
                if atom.name in WATER_O_NAMES or (
                        atom.element is not None and atom.element.symbol == "O"):
                    out.append(atom.index)
                    break
    return np.array(out, dtype=int)


def _water_atoms_for_oxygen(mdtop, o_idx):
    return [a.index for a in mdtop.atom(o_idx).residue.atoms]


def _real_qm_atom_indices(mdtop, atom_indices):
    """Drop virtual sites from topology atom indices before writing QM atoms."""
    real = []
    for ai in atom_indices:
        atom = mdtop.atom(ai)
        el = atom.element.symbol if atom.element is not None else "VS"
        if el == "VS" or (atom.element is not None and atom.element.number == 0):
            continue
        real.append(ai)
    return real


def _validate_qm_water_residue(mdtop, o_idx):
    """Require a selected QM water to contribute exactly O/H/H real atoms.

    Four-site waters such as OPC may contain a virtual charge site; that site is
    allowed in the topology but excluded from the QM coordinates. A selected
    water with missing/extra real atoms would change the QM atom count and is
    therefore rejected.
    """
    res = mdtop.atom(o_idx).residue
    atoms = _water_atoms_for_oxygen(mdtop, o_idx)
    real = _real_qm_atom_indices(mdtop, atoms)
    symbols = []
    for ai in real:
        atom = mdtop.atom(ai)
        symbols.append(atom.element.symbol if atom.element is not None else "")
    if symbols.count("O") != 1 or symbols.count("H") != 2 or len(symbols) != 3:
        raise ValueError(
            f"selected QM water {res.name}{res.resSeq} does not have exactly "
            f"one O and two H real atoms after excluding virtual sites; got "
            f"{symbols}. This would make the QM atom count inconsistent.")
    return atoms, real, res.index


def _whole_residue_mm_cutoff_indices(mdtop, xyz_A, qm_positions_A, qm_set,
                                      cutoff_A, always_keep=()):
    """Return MM atom indices for a whole-residue electrostatic cutoff.

    First find MM atoms within cutoff_A of any QM atom. Then expand those hits
    to every atom in the same topology residue/molecule, excluding real QM atoms
    because they are represented quantum-mechanically. This prevents the old
    atom-wise cutoff from keeping half of a water molecule or part of a charged
    residue, which creates artificial point-charge fragments.
    """
    if cutoff_A <= 0:
        return [i for i in range(mdtop.n_atoms) if i not in qm_set]

    mm_indices = np.array([i for i in range(mdtop.n_atoms) if i not in qm_set],
                          dtype=int)
    if len(mm_indices) == 0:
        return []

    close = _within_cutoff(xyz_A[mm_indices], qm_positions_A, cutoff_A)
    residue_ids = {mdtop.atom(int(i)).residue.index for i in mm_indices[close]}

    keep = set(int(i) for i in always_keep if int(i) not in qm_set)
    for rid in residue_ids:
        res = mdtop.residue(rid)
        for atom in res.atoms:
            if atom.index not in qm_set:
                keep.add(atom.index)

    return sorted(keep)


# ===========================================================================
# Geometry / charge helpers
# ===========================================================================
def _closest_n_water_o(xyz_A, metal_idx, water_o_idx, n):
    if n <= 0 or len(water_o_idx) == 0:
        return np.array([], dtype=int)
    d = np.linalg.norm(xyz_A[water_o_idx] - xyz_A[metal_idx], axis=1)
    return water_o_idx[np.argsort(d)[:n]]


def _place_link_atom(cb_pos_A, ca_pos_A, bond=LINK_H_BOND_A):
    v = ca_pos_A - cb_pos_A
    v = v / np.linalg.norm(v)
    return cb_pos_A + bond * v


def _redistribute_charges(base_charges, boundary_pairs, m2_per_cb):
    """Zero each Cα charge; spread it over that residue's M2 atoms (N, C, HA)."""
    q = base_charges.copy()
    for (cb_idx, ca_idx), m2 in zip(boundary_pairs, m2_per_cb):
        if not m2:
            continue
        share = q[ca_idx] / len(m2)
        q[ca_idx] = 0.0
        for m in m2:
            q[m] += share
    return q


def _enforce_qm_integer_charge(q, qm_atom_indices, m2_per_cb):
    """
    Make the QM fragment's partial-charge sum an exact integer by moving the
    residual onto the MM boundary (M2) atoms.

    When a sidechain is cut at Cα-Cβ, the QM fragment (Cβ and beyond) does NOT
    carry an integer partial charge in an additive force field — the residue's
    charge group was split. But a QM/MM single point declares an INTEGER formal
    charge for the QM region. If the MM field doesn't compensate, the combined
    QM+MM system carries a spurious non-integer net charge sitting right at the
    boundary, which systematically polarizes the QM density.

    This routine computes the QM fragment's real partial-charge sum, rounds to
    the nearest integer (the formal charge the QM engine will use), and spreads
    the residual over the M2 atoms so the MM field becomes consistent with that
    integer. Total system charge is preserved.

    Returns (modified_charges, qm_real_sum, qm_formal_int).
    """
    qm_real = float(sum(q[i] for i in qm_atom_indices))
    qm_formal = round(qm_real)
    residual = qm_real - qm_formal           # excess that belongs in MM
    all_m2 = [m for sub in m2_per_cb for m in sub]
    if all_m2 and abs(residual) > 1e-9:
        per = residual / len(all_m2)
        for m in all_m2:
            q[m] += per
    return q, qm_real, qm_formal


def _within_cutoff(mm_pos_A, qm_pos_A, cutoff):
    from scipy.spatial import cKDTree
    tree = cKDTree(qm_pos_A)
    d, _ = tree.query(mm_pos_A, k=1, distance_upper_bound=cutoff)
    return np.isfinite(d) & (d <= cutoff)


# ===========================================================================
# Core: partition a single frame
# ===========================================================================
def partition_frame(mdtop, charges_full, spec, xyz_A, frame_index,
                    mm_cutoff_A=0.0, box_vectors_A=None,
                    enforce_integer_qm_charge=True):
    """
    Build a FramePartition for one frame.

    xyz_A           : (n_atoms, 3) coordinates in Å
    spec            : QMRegionSpec from build_qm_region_spec
    mm_cutoff_A     : if >0, keep whole MM residues/molecules with at least one
                      MM atom within this distance of any QM atom (0=all).
    box_vectors_A   : optional (3,3) for periodic info passed through to output
    enforce_integer_qm_charge : if True (default), adjust MM boundary charges so
                      the QM fragment's charge is the exact integer the QM engine
                      will use, keeping QM+MM consistent (recommended).
    """
    # Choose the nearest waters to the (first) metal this frame.
    if spec.metal_indices:
        near_o = _closest_n_water_o(xyz_A, spec.metal_indices[0],
                                    spec.water_o_indices, spec.n_waters_near_metal)
    else:
        near_o = np.array([], dtype=int)

    qm_water_atoms = []
    selected_water_resids = []
    for o_idx in near_o:
        water_atoms, _real_water_atoms, resid = _validate_qm_water_residue(mdtop, o_idx)
        qm_water_atoms.extend(water_atoms)
        selected_water_resids.append(resid)

    qm_atoms = sorted(set(spec.qm_static) | set(qm_water_atoms))
    qm_real_atoms = _real_qm_atom_indices(mdtop, qm_atoms)

    # QM coordinates/elements; skip virtual sites (atomic_number 0 / 'VS')
    qm_positions, qm_elements = [], []
    for ai in qm_real_atoms:
        atom = mdtop.atom(ai)
        el = atom.element.symbol if atom.element is not None else "VS"
        qm_positions.append(xyz_A[ai])
        qm_elements.append(el)

    # Link atoms (capping Hs)
    for (cb_idx, ca_idx) in spec.boundary_pairs:
        qm_positions.append(_place_link_atom(xyz_A[cb_idx], xyz_A[ca_idx]))
        qm_elements.append("H")
    n_link = len(spec.boundary_pairs)

    qm_positions = np.asarray(qm_positions)

    # MM point charges: Cα redistribution, then optional integer-charge fix.
    # Do this before cutoff so the retained point charges inherit the corrected
    # boundary electrostatics.
    mm_charges_full = _redistribute_charges(charges_full, spec.boundary_pairs,
                                            spec.m2_per_cb)
    qm_formal = None
    if enforce_integer_qm_charge:
        mm_charges_full, qm_real, qm_formal = _enforce_qm_integer_charge(
            mm_charges_full, qm_atoms, spec.m2_per_cb)

    qm_set = set(qm_atoms)
    always_keep = set()
    for (_cb_idx, ca_idx), m2 in zip(spec.boundary_pairs, spec.m2_per_cb):
        always_keep.add(ca_idx)
        always_keep.update(m2)

    if mm_cutoff_A and mm_cutoff_A > 0:
        mm_indices = _whole_residue_mm_cutoff_indices(
            mdtop, xyz_A, qm_positions, qm_set, mm_cutoff_A, always_keep=always_keep)
        cutoff_mode = "whole_residue"
    else:
        mm_indices = [i for i in range(len(mm_charges_full)) if i not in qm_set]
        cutoff_mode = "all"

    mm_positions = xyz_A[mm_indices]
    mm_charges = mm_charges_full[mm_indices]

    full_mm_indices = [i for i in range(len(mm_charges_full)) if i not in qm_set]
    full_mm_charge_sum = float(np.sum(mm_charges_full[full_mm_indices]))

    return FramePartition(
        frame_index=frame_index,
        qm_elements=qm_elements,
        qm_positions_A=qm_positions,
        mm_charges=mm_charges,
        mm_positions_A=mm_positions,
        n_link_atoms=n_link,
        box_vectors_A=box_vectors_A,
        qm_total_charge=qm_formal,
        qm_atom_indices=qm_real_atoms,
        selected_water_resids=selected_water_resids,
        mm_atom_indices=list(map(int, mm_indices)),
        mm_cutoff_A=float(mm_cutoff_A or 0.0),
        mm_cutoff_mode=cutoff_mode,
        mm_full_charge_sum=full_mm_charge_sum,
    )


# ===========================================================================
# Trajectory iteration
# ===========================================================================
def iter_frame_partitions(top_path, traj_path, region_name="minimal",
                          resnum_offset=DEFAULT_RESNUM_OFFSET,
                          frames="0:None:1", mm_cutoff_A=0.0,
                          enforce_integer_qm_charge=True, strict=True):
    """
    Generator yielding FramePartition objects for the requested frames.

    Handles single-frame restart files (.rst7/.inpcrd/.ncrst) and multi-frame
    trajectories (.dcd/.nc/.xtc/.h5) transparently.
    """
    if region_name not in QM_REGIONS:
        raise ValueError(f"Unknown region '{region_name}'. "
                         f"Choices: {list(QM_REGIONS)}")
    region = QM_REGIONS[region_name]

    mdtop, charges_full = load_topology_and_charges(top_path)
    spec = build_qm_region_spec(mdtop, region, resnum_offset, strict=strict)
    if not spec.metal_indices:
        _fail_or_warn("no metal atoms found — check metal_resnames in the region.",
                      strict=strict)

    start, stop, stride = _parse_frames(frames)

    single = traj_path.endswith((".rst7", ".rst", ".inpcrd", ".ncrst", ".pdb"))
    if single:
        if start > 0:
            return
        if stop is not None and stop <= 0:
            return
        if traj_path.endswith(".pdb"):
            chunks = [md.load_pdb(traj_path)]
        else:
            chunks = [md.load(traj_path, top=mdtop)]
        for chunk in chunks:
            xyz_A = chunk.xyz[0] * 10.0
            box = None
            if chunk.unitcell_vectors is not None:
                box = chunk.unitcell_vectors[0] * 10.0
            yield partition_frame(mdtop, charges_full, spec, xyz_A, 0,
                                  mm_cutoff_A, box,
                                  enforce_integer_qm_charge=enforce_integer_qm_charge)
        return

    # Important: MDTraj's `stride` starts at physical frame 0. If we pass only
    # stride=stride and then label the first returned frame as `start`,
    # frames="100:200:10" would actually process frames 0,10,20,... but name
    # them 100,110,120,... . Use skip=start so the physical frame and label agree.
    chunks = md.iterload(traj_path, top=mdtop, chunk=200, skip=start, stride=stride)

    seen = 0
    for chunk in chunks:
        for j in range(chunk.n_frames):
            global_idx = start + seen * stride
            seen += 1
            if stop is not None and global_idx >= stop:
                return
            xyz_A = chunk.xyz[j] * 10.0
            box = None
            if chunk.unitcell_vectors is not None:
                box = chunk.unitcell_vectors[j] * 10.0
            yield partition_frame(mdtop, charges_full, spec, xyz_A,
                                  global_idx, mm_cutoff_A, box,
                                  enforce_integer_qm_charge=enforce_integer_qm_charge)


def _parse_frames(spec):
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--frames must be start:stop:stride, got {spec!r}")
    start = int(parts[0]) if parts[0] else 0
    stop = None if parts[1] in ("", "None") else int(parts[1])
    stride = int(parts[2]) if parts[2] else 1
    return start, stop, stride


# ===========================================================================
# Diagnostics
# ===========================================================================
def summarize_partition(fp: FramePartition):
    from collections import Counter
    heavy = sum(1 for e in fp.qm_elements if e != "H")
    return {
        "frame": fp.frame_index,
        "n_qm": len(fp.qm_elements),
        "n_qm_heavy": heavy,
        "n_link": fp.n_link_atoms,
        "elements": dict(Counter(fp.qm_elements)),
        "n_mm": len(fp.mm_charges),
        "mm_charge_sum": float(np.sum(fp.mm_charges)),
        "mm_full_charge_sum": fp.mm_full_charge_sum,
        "mm_cutoff_A": fp.mm_cutoff_A,
        "mm_cutoff_mode": fp.mm_cutoff_mode,
        "n_qm_real_atoms": len(fp.qm_atom_indices),
        "selected_water_resids": fp.selected_water_resids,
        "qm_formal_charge": fp.qm_total_charge,
    }
