#!/usr/bin/env python3
"""
qmmm_partition.py -- PartQMMM V1
=================================
Engine-agnostic QM/MM partitioning for MLIP reference-data generation.

V1 design choices
-----------------
1. Fixed chemical QM core: selected protein sidechains + ligand + metal.
2. Variable QM waters: a whole water is promoted to QM when it either
   (a) forms an explicit hydrogen bond to a N/O/S atom in the fixed QM core, or
   (b) is directly Zn-coordinated by a Zn--O distance criterion (default 2.6 A).
3. Triclinic PBC: all distance/angle tests use a minimum-image convention.
4. The original C-alpha/C-beta link-atom boundary treatment is retained:
   C-alpha remains MM, C-beta and beyond are QM, and a 1.09 A link H caps
   C-beta. Two electrostatic boundary treatments are available:
     - shift: C-alpha charge is divided equally over N/C/HA (M2 atoms).
     - rcd: Redistributed Charge and Dipole; C-alpha is zeroed, virtual point
       charges are placed at PBC-aware C-alpha--M2 bond midpoints, and M2
       charges are adjusted so charge and the local bond dipole are preserved.
5. The QM formal charge is frame invariant for a given preset. Adaptive waters
   must be neutral and therefore cannot change it.
6. Positive/negative free ions are detected. V1 does not silently delete a
   close ion charge or promote an ion; instead, a close non-QM ion triggers a
   fail-fast guard by default. This avoids both double counting and hidden
   changes of the total electrostatic embedding charge.
7. No MM electrostatic cutoff is applied in V1: every non-QM topology atom is
   retained as a point charge. This avoids an additional hard cutoff boundary.

Dependencies: numpy, mdtraj
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
import math
import re

import numpy as np

try:
    import mdtraj as md
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "qmmm_partition requires mdtraj (conda install -c conda-forge mdtraj)"
    ) from exc


# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------
DEFAULT_RESNUM_OFFSET = -4
LINK_H_BOND_A = 1.09
DEFAULT_ZN_WATER_CUTOFF_A = 2.6
M2_ATOMS = {"N", "C", "HA"}
WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL", "T3P", "OPC"}
WATER_O_NAMES = {"O", "OW", "OH2"}
HID_HIE_VARIANTS = ("HID", "HIE", "HIP", "HSD", "HSE", "HSP")
POLAR_ELEMENTS = {"N", "O", "S"}

# Explicit formal charge is preferable to inferring chemistry by rounding a
# force-field partial-charge sum. Adaptive waters are required to be neutral.
QM_REGIONS = {
    "minimal": {
        "sidechains": {
            "HIS": [94, 96, 119],
            "THR": [199],
        },
        "ligand_resnames": ["AZM"],
        "metal_resnames": ["ZN", "ZN2", "Zn2+"],
        "formal_charge": +1,
    },
    "larger": {
        "sidechains": {
            "HIS": [64, 94, 96, 119],
            "THR": [199],
            "GLU": [106],
        },
        "ligand_resnames": ["AZM"],
        "metal_resnames": ["ZN", "ZN2", "Zn2+"],
        "formal_charge": 0,
    },
}

SIDECHAIN_ATOMS = {
    "HIS": {"CB", "HB2", "HB3", "CG", "ND1", "HD1", "CE1", "HE1",
            "NE2", "HE2", "CD2", "HD2"},
    "HID": {"CB", "HB2", "HB3", "CG", "ND1", "HD1", "CE1", "HE1",
            "NE2", "CD2", "HD2"},
    "HIE": {"CB", "HB2", "HB3", "CG", "ND1", "CE1", "HE1",
            "NE2", "HE2", "CD2", "HD2"},
    "HIP": {"CB", "HB2", "HB3", "CG", "ND1", "HD1", "CE1", "HE1",
            "NE2", "HE2", "CD2", "HD2"},
    "THR": {"CB", "HB", "OG1", "HG1", "CG2", "HG21", "HG22", "HG23"},
    "GLU": {"CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "OE1", "OE2"},
    "GLH": {"CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "OE1", "OE2", "HE2"},
}

# Common free-ion names. Generic charge-based single-atom detection is also used,
# so these lists are not the only way an ion is recognized.
POSITIVE_ION_RESNAMES = {
    "NA", "NA+", "SOD", "K", "K+", "POT", "LI", "LI+", "CS", "CS+",
    "MG", "MG2", "MG2+", "CA", "CA2", "CA2+",
}
NEGATIVE_ION_RESNAMES = {
    "CL", "CL-", "CLA", "BR", "BR-", "IOD", "I-", "F", "F-",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WaterRecord:
    residue_index: int
    residue_name: str
    residue_seq: int
    oxygen_index: int
    hydrogen_indices: tuple[int, ...]
    all_atom_indices: tuple[int, ...]   # includes OPC/TIP4P virtual sites
    real_atom_indices: tuple[int, ...]  # O/H/H only for QM XYZ


@dataclass(frozen=True)
class IonRecord:
    residue_index: int
    residue_name: str
    residue_seq: int
    atom_indices: tuple[int, ...]
    net_charge: float


@dataclass
class QMRegionSpec:
    qm_static: list[int]
    boundary_pairs: list[tuple[int, int]]  # (CB, CA)
    m2_per_cb: list[list[int]]
    metal_indices: list[int]
    formal_charge: int
    water_records: list[WaterRecord]
    core_hbond_donors: dict[int, tuple[int, ...]]
    core_hbond_acceptors: list[int]
    ion_records: list[IonRecord]
    anchor_index: int


@dataclass
class FramePartition:
    frame_index: int
    qm_elements: list[str]
    qm_positions_A: np.ndarray
    mm_charges: np.ndarray
    mm_positions_A: np.ndarray
    qm_formal_charge: int
    system_charge: float
    mm_charge_sum: float
    charge_error: float
    qm_atom_indices: list[int] = field(default_factory=list)  # real atoms only
    qm_all_topology_indices: list[int] = field(default_factory=list)  # incl. virtual sites
    mm_atom_indices: list[int] = field(default_factory=list)
    mm_site_types: list[str] = field(default_factory=list)
    mm_site_labels: list[str] = field(default_factory=list)
    boundary_charge_method: str = "shift"
    n_boundary_virtual_sites: int = 0
    boundary_ff_qm_sum: float = 0.0
    boundary_charge_residual: float = 0.0
    selected_water_resids: list[int] = field(default_factory=list)
    selected_hbond_water_resids: list[int] = field(default_factory=list)
    selected_zn_coordinated_water_resids: list[int] = field(default_factory=list)
    selected_water_reasons: dict[int, list[str]] = field(default_factory=dict)
    n_link_atoms: int = 0
    close_ions: list[str] = field(default_factory=list)
    box_vectors_A: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# AMBER topology / charge loading
# ---------------------------------------------------------------------------
def _parse_fortran_format(fmt_line: str) -> tuple[int, str]:
    """Return (field_width, type_code) for simple AMBER prmtop formats."""
    m = re.search(r"%FORMAT\(\s*\d+\s*([aAiIeEdD])\s*(\d+)", fmt_line)
    if not m:
        raise ValueError(f"Unsupported prmtop format line: {fmt_line!r}")
    return int(m.group(2)), m.group(1).upper()


def _read_prmtop_flag(path: str | Path, flag: str) -> tuple[str, list[str]]:
    fmt = None
    lines: list[str] = []
    in_flag = False
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("%FLAG"):
                name = line.split()[1].strip()
                if in_flag and name != flag:
                    break
                in_flag = (name == flag)
                continue
            if in_flag and line.startswith("%FORMAT"):
                fmt = line.strip()
                continue
            if in_flag and line.startswith("%"):
                continue
            if in_flag:
                lines.append(line.rstrip("\n"))
    if fmt is None:
        raise ValueError(f"%FLAG {flag} not found in {path}")
    return fmt, lines


def _parse_fixed_width(lines: Iterable[str], width: int, kind: str):
    out = []
    for line in lines:
        for start in range(0, len(line), width):
            token = line[start:start + width]
            if not token.strip():
                continue
            if kind == "I":
                out.append(int(token))
            elif kind in {"E", "D"}:
                out.append(float(token.replace("D", "E")))
            else:
                out.append(token.strip())
    return out


def _load_amber_charges(prmtop_path: str | Path) -> np.ndarray:
    """Read AMBER CHARGE directly; values are stored in units of e*18.2223."""
    fmt, lines = _read_prmtop_flag(prmtop_path, "CHARGE")
    width, kind = _parse_fortran_format(fmt)
    raw = np.asarray(_parse_fixed_width(lines, width, kind), dtype=float)
    return raw / 18.2223


def load_topology_and_charges(top_path: str | Path):
    """Return (mdtraj.Topology, charges[e]). Production path is parm7/prmtop."""
    path = str(top_path)
    lower = path.lower()
    if lower.endswith((".parm7", ".prmtop", ".top")):
        mdtop = md.load_prmtop(path)
        charges = _load_amber_charges(path)
        if len(charges) != mdtop.n_atoms:
            raise ValueError(
                f"Topology/charge size mismatch: {mdtop.n_atoms} atoms but "
                f"{len(charges)} charges"
            )
        return mdtop, charges
    if lower.endswith(".pdb"):
        t = md.load_pdb(path)
        print(
            "WARNING: PDB topology has no MM partial charges; using zeros. "
            "Use parm7/prmtop for production QM/MM partitions."
        )
        return t.topology, np.zeros(t.topology.n_atoms, dtype=float)
    raise ValueError("V1 supports .parm7/.prmtop/.top or .pdb topology files")


# ---------------------------------------------------------------------------
# PBC helpers (triclinic minimum image)
# ---------------------------------------------------------------------------
def minimum_image_displacement(delta_A: np.ndarray,
                               box_vectors_A: Optional[np.ndarray]) -> np.ndarray:
    """
    Apply a robust triclinic minimum-image convention to one or many vectors.

    Component-wise fractional wrapping can choose a non-shortest Cartesian
    image for a skewed triclinic cell. We therefore test the 27 lattice images
    neighbouring the nearest fractional-cell translation and retain the
    shortest Cartesian displacement.
    """
    delta = np.asarray(delta_A, dtype=float)
    if box_vectors_A is None:
        return delta.copy()
    box = np.asarray(box_vectors_A, dtype=float)
    if box.shape != (3, 3):
        raise ValueError(f"box_vectors_A must be (3,3), got {box.shape}")
    if abs(np.linalg.det(box)) < 1e-10:
        raise ValueError("Periodic box is singular")
    original_shape = delta.shape
    if original_shape[-1] != 3:
        raise ValueError(
            f"displacement vectors must end in dimension 3, got {original_shape}"
        )
    flat = delta.reshape(-1, 3)
    frac = flat @ np.linalg.inv(box)
    centre = np.round(frac)
    best = None
    best_d2 = None
    for i in (-1.0, 0.0, 1.0):
        for j in (-1.0, 0.0, 1.0):
            for k in (-1.0, 0.0, 1.0):
                candidate = (frac - (centre + np.array([i, j, k]))) @ box
                d2 = np.einsum("ij,ij->i", candidate, candidate)
                if best is None:
                    best = candidate.copy()
                    best_d2 = d2.copy()
                else:
                    mask = d2 < best_d2
                    best[mask] = candidate[mask]
                    best_d2[mask] = d2[mask]
    return best.reshape(original_shape)


def pbc_distance(pos_a_A: np.ndarray, pos_b_A: np.ndarray,
                 box_vectors_A: Optional[np.ndarray]) -> float:
    return float(np.linalg.norm(
        minimum_image_displacement(np.asarray(pos_b_A) - np.asarray(pos_a_A),
                                   box_vectors_A)
    ))


def pbc_angle_dha(donor_A: np.ndarray, hydrogen_A: np.ndarray,
                  acceptor_A: np.ndarray,
                  box_vectors_A: Optional[np.ndarray]) -> float:
    """Return D-H...A angle in degrees using minimum-image H->D and H->A."""
    h_to_d = minimum_image_displacement(np.asarray(donor_A) - np.asarray(hydrogen_A),
                                        box_vectors_A)
    h_to_a = minimum_image_displacement(np.asarray(acceptor_A) - np.asarray(hydrogen_A),
                                        box_vectors_A)
    nd = np.linalg.norm(h_to_d)
    na = np.linalg.norm(h_to_a)
    if nd < 1e-12 or na < 1e-12:
        return 0.0
    c = float(np.dot(h_to_d, h_to_a) / (nd * na))
    return math.degrees(math.acos(np.clip(c, -1.0, 1.0)))


def image_positions_around_anchor(xyz_A: np.ndarray, atom_indices: Iterable[int],
                                  anchor_index: int,
                                  box_vectors_A: Optional[np.ndarray]) -> np.ndarray:
    """Place each requested topology atom in its nearest image to the QM anchor."""
    ids = np.asarray(list(atom_indices), dtype=int)
    if len(ids) == 0:
        return np.empty((0, 3), dtype=float)
    anchor = np.asarray(xyz_A[anchor_index], dtype=float)
    disp = np.asarray(xyz_A[ids], dtype=float) - anchor
    return anchor + minimum_image_displacement(disp, box_vectors_A)


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------
def _element_symbol(atom) -> str:
    if atom.element is None:
        return "VS"
    return atom.element.symbol


def _is_real_atom(atom) -> bool:
    return atom.element is not None and atom.element.number != 0


def _find_residue(mdtop, name: str, resnum: int):
    for res in mdtop.residues:
        if res.name == name and res.resSeq == resnum:
            return res
    return None


def _fail_or_warn(message: str, strict: bool = True):
    if strict:
        raise ValueError(message)
    print(f"WARNING: {message}")


def _neighbor_map(mdtop) -> dict[int, list[int]]:
    out = {i: [] for i in range(mdtop.n_atoms)}
    for bond in mdtop.bonds:
        a, b = bond
        out[a.index].append(b.index)
        out[b.index].append(a.index)
    return out


def _build_water_records(mdtop) -> list[WaterRecord]:
    waters = []
    for res in mdtop.residues:
        if res.name not in WATER_RESNAMES:
            continue
        all_ids = tuple(a.index for a in res.atoms)
        real_ids = tuple(a.index for a in res.atoms if _is_real_atom(a))
        oxy = [a.index for a in res.atoms
               if _is_real_atom(a) and _element_symbol(a) == "O"]
        hyd = [a.index for a in res.atoms
               if _is_real_atom(a) and _element_symbol(a) == "H"]
        if len(oxy) != 1 or len(hyd) != 2 or len(real_ids) != 3:
            raise ValueError(
                f"Water {res.name}{res.resSeq} must contain exactly O/H/H real "
                f"atoms (virtual sites allowed); got {[(_element_symbol(a), a.name) for a in res.atoms]}"
            )
        waters.append(WaterRecord(
            residue_index=res.index,
            residue_name=res.name,
            residue_seq=res.resSeq,
            oxygen_index=oxy[0],
            hydrogen_indices=tuple(hyd),
            all_atom_indices=all_ids,
            real_atom_indices=real_ids,
        ))
    return waters


def _classify_core_hbond_sites(mdtop, qm_static: Iterable[int]):
    """
    Identify donor and acceptor N/O/S atoms in the *fixed* QM core.

    Donor rule: N/O/S with a covalently bonded hydrogen that is itself in the
    fixed QM core.

    Acceptor rule (intentionally conservative):
      - O: acceptor
      - N: acceptor only when it has no bonded H
      - S: acceptor unless directly bonded to O (excludes sulfonyl sulfur;
           retains thio/ring sulfur such as AZM S2)

    This is more chemically meaningful than treating every N/O/S as both donor
    and acceptor. It can later be extended with residue-specific overrides.
    """
    qm_set = set(int(i) for i in qm_static)
    neighbors = _neighbor_map(mdtop)
    donors: dict[int, tuple[int, ...]] = {}
    acceptors: list[int] = []

    for i in sorted(qm_set):
        atom = mdtop.atom(i)
        el = _element_symbol(atom)
        if el not in POLAR_ELEMENTS:
            continue
        bonded_h = tuple(
            j for j in neighbors[i]
            if j in qm_set and _element_symbol(mdtop.atom(j)) == "H"
        )
        if bonded_h:
            donors[i] = bonded_h

        if el == "O":
            acceptors.append(i)
        elif el == "N":
            if not bonded_h:
                acceptors.append(i)
        elif el == "S":
            bonded_o = any(_element_symbol(mdtop.atom(j)) == "O" for j in neighbors[i])
            if not bonded_o:
                acceptors.append(i)

    return donors, acceptors


def _detect_free_ions(mdtop, charges: np.ndarray,
                      excluded_residue_names: set[str]) -> list[IonRecord]:
    """Detect common or charge-bearing small free ions; exclude known QM metals."""
    ions = []
    for res in mdtop.residues:
        if res.name in WATER_RESNAMES or res.name in excluded_residue_names:
            continue
        atom_ids = tuple(a.index for a in res.atoms)
        real_atoms = [a for a in res.atoms if _is_real_atom(a)]
        q = float(np.sum(charges[list(atom_ids)]))
        name_u = res.name.upper()
        named = name_u in {x.upper() for x in POSITIVE_ION_RESNAMES | NEGATIVE_ION_RESNAMES}
        # Generic catch: isolated one-atom residue with approximately integer charge.
        generic = len(real_atoms) == 1 and abs(q) >= 0.5
        if named or generic:
            ions.append(IonRecord(
                residue_index=res.index,
                residue_name=res.name,
                residue_seq=res.resSeq,
                atom_indices=atom_ids,
                net_charge=q,
            ))
    return ions


def build_qm_region_spec(mdtop, charges: np.ndarray, region: dict,
                         resnum_offset: int = DEFAULT_RESNUM_OFFSET,
                         strict: bool = True) -> QMRegionSpec:
    qm_static: list[int] = []
    boundary_pairs: list[tuple[int, int]] = []
    m2_per_cb: list[list[int]] = []

    for requested_name, resnums in region["sidechains"].items():
        for pdb_resnum in resnums:
            resnum = pdb_resnum + resnum_offset
            res = _find_residue(mdtop, requested_name, resnum)
            if res is None and requested_name == "HIS":
                for alt in HID_HIE_VARIANTS:
                    res = _find_residue(mdtop, alt, resnum)
                    if res is not None:
                        break
            if res is None:
                _fail_or_warn(
                    f"required sidechain {requested_name}{pdb_resnum} "
                    f"(topology resSeq {resnum}) not found",
                    strict,
                )
                continue

            allowed = SIDECHAIN_ATOMS.get(res.name,
                                          SIDECHAIN_ATOMS.get(requested_name, set()))
            ca_idx = None
            cb_idx = None
            m2 = []
            found_m2 = set()
            for atom in res.atoms:
                if atom.name == "CA":
                    ca_idx = atom.index
                elif atom.name == "CB":
                    cb_idx = atom.index
                    qm_static.append(atom.index)
                elif atom.name in allowed:
                    qm_static.append(atom.index)
                elif atom.name in M2_ATOMS:
                    m2.append(atom.index)
                    found_m2.add(atom.name)

            if ca_idx is None or cb_idx is None:
                _fail_or_warn(
                    f"missing CA/CB boundary for {res.name}{res.resSeq}", strict
                )
                continue
            missing = M2_ATOMS - found_m2
            if missing:
                _fail_or_warn(
                    f"missing M2 atoms {sorted(missing)} for {res.name}{res.resSeq}",
                    strict,
                )
            boundary_pairs.append((cb_idx, ca_idx))
            m2_per_cb.append(m2)

    ligand_hits = 0
    for res in mdtop.residues:
        if res.name in region.get("ligand_resnames", []):
            ligand_hits += 1
            qm_static.extend(a.index for a in res.atoms)
    if region.get("ligand_resnames") and ligand_hits == 0:
        _fail_or_warn(
            f"no ligand found among {region['ligand_resnames']}", strict
        )

    metal_indices: list[int] = []
    metal_hits = 0
    metal_names = set(region.get("metal_resnames", []))
    for res in mdtop.residues:
        if res.name in metal_names:
            metal_hits += 1
            ids = [a.index for a in res.atoms]
            qm_static.extend(ids)
            metal_indices.extend(ids)
    if metal_names and metal_hits == 0:
        _fail_or_warn(f"no metal found among {sorted(metal_names)}", strict)

    qm_static = sorted(set(qm_static))
    if not qm_static:
        raise ValueError("Fixed QM core is empty")

    donors, acceptors = _classify_core_hbond_sites(mdtop, qm_static)
    waters = _build_water_records(mdtop)
    ions = _detect_free_ions(mdtop, charges, metal_names)
    anchor = metal_indices[0] if metal_indices else qm_static[0]

    formal_charge = int(region["formal_charge"])
    # Useful sanity check, but the explicit chemical formal charge remains authoritative.
    core_ff_sum = float(np.sum(charges[qm_static]))
    if abs(core_ff_sum - formal_charge) > 0.75:
        _fail_or_warn(
            f"fixed QM-core FF partial-charge sum {core_ff_sum:+.4f} e is far "
            f"from configured formal charge {formal_charge:+d}; check region/protonation",
            strict,
        )

    return QMRegionSpec(
        qm_static=qm_static,
        boundary_pairs=boundary_pairs,
        m2_per_cb=m2_per_cb,
        metal_indices=metal_indices,
        formal_charge=formal_charge,
        water_records=waters,
        core_hbond_donors=donors,
        core_hbond_acceptors=acceptors,
        ion_records=ions,
        anchor_index=anchor,
    )


# ---------------------------------------------------------------------------
# Hydrogen-bond water selection
# ---------------------------------------------------------------------------
def _select_hbonded_waters(mdtop, spec: QMRegionSpec, xyz_A: np.ndarray,
                           box_vectors_A: Optional[np.ndarray],
                           distance_cutoff_A: float = 3.5,
                           angle_cutoff_deg: float = 140.0):
    """
    Select whole waters that directly H-bond to fixed-core N/O/S sites.

    Two directions are evaluated:
      water donor: O_w-H_w...A_core
      core donor:  D_core-H_core...O_w

    The donor-acceptor distance and D-H...A angle must both satisfy the cutoffs.
    """
    waters = spec.water_records
    if not waters:
        return [], {}
    reasons: dict[int, list[str]] = {}
    o_indices = np.asarray([w.oxygen_index for w in waters], dtype=int)
    o_positions = xyz_A[o_indices]

    # Vectorize the distance screening so robust triclinic MIC remains cheap.
    for acc in spec.core_hbond_acceptors:
        disp = minimum_image_displacement(o_positions - xyz_A[acc], box_vectors_A)
        distances = np.linalg.norm(disp, axis=1)
        for wi in np.flatnonzero(distances <= distance_cutoff_A):
            water = waters[int(wi)]
            da = float(distances[wi])
            for h in water.hydrogen_indices:
                angle = pbc_angle_dha(
                    xyz_A[water.oxygen_index], xyz_A[h], xyz_A[acc], box_vectors_A
                )
                if angle >= angle_cutoff_deg:
                    a = mdtop.atom(acc)
                    reasons.setdefault(water.residue_index, []).append(
                        f"water-donor -> {a.residue.name}{a.residue.resSeq}:{a.name} "
                        f"D-A={da:.3f}A angle={angle:.1f}deg"
                    )
                    break

    for donor, hydrogens in spec.core_hbond_donors.items():
        disp = minimum_image_displacement(o_positions - xyz_A[donor], box_vectors_A)
        distances = np.linalg.norm(disp, axis=1)
        for wi in np.flatnonzero(distances <= distance_cutoff_A):
            water = waters[int(wi)]
            da = float(distances[wi])
            for h in hydrogens:
                angle = pbc_angle_dha(
                    xyz_A[donor], xyz_A[h], xyz_A[water.oxygen_index], box_vectors_A
                )
                if angle >= angle_cutoff_deg:
                    d = mdtop.atom(donor)
                    reasons.setdefault(water.residue_index, []).append(
                        f"{d.residue.name}{d.residue.resSeq}:{d.name}-donor -> water "
                        f"D-A={da:.3f}A angle={angle:.1f}deg"
                    )
                    break

    return [w for w in waters if w.residue_index in reasons], reasons


def _select_zn_coordinated_waters(mdtop, spec: QMRegionSpec, xyz_A: np.ndarray,
                                   box_vectors_A: Optional[np.ndarray],
                                   zn_water_cutoff_A: float = DEFAULT_ZN_WATER_CUTOFF_A):
    """Select whole waters whose oxygen directly coordinates a QM Zn atom.

    The criterion is purely geometric and PBC aware: a water is selected when
    its O atom is within ``zn_water_cutoff_A`` of any configured QM metal atom.
    A non-positive cutoff disables this pathway.  The current hCAII presets
    contain one Zn atom, but the implementation supports multiple metal indices.
    """
    waters = spec.water_records
    if zn_water_cutoff_A <= 0 or not waters or not spec.metal_indices:
        return [], {}

    reasons: dict[int, list[str]] = {}
    o_indices = np.asarray([w.oxygen_index for w in waters], dtype=int)
    o_positions = xyz_A[o_indices]
    for metal_idx in spec.metal_indices:
        metal_atom = mdtop.atom(metal_idx)
        disp = minimum_image_displacement(o_positions - xyz_A[metal_idx], box_vectors_A)
        distances = np.linalg.norm(disp, axis=1)
        for wi in np.flatnonzero(distances <= zn_water_cutoff_A):
            water = waters[int(wi)]
            d = float(distances[wi])
            reasons.setdefault(water.residue_index, []).append(
                f"Zn-coordination -> {metal_atom.residue.name}{metal_atom.residue.resSeq}:"
                f"{metal_atom.name} Zn-O={d:.3f}A cutoff={zn_water_cutoff_A:.3f}A"
            )
    return [w for w in waters if w.residue_index in reasons], reasons


def _select_adaptive_waters(mdtop, spec: QMRegionSpec, xyz_A: np.ndarray,
                            box_vectors_A: Optional[np.ndarray],
                            hbond_distance_A: float = 3.5,
                            hbond_angle_deg: float = 140.0,
                            zn_water_cutoff_A: float = DEFAULT_ZN_WATER_CUTOFF_A):
    """Union of H-bonded and directly Zn-coordinated whole waters."""
    hbonded, hbond_reasons = _select_hbonded_waters(
        mdtop, spec, xyz_A, box_vectors_A, hbond_distance_A, hbond_angle_deg
    )
    zn_waters, zn_reasons = _select_zn_coordinated_waters(
        mdtop, spec, xyz_A, box_vectors_A, zn_water_cutoff_A
    )
    hbond_ids = {w.residue_index for w in hbonded}
    zn_ids = {w.residue_index for w in zn_waters}
    selected_ids = hbond_ids | zn_ids
    reasons: dict[int, list[str]] = {}
    for source in (hbond_reasons, zn_reasons):
        for resid, entries in source.items():
            reasons.setdefault(resid, []).extend(entries)
    selected = [w for w in spec.water_records if w.residue_index in selected_ids]
    return selected, reasons, hbond_ids, zn_ids


# ---------------------------------------------------------------------------
# Link atom / boundary charge treatment
# ---------------------------------------------------------------------------
def _place_link_atom(cb_pos_A: np.ndarray, ca_pos_A: np.ndarray,
                     box_vectors_A: Optional[np.ndarray],
                     bond_A: float = LINK_H_BOND_A) -> np.ndarray:
    v = minimum_image_displacement(np.asarray(ca_pos_A) - np.asarray(cb_pos_A),
                                   box_vectors_A)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Cannot place link atom: CA and CB coincide")
    return np.asarray(cb_pos_A) + bond_A * v / norm


def _shift_boundary_charges(charges: np.ndarray,
                            boundary_pairs: list[tuple[int, int]],
                            m2_per_cb: list[list[int]]) -> np.ndarray:
    """Original PartQMMM charge-shift scheme: CA -> equal shares on M2 atoms."""
    q = np.asarray(charges, dtype=float).copy()
    for (_cb_idx, ca_idx), m2 in zip(boundary_pairs, m2_per_cb):
        if not m2:
            raise ValueError(f"No M2 atoms available for CA index {ca_idx}")
        share = q[ca_idx] / len(m2)
        q[ca_idx] = 0.0
        for idx in m2:
            q[idx] += share
    return q


def _rcd_boundary_charges(charges: np.ndarray,
                          xyz_A: np.ndarray,
                          boundary_pairs: list[tuple[int, int]],
                          m2_per_cb: list[list[int]],
                          anchor_index: int,
                          box_vectors_A: Optional[np.ndarray]):
    """
    Redistributed Charge and Dipole (RCD) treatment for each CA boundary.

    For an MM1 atom (CA) with charge q and n MM2 neighbours:
      * q(CA) -> 0
      * each MM2 charge is reduced by q/n
      * a virtual charge +2q/n is placed at the CA--MM2 midpoint

    Midpoints are built from a triclinic minimum-image CA->MM2 bond vector.
    """
    q = np.asarray(charges, dtype=float).copy()
    xyz = np.asarray(xyz_A, dtype=float)
    anchor = np.asarray(xyz[anchor_index], dtype=float)
    virtual_charges: list[float] = []
    virtual_positions: list[np.ndarray] = []
    virtual_labels: list[str] = []

    for boundary_no, ((_cb_idx, ca_idx), m2) in enumerate(
            zip(boundary_pairs, m2_per_cb)):
        if not m2:
            raise ValueError(f"No M2 atoms available for CA index {ca_idx}")
        q_ca = float(q[ca_idx])
        fraction = q_ca / len(m2)
        q[ca_idx] = 0.0

        ca_imaged = anchor + minimum_image_displacement(
            xyz[ca_idx] - anchor, box_vectors_A
        )
        for m2_idx in m2:
            q[m2_idx] -= fraction
            ca_to_m2 = minimum_image_displacement(
                xyz[m2_idx] - xyz[ca_idx], box_vectors_A
            )
            virtual_positions.append(ca_imaged + 0.5 * ca_to_m2)
            virtual_charges.append(2.0 * fraction)
            virtual_labels.append(
                f"RCD_b{boundary_no}:CA{ca_idx}-M2{m2_idx}"
            )

    vpos = (
        np.asarray(virtual_positions, dtype=float)
        if virtual_positions else np.empty((0, 3), dtype=float)
    )
    return np.asarray(q), np.asarray(virtual_charges), vpos, virtual_labels


def _apply_boundary_charge_method(charges: np.ndarray,
                                  xyz_A: np.ndarray,
                                  spec: QMRegionSpec,
                                  box_vectors_A: Optional[np.ndarray],
                                  method: str):
    """Apply only the local covalent-boundary electrostatic treatment."""
    method = str(method).lower()
    if method == "shift":
        return (
            _shift_boundary_charges(charges, spec.boundary_pairs, spec.m2_per_cb),
            np.empty((0,), dtype=float),
            np.empty((0, 3), dtype=float),
            [],
        )
    if method == "rcd":
        return _rcd_boundary_charges(
            charges, xyz_A, spec.boundary_pairs, spec.m2_per_cb,
            spec.anchor_index, box_vectors_A,
        )
    raise ValueError(
        f"Unknown boundary charge method {method!r}; choose 'shift' or 'rcd'"
    )


def _make_embedding_charge_consistent(q: np.ndarray,
                                      qm_all_indices: Iterable[int],
                                      formal_qm_charge: int,
                                      m2_per_cb: list[list[int]]):
    """
    Shift the FF-fragment/formal-charge residual onto M2 atoms so that
    Q_QM(formal) + Q_MM(point charges) == Q_system(FF charges).
    """
    q = q.copy()
    qm_ids = list(qm_all_indices)
    ff_qm = float(np.sum(q[qm_ids]))
    residual = ff_qm - float(formal_qm_charge)
    all_m2 = [idx for sub in m2_per_cb for idx in sub]
    if abs(residual) > 1e-12:
        if not all_m2:
            raise ValueError(
                "QM formal charge differs from FF fragment charge but there are no "
                "M2 atoms available for the boundary correction"
            )
        correction = residual / len(all_m2)
        for idx in all_m2:
            q[idx] += correction
    return q, ff_qm, residual


# ---------------------------------------------------------------------------
# Ion guard
# ---------------------------------------------------------------------------
def _find_close_mm_ions(mdtop, spec: QMRegionSpec, xyz_A: np.ndarray,
                        box_vectors_A: Optional[np.ndarray],
                        qm_all_set: set[int], guard_distance_A: float) -> list[str]:
    if guard_distance_A <= 0:
        return []
    qm_real = [i for i in qm_all_set if _is_real_atom(mdtop.atom(i))]
    alerts = []
    for ion in spec.ion_records:
        # If a future region explicitly places an ion in QM, it must not also be MM.
        if set(ion.atom_indices).issubset(qm_all_set):
            continue
        best = float("inf")
        for ii in ion.atom_indices:
            if not _is_real_atom(mdtop.atom(ii)):
                continue
            for qi in qm_real:
                best = min(best, pbc_distance(xyz_A[ii], xyz_A[qi], box_vectors_A))
        if best <= guard_distance_A:
            sign = "+" if ion.net_charge >= 0 else "-"
            alerts.append(
                f"{ion.residue_name}{ion.residue_seq} ({ion.net_charge:{sign}.3f} e) "
                f"is {best:.3f} A from the QM region"
            )
    return alerts


# ---------------------------------------------------------------------------
# Core per-frame partition
# ---------------------------------------------------------------------------
def partition_frame(mdtop, charges_full: np.ndarray, spec: QMRegionSpec,
                    xyz_A: np.ndarray, frame_index: int,
                    box_vectors_A: Optional[np.ndarray] = None,
                    hbond_distance_A: float = 3.5,
                    hbond_angle_deg: float = 140.0,
                    zn_water_cutoff_A: float = DEFAULT_ZN_WATER_CUTOFF_A,
                    boundary_charge_method: str = "shift",
                    ion_guard_A: float = 4.0,
                    fail_on_close_ion: bool = True,
                    charge_tolerance: float = 1e-5,
                    neutral_fragment_tolerance: float = 1e-4) -> FramePartition:
    xyz_A = np.asarray(xyz_A, dtype=float)
    if xyz_A.shape != (mdtop.n_atoms, 3):
        raise ValueError(
            f"xyz shape {xyz_A.shape} does not match topology ({mdtop.n_atoms}, 3)"
        )

    selected_waters, water_reasons, hbond_water_ids, zn_water_ids = _select_adaptive_waters(
        mdtop, spec, xyz_A, box_vectors_A,
        hbond_distance_A=hbond_distance_A,
        hbond_angle_deg=hbond_angle_deg,
        zn_water_cutoff_A=zn_water_cutoff_A,
    )

    qm_all = set(spec.qm_static)
    selected_water_resids = []
    for water in selected_waters:
        q_water = float(np.sum(charges_full[list(water.all_atom_indices)]))
        if abs(q_water) > neutral_fragment_tolerance:
            raise ValueError(
                f"Adaptive water {water.residue_name}{water.residue_seq} has net "
                f"FF charge {q_water:+.8f} e; V1 requires neutral adaptive fragments "
                "so that the QM formal charge is frame invariant."
            )
        qm_all.update(water.all_atom_indices)  # removes virtual site from MM too
        selected_water_resids.append(water.residue_index)

    qm_real = sorted(i for i in qm_all if _is_real_atom(mdtop.atom(i)))
    qm_all_sorted = sorted(qm_all)

    # Guard against close free ions. V1 intentionally fails rather than silently
    # deleting a charge, because silent deletion breaks total-charge consistency.
    close_ions = _find_close_mm_ions(
        mdtop, spec, xyz_A, box_vectors_A, qm_all, ion_guard_A
    )
    if close_ions and fail_on_close_ion:
        raise ValueError(
            "Close MM ion(s) detected. V1 will not silently delete/promote ions:\n  "
            + "\n  ".join(close_ions)
        )

    # Reimage all output coordinates around the fixed QM anchor so PBC-wrapped
    # trajectories produce a locally coherent QM/MM embedding frame.
    imaged_qm = image_positions_around_anchor(
        xyz_A, qm_real, spec.anchor_index, box_vectors_A
    )
    image_by_index = {idx: pos for idx, pos in zip(qm_real, imaged_qm)}

    qm_elements = [_element_symbol(mdtop.atom(i)) for i in qm_real]
    qm_positions = [image_by_index[i] for i in qm_real]

    # Same original C-beta -> C-alpha link-H construction, now PBC aware.
    for cb_idx, ca_idx in spec.boundary_pairs:
        cb_imaged = image_by_index[cb_idx]
        ca_relative = cb_imaged + minimum_image_displacement(
            xyz_A[ca_idx] - xyz_A[cb_idx], box_vectors_A
        )
        link = _place_link_atom(cb_imaged, ca_relative, None, LINK_H_BOND_A)
        qm_elements.append("H")
        qm_positions.append(link)

    qm_positions_A = np.asarray(qm_positions, dtype=float)

    # Step 1: local covalent-boundary electrostatics. RCD virtual sites replace
    # the original CA electrostatic contribution; the CA charge is zero and must
    # not also remain active (no double counting).
    q_embed, boundary_virtual_q, boundary_virtual_pos, boundary_virtual_labels = (
        _apply_boundary_charge_method(
            charges_full, xyz_A, spec, box_vectors_A, boundary_charge_method
        )
    )

    # Step 2: independently reconcile the FF fragment charge with the chemically
    # defined integer QM charge. This bookkeeping correction remains separate
    # from the chosen local shift/RCD construction.
    q_embed, ff_qm_sum, residual = _make_embedding_charge_consistent(
        q_embed, qm_all_sorted, spec.formal_charge, spec.m2_per_cb
    )

    mm_indices = [i for i in range(mdtop.n_atoms) if i not in qm_all]
    mm_set = set(mm_indices)
    if qm_all & mm_set:
        raise AssertionError("A topology atom is present in both QM and MM regions")

    mm_real_charges = q_embed[mm_indices]
    mm_real_positions_A = image_positions_around_anchor(
        xyz_A, mm_indices, spec.anchor_index, box_vectors_A
    )
    if len(boundary_virtual_q):
        mm_charges = np.concatenate([mm_real_charges, boundary_virtual_q])
        mm_positions_A = np.vstack([mm_real_positions_A, boundary_virtual_pos])
    else:
        mm_charges = mm_real_charges
        mm_positions_A = mm_real_positions_A

    mm_site_types = ["topology"] * len(mm_indices) + [
        "rcd_virtual"
    ] * len(boundary_virtual_q)
    mm_site_labels = [f"atom:{i}" for i in mm_indices] + boundary_virtual_labels

    system_charge = float(np.sum(charges_full))
    mm_charge_sum = float(np.sum(mm_charges))
    combined = float(spec.formal_charge) + mm_charge_sum
    charge_error = combined - system_charge
    if abs(charge_error) > charge_tolerance:
        raise ValueError(
            "QM+MM charge inconsistency: "
            f"Q_QM={spec.formal_charge:+d}, Q_MM={mm_charge_sum:+.8f}, "
            f"sum={combined:+.8f}, original system={system_charge:+.8f}, "
            f"error={charge_error:+.3e}"
        )

    # Frame-invariant QM formal charge check. Water count may vary; charge may not.
    if spec.formal_charge != int(spec.formal_charge):  # defensive
        raise AssertionError("QM formal charge is not an integer")

    return FramePartition(
        frame_index=frame_index,
        qm_elements=qm_elements,
        qm_positions_A=qm_positions_A,
        mm_charges=mm_charges,
        mm_positions_A=mm_positions_A,
        qm_formal_charge=spec.formal_charge,
        system_charge=system_charge,
        mm_charge_sum=mm_charge_sum,
        charge_error=charge_error,
        qm_atom_indices=qm_real,
        qm_all_topology_indices=qm_all_sorted,
        mm_atom_indices=mm_indices,
        mm_site_types=mm_site_types,
        mm_site_labels=mm_site_labels,
        boundary_charge_method=str(boundary_charge_method).lower(),
        n_boundary_virtual_sites=len(boundary_virtual_q),
        boundary_ff_qm_sum=ff_qm_sum,
        boundary_charge_residual=residual,
        selected_water_resids=selected_water_resids,
        selected_hbond_water_resids=sorted(hbond_water_ids),
        selected_zn_coordinated_water_resids=sorted(zn_water_ids),
        selected_water_reasons=water_reasons,
        n_link_atoms=len(spec.boundary_pairs),
        close_ions=close_ions,
        box_vectors_A=None if box_vectors_A is None else np.asarray(box_vectors_A),
    )


# ---------------------------------------------------------------------------
# Trajectory iteration
# ---------------------------------------------------------------------------
def _parse_frames(spec: str) -> tuple[int, Optional[int], int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--frames must be start:stop:stride, got {spec!r}")
    start = int(parts[0]) if parts[0] else 0
    stop = None if parts[1] in ("", "None") else int(parts[1])
    stride = int(parts[2]) if parts[2] else 1
    if start < 0 or stride <= 0:
        raise ValueError("frame start must be >=0 and stride must be >0")
    return start, stop, stride


def iter_frame_partitions(top_path: str | Path, traj_path: str | Path,
                          region_name: str = "minimal",
                          resnum_offset: int = DEFAULT_RESNUM_OFFSET,
                          frames: str = "0:None:1",
                          hbond_distance_A: float = 3.5,
                          hbond_angle_deg: float = 140.0,
                          zn_water_cutoff_A: float = DEFAULT_ZN_WATER_CUTOFF_A,
                          boundary_charge_method: str = "shift",
                          ion_guard_A: float = 4.0,
                          fail_on_close_ion: bool = True,
                          strict: bool = True):
    if region_name not in QM_REGIONS:
        raise ValueError(f"Unknown region {region_name!r}; choices={list(QM_REGIONS)}")

    mdtop, charges = load_topology_and_charges(top_path)
    spec = build_qm_region_spec(
        mdtop, charges, QM_REGIONS[region_name],
        resnum_offset=resnum_offset, strict=strict,
    )
    start, stop, stride = _parse_frames(frames)
    traj_path = str(traj_path)
    lower = traj_path.lower()

    single = lower.endswith((".rst7", ".rst", ".inpcrd", ".ncrst", ".pdb"))
    if single:
        if start > 0 or (stop is not None and stop <= 0):
            return
        if lower.endswith(".pdb"):
            traj = md.load_pdb(traj_path)
        else:
            traj = md.load(traj_path, top=mdtop)
        xyz_A = traj.xyz[0] * 10.0
        box = None
        if traj.unitcell_vectors is not None:
            box = traj.unitcell_vectors[0] * 10.0
        yield partition_frame(
            mdtop, charges, spec, xyz_A, 0, box,
            hbond_distance_A=hbond_distance_A,
            hbond_angle_deg=hbond_angle_deg,
            zn_water_cutoff_A=zn_water_cutoff_A,
            boundary_charge_method=boundary_charge_method,
            ion_guard_A=ion_guard_A,
            fail_on_close_ion=fail_on_close_ion,
        )
        return

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
            yield partition_frame(
                mdtop, charges, spec, xyz_A, global_idx, box,
                hbond_distance_A=hbond_distance_A,
                hbond_angle_deg=hbond_angle_deg,
                zn_water_cutoff_A=zn_water_cutoff_A,
                boundary_charge_method=boundary_charge_method,
                ion_guard_A=ion_guard_A,
                fail_on_close_ion=fail_on_close_ion,
            )


def summarize_partition(fp: FramePartition) -> dict:
    return {
        "frame": fp.frame_index,
        "n_qm_xyz_atoms": len(fp.qm_elements),
        "n_qm_real_topology_atoms": len(fp.qm_atom_indices),
        "n_qm_all_topology_atoms": len(fp.qm_all_topology_indices),
        "n_link_atoms": fp.n_link_atoms,
        "n_selected_waters": len(fp.selected_water_resids),
        "n_hbond_selected_waters": len(fp.selected_hbond_water_resids),
        "n_zn_coordinated_selected_waters": len(fp.selected_zn_coordinated_water_resids),
        "selected_water_resids": fp.selected_water_resids,
        "selected_hbond_water_resids": fp.selected_hbond_water_resids,
        "selected_zn_coordinated_water_resids": fp.selected_zn_coordinated_water_resids,
        "n_mm_point_charges": len(fp.mm_charges),
        "n_mm_topology_sites": len(fp.mm_atom_indices),
        "boundary_charge_method": fp.boundary_charge_method,
        "n_boundary_virtual_sites": fp.n_boundary_virtual_sites,
        "raw_qm_ff_charge_sum": fp.boundary_ff_qm_sum,
        "formal_charge_residual_to_m2": fp.boundary_charge_residual,
        "qm_formal_charge": fp.qm_formal_charge,
        "mm_charge_sum": fp.mm_charge_sum,
        "system_charge": fp.system_charge,
        "qm_plus_mm_charge": fp.qm_formal_charge + fp.mm_charge_sum,
        "charge_error": fp.charge_error,
        "close_ions": fp.close_ions,
    }
