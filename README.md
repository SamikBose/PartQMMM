# PartQMMM
## QM/MM Subsystem Partitioning & QM-Engine Input Generation

Tools to partition an MD trajectory into QM and MM subsystems (link-atom
boundary scheme) and generate **electrostatic-embedding** QM/MM single-point
inputs. Designed for generating reference data to train environment-aware
machine-learning interatomic potentials (e.g. AIMNet2-MLMM).

## Design

Two layers, so new QM engines are easy to add:

```
qmmm_partition.py   <- engine-agnostic library (the helper)
                       trajectory -> per-frame FramePartition
                       (QM elements+coords, MM charges+coords, link atoms)

orca_inputs.py      <- ORCA driver (imports qmmm_partition)
                       FramePartition -> frame_*.inp + frame_*.pc

(future) psi4_inputs.py, qchem_inputs.py  <- same backend, different writer
```

The partition layer does the science (region selection, link-atom placement,
boundary-charge redistribution); the driver layer does the file format.

## Boundary scheme (link-atom method)

- QM/MM covalent cuts are made at **Cα-Cβ** bonds for protein sidechains.
- A **link H** caps the QM side, placed along Cβ→Cα at 1.09 Å from Cβ.
- The MM boundary atom (Cα) charge is **zeroed and redistributed** equally over
  its backbone neighbors (N, C, HA) — Walker-Crowley-Case
  (*J. Comput. Chem.* 2008, 29, 1019). Keeps total charge integral and avoids
  over-polarizing the QM density with a fractional charge on the cut bond.
- This has been previously implemented and benchmarked as charged smearing scheme in
  (*J. Phys. Chem. B* 2016, 120, 4410−4420) by Bose and co-authors.

## QM regions (for hCA II + acetazolamide)

- `minimal` (~36 heavy / ~65 all-atom): Zn + His94/96/119 + Thr199 sidechains
  + AZM + 1 active-site water.
- `larger` (~50 heavy / ~85 all-atom): adds His64, Glu106, and 3 waters.

**QM-region net charge** (minimal, Zn in QM): Zn(+2) + AZM(−1) + neutral
His/Thr/water = **+1**. Pass `--charge 1` to ORCA. Recompute if you change the
region.

## Charge integrality at the boundary (important)

Cutting a sidechain at Cα-Cβ splits the residue's charge group, so the QM
fragment does NOT carry an integer partial charge in an additive force field
(e.g. for hCA II minimal the four cut sidechains leave a ~+0.10 e residual). A
QM/MM single point, however, declares an INTEGER formal charge for the QM
region. If the MM field isn't made consistent, the combined QM+MM system carries
a spurious non-integer charge sitting at the boundary, which systematically
polarizes the QM density in every frame — a bias the trained MLIP would learn.

By default the partition **enforces QM integer charge**: it rounds the QM
fragment's real partial-charge sum to the nearest integer (the formal charge the
QM engine uses) and spreads the residual over the MM boundary (M2) atoms so the
MM field is consistent with that integer. Total charge is preserved. Disable
with `enforce_integer_qm_charge=False` in the library if you want the raw
complement. The detected integer is reported as `qm_formal_charge`; the ORCA
driver cross-checks it against `--charge` and warns on mismatch.

## Usage

Dry-run one frame first (prints diagnostics, writes one input pair):

```bash
python orca_inputs.py --top system_1264.parm7 --traj prod.dcd \
    --region minimal --output-dir orca_min --frames 0:None:10 \
    --method "wB97M-V" --basis "def2-TZVP" --charge 1 --mult 1 \
    --engrad --dry-run
```

Full run (every 10th frame, with gradients for force training):

```bash
python orca_inputs.py --top system_1264.parm7 --traj prod.dcd \
    --region minimal --output-dir orca_min --frames 0:None:10 \
    --method "wB97M-V" --basis "def2-TZVP" --charge 1 --mult 1 \
    --engrad --nprocs 8 --maxcore 3000
```

### Key options

- `--resnum-offset` : PDB→topology residue-number offset. Default **−4**
  (tleap/CHARMM-GUI AMBER renumbering with 3HS4 residues 1–3 unresolved). Set 0
  if your topology preserves PDB numbering. [TODO: get rid of this dependency from the partition file]
- `--mm-cutoff` : keep **whole MM residues/molecules** when at least one of
  their MM atoms lies within this Å of any QM atom (0 = all). This avoids the
  old atom-wise cutoff artifact where half-waters or partial charged residues
  produced nonphysical fractional point-charge fragments. The retained shell can
  still have a nonzero integer net charge; the dry-run diagnostics report both
  retained-shell and full-field MM charge sums.
- `--engrad` : request ORCA gradients (needed if the MLIP trains on forces).
- `--frames start:stop:stride`.

## Outputs (per frame)

- `frame_{idx}.inp` — ORCA input; QM coords inline, MM charges referenced via
  `%pointcharges` (electrostatic embedding: QM density polarizes in the MM
  field).
- `frame_{idx}.pc` — ORCA point-charge file (count line, then `q x y z`).
[TODO: Better numbering based on run,frame idx from MD]

## Dependencies

`mdtraj`, `parmed`, `numpy` (and `scipy` only if `--mm-cutoff` is used). No
OpenMM dependency (topology read via mdtraj/parmed). 

[Right now works with amber ff because of the better metal ffs but can be extended 
easily to charmm]

## Notes 
- Virtual sites (OPC `EPW`, atomic number 0) are excluded from the QM region
  (QM engines reject them); the whole water is still removed from the MM field
  when selected into QM.
- Missing required QM sidechains, ligand, metal, or boundary atoms now fail
  fast by default. Use `--allow-missing-qm` only for exploratory debugging.
- DCD/multi-frame `--frames start:stop:stride` uses physical trajectory frame
  indices. For example, `2:4:1` processes actual frames 2 and 3, not frames 0
  and 1 relabeled as 2 and 3.
- Single-frame restart files (`.rst7`/`.inpcrd`/`.ncrst`) and multi-frame
  trajectories are both handled.
- NPT trajectories: box vectors are read per frame (carried on FramePartition)
  in case a future writer needs periodic info.
