# PartQMMM Ensemble Analysis

This folder characterizes the MD ensemble using the **same root `qmmm_partition.py` used to generate QM/MM partitions**.

The current adaptive-water definition is:

```text
QM water = H-bonded to fixed-core N/O/S
           OR directly Zn coordinated (default Zn-O <= 2.6 A)
```

All H-bond and Zn--water distance tests use the same triclinic minimum-image implementation as production partitioning.

## What is measured

For every analyzed frame:

- total adaptive-QM water count and identities;
- H-bond-selected water count and identities;
- Zn-coordinated water count and identities;
- waters satisfying both criteria;
- water additions/removals and membership similarity;
- structured water--QM-core H-bond contacts;
- structured Zn--water contacts and Zn--O distances;
- nearest Zn-water O distance and whether that nearest water is QM;
- ion proximity;
- nearest physical/MM embedding charges;
- MM point-charge populations inside configurable shells;
- electrostatic potential and field at every QM atom and link H;
- field specifically at Zn (`E_Zn_V_per_A`);
- real-QM-only and link-inclusive maximum fields;
- high-field outliers;
- QM/MM/system charge consistency.

The electrostatic calculation uses the exact finite embedding produced after adaptive-water removal and the selected `shift` or `rcd` boundary treatment.

## Dependencies

From the repository root:

```bash
pip install -r analysis/requirements.txt
```

## Five-run configuration

Example:

```json
{
  "top": "/path/system_1264.parm7",
  "runs": [
    {"name": "run1", "traj": "/path/hca2_azm_withwater_run1_combined.dcd"},
    {"name": "run2", "traj": "/path/hca2_azm_withwater_run2_combined.dcd"},
    {"name": "run3", "traj": "/path/hca2_azm_withwater_run3_combined.dcd"},
    {"name": "run4", "traj": "/path/hca2_azm_withwater_run4_combined.dcd"},
    {"name": "run5", "traj": "/path/hca2_azm_withwater_run5_combined.dcd"}
  ]
}
```

## Recommended redo after adding Zn-bound waters

Stride-10 RCD survey:

```bash
python analysis/analyze_ensemble.py \
  --config analysis/five_runs.json \
  --output-dir ensemble_analysis_rcd_znwater_stride10 \
  --region larger \
  --boundary-charge-method rcd \
  --zn-water-cutoff 2.6 \
  --frames 0:None:10 \
  --outlier-percentile 99 \
  --progress-every 100
```

Full-frame analysis uses `--frames 0:None:1`.

To reproduce the old H-bond-only definition for a controlled comparison:

```bash
--zn-water-cutoff 0
```

## Main outputs

For each region/method directory:

- `frame_metrics.csv`;
- `water_membership.csv`;
- `water_hbond_contacts.csv`;
- `water_zn_contacts.csv`;
- `qm_atom_fields.csv`;
- `water_occupancy.csv`;
- `partner_occupancy.csv`;
- `water_residence_events.csv`;
- `run_summary.csv`;
- `field_outliers.csv`;
- `analysis_report.md`;
- `analysis_metadata.json`;
- `plots/`.

### Important new frame metrics

```text
n_adaptive_qm_waters
adaptive_qm_water_residue_indices
n_hbonded_waters
hbonded_water_residue_indices
n_zn_coordinated_waters
zn_coordinated_water_residue_indices
n_waters_selected_by_both
nearest_zn_water_A
nearest_zn_water_is_qm
nearest_zn_water_is_zn_coordinated
E_Zn_V_per_A
potential_Zn_V
nearest_mm_charge_to_Zn_A
Emax_real_qm_V_per_A
```

## New Zn-focused plots

In addition to the existing field and water figures, the plotting script generates:

- `adaptive_water_count_hist.png`;
- `zn_coordinated_water_count_hist.png`;
- `Emax_real_qm_hist.png`;
- `Zn_field_vs_nearest_zn_water.png`;
- `runX_Zn_water_distance_timeseries.png`;
- `runX_Zn_field_timeseries.png`.

The Zn-water distance plots mark the configured cutoff.

## Shift vs RCD

Analyze identical frames with both methods, then:

```bash
python analysis/compare_boundary_methods.py \
  shift/frame_metrics.csv \
  rcd/frame_metrics.csv \
  --output boundary_comparison.csv
```

The comparison now includes field/potential diagnostics specifically at Zn.

## Tests

```bash
python analysis/test_analysis.py \
  --parm7 /path/system_1264.parm7 \
  --pdb /path/hca2_azm_withwater_run5_combined.pdb
```

The real-PDB smoke test checks the original H-bond selection and charge conservation; a synthetic 2.1 A Zn--water geometry verifies the new coordination pathway.

## Emax atom identity and site-specific field diagnostics

The current analysis version explicitly reports **which QM site carries the
frame-level `Emax`**.  `frame_metrics.csv` now includes fields such as:

```text
Emax_V_per_A
Emax_atom_label
Emax_is_link_atom
Emax_link_index
Emax_topology_atom_index
Emax_topology_atom_id_1based
Emax_resname
Emax_pdb_resnum
Emax_atom_name

Emax_real_qm_V_per_A
Emax_real_atom_label
Emax_real_topology_atom_index
Emax_real_topology_atom_id_1based
```

Synthetic C-beta link hydrogens are explicitly labeled `LINK_H...` and have no
topology atom id.  During a run, progress lines now look like:

```text
analyzed 500 frames; ... Emax=2.61 V/A at LINK_H3:... [LINK_H3; synthetic QM link atom]
```

or for a physical QM atom:

```text
Emax=2.61 V/A at HIE94:HE2 [topology_index=..., atom_id_1based=...]
```

At the end of each trajectory, the terminal also reports the fraction of frames
whose Emax is on a link H, the most frequent Emax-carrying sites, and compact
site-field statistics.

A new `site_specific_fields.csv` provides a smaller, chemically focused table
for time-series work.  It contains:

- Zn;
- QM histidine ring nitrogens (`ND1`, `NE2`);
- QM glutamate carboxylate oxygens (`OE1`, `OE2`);
- oxygen atoms of every adaptive QM water.

The full `qm_atom_fields.csv` still contains **all** real QM atoms plus link H
sites.  Therefore no atom-level information is lost.

### New field plots

In `plots/` the suite additionally generates:

```text
Emax_atom_frequency.png
Emax_link_vs_real.png
runX_Emax_all_vs_real_timeseries.png
runX_Emax_atom_frequency.png
runX_Zn_site_field_timeseries.png
runX_HIS_N_site_fields_timeseries.png
runX_GLU_O_site_fields_timeseries.png
runX_QM_water_O_site_fields_timeseries.png
runX_QM_water_O_field_mean_max_timeseries.png
runX_QM_water_O_top10_site_fields_timeseries.png
```

For adaptive water plots, every point in
`runX_QM_water_O_site_fields_timeseries.png` is the MM field at one **QM water
oxygen** in one sampled frame.  The companion CSV retains the exact water
residue identity.
