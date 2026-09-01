#!/usr/bin/env python3
"""Analyze PartQMMM structural boundaries and MM electrostatic embedding over MD runs.

Outputs per region/boundary method:
  frame_metrics.csv          one row per analyzed MD frame
  water_membership.csv       selected adaptive-QM water identity per frame
  water_hbond_contacts.csv   structured water<->QM-core H-bond contacts
  water_zn_contacts.csv      structured direct Zn--water coordination contacts
  qm_atom_fields.csv         electrostatic potential/field at every QM + link site
  site_specific_fields.csv   Zn, His-N, Glu-O, and adaptive-QM-water-O fields
  water_occupancy.csv        per-water occupancies by run and combined
  partner_occupancy.csv      AZM/Thr/His/etc H-bond occupancies
  water_residence_events.csv continuous membership episodes in sampled frames
  run_summary.csv            compact per-run statistics
  field_outliers.csv         high-field frames with robust statistics
  analysis_report.md         human-readable summary
  analysis_metadata.json     settings and topology charge metadata

The suite uses qmmm_partition.py directly, ensuring adaptive-water membership
(H-bond OR direct Zn coordination), PBC handling, link atoms, charge consistency, and shift/RCD boundary fields are
identical to the label-generation workflow.
"""
from __future__ import annotations

import argparse
import json
import math
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import qmmm_partition as qp
from analysis_core import (
    iter_trajectory_frames,
    structured_hbond_contacts,
    zn_coordinated_water_contacts,
    nearest_zn_water,
    ion_proximity_metrics,
    embedding_field_metrics,
    site_specific_field_rows,
    water_contact_category_counts,
)


def parse_shells(text: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not vals or any(x <= 0 for x in vals):
        raise argparse.ArgumentTypeError("shell radii must be positive comma-separated Angstrom values")
    return vals


def load_run_config(args) -> tuple[str, list[dict]]:
    top = args.top
    runs = []
    if args.config:
        cfg_path = Path(args.config)
        cfg = json.loads(cfg_path.read_text())
        if top is None:
            cfg_top = cfg.get("top")
            if cfg_top:
                cfg_top_path = Path(cfg_top)
                if not cfg_top_path.is_absolute():
                    cfg_top_path = (cfg_path.parent / cfg_top_path).resolve()
                top = str(cfg_top_path)
        for item in cfg.get("runs", []):
            p = Path(item["traj"])
            if not p.is_absolute():
                p = (cfg_path.parent / p).resolve()
            runs.append({"name": item.get("name", p.stem), "traj": str(p)})
    if args.traj:
        names = args.run_name or []
        if names and len(names) != len(args.traj):
            raise ValueError("--run-name must be supplied exactly once per --traj, or omitted")
        for i, traj in enumerate(args.traj):
            runs.append({"name": names[i] if names else Path(traj).stem, "traj": traj})
    if not top:
        raise ValueError("topology required via --top or config JSON")
    if not runs:
        raise ValueError("provide trajectories using --traj or --config")
    names = [r["name"] for r in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"run names must be unique; got {names}")
    return str(top), runs


def make_residence_events(membership_by_run: dict[str, list[tuple[int, set[int]]]]) -> list[dict]:
    rows = []
    for run, timeline in membership_by_run.items():
        active: dict[int, dict] = {}
        for sample_i, (frame, waters) in enumerate(timeline):
            # Close waters that disappeared at this sampled transition.
            for wid in list(active):
                if wid not in waters:
                    ev = active.pop(wid)
                    ev["end_frame"] = timeline[sample_i - 1][0]
                    ev["n_sampled_frames"] = sample_i - ev.pop("start_sample_i")
                    rows.append(ev)
            # Open newly present waters.
            for wid in waters:
                if wid not in active:
                    active[wid] = {
                        "run": run,
                        "water_residue_index": wid,
                        "start_frame": frame,
                        "start_sample_i": sample_i,
                    }
        if timeline:
            last_frame = timeline[-1][0]
            n_samples = len(timeline)
            for wid, ev in active.items():
                ev["end_frame"] = last_frame
                ev["n_sampled_frames"] = n_samples - ev.pop("start_sample_i")
                rows.append(ev)
    return rows


def water_occupancy_table(frame_df: pd.DataFrame, membership_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, fsub in frame_df.groupby("run", sort=False):
        nframes = len(fsub)
        msub = membership_df[membership_df["run"] == run]
        for wid, g in msub.groupby("water_residue_index"):
            rows.append({
                "scope": run,
                "water_residue_index": int(wid),
                "water_resseq": int(g["water_resseq"].iloc[0]),
                "frames_selected": int(g["frame"].nunique()),
                "frames_total": int(nframes),
                "occupancy_fraction": float(g["frame"].nunique() / nframes),
            })
    nframes = len(frame_df)
    for wid, g in membership_df.groupby("water_residue_index"):
        rows.append({
            "scope": "ALL_RUNS",
            "water_residue_index": int(wid),
            "water_resseq": int(g["water_resseq"].iloc[0]),
            "frames_selected": int(len(g[["run", "frame"]].drop_duplicates())),
            "frames_total": int(nframes),
            "occupancy_fraction": float(len(g[["run", "frame"]].drop_duplicates()) / nframes),
        })
    return pd.DataFrame(rows).sort_values(["scope", "occupancy_fraction"], ascending=[True, False]) if rows else pd.DataFrame()


def partner_occupancy_table(frame_df: pd.DataFrame, contact_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [(run, frame_df[frame_df.run == run]) for run in frame_df.run.unique()]
    scopes.append(("ALL_RUNS", frame_df))
    for scope, fsub in scopes:
        csub = contact_df if scope == "ALL_RUNS" else contact_df[contact_df.run == scope]
        nframes = len(fsub)
        for group, g in csub.groupby("partner_group"):
            frame_water = g[["run", "frame", "water_residue_index"]].drop_duplicates()
            contact_frames = frame_water[["run", "frame"]].drop_duplicates()
            waters_per_frame = frame_water.groupby(["run", "frame"]).size()
            rows.append({
                "scope": scope,
                "partner_group": group,
                "frames_with_hbonded_water": int(len(contact_frames)),
                "frames_total": int(nframes),
                "occupancy_fraction": float(len(contact_frames) / nframes) if nframes else math.nan,
                "mean_contact_waters_when_present": float(waters_per_frame.mean()) if len(waters_per_frame) else 0.0,
            })
    return pd.DataFrame(rows).sort_values(["scope", "occupancy_fraction"], ascending=[True, False]) if rows else pd.DataFrame()


def summarize_runs(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, g in frame_df.groupby("run", sort=False):
        row = {
            "run": run,
            "n_frames": len(g),
            "mean_adaptive_qm_waters": float(g.n_adaptive_qm_waters.mean()),
            "max_adaptive_qm_waters": int(g.n_adaptive_qm_waters.max()),
            "mean_hbonded_waters": float(g.n_hbonded_waters.mean()),
            "max_hbonded_waters": int(g.n_hbonded_waters.max()),
            "mean_zn_coordinated_waters": float(g.n_zn_coordinated_waters.mean()),
            "fraction_frames_with_zn_coordinated_water": float((g.n_zn_coordinated_waters > 0).mean()),
            "fraction_frames_water_membership_changed": float((g.n_water_added + g.n_water_removed > 0).mean()),
            "mean_Emax_V_per_A": float(g.Emax_V_per_A.mean()),
            "p95_Emax_V_per_A": float(g.Emax_V_per_A.quantile(0.95)),
            "p99_Emax_V_per_A": float(g.Emax_V_per_A.quantile(0.99)),
            "max_Emax_V_per_A": float(g.Emax_V_per_A.max()),
            "mean_Emax_real_qm_V_per_A": float(g.Emax_real_qm_V_per_A.mean()),
            "max_Emax_real_qm_V_per_A": float(g.Emax_real_qm_V_per_A.max()),
            "fraction_frames_Emax_on_link_atom": float(g.Emax_is_link_atom.astype(bool).mean()) if "Emax_is_link_atom" in g else math.nan,
            "most_frequent_Emax_atom": str(g.Emax_atom_label.value_counts().index[0]) if "Emax_atom_label" in g and len(g.Emax_atom_label.dropna()) else "",
            "min_nearest_mm_charge_to_qm_A": float(g.nearest_mm_charge_to_qm_A.min()),
            "median_nearest_zn_water_A": float(g.nearest_zn_water_A.median()),
            "min_nearest_free_ion_A": float(g.nearest_free_ion_A.min()) if g.nearest_free_ion_A.notna().any() else math.nan,
            "max_abs_charge_error_e": float(g.charge_error_e.abs().max()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def format_emax_identity(emb: dict) -> str:
    """Compact human-readable identity of the site carrying Emax."""
    label = str(emb.get("Emax_atom_label") or "UNKNOWN")
    if bool(emb.get("Emax_is_link_atom", False)):
        li = emb.get("Emax_link_index")
        return f"{label} [LINK_H{li if li is not None else ''}; synthetic QM link atom]"
    idx = emb.get("Emax_topology_atom_index")
    aid = emb.get("Emax_topology_atom_id_1based")
    bits = []
    if idx is not None and not (isinstance(idx, float) and math.isnan(idx)):
        bits.append(f"topology_index={int(idx)}")
    if aid is not None and not (isinstance(aid, float) and math.isnan(aid)):
        bits.append(f"atom_id_1based={int(aid)}")
    suffix = f" [{', '.join(bits)}]" if bits else ""
    return f"{label}{suffix}"


def print_run_field_summary(run_name: str, run_frames: list[dict], run_sites: list[dict]):
    """Print a concise end-of-run field summary to the terminal."""
    if not run_frames:
        return
    f = pd.DataFrame(run_frames)
    print(f"  field summary for {run_name}:")
    if "Emax_is_link_atom" in f:
        frac_link = 100.0 * f.Emax_is_link_atom.astype(bool).mean()
        print(f"    Emax occurs on a synthetic link H in {frac_link:.1f}% of analyzed frames")
    if "Emax_atom_label" in f:
        top = f.Emax_atom_label.value_counts().head(5)
        if len(top):
            print("    most frequent Emax sites:")
            for label, count in top.items():
                tag = " [LINK]" if str(label).startswith("LINK_H") else ""
                print(f"      {label}{tag}: {int(count)} frames ({100.0*count/len(f):.1f}%)")
    if run_sites:
        s = pd.DataFrame(run_sites)
        for cls, label in (("ZN", "Zn"), ("HIS_N", "His N"), ("GLU_O", "Glu O"), ("QM_WATER_O", "QM-water O")):
            g = s[s.site_class == cls]
            if len(g):
                print(
                    f"    {label} field: mean={g.field_magnitude_V_per_A.mean():.4f} V/A, "
                    f"max={g.field_magnitude_V_per_A.max():.4f} V/A, n={len(g)}"
                )


def field_outliers(frame_df: pd.DataFrame, percentile: float) -> pd.DataFrame:
    x = frame_df["Emax_V_per_A"].to_numpy(float)
    threshold = float(np.nanpercentile(x, percentile))
    median = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - median)))
    robust_z = np.zeros(len(x), dtype=float)
    if mad > 0:
        robust_z = 0.67448975 * (x - median) / mad
    out = frame_df.copy()
    out["Emax_percentile_threshold_V_per_A"] = threshold
    out["Emax_robust_z"] = robust_z
    out["is_high_field_outlier"] = out["Emax_V_per_A"] >= threshold
    return out[out.is_high_field_outlier].sort_values("Emax_V_per_A", ascending=False)


def write_report(path: Path, frame_df: pd.DataFrame, run_summary: pd.DataFrame,
                 water_occ: pd.DataFrame, partner_occ: pd.DataFrame,
                 outliers: pd.DataFrame, region: str, method: str, percentile: float):
    lines = [
        f"# PartQMMM ensemble analysis: {region} / {method}", "",
        f"Analyzed **{len(frame_df)} frames** across **{frame_df.run.nunique()} runs**.", "",
        "## Global diagnostics", "",
        f"- Total adaptive QM waters/frame (H-bond OR Zn coordination): mean {frame_df.n_adaptive_qm_waters.mean():.3f}, range {int(frame_df.n_adaptive_qm_waters.min())}-{int(frame_df.n_adaptive_qm_waters.max())}.",
        f"- H-bond-selected waters/frame: mean {frame_df.n_hbonded_waters.mean():.3f}.",
        f"- Zn-coordinated waters/frame (Zn-O <= {frame_df.zn_water_cutoff_A.iloc[0]:.3f} A): mean {frame_df.n_zn_coordinated_waters.mean():.3f}; present in {100*(frame_df.n_zn_coordinated_waters>0).mean():.2f}% of frames.",
        f"- Frames with a water-membership change relative to the prior sampled frame: {100*(frame_df.n_water_added.add(frame_df.n_water_removed)>0).mean():.2f}%.",
        f"- Emax: median {frame_df.Emax_V_per_A.median():.4f} V/A; p95 {frame_df.Emax_V_per_A.quantile(.95):.4f}; p99 {frame_df.Emax_V_per_A.quantile(.99):.4f}; max {frame_df.Emax_V_per_A.max():.4f}.",
        f"- Emax on a synthetic link H: {100*frame_df.Emax_is_link_atom.astype(bool).mean():.2f}% of frames." if "Emax_is_link_atom" in frame_df else "",
        f"- Most frequent Emax site: {frame_df.Emax_atom_label.value_counts().index[0]} ({int(frame_df.Emax_atom_label.value_counts().iloc[0])} frames)." if "Emax_atom_label" in frame_df and len(frame_df.Emax_atom_label.dropna()) else "",
        f"- Closest nonzero MM embedding charge to QM over all frames: {frame_df.nearest_mm_charge_to_qm_A.min():.3f} A.",
        f"- High-field outlier definition: top {100-percentile:.2f}% (>= {frame_df.Emax_V_per_A.quantile(percentile/100):.4f} V/A); {len(outliers)} frames flagged.",
        "",
        "## Run summary", "",
        run_summary.to_markdown(index=False) if len(run_summary) else "No run summary.",
        "",
        "## Highest-occupancy adaptive waters", "",
    ]
    if len(water_occ):
        allw = water_occ[water_occ.scope == "ALL_RUNS"].head(15)
        lines.append(allw.to_markdown(index=False))
    else:
        lines.append("No adaptive waters were selected.")
    lines += ["", "## Core-site water H-bond occupancy", ""]
    if len(partner_occ):
        lines.append(partner_occ[partner_occ.scope == "ALL_RUNS"].to_markdown(index=False))
    else:
        lines.append("No water-core H bonds were detected.")
    lines += [
        "", "## Interpretation note", "",
        "Electric potential/field values are computed from the exact finite point-charge embedding that PartQMMM generates after adaptive-water removal and the selected boundary-charge method. They are therefore diagnostics of the field the QM calculation actually sees; they are not PME/Ewald electrostatics of the parent MD simulation.", "",
    ]
    path.write_text("\n".join(lines))


def analyze_one_region(top_path: str, runs: list[dict], outdir: Path, args, region_name: str):
    mdtop, charges = qp.load_topology_and_charges(top_path)
    spec = qp.build_qm_region_spec(
        mdtop, charges, qp.QM_REGIONS[region_name],
        resnum_offset=args.resnum_offset, strict=True,
    )
    region_dir = outdir / f"{region_name}_{args.boundary_charge_method}"
    region_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    membership_rows = []
    contact_rows = []
    zn_contact_rows = []
    site_specific_rows = []
    membership_by_run: dict[str, list[tuple[int, set[int]]]] = defaultdict(list)
    field_path = region_dir / "qm_atom_fields.csv"
    field_fh = None
    field_writer = None
    if not args.skip_per_atom_fields:
        field_fh = field_path.open("w", newline="")

    expected_charge = spec.formal_charge
    for run in runs:
        run_name = run["name"]
        prev_waters: set[int] | None = None
        print(f"[{region_name}/{args.boundary_charge_method}] {run_name}: {run['traj']}")
        n_run = 0
        run_frame_start = len(frame_rows)
        run_site_start = len(site_specific_rows)
        for raw in iter_trajectory_frames(run["traj"], mdtop, args.frames):
            fp = qp.partition_frame(
                mdtop, charges, spec, raw.xyz_A, raw.frame_index,
                box_vectors_A=raw.box_vectors_A,
                hbond_distance_A=args.hbond_distance,
                hbond_angle_deg=args.hbond_angle,
                zn_water_cutoff_A=args.zn_water_cutoff,
                boundary_charge_method=args.boundary_charge_method,
                ion_guard_A=args.ion_guard,
                fail_on_close_ion=False,
            )
            if fp.qm_formal_charge != expected_charge:
                raise RuntimeError(
                    f"QM formal charge changed at {run_name} frame {raw.frame_index}: "
                    f"{fp.qm_formal_charge:+d} != expected {expected_charge:+d}"
                )

            contacts = structured_hbond_contacts(
                mdtop, spec, raw.xyz_A, raw.box_vectors_A, args.resnum_offset,
                args.hbond_distance, args.hbond_angle,
            )
            zn_contacts = zn_coordinated_water_contacts(
                mdtop, spec, raw.xyz_A, raw.box_vectors_A, args.zn_water_cutoff
            )
            selected = set(fp.selected_water_resids)
            hbond_selected = {int(c["water_residue_index"]) for c in contacts}
            zn_selected = {int(c["water_residue_index"]) for c in zn_contacts}
            expected_selected = hbond_selected | zn_selected
            if selected != expected_selected:
                raise AssertionError(
                    f"analysis adaptive-water contacts disagree with partitioner at {run_name} "
                    f"frame {raw.frame_index}: partition={sorted(selected)} "
                    f"hbond={sorted(hbond_selected)} zn={sorted(zn_selected)}"
                )
            if hbond_selected != set(fp.selected_hbond_water_resids):
                raise AssertionError("partitioner H-bond water bookkeeping disagrees with analysis")
            if zn_selected != set(fp.selected_zn_coordinated_water_resids):
                raise AssertionError("partitioner Zn-water bookkeeping disagrees with analysis")

            zn = nearest_zn_water(spec, raw.xyz_A, raw.box_vectors_A)
            nearest_zn_id = zn.get("nearest_zn_water_residue_index")
            zn["nearest_zn_water_is_qm"] = bool(nearest_zn_id in selected) if nearest_zn_id is not None else False
            zn["nearest_zn_water_is_zn_coordinated"] = bool(nearest_zn_id in zn_selected) if nearest_zn_id is not None else False
            zn["nearest_zn_water_is_hbonded"] = bool(nearest_zn_id in hbond_selected) if nearest_zn_id is not None else False
            ions = ion_proximity_metrics(
                mdtop, spec, raw.xyz_A, raw.box_vectors_A, fp.qm_atom_indices
            )
            emb, per_qm = embedding_field_metrics(
                mdtop, spec, fp, args.resnum_offset,
                shell_radii_A=args.shell_radii,
                charge_threshold_e=args.mm_charge_threshold,
                chunk_size=args.field_chunk_size,
            )
            cats = water_contact_category_counts(contacts)
            frame_site_rows = site_specific_field_rows(per_qm)
            for sr in frame_site_rows:
                site_specific_rows.append({
                    "run": run_name,
                    "frame": raw.frame_index,
                    "time_ps": raw.time_ps,
                    **sr,
                })

            if prev_waters is None:
                added, removed = set(), set()
                jaccard = math.nan
            else:
                added = selected - prev_waters
                removed = prev_waters - selected
                union = selected | prev_waters
                jaccard = len(selected & prev_waters) / len(union) if union else 1.0
            prev_waters = selected.copy()
            membership_by_run[run_name].append((raw.frame_index, selected.copy()))

            row = {
                "run": run_name,
                "frame": raw.frame_index,
                "time_ps": raw.time_ps,
                "region": region_name,
                "boundary_charge_method": args.boundary_charge_method,
                "n_adaptive_qm_waters": len(selected),
                "adaptive_qm_water_residue_indices": ";".join(map(str, sorted(selected))),
                "n_hbonded_waters": len(hbond_selected),
                "hbonded_water_residue_indices": ";".join(map(str, sorted(hbond_selected))),
                "n_zn_coordinated_waters": len(zn_selected),
                "zn_coordinated_water_residue_indices": ";".join(map(str, sorted(zn_selected))),
                "n_waters_selected_by_both": len(hbond_selected & zn_selected),
                "zn_water_cutoff_A": float(args.zn_water_cutoff),
                "n_water_added": len(added),
                "n_water_removed": len(removed),
                "water_added_residue_indices": ";".join(map(str, sorted(added))),
                "water_removed_residue_indices": ";".join(map(str, sorted(removed))),
                "water_membership_jaccard": jaccard,
                "n_qm_real_atoms": len(fp.qm_atom_indices),
                "n_qm_sites_including_links": len(fp.qm_elements),
                "n_link_atoms": fp.n_link_atoms,
                "qm_formal_charge": fp.qm_formal_charge,
                "mm_charge_sum_e": fp.mm_charge_sum,
                "system_charge_e": fp.system_charge,
                "charge_error_e": fp.charge_error,
                "boundary_ff_qm_sum_e": fp.boundary_ff_qm_sum,
                "boundary_charge_residual_e": fp.boundary_charge_residual,
                "n_boundary_virtual_sites": fp.n_boundary_virtual_sites,
                "n_detected_close_ions_at_guard": len(fp.close_ions),
                **zn, **ions, **emb, **cats,
            }
            frame_rows.append(row)

            water_by_id = {w.residue_index: w for w in spec.water_records}
            for wid in sorted(selected):
                w = water_by_id[wid]
                membership_rows.append({
                    "run": run_name,
                    "frame": raw.frame_index,
                    "time_ps": raw.time_ps,
                    "water_residue_index": int(w.residue_index),
                    "water_resseq": int(w.residue_seq),
                    "water_oxygen_index": int(w.oxygen_index),
                    "n_hbond_contacts": sum(c["water_residue_index"] == wid for c in contacts),
                    "is_hbond_selected": wid in hbond_selected,
                    "is_zn_coordinated": wid in zn_selected,
                    "selection_modes": ";".join([m for m, ok in (("hbond", wid in hbond_selected), ("zn_coordination", wid in zn_selected)) if ok]),
                })
            for c in contacts:
                contact_rows.append({"run": run_name, "frame": raw.frame_index, "time_ps": raw.time_ps, **c})
            for c in zn_contacts:
                zn_contact_rows.append({"run": run_name, "frame": raw.frame_index, "time_ps": raw.time_ps, **c})
            if field_fh is not None:
                for fq in per_qm:
                    field_record = {"run": run_name, "frame": raw.frame_index, "time_ps": raw.time_ps, **fq}
                    if field_writer is None:
                        field_writer = csv.DictWriter(field_fh, fieldnames=list(field_record.keys()))
                        field_writer.writeheader()
                    field_writer.writerow(field_record)

            n_run += 1
            if args.progress_every and n_run % args.progress_every == 0:
                print(
                    f"  analyzed {n_run} frames; latest frame={raw.frame_index}, "
                    f"waters={len(selected)} (hbond={len(hbond_selected)}, zn={len(zn_selected)}), "
                    f"Emax={emb['Emax_V_per_A']:.4f} V/A at {format_emax_identity(emb)}"
                )
        if n_run == 0:
            raise RuntimeError(f"no frames were analyzed for {run_name}; check --frames")
        print_run_field_summary(
            run_name,
            frame_rows[run_frame_start:],
            site_specific_rows[run_site_start:],
        )

    if field_fh is not None:
        field_fh.close()

    frame_df = pd.DataFrame(frame_rows)
    membership_df = pd.DataFrame(membership_rows, columns=[
        "run", "frame", "time_ps", "water_residue_index", "water_resseq",
        "water_oxygen_index", "n_hbond_contacts", "is_hbond_selected",
        "is_zn_coordinated", "selection_modes"
    ])
    contact_df = pd.DataFrame(contact_rows)
    zn_contact_df = pd.DataFrame(zn_contact_rows, columns=[
        "run", "frame", "time_ps", "water_residue_index", "water_resseq",
        "water_oxygen_index", "metal_atom_index", "metal_resname",
        "metal_resseq", "metal_atom_name", "zn_o_distance_A", "zn_water_cutoff_A"
    ])
    site_specific_df = pd.DataFrame(site_specific_rows)

    frame_df.to_csv(region_dir / "frame_metrics.csv", index=False)
    membership_df.to_csv(region_dir / "water_membership.csv", index=False)
    contact_df.to_csv(region_dir / "water_hbond_contacts.csv", index=False)
    zn_contact_df.to_csv(region_dir / "water_zn_contacts.csv", index=False)
    site_specific_df.to_csv(region_dir / "site_specific_fields.csv", index=False)

    water_occ = water_occupancy_table(frame_df, membership_df)
    partner_occ = partner_occupancy_table(frame_df, contact_df) if len(contact_df) else pd.DataFrame()
    residence = pd.DataFrame(make_residence_events(membership_by_run))
    run_summary = summarize_runs(frame_df)
    outliers = field_outliers(frame_df, args.outlier_percentile)

    water_occ.to_csv(region_dir / "water_occupancy.csv", index=False)
    partner_occ.to_csv(region_dir / "partner_occupancy.csv", index=False)
    residence.to_csv(region_dir / "water_residence_events.csv", index=False)
    run_summary.to_csv(region_dir / "run_summary.csv", index=False)
    outliers.to_csv(region_dir / "field_outliers.csv", index=False)

    metadata = {
        "topology": str(Path(top_path).resolve()),
        "runs": runs,
        "region": region_name,
        "boundary_charge_method": args.boundary_charge_method,
        "resnum_offset": args.resnum_offset,
        "frames": args.frames,
        "hbond_distance_A": args.hbond_distance,
        "hbond_angle_deg": args.hbond_angle,
        "zn_water_cutoff_A": args.zn_water_cutoff,
        "ion_guard_A": args.ion_guard,
        "shell_radii_A": list(args.shell_radii),
        "mm_charge_threshold_e": args.mm_charge_threshold,
        "system_charge_e": float(np.sum(charges)),
        "qm_formal_charge": int(spec.formal_charge),
        "n_topology_atoms": int(mdtop.n_atoms),
        "n_waters": int(len(spec.water_records)),
        "n_detected_free_ions": int(len(spec.ion_records)),
    }
    (region_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_report(
        region_dir / "analysis_report.md", frame_df, run_summary, water_occ,
        partner_occ, outliers, region_name, args.boundary_charge_method,
        args.outlier_percentile,
    )

    if not args.no_plots:
        try:
            from plot_analysis import make_plots
            make_plots(region_dir)
        except Exception as exc:
            print(f"WARNING: plot generation failed: {exc}")
    print(f"Wrote analysis to {region_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", help="AMBER parm7/prmtop shared by all runs")
    ap.add_argument("--traj", action="append", help="trajectory; repeat once per run")
    ap.add_argument("--run-name", action="append", help="run label; repeat once per --traj")
    ap.add_argument("--config", help="JSON file containing {'top': ..., 'runs':[{'name':...,'traj':...}, ...]}")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--region", action="append", choices=list(qp.QM_REGIONS), help="repeat to analyze multiple regions; default minimal")
    ap.add_argument("--resnum-offset", type=int, default=qp.DEFAULT_RESNUM_OFFSET)
    ap.add_argument("--frames", default="0:None:1", help="start:stop:stride applied independently to every run")
    ap.add_argument("--hbond-distance", type=float, default=3.5)
    ap.add_argument("--hbond-angle", type=float, default=140.0)
    ap.add_argument(
        "--zn-water-cutoff", type=float, default=qp.DEFAULT_ZN_WATER_CUTOFF_A,
        help="Zn-O cutoff in Angstrom for direct Zn-coordinated adaptive QM waters (default 2.6; 0 disables)",
    )
    ap.add_argument("--boundary-charge-method", choices=("shift", "rcd"), default="shift")
    ap.add_argument("--ion-guard", type=float, default=4.0, help="distance used only for counting existing PartQMMM ion-guard alerts")
    ap.add_argument("--shell-radii", type=parse_shells, default=(3.0, 4.0, 5.0), help="comma-separated MM-charge shell radii in Angstrom")
    ap.add_argument("--mm-charge-threshold", type=float, default=1e-8, help="ignore |q| below this value for nearest-site/shell/field diagnostics")
    ap.add_argument("--field-chunk-size", type=int, default=5000)
    ap.add_argument("--skip-per-atom-fields", action="store_true", help="do not write the potentially large qm_atom_fields.csv; frame Emax/potential diagnostics are still computed")
    ap.add_argument("--outlier-percentile", type=float, default=99.0, help="flag frames at/above this Emax percentile")
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    if not (50.0 <= args.outlier_percentile < 100.0):
        ap.error("--outlier-percentile must be >=50 and <100")

    if args.zn_water_cutoff < 0:
        ap.error("--zn-water-cutoff must be >= 0 (0 disables Zn-water selection)")

    top, runs = load_run_config(args)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    regions = args.region or ["minimal"]
    root_manifest = {
        "topology": str(Path(top).resolve()),
        "runs": runs,
        "regions": regions,
        "boundary_charge_method": args.boundary_charge_method,
        "zn_water_cutoff_A": args.zn_water_cutoff,
    }
    (outdir / "suite_manifest.json").write_text(json.dumps(root_manifest, indent=2))
    for region in regions:
        analyze_one_region(top, runs, outdir, args, region)


if __name__ == "__main__":
    main()
