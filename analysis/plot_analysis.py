#!/usr/bin/env python3
"""Generate diagnostic plots from an analyze_ensemble.py output directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _integer_hist(series, xlabel, path):
    x = series.dropna()
    if not len(x):
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bins = range(int(x.min()), int(x.max()) + 2)
    ax.hist(x, bins=bins, align="left", rwidth=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frames")
    _save(fig, path)


def _bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def _plot_fixed_sites(site_df, run, site_class, plotdir, stem, title):
    g = site_df[(site_df.run == run) & (site_df.site_class == site_class)].copy()
    if not len(g):
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, h in g.groupby("site_label", sort=True):
        h = h.sort_values("frame")
        ax.plot(h.frame, h.field_magnitude_V_per_A, linewidth=1.0, label=str(label))
    ax.set_xlabel("MD frame")
    ax.set_ylabel("MM field magnitude (V/A)")
    ax.set_title(title)
    if g.site_label.nunique() <= 12:
        ax.legend(fontsize=7, ncol=2)
    _save(fig, plotdir / f"{run}_{stem}.png")


def _plot_water_sites(site_df, run, plotdir):
    g = site_df[(site_df.run == run) & (site_df.site_class == "QM_WATER_O")].copy()
    if not len(g):
        return

    # Every dot is one adaptive-QM water oxygen in one sampled frame.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(g.frame, g.field_magnitude_V_per_A, s=8, alpha=0.45)
    ax.set_xlabel("MD frame")
    ax.set_ylabel("MM field at adaptive-QM water O (V/A)")
    ax.set_title(f"{run}: all adaptive-QM water oxygen fields")
    _save(fig, plotdir / f"{run}_QM_water_O_site_fields_timeseries.png")

    # Compact envelope: mean and maximum over all QM-water oxygen sites per frame.
    agg = g.groupby("frame").field_magnitude_V_per_A.agg(["mean", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(agg.frame, agg["mean"], label="Mean QM-water O field")
    ax.plot(agg.frame, agg["max"], label="Max QM-water O field")
    ax.set_xlabel("MD frame")
    ax.set_ylabel("MM field magnitude (V/A)")
    ax.set_title(f"{run}: adaptive-QM water oxygen field envelope")
    ax.legend(fontsize=8)
    _save(fig, plotdir / f"{run}_QM_water_O_field_mean_max_timeseries.png")

    # Individual time series for the most persistent water identities only.
    top_labels = g.groupby("site_label").frame.nunique().sort_values(ascending=False).head(10).index
    gt = g[g.site_label.isin(top_labels)]
    if len(gt):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label, h in gt.groupby("site_label", sort=False):
            h = h.sort_values("frame")
            ax.plot(h.frame, h.field_magnitude_V_per_A, marker=".", markersize=2,
                    linewidth=0.8, label=str(label))
        ax.set_xlabel("MD frame")
        ax.set_ylabel("MM field at water O (V/A)")
        ax.set_title(f"{run}: 10 most persistent adaptive-QM water O sites")
        ax.legend(fontsize=6, ncol=2)
        _save(fig, plotdir / f"{run}_QM_water_O_top10_site_fields_timeseries.png")


def make_plots(directory: str | Path):
    d = Path(directory)
    f = pd.read_csv(d / "frame_metrics.csv")
    plotdir = d / "plots"
    plotdir.mkdir(exist_ok=True)

    site_path = d / "site_specific_fields.csv"
    site_df = pd.read_csv(site_path) if site_path.exists() else pd.DataFrame()

    if "n_adaptive_qm_waters" in f:
        _integer_hist(
            f.n_adaptive_qm_waters,
            "Adaptive QM waters per frame (H-bond OR Zn coordination)",
            plotdir / "adaptive_water_count_hist.png",
        )
    if "n_hbonded_waters" in f:
        _integer_hist(
            f.n_hbonded_waters,
            "H-bond-selected QM waters per frame",
            plotdir / "hbonded_water_count_hist.png",
        )
    if "n_zn_coordinated_waters" in f:
        _integer_hist(
            f.n_zn_coordinated_waters,
            "Zn-coordinated QM waters per frame",
            plotdir / "zn_coordinated_water_count_hist.png",
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(f.nearest_zn_water_A.dropna(), bins=40)
    if "zn_water_cutoff_A" in f and f.zn_water_cutoff_A.notna().any():
        ax.axvline(float(f.zn_water_cutoff_A.dropna().iloc[0]), linestyle="--")
    ax.set_xlabel("Nearest Zn-water O distance (A)")
    ax.set_ylabel("Frames")
    _save(fig, plotdir / "nearest_zn_water_hist.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(f.Emax_V_per_A.dropna(), bins=50)
    ax.set_xlabel("Maximum MM field on all QM sites, Emax (V/A)")
    ax.set_ylabel("Frames")
    _save(fig, plotdir / "Emax_hist.png")

    if "Emax_real_qm_V_per_A" in f:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.hist(f.Emax_real_qm_V_per_A.dropna(), bins=50)
        ax.set_xlabel("Maximum MM field on real QM atoms (V/A)")
        ax.set_ylabel("Frames")
        _save(fig, plotdir / "Emax_real_qm_hist.png")

    # Which exact site carries the link-inclusive Emax?
    if "Emax_atom_label" in f:
        labels = f.Emax_atom_label.fillna("UNKNOWN").astype(str)
        if "Emax_is_link_atom" in f:
            link = _bool_series(f.Emax_is_link_atom)
            display = labels.copy()
            display.loc[link] = display.loc[link] + " [LINK]"
        else:
            display = labels
        counts = display.value_counts().head(20).sort_values()
        if len(counts):
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(counts.index, counts.values)
            ax.set_xlabel("Frames carrying Emax")
            ax.set_ylabel("QM site")
            ax.set_title("Most frequent atoms/sites carrying frame-level Emax")
            _save(fig, plotdir / "Emax_atom_frequency.png")

    if "Emax_is_link_atom" in f:
        link = _bool_series(f.Emax_is_link_atom)
        counts = pd.Series({"Real QM atom": int((~link).sum()), "Synthetic link H": int(link.sum())})
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Frames")
        ax.set_title("Location of frame-level Emax")
        _save(fig, plotdir / "Emax_link_vs_real.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(f.nearest_mm_charge_to_qm_A, f.Emax_V_per_A, s=10, alpha=0.6)
    ax.set_xlabel("Nearest nonzero MM embedding charge to real QM (A)")
    ax.set_ylabel("Emax (V/A)")
    _save(fig, plotdir / "Emax_vs_nearest_mm_charge.png")

    if "E_Zn_V_per_A" in f:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(f.nearest_zn_water_A, f.E_Zn_V_per_A, s=10, alpha=0.6)
        if "zn_water_cutoff_A" in f and f.zn_water_cutoff_A.notna().any():
            ax.axvline(float(f.zn_water_cutoff_A.dropna().iloc[0]), linestyle="--")
        ax.set_xlabel("Nearest Zn-water O distance (A)")
        ax.set_ylabel("MM field magnitude at Zn (V/A)")
        _save(fig, plotdir / "Zn_field_vs_nearest_zn_water.png")

    for run, g in f.groupby("run", sort=False):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(g.frame, g.Emax_real_qm_V_per_A if "Emax_real_qm_V_per_A" in g else g.Emax_V_per_A)
        ax.set_xlabel("MD frame")
        ax.set_ylabel("Emax on real QM atoms (V/A)" if "Emax_real_qm_V_per_A" in g else "Emax (V/A)")
        ax.set_title(str(run))
        _save(fig, plotdir / f"{run}_Emax_timeseries.png")

        # Link-inclusive Emax vs real-QM Emax exposes when the synthetic link H
        # is responsible for an apparently large frame-level maximum.
        if "Emax_real_qm_V_per_A" in g:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(g.frame, g.Emax_V_per_A, label="All QM sites incl. link H")
            ax.plot(g.frame, g.Emax_real_qm_V_per_A, label="Real QM atoms only")
            ax.set_xlabel("MD frame")
            ax.set_ylabel("Maximum MM field (V/A)")
            ax.set_title(f"{run}: Emax with/without synthetic link H")
            ax.legend(fontsize=8)
            _save(fig, plotdir / f"{run}_Emax_all_vs_real_timeseries.png")

        if "Emax_atom_label" in g:
            labels = g.Emax_atom_label.fillna("UNKNOWN").astype(str)
            if "Emax_is_link_atom" in g:
                link = _bool_series(g.Emax_is_link_atom)
                labels.loc[link] = labels.loc[link] + " [LINK]"
            counts = labels.value_counts().head(15).sort_values()
            if len(counts):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(counts.index, counts.values)
                ax.set_xlabel("Frames carrying Emax")
                ax.set_ylabel("QM site")
                ax.set_title(f"{run}: most frequent Emax sites")
                _save(fig, plotdir / f"{run}_Emax_atom_frequency.png")

        water_col = "n_adaptive_qm_waters" if "n_adaptive_qm_waters" in g else "n_hbonded_waters"
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(g.frame, g[water_col])
        ax.set_xlabel("MD frame")
        ax.set_ylabel("Adaptive QM waters")
        ax.set_title(str(run))
        _save(fig, plotdir / f"{run}_water_count_timeseries.png")

        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(g.frame, g.nearest_zn_water_A)
        if "zn_water_cutoff_A" in g and g.zn_water_cutoff_A.notna().any():
            ax.axhline(float(g.zn_water_cutoff_A.dropna().iloc[0]), linestyle="--")
        ax.set_xlabel("MD frame")
        ax.set_ylabel("Nearest Zn-water O distance (A)")
        ax.set_title(str(run))
        _save(fig, plotdir / f"{run}_Zn_water_distance_timeseries.png")

        if "E_Zn_V_per_A" in g:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(g.frame, g.E_Zn_V_per_A)
            ax.set_xlabel("MD frame")
            ax.set_ylabel("MM field magnitude at Zn (V/A)")
            ax.set_title(str(run))
            _save(fig, plotdir / f"{run}_Zn_field_timeseries.png")

        if len(site_df):
            _plot_fixed_sites(
                site_df, run, "ZN", plotdir,
                "Zn_site_field_timeseries", f"{run}: Zn site-specific MM field",
            )
            _plot_fixed_sites(
                site_df, run, "HIS_N", plotdir,
                "HIS_N_site_fields_timeseries", f"{run}: QM histidine N-site MM fields",
            )
            _plot_fixed_sites(
                site_df, run, "GLU_O", plotdir,
                "GLU_O_site_fields_timeseries", f"{run}: QM glutamate O-site MM fields",
            )
            _plot_water_sites(site_df, run, plotdir)

    occ_path = d / "water_occupancy.csv"
    if occ_path.exists():
        occ = pd.read_csv(occ_path)
        occ = occ[occ.scope == "ALL_RUNS"].head(20)
        if len(occ):
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [str(int(x)) for x in occ.water_residue_index]
            ax.bar(labels, occ.occupancy_fraction)
            ax.set_xlabel("Water residue index")
            ax.set_ylabel("Adaptive-QM occupancy fraction")
            ax.tick_params(axis="x", rotation=60)
            _save(fig, plotdir / "top_water_occupancies.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("analysis_dir")
    args = ap.parse_args()
    make_plots(args.analysis_dir)


if __name__ == "__main__":
    main()
