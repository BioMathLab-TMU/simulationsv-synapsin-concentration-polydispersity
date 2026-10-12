"""Regenerate compact audit plots from outputs/ after analysis.run.

These are editable diagnostic plots, not replacements for the frozen article
figure files in figures/. Colors follow the manuscript's blue/orange/purple/
gray palette; line style distinguishes radius distributions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run import OUT
from .paper_figures import VIOLET, CYAN, MAGENTA, AMBER, MUTED

BLUE = CYAN
ORANGE = AMBER
PURPLE = MAGENTA
GRAY = VIOLET
CONDITIONS = ("C10_mono", "C10_poly", "C60_mono", "C60_poly")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 240})


def save(fig, stem: str) -> None:
    destination = OUT / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(destination / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def time_courses(runs: pd.DataFrame) -> None:
    panels = (
        ("largest_component_vesicle_fraction", "SV in largest component", BLUE),
        ("vesicle_recruited_fraction", "SV in any nonsingleton", BLUE),
        ("synapsin_recruited_fraction", "Synapsin in any nonsingleton", ORANGE),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.35), constrained_layout=True)
    for condition in CONDITIONS:
        run_ids = runs.loc[runs.Study_Condition.eq(condition), "Study_Run_ID"]
        curves = [pd.read_csv(OUT / run_id / "frames.csv") for run_id in run_ids]
        if len(curves) != 3:
            raise ValueError(f"Expected three seed trajectories in {condition}")
        times = curves[0].time_ms.to_numpy(float)
        if any(not np.allclose(frame.time_ms, times) for frame in curves[1:]):
            raise ValueError("Unaligned save schedules")
        line_style = "-" if condition.endswith("mono") else "--"
        opacity = 1 if condition.startswith("C60") else .7
        for axis, (field, title, color) in zip(axes, panels):
            values = np.stack([frame[field].to_numpy(float) for frame in curves])
            for individual in values:
                axis.plot(times, individual, color=color if condition.startswith("C60") else GRAY, alpha=.08, linewidth=.6)
            axis.plot(times, values.mean(axis=0), color=color if condition.startswith("C60") else GRAY, linestyle=line_style, alpha=opacity, linewidth=1.8, label=condition.replace("_", " "))
            axis.set(title=title, xlabel="Physical time (ms)", ylabel="Fraction", xlim=(times.min(), times.max()), ylim=(0, 1.02))
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    save(fig, "audit_connectivity_timecourses")


def paired_contrasts(contrasts: pd.DataFrame) -> None:
    measures = ("late_mean_largest_component_vesicle_fraction", "late_mean_vesicle_recruited_fraction", "late_mean_synapsin_recruited_fraction")
    selected = ("concentration_mono", "concentration_poly", "dispersion_10", "dispersion_60", "factor_interaction")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    for axis, metric, title in zip(axes, measures, ("SV in largest component", "SV recruitment", "Synapsin recruitment")):
        frame = contrasts[contrasts.metric.eq(metric)].set_index("contrast").loc[list(selected)]
        for y, (name, row) in enumerate(frame.iterrows()):
            color = BLUE if name.startswith("concentration") else PURPLE
            axis.errorbar(row.mean_paired_difference, y, xerr=[[row.mean_paired_difference - row.ci95_low], [row.ci95_high - row.mean_paired_difference]], fmt="D", color=color, capsize=2.5, markersize=4)
        axis.axvline(0, color=MUTED, linestyle=":", linewidth=1)
        axis.set(yticks=range(len(selected)), yticklabels=[name.replace("_", " ") for name in selected], xlabel="Seed-paired difference", title=title)
    save(fig, "audit_seed_paired_effects")


def mesoscale() -> None:
    rdf = pd.read_csv(OUT / "RDF_Run_Level.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    for axis, pair, color in zip(axes, ("SS", "SV", "VV"), (ORANGE, PURPLE, BLUE)):
        for condition in CONDITIONS:
            observed = rdf[rdf.pair.eq(pair) & rdf.Study_Condition.eq(condition)]
            curve = observed.groupby("r_mid_nm").rdf_geometric_null_ratio.mean()
            axis.plot(curve.index, curve, linestyle="-" if condition.endswith("mono") else "--", color=color if condition.startswith("C60") else GRAY, linewidth=1.5, label=condition.replace("_", " "))
        axis.axhline(1., color=MUTED, linestyle=":", linewidth=.8)
        axis.set(title=pair, xlabel="Center separation (nm)", ylabel="Pair ratio to geometric null", xlim=(0, 120))
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "audit_geometry_matched_rdf")


def pca() -> None:
    scores = pd.read_csv(OUT / "exploratory_pca_scores.csv")
    runs = pd.read_csv(OUT / "run_level.csv")
    frame = scores.merge(runs[["Study_Run_ID", "Study_Condition"]], on="Study_Run_ID", validate="one_to_one")
    fig, axis = plt.subplots(figsize=(5.2, 4.1), constrained_layout=True)
    for condition in CONDITIONS:
        selected = frame[frame.Study_Condition.eq(condition)]
        axis.scatter(selected.PC1, selected.PC2, color=PURPLE if condition.startswith("C60") else GRAY, marker="o" if condition.endswith("mono") else "s", s=38, label=condition.replace("_", " "))
    axis.set(xlabel="PC1 (85.13% of variance)", ylabel="PC2 (10.21% of variance)", title="Exploratory run-level PCA")
    axis.legend(fontsize=7, frameon=False)
    save(fig, "audit_exploratory_pca")


def main() -> None:
    style()
    if not (OUT / "validation.json").is_file():
        raise FileNotFoundError("Run `python -m analysis.run` first")
    runs = pd.read_csv(OUT / "run_level.csv")
    time_courses(runs)
    paired_contrasts(pd.read_csv(OUT / "paired_contrasts.csv"))
    mesoscale()
    pca()
    print("ARTICLE_TWO_AUDIT_PLOTS_OK count=4")


if __name__ == "__main__":
    main()
