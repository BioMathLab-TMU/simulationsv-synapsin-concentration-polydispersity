"""Regenerate article-two figures from the published run data and audit tables.

The semantic palette is taken from the archived Single_Execution notebooks:
deep violet #5c63a2, cyan #20a5e4, magenta #995286 and amber #f4ab33.
Color encodes concentration in comparison plots; line style/marker encodes
radius distribution. No frame is treated as an independent replicate.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .run import OUT, ROOT

FIG = ROOT / "figures"
REF = ROOT / "reference"
CONDITIONS = ("C10_mono", "C10_poly", "C60_mono", "C60_poly")
VIOLET, CYAN, MAGENTA, AMBER = "#5c63a2", "#20a5e4", "#995286", "#f4ab33"
INK, MUTED, GRID = "#222222", "#5d6470", "#e5e8ec"
COLOR = {"C10_mono": VIOLET, "C10_poly": VIOLET, "C60_mono": CYAN, "C60_poly": CYAN}
LABEL = {"C10_mono": "10 µM · mono", "C10_poly": "10 µM · poly", "C60_mono": "60 µM · mono", "C60_poly": "60 µM · poly"}
MARKER = {"C10_mono": "o", "C10_poly": "s", "C60_mono": "o", "C60_poly": "s"}
STYLE = {"C10_mono": "-", "C10_poly": (0, (4, 2)), "C60_mono": "-", "C60_poly": (0, (4, 2))}


def theme() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.2, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "grid.color": GRID, "grid.linewidth": .55,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.08)
    fig.savefig(FIG / f"{name}.png", dpi=320, bbox_inches="tight", pad_inches=.08)
    fig.savefig(FIG / f"{name}.tif", dpi=320, bbox_inches="tight", pad_inches=.08,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-.12, 1.07, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def key(fig: plt.Figure, y: float = .985) -> None:
    handles = [Line2D([0], [0], color=COLOR[c], ls=STYLE[c], marker=MARKER[c],
                      markerfacecolor="white" if c.endswith("poly") else COLOR[c],
                      markersize=5, linewidth=1.8, label=LABEL[c]) for c in CONDITIONS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, y), columnspacing=1.4, handlelength=2.4, fontsize=8.3)


def run_table() -> pd.DataFrame:
    runs = pd.read_csv(OUT / "run_level.csv")
    assert len(runs) == 12 and runs.groupby("Study_Condition").size().eq(3).all()
    return runs


def condition_points(ax: plt.Axes, runs: pd.DataFrame, field: str, *, ylimits=None) -> None:
    for i, c in enumerate(CONDITIONS):
        vals = runs.loc[runs.Study_Condition.eq(c), field].to_numpy(float)
        assert len(vals) == 3 and np.isfinite(vals).all()
        x = i + 1
        # Put the range and mean beside the three seeds: a mean bar drawn
        # through the seed symbols hides both their fill and their spread.
        summary_x = x + .21
        ax.plot([summary_x, summary_x], [vals.min(), vals.max()],
                color=COLOR[c], lw=1.4, alpha=.9, zorder=2)
        ax.scatter(x + np.array([-.21, -.09, .03]), vals, s=48,
                   marker=MARKER[c], facecolor="white" if c.endswith("poly") else COLOR[c],
                   edgecolor=COLOR[c], lw=1.5, zorder=3)
        ax.scatter([summary_x], [vals.mean()], marker="_", s=180,
                   lw=2.0, color=INK, zorder=4)
    ax.set_xticks([1, 2, 3, 4], ["Mono", "Poly", "Mono", "Poly"])
    ax.set_xlim(.55, 4.47)
    ax.axvline(2.5, color=GRID, lw=.8)
    ax.text(1.5, -.25, "10 µM", transform=ax.get_xaxis_transform(), ha="center", color=VIOLET, fontweight="bold")
    ax.text(3.5, -.25, "60 µM", transform=ax.get_xaxis_transform(), ha="center", color=CYAN, fontweight="bold")
    if ylimits is not None:
        ax.set_ylim(*ylimits)


def connectivity(runs: pd.DataFrame) -> None:
    fields = (
        ("largest_component_vesicle_fraction", "SVs in largest component", "0–1 fraction"),
        ("vesicle_recruited_fraction", "SVs in any nonsingleton", "0–1 fraction"),
        ("synapsin_recruited_fraction", "Synapsin in any nonsingleton", "0–1 fraction"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.8), layout="constrained",
                             gridspec_kw={"height_ratios": [1.45, 1]})
    fig.set_layout_engine("constrained", h_pad=.3, w_pad=.15)
    for j, (field, title, _) in enumerate(fields):
        ax = axes[0, j]
        for c in CONDITIONS:
            per_seed = [pd.read_csv(OUT / rid / "frames.csv", usecols=["time_ms", field])
                        for rid in runs.loc[runs.Study_Condition.eq(c), "Study_Run_ID"]]
            x = per_seed[0].time_ms.to_numpy(float)
            assert all(np.allclose(f.time_ms, x) for f in per_seed[1:])
            ys = np.stack([f[field].to_numpy(float) for f in per_seed])
            for y in ys:
                ax.plot(x, y, color=COLOR[c], lw=.65, alpha=.13, ls=STYLE[c])
            ax.plot(x, ys.mean(axis=0), color=COLOR[c], lw=1.9, ls=STYLE[c])
        ax.axvspan(19.74, 28, color=VIOLET, alpha=.055, lw=0)
        ax.set(xlim=(0, 28), ylim=(0, 1.03), xlabel="Time (ms)")
        if j == 0:
            ax.set_ylabel("Fraction")
        panel(ax, chr(97+j), title)
        lower = axes[1, j]
        condition_points(lower, runs, "late_mean_" + field, ylimits=(-.025, 1.055))
        if j == 0:
            lower.set_ylabel("Late-window fraction")
        panel(lower, chr(100+j), "Terminal-window summaries")
    key(fig, 1.008)
    save(fig, "fig1_connectivity")


def structure(runs: pd.DataFrame) -> None:
    items = (
        ("late_mean_weight_average_finite_cluster_size", "Finite-component size", "Agents per finite component", (0, None)),
        ("late_particle_weighted_shape_anisotropy", "Component anisotropy", "Relative shape anisotropy κ²", (-.025, .55)),
        ("late_edge_structure_mean_type_assortativity", "Mixed-component assortativity", "Newman assortativity", (-1.05, .05)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.9, 3.65), layout="constrained")
    for j, (field, title, ylabel, limit) in enumerate(items):
        condition_points(axes[j], runs, field, ylimits=limit)
        axes[j].set_ylabel(ylabel)
        panel(axes[j], chr(97+j), title)
    save(fig, "fig2_structure")


def robustness() -> None:
    shells = pd.read_csv(OUT / "run_level.csv")[["Study_Run_ID", "Study_Condition"]].merge(
        pd.read_csv(REF / "Shell_Sensitivity_Run_Level_v5.csv"),
        on=["Study_Run_ID", "Study_Condition"], validate="one_to_many")
    paired = pd.read_csv(OUT / "paired_contrasts.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.55), layout="constrained")
    for c in CONDITIONS:
        s = shells[shells.Study_Condition.eq(c)].groupby("analysis_shell_nm")["late_mean_largest_component_vesicle_fraction"].mean()
        axes[0].plot(s.index, s.values, color=COLOR[c], ls=STYLE[c], marker=MARKER[c],
                     markerfacecolor="white" if c.endswith("poly") else COLOR[c],
                     linewidth=1.7, markersize=4.5)
    axes[0].set(xlabel="Outer-surface graph shell (nm)", ylabel="SVs in largest component", ylim=(0, 1.04), xticks=[0,2,4,6])
    panel(axes[0], "a", "Graph definition")
    labels = ("10 µM", "60 µM", "Interaction")
    names = ("dispersion_10", "dispersion_60", "factor_interaction")
    for j, metric in enumerate(("late_mean_largest_component_vesicle_fraction", "late_mean_vesicle_recruited_fraction"), 1):
        ax = axes[j]
        for k, (name, label) in enumerate(zip(names, labels)):
            row = paired.loc[paired.metric.eq(metric) & paired.contrast.eq(name)].iloc[0]
            delta = row.mean_paired_difference
            ax.plot([row.ci95_low, row.ci95_high], [k, k], color=MAGENTA, lw=1.4)
            ax.scatter(delta, k, color=MAGENTA, marker="D", s=31, zorder=3)
        ax.axvline(0, color=MUTED, ls=(0,(2,2)), lw=1)
        ax.set(yticks=np.arange(3), yticklabels=labels, xlabel="Seed-paired difference in fraction", ylim=(-.6, 2.5))
        ax.invert_yaxis()
        panel(ax, chr(97+j), "Largest component" if j == 1 else "Any nonsingleton")
    axes[0].set_xlim(-.1, 8.1)
    axes[0].text(6.3, .98, "60 µM", color=CYAN, va="center", fontsize=8)
    axes[0].text(6.3, .11, "10 µM", color=VIOLET, va="center", fontsize=8)
    save(fig, "fig3_robustness")


def density() -> None:
    hist = pd.read_csv(OUT / "Local_Density_Histograms_Run_Level.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.65), layout="constrained")
    for j, species in enumerate(("vesicle", "synapsin")):
        ax = axes[j]
        for c in CONDITIONS:
            s = hist[hist.species.eq(species) & hist.probe_radius_nm.eq(50) & hist.Study_Condition.eq(c)]
            zero = s[s.bin_kind.eq("zero")].probability_mass.mean()
            p = s[s.bin_kind.eq("positive")].groupby(["ratio_low", "ratio_high"]).probability_mass.mean().reset_index()
            x = np.log10(np.sqrt(p.ratio_low.to_numpy(float) * p.ratio_high.to_numpy(float)))
            ax.plot(x, p.probability_mass, color=COLOR[c], ls=STYLE[c], lw=1.6)
            ax.text(.04 + (.47 if c.startswith("C60") else 0), .93 - (.10 if c.endswith("poly") else 0),
                    f"P(ρ=0), {LABEL[c]}: {zero:.2f}", transform=ax.transAxes, fontsize=7.1,
                    color=COLOR[c], va="top")
        ax.set(xlim=(-3, 3), xlabel="log₁₀(local / global number density)", ylabel="Probe probability")
        panel(ax, chr(97+j), species.capitalize() + " · 50 nm probes")
    save(fig, "supp_density")


def rdf() -> None:
    data = pd.read_csv(OUT / "RDF_Run_Level.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.45), layout="constrained")
    for j, pair in enumerate(("SS", "SV", "VV")):
        ax = axes[j]
        for c in CONDITIONS:
            s = data[data.pair.eq(pair) & data.Study_Condition.eq(c) & data.baseline_expected_pairs_at_least_five]
            curve = s.groupby("r_mid_nm").rdf_geometric_null_ratio.mean()
            ax.plot(curve.index, curve.values, color=COLOR[c], ls=STYLE[c], lw=1.6)
        ax.axhline(1, color=MUTED, ls=(0,(2,2)), lw=.9)
        ax.set(xlim=(0,120), xlabel="Center distance (nm)", ylabel="Ratio to geometric null")
        panel(ax, chr(97+j), {"SS":"Synapsin–synapsin", "SV":"Synapsin–vesicle", "VV":"Vesicle–vesicle"}[pair])
    key(fig, -.08)
    save(fig, "supp_rdf")


def machine_learning() -> None:
    scores = pd.read_csv(OUT / "exploratory_pca_scores.csv").merge(
        run_table()[["Study_Run_ID", "Study_Condition"]], on="Study_Run_ID", validate="one_to_one")
    check = json.loads((OUT / "inference_checks.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.75), layout="constrained")
    for c in CONDITIONS:
        s = scores[scores.Study_Condition.eq(c)]
        axes[0].scatter(s.PC1, s.PC2, marker=MARKER[c], s=49,
                        facecolor="white" if c.endswith("poly") else COLOR[c],
                        edgecolor=COLOR[c], lw=1.25, label=LABEL[c])
    axes[0].set(xlabel=f"PC1 ({100*scores.PC1_explained_variance.iloc[0]:.1f}% variance)",
                ylabel=f"PC2 ({100*scores.PC2_explained_variance.iloc[0]:.1f}% variance)")
    panel(axes[0], "a", "Run-level structure · n = 12")
    perf = pd.read_csv(OUT / "exploratory_ml.csv").set_index("task")
    axes[1].bar([0,1], [perf.loc["concentration", "balanced_accuracy"], perf.loc["radius_dispersion", "balanced_accuracy"]],
                color=[CYAN, MAGENTA], width=.53, zorder=3)
    axes[1].axhline(.5, color=MUTED, ls=(0,(2,2)), lw=.9)
    axes[1].text(0, .92, "12/12", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")
    axes[1].text(1, .62, "8/12", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")
    axes[1].set(xticks=[0,1], xticklabels=["Concentration", "Radius spread"],
                ylabel="Held-seed balanced accuracy", ylim=(0,1.14))
    panel(axes[1], "b", "Exploratory classification")
    assert check["classifier_seed_block_joint_label_remapping"]["tail_fraction_ge_observed"] == .25
    key(fig, -.08)
    save(fig, "supp_ml")


def geometry_numerics() -> None:
    runs = run_table()
    geometry = pd.read_csv(REF / "radius_geometry.csv")
    audit = pd.read_csv(REF / "numerical_audit.csv")
    windows = pd.read_csv(REF / "late_window_sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), layout="constrained")
    condition_points(axes[0], geometry, "nominal_core_volume_fraction")
    axes[0].set_ylabel("Nominal core-volume fraction")
    panel(axes[0], "a", "Realized geometry")
    audit["drift_limit_percent"] = audit.drift_limited_proposal_fraction * 100
    condition_points(axes[1], audit, "drift_limit_percent")
    axes[1].set_ylabel("Drift-limited proposals (%)")
    panel(axes[1], "b", "Numerical intervention")
    windows = windows[windows.metric.eq("late_mean_largest_component_vesicle_fraction")]
    for c in CONDITIONS:
        s = windows[windows.Study_Condition.eq(c)].groupby("late_fraction").value.mean()
        axes[2].plot(100*s.index, s.values, color=COLOR[c], ls=STYLE[c], marker=MARKER[c],
                     markerfacecolor="white" if c.endswith("poly") else COLOR[c], lw=1.7)
    axes[2].set(xlabel="Terminal saved frames used (%)", ylabel="SVs in largest component", ylim=(0,1.04), xticks=[20,30,40])
    panel(axes[2], "c", "Observation window")
    save(fig, "supp_geometry_numerics")


def snapshots() -> None:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), layout="constrained")
    for j, c in enumerate(CONDITIONS):
        run = next(r for r in manifest["runs"] if r["condition_id"] == c and r["seed"] == 9210001)
        cfg = json.loads((ROOT / run["config"]).read_text(encoding="utf-8"))
        env = cfg["environment"][0]
        width = np.asarray(env["steps"], float)
        lengths = np.asarray([env[k] for k in ("length", "width", "height")], float)
        center = (env["out"] + 1) * width + np.ceil(lengths / width - 1e-9) * width / 2
        with h5py.File(ROOT / run["trajectory"], "r") as handle:
            last = handle["step_40000"]
            pos = last["positions"][:] - center
            types = last["types"][:].reshape(-1)
        ax = axes.flat[j]
        for t, color, size, name in ((1, AMBER, 2.1, "Synapsin"), (2, CYAN, 15, "SV")):
            p = pos[types == t]
            ax.scatter(p[:,0], p[:,1], s=size, color=color, alpha=.45 if t == 1 else .82,
                       linewidths=0, rasterized=True, label=name)
        ax.add_patch(plt.Circle((0,0), lengths[0]/2, fill=False, color=MUTED, lw=.9))
        ax.set(xlim=(-300,300), ylim=(-300,300), aspect="equal", xlabel="x (nm)", ylabel="y (nm)")
        panel(ax, chr(97+j), LABEL[c])
    axes.flat[0].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "supp_snapshots")


def experimental_context() -> None:
    """Published cryo-ET diameter summary is contextual, not a direct fit."""
    geom = pd.read_csv(REF / "radius_geometry.csv")
    low = geom[geom.Study_Condition.str.startswith("C10")]
    mono = low[low.Study_Condition.eq("C10_mono")]
    poly = low[low.Study_Condition.eq("C10_poly")]
    values = [2*mono.core_mean_nm.mean(), 2*poly.core_mean_nm.mean(), 41.6]
    spreads = [0, 2*poly.core_sd_population_nm.mean(), 5.6]
    labels = ["Model · single radius", "Model · heterogeneous", "Untreated synaptosomes\nFernández-Busnadiego et al. (2010)"]
    fig, ax = plt.subplots(figsize=(7.0, 3.1), layout="constrained")
    for i, (mean, sd, color) in enumerate(zip(values, spreads, (VIOLET, CYAN, MAGENTA))):
        ax.plot([mean-sd, mean+sd], [i,i], color=color, lw=3, solid_capstyle="round")
        ax.scatter(mean, i, color=color, marker="D", s=47, zorder=3)
        ax.text(mean+sd+.65, i, f"{mean:.1f} ± {sd:.1f} nm", va="center", fontsize=8)
    ax.set(yticks=range(3), yticklabels=labels, xlim=(28,55), xlabel="Vesicle membrane-core diameter (nm)")
    ax.invert_yaxis()
    panel(ax, "a", "Diameter scale · mean ± across-vesicle SD")
    save(fig, "supp_experimental_context")


def main() -> None:
    theme()
    if not (OUT / "validation.json").is_file() or not (OUT / "inference_checks.json").is_file():
        raise FileNotFoundError("Run python -m analysis.run and python -m analysis.inference_checks first")
    runs = run_table()
    connectivity(runs)
    structure(runs)
    robustness()
    density()
    rdf()
    machine_learning()
    geometry_numerics()
    snapshots()
    experimental_context()
    print("ARTICLE_TWO_PAPER_FIGURES_OK count=9")


if __name__ == "__main__":
    main()
