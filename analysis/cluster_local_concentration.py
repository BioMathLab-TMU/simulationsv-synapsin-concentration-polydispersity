"""Recompute vesicle-associated local synapsin concentrations from saved frames.

The 2-nm graph identifies vesicles in mixed synapsin/vesicle components.
At each such vesicle whose complete probe ball is inside the compartment,
count nearby synapsin centers and divide by the geometric probe-ball volume.
This is an operational local number concentration, not a fitted phase density.
The 60 late frames are averaged within a run; the three seed blocks are the
replication units. Radius-distribution arms are averaged within each block.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


AVOGADRO = 6.02214076e23
RADII_NM = (35.0, 50.0, 70.0)
SHELL_NM = 2.0


def main(root: Path, output: Path) -> None:
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == 12
    from analysis.mesoscale import Domain
    from analysis.graph import _connected_components_from_pairs

    rows = []
    for run in manifest["runs"]:
        raw_path = root / run["trajectory"]
        with raw_path.open("rb") as stream:
            raw_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if raw_hash != run["published_sha256"]:
            raise ValueError(f"Trajectory hash mismatch: {run['run_id']}")
        config_path = root / run["config"]
        with config_path.open("rb") as stream:
            config_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if config_hash != run["config_sha256"]:
            raise ValueError(f"Configuration hash mismatch: {run['run_id']}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        domain = Domain.from_dashboard(config)
        if not math.isclose(domain.volume_nm3, 1e8, rel_tol=1e-9):
            raise ValueError(f"Unexpected compartment volume: {run['run_id']}")
        frame_rows = {r: [] for r in RADII_NM}
        with h5py.File(raw_path, "r") as h5:
            steps = sorted((int(k.rsplit("_", 1)[1]) for k in h5 if k.startswith("step_")))
            if steps != list(range(200, 40001, 200)):
                raise ValueError(f"Incomplete frame schedule: {run['run_id']}")
            for step in steps[-60:]:
                group = h5[f"step_{step}"]
                positions = np.asarray(group["positions"][:], dtype=float)
                types = np.asarray(group["types"][:], dtype=int)
                radii = np.asarray(group["radii"][:], dtype=float)
                if not np.isfinite(positions).all() or not np.isin(types, (1, 2)).all():
                    raise ValueError(f"Bad frame: {run['run_id']} {step}")
                candidates = cKDTree(positions).query_pairs(2 * float(radii.max()) + SHELL_NM, output_type="ndarray")
                if len(candidates):
                    i, j = candidates[:, 0], candidates[:, 1]
                    gaps = np.linalg.norm(positions[i] - positions[j], axis=1) - radii[i] - radii[j]
                    edges = candidates[gaps <= SHELL_NM]
                else:
                    edges = np.empty((0, 2), dtype=int)
                n_components, labels, _ = _connected_components_from_pairs(len(types), edges[:, 0], edges[:, 1])
                syn_counts = np.bincount(labels[types == 1], minlength=n_components)
                ves_counts = np.bincount(labels[types == 2], minlength=n_components)
                ves_pos = positions[types == 2]
                ves_labels = labels[types == 2]
                mixed_vesicle = (syn_counts[ves_labels] > 0) & (ves_counts[ves_labels] > 0)
                syn_tree = cKDTree(positions[types == 1])
                syn_labels = labels[types == 1]
                clearance = domain.clearance(ves_pos)
                for radius in RADII_NM:
                    eligible = mixed_vesicle & (clearance >= radius)
                    if not eligible.any():
                        raise ValueError(f"No eligible vesicles: {run['run_id']} {step} {radius}")
                    neighbor_ids = syn_tree.query_ball_point(ves_pos[eligible], radius)
                    counts = np.asarray([len(ids) for ids in neighbor_ids], dtype=float)
                    interior_all = clearance >= radius
                    all_counts = np.asarray(syn_tree.query_ball_point(ves_pos[interior_all], radius, return_length=True), dtype=float)
                    nonmixed_eligible = (~mixed_vesicle) & interior_all
                    nonmixed_counts = np.asarray(syn_tree.query_ball_point(ves_pos[nonmixed_eligible], radius, return_length=True), dtype=float)
                    same_component_counts = np.asarray([
                        np.count_nonzero(syn_labels[ids] == label)
                        for ids, label in zip(neighbor_ids, ves_labels[eligible])
                    ], dtype=float)
                    volume_nm3 = 4 * math.pi * radius**3 / 3
                    c_uM = counts * 1e30 / (AVOGADRO * volume_nm3)
                    c_same_uM = same_component_counts * 1e30 / (AVOGADRO * volume_nm3)
                    frame_rows[radius].append((len(counts), float(c_uM.mean()), float(np.median(c_uM)), float((c_uM == 0).mean()), float(mixed_vesicle.mean()), float(c_same_uM.mean()), float(all_counts.mean() * 1e30 / (AVOGADRO * volume_nm3)), len(nonmixed_counts), float(nonmixed_counts.mean() * 1e30 / (AVOGADRO * volume_nm3)) if len(nonmixed_counts) else math.nan))
        for radius, values in frame_rows.items():
            arr = np.asarray(values)
            rows.append({
                "run_id": run["run_id"], "condition": run["condition_id"], "seed": run["seed"],
                "bulk_synapsin_uM": run["synapsin_concentration_uM"], "probe_radius_nm": radius,
                "late_frames": len(values), "mean_eligible_vesicles_per_frame": float(arr[:, 0].mean()),
                "mean_mixed_vesicle_fraction": float(arr[:, 4].mean()),
                "local_synapsin_uM": float(arr[:, 1].mean()),
                "same_component_local_synapsin_uM": float(arr[:, 5].mean()),
                "all_vesicle_local_synapsin_uM": float(arr[:, 6].mean()),
                "mean_nonmixed_vesicles_per_frame": float(arr[:, 7].mean()),
                "nonmixed_local_synapsin_uM": float(np.nanmean(arr[:, 8])) if np.isfinite(arr[:, 8]).any() else math.nan,
                "median_local_synapsin_uM_frame_average": float(arr[:, 2].mean()),
                "zero_synapsin_neighborhood_fraction": float(arr[:, 3].mean()),
            })
        print(f"CLUSTER_PROBE {run['run_id']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for concentration in (10, 60):
        for radius in RADII_NM:
            blocked = [np.mean([
                row["local_synapsin_uM"] for row in rows
                if row["bulk_synapsin_uM"] == concentration
                and row["probe_radius_nm"] == radius
                and row["seed"] == seed
            ]) for seed in (9210001, 9210002, 9210003)]
            print(f"SUMMARY bulk={concentration} radius={radius} blocked_mean={np.mean(blocked):.3f} blocked_min={min(blocked):.3f} blocked_max={max(blocked):.3f} ratio_to_bulk={np.mean(blocked) / concentration:.3f}")
    print(f"WROTE {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=default_root / "outputs/cluster_local_concentration.csv")
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
