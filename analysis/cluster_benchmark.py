"""Recompute a vesicle-weighted >50 component benchmark from saved frames.

Fernandez-Busnadiego et al. (JCB 2010, Fig. S3) use visible connectors,
whereas these model components use mixed-agent geometric-proximity edges.
The resulting percentages are descriptive side-by-side quantities, not a
measurement-matched experimental validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .graph import _connected_components_from_pairs
from .run import MANIFEST, OUT, ROOT, check_manifest


def frame_benchmark(group: h5py.Group, shell_nm: float = 2.0) -> dict[str, float]:
    positions = group["positions"][:]
    radii = group["radii"][:]
    types = group["types"][:]
    candidates = cKDTree(positions).query_pairs(
        2.0 * float(np.max(radii)) + 6.0, output_type="ndarray"
    )
    if len(candidates):
        i, j = candidates[:, 0], candidates[:, 1]
        gap = np.linalg.norm(positions[i] - positions[j], axis=1) - radii[i] - radii[j]
        edges = candidates[gap <= shell_nm]
    else:
        edges = np.empty((0, 2), dtype=np.int64)
    n_components, labels, sizes = _connected_components_from_pairs(
        len(types), edges[:, 0], edges[:, 1]
    )
    vesicle_mask = types == 2
    n_vesicles = int(np.sum(vesicle_mask))
    assert n_vesicles == 200
    vesicle_counts = np.bincount(labels[vesicle_mask], minlength=n_components)
    return {
        "n_components_all": int(n_components),
        "largest_component_vesicle_fraction": float(
            vesicle_counts[int(np.argmax(sizes))] / n_vesicles
        ),
        "vesicle_recruited_fraction": float(
            np.sum(sizes[labels[vesicle_mask]] >= 2) / n_vesicles
        ),
        "vesicles_in_components_gt50_fraction": float(
            np.sum(vesicle_counts[vesicle_counts > 50]) / n_vesicles
        ),
        "vesicles_in_components_2_to_50_fraction": float(
            np.sum(vesicle_counts[(vesicle_counts >= 2) & (vesicle_counts <= 50)])
            / n_vesicles
        ),
        "vesicles_alone_in_mixed_components_fraction": float(
            np.sum(vesicle_counts[vesicle_counts == 1]) / n_vesicles
        ),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check_manifest(manifest)
    rows = []
    for run in manifest["runs"]:
        reference = pd.read_csv(OUT / run["run_id"] / "frames.csv").set_index("simulation_step")
        with h5py.File(ROOT / run["trajectory"], "r") as trajectory:
            for step in range(28200, 40001, 200):
                value = frame_benchmark(trajectory[f"step_{step}"])
                expected = reference.loc[step]
                for field in (
                    "n_components_all",
                    "largest_component_vesicle_fraction",
                    "vesicle_recruited_fraction",
                ):
                    if not np.isclose(value[field], expected[field], atol=1e-12):
                        raise AssertionError(f"Graph mismatch: {run['run_id']} step={step} {field}")
                rows.append({
                    "run_id": run["run_id"],
                    "condition": run["condition_id"],
                    "seed": run["seed"],
                    "step": step,
                    **value,
                })
        print(f"BENCHMARK_VERIFIED {run['run_id']}", flush=True)
    frame_table = pd.DataFrame(rows)
    run_table = frame_table.groupby(["run_id", "condition", "seed"], as_index=False).mean(numeric_only=True)
    run_table = run_table.drop(columns=["step"])
    OUT.mkdir(exist_ok=True)
    frame_table.to_csv(OUT / "cluster_benchmark_frames.csv", index=False)
    run_table.to_csv(OUT / "cluster_benchmark_runs.csv", index=False)
    summary = run_table.groupby("condition")["vesicles_in_components_gt50_fraction"].agg(["mean", "min", "max"])
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    print("BENCHMARK_COMPLETE frames=720 runs=12 shell_nm=2 threshold_vesicles_gt=50", flush=True)


if __name__ == "__main__":
    main()
