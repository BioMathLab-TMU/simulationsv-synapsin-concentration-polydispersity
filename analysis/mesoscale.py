"""Interior density and radius-aware geometric-null RDF diagnostics."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .graph import _section
CODE_VERSION = "1.0.0"
DEFAULTS = {"probe_radii_nm": [35., 50., 70.], "spatial_probes": 4096,
            "late_frames_maximum": 6, "ideal_gas_replicas": 4,
            "rdf_maximum_nm": 120., "rdf_bin_width_nm": 4.,
            "late_window_fraction": .30, "random_seed": 1729}
def resolve_dataset_names(group):
    return {key: key if key in group else None for key in ("positions", "radii", "types")}
@dataclass(frozen=True)
class Domain:
    shape: int
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_dashboard(cls, dashboard):
        env = _section(dashboard, "environment")
        lengths = np.array([env["length"], env["width"], env["height"]], dtype=float)
        steps = np.asarray(env["steps"], dtype=float)
        out = float(env["out"])
        if lengths.shape != (3,) or steps.shape != (3,) or not np.isfinite(lengths).all() or not np.isfinite(steps).all() or (lengths <= 0).any() or (steps <= 0).any() or not math.isfinite(out) or out < 0 or not out.is_integer():
            raise ValueError("Domain requires recorded positive lengths/steps and integer out")
        counts = np.ceil(lengths / steps - 1e-9)
        # Exactly the analytic bounds used by GF::checkBoundary.
        lower = (out + 1) * steps
        upper = (out + counts + 1) * steps
        shape = int(env["shape"])
        if shape not in (0, 2, 3, 4):
            raise ValueError("Imported/unknown domain requires explicit geometry; no fabricated boundaries")
        return cls(shape, lower, upper)

    @property
    def center(self):
        value = (self.lower + self.upper) / 2
        if self.shape == 3:
            value[2] = self.upper[2]
        return value

    @property
    def radius(self):
        return float(np.min((self.upper - self.lower)[:2]) / 2)

    @property
    def volume_nm3(self):
        if self.shape == 0:
            return float(np.prod(self.upper - self.lower))
        if self.shape == 4:
            return float(np.pi * self.radius**2 * (self.upper[2] - self.lower[2]))
        return float(4 * np.pi / 3 * self.radius**3 / (2 if self.shape == 3 else 1))

    def clearance(self, positions):
        x = np.asarray(positions, dtype=float)
        planar = np.min(np.minimum(x - self.lower, self.upper - x), axis=1)
        if self.shape == 0:
            return planar
        offset = x - self.center
        if self.shape == 4:
            radial = self.radius - np.linalg.norm(offset[:, :2], axis=1)
            return np.minimum(radial, np.minimum(x[:, 2] - self.lower[2], self.upper[2] - x[:, 2]))
        radial = self.radius - np.linalg.norm(offset, axis=1)
        return np.minimum(radial, self.upper[2] - x[:, 2]) if self.shape == 3 else radial

    def sample(self, radii, rng):
        """Uniform centers in each particle's radius-accessible analytic domain."""
        radii = np.asarray(radii, dtype=float).reshape(-1)
        if not np.isfinite(radii).all() or (radii < 0).any():
            raise ValueError("Sampling radii must be finite and nonnegative")
        if self.shape == 0:
            spans = self.upper - self.lower - 2 * radii[:, None]
            if (spans <= 0).any():
                raise ValueError("Particle does not fit in domain")
            return self.lower + radii[:, None] + rng.random((len(radii), 3)) * spans
        accessible = self.radius - radii
        if (accessible <= 0).any():
            raise ValueError("Particle does not fit in curved domain")
        if self.shape == 4:
            angle = rng.uniform(0, 2 * np.pi, len(radii))
            rho = np.sqrt(rng.random(len(radii))) * accessible
            result = np.column_stack([rho * np.cos(angle), rho * np.sin(angle), np.zeros(len(radii))]) + self.center
            span = self.upper[2] - self.lower[2] - 2 * radii
            if (span <= 0).any():
                raise ValueError("Particle does not fit axially")
            result[:, 2] = self.lower[2] + radii + rng.random(len(radii)) * span
            return result
        result = np.empty((len(radii), 3))
        remaining = np.arange(len(radii))
        attempts = 0
        while len(remaining):
            direction = rng.normal(size=(len(remaining), 3))
            direction /= np.linalg.norm(direction, axis=1)[:, None]
            if self.shape == 3:
                direction[:, 2] = -np.abs(direction[:, 2])
            candidate = self.center + direction * (rng.random(len(remaining))**(1 / 3) * accessible[remaining])[:, None]
            accepted = self.clearance(candidate) >= radii[remaining] - 1e-10
            result[remaining[accepted]] = candidate[accepted]
            remaining = remaining[~accepted]
            attempts += 1
            if attempts > 10000:
                raise ValueError("Insufficient radius-accessible domain")
        return result

def pair_histograms(positions, types, edges):
    """Pair counts in (r_low,r_high], without materializing O(N^2) edges."""
    positions, types = np.asarray(positions), np.asarray(types)
    trees = {t: cKDTree(positions[types == t]) for t in (1, 2)}
    result = {}
    for name, a, b in (("SS", 1, 1), ("SV", 1, 2), ("VV", 2, 2)):
        cumulative = np.asarray(trees[a].count_neighbors(trees[b], edges), dtype=float)
        if a == b:
            cumulative = (cumulative - np.sum(types == a)) / 2
        result[name] = np.diff(cumulative)
    return result

def probe_densities(positions, types, probes, domain, radius):
    probes = np.asarray(probes)[domain.clearance(probes) >= radius]
    if not len(probes):
        raise ValueError("No fully interior probes for this analysis scale")
    result = {}
    for name, t in (("synapsin", 1), ("vesicle", 2)):
        tree = cKDTree(np.asarray(positions)[np.asarray(types) == t])
        counts = tree.query_ball_point(probes, radius, return_length=True, workers=1)
        result[name] = np.asarray(counts, dtype=float) / (4 * np.pi / 3 * radius**3) * 1e9
    return result, len(probes)

def late_window_drift(frames):
    ordered = frames.sort_values("frame_index")
    late = ordered.tail(max(1, math.ceil(len(ordered) * 0.30)))
    rows = []
    for metric in ("largest_component_vesicle_fraction", "vesicle_recruited_fraction", "synapsin_recruited_fraction"):
        if metric not in late:
            continue
        if len(late) < 4:
            rows.append({"metric": metric, "status": "insufficient_late_frames", "late_half_difference": np.nan})
            continue
        midpoint = len(late) // 2
        first = pd.to_numeric(late.iloc[:midpoint][metric], errors="coerce").to_numpy(float)
        last = pd.to_numeric(late.iloc[midpoint:][metric], errors="coerce").to_numpy(float)
        valid = np.isfinite(first).all() and np.isfinite(last).all()
        rows.append({"metric": metric, "status": "descriptive_only" if valid else "invalid_values", "late_half_difference": float(np.mean(last) - np.mean(first)) if valid else np.nan})
    return pd.DataFrame(rows)

def analyze_run(manifest, config=None):
    config = dict(DEFAULTS if config is None else config)
    dashboard = json.loads(Path(manifest["paths"]["dashboard"]).read_text(encoding="utf-8-sig"))
    domain = Domain.from_dashboard(dashboard)
    rng = np.random.default_rng(config["random_seed"])
    probes = domain.sample(np.zeros(config["spatial_probes"]), rng)
    edges = np.arange(0, config["rdf_maximum_nm"] + config["rdf_bin_width_nm"] / 2, config["rdf_bin_width_nm"])
    density_rows, hist_rows, rdf_rows = [], [], []
    with h5py.File(manifest["paths"]["h5"], "r") as h5:
        steps = sorted([name for name in h5 if name.startswith("step_")], key=lambda name: int(name.rsplit("_", 1)[-1]))
        late_count = max(1, math.ceil(len(steps) * config["late_window_fraction"]))
        selected = np.unique(np.linspace(len(steps) - late_count, len(steps) - 1, min(late_count, config["late_frames_maximum"]), dtype=int))
        reference = h5[steps[0]]
        names = resolve_dataset_names(reference)
        radii = np.asarray(reference[names["radii"]][:], dtype=float).reshape(-1)
        types = np.asarray(reference[names["types"]][:], dtype=int).reshape(-1)
        if len(types) == 0 or not np.isin(types, [1, 2]).all():
            raise ValueError("RDF/density requires recorded supported species")
        null_counts = {name: np.zeros(len(edges) - 1) for name in ("SS", "SV", "VV")}
        for _ in range(config["ideal_gas_replicas"]):
            null = pair_histograms(domain.sample(radii, rng), types, edges)
            for name in null_counts:
                null_counts[name] += null[name] / config["ideal_gas_replicas"]
        actual_counts = {name: np.zeros(len(edges) - 1) for name in null_counts}
        collected = {}
        # Histogram zero separately; positive rho/rho_global uses log bins.
        positive_edges = np.geomspace(1e-3, 1e3, 49)
        for frame_index in selected:
            group = h5[steps[int(frame_index)]]
            names = resolve_dataset_names(group)
            positions = np.asarray(group[names["positions"]][:], dtype=float)
            current_types = np.asarray(group[names["types"]][:], dtype=int).reshape(-1)
            current_radii = np.asarray(group[names["radii"]][:], dtype=float).reshape(-1)
            # Row ordering may change: species/radius multisets, not ID ordering,
            # are relevant to snapshot RDF and fixed-radius geometry.
            if not np.array_equal(np.sort(current_types), np.sort(types)) or any(not np.allclose(np.sort(current_radii[current_types == t]), np.sort(radii[types == t]), rtol=0, atol=1e-6) for t in (1, 2)):
                raise ValueError("Particle population or radius distribution changed")
            if not np.isfinite(positions).all() or (domain.clearance(positions) < current_radii - 1e-4).any():
                raise ValueError("Saved positions violate recorded radius-aware domain")
            actual = pair_histograms(positions, current_types, edges)
            for name in actual_counts:
                actual_counts[name] += actual[name] / len(selected)
            for radius in config["probe_radii_nm"]:
                values, retained = probe_densities(positions, current_types, probes, domain, radius)
                for species, samples in values.items():
                    collected.setdefault((radius, species, retained), []).append(samples)
        for (radius, species, retained), chunks in collected.items():
            values = np.concatenate(chunks)
            species_type = 1 if species == "synapsin" else 2
            global_density = np.sum(types == species_type) / domain.volume_nm3 * 1e9
            ratio = values / global_density if global_density > 0 else np.full(len(values), np.nan)
            density_rows.append({"probe_radius_nm": radius, "species": species, "interior_probe_count": retained, "interior_probe_fraction": retained / len(probes), "selected_frame_count": len(selected), "global_number_density_per_um3": global_density, "mean_number_density_per_um3": float(np.mean(values)), "sd_number_density_per_um3": float(np.std(values)), "q05_number_density_per_um3": float(np.quantile(values, .05)), "q50_number_density_per_um3": float(np.quantile(values, .50)), "q95_number_density_per_um3": float(np.quantile(values, .95)), "zero_density_fraction": float(np.mean(values == 0))})
            if global_density <= 0:
                continue
            hist, _ = np.histogram(ratio[ratio > 0], positive_edges)
            # Explicit zero, underflow, overflow guarantee probability mass preservation.
            bins = [(-1.0, 0.0, "zero", np.sum(ratio == 0)), (0.0, positive_edges[0], "underflow", np.sum((ratio > 0) & (ratio < positive_edges[0])))]
            bins += [(a, b, "positive", n) for a, b, n in zip(positive_edges[:-1], positive_edges[1:], hist)]
            bins += [(positive_edges[-1], np.inf, "overflow", np.sum(ratio > positive_edges[-1]))]
            for index, (low, high, kind, count) in enumerate(bins):
                hist_rows.append({"probe_radius_nm": radius, "species": species, "bin_index": index, "ratio_low": low, "ratio_high": high, "bin_kind": kind, "probability_mass": float(count / len(values))})
        for name, counts in actual_counts.items():
            baseline = null_counts[name]
            for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                rdf_rows.append({"pair": name, "r_low_nm": low, "r_high_nm": high, "r_mid_nm": (low + high) / 2, "mean_observed_pairs": counts[index], "mean_ideal_gas_pairs": baseline[index], "rdf_geometric_null_ratio": counts[index] / baseline[index] if baseline[index] >= 5 else np.nan, "baseline_expected_pairs_at_least_five": bool(baseline[index] >= 5)})
    provenance = {"code_version": CODE_VERSION, "analysis_config": config, "selected_frame_indices": selected.tolist(), "domain_shape": domain.shape, "domain_lower_nm": domain.lower.tolist(), "domain_upper_nm": domain.upper.tolist(), "geometric_compartment_volume_nm3": domain.volume_nm3, "claim_scope": "descriptive_density_and_radius_aware_ideal_gas_RDF_not_LLPS_proof", "density_scope": "uniform_spatial_probes_conditional_on_fully_interior_balls", "null_does_not_correct": ["excluded_volume", "interactions", "equilibration"], "raw_data_modified": False}
    return pd.DataFrame(density_rows), pd.DataFrame(hist_rows), pd.DataFrame(rdf_rows), provenance
