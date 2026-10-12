"""Reviewed non-periodic surface-gap graph calculations."""
from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
BOUNDARY_CONDITION = "closed_reflecting_no_flux"
OUTPUT_SCHEMA_VERSION = "concentration-graphs-1.0"
def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())

def _first_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, list):
        value = value[0] if value else {}
    return dict(value) if isinstance(value, dict) else {}

def _section(data: Mapping[str, Any], name: str) -> Dict[str, Any]:
    normalized = {_normalized_key(str(key)): value for key, value in data.items()}
    return _first_mapping(normalized.get(_normalized_key(name), {}))

def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, (list, tuple, np.ndarray)) and np.size(value) == 1:
        value = np.asarray(value).reshape(-1)[0]
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else math.nan

def _drift_proxy(
    force_magnitude: np.ndarray,
    radii: np.ndarray,
    types: np.ndarray,
    eta: Optional[float],
    drag_syn: Optional[float],
    drag_ves: Optional[float],
    synapsin_type: int,
    vesicle_type: int,
) -> np.ndarray:
    if eta is None or eta <= 0:
        return np.full(len(radii), np.nan, dtype=np.float64)
    drag = np.full(len(radii), np.nan, dtype=np.float64)
    syn = types == synapsin_type
    ves = types == vesicle_type
    if drag_syn is not None and drag_syn > 0:
        drag[syn] = 6.0 * np.pi * eta * radii[syn] * drag_syn
    if drag_ves is not None and drag_ves > 0:
        drag[ves] = 6.0 * np.pi * eta * radii[ves] * drag_ves
    result = np.full(len(radii), np.nan, dtype=np.float64)
    valid = np.isfinite(force_magnitude) & np.isfinite(drag) & (drag > 0)
    result[valid] = force_magnitude[valid] / drag[valid]
    return result

def _frame_context(task: Mapping[str, Any]) -> Dict[str, Any]:
    simulation_step = int(task["simulation_step"])
    dt = task.get("time_step_ms")
    recorded_time = _finite_number(task.get("physical_time_ms"))
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_config_id": task["analysis_config_id"],
        "Run_ID": task["run_id"],
        "Condition_ID": task["condition_id"],
        "seed": task.get("seed"),
        "seed_status": task["seed_status"],
        "frame_index": int(task["frame_index"]),
        "simulation_step": simulation_step,
        "time_step": simulation_step,
        "time_ms": (
            recorded_time
            if recorded_time is not None
            else float(simulation_step * dt) if dt is not None else math.nan
        ),
        "time_source": (
            "hdf5_physical_time_ms"
            if recorded_time is not None
            else "simulation_step_x_time_step_ms" if dt is not None else "simulation_step_only"
        ),
        "boundary_condition": BOUNDARY_CONDITION,
        "particle_id_source": task["particle_id_source"],
    }

def _type_assortativity(e_ss: int, e_sv: int, e_vv: int) -> float:
    total = e_ss + e_sv + e_vv
    if total == 0:
        return math.nan
    a_syn = (2.0 * e_ss + e_sv) / (2.0 * total)
    a_ves = (2.0 * e_vv + e_sv) / (2.0 * total)
    expected_trace = a_syn * a_syn + a_ves * a_ves
    denominator = 1.0 - expected_trace
    if denominator <= np.finfo(float).eps:
        return math.nan
    observed_trace = (e_ss + e_vv) / total
    return float((observed_trace - expected_trace) / denominator)

def _connected_components_from_pairs(
    count: int, pair_i: np.ndarray, pair_j: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    if len(pair_i):
        rows = np.concatenate([pair_i, pair_j])
        cols = np.concatenate([pair_j, pair_i])
        adjacency = csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(count, count)
        )
    else:
        adjacency = csr_matrix((count, count), dtype=np.uint8)
    n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=n_components)
    return int(n_components), labels, sizes

def _weight_average_finite_cluster_size(component_sizes: np.ndarray) -> float:
    """Susceptibility-like finite-cluster size with one largest component removed."""

    sizes = np.asarray(component_sizes, dtype=np.float64)
    if sizes.size <= 1:
        return 0.0
    finite = np.delete(sizes, int(np.argmax(sizes)))
    denominator = float(np.sum(finite))
    return float(np.sum(finite * finite) / denominator) if denominator > 0 else 0.0

def _analyze_frame_arrays(
    task: Mapping[str, Any],
    positions: np.ndarray,
    types: np.ndarray,
    radii: np.ndarray,
    particle_ids: np.ndarray,
    force_magnitude: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    positions = np.asarray(positions, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64).reshape(-1)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    particle_ids = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
    force_magnitude = np.asarray(force_magnitude, dtype=np.float64).reshape(-1)
    count = len(positions)
    if positions.shape != (count, 3) or any(len(array) != count for array in (types, radii, particle_ids, force_magnitude)):
        raise ValueError(f"Inconsistent arrays in {task['step_name']}")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError(f"Invalid positions/radii in {task['step_name']}")
    if len(np.unique(particle_ids)) != count:
        raise ValueError(f"Duplicate particle IDs in {task['step_name']}")

    synapsin_type = int(task["synapsin_type"])
    vesicle_type = int(task["vesicle_type"])
    observed_types = set(int(value) for value in np.unique(types))
    allowed_types = {synapsin_type, vesicle_type}
    if not observed_types.issubset(allowed_types):
        raise ValueError(
            f"Unexpected particle type codes {sorted(observed_types - allowed_types)} in {task['step_name']}; "
            f"configured synapsin={synapsin_type}, vesicle={vesicle_type}"
        )

    force_over_drag = _drift_proxy(
        force_magnitude,
        radii,
        types,
        task.get("eta"),
        task.get("drag_syn"),
        task.get("drag_ves"),
        synapsin_type,
        vesicle_type,
    )
    if task.get("boundary_condition", BOUNDARY_CONDITION) != BOUNDARY_CONDITION:
        raise ValueError("Only direct non-periodic geometry is valid for this project")
    tree = cKDTree(positions)
    shells_nm = sorted(float(value) for value in task["shells_nm"])
    primary_shell_nm = float(task["primary_shell_nm"])
    search_radius = 2.0 * float(np.max(radii)) + max(shells_nm)
    candidate_pairs = tree.query_pairs(search_radius, output_type="ndarray")
    if candidate_pairs.size:
        pair_i = candidate_pairs[:, 0].astype(np.int64, copy=False)
        pair_j = candidate_pairs[:, 1].astype(np.int64, copy=False)
        center_distance = np.linalg.norm(positions[pair_i] - positions[pair_j], axis=1)
    else:
        pair_i = np.empty(0, dtype=np.int64)
        pair_j = np.empty(0, dtype=np.int64)
        center_distance = np.empty(0, dtype=np.float64)
    pair_type_i = types[pair_i]
    pair_type_j = types[pair_j]
    pair_ss = (pair_type_i == synapsin_type) & (pair_type_j == synapsin_type)
    pair_vv = (pair_type_i == vesicle_type) & (pair_type_j == vesicle_type)
    pair_sv = ((pair_type_i == synapsin_type) & (pair_type_j == vesicle_type)) | (
        (pair_type_i == vesicle_type) & (pair_type_j == synapsin_type)
    )
    surface_distance = center_distance - radii[pair_i] - radii[pair_j]
    primary_edge = surface_distance <= primary_shell_nm
    edge_i = pair_i[primary_edge]
    edge_j = pair_j[primary_edge]
    edge_surface_distance = surface_distance[primary_edge]
    edge_type_i = types[edge_i]
    edge_type_j = types[edge_j]
    edge_ss = (edge_type_i == synapsin_type) & (edge_type_j == synapsin_type)
    edge_vv = (edge_type_i == vesicle_type) & (edge_type_j == vesicle_type)
    edge_sv = ((edge_type_i == synapsin_type) & (edge_type_j == vesicle_type)) | (
        (edge_type_i == vesicle_type) & (edge_type_j == synapsin_type)
    )
    overlap_depth = np.maximum(0.0, -edge_surface_distance)
    core_allowance = float(task["core_allowed_overlap_nm"])
    corona_thickness = float(task["vesicle_corona_thickness_nm"])
    pair_allowance = (
        core_allowance
        + corona_thickness * (edge_type_i == vesicle_type)
        + corona_thickness * (edge_type_j == vesicle_type)
    )
    allowance_excess = np.maximum(0.0, overlap_depth - pair_allowance)
    projection_tolerance = float(task["overlap_projection_residual_tolerance_nm"])
    analysis_slack_nm = 1e-6
    excess_beyond_tolerance = np.maximum(0.0, allowance_excess - projection_tolerance)
    excess_beyond_tolerance[excess_beyond_tolerance <= analysis_slack_nm] = 0.0

    n_components, component_labels, component_sizes = _connected_components_from_pairs(
        count, edge_i, edge_j
    )
    primary_min_size = int(task["min_cluster_size"])
    valid_components = np.flatnonzero(component_sizes >= primary_min_size)
    base = {**_frame_context(task), "analysis_shell_nm": primary_shell_nm}

    sensitivity_rows: List[Dict[str, Any]] = []
    for shell_nm in shells_nm:
        shell_edge = surface_distance <= shell_nm
        shell_i = pair_i[shell_edge]
        shell_j = pair_j[shell_edge]
        shell_n, shell_labels, shell_sizes = _connected_components_from_pairs(count, shell_i, shell_j)
        shell_type_i = types[shell_i]
        shell_type_j = types[shell_j]
        shell_ss = (shell_type_i == synapsin_type) & (shell_type_j == synapsin_type)
        shell_vv = (shell_type_i == vesicle_type) & (shell_type_j == vesicle_type)
        shell_sv = ~shell_ss & ~shell_vv
        shell_largest = int(shell_sizes.max()) if len(shell_sizes) else 0
        shell_recruited = shell_sizes[shell_labels] >= 2
        shell_largest_mask = shell_labels == int(np.argmax(shell_sizes))
        shell_syn = types == synapsin_type
        shell_ves = types == vesicle_type
        sensitivity_rows.append(
            {
                **{key: base[key] for key in (
                    "schema_version", "analysis_config_id", "Run_ID", "Condition_ID", "seed",
                    "seed_status", "frame_index", "simulation_step", "time_step", "time_ms"
                )},
                "analysis_shell_nm": float(shell_nm),
                "n_components_all": int(shell_n),
                "monomer_fraction": float(np.sum(shell_sizes == 1) / count),
                "largest_component_size": shell_largest,
                "largest_component_fraction": float(shell_largest / count),
                "largest_component_vesicle_fraction": float(np.sum(shell_largest_mask & shell_ves) / np.sum(shell_ves)) if np.any(shell_ves) else math.nan,
                "vesicle_recruited_fraction": float(np.sum(shell_recruited & shell_ves) / np.sum(shell_ves)) if np.any(shell_ves) else math.nan,
                "synapsin_recruited_fraction": float(np.sum(shell_recruited & shell_syn) / np.sum(shell_syn)) if np.any(shell_syn) else math.nan,
                "weight_average_finite_cluster_size": _weight_average_finite_cluster_size(shell_sizes),
                "adjacency_edges_SS": int(np.sum(shell_ss)),
                "adjacency_edges_SV": int(np.sum(shell_sv)),
                "adjacency_edges_VV": int(np.sum(shell_vv)),
            }
        )

    valid_particle_count = int(component_sizes[valid_components].sum()) if len(valid_components) else 0
    largest_size = int(component_sizes.max()) if len(component_sizes) else 0
    recruited_mask = component_sizes[component_labels] >= 2
    syn_all = types == synapsin_type
    ves_all = types == vesicle_type
    n_syn_all = int(np.sum(syn_all))
    n_ves_all = int(np.sum(ves_all))
    largest_label = int(np.argmax(component_sizes)) if len(component_sizes) else -1
    largest_mask = component_labels == largest_label
    frame_summary: Dict[str, Any] = {
        **base,
        "n_particles": count,
        "n_synapsins": n_syn_all,
        "n_vesicles": n_ves_all,
        "n_components_all": int(n_components),
        "n_valid_clusters": int(len(valid_components)),
        "valid_clustered_fraction": float(valid_particle_count / count),
        "largest_component_size": largest_size,
        "largest_component_fraction": float(largest_size / count),
        "monomer_fraction": float(np.sum(component_sizes == 1) / count),
        "weight_average_finite_cluster_size": _weight_average_finite_cluster_size(component_sizes),
        "vesicle_recruited_fraction": float(np.sum(recruited_mask & ves_all) / n_ves_all) if n_ves_all else math.nan,
        "synapsin_recruited_fraction": float(np.sum(recruited_mask & syn_all) / n_syn_all) if n_syn_all else math.nan,
        "largest_component_vesicle_fraction": float(np.sum(largest_mask & ves_all) / n_ves_all) if n_ves_all else math.nan,
        "mean_valid_cluster_size": float(np.mean(component_sizes[valid_components])) if len(valid_components) else 0.0,
        "adjacency_edges_SS": int(np.sum(edge_ss)),
        "adjacency_edges_SV": int(np.sum(edge_sv)),
        "adjacency_edges_VV": int(np.sum(edge_vv)),
        "mean_adjacency_degree": float(2.0 * len(edge_i) / count),
        "overlap_fraction_among_adjacency_edges": (
            float(np.mean(edge_surface_distance < 0.0)) if len(edge_surface_distance) else 0.0
        ),
        "overlap_pair_count": int(np.sum(overlap_depth > 0.0)),
        "max_outer_overlap_depth_nm": float(np.max(overlap_depth)) if len(overlap_depth) else 0.0,
        "projection_residual_tolerance_nm": projection_tolerance,
        "overlap_pairs_above_allowed_plus_tolerance": int(np.sum(excess_beyond_tolerance > 0.0)),
        "max_pair_allowance_excess_nm": float(np.max(allowance_excess)) if len(allowance_excess) else 0.0,
        "max_excess_beyond_projection_tolerance_nm": (
            float(np.max(excess_beyond_tolerance)) if len(excess_beyond_tolerance) else 0.0
        ),
    }

    clusters: List[Dict[str, Any]] = []
    for component in valid_components:
        member_index = np.flatnonzero(component_labels == component)
        member_mask = np.zeros(count, dtype=bool)
        member_mask[member_index] = True
        internal_edge = member_mask[edge_i] & member_mask[edge_j]
        e_ss = int(np.sum(internal_edge & edge_ss))
        e_vv = int(np.sum(internal_edge & edge_vv))
        e_sv = int(np.sum(internal_edge & edge_sv))
        e_total = e_ss + e_vv + e_sv

        c_positions = positions[member_index].copy()
        c_types = types[member_index]
        c_radii = radii[member_index]
        c_force = force_magnitude[member_index]
        c_drift = force_over_drag[member_index]
        syn_mask = c_types == synapsin_type
        ves_mask = c_types == vesicle_type
        n_syn = int(np.sum(syn_mask))
        n_ves = int(np.sum(ves_mask))
        cluster_size = len(member_index)

        center = np.mean(c_positions, axis=0)
        centered = c_positions - center
        gyration = centered.T @ centered / cluster_size
        eigenvalues = np.sort(np.clip(np.linalg.eigvalsh(gyration), 0.0, None))[::-1]
        l1, l2, l3 = eigenvalues
        rg_sq = float(np.sum(eigenvalues))
        radius_of_gyration = math.sqrt(rg_sq)
        shape_anisotropy = (
            1.0 - 3.0 * float(l1 * l2 + l2 * l3 + l3 * l1) / (rg_sq * rg_sq)
            if rg_sq > 0
            else 0.0
        )
        ellipsoid_volume = float((4.0 * np.pi / 3.0) * math.sqrt(max(0.0, 125.0 * l1 * l2 * l3)))
        particle_volume = float(np.sum((4.0 * np.pi / 3.0) * c_radii**3))
        occupancy_proxy = particle_volume / ellipsoid_volume if ellipsoid_volume > 0 else math.nan
        phi_in_range = bool(0.0 <= occupancy_proxy <= 1.0) if math.isfinite(occupancy_proxy) else None

        radial_distance = np.linalg.norm(centered, axis=1)
        if n_syn and n_ves and radius_of_gyration > 0:
            mean_syn_radius = float(np.mean(radial_distance[syn_mask]))
            mean_ves_radius = float(np.mean(radial_distance[ves_mask]))
            core_shell_signed = (mean_syn_radius - mean_ves_radius) / radius_of_gyration
            core_shell_strength = float(
                wasserstein_distance(radial_distance[syn_mask], radial_distance[ves_mask])
                / radius_of_gyration
            )
            core_species = "vesicle" if core_shell_signed > 0 else "synapsin" if core_shell_signed < 0 else "none"
        else:
            core_shell_signed = math.nan
            core_shell_strength = math.nan
            core_species = "single_species"

        observed_cross = float(e_sv / e_total) if e_total else math.nan
        random_cross = (
            float(2.0 * n_syn * n_ves / (cluster_size * (cluster_size - 1)))
            if cluster_size > 1 and n_syn and n_ves
            else math.nan
        )
        cross_enrichment = (
            float(observed_cross / random_cross)
            if math.isfinite(observed_cross) and math.isfinite(random_cross) and random_cross > 0
            else math.nan
        )
        excess_segregation = 1.0 - cross_enrichment if math.isfinite(cross_enrichment) else math.nan
        assortativity = _type_assortativity(e_ss, e_sv, e_vv) if n_syn and n_ves else math.nan
        drift_syn = _nanmean(c_drift[syn_mask]) if n_syn else math.nan
        drift_ves = _nanmean(c_drift[ves_mask]) if n_ves else math.nan
        clusters.append(
            {
                **base,
                "cluster_id": f"{task['run_id']}::cfg_{task['analysis_config_id']}::frame_{task['frame_index']}::component_{int(component)}",
                "component_label": int(component),
                "cluster_size": cluster_size,
                "cluster_fraction": float(cluster_size / count),
                "n_vesicles": n_ves,
                "n_synapsins": n_syn,
                "gyration_ellipsoid_volume_nm3": ellipsoid_volume,
                "particle_volume_nm3": particle_volume,
                "ellipsoid_occupancy_proxy": occupancy_proxy,
                "packing_fraction_phi": occupancy_proxy,
                "phi": occupancy_proxy,
                "phi_proxy_in_unit_interval": phi_in_range,
                "radius_of_gyration_nm": radius_of_gyration,
                "radius_of_gyration": radius_of_gyration,
                "shape_anisotropy": float(shape_anisotropy),
                "core_shell_index": float(core_shell_signed),
                "core_shell_strength": core_shell_strength,
                "core_species": core_species,
                "type_assortativity": assortativity,
                "segregation_index": assortativity,
                "excess_segregation_index": excess_segregation,
                "observed_cross_edge_fraction": observed_cross,
                "random_mixing_cross_fraction": random_cross,
                "cross_contact_enrichment": cross_enrichment,
                "homophily_S": float((2 * e_ss) / (2 * e_ss + e_sv)) if (2 * e_ss + e_sv) else math.nan,
                "homophily_V": float((2 * e_vv) / (2 * e_vv + e_sv)) if (2 * e_vv + e_sv) else math.nan,
                "mean_neighbors_SS": float(2 * e_ss / n_syn) if n_syn else math.nan,
                "mean_neighbors_VV": float(2 * e_vv / n_ves) if n_ves else math.nan,
                "mean_neighbors_cross_S": float(e_sv / n_syn) if n_syn else math.nan,
                "mean_neighbors_cross_V": float(e_sv / n_ves) if n_ves else math.nan,
                "mean_force_S": _nanmean(c_force[syn_mask]) if n_syn else math.nan,
                "mean_force_V": _nanmean(c_force[ves_mask]) if n_ves else math.nan,
                "deterministic_drift_proxy_S": drift_syn,
                "deterministic_drift_proxy_V": drift_ves,
                "force_over_drag_S": drift_syn,
                "force_over_drag_V": drift_ves,
                "mean_velocity_S": drift_syn,
                "mean_velocity_V": drift_ves,
                "member_ids": particle_ids[member_index].tolist() if task["store_members"] else None,
            }
        )
    return clusters, frame_summary, sensitivity_rows

def _weighted_mean(frame: pd.DataFrame, value_column: str, weight_column: str) -> float:
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(frame[weight_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else math.nan
