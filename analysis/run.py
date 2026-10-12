"""Rebuild the reported structural, mesoscale and exploratory ML readouts.

Run with ``python -m analysis.run`` from the repository root. Results are
written to the ignored ``outputs/`` directory; reference/ remains immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import graph, mesoscale

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifest.json"
OUT = ROOT / "outputs"
METRICS = (
    "late_mean_largest_component_vesicle_fraction",
    "late_mean_vesicle_recruited_fraction",
    "late_mean_synapsin_recruited_fraction",
)
FEATURES = METRICS + (
    "late_mean_largest_component_fraction",
    "late_mean_monomer_fraction",
    "late_mean_weight_average_finite_cluster_size",
    "late_particle_weighted_shape_anisotropy",
    "late_particle_weighted_core_shell_signed",
    "late_edge_structure_mean_type_assortativity",
)


def checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def check_manifest(manifest: dict) -> None:
    runs = manifest["runs"]
    if len(runs) != 12 or len({r["run_id"] for r in runs}) != 12:
        raise ValueError("Expected exactly 12 unique run IDs")
    for run in runs:
        config, trajectory = ROOT / run["config"], ROOT / run["trajectory"]
        if not config.is_file() or not trajectory.is_file():
            raise FileNotFoundError(f"Missing config/data for {run['run_id']}")
        if checksum(config) != run["config_sha256"] or checksum(trajectory) != run["published_sha256"]:
            raise ValueError(f"SHA256 mismatch: {run['run_id']}")
        if not run["exported_numeric_identity_verified"]:
            raise ValueError(f"Unverified numeric export: {run['run_id']}")
        cfg = json.loads(config.read_text(encoding="utf-8"))
        if cfg["total"][0]["seed"] != run["seed"] or cfg["total"][0]["address"] != run["run_id"]:
            raise ValueError(f"Config identity mismatch: {run['run_id']}")
        if not math.isclose(cfg["times"][0]["time_step"], .0007, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Fixed integration step differs")
        if cfg["environment"][0]["shape"] != 2 or cfg["environment"][0]["levelNei"] != 1:
            raise ValueError("Unexpected geometry/search depth")
    cells = {(r["synapsin_concentration_uM"], r["vesicle_core_radius_std_nm"]) for r in runs}
    if cells != {(10., 0.), (10., 2.5), (60., 0.), (60., 2.5)}:
        raise ValueError("Incomplete 2x2 design")
    for concentration, dispersion in cells:
        if {r["seed"] for r in runs if (r["synapsin_concentration_uM"], r["vesicle_core_radius_std_nm"]) == (concentration, dispersion)} != {9210001, 9210002, 9210003}:
            raise ValueError("Incomplete blocked seed design")


def structural_run(run: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = json.loads((ROOT / run["config"]).read_text(encoding="utf-8"))
    force = config["forces_parms"][0]
    physical = config["parameters"][0]
    frames, shells, late_micro = [], [], []
    with h5py.File(ROOT / run["trajectory"], "r") as raw:
        step_names = sorted((name for name in raw if name.startswith("step_")), key=lambda s: int(s.rsplit("_", 1)[1]))
        expected = list(range(200, 40001, 200))
        observed = [int(s.rsplit("_", 1)[1]) for s in step_names]
        if observed != expected:
            raise ValueError(f"Incomplete saved-frame schedule: {run['run_id']}")
        if "run_completion" not in raw:
            raise ValueError("Missing numerical completion diagnostics")
        completion = raw["run_completion"]
        scalar = lambda key: float(np.asarray(completion[key][()]).reshape(-1)[0])
        for key, value in (("run_completed", 1), ("all_expected_frames_saved", 1), ("dynamic_steps_recorded", 40000), ("saved_hdf5_frame_count", 200), ("final_state_step", 40000), ("particle_count_at_completion", run["synapsin_count"] + run["vesicle_count"])):
            if scalar(key) != value:
                raise ValueError(f"Runtime completion check failed for {run['run_id']}:{key}")
        for key in ("projection_nonconverged_steps", "projection_iteration_limit_steps", "nonfinite_euler_proposal_count", "nonfinite_position_update_count", "boundary_rejected_update_count"):
            if scalar(key) != 0:
                raise ValueError(f"Numerical QC failed for {run['run_id']}:{key}")
        if scalar("maximum_projection_residual_excess_nm") > .005001 or scalar("maximum_projection_iterations_used") > 128:
            raise ValueError(f"Projection guardrail failed for {run['run_id']}")
        first = raw[step_names[0]]
        first_types = first["types"][:]
        vesicle_mask = first_types == 2
        if int(vesicle_mask.sum()) != 200 or not np.allclose(first["radii"][:][vesicle_mask] - first["core_radii"][:][vesicle_mask], 4.8, atol=3e-6):
            raise ValueError(f"Vesicle geometry/corona QC failed for {run['run_id']}")
        for frame_index, name in enumerate(step_names):
            group = raw[name]
            positions = group["positions"][:]
            radii = group["radii"][:]
            types = group["types"][:]
            ids = group["ids"][:]
            force_mag = group["total_force_magnitude"][:]
            if len(positions) != run["synapsin_count"] + run["vesicle_count"]:
                raise ValueError(f"Particle count mismatch: {run['run_id']}")
            context = {
                "step_name": name,
                "run_id": run["run_id"],
                "condition_id": run["condition_id"],
                "seed": run["seed"],
                "seed_status": "known",
                "analysis_config_id": "shell2_late30",
                "frame_index": frame_index,
                "simulation_step": observed[frame_index],
                "physical_time_ms": float(group["physical_time_ms"][0]),
                "time_step_ms": run["dt_ms"],
                "particle_id_source": "saved_ids",
                "synapsin_type": 1,
                "vesicle_type": 2,
                "eta": physical["eta"],
                "drag_syn": config["synapsinI"][0]["drag_syn"],
                "drag_ves": config["vesicles"][0]["drag_ves"],
                "boundary_condition": "closed_reflecting_no_flux",
                "primary_shell_nm": 2.,
                "shells_nm": (0., 2., 4., 6.),
                "core_allowed_overlap_nm": force["core_allowed_overlap_nm"],
                "vesicle_corona_thickness_nm": config["vesicles"][0]["corona_thickness_nm"],
                "overlap_projection_residual_tolerance_nm": force["overlap_projection_residual_tolerance_nm"],
                "min_cluster_size": 2,
                "store_members": False,
            }
            micro, frame, sensitivity = graph._analyze_frame_arrays(context, positions, types, radii, ids, force_mag)
            frames.append(frame)
            shells.extend(sensitivity)
            if frame_index >= 140:
                late_micro.extend(micro)
    frame_table = pd.DataFrame(frames)
    shell_table = pd.DataFrame(shells)
    late = frame_table.tail(60)
    micro = pd.DataFrame(late_micro)
    record = {"Study_Run_ID": run["run_id"], "Study_Condition": run["condition_id"], "Design_Seed": run["seed"], "frame_count": len(frames), "late_window_frame_count": 60}
    for name in ("largest_component_vesicle_fraction", "vesicle_recruited_fraction", "synapsin_recruited_fraction", "largest_component_fraction", "monomer_fraction", "weight_average_finite_cluster_size"):
        record["late_mean_" + name] = float(late[name].mean())
    record["late_particle_weighted_shape_anisotropy"] = graph._weighted_mean(micro, "shape_anisotropy", "cluster_size")
    record["late_particle_weighted_core_shell_signed"] = graph._weighted_mean(micro, "core_shell_index", "cluster_size")
    record["late_edge_structure_mean_type_assortativity"] = float(micro["type_assortativity"].mean())
    record["maximum_pair_allowance_excess_nm_over_saved_frames"] = float(frame_table["max_pair_allowance_excess_nm"].max())
    record["frames_above_allowed_plus_tolerance_fraction"] = float(np.mean(frame_table["overlap_pairs_above_allowed_plus_tolerance"] > 0))
    return frame_table, shell_table, record


def compare_reference(actual: pd.DataFrame, reference_name: str, keys: list[str], numeric: list[str], atol=1e-9) -> dict:
    expected = pd.read_csv(ROOT / "reference" / reference_name)
    merged = actual.merge(expected, on=keys, suffixes=("_new", "_reference"), validate="one_to_one")
    if len(merged) != len(actual):
        raise ValueError(f"Reference join cardinality mismatch: {reference_name}")
    worst = 0.0
    for column in numeric:
        current = pd.to_numeric(merged[column + "_new"], errors="coerce").to_numpy(float)
        previous = pd.to_numeric(merged[column + "_reference"], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(current), np.isnan(previous)):
            raise AssertionError(f"NaN pattern mismatch in {reference_name}:{column}")
        difference = np.abs(current - previous)
        worst = max(worst, float(np.nanmax(difference)) if np.isfinite(difference).any() else 0.)
        if not np.allclose(current, previous, atol=atol, rtol=1e-8, equal_nan=True):
            raise AssertionError(f"Recomputed values differ in {reference_name}:{column}; max={worst}")
    return {"table": reference_name, "rows_checked": len(merged), "numeric_columns_checked": len(numeric), "maximum_absolute_difference": worst}


def paired_stats(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in FEATURES:
        values = runs.pivot(index="Design_Seed", columns="Study_Condition", values=metric)
        contrasts = {
            "concentration_mono": values.C60_mono - values.C10_mono,
            "concentration_poly": values.C60_poly - values.C10_poly,
            "dispersion_10": values.C10_poly - values.C10_mono,
            "dispersion_60": values.C60_poly - values.C60_mono,
            "factor_interaction": (values.C60_poly - values.C60_mono) - (values.C10_poly - values.C10_mono),
        }
        for label, series in contrasts.items():
            delta = series.to_numpy(float)
            mean = float(delta.mean())
            sd = float(delta.std(ddof=1))
            half = float(stats.t.ppf(.975, 2) * sd / np.sqrt(3))
            signs = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
            null = np.abs((signs * delta).mean(axis=1))
            p = float(np.mean(null >= abs(mean) - 1e-12))
            paired_t = float(2 * stats.t.sf(abs(mean / (sd / np.sqrt(3))), 2)) if sd > 0 else math.nan
            rows.append({"metric": metric, "contrast": label, "n_seed_blocks": 3, "mean_paired_difference": mean, "sd_difference": sd, "ci95_low": mean - half, "ci95_high": mean + half, "cohen_dz": mean / sd if sd > 0 else math.nan, "paired_t_p_raw": paired_t, "signflip_two_sided_p": p})
    return pd.DataFrame(rows)


def planned_contrasts(paired: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "dispersion_10": ("polydispersity_within_concentration", "C=10 uM"),
        "dispersion_60": ("polydispersity_within_concentration", "C=60 uM"),
        "concentration_mono": ("concentration_adjacent_within_polydispersity", "Rsd=0 nm"),
        "concentration_poly": ("concentration_adjacent_within_polydispersity", "Rsd=2.5 nm"),
    }
    result = paired[paired.contrast.isin(labels)].copy()
    result["contrast_family"] = result.contrast.map(lambda c: labels[c][0])
    result["stratum"] = result.contrast.map(lambda c: labels[c][1])
    result["mean_difference"] = result.mean_paired_difference
    result["exact_sign_flip_p_raw"] = result.signflip_two_sided_p
    for source, destination in (("paired_t_p_raw", "paired_t_p_holm"), ("exact_sign_flip_p_raw", "exact_sign_flip_p_holm")):
        result[destination] = np.nan
        for _, group in result.groupby(["contrast_family", "metric"]):
            raw = group[source].to_numpy(float)
            finite = np.flatnonzero(np.isfinite(raw))
            order = finite[np.argsort(raw[finite])]
            corrected = np.full(len(group), np.nan)
            running = 0.
            for rank, index in enumerate(order):
                running = max(running, min(1., raw[index] * (len(finite) - rank)))
                corrected[index] = running
            result.loc[group.index, destination] = corrected
    return result


def check_paired_reference(paired: pd.DataFrame) -> dict:
    expected = pd.read_csv(ROOT / "reference/independent_contrasts.csv")
    crosswalk = {"dispersion_10": "size_at_10", "dispersion_60": "size_at_60", "factor_interaction": "difference_in_differences"}
    trial = paired.copy()
    trial["contrast"] = trial.contrast.replace(crosswalk)
    trial = trial.rename(columns={"mean_paired_difference": "mean_difference"})
    verified = compare_reference(trial, "independent_contrasts.csv", ["metric", "contrast"], ["mean_difference", "ci95_low", "ci95_high"])
    joint = trial.merge(expected, on=["metric", "contrast"], suffixes=("_new", "_reference"), validate="one_to_one")
    known = joint.exact_sign_flip_p.notna()
    if not np.allclose(joint.loc[known, "signflip_two_sided_p"], joint.loc[known, "exact_sign_flip_p"], atol=1e-12):
        raise AssertionError("Seed-block sign-flip tests disagree with reviewed table")
    verified["signflip_p_values_checked"] = int(known.sum())
    return verified


def blocked_hc3(runs: pd.DataFrame) -> pd.DataFrame:
    """Seed-blocked 2x2 OLS with the recorded HC3 sandwich covariance."""
    ordered = runs.sort_values(["Design_Seed", "Study_Condition"]).reset_index(drop=True)
    seed = ordered.Design_Seed.to_numpy(int)
    concentration = ordered.Study_Condition.str.startswith("C60").to_numpy(float)
    dispersion = ordered.Study_Condition.str.endswith("poly").to_numpy(float)
    x = np.column_stack([np.ones(len(ordered)), seed == 9210002, seed == 9210003, concentration, dispersion, concentration * dispersion]).astype(float)
    rank = int(np.linalg.matrix_rank(x))
    df = len(ordered) - rank
    if len(ordered) != 12 or rank != 6 or df != 6:
        raise ValueError("Expected full-rank blocked 2x2 design")
    inverse = np.linalg.pinv(x.T @ x)
    leverage = np.clip(np.diag(x @ inverse @ x.T), 0, 1 - 1e-12)
    critical = stats.t.ppf(.975, df)
    terms = ["synapsin_concentration_uM[60.0] vs 10.0", "vesicle_core_radius_std_nm[2.5] vs 0.0", "synapsin_concentration_uM[60.0]:vesicle_core_radius_std_nm[2.5]"]
    rows = []
    for metric in FEATURES:
        y = ordered[metric].to_numpy(float)
        beta = inverse @ x.T @ y
        residual = y - x @ beta
        adjusted = residual / (1 - leverage)
        covariance = inverse @ (x.T @ ((adjusted**2)[:, None] * x)) @ inverse
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
        residual_variance = float(np.sum(residual**2) / df)
        cooks = residual**2 / x.shape[1] / residual_variance * leverage / (1 - leverage)**2 if residual_variance > 0 else np.zeros(len(y))
        normality_p = float(stats.shapiro(residual).pvalue)
        for term, index in zip(terms, (3, 4, 5)):
            se, estimate = float(standard_errors[index]), float(beta[index])
            statistic = estimate / se if se > 0 else math.nan
            p = float(2 * stats.t.sf(abs(statistic), df)) if math.isfinite(statistic) else math.nan
            rows.append({"metric": metric, "term": term, "estimate_raw_scale": estimate, "hc3_standard_error": se, "ci95_low": estimate - critical * se, "ci95_high": estimate + critical * se, "t_statistic": statistic, "p_raw": p, "n_runs": 12, "n_seeds": 3, "model_rank": rank, "df_residual": df, "covariance_estimator": "HC3", "residual_shapiro_p": normality_p, "max_cooks_distance": float(np.max(cooks))})
    result = pd.DataFrame(rows)
    # Holm correction over the three factorial terms within each observable.
    adjusted_values = []
    for _, group in result.groupby("metric", sort=False):
        raw = group.p_raw.to_numpy(float)
        ranked = np.argsort(raw)
        corrected = np.empty(3)
        running = 0.
        for k, at in enumerate(ranked):
            running = max(running, min(1., raw[at] * (3 - k)))
            corrected[at] = running
        adjusted_values.extend(corrected)
    result["p_holm"] = adjusted_values
    return result


def check_blocked_reference(fits: pd.DataFrame) -> dict:
    columns = ["estimate_raw_scale", "hc3_standard_error", "ci95_low", "ci95_high", "t_statistic", "p_raw", "residual_shapiro_p", "max_cooks_distance", "p_holm"]
    return compare_reference(fits, "Blocked_Factorial_Effects_v5.csv", ["metric", "term"], columns, atol=1e-9)


def exploratory_ml(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = runs[list(FEATURES)].to_numpy(float)
    seed = runs.Design_Seed.to_numpy(int)
    rows, predictions_out = [], []
    for label, target in (("concentration", (runs.Study_Condition.str.startswith("C60")).astype(int)), ("radius_dispersion", (runs.Study_Condition.str.endswith("poly")).astype(int))):
        y = np.asarray(target)
        predictions = np.full(len(y), -1)
        for held_seed in np.unique(seed):
            train, test = seed != held_seed, seed == held_seed
            pipeline = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("model", LogisticRegression(C=1, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=20260909))])
            pipeline.fit(x[train], y[train])
            predictions[test] = pipeline.predict(x[test])
        rows.append({"task": label, "n_runs": len(y), "n_held_out_seeds": len(np.unique(seed)), "balanced_accuracy": float(balanced_accuracy_score(y, predictions)), "macro_f1": float(f1_score(y, predictions, average="macro"))})
        levels = (10., 60.) if label == "concentration" else (0., 2.5)
        for record, observed, predicted in zip(runs.itertuples(), y, predictions):
            predictions_out.append({"Study_Run_ID": record.Study_Run_ID, "Design_Seed": record.Design_Seed, "task": label, "observed_label": levels[observed], "predicted_label": levels[predicted], "correct": bool(observed == predicted)})
    transformed = StandardScaler().fit_transform(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(x))
    pca = PCA(n_components=3, random_state=20260909).fit(transformed)
    coordinates = pca.transform(transformed)
    pca_scores = pd.DataFrame({"Study_Run_ID": runs.Study_Run_ID, "Design_Seed": runs.Design_Seed})
    loadings = []
    for index in range(3):
        pca_scores[f"PC{index + 1}"] = coordinates[:, index]
        pca_scores[f"PC{index + 1}_explained_variance"] = pca.explained_variance_ratio_[index]
        for feature, loading in zip(FEATURES, pca.components_[index]):
            loadings.append({"component": f"PC{index + 1}", "feature": feature, "loading": float(loading), "explained_variance_ratio": float(pca.explained_variance_ratio_[index])})
    note = {"feature_names": FEATURES, "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "scope": "descriptive/exploratory classification of imposed design factors; not LLPS detection"}
    return pd.DataFrame(rows), note, pd.DataFrame(predictions_out), pca_scores, pd.DataFrame(loadings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check_manifest(manifest)
    print("MANIFEST_AND_HASHES_OK runs=12", flush=True)
    if args.manifest_only:
        return
    OUT.mkdir(exist_ok=True)
    summaries, density, histogram, rdf, validation = [], [], [], [], []
    for run in manifest["runs"]:
        frames, shells, summary = structural_run(run)
        summaries.append(summary)
        run_out = OUT / run["run_id"]
        run_out.mkdir(exist_ok=True)
        frames.to_csv(run_out / "frames.csv", index=False)
        shells.to_csv(run_out / "shells.csv", index=False)
        verification = compare_reference(pd.DataFrame([summary]), "run_level.csv", ["Study_Run_ID"], list(FEATURES) + ["maximum_pair_allowance_excess_nm_over_saved_frames", "frames_above_allowed_plus_tolerance_fraction"])
        validation.append(verification)
        external = {"paths": {"dashboard": ROOT / run["config"], "h5": ROOT / run["trajectory"]}}
        local_density, local_histogram, local_rdf, provenance = mesoscale.analyze_run(external)
        for table, collection in ((local_density, density), (local_histogram, histogram), (local_rdf, rdf)):
            table["Study_Run_ID"] = run["run_id"]
            table["Study_Condition"] = run["condition_id"]
            table["Design_Seed"] = run["seed"]
            collection.append(table)
        (run_out / "mesoscale_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(f"ANALYSIS_VERIFIED {run['run_id']}", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "run_level.csv", index=False)
    for name, tables, columns in (
        ("Local_Density_Run_Level.csv", density, ["interior_probe_count", "mean_number_density_per_um3", "q50_number_density_per_um3", "zero_density_fraction"]),
        ("Local_Density_Histograms_Run_Level.csv", histogram, ["probability_mass"]),
        ("RDF_Run_Level.csv", rdf, ["mean_observed_pairs", "mean_ideal_gas_pairs", "rdf_geometric_null_ratio"]),
    ):
        combined = pd.concat(tables, ignore_index=True)
        key = ["Study_Run_ID"] + (["probe_radius_nm", "species", "bin_index"] if "Histograms" in name else ["probe_radius_nm", "species"] if "Density" in name else ["pair", "r_low_nm"])
        validation.append(compare_reference(combined, name, key, columns, atol=1e-6))
        combined.to_csv(OUT / name, index=False)
    paired = paired_stats(summary)
    validation.append(check_paired_reference(paired))
    paired.to_csv(OUT / "paired_contrasts.csv", index=False)
    planned = planned_contrasts(paired)
    validation.append(compare_reference(planned, "Planned_Contrasts_v5.csv", ["metric", "contrast_family", "stratum"], ["mean_difference", "sd_difference", "ci95_low", "ci95_high", "cohen_dz", "paired_t_p_raw", "exact_sign_flip_p_raw", "paired_t_p_holm", "exact_sign_flip_p_holm"], atol=1e-9))
    planned.to_csv(OUT / "planned_contrasts.csv", index=False)
    factorial = blocked_hc3(summary)
    validation.append(check_blocked_reference(factorial))
    factorial.to_csv(OUT / "blocked_hc3.csv", index=False)
    scores, pca, predictions, pca_scores, loadings = exploratory_ml(summary)
    reference_ml = pd.read_csv(ROOT / "reference/ML_CV_Performance_v5.csv")
    checked_ml = scores.merge(reference_ml, on="task", suffixes=("_new", "_reference"), validate="one_to_one")
    if len(checked_ml) != 2:
        raise AssertionError("Expected both concentration and radius ML checks")
    for name in ("balanced_accuracy", "macro_f1"):
        if not np.allclose(checked_ml[name + "_new"], checked_ml[name + "_reference"], atol=1e-12):
            raise AssertionError(f"Exploratory ML discrepancy: {name}")
    validation.append({"table": "ML_CV_Performance_v5.csv", "rows_checked": 2, "numeric_columns_checked": 2, "maximum_absolute_difference": 0.0})
    scores.to_csv(OUT / "exploratory_ml.csv", index=False)
    predictions.to_csv(OUT / "exploratory_ml_predictions.csv", index=False)
    pca_scores.to_csv(OUT / "exploratory_pca_scores.csv", index=False)
    loadings.to_csv(OUT / "exploratory_pca_loadings.csv", index=False)
    validation.append(compare_reference(predictions, "ML_CV_Predictions_v5.csv", ["Study_Run_ID", "task"], ["observed_label", "predicted_label"], atol=1e-12))
    validation.append(compare_reference(pca_scores, "ML_PCA_Scores_v5.csv", ["Study_Run_ID"], ["PC1", "PC2", "PC3", "PC1_explained_variance", "PC2_explained_variance", "PC3_explained_variance"], atol=1e-9))
    validation.append(compare_reference(loadings, "ML_PCA_Loadings_v5.csv", ["component", "feature"], ["loading", "explained_variance_ratio"], atol=1e-9))
    (OUT / "exploratory_pca.json").write_text(json.dumps(pca, indent=2) + "\n", encoding="utf-8")
    report = {"dataset_count": 12, "numerical_reference_checks": validation, "graph_shell_nm": 2, "late_frames_per_run": 60, "contrast_count": len(paired), "status": "all checks passed"}
    (OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ARTICLE_TWO_REANALYSIS_COMPLETE runs=12 status=all_checks_passed", flush=True)


if __name__ == "__main__":
    main()
