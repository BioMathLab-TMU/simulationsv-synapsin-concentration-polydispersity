"""Small-sample sensitivity checks for the article-two analysis.

The three seed labels, rather than frames or radius arms, define blocks.
The label-remapping exercise is descriptive: the design factors were not
assigned by a randomized experiment, so its tail fraction is not a
population-level or causal p-value.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .run import FEATURES, OUT


def _held_seed_accuracy(features: np.ndarray, labels: np.ndarray, seeds: np.ndarray) -> float:
    predicted = np.full(len(labels), -1, dtype=int)
    for held in np.unique(seeds):
        train, test = seeds != held, seeds == held
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=20260909)),
        ])
        model.fit(features[train], labels[train])
        predicted[test] = model.predict(features[test])
    return float(balanced_accuracy_score(labels, predicted))


def calculate(runs: pd.DataFrame) -> dict:
    runs = runs.sort_values(["Design_Seed", "Study_Condition"]).reset_index(drop=True)
    seeds = runs.Design_Seed.to_numpy(int)
    labels = runs.Study_Condition.str.startswith("C60").to_numpy(int)
    x = runs[list(FEATURES)].to_numpy(float)
    observed = _held_seed_accuracy(x, labels, seeds)
    unique_seeds = np.unique(seeds)
    null = []
    # Concentration labels move together across radius arms within each seed.
    # This preserves the four-cell layout in every held-out seed block.
    for flips in itertools.product((0, 1), repeat=len(unique_seeds)):
        remapped = labels.copy()
        for seed, flip in zip(unique_seeds, flips):
            if flip:
                remapped[seeds == seed] = 1 - remapped[seeds == seed]
        null.append(_held_seed_accuracy(x, remapped, seeds))
    null = np.asarray(null)
    pivot = runs.pivot(index="Design_Seed", columns="Study_Condition", values="late_mean_largest_component_vesicle_fraction")
    mono = (pivot.C60_mono - pivot.C10_mono).to_numpy(float)
    poly = (pivot.C60_poly - pivot.C10_poly).to_numpy(float)
    block_mean = (mono + poly) / 2
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    block_null = (signs * block_mean).mean(axis=1)
    first = []
    for row in runs.itertuples():
        frames = pd.read_csv(OUT / row.Study_Run_ID / "frames.csv", usecols=["largest_component_vesicle_fraction", "time_ms"])
        first.append({"condition": row.Study_Condition,
                      "seed": int(row.Design_Seed),
                      "first_saved_frame_ms": float(frames.time_ms.iloc[0]),
                      "first_saved_frame_value": float(frames.largest_component_vesicle_fraction.iloc[0]),
                      "first_ten_frame_mean": float(frames.largest_component_vesicle_fraction.head(10).mean())})
    early = pd.DataFrame(first)
    return {
        "independent_unit": "seed block (n=3)",
        "classifier_seed_block_joint_label_remapping": {
            "observed_balanced_accuracy": observed,
            "remapping_count": int(len(null)),
            "null_balanced_accuracies": null.tolist(),
            "tail_fraction_ge_observed": float(np.mean(null >= observed - 1e-12)),
            "interpretation": "descriptive exact label-remapping sensitivity; not a randomized-treatment or population-inference p-value",
        },
        "concentration_lsv_seed_block_contrasts": {
            "mono_by_seed": mono.tolist(),
            "poly_by_seed": poly.tolist(),
            "within_seed_arm_average": block_mean.tolist(),
            "all_six_same_sign_descriptive": bool(np.all(np.r_[mono, poly] > 0)),
            "blockwise_two_sided_signflip_p": float(np.mean(np.abs(block_null) >= abs(block_mean.mean()) - 1e-12)),
            "two_sided_exact_p_floor_n3": .25,
        },
        "early_window": {
            "definition": "first saved frame (0.14 ms) and first ten saved frames (0.14-1.4 ms); descriptive only",
            "by_run": first,
            "by_condition": early.groupby("condition")[["first_saved_frame_value", "first_ten_frame_mean"]].mean().reset_index().to_dict("records"),
        },
        "radius_margin_0p05": "Exploratory descriptive compatibility check only; margin was selected after viewing data, so no formal equivalence claim.",
    }


def main() -> None:
    runs = pd.read_csv(OUT / "run_level.csv")
    result = calculate(runs)
    destination = OUT / "inference_checks.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"ARTICLE_TWO_INFERENCE_CHECKS_OK remappings={result['classifier_seed_block_joint_label_remapping']['remapping_count']}")


if __name__ == "__main__":
    main()
