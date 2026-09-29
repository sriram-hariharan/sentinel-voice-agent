"""Run the frozen, bounded V2-C3 Step 5 finalist search.

This module reuses the Step 4 development data, folds, frozen BGE cache,
metrics, safety gates, ranking, latency, and size implementations. It never
creates folds or embeddings and never opens final-evaluation data.
"""

from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

tournament = import_module(
    "scripts.run_v2c3_model_tournament"
    if __package__
    else "run_v2c3_model_tournament"
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
CONFIG_PATH = ML_DIRECTORY / "v2c3_tuning_config.json"
REPORT_SCHEMA_VERSION = "v2c3-bounded-tuning-report.v1"
EXPECTED_FINALISTS = [
    "direct_v2c1_tfidf_balanced_svc",
    "direct_bge_balanced_svc",
    "direct_lsa_balanced_svc",
]
EXPECTED_VARIANT_IDS = [
    "tfidf_svc_c_0_5",
    "tfidf_svc_c_0_75",
    "tfidf_svc_c_1_0",
    "tfidf_svc_c_1_5",
    "tfidf_svc_c_2_0",
    "bge_svc_c_0_25",
    "bge_svc_c_0_5",
    "bge_svc_c_1_0",
    "bge_svc_c_2_0",
    "bge_svc_c_4_0",
    "lsa_svc_components_128_c_1_0",
    "lsa_svc_components_256_c_0_5",
    "lsa_svc_components_256_c_1_0",
    "lsa_svc_components_256_c_2_0",
    "lsa_svc_components_384_c_1_0",
]


def repository_path(relative_path: str) -> Path:
    path = (REPOSITORY_ROOT / relative_path).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    validate_config(config)
    return config


def _expected_variant_parameters() -> dict[str, dict[str, float | int]]:
    return {
        "tfidf_svc_c_0_5": {"C": 0.5},
        "tfidf_svc_c_0_75": {"C": 0.75},
        "tfidf_svc_c_1_0": {"C": 1.0},
        "tfidf_svc_c_1_5": {"C": 1.5},
        "tfidf_svc_c_2_0": {"C": 2.0},
        "bge_svc_c_0_25": {"C": 0.25},
        "bge_svc_c_0_5": {"C": 0.5},
        "bge_svc_c_1_0": {"C": 1.0},
        "bge_svc_c_2_0": {"C": 2.0},
        "bge_svc_c_4_0": {"C": 4.0},
        "lsa_svc_components_128_c_1_0": {"n_components": 128, "C": 1.0},
        "lsa_svc_components_256_c_0_5": {"n_components": 256, "C": 0.5},
        "lsa_svc_components_256_c_1_0": {"n_components": 256, "C": 1.0},
        "lsa_svc_components_256_c_2_0": {"n_components": 256, "C": 2.0},
        "lsa_svc_components_384_c_1_0": {"n_components": 384, "C": 1.0},
    }


def validate_config(config: dict[str, Any]) -> None:
    if config["schema_version"] != "v2c3-bounded-tuning-config.v1":
        raise ValueError("unexpected tuning config schema")
    if config["hyperparameter_search_type"] != "bounded_predeclared_manual_search":
        raise ValueError("Step 5 must remain a bounded predeclared manual search")
    if config["tuning_performed"] is not False:
        raise ValueError("the frozen tuning config cannot contain tuning results")
    if config["final_evaluation_performed"] is not False:
        raise ValueError("final evaluation cannot occur in the tuning config")
    if config["frozen_finalist_ids"] != EXPECTED_FINALISTS:
        raise ValueError("the three frozen Step 4 finalists changed")
    variants = config["variants"]
    if len(variants) != 15 or [row["id"] for row in variants] != EXPECTED_VARIANT_IDS:
        raise ValueError("the exact ordered set of 15 tuning variants changed")
    expected_parameters = _expected_variant_parameters()
    expected_family = {
        **{variant_id: "tfidf_svc" for variant_id in EXPECTED_VARIANT_IDS[:5]},
        **{variant_id: "bge_svc" for variant_id in EXPECTED_VARIANT_IDS[5:10]},
        **{variant_id: "lsa_svc" for variant_id in EXPECTED_VARIANT_IDS[10:]},
    }
    expected_representation = {
        "tfidf_svc": "v2c1_word_char_tfidf",
        "bge_svc": "bge_small_en_v1_5",
        "lsa_svc": "tfidf_lsa",
    }
    expected_finalist = dict(
        zip(("tfidf_svc", "bge_svc", "lsa_svc"), EXPECTED_FINALISTS, strict=True)
    )
    for variant in variants:
        family = expected_family[variant["id"]]
        if variant["family"] != family:
            raise ValueError(f"tuning family changed for {variant['id']}")
        if variant["parameters"] != expected_parameters[variant["id"]]:
            raise ValueError(f"tuning parameters changed for {variant['id']}")
        if variant["representation"] != expected_representation[family]:
            raise ValueError(f"representation changed for {variant['id']}")
        if variant["finalist_id"] != expected_finalist[family]:
            raise ValueError(f"finalist lineage changed for {variant['id']}")
        if variant["architecture"] != "direct_9_way":
            raise ValueError("Step 5 cannot change architecture")
    scope = config["search_scope"]
    if (
        scope["variant_count"] != 15
        or scope["classifier_family"] != "LinearSVC"
        or scope["class_weight"] != "balanced"
        or scope["cartesian_grid_generation_allowed"] is not False
    ):
        raise ValueError("bounded search scope changed")
    folds = config["frozen_inputs"]["cv_folds"]
    if (
        folds["n_splits"] != 5
        or folds["method"] != "StratifiedGroupKFold"
        or folds["new_fold_generation_allowed"] is not False
    ):
        raise ValueError("the existing five-fold assignment must be reused")
    report_flags = config["report"]
    required_false = (
        "final_lockbox_used",
        "challenge_set_used",
        "cfpb_used",
        "new_fold_generation_performed",
        "embedding_fine_tuning_performed",
        "broad_search_performed",
    )
    if any(report_flags[field] is not False for field in required_false):
        raise ValueError("a prohibited Step 5 report flag changed")
    if report_flags["bounded_hyperparameter_search_performed"] is not True:
        raise ValueError("the Step 5 report must identify the bounded search")
    allowed = set(config["allowed_read_paths"])
    prohibited = set(config["prohibited_data_sources"])
    if allowed & prohibited:
        raise ValueError("a prohibited source appears in the read allowlist")
    expected_allowed = {"data/evals/v2/ml/v2c3_tuning_config.json"} | {
        config["frozen_inputs"][name]["path"]
        for name in (
            "development_dataset",
            "experiment_contract",
            "tournament_config",
            "tournament_report",
            "cv_folds",
        )
    } | {
        "data/evals/v2/ml/local/v2c3_bge_small_en_v1_5.npz",
        "data/evals/v2/ml/local/v2c3_bge_small_en_v1_5.metadata.json",
    }
    if allowed != expected_allowed:
        raise ValueError("the Step 5 read allowlist changed")


def validate_frozen_file(spec: dict[str, Any]) -> Path:
    path = repository_path(spec["path"])
    actual = tournament.sha256_file(path)
    if actual != spec["sha256"]:
        raise ValueError(f"frozen input hash mismatch for {spec['path']}: {actual}")
    return path


def load_and_validate_inputs(
    config: dict[str, Any],
) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    development_path = validate_frozen_file(frozen["development_dataset"])
    contract_path = validate_frozen_file(frozen["experiment_contract"])
    tournament_config_path = validate_frozen_file(frozen["tournament_config"])
    tournament_report_path = validate_frozen_file(frozen["tournament_report"])
    fold_path = validate_frozen_file(frozen["cv_folds"])

    step_4_config = read_json(tournament_config_path)
    tournament.validate_config(step_4_config)
    if step_4_config["development_dataset"]["path"] != frozen["development_dataset"]["path"]:
        raise ValueError("Step 4 development path differs from the tuning boundary")
    if step_4_config["development_dataset"]["sha256"] != frozen["development_dataset"]["sha256"]:
        raise ValueError("Step 4 development hash differs from the tuning boundary")
    if step_4_config["experiment_contract"]["sha256"] != frozen["experiment_contract"]["sha256"]:
        raise ValueError("Step 4 contract hash differs from the tuning boundary")

    step_4_report = read_json(tournament_report_path)
    if step_4_report["schema_version"] != "v2c3-model-tournament-report.v1":
        raise ValueError("unexpected Step 4 report schema")
    if step_4_report["finalist_shortlist"] != config["frozen_finalist_ids"]:
        raise ValueError("Step 4 report does not contain the exact frozen finalists")
    if step_4_report["eligible_winner"] != config["step_4_winner"]:
        raise ValueError("Step 4 winner changed")
    if step_4_report["tournament_config_sha256"] != frozen["tournament_config"]["sha256"]:
        raise ValueError("Step 4 report points to a different tournament config")
    if step_4_report["development_dataset_sha256"] != frozen["development_dataset"][
        "sha256"
    ]:
        raise ValueError("Step 4 report points to different development data")
    if step_4_report["experiment_contract_sha256"] != frozen["experiment_contract"][
        "sha256"
    ]:
        raise ValueError("Step 4 report points to a different experiment contract")
    if step_4_report["fold_artifact_sha256"] != frozen["cv_folds"]["sha256"]:
        raise ValueError("Step 4 report points to a different fold artifact")
    if step_4_report["fold_assignment_sha256"] != frozen["cv_folds"][
        "assignment_sha256"
    ]:
        raise ValueError("Step 4 report points to a different fold assignment")
    for field in (
        "final_lockbox_used",
        "challenge_set_used",
        "cfpb_used",
        "hyperparameter_search_performed",
    ):
        if step_4_report[field] is not False:
            raise ValueError(f"Step 4 report violates the tuning boundary: {field}")

    step_4_classifier = step_4_config["classifiers"]["linear_svc_balanced"]
    frozen_classifier = config["step_4_balanced_linear_svc_preset"]
    if frozen_classifier["class"] != step_4_classifier["class"]:
        raise ValueError("Step 4 finalist classifier family changed")
    expected_classifier_parameters = dict(step_4_classifier["parameters"])
    expected_classifier_parameters.pop("C")
    if frozen_classifier["parameters_except_c"] != expected_classifier_parameters:
        raise ValueError("Step 4 balanced LinearSVC behavior changed")
    lsa_constraint = config["representation_constraints"]["lsa"]
    step_4_lsa = step_4_config["representations"]["tfidf_lsa"]
    if (
        lsa_constraint["tfidf_base"] != step_4_lsa["tfidf_base"]
        or lsa_constraint["truncated_svd_algorithm"]
        != step_4_lsa["truncated_svd"]["algorithm"]
        or lsa_constraint["truncated_svd_n_iter"]
        != step_4_lsa["truncated_svd"]["n_iter"]
        or lsa_constraint["truncated_svd_random_state"]
        != step_4_lsa["truncated_svd"]["random_state"]
        or lsa_constraint["normalizer_norm"] != step_4_lsa["normalizer"]["norm"]
        or lsa_constraint["normalizer_copy"] != step_4_lsa["normalizer"]["copy"]
    ):
        raise ValueError("Step 4 LSA behavior changed")
    step_4_gates = step_4_config["safety_gates"]
    tuning_gates = {
        key: value
        for key, value in config["safety_gates"].items()
        if key != "unchanged_from_step_4"
    }
    if tuning_gates != step_4_gates:
        raise ValueError("Step 5 safety gates differ from Step 4")
    if (
        config["ranking"]["practical_tie_absolute_difference_below"]
        != step_4_config["ranking"]["practical_tie_absolute_difference_below"]
        or config["ranking"]["tie_break_order"]
        != step_4_config["ranking"]["tie_break_order"]
    ):
        raise ValueError("Step 5 ranking differs from Step 4")
    for policy_name in ("latency_measurement_policy", "model_size_measurement_policy"):
        tuning_policy = {
            key: value
            for key, value in config[policy_name].items()
            if key not in {"implementation_source", "unchanged_from_step_4"}
        }
        if tuning_policy != step_4_config[policy_name]:
            raise ValueError(f"Step 5 {policy_name} differs from Step 4")

    _, examples, loaded_development_path = tournament.load_development_data(step_4_config)
    if loaded_development_path != development_path:
        raise ValueError("development loader resolved an unexpected path")
    fold_artifact = read_json(fold_path)
    folds = tournament.validate_fold_artifact(fold_artifact, examples, step_4_config)
    fold_spec = frozen["cv_folds"]
    if fold_artifact["assignment_sha256"] != fold_spec["assignment_sha256"]:
        raise ValueError("frozen fold assignment hash changed")
    if len(fold_artifact["assignments"]) != fold_spec["example_count"]:
        raise ValueError("frozen folds do not represent all development examples")
    embedding_policy = config["embedding_cache_policy"]
    configured_cache_path = repository_path(embedding_policy["cache_path"])
    configured_metadata_path = repository_path(embedding_policy["cache_metadata_path"])
    if tournament.sha256_file(configured_cache_path) != embedding_policy["cache_file_sha256"]:
        raise ValueError("frozen BGE cache file hash changed")
    if (
        tournament.sha256_file(configured_metadata_path)
        != embedding_policy["cache_metadata_sha256"]
    ):
        raise ValueError("frozen BGE cache metadata hash changed")
    embeddings = tournament.validate_embedding_cache(step_4_config, examples)
    embedding_cache_path, embedding_metadata_path = tournament.cache_paths(step_4_config)
    if (
        embedding_cache_path != configured_cache_path
        or embedding_metadata_path != configured_metadata_path
    ):
        raise ValueError("Step 4 and Step 5 BGE cache paths differ")
    return {
        "development_path": development_path,
        "contract_path": contract_path,
        "tournament_config_path": tournament_config_path,
        "tournament_report_path": tournament_report_path,
        "fold_path": fold_path,
        "step_4_config": step_4_config,
        "step_4_report": step_4_report,
        "examples": examples,
        "fold_artifact": fold_artifact,
        "folds": folds,
        "embeddings": embeddings,
        "embedding_cache_path": embedding_cache_path,
        "embedding_metadata_path": embedding_metadata_path,
    }


def runtime_definition(
    variant: dict[str, Any], step_4_config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_config = deepcopy(step_4_config)
    classifier = deepcopy(runtime_config["classifiers"]["linear_svc_balanced"])
    classifier["parameters"]["C"] = variant["parameters"]["C"]
    runtime_config["classifiers"]["step5_balanced_linear_svc"] = classifier
    if variant["family"] == "lsa_svc":
        representation = deepcopy(runtime_config["representations"]["tfidf_lsa"])
        representation["truncated_svd"]["n_components"] = variant["parameters"][
            "n_components"
        ]
        runtime_config["representations"]["tfidf_lsa"] = representation
        representation_name = "tfidf_lsa"
    else:
        representation_name = variant["representation"]
    candidate = {
        "id": variant["id"],
        "architecture": "direct_9_way",
        "representation": representation_name,
        "classifier": "step5_balanced_linear_svc",
    }
    return candidate, runtime_config


def evaluate_variant(
    variant: dict[str, Any], inputs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate, runtime_config = runtime_definition(variant, inputs["step_4_config"])
    embeddings = inputs["embeddings"] if variant["family"] == "bge_svc" else None
    result, _ = tournament.run_candidate_cv(
        candidate,
        inputs["examples"],
        inputs["folds"],
        runtime_config,
        embeddings,
    )
    result["tuning_variant"] = variant
    return result, candidate, runtime_config


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return tournament.rank_eligible_candidates(results)


def best_by_family(
    results: list[dict[str, Any]], families: tuple[str, ...] = ("tfidf_svc", "bge_svc", "lsa_svc")
) -> dict[str, str | None]:
    best: dict[str, str | None] = {}
    for family in families:
        family_results = [
            result for result in results if result["tuning_variant"]["family"] == family
        ]
        ranked = rank_results(family_results)
        best[family] = ranked[0]["candidate_id"] if ranked else None
    return best


def preflight(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ready",
        "tuning_or_model_fitting_performed": False,
        "variant_count": len(config["variants"]),
        "frozen_finalist_ids": config["frozen_finalist_ids"],
        "development_example_count": len(inputs["examples"]),
        "development_dataset_sha256": config["frozen_inputs"]["development_dataset"][
            "sha256"
        ],
        "fold_assignment_sha256": inputs["fold_artifact"]["assignment_sha256"],
        "folds_reused": True,
        "new_fold_generation_performed": False,
        "bge_cache_valid": True,
        "bge_cache_file_sha256": tournament.sha256_file(inputs["embedding_cache_path"]),
        "bge_cache_metadata_sha256": tournament.sha256_file(
            inputs["embedding_metadata_path"]
        ),
        "embedding_regeneration_performed": False,
        "prohibited_data_read": False,
    }


def run_tuning(
    config: dict[str, Any], inputs: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    results: list[dict[str, Any]] = []
    runtime_definitions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for variant in config["variants"]:
        result, candidate, runtime_config = evaluate_variant(variant, inputs)
        results.append(result)
        runtime_definitions.append((candidate, runtime_config))

    labels = np.asarray([row["intent"] for row in inputs["examples"]])
    for variant, result, runtime in zip(
        config["variants"], results, runtime_definitions, strict=True
    ):
        candidate, runtime_config = runtime
        embeddings = inputs["embeddings"] if variant["family"] == "bge_svc" else None
        latency, model_size = tournament.measure_full_development_candidate(
            candidate,
            inputs["examples"],
            labels,
            runtime_config,
            embeddings,
        )
        result["latency"] = latency
        result["model_size"] = model_size

    ranked = rank_results(results)
    rank_by_id = {row["candidate_id"]: index for index, row in enumerate(ranked, start=1)}
    for result in results:
        result["eligible_rank"] = rank_by_id.get(result["candidate_id"])
    winner = ranked[0]["candidate_id"] if ranked else None
    sample_indices = tournament.latency_sample_indices(
        inputs["examples"], inputs["step_4_config"]
    )
    sample_ids = [inputs["examples"][int(index)]["example_id"] for index in sample_indices]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_phase": "V2-C3-Step-5",
        "tuning_config_sha256": tournament.sha256_file(CONFIG_PATH),
        "development_dataset_sha256": tournament.sha256_file(inputs["development_path"]),
        "experiment_contract_sha256": tournament.sha256_file(inputs["contract_path"]),
        "tournament_config_sha256": tournament.sha256_file(
            inputs["tournament_config_path"]
        ),
        "tournament_report_sha256": tournament.sha256_file(
            inputs["tournament_report_path"]
        ),
        "fold_artifact_sha256": tournament.sha256_file(inputs["fold_path"]),
        "fold_assignment_sha256": inputs["fold_artifact"]["assignment_sha256"],
        "bge_cache_file_sha256": tournament.sha256_file(inputs["embedding_cache_path"]),
        "bge_cache_metadata_sha256": tournament.sha256_file(
            inputs["embedding_metadata_path"]
        ),
        "frozen_finalist_ids": config["frozen_finalist_ids"],
        "exact_tuning_variants": config["variants"],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "fastembed": metadata.version("fastembed"),
            "joblib": metadata.version("joblib"),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "fold_summaries": inputs["fold_artifact"]["fold_summaries"],
        "variant_results": results,
        "deterministic_eligible_ranking": [row["candidate_id"] for row in ranked],
        "best_variant_by_family": best_by_family(results),
        "overall_tuned_winner": winner,
        "no_eligible_tuned_winner": winner is None,
        "failed_safety_variants": [
            row["candidate_id"]
            for row in results
            if not row["safety_gate_result"]["eligible"]
        ],
        "safety_gates": config["safety_gates"],
        "ranking_rules": config["ranking"],
        "latency_measurement_policy": config["latency_measurement_policy"],
        "latency_sample_example_ids_sha256": tournament.sha256_bytes(
            tournament.stable_json_bytes(sample_ids)
        ),
        "model_size_measurement_policy": config["model_size_measurement_policy"],
        "final_lockbox_used": False,
        "challenge_set_used": False,
        "cfpb_used": False,
        "new_fold_generation_performed": False,
        "embedding_fine_tuning_performed": False,
        "embedding_regeneration_performed": False,
        "broad_search_performed": False,
        "bounded_hyperparameter_search_performed": True,
        "tuning_performed": True,
        "raw_text_persisted": False,
        "final_evaluation_performed": False,
        "runtime_integration_performed": False,
        "final_model_artifact_frozen": False,
    }
    report_path = repository_path(config["report"]["path"])
    tournament.write_json(report_path, report)
    return report, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    inputs = load_and_validate_inputs(config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config, inputs), indent=2, sort_keys=True))
        return
    _, report_path = run_tuning(config, inputs)
    print(f"wrote bounded tuning report: {report_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
