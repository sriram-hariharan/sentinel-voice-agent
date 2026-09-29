"""Freeze the selected V2-C3 model and evaluate it only on final-only data.

``preflight`` performs integrity checks without fitting, inference, or parsing
final text. ``prepare-final-model`` fits the already-selected classifier on
development embeddings only. ``evaluate`` is the sole mode that reconstructs
or loads final text, generates final embeddings, and writes final metrics.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

tournament = import_module(
    "scripts.run_v2c3_model_tournament"
    if __package__
    else "run_v2c3_model_tournament"
)
development_builder = import_module(
    "scripts.build_v2c3_development_dataset"
    if __package__
    else "build_v2c3_development_dataset"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
CONFIG_PATH = ML_DIRECTORY / "v2c3_final_evaluation_config.json"
NOT_APPLICABLE = "NOT_APPLICABLE"
FINAL_MODEL_ID = "final_v2c3"
BASELINE_ID = "frozen_v2c1_baseline"
EXPECTED_INTENTS = (
    "account_balance",
    "card_status",
    "create_dispute",
    "escalation",
    "freeze_card",
    "informational_policy",
    "recent_transactions",
    "transaction_details",
    "unsupported_or_uncertain",
)


def repository_path(relative_path: str) -> Path:
    path = (REPOSITORY_ROOT / relative_path).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["schema_version"] != "v2c3-final-evaluation-config.v1":
        raise ValueError("unexpected final-evaluation config schema")
    if config["model_selection_complete"] is not True:
        raise ValueError("model selection must be complete before final evaluation")
    if config["final_evaluation_performed"] is not False:
        raise ValueError("the frozen config cannot contain final results")
    selected = config["selected_model"]
    if (
        selected["id"] != FINAL_MODEL_ID
        or selected["selected_tuning_variant"] != "bge_svc_c_4_0"
    ):
        raise ValueError("the exact selected V2-C3 model changed")
    representation = selected["representation"]
    if (
        representation["model_identifier"] != "BAAI/bge-small-en-v1.5"
        or representation["dimensions"] != 384
        or representation["fine_tuning"] is not False
        or representation["external_api"] is not False
    ):
        raise ValueError("the selected frozen BGE representation changed")
    classifier = selected["classifier"]
    if classifier["class"] != "LinearSVC":
        raise ValueError("the selected classifier family changed")
    expected_parameters = {
        "C": 4.0,
        "class_weight": "balanced",
        "random_state": 20260928,
        "penalty": "l2",
        "loss": "squared_hinge",
        "dual": "auto",
        "tol": 0.0001,
        "max_iter": 2000,
        "fit_intercept": True,
        "intercept_scaling": 1.0,
        "multi_class": "ovr",
        "verbose": 0,
    }
    if classifier["parameters"] != expected_parameters:
        raise ValueError("the selected LinearSVC parameters changed")
    training = selected["training"]
    if (
        training["expected_example_count"] != 8198
        or training["use_all_development_examples"] is not True
        or training["cross_validation_at_final_fit"] is not False
        or training["automatic_embedding_regeneration"] is not False
        or training["final_evaluation_data_allowed"] is not False
    ):
        raise ValueError("the final development-only fit policy changed")
    lockbox = config["external_lockbox"]
    if lockbox["expected_count"] != 1922 or lockbox["expected_source_counts"] != {
        "banking77": 1346,
        "clinc150_oos": 576,
    }:
        raise ValueError("the external lockbox composition changed")
    if lockbox["allowed_source_slices"] != {
        "banking77": ["train"],
        "clinc150_oos": ["train", "val"],
    }:
        raise ValueError("the approved lockbox source slices changed")
    challenge = config["challenge_set"]
    if (
        challenge["expected_count"] != 270
        or tuple(challenge["expected_intent_counts"]) != EXPECTED_INTENTS
        or set(challenge["expected_intent_counts"].values()) != {30}
    ):
        raise ValueError("the balanced challenge composition changed")
    if config["evaluated_models"] != [FINAL_MODEL_ID, BASELINE_ID]:
        raise ValueError("final evaluation must contain exactly two frozen models")
    if config["comparator"]["eligible_to_replace_selected_model"] is not False:
        raise ValueError("the historical comparator cannot replace V2-C3")
    expected_baseline_parameters = {
        "C": 0.5,
        "class_weight": None,
        "random_state": 20260926,
        "penalty": "l2",
        "loss": "squared_hinge",
        "dual": "auto",
        "tol": 0.0001,
        "max_iter": 1000,
        "fit_intercept": True,
        "intercept_scaling": 1.0,
        "multi_class": "ovr",
        "verbose": 0,
    }
    baseline = config["comparator"]["definition"]
    if (
        baseline["classifier"]["class"] != "LinearSVC"
        or baseline["classifier"]["parameters"] != expected_baseline_parameters
        or baseline["representation"]["class"] != "FeatureUnion"
    ):
        raise ValueError("the frozen V2-C1 comparator definition changed")
    metrics = config["metrics"]
    if (
        metrics["datasets_reported_separately"] is not True
        or metrics["combined_final_metric_allowed"] is not False
        or metrics["protected_write_recall_zero_denominator"] != NOT_APPLICABLE
    ):
        raise ValueError("the final metric boundary changed")
    acceptance = config["final_acceptance_policy"]
    if acceptance["challenge_set"] != {
        "protected_write_false_positive_rate_maximum": 0.01,
        "protected_write_recall_minimum": 0.8,
        "unsupported_or_uncertain_recall_minimum": 0.8,
    }:
        raise ValueError("challenge safety acceptance gates changed")
    if acceptance["external_lockbox"] != {
        "protected_write_false_positive_rate_maximum": 0.01,
        "unsupported_or_uncertain_recall_minimum": 0.8,
        "protected_write_recall": (
            "NOT_APPLICABLE when the gold denominator is zero"
        ),
    }:
        raise ValueError("external lockbox safety acceptance gates changed")
    if (
        acceptance["macro_f1_or_accuracy_gate"] is not False
        or acceptance["retuning_after_final_results_allowed"] is not False
        or acceptance["model_switching_after_final_results_allowed"] is not False
    ):
        raise ValueError("post-final model selection is prohibited")
    if config["report"]["cfpb_used"] is not False:
        raise ValueError("CFPB must remain outside Step 6")
    report_contract = config["report"]
    if report_contract["dataset_result_sections"] != [
        "external_lockbox_results",
        "challenge_set_results",
    ] or report_contract["required_model_ids_per_dataset"] != [
        FINAL_MODEL_ID,
        BASELINE_ID,
    ]:
        raise ValueError("the separate final-report schema changed")
    validate_source_boundaries(config)


def validate_source_boundaries(config: dict[str, Any]) -> None:
    """Require the exact approved paths; prohibited splits cannot be substituted."""
    training = config["selected_model"]["training"]
    if training["dataset_path"] != "data/evals/v2/ml/v2c3_development_dataset.json":
        raise ValueError("final fitting must use only the frozen development dataset")
    lockbox = config["external_lockbox"]
    expected_source_paths = {
        "banking77_train": "data/evals/v2/external/raw/banking77/train.csv",
        "banking77_raw_manifest": "data/evals/v2/external/raw/banking77/manifest.json",
        "banking77_mapping": "data/evals/v2/external/banking77_intent_mapping.json",
        "clinc_source": "data/evals/v2/external/raw/clinc_oos/data_full.json",
        "clinc_raw_manifest": "data/evals/v2/external/raw/clinc_oos/manifest.json",
        "clinc_mapping": "data/evals/v2/external/clinc_finance_intent_mapping.json",
    }
    actual_source_paths = {
        name: spec["path"] for name, spec in lockbox["source_inputs"].items()
    }
    if actual_source_paths != expected_source_paths:
        raise ValueError("external reconstruction sources changed")
    if config["challenge_set"]["path"] != (
        "data/evals/v2/ml/v2c3_challenge_set.json"
    ):
        raise ValueError("challenge source changed")
    forbidden_fragments = ("cfpb", "banking77/test", "oos_test")
    governed_paths = [
        training["dataset_path"],
        lockbox["manifest_path"],
        config["challenge_set"]["path"],
        *actual_source_paths.values(),
    ]
    for path in governed_paths:
        lowered = path.lower()
        if any(fragment in lowered for fragment in forbidden_fragments):
            raise ValueError(f"prohibited Step 6 source path: {path}")


def validate_hash(path: Path, expected: str, label: str) -> str:
    actual = tournament.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def validate_hashed_spec(spec: Mapping[str, Any], label: str) -> Path:
    path = repository_path(str(spec["path"]))
    validate_hash(path, str(spec["sha256"]), label)
    return path


def validate_selection_lineage(config: dict[str, Any]) -> dict[str, Any]:
    frozen = config["frozen_input_hashes"]
    paths = {
        name: validate_hashed_spec(spec, name)
        for name, spec in frozen.items()
        if name
        not in {"development_embedding_cache", "development_embedding_cache_metadata"}
    }
    tuning_report = read_json(paths["tuning_report"])
    if tuning_report["overall_tuned_winner"] != "bge_svc_c_4_0":
        raise ValueError("the tuning report does not select bge_svc_c_4_0")
    if tuning_report["no_eligible_tuned_winner"] is not False:
        raise ValueError("the selected tuning winner is not eligible")
    selected_results = [
        row
        for row in tuning_report["variant_results"]
        if row["candidate_id"] == "bge_svc_c_4_0"
    ]
    if len(selected_results) != 1:
        raise ValueError("the tuning report must contain one selected result")
    selected_definition = selected_results[0]["definition"]
    if selected_definition["classifier_definition"] != config["selected_model"][
        "classifier"
    ]:
        raise ValueError("selected classifier differs from the tuning report")
    selected_representation = selected_definition["representation_definition"]
    configured_representation = config["selected_model"]["representation"]
    if (
        selected_representation["model_identifier"]
        != configured_representation["model_identifier"]
        or selected_representation["dimensions"]
        != configured_representation["dimensions"]
        or selected_representation["fine_tuning"]
        != configured_representation["fine_tuning"]
        or selected_representation["external_api"]
        != configured_representation["external_api"]
        or selected_representation["transform"] != "fastembed_passage_embedding"
        or configured_representation["fastembed_method"] != "passage_embed"
    ):
        raise ValueError("selected representation differs from the tuning report")
    return {"paths": paths, "tuning_report": tuning_report}


def validate_comparator(
    config: dict[str, Any], *, load_artifact: bool
) -> dict[str, Any]:
    comparator = config["comparator"]
    artifact_path = repository_path(comparator["artifact_path"])
    validate_hash(artifact_path, comparator["artifact_sha256"], "V2-C1 artifact")
    if artifact_path.stat().st_size != comparator["artifact_size_bytes"]:
        raise ValueError("V2-C1 artifact size changed")
    for prefix in ("model_selection", "classifier_config", "classifier_report"):
        validate_hash(
            repository_path(comparator[f"{prefix}_path"]),
            comparator[f"{prefix}_sha256"],
            f"V2-C1 {prefix}",
        )
    model_selection = read_json(repository_path(comparator["model_selection_path"]))
    if model_selection["selected_intent_model"] != {
        "c": 0.5,
        "model": "linear_svm",
        "probability_metrics": None,
        "protected_write_recall": 0.875,
        "validation_accuracy": 0.8425925925925926,
        "validation_macro_f1": 0.8438335416596286,
    }:
        raise ValueError("V2-C1 selected intent model changed")
    result: dict[str, Any] = {"artifact_path": artifact_path}
    if not load_artifact:
        return result
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise TypeError("V2-C1 artifact must contain a dictionary")
    if artifact.get("trusted_local_artifact") is not True:
        raise ValueError("V2-C1 comparator is not marked trusted local")
    if artifact.get("runtime_authority") is not False:
        raise ValueError("V2-C1 comparator cannot have runtime authority")
    if tuple(artifact.get("intent_label_order", [])) != tuple(
        model_selection["intent_label_order"]
    ):
        raise ValueError("V2-C1 comparator label order changed")
    intent_model = artifact[comparator["artifact_component"]]
    parameters = intent_model.get_params(deep=True)
    frozen_definition = comparator["definition"]
    for name, expected in frozen_definition["classifier"]["parameters"].items():
        if parameters[f"classifier__{name}"] != expected:
            raise ValueError(f"V2-C1 classifier parameter changed: {name}")
    for branch, prefix in (("word_tfidf", "word"), ("character_tfidf", "character")):
        for name, expected in frozen_definition["representation"][branch].items():
            actual = parameters[f"features__{prefix}__{name}"]
            if name == "ngram_range":
                actual = list(actual)
            if actual != expected:
                raise ValueError(f"V2-C1 feature parameter changed: {branch}.{name}")
    result["intent_model"] = intent_model
    return result


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    lineage = validate_selection_lineage(config)
    comparator = validate_comparator(config, load_artifact=False)
    for source_name, spec in config["external_lockbox"]["source_inputs"].items():
        validate_hashed_spec(spec, f"lockbox source {source_name}")
    lockbox_manifest_path = repository_path(config["external_lockbox"]["manifest_path"])
    validate_hash(
        lockbox_manifest_path,
        config["external_lockbox"]["manifest_sha256"],
        "external lockbox manifest",
    )
    lockbox_manifest = read_json(lockbox_manifest_path)
    lockbox_counts = Counter(row["source_id"] for row in lockbox_manifest["members"])
    if (
        lockbox_manifest["member_count"]
        != config["external_lockbox"]["expected_count"]
        or len(lockbox_manifest["members"]) != lockbox_manifest["member_count"]
    ):
        raise ValueError("external lockbox count changed")
    if dict(sorted(lockbox_counts.items())) != config["external_lockbox"][
        "expected_source_counts"
    ]:
        raise ValueError("external lockbox source counts changed")
    if lockbox_manifest["contains_text"] is not False:
        raise ValueError("external lockbox manifest must remain text-free")
    if any("text" in member for member in lockbox_manifest["members"]):
        raise ValueError("external lockbox manifest contains raw text")
    lockbox_intents = dict(
        sorted(Counter(row["intent"] for row in lockbox_manifest["members"]).items())
    )
    if lockbox_intents != config["external_lockbox"]["expected_intent_counts"]:
        raise ValueError("external lockbox intent counts changed")
    challenge_path = repository_path(config["challenge_set"]["path"])
    challenge_manifest_path = repository_path(config["challenge_set"]["manifest_path"])
    validate_hash(challenge_path, config["challenge_set"]["sha256"], "challenge set")
    validate_hash(
        challenge_manifest_path,
        config["challenge_set"]["manifest_sha256"],
        "challenge manifest",
    )
    challenge_manifest = read_json(challenge_manifest_path)
    if challenge_manifest["counts"]["example_count"] != 270:
        raise ValueError("challenge count changed")
    if challenge_manifest["counts_by_intent"] != config["challenge_set"][
        "expected_intent_counts"
    ]:
        raise ValueError("challenge intent balance changed")
    step_4_config = read_json(lineage["paths"]["tournament_config"])
    tournament.validate_config(step_4_config)
    _, development_examples, _ = tournament.load_development_data(step_4_config)
    if len(development_examples) != 8198:
        raise ValueError("development example count changed")
    cache_spec = config["frozen_input_hashes"]["development_embedding_cache"]
    cache_metadata_spec = config["frozen_input_hashes"][
        "development_embedding_cache_metadata"
    ]
    cache_path = validate_hashed_spec(cache_spec, "development BGE cache")
    cache_metadata_path = validate_hashed_spec(
        cache_metadata_spec, "development BGE cache metadata"
    )
    development_embeddings = tournament.validate_embedding_cache(
        step_4_config, development_examples
    )
    final_artifact_path = repository_path(config["selected_model"]["artifact"]["path"])
    return {
        "status": "ready",
        "model_fitting_performed": False,
        "model_inference_performed": False,
        "final_text_parsed": False,
        "final_test_embeddings_generated": False,
        "selected_tuning_variant": "bge_svc_c_4_0",
        "development_example_count": len(development_examples),
        "development_embedding_shape": list(development_embeddings.shape),
        "external_lockbox_count": lockbox_manifest["member_count"],
        "challenge_count": challenge_manifest["counts"]["example_count"],
        "comparator_artifact_sha256": config["comparator"]["artifact_sha256"],
        "development_cache_sha256": tournament.sha256_file(cache_path),
        "development_cache_metadata_sha256": tournament.sha256_file(
            cache_metadata_path
        ),
        "selection_lineage_valid": bool(lineage),
        "comparator_present": comparator["artifact_path"].is_file(),
        "final_model_artifact_present": final_artifact_path.is_file(),
        "cfpb_used": False,
        "forbidden_historical_test_source_used": False,
    }


def load_model_preparation_inputs(config: dict[str, Any]) -> dict[str, Any]:
    lineage = validate_selection_lineage(config)
    step_4_config = read_json(lineage["paths"]["tournament_config"])
    tournament.validate_config(step_4_config)
    _, examples, _ = tournament.load_development_data(step_4_config)
    if len(examples) != config["selected_model"]["training"]["expected_example_count"]:
        raise ValueError("development example count changed")
    cache_spec = config["frozen_input_hashes"]["development_embedding_cache"]
    cache_metadata_spec = config["frozen_input_hashes"][
        "development_embedding_cache_metadata"
    ]
    validate_hashed_spec(cache_spec, "development BGE cache")
    validate_hashed_spec(cache_metadata_spec, "development BGE cache metadata")
    embeddings = tournament.validate_embedding_cache(step_4_config, examples)
    return {
        "examples": examples,
        "embeddings": embeddings,
        "step_4_config": step_4_config,
    }


def prepare_final_model(
    config: dict[str, Any], inputs: dict[str, Any]
) -> tuple[Path, Path]:
    labels = np.asarray([row["intent"] for row in inputs["examples"]])
    classifier = LinearSVC(**config["selected_model"]["classifier"]["parameters"])
    classifier.fit(inputs["embeddings"], labels)
    artifact_spec = config["selected_model"]["artifact"]
    artifact_path = repository_path(artifact_spec["path"])
    metadata_path = repository_path(artifact_spec["metadata_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "artifact_schema_version": artifact_spec["artifact_schema_version"],
        "selected_model_id": FINAL_MODEL_ID,
        "selected_tuning_variant": "bge_svc_c_4_0",
        "representation": config["selected_model"]["representation"],
        "classifier_parameters": config["selected_model"]["classifier"]["parameters"],
        "intent_label_order": inputs["step_4_config"]["intent_taxonomy"],
        "classifier": classifier,
        "training_example_count": len(inputs["examples"]),
        "trusted_local_artifact": True,
        "runtime_authority": False,
        "final_evaluation_performed": False,
    }
    with tempfile.NamedTemporaryFile(
        dir=artifact_path.parent, suffix=".joblib", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        joblib.dump(artifact_payload, temporary_path, compress=3)
        temporary_path.replace(artifact_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    metadata_payload = {
        "schema_version": artifact_spec["metadata_schema_version"],
        "artifact_path": artifact_spec["path"],
        "artifact_sha256": tournament.sha256_file(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "selected_model_id": FINAL_MODEL_ID,
        "selected_tuning_variant": "bge_svc_c_4_0",
        "classifier_parameters": config["selected_model"]["classifier"]["parameters"],
        "development_dataset_sha256": config["frozen_input_hashes"][
            "development_dataset"
        ]["sha256"],
        "development_embedding_cache_sha256": config["frozen_input_hashes"][
            "development_embedding_cache"
        ]["sha256"],
        "development_embedding_cache_metadata_sha256": config[
            "frozen_input_hashes"
        ]["development_embedding_cache_metadata"]["sha256"],
        "bge_model_identifier": config["selected_model"]["representation"][
            "model_identifier"
        ],
        "bge_dimensions": config["selected_model"]["representation"]["dimensions"],
        "ordered_example_ids_sha256": tournament.ordered_ids_sha256(
            inputs["examples"]
        ),
        "training_example_count": len(inputs["examples"]),
        "tuning_config_sha256": config["frozen_input_hashes"]["tuning_config"][
            "sha256"
        ],
        "tuning_report_sha256": config["frozen_input_hashes"]["tuning_report"][
            "sha256"
        ],
        "final_evaluation_config_sha256": tournament.sha256_file(CONFIG_PATH),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "development_only_training": True,
        "final_evaluation_performed": False,
        "final_evaluation_data_used_for_training": False,
        "runtime_authority": False,
    }
    tournament.write_json(metadata_path, metadata_payload)
    return artifact_path, metadata_path


def validate_final_model_artifact(config: dict[str, Any]) -> dict[str, Any]:
    artifact_spec = config["selected_model"]["artifact"]
    artifact_path = repository_path(artifact_spec["path"])
    metadata_path = repository_path(artifact_spec["metadata_path"])
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "prepared final model and metadata are required before evaluate"
        )
    model_metadata = read_json(metadata_path)
    if model_metadata["schema_version"] != artifact_spec["metadata_schema_version"]:
        raise ValueError("unexpected final-model metadata schema")
    if model_metadata["artifact_sha256"] != tournament.sha256_file(artifact_path):
        raise ValueError("final-model artifact hash differs from metadata")
    if model_metadata["artifact_size_bytes"] != artifact_path.stat().st_size:
        raise ValueError("final-model artifact size differs from metadata")
    if model_metadata["selected_tuning_variant"] != "bge_svc_c_4_0":
        raise ValueError("final-model metadata identifies a different variant")
    if (
        model_metadata["selected_model_id"] != FINAL_MODEL_ID
        or model_metadata["training_example_count"] != 8198
        or model_metadata["bge_model_identifier"]
        != config["selected_model"]["representation"]["model_identifier"]
        or model_metadata["bge_dimensions"] != 384
    ):
        raise ValueError("final-model metadata definition changed")
    if model_metadata["classifier_parameters"] != config["selected_model"][
        "classifier"
    ]["parameters"]:
        raise ValueError("final-model metadata parameters changed")
    expected_metadata_lineage = {
        "development_dataset_sha256": config["frozen_input_hashes"][
            "development_dataset"
        ]["sha256"],
        "development_embedding_cache_sha256": config["frozen_input_hashes"][
            "development_embedding_cache"
        ]["sha256"],
        "development_embedding_cache_metadata_sha256": config[
            "frozen_input_hashes"
        ]["development_embedding_cache_metadata"]["sha256"],
        "ordered_example_ids_sha256": config["frozen_input_hashes"][
            "development_embedding_cache_metadata"
        ]["ordered_example_ids_sha256"],
        "tuning_config_sha256": config["frozen_input_hashes"]["tuning_config"][
            "sha256"
        ],
        "tuning_report_sha256": config["frozen_input_hashes"]["tuning_report"][
            "sha256"
        ],
        "final_evaluation_config_sha256": tournament.sha256_file(CONFIG_PATH),
    }
    for field, expected in expected_metadata_lineage.items():
        if model_metadata[field] != expected:
            raise ValueError(f"final-model metadata lineage changed: {field}")
    for field in (
        "development_only_training",
        "final_evaluation_data_used_for_training",
        "final_evaluation_performed",
        "runtime_authority",
    ):
        expected = field == "development_only_training"
        if model_metadata[field] is not expected:
            raise ValueError(f"invalid final-model metadata flag: {field}")
    artifact = joblib.load(artifact_path)
    if artifact["artifact_schema_version"] != artifact_spec["artifact_schema_version"]:
        raise ValueError("unexpected final-model artifact schema")
    if artifact["selected_tuning_variant"] != "bge_svc_c_4_0":
        raise ValueError("final-model artifact identifies a different variant")
    if artifact["selected_model_id"] != FINAL_MODEL_ID:
        raise ValueError("final-model artifact identifies a different model")
    if artifact["classifier_parameters"] != config["selected_model"]["classifier"][
        "parameters"
    ]:
        raise ValueError("final-model artifact parameters changed")
    if artifact["training_example_count"] != 8198:
        raise ValueError("final-model artifact was not fit on all development examples")
    if (
        artifact["trusted_local_artifact"] is not True
        or artifact["runtime_authority"] is not False
        or artifact["final_evaluation_performed"] is not False
    ):
        raise ValueError("invalid final-model artifact boundary flags")
    if tuple(artifact["intent_label_order"]) != EXPECTED_INTENTS:
        raise ValueError("final-model artifact label order changed")
    classifier = artifact["classifier"]
    if not isinstance(classifier, LinearSVC) or classifier.get_params(deep=False) != (
        config["selected_model"]["classifier"]["parameters"]
    ):
        raise ValueError("serialized final classifier definition changed")
    if tuple(classifier.classes_) != EXPECTED_INTENTS:
        raise ValueError("serialized final classifier fitted classes changed")
    return {
        "artifact_path": artifact_path,
        "metadata_path": metadata_path,
        "metadata": model_metadata,
        "classifier": classifier,
        "intent_label_order": artifact["intent_label_order"],
    }


def _load_frozen_mappings(
    config: dict[str, Any], contract: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    sources = config["external_lockbox"]["source_inputs"]
    banking_mapping = read_json(repository_path(sources["banking77_mapping"]["path"]))
    clinc_mapping = read_json(repository_path(sources["clinc_mapping"]["path"]))
    intents = contract["intent_taxonomy"]
    return (
        development_builder.load_frozen_mappings(
            banking_mapping, development_builder.BANKING77_SOURCE_ID, intents
        ),
        development_builder.load_frozen_mappings(
            clinc_mapping, development_builder.CLINC_SOURCE_ID, intents
        ),
    )


def reconstruct_lockbox_members(
    members: Sequence[Mapping[str, Any]],
    banking_rows: Sequence[Mapping[str, str]],
    clinc_source: Mapping[str, Sequence[Sequence[str]]],
    banking_mappings: Mapping[str, Mapping[str, Any]],
    clinc_mappings: Mapping[str, Mapping[str, Any]],
    risk_by_intent: Mapping[str, str],
    banking_revision: str,
    clinc_revision: str,
) -> list[dict[str, Any]]:
    reconstructed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for member in members:
        example_id = str(member["example_id"])
        if example_id in seen_ids:
            raise ValueError(f"duplicate lockbox member: {example_id}")
        seen_ids.add(example_id)
        source_id = member["source_id"]
        split = member["source_split"]
        row_index = member["source_row_index"]
        if not isinstance(row_index, int) or row_index < 1:
            raise ValueError(f"invalid source row index: {example_id}")
        if source_id == development_builder.BANKING77_SOURCE_ID:
            if split != "train" or member["source_revision"] != banking_revision:
                raise ValueError(f"unapproved BANKING77 source identity: {example_id}")
            try:
                source_row = banking_rows[row_index - 1]
            except IndexError as exc:
                raise ValueError(f"missing BANKING77 source row: {example_id}") from exc
            text = source_row["text"]
            source_label = source_row["category"]
            expected_id = f"banking77:{banking_revision}:train:{row_index:06d}"
            mappings = banking_mappings
        elif source_id == development_builder.CLINC_SOURCE_ID:
            if (
                split not in {"train", "val"}
                or member["source_revision"] != clinc_revision
            ):
                raise ValueError(f"unapproved CLINC source identity: {example_id}")
            try:
                text, source_label = clinc_source[split][row_index - 1]
            except (KeyError, IndexError) as exc:
                raise ValueError(f"missing CLINC source row: {example_id}") from exc
            expected_id = f"clinc:{clinc_revision}:{split}:{row_index:06d}"
            mappings = clinc_mappings
        else:
            raise ValueError(f"unapproved lockbox source: {source_id}")
        if example_id != expected_id:
            raise ValueError(f"lockbox example ID mismatch: {example_id}")
        if source_label != member["source_label"]:
            raise ValueError(f"lockbox source label mismatch: {example_id}")
        mapping = mappings.get(source_label)
        if mapping is None:
            raise ValueError(f"frozen mapping missing for {example_id}")
        target = development_builder._mapped_target(mapping)
        if target != member["intent"] or mapping["mapping_status"] != member[
            "mapping_status"
        ]:
            raise ValueError(f"lockbox frozen target mismatch: {example_id}")
        if risk_by_intent[target] != member["risk"]:
            raise ValueError(f"lockbox frozen risk mismatch: {example_id}")
        if development_builder.text_sha256(text) != member["text_sha256"]:
            raise ValueError(f"lockbox text hash mismatch: {example_id}")
        if (
            development_builder.normalized_text_sha256(text)
            != member["normalized_text_sha256"]
        ):
            raise ValueError(f"lockbox normalized hash mismatch: {example_id}")
        row = dict(member)
        row["text"] = text
        reconstructed.append(row)
    return reconstructed


def reconstruct_external_lockbox(config: dict[str, Any]) -> list[dict[str, Any]]:
    lockbox = config["external_lockbox"]
    sources = lockbox["source_inputs"]
    manifest = read_json(repository_path(lockbox["manifest_path"]))
    contract = read_json(
        repository_path(config["frozen_input_hashes"]["experiment_contract"]["path"])
    )
    banking_bytes = repository_path(sources["banking77_train"]["path"]).read_bytes()
    banking_rows = development_builder._parse_banking77_rows(banking_bytes)
    clinc_payload = read_json(repository_path(sources["clinc_source"]["path"]))
    banking_raw_manifest = read_json(
        repository_path(sources["banking77_raw_manifest"]["path"])
    )
    clinc_raw_manifest = read_json(
        repository_path(sources["clinc_raw_manifest"]["path"])
    )
    banking_mappings, clinc_mappings = _load_frozen_mappings(config, contract)
    rows = reconstruct_lockbox_members(
        manifest["members"],
        banking_rows,
        {split: clinc_payload[split] for split in ("train", "val")},
        banking_mappings,
        clinc_mappings,
        contract["risk_by_intent"],
        banking_raw_manifest["source_revision"],
        clinc_raw_manifest["pinned_revision"],
    )
    if len(rows) != lockbox["expected_count"]:
        raise ValueError("reconstructed lockbox count changed")
    source_counts = dict(sorted(Counter(row["source_id"] for row in rows).items()))
    if source_counts != lockbox["expected_source_counts"]:
        raise ValueError("reconstructed lockbox source counts changed")
    intent_counts = dict(sorted(Counter(row["intent"] for row in rows).items()))
    if intent_counts != lockbox["expected_intent_counts"]:
        raise ValueError("reconstructed lockbox intent counts changed")
    return rows


def load_challenge_examples(config: dict[str, Any]) -> list[dict[str, Any]]:
    challenge = config["challenge_set"]
    payload = read_json(repository_path(challenge["path"]))
    if payload["schema_version"] != "v2c3-challenge-set.v1":
        raise ValueError("unexpected challenge schema")
    if (
        payload["example_count"] != challenge["expected_count"]
        or payload["data_role"] != challenge["data_role"]
        or payload["training_eligible"] is not False
        or payload["model_selection_eligible"] is not False
        or payload["threshold_selection_eligible"] is not False
    ):
        raise ValueError("challenge final-only policy changed")
    examples = payload["examples"]
    counts = dict(sorted(Counter(row["intent"] for row in examples).items()))
    if counts != challenge["expected_intent_counts"]:
        raise ValueError("challenge intent balance changed")
    for row in examples:
        if development_builder.text_sha256(row["text"]) != row["text_sha256"]:
            raise ValueError(f"challenge text hash mismatch: {row['example_id']}")
        if (
            development_builder.normalized_text_sha256(row["text"])
            != row["normalized_text_sha256"]
        ):
            raise ValueError(
                f"challenge normalized hash mismatch: {row['example_id']}"
            )
    return examples


def embed_final_texts(texts: Sequence[str], config: dict[str, Any]) -> np.ndarray:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for final evaluation") from exc
    model_identifier = config["selected_model"]["representation"]["model_identifier"]
    try:
        model = TextEmbedding(model_name=model_identifier)
        embeddings = np.asarray(
            list(model.passage_embed(list(texts), batch_size=256)), dtype=np.float32
        )
    except Exception as exc:
        raise RuntimeError(
            f"FastEmbed could not embed final data with {model_identifier}"
        ) from exc
    expected_shape = (
        len(texts),
        config["selected_model"]["representation"]["dimensions"],
    )
    if embeddings.shape != expected_shape or not np.isfinite(embeddings).all():
        raise ValueError(f"unexpected final embedding matrix: {embeddings.shape}")
    return embeddings


def final_dataset_metrics(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    step_4_config: dict[str, Any],
    *,
    include_challenge_classes: bool,
) -> dict[str, Any]:
    expected = [str(row["intent"]) for row in examples]
    predicted = list(predictions)
    labels = step_4_config["intent_taxonomy"]
    present_labels = [label for label in labels if label in set(expected)]
    common = tournament.classification_metrics(expected, predicted, labels)
    safety = tournament.safety_metrics(expected, predicted, step_4_config)
    protected_recall: float | str = safety["protected_write_recall"]
    if protected_recall is None:
        protected_recall = NOT_APPLICABLE
    result = {
        "count": len(examples),
        "accuracy": common["accuracy"],
        "macro_f1": float(
            f1_score(
                expected,
                predicted,
                labels=present_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_scope": "gold_classes_present",
        "weighted_f1": common["weighted_f1"],
        "per_class": common["per_class"],
        "confusion_matrix": common["confusion_matrix"],
        "supported_intent_accuracy": safety["supported_intent_accuracy"],
        "unsupported_or_uncertain_recall": safety[
            "unsupported_or_uncertain_recall"
        ],
        "false_supported_count": safety["false_supported_count"],
        "false_supported_rate": safety["false_supported_rate"],
        "protected_write_recall": protected_recall,
        "protected_write_false_positive_count": safety[
            "protected_write_false_positive_count"
        ],
        "protected_write_false_positive_rate": safety[
            "protected_write_false_positive_rate"
        ],
    }
    if include_challenge_classes:
        result.update(
            {
                "freeze_card_metrics": common["per_class"]["freeze_card"],
                "create_dispute_metrics": common["per_class"]["create_dispute"],
                "escalation_metrics": common["per_class"]["escalation"],
            }
        )
    return result


def external_metrics(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    step_4_config: dict[str, Any],
) -> dict[str, Any]:
    result = final_dataset_metrics(
        examples, predictions, step_4_config, include_challenge_classes=False
    )
    result["source_breakdown"] = {}
    for source_id in ("banking77", "clinc150_oos"):
        indices = [
            index for index, row in enumerate(examples) if row["source_id"] == source_id
        ]
        source_examples = [examples[index] for index in indices]
        source_predictions = [predictions[index] for index in indices]
        result["source_breakdown"][source_id] = final_dataset_metrics(
            source_examples,
            source_predictions,
            step_4_config,
            include_challenge_classes=False,
        )
    return result


def comparison_deltas(
    final_metrics: Mapping[str, Any], baseline_metrics: Mapping[str, Any]
) -> dict[str, float | str]:
    fields = {
        "accuracy_delta": "accuracy",
        "macro_f1_delta": "macro_f1",
        "unsupported_recall_delta": "unsupported_or_uncertain_recall",
        "false_supported_rate_delta": "false_supported_rate",
        "protected_write_recall_delta": "protected_write_recall",
        "protected_write_fpr_delta": "protected_write_false_positive_rate",
    }
    deltas: dict[str, float | str] = {}
    for output_name, metric_name in fields.items():
        final_value = final_metrics[metric_name]
        baseline_value = baseline_metrics[metric_name]
        if final_value == NOT_APPLICABLE or baseline_value == NOT_APPLICABLE:
            deltas[output_name] = NOT_APPLICABLE
        else:
            deltas[output_name] = float(final_value) - float(baseline_value)
    return deltas


def final_acceptance(
    external: Mapping[str, Any], challenge: Mapping[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    policy = config["final_acceptance_policy"]
    checks = {
        "challenge_protected_write_false_positive_rate": {
            "value": challenge["protected_write_false_positive_rate"],
            "operator": "<=",
            "threshold": policy["challenge_set"][
                "protected_write_false_positive_rate_maximum"
            ],
        },
        "challenge_protected_write_recall": {
            "value": challenge["protected_write_recall"],
            "operator": ">=",
            "threshold": policy["challenge_set"]["protected_write_recall_minimum"],
        },
        "challenge_unsupported_recall": {
            "value": challenge["unsupported_or_uncertain_recall"],
            "operator": ">=",
            "threshold": policy["challenge_set"][
                "unsupported_or_uncertain_recall_minimum"
            ],
        },
        "external_protected_write_false_positive_rate": {
            "value": external["protected_write_false_positive_rate"],
            "operator": "<=",
            "threshold": policy["external_lockbox"][
                "protected_write_false_positive_rate_maximum"
            ],
        },
        "external_unsupported_recall": {
            "value": external["unsupported_or_uncertain_recall"],
            "operator": ">=",
            "threshold": policy["external_lockbox"][
                "unsupported_or_uncertain_recall_minimum"
            ],
        },
    }
    for check in checks.values():
        if check["operator"] == "<=":
            check["passed"] = check["value"] <= check["threshold"]
        else:
            check["passed"] = check["value"] >= check["threshold"]
    return {
        "checks": checks,
        "final_safety_acceptable": all(check["passed"] for check in checks.values()),
        "macro_f1_or_accuracy_used_as_gate": False,
        "retuning_allowed_after_result": False,
    }


def resolved_frozen_hashes(config: dict[str, Any]) -> dict[str, Any]:
    """Return every frozen artifact hash recorded by the Step 6 contract."""
    return {
        "selection_lineage": {
            name: tournament.sha256_file(repository_path(spec["path"]))
            for name, spec in config["frozen_input_hashes"].items()
        },
        "external_lockbox": {
            "manifest": tournament.sha256_file(
                repository_path(config["external_lockbox"]["manifest_path"])
            ),
            "source_inputs": {
                name: tournament.sha256_file(repository_path(spec["path"]))
                for name, spec in config["external_lockbox"][
                    "source_inputs"
                ].items()
            },
        },
        "challenge_set": {
            "data": tournament.sha256_file(
                repository_path(config["challenge_set"]["path"])
            ),
            "manifest": tournament.sha256_file(
                repository_path(config["challenge_set"]["manifest_path"])
            ),
        },
        "comparator": {
            "artifact": tournament.sha256_file(
                repository_path(config["comparator"]["artifact_path"])
            ),
            "model_selection": tournament.sha256_file(
                repository_path(config["comparator"]["model_selection_path"])
            ),
            "classifier_config": tournament.sha256_file(
                repository_path(config["comparator"]["classifier_config_path"])
            ),
            "classifier_report": tournament.sha256_file(
                repository_path(config["comparator"]["classifier_report_path"])
            ),
        },
    }


def validate_report_payload(report: dict[str, Any], config: dict[str, Any]) -> None:
    required_fields = config["report"]["required_top_level_fields"]
    if set(report) != set(required_fields):
        missing = sorted(set(required_fields) - set(report))
        extra = sorted(set(report) - set(required_fields))
        raise ValueError(
            f"final report schema mismatch: missing={missing}, extra={extra}"
        )
    for section in config["report"]["dataset_result_sections"]:
        if list(report[section]) != config["report"][
            "required_model_ids_per_dataset"
        ]:
            raise ValueError(f"final report model set changed: {section}")
    if (
        report["raw_text_persisted"] is not False
        or report["combined_final_metric_reported"] is not False
        or report["cfpb_used"] is not False
    ):
        raise ValueError("final report boundary flags changed")


def evaluate(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    integrity_verification = preflight(config)
    final_model = validate_final_model_artifact(config)
    comparator = validate_comparator(config, load_artifact=True)
    step_4_config = read_json(
        repository_path(config["frozen_input_hashes"]["tournament_config"]["path"])
    )
    lockbox = reconstruct_external_lockbox(config)
    challenge = load_challenge_examples(config)
    lockbox_texts = [row["text"] for row in lockbox]
    challenge_texts = [row["text"] for row in challenge]
    lockbox_embeddings = embed_final_texts(lockbox_texts, config)
    challenge_embeddings = embed_final_texts(challenge_texts, config)
    final_lockbox_predictions = final_model["classifier"].predict(
        lockbox_embeddings
    ).tolist()
    final_challenge_predictions = final_model["classifier"].predict(
        challenge_embeddings
    ).tolist()
    baseline_lockbox_predictions = comparator["intent_model"].predict(
        lockbox_texts
    ).tolist()
    baseline_challenge_predictions = comparator["intent_model"].predict(
        challenge_texts
    ).tolist()
    external_results = {
        FINAL_MODEL_ID: external_metrics(
            lockbox, final_lockbox_predictions, step_4_config
        ),
        BASELINE_ID: external_metrics(
            lockbox, baseline_lockbox_predictions, step_4_config
        ),
    }
    challenge_results = {
        FINAL_MODEL_ID: final_dataset_metrics(
            challenge,
            final_challenge_predictions,
            step_4_config,
            include_challenge_classes=True,
        ),
        BASELINE_ID: final_dataset_metrics(
            challenge,
            baseline_challenge_predictions,
            step_4_config,
            include_challenge_classes=True,
        ),
    }
    acceptance = final_acceptance(
        external_results[FINAL_MODEL_ID], challenge_results[FINAL_MODEL_ID], config
    )
    model_metadata = final_model["metadata"]
    report = {
        "schema_version": config["report"]["schema_version"],
        "experiment_phase": "V2-C3-Step-6",
        "final_evaluation_config_sha256": tournament.sha256_file(CONFIG_PATH),
        "frozen_input_hashes": resolved_frozen_hashes(config),
        "final_v2c3_artifact_sha256": model_metadata["artifact_sha256"],
        "final_v2c3_artifact_metadata_sha256": tournament.sha256_file(
            final_model["metadata_path"]
        ),
        "comparator_artifact_sha256": config["comparator"]["artifact_sha256"],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "fastembed": metadata.version("fastembed"),
            "joblib": metadata.version("joblib"),
        },
        "dataset_integrity": {
            "external_lockbox": {
                "count": len(lockbox),
                "source_counts": dict(
                    sorted(Counter(row["source_id"] for row in lockbox).items())
                ),
                "all_member_hashes_verified": True,
                "membership_changed": False,
            },
            "challenge_set": {
                "count": len(challenge),
                "intent_counts": dict(
                    sorted(Counter(row["intent"] for row in challenge).items())
                ),
                "all_example_hashes_verified": True,
            },
        },
        "integrity_verification": integrity_verification,
        "external_lockbox_results": external_results,
        "challenge_set_results": challenge_results,
        "comparison_role": "final selected model vs frozen historical baseline",
        "comparison_deltas": {
            "external_lockbox": comparison_deltas(
                external_results[FINAL_MODEL_ID], external_results[BASELINE_ID]
            ),
            "challenge_set": comparison_deltas(
                challenge_results[FINAL_MODEL_ID], challenge_results[BASELINE_ID]
            ),
        },
        "final_acceptance_policy": config["final_acceptance_policy"],
        "safety_gate_results": acceptance["checks"],
        "final_safety_acceptable": acceptance["final_safety_acceptable"],
        "model_selection_completed_before_final_evaluation": True,
        "final_lockbox_used": True,
        "challenge_set_used": True,
        "cfpb_used": False,
        "hyperparameter_tuning_on_final_data": False,
        "threshold_tuning_on_final_data": False,
        "model_switching_after_final_results": False,
        "final_model_training_used_development_only": True,
        "combined_final_metric_reported": False,
        "raw_text_persisted": False,
        "runtime_integration_performed": False,
    }
    validate_report_payload(report, config)
    report_path = repository_path(config["report"]["path"])
    tournament.write_json(report_path, report)
    return report, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "prepare-final-model", "evaluate")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
        return
    if args.mode == "prepare-final-model":
        inputs = load_model_preparation_inputs(config)
        artifact_path, metadata_path = prepare_final_model(config, inputs)
        print(f"wrote final classifier: {artifact_path.relative_to(REPOSITORY_ROOT)}")
        print(f"wrote model metadata: {metadata_path.relative_to(REPOSITORY_ROOT)}")
        return
    _, report_path = evaluate(config)
    print(f"wrote final evaluation report: {report_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
