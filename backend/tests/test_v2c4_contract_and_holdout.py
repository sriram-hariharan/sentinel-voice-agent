from __future__ import annotations

import ast
import copy
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import build_v2c4_safety_holdout as builder

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "data/evals/v2/ml/v2c4_experiment_contract.json"
ERROR_CONFIG_PATH = ROOT / "data/evals/v2/ml/v2c4_error_analysis_config.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return load_json(CONTRACT_PATH)


@pytest.fixture(scope="module")
def error_config() -> dict[str, Any]:
    return load_json(ERROR_CONFIG_PATH)


@pytest.fixture(scope="module")
def artifacts() -> builder.BuiltArtifacts:
    return builder.build_artifacts()


def test_v2c3_final_sets_are_consumed_not_untouched(
    contract: dict[str, Any],
) -> None:
    consumed = contract["consumed_data_policy"]
    challenge = consumed["v2c3_challenge_set"]
    external = consumed["v2c3_external_lockbox"]

    assert challenge["diagnostic_role"] == "consumed_v2c3_safety_challenge"
    assert external["diagnostic_role"] == "consumed_v2c3_external_regression"
    for policy in (challenge, external):
        assert policy["v2c4_final_acceptance_eligible"] is False
        assert policy["may_be_described_as_untouched"] is False
        assert policy["model_selection_eligible"] is False
        assert policy["threshold_selection_eligible"] is False


def test_frozen_v2c3_outcome_is_recorded_without_reinterpretation(
    contract: dict[str, Any],
) -> None:
    outcome = contract["frozen_v2c3_outcome"]
    assert outcome["external_lockbox_required_safety_gates_passed"] is True
    assert outcome["challenge_protected_write_false_positive_rate"] == (
        0.01904761904761905
    )
    assert outcome["challenge_protected_write_recall"] == 0.75
    assert outcome["challenge_unsupported_or_uncertain_recall"] == 0.7
    assert outcome["failed_challenge_gate_count"] == 3
    assert outcome["final_safety_acceptable"] is False
    assert outcome["reinterpretation_allowed"] is False
    assert outcome["same_phase_retuning_allowed"] is False


def test_parent_v2c3_hashes_match_frozen_files(contract: dict[str, Any]) -> None:
    for name, spec in contract["parent_v2c3_hashes"].items():
        assert builder.development_builder.sha256_bytes(
            (ROOT / spec["path"]).read_bytes()
        ) == spec["sha256"], name


def test_safety_gates_are_unchanged(contract: dict[str, Any]) -> None:
    assert contract["acceptance_gates"] == {
        "protected_write_false_positive_rate_maximum": 0.01,
        "protected_write_recall_minimum": 0.8,
        "unsupported_or_uncertain_recall_minimum": 0.8,
        "all_gates_required": True,
        "post_hoc_relaxation_allowed": False,
        "macro_f1_or_accuracy_gate": False,
    }
    assert contract["safety_metrics"]["definitions_unchanged_from_v2c3"] is True


def test_deterministic_rebuild_matches_tracked_artifacts(
    artifacts: builder.BuiltArtifacts,
) -> None:
    repeated = builder.build_artifacts()
    assert repeated.holdout_bytes == artifacts.holdout_bytes
    assert repeated.manifest_bytes == artifacts.manifest_bytes
    assert artifacts.holdout_bytes == builder.DEFAULT_PATHS.holdout_output.read_bytes()
    assert artifacts.manifest_bytes == (
        builder.DEFAULT_PATHS.manifest_output.read_bytes()
    )


def test_holdout_has_exactly_360_and_40_per_intent(
    artifacts: builder.BuiltArtifacts, contract: dict[str, Any]
) -> None:
    examples = artifacts.holdout_payload["examples"]
    counts = Counter(row["intent"] for row in examples)
    assert len(examples) == 360
    assert set(counts) == set(contract["intent_taxonomy"])
    assert set(counts.values()) == {40}
    assert artifacts.manifest_payload["counts_by_intent"] == dict(
        sorted(counts.items())
    )
    assert artifacts.manifest_payload["validation"][
        "balanced_40_per_intent"
    ] is True


def test_every_record_is_fresh_final_holdout_only(
    artifacts: builder.BuiltArtifacts,
) -> None:
    payload = artifacts.holdout_payload
    policy = artifacts.manifest_payload["evaluation_policy"]
    examples = payload["examples"]
    assert payload["data_role"] == "v2c4_final_safety_holdout"
    assert policy["data_role"] == "v2c4_final_safety_holdout"
    for field in (
        "training_eligible",
        "model_selection_eligible",
        "threshold_selection_eligible",
        "error_analysis_eligible",
    ):
        assert payload[field] is False
        assert policy[field] is False
        assert all(row[field] is False for row in examples)
    assert all(
        "prediction" not in row and "predicted_intent" not in row
        for row in examples
    )
    assert payload["model_predictions_present"] is False
    assert payload["training_or_evaluation_performed"] is False


def test_holdout_is_disjoint_from_v2c3_development_challenge_and_lockbox(
    artifacts: builder.BuiltArtifacts,
) -> None:
    hashes = {
        row["normalized_text_sha256"]
        for row in artifacts.holdout_payload["examples"]
    }
    development = load_json(builder.DEFAULT_PATHS.v2c3_development)
    challenge = load_json(builder.DEFAULT_PATHS.v2c3_challenge)
    lockbox = load_json(builder.DEFAULT_PATHS.v2c3_external_lockbox_manifest)
    reference_sets = (
        {row["normalized_text_sha256"] for row in development["examples"]},
        {row["normalized_text_sha256"] for row in challenge["examples"]},
        {row["normalized_text_sha256"] for row in lockbox["members"]},
    )
    assert len(hashes) == 360
    assert all(hashes.isdisjoint(reference) for reference in reference_sets)
    validation = artifacts.manifest_payload["validation"]
    assert validation["v2c3_development_normalized_hash_overlap"] == 0
    assert validation["consumed_v2c3_challenge_normalized_hash_overlap"] == 0
    assert validation[
        "consumed_v2c3_external_lockbox_normalized_hash_overlap"
    ] == 0


def test_lineage_is_unique_and_separate_from_v2c3_challenge(
    artifacts: builder.BuiltArtifacts,
) -> None:
    lineages = {
        row["lineage_id"] for row in artifacts.holdout_payload["examples"]
    }
    old_challenge = load_json(builder.DEFAULT_PATHS.v2c3_challenge)
    old_lineages = {row["lineage_id"] for row in old_challenge["examples"]}
    assert len(lineages) == 360
    assert lineages.isdisjoint(old_lineages)
    assert all(lineage.startswith("v2c4-holdout:") for lineage in lineages)
    assert artifacts.manifest_payload["validation"][
        "lineage_overlap_with_v2c3_challenge"
    ] == 0


def test_safety_sensitive_pattern_coverage_is_complete(
    artifacts: builder.BuiltArtifacts,
) -> None:
    examples = artifacts.holdout_payload["examples"]
    patterns = {
        row["safety_pattern"]
        for row in examples
        if row["intent"] in builder.SAFETY_SENSITIVE_INTENTS
    }
    assert builder.REQUIRED_SAFETY_PATTERNS <= patterns
    assert artifacts.manifest_payload["validation"][
        "required_safety_patterns_present"
    ] is True


def test_every_non_protected_lane_has_protected_mention_hard_negatives(
    artifacts: builder.BuiltArtifacts, contract: dict[str, Any]
) -> None:
    examples = artifacts.holdout_payload["examples"]
    counts = Counter(
        row["intent"]
        for row in examples
        if builder.PROTECTED_MENTION_HARD_NEGATIVE in row["tags"]
    )
    non_protected = set(contract["intent_taxonomy"]) - builder.PROTECTED_WRITE_INTENTS
    assert all(counts[intent] >= 4 for intent in non_protected)
    assert artifacts.manifest_payload["validation"][
        "required_hard_negative_categories_present"
    ] is True


def test_protected_gold_examples_are_explicit_current_actions(
    artifacts: builder.BuiltArtifacts,
) -> None:
    protected = [
        row
        for row in artifacts.holdout_payload["examples"]
        if row["intent"] in builder.PROTECTED_WRITE_INTENTS
    ]
    assert len(protected) == 80
    assert all("explicit_current_action" in row["tags"] for row in protected)
    assert all("protected_write_positive" in row["tags"] for row in protected)


def test_human_reviewed_seed_corrections_are_frozen(
    artifacts: builder.BuiltArtifacts,
) -> None:
    examples = {
        row["example_id"]: row for row in artifacts.holdout_payload["examples"]
    }
    corrected_text = {
        "v2c4-holdout:account_balance:005": (
            "I was looking at a disputed charge, but right now I only need my "
            "account balance."
        ),
        "v2c4-holdout:card_status:038": (
            "Check the status of my travel card rather than my everyday card."
        ),
        "v2c4-holdout:escalation:010": (
            "Don't freeze my card or dispute anything; just transfer me to "
            "customer support."
        ),
        "v2c4-holdout:recent_transactions:010": (
            "Don't freeze my card or open a dispute; just show me my latest "
            "purchases."
        ),
        "v2c4-holdout:transaction_details:010": (
            "Don't freeze my card or dispute anything; just tell me more about "
            "that payment."
        ),
        "v2c4-holdout:unsupported_or_uncertain:005": (
            "I noticed the freeze-card option, but I don't want anything changed "
            "right now."
        ),
        "v2c4-holdout:unsupported_or_uncertain:008": (
            "I saw a dispute notice earlier, but I'm not asking you to do anything "
            "right now."
        ),
    }
    assert {
        example_id: examples[example_id]["text"] for example_id in corrected_text
    } == corrected_text
    assert "status" in examples["v2c4-holdout:card_status:038"]["text"].lower()
    assert all(
        "protected action" not in row["text"].lower()
        for row in examples.values()
    )

    for example_id in (
        "v2c4-holdout:unsupported_or_uncertain:005",
        "v2c4-holdout:unsupported_or_uncertain:008",
    ):
        assert examples[example_id]["intent"] == "unsupported_or_uncertain"
        assert builder.PROTECTED_MENTION_HARD_NEGATIVE in examples[example_id]["tags"]

    ambiguous = examples["v2c4-holdout:unsupported_or_uncertain:030"]
    assert ambiguous["text"] == "Handle the dispute thing for me somehow."
    assert ambiguous["intent"] == "unsupported_or_uncertain"


def test_information_then_current_action_is_a_protected_positive_pattern(
    artifacts: builder.BuiltArtifacts,
) -> None:
    examples = {
        row["example_id"]: row for row in artifacts.holdout_payload["examples"]
    }
    expected = {
        *(f"v2c4-holdout:freeze_card:{ordinal:03d}" for ordinal in range(33, 37)),
        *(
            f"v2c4-holdout:create_dispute:{ordinal:03d}"
            for ordinal in range(33, 37)
        ),
    }
    assert "information_then_current_action" in builder.REQUIRED_SAFETY_PATTERNS
    assert {
        row["example_id"]
        for row in examples.values()
        if row["safety_pattern"] == "information_then_current_action"
    } == expected
    for example_id in expected:
        assert examples[example_id]["safety_pattern"] == (
            "information_then_current_action"
        )
        assert examples[example_id]["intent"] in builder.PROTECTED_WRITE_INTENTS
        assert examples[example_id]["risk"] == "PROTECTED_WRITE"
        assert "protected_write_positive" in examples[example_id]["tags"]


def test_synthetic_overlap_and_old_lineage_are_rejected() -> None:
    seed = load_json(builder.DEFAULT_PATHS.seed)
    contract = load_json(builder.DEFAULT_PATHS.contract)
    first_family = seed["families"][0]
    first_text = first_family["texts"][0]
    first_lineage = f"v2c4-holdout:{first_family['family_id']}:01"

    overlap_reference = builder.ReferenceMetadata(
        normalized_hashes={
            builder.development_builder.normalized_text_sha256(first_text)
        },
        lineage_ids=set(),
        hash_counts={},
    )
    with pytest.raises(ValueError, match="overlaps V2-C3 data"):
        builder._build_records(
            seed,
            contract["intent_taxonomy"],
            contract["risk_by_intent"],
            overlap_reference,
        )

    lineage_reference = builder.ReferenceMetadata(
        normalized_hashes=set(),
        lineage_ids={first_lineage},
        hash_counts={},
    )
    with pytest.raises(ValueError, match="duplicates V2-C3 challenge lineage"):
        builder._build_records(
            seed,
            contract["intent_taxonomy"],
            contract["risk_by_intent"],
            lineage_reference,
        )


def test_normalized_duplicate_is_rejected() -> None:
    seed = load_json(builder.DEFAULT_PATHS.seed)
    contract = load_json(builder.DEFAULT_PATHS.contract)
    changed = copy.deepcopy(seed)
    changed["families"][0]["texts"][1] = changed["families"][0]["texts"][0]
    with pytest.raises(ValueError, match="normalized-text duplicates"):
        builder._build_records(
            changed,
            contract["intent_taxonomy"],
            contract["risk_by_intent"],
            builder.ReferenceMetadata(set(), set(), {}),
        )


def test_error_analysis_categories_are_frozen_before_results(
    error_config: dict[str, Any],
) -> None:
    expected = {
        "unsupported_to_supported",
        "supported_to_unsupported",
        "protected_to_wrong_protected_intent",
        "protected_to_non_protected",
        "non_protected_to_protected",
        "freeze_card_create_dispute_confusion",
        "informational_mention_vs_action_request",
        "ambiguity_or_insufficient_context",
        "lexical_trigger_over_reliance",
    }
    assert {row["id"] for row in error_config["categories"]} == expected
    assert error_config["frozen_before_error_analysis"] is True
    assert error_config["error_analysis_performed"] is False
    assert error_config["results_present"] is False
    assert error_config["scope"]["individual_example_review_started"] is False


def test_cfpb_and_historical_public_tests_remain_excluded(
    contract: dict[str, Any], artifacts: builder.BuiltArtifacts
) -> None:
    prohibited = " ".join(contract["prohibited_inputs"]).lower()
    assert "cfpb" in prohibited
    assert "banking77 test" in prohibited
    assert "clinc test" in prohibited
    assert "clinc oos_test" in prohibited
    assert contract["cfpb_policy"]["read_by_v2c4_holdout_builder"] is False
    input_paths = set(artifacts.manifest_payload["input_paths"])
    assert not any("cfpb" in name for name in input_paths)
    assert not any("banking77" in name for name in input_paths)
    assert not any("clinc" in name for name in input_paths)


def test_builder_has_no_training_inference_network_or_runtime_integration() -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots.isdisjoint(
        {
            "fastembed",
            "httpx",
            "joblib",
            "numpy",
            "openai",
            "requests",
            "sklearn",
            "torch",
            "urllib",
        }
    )
    called_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names.update(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    assert called_names.isdisjoint(
        {
            "cross_validate",
            "evaluate",
            "fit",
            "fit_transform",
            "predict",
            "predict_proba",
            "transform",
        }
    )
    assert "backend.app" not in source


def test_check_detects_artifact_drift(
    artifacts: builder.BuiltArtifacts, tmp_path: Path
) -> None:
    paths = replace(
        builder.DEFAULT_PATHS,
        holdout_output=tmp_path / "holdout.json",
        manifest_output=tmp_path / "manifest.json",
    )
    builder.write_artifacts(artifacts, paths)
    builder.check_artifacts(artifacts, paths)
    paths.holdout_output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from deterministic build"):
        builder.check_artifacts(artifacts, paths)
