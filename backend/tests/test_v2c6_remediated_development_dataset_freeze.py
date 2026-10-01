from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import validate_and_freeze_v2c6_remediated_development_dataset as freeze

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inputs() -> freeze.LoadedInputs:
    return freeze.load_inputs()


@pytest.fixture(scope="module")
def report(inputs: freeze.LoadedInputs) -> dict[str, Any]:
    return freeze.validate_inputs(inputs)


@pytest.fixture(scope="module")
def payload(
    inputs: freeze.LoadedInputs,
    report: dict[str, Any],
) -> dict[str, Any]:
    return freeze.build_freeze_payload(inputs, report)


def recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(recursive_keys(child))
    return keys


def group_record(group_id: str, intent: str) -> dict[str, str]:
    return {"group_id": group_id, "intent": intent}


def test_exact_9008_record_composition(report: dict[str, Any]) -> None:
    assert report["total_record_count"] == 9008
    assert report["inherited_record_count"] == 8198
    assert report["remediation_record_count"] == 810
    assert 8198 + 810 == 9008


def test_frozen_taxonomy_and_protected_intents_are_exact(
    report: dict[str, Any],
) -> None:
    assert tuple(report["taxonomy_intent_labels"]) == freeze.INTENT_LABEL_ORDER
    assert len(report["taxonomy_intent_labels"]) == 16
    assert tuple(report["protected_write_intents"]) == (
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    )
    assert set(report["risk_mapping"]) == set(freeze.INTENT_LABEL_ORDER)


def test_remediation_intent_and_source_family_requirements(
    report: dict[str, Any],
) -> None:
    by_intent = report["remediation_source_family_counts_by_intent"]

    assert report["remediation_intent_counts"] == {
        **{intent: 90 for intent in freeze.SUPPORTED_PRIMARY_INTENTS},
        freeze.UNSUPPORTED_INTENT: 180,
    }
    assert set(report["remediation_source_family_counts"]) == set(
        freeze.EXPECTED_SOURCE_FAMILIES
    )
    assert report["remediation_source_family_counts"] == {
        family_id: 270 for family_id in freeze.EXPECTED_SOURCE_FAMILIES
    }
    for intent in freeze.SUPPORTED_PRIMARY_INTENTS:
        assert sum(by_intent[intent].values()) == 90
        assert set(by_intent[intent]) == set(freeze.EXPECTED_SOURCE_FAMILIES)
        assert set(by_intent[intent].values()) == {30}
    assert sum(by_intent[freeze.UNSUPPORTED_INTENT].values()) == 180
    assert set(by_intent[freeze.UNSUPPORTED_INTENT].values()) == {60}


def test_review_provenance_reconciles_as_ai_assisted(
    report: dict[str, Any],
) -> None:
    provenance = report["review_provenance"]

    assert provenance["review_method"] == "ai_assisted_review"
    assert provenance["reviewer_type"] == "AI-assisted semantic reviewer"
    assert provenance["review_record_count"] == 810
    assert provenance["approved_count"] == 810
    assert provenance["rejected_count"] == 0
    assert provenance["needs_revision_count"] == 0
    assert provenance["human_review_record_count"] == 0
    assert provenance["ai_assisted_review_record_count"] == 810
    assert provenance["human_adjudication_required_count"] == 0
    assert provenance["human_adjudication_completed_count"] == 0


def test_ai_review_is_not_relabelled_as_human_review(
    inputs: freeze.LoadedInputs,
) -> None:
    modified_payloads = copy.deepcopy(inputs.payloads)
    modified_payloads["remediation_manifest"]["review_provenance"][
        "human_review_record_count"
    ] = 810
    modified = replace(inputs, payloads=modified_payloads)
    records = modified.payloads["remediation_examples"]["examples"]

    with pytest.raises(ValueError, match="AI-assisted review provenance"):
        freeze.validate_review_provenance(modified, records)


def test_unsupported_subtype_coverage_is_complete_and_balanced(
    report: dict[str, Any],
) -> None:
    assert report["unsupported_subtype_counts"] == {
        subtype: 36 for subtype in freeze.UNSUPPORTED_SUBTYPES
    }


def test_hard_negative_coverage_is_complete(report: dict[str, Any]) -> None:
    coverage = report["hard_negative_coverage"]

    assert coverage["all_required_pairs_complete"] is True
    assert coverage["pair_count"] == 10
    assert all(pair["coverage_complete"] for pair in coverage["pairs"])


def test_duplicate_and_conflict_gates_pass(report: dict[str, Any]) -> None:
    duplicate = report["duplicate_validation"]

    assert duplicate["duplicate_gates_passed"] is True
    assert duplicate["exact_development_overlap_count"] == 0
    assert duplicate["normalized_development_overlap_count"] == 0
    assert duplicate["exact_within_new_cluster_count"] == 0
    assert duplicate["normalized_within_new_cluster_count"] == 0
    assert duplicate["cross_intent_normalized_conflict_count"] == 0
    assert duplicate["semantic_or_embedding_similarity_used"] is False


def test_source_aware_and_group_aware_readiness(report: dict[str, Any]) -> None:
    readiness = report["group_and_source_aware_readiness"]

    assert readiness["minimum_three_source_families_every_primary_intent"] is True
    assert readiness[
        "whole_family_holdout_feasible_without_dropping_primary_intent"
    ] is True
    assert readiness["whole_family_holdout_candidate_count"] == 3
    assert readiness["group_aware_cv_ready"] is True
    assert readiness["group_ids_do_not_cross_prohibited_split_boundaries"] is True
    assert readiness["source_family_independence_semantically_proven"] is False
    assert readiness["step29g_ready"] is True


def test_actual_dataset_preserves_valid_historical_group_structure(
    report: dict[str, Any],
) -> None:
    evidence = report["structural_grouping_evidence"]

    assert evidence["inherited_group_membership_preserved"] is True
    assert evidence["inherited_unique_group_count"] > 0
    assert evidence["inherited_cross_intent_group_count"] > 0
    assert evidence["remediation_unique_group_count"] == 810
    assert evidence["remediation_group_ids_unique"] is True
    assert evidence["remediation_group_collision_with_inherited_count"] == 0
    assert evidence["remediation_group_ids_preserved"] is True
    assert evidence["group_aware_cv_ready"] is True


def test_inherited_mixed_intent_groups_are_accepted_as_atomic_units() -> None:
    inherited = [
        group_record("historical-shared-group", "account_balance"),
        group_record("historical-shared-group", "transaction_details"),
    ]
    remediation = [group_record("new-isolated-group", "account_blocked")]
    combined = [*inherited, *remediation]

    evidence = freeze.validate_group_structure(
        inherited,
        remediation,
        combined,
    )

    assert evidence["inherited_cross_intent_group_count"] == 1
    assert evidence["inherited_group_label_purity_required"] is False
    assert evidence["group_ids_are_indivisible_split_units"] is True
    assert evidence["group_aware_split_unit"] == "group_id"


def test_changed_inherited_group_membership_fails_lineage_validation() -> None:
    inherited = [group_record("historical-group", "account_balance")]
    combined = [group_record("rewritten-group", "account_balance")]

    with pytest.raises(ValueError, match="inherited V2-C5 development records changed"):
        freeze.validate_inherited_prefix(combined, inherited)


def test_remediation_group_ids_must_be_nonempty() -> None:
    inherited = [group_record("historical-group", "account_balance")]
    remediation = [group_record("", "account_blocked")]
    combined = [*inherited, *remediation]

    with pytest.raises(ValueError, match="group_id must be a nonempty string"):
        freeze.validate_group_structure(inherited, remediation, combined)


def test_remediation_group_ids_must_be_unique() -> None:
    inherited = [group_record("historical-group", "account_balance")]
    remediation = [
        group_record("duplicate-new-group", "account_blocked"),
        group_record("duplicate-new-group", "cancel_transfer"),
    ]
    combined = [*inherited, *remediation]

    with pytest.raises(ValueError, match="remediation group IDs must be unique"):
        freeze.validate_group_structure(inherited, remediation, combined)


def test_remediation_group_id_collision_with_inherited_group_fails() -> None:
    inherited = [group_record("shared-group", "account_balance")]
    remediation = [group_record("shared-group", "account_blocked")]
    combined = [*inherited, *remediation]

    with pytest.raises(ValueError, match="collides with inherited group ID"):
        freeze.validate_group_structure(inherited, remediation, combined)


def test_combined_representation_must_preserve_remediation_group_ids() -> None:
    inherited = [group_record("historical-group", "account_balance")]
    remediation = [group_record("new-group", "account_blocked")]
    combined = [
        *inherited,
        group_record("changed-new-group", "account_blocked"),
    ]

    with pytest.raises(ValueError, match="group IDs were not preserved"):
        freeze.validate_group_structure(inherited, remediation, combined)


def test_authoring_batch_provenance_remains_auditable(
    report: dict[str, Any],
) -> None:
    batches = report["authoring_batch_statistics_by_source_family"]

    assert set(batches) == set(freeze.EXPECTED_SOURCE_FAMILIES)
    assert batches["v2c6_sf1_definition_direct"][
        "unique_authoring_batch_count"
    ] == 12
    assert batches["v2c6_sf2_scenario_narrative"][
        "unique_authoring_batch_count"
    ] == 1
    assert batches["v2c6_sf3_boundary_conversational"][
        "unique_authoring_batch_count"
    ] == 1
    assert all(
        family["counts_by_authoring_batch_id"] for family in batches.values()
    )


def test_manifest_and_dataset_hashes_reconcile(
    inputs: freeze.LoadedInputs,
    payload: dict[str, Any],
) -> None:
    assert inputs.sha256["dataset"] == (
        "d7f78d7a76799f47bfdc3c1291505d1b964b9d143245d2d353931e8cb4f4a493"
    )
    assert payload["frozen_dataset_sha256"] == inputs.sha256["dataset"]
    assert payload["parent_artifact_lineage"]["combined_manifest"][
        "sha256"
    ] == freeze.EXPECTED_SOURCE_SHA256["combined_manifest"]
    assert payload["parent_artifact_lineage"]["remediation_manifest"][
        "sha256"
    ] == freeze.EXPECTED_SOURCE_SHA256["remediation_manifest"]


def test_freeze_artifact_schema_and_next_required(
    payload: dict[str, Any],
) -> None:
    assert payload["schema_version"] == (
        "v2c6-remediated-development-dataset-freeze.v1"
    )
    assert payload["phase"] == "V2-C6 Step 29F"
    assert payload["freeze_status"] == "FROZEN"
    assert payload["frozen_dataset_record_count"] == 9008
    assert payload["inherited_v2c5_record_count"] == 8198
    assert payload["remediation_record_count"] == 810
    assert payload["next_required"] == (
        "v2c6_source_aware_model_selection_contract"
    )


def test_freeze_artifact_contains_no_raw_utterance_text(
    payload: dict[str, Any],
) -> None:
    keys = recursive_keys(payload)

    assert "text" not in keys
    assert "examples" not in keys
    assert "utterances" not in keys


def test_known_limitations_are_explicit(payload: dict[str, Any]) -> None:
    limitations = " ".join(payload["known_limitations"])

    assert "not independent human annotation" in limitations
    assert "do not prove semantic independence" in limitations
    assert "source-aware development evaluation" in limitations
    assert "fresh untouched final holdout" in limitations
    assert "indivisible split units" in limitations
    assert "not required to be label-pure" in limitations


def test_v2c5_final_holdout_path_is_prohibited(tmp_path: Path) -> None:
    paths = replace(
        freeze.DEFAULT_PATHS,
        prohibited_holdout=tmp_path / "v2c5_final_holdout.json",
    )
    reader_called = False

    def forbidden_reader(path: Path) -> bytes:
        nonlocal reader_called
        reader_called = True
        return b""

    with pytest.raises(PermissionError, match="holdout access is prohibited"):
        freeze.guarded_read_bytes(
            paths.prohibited_holdout,
            paths,
            forbidden_reader,
        )
    assert reader_called is False


def test_source_loading_never_opens_final_holdout() -> None:
    opened: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        assert path.resolve() != freeze.DEFAULT_PATHS.prohibited_holdout.resolve()
        opened.append(path.resolve())
        return path.read_bytes()

    freeze.load_inputs(reader=recording_reader)

    assert freeze.DEFAULT_PATHS.prohibited_holdout.resolve() not in opened


def test_check_mode_writes_nothing(tmp_path: Path) -> None:
    paths = replace(
        freeze.DEFAULT_PATHS,
        freeze_output=tmp_path / freeze.DEFAULT_PATHS.freeze_output.name,
    )

    result = freeze.check(paths)

    assert result["files_written"] is False
    assert result["ready_to_freeze"] is True
    assert result["freeze_artifact_present"] is False
    assert list(tmp_path.iterdir()) == []


def test_deterministic_freeze_serialization(
    inputs: freeze.LoadedInputs,
    report: dict[str, Any],
) -> None:
    first = freeze.stable_json_bytes(freeze.build_freeze_payload(inputs, report))
    second = freeze.stable_json_bytes(freeze.build_freeze_payload(inputs, report))

    assert first == second
    assert first.endswith(b"\n")


def test_freeze_artifact_write_is_create_once(
    tmp_path: Path,
    payload: dict[str, Any],
) -> None:
    output = tmp_path / "freeze.json"
    content = freeze.stable_json_bytes(payload)

    freeze.durable_create(output, content)

    assert output.read_bytes() == content
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze.durable_create(output, content)


def test_no_model_embedding_or_evaluation_work_is_introduced() -> None:
    path = ROOT / freeze.SCRIPT_RELATIVE_PATH
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert ".fit(" not in source
    assert ".predict(" not in source
    assert not {"fastembed", "sentence_transformers", "sklearn", "torch"} & imported


def test_governance_confirms_no_model_work_and_runtime_unchanged(
    payload: dict[str, Any],
) -> None:
    governance = payload["governance"]

    assert governance["model_selection_performed"] is False
    assert governance["model_training_performed"] is False
    assert governance["embeddings_generated"] is False
    assert governance["model_inference_performed"] is False
    assert governance["threshold_tuning_performed"] is False
    assert governance["runtime_behavior_changed"] is False
    assert governance["v2c5_raw_final_holdout_accessed"] is False
    assert governance["approved_status_implies_human_review"] is False
    assert governance["human_review_is_universal_build_gate"] is False
    assert governance["independent_human_annotation_claimed"] is False
    assert governance["semantic_review_completed"] is True
    assert governance["development_dataset_frozen"] is True
