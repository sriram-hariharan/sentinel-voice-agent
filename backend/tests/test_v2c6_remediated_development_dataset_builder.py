from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import build_v2c6_remediated_development_dataset as builder

ROOT = Path(__file__).resolve().parents[2]

RISK_BY_INTENT = {
    "account_balance": "PRIVATE_READ",
    "account_blocked": "PRIVATE_READ",
    "cancel_transfer": "PROTECTED_WRITE",
    "card_status": "PRIVATE_READ",
    "close_account": "PROTECTED_WRITE",
    "create_dispute": "PROTECTED_WRITE",
    "escalation": "ESCALATION_OR_UNCERTAIN",
    "freeze_card": "PROTECTED_WRITE",
    "informational_policy": "PUBLIC",
    "lost_or_stolen_phone": "ESCALATION_OR_UNCERTAIN",
    "passcode_recovery": "ESCALATION_OR_UNCERTAIN",
    "recent_transactions": "PRIVATE_READ",
    "transaction_details": "PRIVATE_READ",
    "transfer_failed_or_declined": "PRIVATE_READ",
    "transfer_pending": "PRIVATE_READ",
    "unsupported_or_uncertain": "ESCALATION_OR_UNCERTAIN",
}


def make_record(
    record_id: str,
    intent: str,
    family_index: int,
    *,
    boundary_target: str = "none",
    hard_negative: bool = False,
    review_status: str = "approved",
    unsupported_subtype: str | None = None,
) -> dict[str, Any]:
    family_id = f"family:{intent}:{family_index}"
    record: dict[str, Any] = {
        "authoring_batch_id": f"batch:{family_id}",
        "authoring_method": "human_authored",
        "boundary_target": boundary_target,
        "group_id": f"group:{record_id}",
        "intent": intent,
        "is_hard_negative": hard_negative,
        "record_id": record_id,
        "review_status": review_status,
        "risk_level": RISK_BY_INTENT[intent],
        "source_family_id": family_id,
        "source_family_independence_basis": f"independent basis for {family_id}",
        "source_revision": f"revision:{family_id}",
        "text": f"Independent synthetic fixture text for {record_id}.",
    }
    if intent == builder.UNSUPPORTED_INTENT:
        record["unsupported_subtype"] = (
            unsupported_subtype or builder.UNSUPPORTED_SUBTYPES[0]
        )
    return record


def complete_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {intent: 0 for intent in builder.PRIMARY_INTENTS}
    subtype_index = 0
    for pair_index, (side_a, side_b) in enumerate(builder.HARD_NEGATIVE_PAIRS):
        for side, target in ((side_a, side_b), (side_b, side_a)):
            family_index = occurrences[side] % 3
            occurrences[side] += 1
            subtype = None
            if side == builder.UNSUPPORTED_INTENT:
                subtype = builder.UNSUPPORTED_SUBTYPES[
                    subtype_index % len(builder.UNSUPPORTED_SUBTYPES)
                ]
                subtype_index += 1
            records.append(
                make_record(
                    f"v2c6-fixture:hard:{pair_index}:{side}",
                    side,
                    family_index,
                    boundary_target=target,
                    hard_negative=True,
                    unsupported_subtype=subtype,
                )
            )
    for intent in builder.PRIMARY_INTENTS:
        used = {
            int(str(record["source_family_id"]).rsplit(":", 1)[1])
            for record in records
            if record["intent"] == intent
        }
        for family_index in sorted(set(range(3)) - used):
            records.append(
                make_record(
                    f"v2c6-fixture:coverage:{intent}:{family_index}",
                    intent,
                    family_index,
                    unsupported_subtype=(
                        builder.UNSUPPORTED_SUBTYPES[family_index]
                        if intent == builder.UNSUPPORTED_INTENT
                        else None
                    ),
                )
            )
    return records


def authoring_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": "V2-C6 Step 29E2",
        "records": records,
        "schema_version": builder.AUTHORING_INPUT_SCHEMA,
    }


@pytest.fixture
def sources() -> builder.FrozenSources:
    old_text = "Synthetic frozen development fixture."
    old_record = {
        "data_role": "development",
        "example_id": "historical-fixture:0001",
        "group_id": "historical-group:0001",
        "intent": "account_balance",
        "normalized_text_sha256": builder.normalized_text_sha256(old_text),
        "risk": "PRIVATE_READ",
        "text": old_text,
        "text_sha256": builder.text_sha256(old_text),
    }
    artifacts = {
        "dataset_contract": {
            "path": "data/evals/v2/ml/v2c6_remediation_dataset_contract.json",
            "sha256": builder.DATASET_CONTRACT_SHA256,
        },
        "development_dataset": {
            "path": "data/evals/v2/ml/v2c5_expanded_development_dataset.json",
            "sha256": builder.DEVELOPMENT_DATASET_SHA256,
        },
        "taxonomy": {
            "path": "data/evals/v2/ml/v2c5_taxonomy_freeze.json",
            "sha256": builder.TAXONOMY_SHA256,
        },
    }
    return builder.FrozenSources(
        dataset_contract={},
        design_contract={},
        diagnosis={},
        diagnosis_manifest={},
        development_dataset={"examples": [old_record]},
        development_manifest={},
        taxonomy={},
        taxonomy_manifest={},
        development_examples=(old_record,),
        intent_labels=tuple(RISK_BY_INTENT),
        risk_by_intent=RISK_BY_INTENT,
        source_artifacts=artifacts,
    )


def temporary_paths(tmp_path: Path) -> builder.BuildPaths:
    return replace(
        builder.DEFAULT_PATHS,
        repository_root=tmp_path,
        authoring_input=tmp_path / "local/remediation.json",
        remediation_output=tmp_path / "remediation.json",
        remediation_manifest_output=tmp_path / "remediation.manifest.json",
        combined_output=tmp_path / "combined.json",
        combined_manifest_output=tmp_path / "combined.manifest.json",
        prohibited_holdout=tmp_path / "v2c5_final_holdout.json",
        builder=ROOT / builder.SCRIPT_RELATIVE_PATH,
    )


def validate(
    records: list[dict[str, Any]],
    sources: builder.FrozenSources,
    *,
    strict: bool = False,
) -> builder.AuthoringValidation:
    payload = authoring_payload(records)
    return builder.validate_authoring_payload(
        payload,
        builder.stable_json_bytes(payload),
        sources,
        enforce_build_requirements=strict,
    )


def write_authoring_input(
    paths: builder.BuildPaths,
    records: list[dict[str, Any]],
) -> None:
    paths.authoring_input.parent.mkdir(parents=True, exist_ok=True)
    paths.authoring_input.write_bytes(
        builder.stable_json_bytes(authoring_payload(records))
    )


def test_expected_contract_design_and_diagnosis_lineage() -> None:
    assert builder.DATASET_CONTRACT_SHA256 == (
        "2f251cb814f06058dd24a8b86b207fa85b14df98a5c17f2853bed50ffea40155"
    )
    assert builder.DESIGN_CONTRACT_SHA256 == (
        "ffa8ec7f20b99cc22da3473dd50d32c7aa611cf548ecb205d5c53b9e5409a442"
    )
    assert builder.DIAGNOSIS_SHA256 == (
        "58177bd3aacf52bbff2268fd834e883457e8241f36e35f5e84943f196f688536"
    )
    assert builder.DIAGNOSIS_MANIFEST_SHA256 == (
        "bb04f4bf82704331ab4cd0c1dc5ed3a9d87761a30013315e8c52055b3c550305"
    )


def test_prohibited_v2c5_holdout_path_guard(tmp_path: Path) -> None:
    paths = temporary_paths(tmp_path)
    reader_called = False

    def forbidden_reader(path: Path) -> bytes:
        nonlocal reader_called
        reader_called = True
        return b""

    with pytest.raises(PermissionError, match="holdout access is prohibited"):
        builder.guarded_read_bytes(paths.prohibited_holdout, paths, forbidden_reader)
    assert reader_called is False


def test_preflight_never_opens_holdout(
    tmp_path: Path, sources: builder.FrozenSources
) -> None:
    paths = temporary_paths(tmp_path)
    write_authoring_input(paths, complete_records())
    opened: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        assert path.resolve() != paths.prohibited_holdout.resolve()
        opened.append(path.resolve())
        return path.read_bytes()

    result = builder.preflight(paths, recording_reader, sources)

    assert result["files_written"] is False
    assert paths.prohibited_holdout.resolve() not in opened


def test_build_never_opens_holdout(
    tmp_path: Path, sources: builder.FrozenSources
) -> None:
    paths = temporary_paths(tmp_path)
    write_authoring_input(paths, complete_records())
    opened: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        assert path.resolve() != paths.prohibited_holdout.resolve()
        opened.append(path.resolve())
        return path.read_bytes()

    result = builder.build(paths, recording_reader, sources)

    assert result["files_written"] is True
    assert paths.prohibited_holdout.resolve() not in opened


def test_check_results_never_opens_holdout_and_writes_nothing(
    tmp_path: Path, sources: builder.FrozenSources
) -> None:
    paths = temporary_paths(tmp_path)
    write_authoring_input(paths, complete_records())
    builder.build(paths, sources=sources)
    before = {path: path.read_bytes() for path in builder.output_paths(paths)}
    opened: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        assert path.resolve() != paths.prohibited_holdout.resolve()
        opened.append(path.resolve())
        return path.read_bytes()

    result = builder.check_results(paths, recording_reader, sources)

    assert result["files_written"] is False
    assert before == {path: path.read_bytes() for path in builder.output_paths(paths)}
    assert paths.prohibited_holdout.resolve() not in opened


def test_exact_primary_intent_set() -> None:
    assert builder.PRIMARY_INTENTS == (
        "account_blocked",
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
        "transfer_failed_or_declined",
        "transfer_pending",
        "unsupported_or_uncertain",
    )


def test_exact_protected_write_set() -> None:
    assert builder.PROTECTED_WRITE_INTENTS == (
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    )


def test_risk_mapping_validation(sources: builder.FrozenSources) -> None:
    records = complete_records()
    records[0]["risk_level"] = "PUBLIC"

    with pytest.raises(ValueError, match="risk_level disagrees"):
        validate(records, sources)


def test_required_metadata_validation(sources: builder.FrozenSources) -> None:
    records = complete_records()
    del records[0]["source_family_independence_basis"]

    with pytest.raises(ValueError, match="missing fields"):
        validate(records, sources)


def test_authoring_method_allowlist(sources: builder.FrozenSources) -> None:
    records = complete_records()
    records[0]["authoring_method"] = "unreviewed_generation_process"

    with pytest.raises(ValueError, match="invalid authoring method"):
        validate(records, sources)


def test_record_id_uniqueness(sources: builder.FrozenSources) -> None:
    records = complete_records()
    records[1]["record_id"] = records[0]["record_id"]

    with pytest.raises(ValueError, match="duplicate remediation record ID"):
        validate(records, sources)


def test_record_id_cannot_collide_with_frozen_development(
    sources: builder.FrozenSources,
) -> None:
    records = complete_records()
    records[0]["record_id"] = sources.development_examples[0]["example_id"]

    with pytest.raises(ValueError, match="collides with development"):
        validate(records, sources)


def test_group_id_is_required(sources: builder.FrozenSources) -> None:
    records = complete_records()
    records[0]["group_id"] = ""

    with pytest.raises(ValueError, match="group_id must be"):
        validate(records, sources)


def test_group_cannot_cross_intent_boundary(sources: builder.FrozenSources) -> None:
    records = complete_records()
    different_intent = next(
        record for record in records[1:] if record["intent"] != records[0]["intent"]
    )
    different_intent["group_id"] = records[0]["group_id"]

    with pytest.raises(ValueError, match="group crosses intent"):
        validate(records, sources)


def test_source_family_minimum_is_a_build_gate(
    sources: builder.FrozenSources,
) -> None:
    records = [
        record
        for record in complete_records()
        if not (
            record["intent"] == "close_account"
            and record["source_family_id"].endswith(":2")
        )
    ]
    report = validate(records, sources).report

    assert report["source_family_statistics"]["close_account"][
        "source_family_minimum_met"
    ] is False
    with pytest.raises(ValueError, match="minimum_source_families"):
        validate(records, sources, strict=True)


def test_per_family_minimum_is_reported_as_planning_target(
    sources: builder.FrozenSources,
) -> None:
    report = validate(complete_records(), sources).report

    assert report["planning_volume"][
        "planning_target_is_mandatory_build_gate"
    ] is False
    assert report["source_family_statistics"]["account_blocked"][
        "per_family_30_record_target_met"
    ] is False


def test_source_family_statistics(sources: builder.FrozenSources) -> None:
    report = validate(complete_records(), sources).report[
        "source_family_statistics"
    ]

    assert set(report) == set(builder.PRIMARY_INTENTS)
    assert all(value["unique_source_family_count"] >= 3 for value in report.values())
    assert all(value["total_new_records"] > 0 for value in report.values())
    assert all(value["unique_group_count"] > 0 for value in report.values())
    assert all(value["unique_normalized_text_count"] > 0 for value in report.values())
    assert all(value["diversity_by_source_family"] for value in report.values())
    assert all(
        value["dominant_wording_review"]
        == "human_review_required_no_automatic_threshold"
        for value in report.values()
    )


def test_identical_provenance_cannot_alias_multiple_family_ids(
    sources: builder.FrozenSources,
) -> None:
    records = complete_records()
    first = records[0]
    other = next(
        record
        for record in records
        if record["source_family_id"] != first["source_family_id"]
    )
    other_family_id = other["source_family_id"]
    for record in records:
        if record["source_family_id"] == other_family_id:
            for field in (
                "source_family_independence_basis",
                "source_revision",
                "authoring_batch_id",
                "authoring_method",
            ):
                record[field] = first[field]

    with pytest.raises(ValueError, match="identical provenance"):
        validate(records, sources)


def test_random_seed_does_not_create_an_independent_family(
    sources: builder.FrozenSources,
) -> None:
    records = complete_records()
    records[0]["source_family_independence_basis"] = (
        "Different random seed using the same prompt"
    )

    with pytest.raises(ValueError, match="basis is not independent"):
        validate(records, sources)


def test_review_state_allowlist(sources: builder.FrozenSources) -> None:
    records = complete_records()
    records[0]["review_status"] = "auto_approved"

    with pytest.raises(ValueError, match="invalid review status"):
        validate(records, sources)


def test_only_approved_records_are_included(sources: builder.FrozenSources) -> None:
    records = complete_records()
    rejected = make_record(
        "v2c6-fixture:rejected",
        "account_blocked",
        0,
        review_status="rejected",
    )
    rejected["source_family_id"] = records[0]["source_family_id"]
    rejected["source_family_independence_basis"] = records[0][
        "source_family_independence_basis"
    ]
    rejected["source_revision"] = records[0]["source_revision"]
    rejected["authoring_batch_id"] = records[0]["authoring_batch_id"]
    records.append(rejected)

    result = validate(records, sources)

    assert result.report["excluded_record_count"] == 1
    assert all(
        record["review_status"] == "approved"
        for record in result.approved_records
    )
    assert rejected["record_id"] not in {
        record["record_id"] for record in result.approved_records
    }


def test_unsupported_subtype_allowlist(sources: builder.FrozenSources) -> None:
    records = complete_records()
    unsupported = next(
        record for record in records if record["intent"] == builder.UNSUPPORTED_INTENT
    )
    unsupported["unsupported_subtype"] = "new_runtime_label"

    with pytest.raises(ValueError, match="invalid unsupported subtype"):
        validate(records, sources)


def test_all_unsupported_subtype_coverage_is_required(
    sources: builder.FrozenSources,
) -> None:
    records = complete_records()
    for record in records:
        if record.get("unsupported_subtype") == builder.UNSUPPORTED_SUBTYPES[-1]:
            record["unsupported_subtype"] = builder.UNSUPPORTED_SUBTYPES[0]
    report = validate(records, sources).report

    assert report["unsupported_subtype_coverage_complete"] is False
    with pytest.raises(ValueError, match="unsupported_subtype_coverage"):
        validate(records, sources, strict=True)


def test_exact_hard_negative_pair_set() -> None:
    assert builder.HARD_NEGATIVE_PAIRS == (
        ("close_account", "unsupported_or_uncertain"),
        ("transfer_failed_or_declined", "unsupported_or_uncertain"),
        ("account_blocked", "unsupported_or_uncertain"),
        ("cancel_transfer", "unsupported_or_uncertain"),
        ("create_dispute", "unsupported_or_uncertain"),
        ("freeze_card", "unsupported_or_uncertain"),
        ("transfer_pending", "unsupported_or_uncertain"),
        ("cancel_transfer", "transfer_pending"),
        ("transfer_failed_or_declined", "transfer_pending"),
        ("account_blocked", "transfer_failed_or_declined"),
    )


def test_hard_negative_coverage_reporting(sources: builder.FrozenSources) -> None:
    coverage = validate(complete_records(), sources).report[
        "hard_negative_coverage"
    ]

    assert coverage["pair_count"] == 10
    assert coverage["all_required_pairs_complete"] is True
    assert all(pair["side_a_count"] > 0 for pair in coverage["pairs"])
    assert all(pair["side_b_count"] > 0 for pair in coverage["pairs"])
    assert all(pair["approved_count"] > 1 for pair in coverage["pairs"])


def test_invented_hard_negative_pair_is_rejected(
    sources: builder.FrozenSources,
) -> None:
    records = complete_records()
    records[0]["boundary_target"] = "account_balance"

    with pytest.raises(ValueError, match="pair is not frozen"):
        validate(records, sources)


def test_nfkc_lowercase_whitespace_normalization() -> None:
    assert builder.normalize_text("  ＣＡＲＤ\t\nStatus  ") == "card status"


def test_exact_duplicate_detection(sources: builder.FrozenSources) -> None:
    records = complete_records()
    duplicate = copy.deepcopy(records[0])
    duplicate["record_id"] = "v2c6-fixture:exact-duplicate"
    duplicate["group_id"] = "group:v2c6-fixture:exact-duplicate"
    records.append(duplicate)

    report = validate(records, sources).report["duplicate_validation"]

    assert report["exact_within_new_cluster_count"] == 1
    assert report["duplicate_gates_passed"] is False


def test_normalized_duplicate_detection(sources: builder.FrozenSources) -> None:
    records = complete_records()
    duplicate = copy.deepcopy(records[0])
    duplicate["record_id"] = "v2c6-fixture:normalized-duplicate"
    duplicate["group_id"] = "group:v2c6-fixture:normalized-duplicate"
    duplicate["text"] = f"  {str(records[0]['text']).upper()}  "
    records.append(duplicate)

    report = validate(records, sources).report["duplicate_validation"]

    assert report["exact_within_new_cluster_count"] == 0
    assert report["normalized_within_new_cluster_count"] == 1


def test_existing_development_duplicate_detection(
    sources: builder.FrozenSources,
) -> None:
    records = complete_records()
    records[0]["text"] = str(sources.development_examples[0]["text"])

    report = validate(records, sources).report["duplicate_validation"]

    assert report["exact_development_overlap_count"] == 1
    assert report["normalized_development_overlap_count"] == 1
    assert report["exact_development_overlaps"][0].get("text") is None


def test_cross_intent_normalized_conflict_detection(
    sources: builder.FrozenSources,
) -> None:
    records = complete_records()
    duplicate = copy.deepcopy(records[0])
    duplicate["record_id"] = "v2c6-fixture:cross-intent"
    duplicate["group_id"] = "group:v2c6-fixture:cross-intent"
    duplicate["intent"] = "freeze_card"
    duplicate["risk_level"] = RISK_BY_INTENT["freeze_card"]
    duplicate["is_hard_negative"] = False
    duplicate["boundary_target"] = "none"
    duplicate["text"] = f" {str(records[0]['text']).upper()} "
    records.append(duplicate)

    report = validate(records, sources).report["duplicate_validation"]

    assert report["cross_intent_normalized_conflict_count"] == 1
    assert report["raw_text_persisted_in_diagnostics"] is False


def test_no_embedding_similarity_is_used(sources: builder.FrozenSources) -> None:
    report = validate(complete_records(), sources).report["duplicate_validation"]
    source = (ROOT / builder.SCRIPT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert report["semantic_or_embedding_similarity_used"] is False
    assert "sentence_transform" not in source.lower()
    assert "fastembed" not in source.lower()


def test_frozen_v2c5_records_remain_unchanged(
    tmp_path: Path, sources: builder.FrozenSources
) -> None:
    paths = temporary_paths(tmp_path)
    validation = validate(complete_records(), sources, strict=True)
    before = copy.deepcopy(sources.development_examples)

    artifacts = builder.build_artifacts(sources, validation, paths)

    assert sources.development_examples == before
    assert artifacts.combined_payload["examples"][: len(before)] == list(before)


def test_combined_count_reconciliation(
    tmp_path: Path, sources: builder.FrozenSources
) -> None:
    paths = temporary_paths(tmp_path)
    validation = validate(complete_records(), sources, strict=True)
    artifacts = builder.build_artifacts(sources, validation, paths)

    assert artifacts.combined_payload["example_count"] == (
        len(sources.development_examples) + len(validation.approved_records)
    )
    counts = artifacts.combined_manifest_payload["counts"]
    assert counts["old_record_count"] + counts["new_record_count"] == counts[
        "total_combined_count"
    ]


def test_deterministic_record_order(
    tmp_path: Path, sources: builder.FrozenSources
) -> None:
    records = list(reversed(complete_records()))
    paths = temporary_paths(tmp_path)
    validation = validate(records, sources, strict=True)
    artifacts = builder.build_artifacts(sources, validation, paths)
    old_count = len(sources.development_examples)

    assert artifacts.combined_payload["examples"][:old_count] == list(
        sources.development_examples
    )
    assert [
        record["example_id"]
        for record in artifacts.combined_payload["examples"][old_count:]
    ] == sorted(record["record_id"] for record in records)


def test_deterministic_serialization() -> None:
    assert builder.stable_json_bytes({"z": 1, "a": 2}) == (
        builder.stable_json_bytes({"a": 2, "z": 1})
    )


def test_manifest_hashes(tmp_path: Path, sources: builder.FrozenSources) -> None:
    paths = temporary_paths(tmp_path)
    payload = authoring_payload(complete_records())
    validation = builder.validate_authoring_payload(
        payload,
        builder.stable_json_bytes(payload),
        sources,
        enforce_build_requirements=True,
    )
    artifacts = builder.build_artifacts(sources, validation, paths)

    assert artifacts.remediation_manifest_payload["remediation_examples"][
        "sha256"
    ] == builder.sha256_bytes(artifacts.remediation_bytes)
    assert artifacts.combined_manifest_payload[
        "remediated_development_dataset"
    ]["sha256"] == builder.sha256_bytes(artifacts.combined_bytes)
    assert artifacts.combined_manifest_payload["remediation_examples"][
        "sha256"
    ] == builder.sha256_bytes(artifacts.remediation_bytes)


def test_no_model_training_api_exists() -> None:
    source = (ROOT / builder.SCRIPT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert ".fit(" not in source
    assert builder.GOVERNANCE_FLAGS["model_training_performed"] is False


def test_no_inference_api_exists() -> None:
    source = (ROOT / builder.SCRIPT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert ".predict(" not in source
    assert builder.GOVERNANCE_FLAGS["model_inference_performed"] is False


def test_no_embedding_api_imports() -> None:
    tree = ast.parse(
        (ROOT / builder.SCRIPT_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not {"fastembed", "sentence_transformers", "torch"} & imported
    assert builder.GOVERNANCE_FLAGS["embeddings_generated"] is False


def test_no_threshold_tuning() -> None:
    assert builder.GOVERNANCE_FLAGS["threshold_tuning_performed"] is False


def test_runtime_is_unchanged() -> None:
    assert builder.GOVERNANCE_FLAGS["runtime_behavior_changed"] is False


def test_source_aware_readiness_report(sources: builder.FrozenSources) -> None:
    report = validate(complete_records(), sources).report[
        "source_aware_readiness"
    ]

    assert report["minimum_three_source_families_every_primary_intent"] is True
    assert report[
        "whole_family_holdout_feasible_without_dropping_primary_intent"
    ] is True
    assert report["group_ids_do_not_cross_prohibited_split_boundaries"] is True
    assert report["group_aware_cv_ready"] is True
    assert report["only_approved_records_included"] is True
    assert report["duplicate_gates_passed"] is True
    assert report["review_gates_passed"] is True
    assert report["source_family_independence_semantically_proven"] is False


def test_future_output_paths_are_exact() -> None:
    assert [path.name for path in builder.output_paths(builder.DEFAULT_PATHS)] == [
        "v2c6_remediation_examples.json",
        "v2c6_remediation_examples.manifest.json",
        "v2c6_remediated_development_dataset.json",
        "v2c6_remediated_development_dataset.manifest.json",
    ]
    assert builder.DEFAULT_PATHS.authoring_input.as_posix().endswith(
        "data/evals/v2/ml/local/v2c6_remediation_authoring_workfile.json"
    )


def test_import_and_preflight_without_input_create_no_artifacts(
    tmp_path: Path, sources: builder.FrozenSources
) -> None:
    paths = temporary_paths(tmp_path)

    result = builder.preflight(paths, sources=sources)

    assert result["authoring_input_present"] is False
    assert result["files_written"] is False
    assert all(not path.exists() for path in builder.output_paths(paths))


def test_cli_modes_are_exact() -> None:
    parser = builder.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }

    assert options == {"--preflight", "--build", "--check-results"}


def test_planning_volume_shortfall_does_not_relax_mandatory_gates(
    sources: builder.FrozenSources,
) -> None:
    validation = validate(complete_records(), sources, strict=True)

    assert validation.report["mandatory_build_gates_passed"] is True
    assert validation.report["planning_volume"]["all_planning_targets_met"] is False


def test_source_independence_limitation_is_explicit(
    sources: builder.FrozenSources,
) -> None:
    report = validate(complete_records(), sources).report

    assert "cannot prove true semantic independence" in report[
        "source_independence_limitation"
    ]
