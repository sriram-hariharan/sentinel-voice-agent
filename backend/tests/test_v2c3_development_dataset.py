import ast
import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import build_v2c3_development_dataset as builder


@pytest.fixture(scope="module")
def artifacts() -> builder.BuiltArtifacts:
    return builder.build_artifacts()


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value)) if value else set()
    return set()


def _synthetic_record(
    *,
    example_id: str,
    text: str,
    intent: str = "account_balance",
    source_id: str = builder.BANKING77_SOURCE_ID,
    group_id: str | None = None,
) -> dict[str, Any]:
    return {
        "data_role": "external_candidate",
        "example_id": example_id,
        "group_id": group_id or f"source:{example_id}",
        "intent": intent,
        "mapping_status": "EXACT_MATCH",
        "normalized_text_sha256": builder.normalized_text_sha256(text),
        "original_example_id": None,
        "original_split": "train",
        "risk": "PRIVATE_READ",
        "source_domain": "banking",
        "source_id": source_id,
        "source_label": "balance",
        "source_revision": "test-revision",
        "source_row_index": 1,
        "source_split": "train",
        "text": text,
        "text_sha256": builder.text_sha256(text),
    }


def test_deterministic_output_matches_tracked_artifacts(
    artifacts: builder.BuiltArtifacts,
) -> None:
    repeated = builder.build_artifacts()

    assert artifacts.development_bytes == repeated.development_bytes
    assert artifacts.lockbox_bytes == repeated.lockbox_bytes
    assert artifacts.manifest_bytes == repeated.manifest_bytes
    assert artifacts.development_bytes == builder.DEFAULT_PATHS.development_output.read_bytes()
    assert artifacts.lockbox_bytes == builder.DEFAULT_PATHS.lockbox_output.read_bytes()
    assert artifacts.manifest_bytes == builder.DEFAULT_PATHS.manifest_output.read_bytes()


def test_internal_train_and_validation_only(
    artifacts: builder.BuiltArtifacts,
) -> None:
    internal = [
        row
        for row in artifacts.development_payload["examples"]
        if row["source_id"] == builder.INTERNAL_SOURCE_ID
    ]

    assert len(internal) == 513
    assert {row["source_split"] for row in internal} == {"train", "validation"}
    assert all(row["example_id"] == row["original_example_id"] for row in internal)
    assert all(row["group_id"] for row in internal)
    assert not any(row["source_split"] == "locked_test" for row in internal)


def test_internal_locked_test_is_excluded(
    artifacts: builder.BuiltArtifacts,
) -> None:
    manifest = artifacts.manifest_payload

    assert manifest["counts"]["internal_development"] == 513
    assert manifest["validation"]["internal_locked_test_used"] is False
    assert "sentinelvoice_v2c1_internal:locked_test" not in manifest[
        "counts_by_source_split"
    ]


def test_banking77_uses_train_only(artifacts: builder.BuiltArtifacts) -> None:
    records = [
        row
        for row in [
            *artifacts.development_payload["examples"],
            *artifacts.lockbox_payload["members"],
        ]
        if row["source_id"] == builder.BANKING77_SOURCE_ID
    ]

    assert records
    assert {row["source_split"] for row in records} == {"train"}
    assert all(1 <= row["source_row_index"] <= 10_003 for row in records)
    assert "banking77_train" in artifacts.manifest_payload["input_paths"]
    assert "banking77_test" not in artifacts.manifest_payload["input_paths"]
    assert artifacts.manifest_payload["validation"]["banking77_test_used"] is False


def test_clinc_uses_finance_train_and_val_only(
    artifacts: builder.BuiltArtifacts,
) -> None:
    records = [
        row
        for row in [
            *artifacts.development_payload["examples"],
            *artifacts.lockbox_payload["members"],
        ]
        if row["source_id"] == builder.CLINC_SOURCE_ID
    ]
    mapping = builder.load_json_object(builder.DEFAULT_PATHS.clinc_mapping)
    finance_labels = {row["source_intent"] for row in mapping["mappings"]}

    assert records
    assert {row["source_split"] for row in records} == {"train", "val"}
    assert {row["source_label"] for row in records} <= finance_labels
    assert artifacts.manifest_payload["validation"]["clinc_test_used"] is False
    assert artifacts.manifest_payload["validation"]["clinc_oos_used"] is False


def test_cfpb_and_other_prohibited_sources_are_excluded(
    artifacts: builder.BuiltArtifacts,
) -> None:
    sources = {
        row["source_id"]
        for row in [
            *artifacts.development_payload["examples"],
            *artifacts.lockbox_payload["members"],
        ]
    }

    assert sources == builder.ALLOWED_SOURCE_IDS
    assert not sources & builder.PROHIBITED_SOURCE_IDS
    assert artifacts.manifest_payload["counts"]["prohibited_source_count"] == 0
    assert artifacts.manifest_payload["validation"]["cfpb_used"] is False


def test_exact_and_unsupported_are_included(
    artifacts: builder.BuiltArtifacts,
) -> None:
    external = [
        row
        for row in [
            *artifacts.development_payload["examples"],
            *artifacts.lockbox_payload["members"],
        ]
        if row["source_id"] != builder.INTERNAL_SOURCE_ID
    ]

    assert {row["mapping_status"] for row in external} == {
        "EXACT_MATCH",
        "UNSUPPORTED",
    }
    assert any(row["mapping_status"] == "EXACT_MATCH" for row in external)
    assert any(row["mapping_status"] == "UNSUPPORTED" for row in external)


def test_near_and_ambiguous_are_excluded_pending_review(
    artifacts: builder.BuiltArtifacts,
) -> None:
    manifest = artifacts.manifest_payload
    all_rows = [
        *artifacts.development_payload["examples"],
        *artifacts.lockbox_payload["members"],
    ]

    assert not any(
        row["mapping_status"] in {"NEAR_MATCH", "AMBIGUOUS"} for row in all_rows
    )
    assert manifest["counts"]["excluded_near_match_count"] > 0
    assert manifest["counts"]["excluded_ambiguous_count"] > 0
    assert manifest["counts"]["excluded_pending_semantic_review"] == (
        manifest["counts"]["excluded_near_match_count"]
        + manifest["counts"]["excluded_ambiguous_count"]
    )


def test_unsupported_always_maps_to_unsupported_or_uncertain(
    artifacts: builder.BuiltArtifacts,
) -> None:
    unsupported = [
        row
        for row in [
            *artifacts.development_payload["examples"],
            *artifacts.lockbox_payload["members"],
        ]
        if row["mapping_status"] == "UNSUPPORTED"
    ]

    assert unsupported
    assert {row["intent"] for row in unsupported} == {"unsupported_or_uncertain"}


def test_protected_write_automatic_mapping_guard() -> None:
    unsafe = {
        "mapping_status": "EXACT_MATCH",
        "sentinelvoice_intent": "freeze_card",
        "source_intent": "unsafe_mapping",
    }

    with pytest.raises(ValueError, match="protected-write mapping"):
        builder._mapped_target(unsafe)


def test_normalization_is_frozen_and_deterministic() -> None:
    first = "  ＨELLO\tWorld\n"
    second = "hello world"

    assert builder.normalize_text(first) == "hello world"
    assert builder.normalize_text(first) == builder.normalize_text(first)
    assert builder.normalized_text_sha256(first) == (
        builder.normalized_text_sha256(second)
    )


def test_exact_normalized_duplicates_share_a_group() -> None:
    records = [
        _synthetic_record(example_id="one", text="Same request"),
        _synthetic_record(
            example_id="two",
            text="  same\trequest ",
            source_id=builder.CLINC_SOURCE_ID,
        ),
    ]

    grouped, summary, forced = builder.apply_duplicate_groups(records)

    assert len({row["group_id"] for row in grouped}) == 1
    assert summary.duplicate_groups_found == 1
    assert summary.duplicate_records == 2
    assert forced == set()


def test_conflicting_duplicate_targets_fail() -> None:
    records = [
        _synthetic_record(example_id="one", text="Same request"),
        _synthetic_record(
            example_id="two",
            text="same request",
            intent="card_status",
            source_id=builder.CLINC_SOURCE_ID,
        ),
    ]

    with pytest.raises(ValueError, match="conflicting target intents"):
        builder.apply_duplicate_groups(records)


def test_no_development_lockbox_group_or_hash_overlap(
    artifacts: builder.BuiltArtifacts,
) -> None:
    development = artifacts.development_payload["examples"]
    lockbox = artifacts.lockbox_payload["members"]

    assert not ({row["group_id"] for row in development} & {row["group_id"] for row in lockbox})
    assert not (
        {row["normalized_text_sha256"] for row in development}
        & {row["normalized_text_sha256"] for row in lockbox}
    )


def test_internal_duplicates_cannot_enter_lockbox() -> None:
    internal = _synthetic_record(
        example_id="internal",
        text="shared request",
        source_id=builder.INTERNAL_SOURCE_ID,
        group_id="internal-group",
    )
    internal["data_role"] = "development"
    internal["mapping_status"] = "INTERNAL_FROZEN"
    external = _synthetic_record(example_id="external", text="shared request")
    grouped, _, forced = builder.apply_duplicate_groups([internal, external])

    development, lockbox, _ = builder.split_external_groups(
        grouped,
        seed=20260928,
        lockbox_fraction=1.0,
        forced_development_groups=forced,
    )

    assert {row["example_id"] for row in development} == {"internal", "external"}
    assert lockbox == []


def test_lockbox_is_text_free(artifacts: builder.BuiltArtifacts) -> None:
    lockbox = artifacts.lockbox_payload

    assert lockbox["contains_text"] is False
    assert "text" not in _all_keys(lockbox)
    assert all("text_sha256" in row for row in lockbox["members"])
    assert all("normalized_text_sha256" in row for row in lockbox["members"])


def test_lockbox_membership_is_deterministic_from_seed(
    artifacts: builder.BuiltArtifacts,
) -> None:
    repeated = builder.build_artifacts()

    assert artifacts.lockbox_payload["seed"] == 20260928
    assert artifacts.lockbox_payload["members"] == repeated.lockbox_payload["members"]
    assert artifacts.manifest_payload["lockbox_policy"][
        "model_output_or_confidence_used"
    ] is False


def test_source_and_intent_stratification_is_approximately_twenty_percent(
    artifacts: builder.BuiltArtifacts,
) -> None:
    plan = artifacts.manifest_payload["lockbox_policy"][
        "source_and_target_strata"
    ]

    assert plan
    for stratum in plan.values():
        target = stratum["target_lockbox_group_count"]
        actual = stratum["actual_lockbox_group_count"]
        eligible = stratum["lockbox_eligible_group_count"]
        assert actual >= target
        assert actual <= target + max(1, int(eligible * 0.02 + 0.5))


def test_risk_is_derived_from_frozen_contract(
    artifacts: builder.BuiltArtifacts,
) -> None:
    contract = builder.load_json_object(builder.DEFAULT_PATHS.contract)
    risk_by_intent = contract["risk_by_intent"]
    records = [
        *artifacts.development_payload["examples"],
        *artifacts.lockbox_payload["members"],
    ]

    assert all(row["risk"] == risk_by_intent[row["intent"]] for row in records)


def test_check_detects_output_drift(
    tmp_path: Path, artifacts: builder.BuiltArtifacts
) -> None:
    paths = replace(
        builder.DEFAULT_PATHS,
        development_output=tmp_path / "development.json",
        lockbox_output=tmp_path / "lockbox.json",
        manifest_output=tmp_path / "manifest.json",
    )
    builder.write_artifacts(artifacts, paths)
    builder.check_artifacts(artifacts, paths)
    paths.lockbox_output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from deterministic build"):
        builder.check_artifacts(artifacts, paths)


def test_prohibited_source_configuration_fails() -> None:
    configured = set(builder.ALLOWED_SOURCE_IDS)
    configured.add("cfpb_consumer_complaint_narratives_archive")

    with pytest.raises(ValueError, match="prohibited V2-C3 sources"):
        builder.validate_configured_source_ids(configured)


def test_contamination_guards_reject_locked_test_and_clinc_oos(
    artifacts: builder.BuiltArtifacts,
) -> None:
    contract = builder.load_json_object(builder.DEFAULT_PATHS.contract)
    risk_by_intent = contract["risk_by_intent"]
    development = copy.deepcopy(artifacts.development_payload["examples"])
    locked = copy.deepcopy(development[0])
    locked["example_id"] = "internal-locked-test-contamination"
    locked["source_split"] = "locked_test"

    with pytest.raises(ValueError, match="locked_test contamination"):
        builder.validate_partition(
            [*development, locked],
            [],
            contract["intent_taxonomy"],
            risk_by_intent,
        )

    clinc_oos = copy.deepcopy(development[0])
    clinc_oos["example_id"] = "clinc-oos-contamination"
    clinc_oos["source_id"] = builder.CLINC_SOURCE_ID
    clinc_oos["source_split"] = "oos_train"
    with pytest.raises(ValueError, match="CLINC test or OOS contamination"):
        builder.validate_partition(
            [*development, clinc_oos],
            [],
            contract["intent_taxonomy"],
            risk_by_intent,
        )


def test_builder_contains_no_model_fitting_or_inference() -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert imported_roots <= {
        "__future__",
        "argparse",
        "collections",
        "csv",
        "dataclasses",
        "hashlib",
        "io",
        "json",
        "os",
        "pathlib",
        "tempfile",
        "typing",
        "unicodedata",
    }
    assert ".fit(" not in source
    assert ".fit_transform(" not in source
    assert ".predict(" not in source
    assert "sklearn" not in imported_roots
    assert "sentence_transformers" not in imported_roots
    assert "fastembed" not in imported_roots
    assert "joblib" not in imported_roots
