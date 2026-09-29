from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from scripts import build_v2c4_development_data as builder

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/build_v2c4_development_data.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifacts() -> builder.BuiltArtifacts:
    return builder.build_artifacts()


@pytest.fixture(scope="module")
def augmentation(artifacts: builder.BuiltArtifacts) -> list[dict[str, Any]]:
    return artifacts.augmentation_payload["examples"]


@pytest.fixture(scope="module")
def probe(artifacts: builder.BuiltArtifacts) -> list[dict[str, Any]]:
    return artifacts.probe_payload["examples"]


def test_training_augmentation_exact_lane_counts(
    augmentation: list[dict[str, Any]],
) -> None:
    assert len(augmentation) == 360
    assert Counter(row["design_lane"] for row in augmentation) == {
        "protected_positive": 120,
        "protected_action_hard_negative": 120,
        "unsupported_or_scope": 120,
    }


def test_protected_positive_balance_and_risk(
    augmentation: list[dict[str, Any]],
) -> None:
    positives = [
        row for row in augmentation if row["design_lane"] == "protected_positive"
    ]
    assert Counter(row["intent"] for row in positives) == {
        "create_dispute": 60,
        "freeze_card": 60,
    }
    assert {row["risk"] for row in positives} == {"PROTECTED_WRITE"}


def test_hard_negatives_use_frozen_multiple_intent_distribution(
    augmentation: list[dict[str, Any]],
) -> None:
    negatives = [
        row
        for row in augmentation
        if row["design_lane"] == "protected_action_hard_negative"
    ]
    expected = {
        "account_balance": 12,
        "card_status": 18,
        "escalation": 12,
        "informational_policy": 24,
        "recent_transactions": 18,
        "transaction_details": 18,
        "unsupported_or_uncertain": 18,
    }
    assert Counter(row["intent"] for row in negatives) == expected
    assert len(expected) == 7
    assert len(negatives) == 120
    assert any(row["intent"] != "unsupported_or_uncertain" for row in negatives)


def test_unsupported_scope_lane_has_correct_label(
    augmentation: list[dict[str, Any]],
) -> None:
    unsupported = [
        row
        for row in augmentation
        if row["design_lane"] == "unsupported_or_scope"
    ]
    assert len(unsupported) == 120
    assert {row["intent"] for row in unsupported} == {"unsupported_or_uncertain"}
    assert {row["risk"] for row in unsupported} == {"ESCALATION_OR_UNCERTAIN"}


def test_augmentation_eligibility_flags(
    augmentation: list[dict[str, Any]],
) -> None:
    assert all(row["training_eligible"] is True for row in augmentation)
    assert all(row["model_selection_eligible"] is False for row in augmentation)
    assert all(
        row["threshold_selection_eligible"] is False for row in augmentation
    )
    assert all(row["final_acceptance_evidence"] is False for row in augmentation)


def test_selection_probe_exact_intent_and_risk_counts(
    probe: list[dict[str, Any]],
) -> None:
    assert len(probe) == 270
    assert Counter(row["intent"] for row in probe) == dict.fromkeys(
        builder.EXPECTED_INTENTS, 30
    )
    assert Counter(row["risk"] for row in probe) == {
        "ESCALATION_OR_UNCERTAIN": 60,
        "PRIVATE_READ": 120,
        "PROTECTED_WRITE": 60,
        "PUBLIC": 30,
    }


def test_selection_probe_eligibility_flags(probe: list[dict[str, Any]]) -> None:
    assert all(row["training_eligible"] is False for row in probe)
    assert all(row["model_selection_eligible"] is True for row in probe)
    assert all(row["threshold_selection_eligible"] is False for row in probe)
    assert all(row["final_acceptance_evidence"] is False for row in probe)


def test_normalized_texts_are_unique_with_zero_cross_overlap(
    augmentation: list[dict[str, Any]], probe: list[dict[str, Any]]
) -> None:
    augmentation_hashes = {row["normalized_text_sha256"] for row in augmentation}
    probe_hashes = {row["normalized_text_sha256"] for row in probe}
    assert len(augmentation_hashes) == len(augmentation)
    assert len(probe_hashes) == len(probe)
    assert augmentation_hashes.isdisjoint(probe_hashes)


def test_cross_dataset_ids_groups_and_lineages_are_disjoint(
    augmentation: list[dict[str, Any]], probe: list[dict[str, Any]]
) -> None:
    for field in ("example_id", "group_id", "lineage_id"):
        augmentation_values = {row[field] for row in augmentation}
        probe_values = {row[field] for row in probe}
        assert augmentation_values.isdisjoint(probe_values)


def test_zero_overlap_with_existing_v2c3_references(
    artifacts: builder.BuiltArtifacts,
) -> None:
    for manifest in (
        artifacts.augmentation_manifest_payload,
        artifacts.probe_manifest_payload,
    ):
        overlaps = manifest["reference_dataset_overlap_results"]
        assert overlaps["v2c3_development"]["normalized_text_overlap"] == 0
        assert overlaps["consumed_v2c3_challenge"][
            "normalized_text_overlap"
        ] == 0
        assert overlaps["consumed_v2c3_external_lockbox"][
            "normalized_text_overlap"
        ] == 0


def test_reference_overlap_fails_with_safe_record_level_diagnostics() -> None:
    normalized_hash = "a" * 64
    reference_text = "REFERENCE RAW TEXT MUST NOT APPEAR"
    reference = {
        "examples": [
            {
                "example_id": "v2c3-development:fixture:001",
                "data_role": "development",
                "normalized_text_sha256": normalized_hash,
                "text": reference_text,
            }
        ]
    }
    index = builder.reference_index(
        reference, "examples", "v2c3_development"
    )
    assert index == {
        normalized_hash: [
            {
                "reference_id": "v2c3-development:fixture:001",
                "reference_role": "development",
            }
        ]
    }
    new_text = "Show me the newly authored Step 10 request."
    new_records = [
        {
            "data_role": "v2c4_model_selection_probe",
            "example_id": "v2c4-probe:fixture:001",
            "intent": "transaction_details",
            "text": new_text,
            "normalized_text_sha256": normalized_hash,
        }
    ]

    with pytest.raises(ValueError) as caught:
        builder.raise_for_reference_collisions(
            new_records, {"v2c3_development": index}
        )

    message = str(caught.value)
    assert "Leakage detected:" in message
    assert "new=v2c4_model_selection_probe" in message
    assert "example_id=v2c4-probe:fixture:001" in message
    assert "intent=transaction_details" in message
    assert f"normalized_text_sha256={normalized_hash}" in message
    assert "reference=v2c3_development" in message
    assert "reference_role=development" in message
    assert "reference_id=v2c3-development:fixture:001" in message
    assert new_text in message
    assert reference_text not in message


def test_sealed_holdout_files_and_cfpb_are_not_accessed(
    artifacts: builder.BuiltArtifacts,
) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'ML_ROOT / "v2c4_safety_holdout.json"' not in source
    assert 'ML_ROOT / "v2c4_safety_holdout_seed.json"' not in source
    assert "cfpb_narratives" not in source.lower()
    for manifest in (
        artifacts.augmentation_manifest_payload,
        artifacts.probe_manifest_payload,
    ):
        limitation = manifest["sealed_v2c4_holdout_overlap_check"]
        assert limitation["exact_check_performed"] is False
        assert limitation["sealed_holdout_or_seed_opened"] is False
        assert limitation["manifest_path"].endswith(
            "v2c4_safety_holdout.manifest.json"
        )
        assert manifest["source_policy"]["cfpb_used"] is False
        assert manifest["source_policy"][
            "v2c4_final_holdout_used_for_authoring"
        ] is False


def test_required_record_metadata_and_frozen_risk_mapping(
    augmentation: list[dict[str, Any]], probe: list[dict[str, Any]]
) -> None:
    required = {
        "example_id",
        "source_id",
        "data_role",
        "text",
        "text_sha256",
        "normalized_text_sha256",
        "intent",
        "risk",
        "family_id",
        "group_id",
        "lineage_id",
        "design_lane",
        "tags",
        "training_eligible",
        "model_selection_eligible",
        "threshold_selection_eligible",
        "final_acceptance_evidence",
    }
    for row in [*augmentation, *probe]:
        assert required.issubset(row)
        assert row["risk"] == builder.EXPECTED_RISK_BY_INTENT[row["intent"]]


def test_customer_text_excludes_internal_engineering_phrase(
    augmentation: list[dict[str, Any]], probe: list[dict[str, Any]]
) -> None:
    assert all(
        "protected action" not in row["text"].lower()
        for row in [*augmentation, *probe]
    )


def test_seed_provenance_and_required_augmentation_coverage() -> None:
    for seed_path in (
        builder.DEFAULT_PATHS.augmentation_seed,
        builder.DEFAULT_PATHS.probe_seed,
    ):
        seed = load_json(seed_path)
        assert seed["authorship_provenance"] == builder.REQUIRED_PROVENANCE
        assert seed["parent_intervention_plan_sha256"] == (
            builder.development_builder.sha256_bytes(
                builder.DEFAULT_PATHS.intervention_plan.read_bytes()
            )
        )
    augmentation_seed = load_json(builder.DEFAULT_PATHS.augmentation_seed)
    for lane, required_tags in builder.REQUIRED_AUGMENTATION_TAGS_BY_LANE.items():
        actual_tags = {
            tag
            for family in augmentation_seed["families"]
            if family["design_lane"] == lane
            for tag in family["tags"]
        }
        assert required_tags.issubset(actual_tags)


def test_manifests_record_no_training_inference_or_evaluation(
    artifacts: builder.BuiltArtifacts,
) -> None:
    for manifest in (
        artifacts.augmentation_manifest_payload,
        artifacts.probe_manifest_payload,
    ):
        assert manifest["training_or_evaluation_performed"] is False
        assert all(value is False for value in manifest["execution_status"].values())


def test_builder_has_no_model_training_embedding_or_inference_calls() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for prohibited in (
        ".fit(",
        ".fit_transform(",
        ".predict(",
        ".decision_function(",
        "passage_embed(",
        "SentenceTransformer(",
        "LinearSVC(",
        "joblib.load(",
    ):
        assert prohibited not in source


def test_rebuild_is_deterministic_and_parent_plan_is_referenced(
    artifacts: builder.BuiltArtifacts,
) -> None:
    rebuilt = builder.build_artifacts()
    assert rebuilt.augmentation_bytes == artifacts.augmentation_bytes
    assert rebuilt.augmentation_manifest_bytes == artifacts.augmentation_manifest_bytes
    assert rebuilt.probe_bytes == artifacts.probe_bytes
    assert rebuilt.probe_manifest_bytes == artifacts.probe_manifest_bytes
    plan_hash = builder.development_builder.sha256_bytes(
        builder.DEFAULT_PATHS.intervention_plan.read_bytes()
    )
    assert artifacts.augmentation_manifest_payload[
        "parent_intervention_plan_sha256"
    ] == plan_hash
    assert artifacts.probe_manifest_payload[
        "parent_intervention_plan_sha256"
    ] == plan_hash
