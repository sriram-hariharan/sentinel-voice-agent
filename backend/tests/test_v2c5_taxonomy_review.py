from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import review_v2c5_taxonomy as review


@pytest.fixture(scope="module")
def frozen_step18() -> review.FrozenStep18:
    return review.load_frozen_step18()


def concentration(label: str, count: int) -> dict[str, Any]:
    return {
        "counts": {label: count},
        "dominant": label,
        "dominant_proportion": 1.0,
        "proportions": {label: 1.0},
        "unit": "provenance_occurrences",
    }


def synthetic_frozen() -> review.FrozenStep18:
    clusters: list[dict[str, Any]] = []
    corpus_records: list[dict[str, Any]] = []
    for index in range(review.EXPECTED_CLUSTER_COUNT):
        cluster_id = f"cluster-{index:02d}"
        representative_id = f"discovery-{index:02d}-representative"
        boundary_id = f"discovery-{index:02d}-boundary"
        member_count = 30 + (index % 4)
        clusters.append(
            {
                "canonical_cluster_id": cluster_id,
                "member_count": member_count,
                "representative_discovery_ids": [representative_id],
                "boundary_discovery_ids": [boundary_id],
                "source_dataset_concentration": concentration(
                    "synthetic_source",
                    member_count,
                ),
                "native_external_label_concentration": concentration(
                    "synthetic_label",
                    member_count,
                ),
            }
        )
        for discovery_id, role in (
            (representative_id, "representative"),
            (boundary_id, "boundary"),
        ):
            corpus_records.append(
                {
                    "discovery_id": discovery_id,
                    "text": f"fixture utterance {index:02d} {role}",
                    "occurrences": [
                        {
                            "native_external_label": "synthetic_label",
                            "source_dataset": "synthetic_source",
                        }
                    ],
                }
            )
    return review.FrozenStep18(
        contract={"status": "frozen"},
        report={"primary": {"config_id": review.PRIMARY_CONFIG_ID}},
        assignments={},
        step18_manifest={},
        corpus={"primary_cluster_population": corpus_records},
        clusters=clusters,
        source_artifacts={
            "discovery_report": {"path": "report.json", "sha256": "a" * 64},
            "discovery_assignments": {
                "path": "assignments.json",
                "sha256": "b" * 64,
            },
            "step18_manifest": {"path": "manifest.json", "sha256": "c" * 64},
            "discovery_corpus": {"path": "corpus.json", "sha256": "d" * 64},
        },
    )


def review_kwargs(
    cluster_id: str,
    **overrides: Any,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "cluster_id": cluster_id,
        "decision": "REMAIN_UNSUPPORTED",
        "human_theme": "A human supplied theme",
        "confidence": "HIGH",
        "runtime_change_required": False,
        "human_rationale": "A human supplied rationale.",
        "target_existing_intent": None,
        "candidate_intent_name": None,
    }
    values.update(overrides)
    return values


def fully_reviewed_workfile(
    frozen: review.FrozenStep18,
) -> dict[str, Any]:
    payload = review.build_workfile_payload(frozen)
    for record in payload["clusters"]:
        payload = review.apply_review_update(
            payload,
            frozen,
            **review_kwargs(record["canonical_cluster_id"]),
        )
    return payload


def test_frozen_primary_has_exactly_35_clusters_and_pinned_sources(
    frozen_step18: review.FrozenStep18,
) -> None:
    assert frozen_step18.report["primary"]["config_id"] == "mcs30_ms10"
    assert len(frozen_step18.clusters) == 35
    assert frozen_step18.source_artifacts["discovery_report"]["sha256"] == (
        review.REPORT_SHA256
    )
    assert frozen_step18.source_artifacts["discovery_assignments"]["sha256"] == (
        review.ASSIGNMENTS_SHA256
    )
    assert frozen_step18.source_artifacts["step18_manifest"]["sha256"] == (
        review.STEP18_MANIFEST_SHA256
    )
    assert frozen_step18.source_artifacts["discovery_corpus"]["sha256"] == (
        review.CORPUS_SHA256
    )


def test_only_frozen_primary_configuration_is_accepted(
    frozen_step18: review.FrozenStep18,
) -> None:
    altered_report = copy.deepcopy(frozen_step18.report)
    altered_report["primary"]["config_id"] = "mcs15_ms5"

    with pytest.raises(ValueError, match="only primary mcs30_ms10"):
        review.extract_primary_clusters(
            altered_report,
            frozen_step18.assignments,
            frozen_step18.corpus,
        )


def test_workfile_starts_unreviewed_text_free_and_deterministically_ordered() -> None:
    frozen = synthetic_frozen()
    payload = review.build_workfile_payload(frozen)
    records = payload["clusters"]

    assert len(records) == 35
    assert all(record["review_status"] == "UNREVIEWED" for record in records)
    assert all(record["decision"] is None for record in records)
    assert all(record["runtime_change_required"] is None for record in records)
    assert records == sorted(records, key=review.cluster_sort_key)
    assert b"fixture utterance" not in review.stable_json_bytes(payload)


def test_existing_intent_allowlist_excludes_unsupported_bucket() -> None:
    frozen = synthetic_frozen()
    payload = review.build_workfile_payload(frozen)
    cluster_id = payload["clusters"][0]["canonical_cluster_id"]

    with pytest.raises(ValueError, match="allowed existing intent"):
        review.apply_review_update(
            payload,
            frozen,
            **review_kwargs(
                cluster_id,
                decision="MAP_TO_EXISTING_INTENT",
                target_existing_intent="unsupported_or_uncertain",
            ),
        )

    updated = review.apply_review_update(
        payload,
        frozen,
        **review_kwargs(
            cluster_id,
            decision="MAP_TO_EXISTING_INTENT",
            target_existing_intent="account_balance",
        ),
    )
    assert review.find_cluster(updated, cluster_id)["target_existing_intent"] == (
        "account_balance"
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"decision": "MAP_TO_EXISTING_INTENT"}, "allowed existing intent"),
        (
            {
                "decision": "MAP_TO_EXISTING_INTENT",
                "target_existing_intent": "account_balance",
                "candidate_intent_name": "candidate_name",
            },
            "cannot include a candidate",
        ),
        ({"decision": "CANDIDATE_NEW_INTENT"}, "candidate_intent_name"),
        (
            {
                "decision": "CANDIDATE_NEW_INTENT",
                "candidate_intent_name": "cash_withdrawal",
                "target_existing_intent": "account_balance",
            },
            "cannot target an existing",
        ),
        (
            {
                "decision": "CANDIDATE_NEW_INTENT",
                "candidate_intent_name": "freeze_card",
            },
            "already an existing intent",
        ),
        (
            {
                "decision": "NEEDS_SPLIT_REVIEW",
                "candidate_intent_name": "split_candidate",
            },
            "requires empty existing and candidate",
        ),
        ({"human_theme": "   "}, "human_theme"),
        ({"human_rationale": ""}, "human_rationale"),
        ({"confidence": "CERTAIN"}, "HIGH, MEDIUM, or LOW"),
        ({"runtime_change_required": "null"}, "true, false, or null"),
    ],
)
def test_decision_specific_requirements(
    overrides: dict[str, Any],
    match: str,
) -> None:
    frozen = synthetic_frozen()
    payload = review.build_workfile_payload(frozen)
    cluster_id = payload["clusters"][0]["canonical_cluster_id"]

    with pytest.raises((TypeError, ValueError), match=match):
        review.apply_review_update(
            payload,
            frozen,
            **review_kwargs(cluster_id, **overrides),
        )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("true", True), ("false", False), ("null", None)],
)
def test_runtime_change_cli_parses_json_values(
    raw_value: str,
    expected: bool | None,
) -> None:
    args = review.build_parser().parse_args(
        [
            "set",
            "--cluster-id",
            "cluster-test",
            "--decision",
            "INSUFFICIENT_EVIDENCE",
            "--human-theme",
            "Human theme",
            "--confidence",
            "LOW",
            "--runtime-change-required",
            raw_value,
            "--human-rationale",
            "Human rationale",
        ]
    )

    assert args.runtime_change_required is expected


@pytest.mark.parametrize("raw_value", ["True", "FALSE", "none", "1", ""])
def test_runtime_change_parser_rejects_every_other_value(
    raw_value: str,
) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="true, false, or null",
    ):
        review.parse_nullable_boolean(raw_value)


@pytest.mark.parametrize(
    "decision",
    [
        "NEEDS_SPLIT_REVIEW",
        "MIXED_OR_INCOHERENT",
        "INSUFFICIENT_EVIDENCE",
    ],
)
def test_unresolved_decisions_persist_runtime_change_as_json_null(
    decision: str,
) -> None:
    frozen = synthetic_frozen()
    payload = review.build_workfile_payload(frozen)
    cluster_id = payload["clusters"][0]["canonical_cluster_id"]
    updated = review.apply_review_update(
        payload,
        frozen,
        **review_kwargs(
            cluster_id,
            decision=decision,
            runtime_change_required=None,
        ),
    )
    serialized = review.stable_json_bytes(updated)
    persisted = json.loads(serialized)
    record = review.find_cluster(persisted, cluster_id)

    assert record["review_status"] == "REVIEWED"
    assert record["runtime_change_required"] is None
    assert b'"runtime_change_required": null' in serialized


def test_protected_action_names_are_never_inferred_from_theme_or_metadata() -> None:
    frozen = synthetic_frozen()
    payload = review.build_workfile_payload(frozen)
    cluster_id = payload["clusters"][0]["canonical_cluster_id"]
    updated = review.apply_review_update(
        payload,
        frozen,
        **review_kwargs(
            cluster_id,
            human_theme="Questions mentioning a frozen card",
            human_rationale="Topic similarity is not action semantics.",
        ),
    )
    record = review.find_cluster(updated, cluster_id)

    assert record["target_existing_intent"] is None
    assert record["decision"] == "REMAIN_UNSUPPORTED"
    assert review.workfile_governance()[
        "protected_action_intents_inferred_automatically"
    ] is False


def test_set_update_preserves_unrelated_reviewed_records() -> None:
    frozen = synthetic_frozen()
    payload = review.build_workfile_payload(frozen)
    first_id = payload["clusters"][0]["canonical_cluster_id"]
    second_id = payload["clusters"][1]["canonical_cluster_id"]
    first_update = review.apply_review_update(
        payload,
        frozen,
        **review_kwargs(first_id),
    )
    first_record = copy.deepcopy(review.find_cluster(first_update, first_id))

    second_update = review.apply_review_update(
        first_update,
        frozen,
        **review_kwargs(
            second_id,
            decision="CANDIDATE_NEW_INTENT",
            candidate_intent_name="human_named_candidate",
            confidence="MEDIUM",
            runtime_change_required=True,
        ),
    )

    assert review.find_cluster(second_update, first_id) == first_record
    assert review.find_cluster(second_update, second_id)["review_status"] == (
        "REVIEWED"
    )


def test_check_never_overwrites_human_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frozen = synthetic_frozen()
    payload = review.build_workfile_payload(frozen)
    cluster_id = payload["clusters"][0]["canonical_cluster_id"]
    payload = review.apply_review_update(
        payload,
        frozen,
        **review_kwargs(cluster_id),
    )
    workfile = tmp_path / "v2c5_taxonomy_review_workfile.json"
    workfile.write_bytes(review.stable_json_bytes(payload))
    paths = replace(review.DEFAULT_PATHS, workfile=workfile)
    before = workfile.read_bytes()
    monkeypatch.setattr(review, "load_frozen_step18", lambda unused: frozen)

    summary = review.check_workfile(paths)

    assert summary["reviewed"] == 1
    assert workfile.read_bytes() == before


def test_show_joins_frozen_text_and_separates_review_sections() -> None:
    frozen = synthetic_frozen()
    payload = review.build_workfile_payload(frozen)
    cluster = payload["clusters"][0]

    shown = review.display_cluster_payload(cluster, frozen)

    assert shown["section_order"] == [
        "cluster",
        "representative_examples",
        "boundary_examples",
    ]
    assert shown["representative_examples"][0]["text"].endswith(
        "representative"
    )
    assert shown["boundary_examples"][0]["text"].endswith("boundary")
    assert "metadata only" in shown["metadata_notice"]


def test_export_refuses_while_any_cluster_is_unreviewed() -> None:
    frozen = synthetic_frozen()
    payload = review.build_workfile_payload(frozen)

    with pytest.raises(ValueError, match="all 35 clusters"):
        review.build_adjudication_payload(payload, frozen)


def test_export_is_text_free_and_keeps_step20_governance() -> None:
    frozen = synthetic_frozen()
    workfile = fully_reviewed_workfile(frozen)
    adjudication = review.build_adjudication_payload(workfile, frozen)
    serialized = review.stable_json_bytes(adjudication)

    assert b"fixture utterance" not in serialized
    assert adjudication["reviewed_count"] == 35
    assert adjudication["taxonomy_changed"] is False
    assert adjudication["new_intents_created"] is False
    assert adjudication["classifier_training_performed"] is False
    assert adjudication["runtime_behavior_changed"] is False
    assert adjudication["final_taxonomy_frozen"] is False
    assert adjudication["step20_required"] is True
    assert adjudication["human_adjudication_completed"] is True
    review.validate_text_free_artifact(adjudication)


def test_manifest_pins_sources_script_counts_and_governance() -> None:
    frozen = synthetic_frozen()
    workfile = fully_reviewed_workfile(frozen)
    adjudication = review.build_adjudication_payload(workfile, frozen)
    adjudication_bytes = review.stable_json_bytes(adjudication)
    manifest = review.build_adjudication_manifest(
        adjudication_bytes,
        adjudication,
        review.DEFAULT_PATHS,
    )

    assert manifest["source_artifacts"] == frozen.source_artifacts
    assert manifest["review_script"]["path"] == (
        "scripts/review_v2c5_taxonomy.py"
    )
    assert len(manifest["review_script"]["sha256"]) == 64
    assert manifest["cluster_count"] == 35
    assert manifest["reviewed_count"] == 35
    assert manifest["decision_counts"]["REMAIN_UNSUPPORTED"] == 35
    assert manifest["taxonomy_changed"] is False
    assert manifest["human_adjudication_completed"] is True


def test_sealed_v2c4_holdout_is_guarded_and_not_an_input() -> None:
    input_paths = {
        review.DEFAULT_PATHS.contract,
        review.DEFAULT_PATHS.report,
        review.DEFAULT_PATHS.assignments,
        review.DEFAULT_PATHS.step18_manifest,
        review.DEFAULT_PATHS.corpus,
        review.DEFAULT_PATHS.step18_script,
    }
    assert all(
        path.name not in review.FORBIDDEN_INPUT_FILENAMES
        for path in input_paths
    )
    for filename in review.FORBIDDEN_INPUT_FILENAMES:
        with pytest.raises(ValueError, match="sealed V2-C4 holdout"):
            review.read_input_bytes(review.ML_ROOT / filename)
