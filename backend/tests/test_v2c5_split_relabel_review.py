from __future__ import annotations

import ast
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import review_v2c5_split_relabels as review


@pytest.fixture(scope="module")
def frozen_sources() -> review.FrozenSources:
    return review.load_frozen_sources()


@pytest.fixture()
def workfile(frozen_sources: review.FrozenSources) -> dict[str, Any]:
    return review.build_workfile_payload(frozen_sources)


def fully_reviewed_workfile(
    frozen_sources: review.FrozenSources,
) -> dict[str, Any]:
    payload = review.build_workfile_payload(frozen_sources)
    for record in payload["records"]:
        record.update(
            {
                "confidence": "HIGH",
                "review_status": "REVIEWED",
                "reviewer_note": "",
                "selected_final_intent": "informational_policy",
            }
        )
    review.validate_workfile_payload(payload, frozen_sources)
    return payload


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


def test_all_frozen_input_hashes_are_validated(
    frozen_sources: review.FrozenSources,
) -> None:
    assert frozen_sources.source_artifacts == review.source_artifact_metadata(
        review.DEFAULT_PATHS
    )
    for _, path, expected_hash in review.source_path_items(review.DEFAULT_PATHS):
        assert review.sha256_bytes(path.read_bytes()) == expected_hash


def test_manual_population_is_exactly_578(
    frozen_sources: review.FrozenSources,
) -> None:
    assert len(frozen_sources.manual_records) == 578


def test_manual_per_cluster_counts_are_exact(
    frozen_sources: review.FrozenSources,
) -> None:
    counts: dict[str, int] = {}
    for record in frozen_sources.manual_records:
        cluster_id = record["canonical_cluster_id"]
        counts[cluster_id] = counts.get(cluster_id, 0) + 1

    assert counts == {
        "cluster-25a4740ae330f948": 161,
        "cluster-2b49d15c7f2e42b9": 44,
        "cluster-3db20683a0becc1c": 86,
        "cluster-783d0d28bd4c4992": 66,
        "cluster-bab7911ea80f612b": 221,
    }


def test_deterministic_population_is_exactly_397(
    frozen_sources: review.FrozenSources,
) -> None:
    assert len(frozen_sources.deterministic_records) == 397


def test_total_split_coverage_is_exactly_975(
    frozen_sources: review.FrozenSources,
) -> None:
    assert len(frozen_sources.split_records) == 975
    assert len(frozen_sources.manual_records) + len(
        frozen_sources.deterministic_records
    ) == 975


def test_split_records_get_cluster_id_from_primary_assignment_without_mutation(
    frozen_sources: review.FrozenSources,
) -> None:
    corpus = copy.deepcopy(frozen_sources.discovery_corpus)
    assignments = copy.deepcopy(frozen_sources.discovery_assignments)
    development = copy.deepcopy(frozen_sources.development_dataset)
    corpus_before = copy.deepcopy(corpus)
    assignments_before = copy.deepcopy(assignments)
    development_before = copy.deepcopy(development)
    raw_records = corpus["primary_cluster_population"]

    assert all("canonical_cluster_id" not in record for record in raw_records)

    split_records, _, _ = review.partition_split_records(
        corpus,
        assignments,
        review.validate_development_dataset(development),
    )
    primary_assignments = {
        value["discovery_id"]: value["primary_canonical_cluster_id"]
        for value in assignments["assignments"]
    }

    assert all(
        record["canonical_cluster_id"]
        == primary_assignments[record["discovery_id"]]
        for record in split_records
    )
    assert all(
        isinstance(record["canonical_cluster_id"], str)
        and record["canonical_cluster_id"]
        for record in split_records
    )
    assert corpus == corpus_before
    assert assignments == assignments_before
    assert development == development_before


def test_deterministic_native_label_mapping_table_is_exact() -> None:
    assert review.DETERMINISTIC_NATIVE_LABEL_MAPPINGS == {
        "cluster-25a4740ae330f948": {
            "disposable_card_limits": "informational_policy"
        },
        "cluster-2b49d15c7f2e42b9": {
            "card_delivery_estimate": "informational_policy"
        },
        "cluster-3db20683a0becc1c": {
            "__INTERNAL_OR_NULL__": "unsupported_or_uncertain"
        },
        "cluster-783d0d28bd4c4992": {
            "topping_up_by_card": "unsupported_or_uncertain"
        },
        "cluster-a077797c96f814ce": {
            "pending_transfer": "transfer_pending",
            "transfer_not_received_by_recipient": "transfer_pending",
            "transfer_timing": "informational_policy",
        },
        "cluster-bab7911ea80f612b": {},
        "cluster-c29614d54555e30c": {
            "account_blocked": "account_blocked",
            "freeze_account": "unsupported_or_uncertain",
        },
        "cluster-d8ae664886c213a2": {
            "__INTERNAL_OR_NULL__": "unsupported_or_uncertain",
            "credit_limit": "unsupported_or_uncertain",
            "credit_limit_change": "unsupported_or_uncertain",
        },
        "cluster-e92864f71372fce3": {
            "unable_to_verify_identity": "unsupported_or_uncertain",
            "verify_my_identity": "informational_policy",
            "verify_top_up": "unsupported_or_uncertain",
        },
    }
    assert review.DETERMINISTIC_NATIVE_LABEL_MAPPINGS[
        "cluster-c29614d54555e30c"
    ]["freeze_account"] != "freeze_card"


def test_populations_have_no_duplicates_or_overlap(
    frozen_sources: review.FrozenSources,
) -> None:
    manual_ids = {
        value["discovery_id"] for value in frozen_sources.manual_records
    }
    deterministic_ids = {
        value["discovery_id"]
        for value in frozen_sources.deterministic_records
    }

    assert len(manual_ids) == 578
    assert len(deterministic_ids) == 397
    assert manual_ids.isdisjoint(deterministic_ids)


def test_workfile_build_is_deterministic(
    frozen_sources: review.FrozenSources,
) -> None:
    first = review.stable_json_bytes(
        review.build_workfile_payload(frozen_sources)
    )
    second = review.stable_json_bytes(
        review.build_workfile_payload(frozen_sources)
    )

    assert first == second
    assert first.endswith(b"\n")


def test_build_refuses_to_overwrite_existing_workfile(
    frozen_sources: review.FrozenSources,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workfile_path = tmp_path / review.DEFAULT_PATHS.workfile.name
    workfile_path.write_bytes(b"preserve-human-work\n")
    paths = replace(review.DEFAULT_PATHS, workfile=workfile_path)
    monkeypatch.setattr(
        review,
        "load_frozen_sources",
        lambda unused_paths: frozen_sources,
    )

    with pytest.raises(FileExistsError, match="will not overwrite"):
        review.build_workfile(paths)
    assert workfile_path.read_bytes() == b"preserve-human-work\n"


def test_workfile_starts_entirely_unreviewed(
    workfile: dict[str, Any],
) -> None:
    assert len(workfile["records"]) == 578
    assert all(value["review_status"] == "UNREVIEWED" for value in workfile["records"])
    assert all(value["selected_final_intent"] is None for value in workfile["records"])
    assert all(value["confidence"] is None for value in workfile["records"])


def test_source_and_native_labels_are_metadata_only(
    workfile: dict[str, Any],
) -> None:
    serialized = review.stable_json_bytes(workfile)

    assert workfile["governance"]["external_labels_are_metadata_only"] is True
    assert workfile["governance"][
        "native_labels_are_automatic_truth_for_manual_rows"
    ] is False
    assert b'"text"' not in serialized
    assert all("native_label_key" in value for value in workfile["records"])
    assert all(value["selected_final_intent"] is None for value in workfile["records"])


def test_allowed_manual_labels_are_limited_to_info_and_unsupported(
    workfile: dict[str, Any],
) -> None:
    assert {
        tuple(value["allowed_final_intents"]) for value in workfile["records"]
    } == {("informational_policy", "unsupported_or_uncertain")}


def test_protected_intents_are_rejected(
    frozen_sources: review.FrozenSources,
    workfile: dict[str, Any],
) -> None:
    discovery_id = workfile["records"][0]["discovery_id"]

    with pytest.raises(ValueError, match="protected intents are forbidden"):
        review.apply_review_update(
            workfile,
            frozen_sources,
            discovery_id=discovery_id,
            selected_final_intent="freeze_card",
            confidence="HIGH",
            reviewer_note=None,
        )


def test_protected_intents_are_rejected_from_deterministic_population(
    frozen_sources: review.FrozenSources,
) -> None:
    deterministic_records = copy.deepcopy(frozen_sources.deterministic_records)
    deterministic_records[0]["final_intent"] = "freeze_card"

    with pytest.raises(ValueError, match="protected intents are forbidden"):
        review.validate_partition_counts(
            frozen_sources.split_records,
            frozen_sources.manual_records,
            deterministic_records,
        )


def test_wrong_cluster_label_is_rejected(
    frozen_sources: review.FrozenSources,
    workfile: dict[str, Any],
) -> None:
    discovery_id = workfile["records"][0]["discovery_id"]

    with pytest.raises(ValueError, match="not allowed for this cluster"):
        review.apply_review_update(
            workfile,
            frozen_sources,
            discovery_id=discovery_id,
            selected_final_intent="account_blocked",
            confidence="HIGH",
            reviewer_note=None,
        )


def test_check_writes_nothing(
    frozen_sources: review.FrozenSources,
    workfile: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workfile_path = tmp_path / review.DEFAULT_PATHS.workfile.name
    workfile_path.write_bytes(review.stable_json_bytes(workfile))
    paths = replace(
        review.DEFAULT_PATHS,
        workfile=workfile_path,
        adjudication_output=tmp_path / review.DEFAULT_PATHS.adjudication_output.name,
        adjudication_manifest_output=(
            tmp_path / review.DEFAULT_PATHS.adjudication_manifest_output.name
        ),
    )
    monkeypatch.setattr(
        review,
        "load_frozen_sources",
        lambda unused_paths: frozen_sources,
    )
    before = workfile_path.read_bytes()

    summary = review.check_workfile(paths)

    assert summary["total"] == 578
    assert workfile_path.read_bytes() == before
    assert not paths.adjudication_output.exists()
    assert not paths.adjudication_manifest_output.exists()


def test_show_and_next_join_only_frozen_corpus_text(
    frozen_sources: review.FrozenSources,
    workfile: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workfile_path = tmp_path / review.DEFAULT_PATHS.workfile.name
    workfile_path.write_bytes(review.stable_json_bytes(workfile))
    paths = replace(review.DEFAULT_PATHS, workfile=workfile_path)
    monkeypatch.setattr(
        review,
        "load_frozen_sources",
        lambda unused_paths: frozen_sources,
    )
    first = workfile["records"][0]
    expected_text = frozen_sources.corpus_by_id[first["discovery_id"]]["text"]

    shown = review.show_record(first["discovery_id"], paths)
    next_value = review.next_record(paths)

    assert shown["text"] == expected_text
    assert next_value["text"] == expected_text
    assert shown["record"]["allowed_final_intents"] == [
        "informational_policy",
        "unsupported_or_uncertain",
    ]
    assert "metadata/evidence only" in shown["metadata_notice"]
    assert len(shown["branch_semantics"]) == 2


def test_set_persists_only_explicit_human_decision(
    frozen_sources: review.FrozenSources,
    workfile: dict[str, Any],
) -> None:
    target = workfile["records"][0]
    untouched = copy.deepcopy(workfile["records"][1])

    updated = review.apply_review_update(
        workfile,
        frozen_sources,
        discovery_id=target["discovery_id"],
        selected_final_intent="unsupported_or_uncertain",
        confidence="MEDIUM",
        reviewer_note="Direct operation request",
    )
    selected = review.find_record(updated, target["discovery_id"])

    assert selected["review_status"] == "REVIEWED"
    assert selected["selected_final_intent"] == "unsupported_or_uncertain"
    assert selected["confidence"] == "MEDIUM"
    assert selected["reviewer_note"] == "Direct operation request"
    assert updated["records"][1] == untouched


def test_invalid_confidence_is_rejected(
    frozen_sources: review.FrozenSources,
    workfile: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="HIGH, MEDIUM, or LOW"):
        review.apply_review_update(
            workfile,
            frozen_sources,
            discovery_id=workfile["records"][0]["discovery_id"],
            selected_final_intent="informational_policy",
            confidence="CERTAIN",
            reviewer_note=None,
        )


def test_export_is_blocked_while_any_record_is_unreviewed(
    frozen_sources: review.FrozenSources,
    workfile: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="all 578"):
        review.build_adjudication_payload(workfile, frozen_sources)


def test_export_succeeds_after_all_records_are_reviewed_and_is_text_free(
    frozen_sources: review.FrozenSources,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    completed = fully_reviewed_workfile(frozen_sources)
    workfile_path = tmp_path / review.DEFAULT_PATHS.workfile.name
    workfile_path.write_bytes(review.stable_json_bytes(completed))
    paths = replace(
        review.DEFAULT_PATHS,
        workfile=workfile_path,
        adjudication_output=tmp_path / review.DEFAULT_PATHS.adjudication_output.name,
        adjudication_manifest_output=(
            tmp_path / review.DEFAULT_PATHS.adjudication_manifest_output.name
        ),
    )
    monkeypatch.setattr(
        review,
        "load_frozen_sources",
        lambda unused_paths: frozen_sources,
    )

    result = review.export_adjudication(paths)
    exported = json.loads(paths.adjudication_output.read_text(encoding="utf-8"))
    manifest = json.loads(
        paths.adjudication_manifest_output.read_text(encoding="utf-8")
    )
    keys = recursive_keys(exported)

    assert result["summary"]["reviewed"] == 578
    assert exported["all_reviewed"] is True
    assert exported["manual_review_count"] == 578
    assert exported["deterministic_mapping_count"] == 397
    assert exported["split_coverage_count"] == 975
    assert len(exported["manual_adjudications"]) == 578
    assert len(exported["deterministic_mappings"]) == 397
    assert {"text", "normalized_text", "utterance", "examples"}.isdisjoint(keys)
    assert manifest["adjudication"]["sha256"] == review.sha256_bytes(
        paths.adjudication_output.read_bytes()
    )


def test_export_bytes_are_deterministic(
    frozen_sources: review.FrozenSources,
) -> None:
    completed = fully_reviewed_workfile(frozen_sources)

    first = review.expected_export_bytes(
        completed,
        frozen_sources,
        review.DEFAULT_PATHS,
    )
    second = review.expected_export_bytes(
        completed,
        frozen_sources,
        review.DEFAULT_PATHS,
    )

    assert first == second
    assert all(value.endswith(b"\n") for value in first)


@pytest.mark.parametrize(
    "path",
    [
        Path("data/evals/v2/ml/v2c4_safety_holdout.json"),
        Path("data/evals/v2/ml/v2c3_challenge.json"),
        Path("data/evals/v2/ml/cfpb.json"),
        Path("data/evals/v2/ml/external_final_lockbox.json"),
        Path("data/evals/v2/ml/test_dataset.json"),
        Path("data/evals/v2/ml/final_dataset.json"),
        Path("data/evals/v2/ml/cfpb/v2c5_taxonomy_freeze.json"),
    ],
)
def test_final_test_holdout_and_cfpb_inputs_are_prohibited(path: Path) -> None:
    with pytest.raises(ValueError):
        review.guard_source_path(path)


def test_workflow_imports_only_python_standard_library() -> None:
    script_path = Path(review.__file__).resolve()
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    imported_roots = {
        node.names[0].name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
    } | {
        str(node.module).split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    }

    assert imported_roots == {
        "argparse",
        "collections",
        "copy",
        "dataclasses",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "tempfile",
        "typing",
    }
    source = script_path.read_text(encoding="utf-8")
    for prohibited in (
        "fastembed",
        "sklearn",
        "numpy",
        "pandas",
        "HDBSCAN(",
        "UMAP(",
        "fit_predict(",
    ):
        assert prohibited not in source
