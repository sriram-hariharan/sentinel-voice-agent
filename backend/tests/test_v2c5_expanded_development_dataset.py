from __future__ import annotations

import ast
import copy
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import build_v2c5_expanded_development_dataset as builder


@pytest.fixture(scope="module")
def sources() -> builder.FrozenSources:
    return builder.load_frozen_sources()


@pytest.fixture(scope="module")
def artifacts(sources: builder.FrozenSources) -> builder.BuiltArtifacts:
    return builder.build_artifacts(sources=sources)


@pytest.fixture(scope="module")
def output_records(artifacts: builder.BuiltArtifacts) -> list[dict[str, Any]]:
    return artifacts.dataset_payload["examples"]


def first_historical_unsupported(
    sources: builder.FrozenSources,
) -> dict[str, Any]:
    return next(
        record
        for record in sources.development_examples
        if record["intent"] == builder.UNSUPPORTED_INTENT
    )


def records_for_source(
    records: list[dict[str, Any]],
    relabel_source: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["v2c5_relabel_source"] == relabel_source
    ]


def test_all_frozen_input_hashes_are_pinned() -> None:
    for _, path, expected_hash in builder.source_path_items(builder.DEFAULT_PATHS):
        assert builder.sha256_bytes(path.read_bytes()) == expected_hash


def test_output_contains_exactly_8198_occurrences(
    artifacts: builder.BuiltArtifacts,
) -> None:
    assert artifacts.dataset_payload["example_count"] == 8198
    assert len(artifacts.dataset_payload["examples"]) == 8198


def test_historical_unsupported_population_is_exactly_6373(
    sources: builder.FrozenSources,
) -> None:
    assert sum(
        record["intent"] == builder.UNSUPPORTED_INTENT
        for record in sources.development_examples
    ) == 6373


def test_unsupported_occurrences_resolve_to_6372_discovery_texts(
    output_records: list[dict[str, Any]],
) -> None:
    discovery_ids = [
        record["v2c5_discovery_id"]
        for record in output_records
        if record["previous_intent"] == builder.UNSUPPORTED_INTENT
    ]

    assert len(discovery_ids) == 6373
    assert len(set(discovery_ids)) == 6372


def test_normalized_duplicate_is_preserved_with_same_relabel(
    output_records: list[dict[str, Any]],
) -> None:
    counts = Counter(
        record["v2c5_discovery_id"]
        for record in output_records
        if record["previous_intent"] == builder.UNSUPPORTED_INTENT
    )
    duplicate_id = next(key for key, count in counts.items() if count == 2)
    duplicate_records = [
        record
        for record in output_records
        if record["v2c5_discovery_id"] == duplicate_id
    ]

    assert len(duplicate_records) == 2
    assert len({record["example_id"] for record in duplicate_records}) == 2
    assert {
        (
            record["intent"],
            record["v2c5_primary_canonical_cluster_id"],
            record["v2c5_relabel_source"],
        )
        for record in duplicate_records
    } == {
        (
            duplicate_records[0]["intent"],
            duplicate_records[0]["v2c5_primary_canonical_cluster_id"],
            duplicate_records[0]["v2c5_relabel_source"],
        )
    }


def test_existing_supported_intents_are_unchanged(
    sources: builder.FrozenSources,
    output_records: list[dict[str, Any]],
) -> None:
    for source, output in zip(
        sources.development_examples,
        output_records,
        strict=True,
    ):
        if source["intent"] != builder.UNSUPPORTED_INTENT:
            assert output["intent"] == source["intent"]
            assert output["v2c5_relabel_source"] == (
                builder.RETAINED_EXISTING_INTENT
            )


def test_all_output_labels_and_risks_match_frozen_taxonomy(
    sources: builder.FrozenSources,
    artifacts: builder.BuiltArtifacts,
    output_records: list[dict[str, Any]],
) -> None:
    assert {record["intent"] for record in output_records}.issubset(
        sources.intent_labels
    )
    assert all(
        record["risk"] == sources.risk_by_intent[record["intent"]]
        for record in output_records
    )
    assert artifacts.manifest_payload["counts"]["final_intent_counts"] == dict(
        sorted(Counter(record["intent"] for record in output_records).items())
    )


def test_primary_assignment_index_ignores_sensitivity_diagnostics(
    sources: builder.FrozenSources,
) -> None:
    changed = copy.deepcopy(sources.discovery_assignments)
    for record in changed["assignments"]:
        record["sensitivity_assignments"] = {
            "not-primary": {
                "canonical_cluster_id": "cluster-prohibited-diagnostic",
                "is_noise": False,
            }
        }

    rebuilt = builder.build_primary_assignment_index(
        changed,
        sources.corpus_by_discovery_id,
    )

    assert rebuilt == sources.primary_assignments
    assert all(
        set(record)
        == {
            "discovery_id",
            "normalized_text_sha256",
            "primary_canonical_cluster_id",
            "primary_is_noise",
        }
        for record in rebuilt.values()
    )


def test_primary_noise_remains_unsupported(
    output_records: list[dict[str, Any]],
) -> None:
    noise = records_for_source(output_records, builder.PRIMARY_NOISE)

    assert noise
    assert {record["intent"] for record in noise} == {
        builder.UNSUPPORTED_INTENT
    }


def test_accepted_new_intent_cluster_uses_frozen_resolution(
    sources: builder.FrozenSources,
    output_records: list[dict[str, Any]],
) -> None:
    accepted = next(
        record
        for record in sources.taxonomy_freeze["candidate_cluster_resolutions"]
        if record["resolution"] == "ACCEPT_NEW_INTENT"
    )
    output = next(
        record
        for record in output_records
        if record["v2c5_primary_canonical_cluster_id"]
        == accepted["canonical_cluster_id"]
    )

    assert output["intent"] == accepted["final_intent"]
    assert output["v2c5_relabel_source"] == builder.FROZEN_CLUSTER_RESOLUTION


def test_existing_intent_merge_uses_frozen_resolution(
    sources: builder.FrozenSources,
    output_records: list[dict[str, Any]],
) -> None:
    merged = next(
        record
        for record in sources.taxonomy_freeze["candidate_cluster_resolutions"]
        if record["resolution"] == "MERGE_TO_EXISTING_INTENT"
    )
    output = next(
        record
        for record in output_records
        if record["v2c5_primary_canonical_cluster_id"]
        == merged["canonical_cluster_id"]
    )

    assert output["intent"] == merged["final_intent"]
    assert output["v2c5_relabel_source"] == builder.FROZEN_CLUSTER_RESOLUTION


def test_remain_unsupported_cluster_uses_frozen_resolution(
    sources: builder.FrozenSources,
    output_records: list[dict[str, Any]],
) -> None:
    unsupported = next(
        record
        for record in sources.taxonomy_freeze["carry_forward_cluster_resolutions"]
        if record["resolution"] == "REMAIN_UNSUPPORTED"
    )
    output = next(
        record
        for record in output_records
        if record["v2c5_primary_canonical_cluster_id"]
        == unsupported["canonical_cluster_id"]
    )

    assert output["intent"] == builder.UNSUPPORTED_INTENT
    assert output["v2c5_relabel_source"] == builder.FROZEN_CLUSTER_RESOLUTION


def test_split_cluster_uses_step21a_adjudication(
    sources: builder.FrozenSources,
    output_records: list[dict[str, Any]],
) -> None:
    output = records_for_source(output_records, builder.SPLIT_ADJUDICATION)[0]
    decision = sources.split_decisions[output["v2c5_discovery_id"]]

    assert output["intent"] == decision["final_intent"]
    assert output["v2c5_split_adjudication_confidence"] == decision["confidence"]


def test_step21a_has_complete_975_record_coverage(
    sources: builder.FrozenSources,
) -> None:
    assert len(sources.split_decisions) == 975
    assert sum(
        decision["resolution_kind"] == "HUMAN"
        for decision in sources.split_decisions.values()
    ) == 578
    assert sum(
        decision["resolution_kind"] == "DETERMINISTIC"
        for decision in sources.split_decisions.values()
    ) == 397


def test_missing_discovery_join_fails(
    sources: builder.FrozenSources,
) -> None:
    example = first_historical_unsupported(sources)
    normalized_hash = example["normalized_text_sha256"]
    corpus_by_hash = dict(sources.corpus_by_normalized_hash)
    corpus_by_hash.pop(normalized_hash)
    changed = replace(sources, corpus_by_normalized_hash=corpus_by_hash)

    with pytest.raises(ValueError, match="no Step 17 discovery"):
        builder.resolve_unsupported_example(example, changed)


def test_duplicate_primary_assignment_fails(
    sources: builder.FrozenSources,
) -> None:
    changed = copy.deepcopy(sources.discovery_assignments)
    changed["assignments"][-1] = copy.deepcopy(changed["assignments"][0])

    with pytest.raises(ValueError, match="duplicate primary assignment"):
        builder.build_primary_assignment_index(
            changed,
            sources.corpus_by_discovery_id,
        )


def test_missing_primary_assignment_fails(
    sources: builder.FrozenSources,
) -> None:
    example = first_historical_unsupported(sources)
    discovery_id = f"v2c5-discovery:{example['normalized_text_sha256']}"
    assignments = dict(sources.primary_assignments)
    assignments.pop(discovery_id)
    changed = replace(sources, primary_assignments=assignments)

    with pytest.raises(ValueError, match="no unique primary assignment"):
        builder.resolve_unsupported_example(example, changed)


def test_duplicate_step20_cluster_resolution_fails(
    sources: builder.FrozenSources,
) -> None:
    changed = copy.deepcopy(sources.taxonomy_freeze)
    changed["candidate_cluster_resolutions"][1]["canonical_cluster_id"] = changed[
        "candidate_cluster_resolutions"
    ][0]["canonical_cluster_id"]

    with pytest.raises(ValueError, match="duplicate or invalid frozen"):
        builder.build_cluster_resolution_indexes(changed)


def test_step20_resolution_outside_taxonomy_fails(
    sources: builder.FrozenSources,
) -> None:
    changed = copy.deepcopy(sources.taxonomy_freeze)
    changed["candidate_cluster_resolutions"][0]["final_intent"] = "invented_intent"

    with pytest.raises(ValueError, match="outside the taxonomy"):
        builder.build_cluster_resolution_indexes(changed)


def test_wrong_output_risk_fails(
    sources: builder.FrozenSources,
    output_records: list[dict[str, Any]],
) -> None:
    changed = copy.deepcopy(output_records)
    changed[0]["risk"] = "WRONG_RISK"

    with pytest.raises(ValueError, match="risk disagrees"):
        builder.validate_expanded_examples(changed, sources)


def test_protected_write_set_is_exact(
    sources: builder.FrozenSources,
) -> None:
    assert tuple(sources.taxonomy_freeze["protected_write_intents"]) == (
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    )
    changed = copy.deepcopy(sources.taxonomy_freeze)
    changed["protected_write_intents"].append("transfer_pending")

    with pytest.raises(ValueError, match="protected-write intent set"):
        builder.build_cluster_resolution_indexes(changed)


@pytest.mark.parametrize(
    ("source_id", "source_split"),
    [
        ("cfpb", "train"),
        ("banking77", "test"),
        ("clinc150_oos", "test"),
        ("sentinelvoice_v2c1_internal", "final"),
        ("sentinelvoice_v2c1_internal", "holdout"),
    ],
)
def test_prohibited_development_sources_are_rejected(
    sources: builder.FrozenSources,
    source_id: str,
    source_split: str,
) -> None:
    changed = copy.deepcopy(sources.development_dataset)
    changed["examples"][0]["source_id"] = source_id
    changed["examples"][0]["source_split"] = source_split

    with pytest.raises(ValueError, match="prohibited development source"):
        builder.validate_development_dataset(changed)


@pytest.mark.parametrize(
    "path",
    [
        Path("data/evals/v2/ml/v2c4_safety_holdout.json"),
        Path("data/evals/v2/ml/v2c4_selection_probe.json"),
        Path("data/evals/v2/ml/v2c3_external_final_lockbox.json"),
        Path("data/evals/v2/ml/cfpb.json"),
        Path("data/evals/v2/ml/banking77_test.json"),
        Path("data/evals/v2/ml/clinc_test.json"),
    ],
)
def test_prohibited_input_paths_are_rejected(path: Path) -> None:
    with pytest.raises(ValueError):
        builder.guard_source_path(path)


def test_native_external_label_is_metadata_only(
    sources: builder.FrozenSources,
) -> None:
    example = copy.deepcopy(first_historical_unsupported(sources))
    original = builder.resolve_unsupported_example(example, sources)
    example["source_label"] = "freeze_card"

    assert builder.resolve_unsupported_example(example, sources) == original


def test_output_bytes_are_deterministic(
    sources: builder.FrozenSources,
) -> None:
    first = builder.build_artifacts(sources=sources)
    second = builder.build_artifacts(sources=sources)

    assert first.dataset_bytes == second.dataset_bytes
    assert first.manifest_bytes == second.manifest_bytes
    assert first.dataset_bytes.endswith(b"\n")
    assert first.manifest_bytes.endswith(b"\n")


def test_check_writes_nothing(
    sources: builder.FrozenSources,
    tmp_path: Path,
) -> None:
    paths = replace(
        builder.DEFAULT_PATHS,
        dataset_output=tmp_path / builder.DEFAULT_PATHS.dataset_output.name,
        manifest_output=tmp_path / builder.DEFAULT_PATHS.manifest_output.name,
    )
    artifacts = builder.build_artifacts(paths, sources=sources)

    builder.check_artifacts(artifacts, paths)

    assert not paths.dataset_output.exists()
    assert not paths.manifest_output.exists()


def test_builder_uses_only_python_standard_library() -> None:
    script_path = Path(builder.__file__).resolve()
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
        "unicodedata",
    }
    for prohibited in (
        "fastembed",
        "sklearn",
        "numpy",
        "pandas",
        "fit_predict(",
        "predict_proba(",
        "sensitivity_assignments][",
    ):
        assert prohibited not in source
