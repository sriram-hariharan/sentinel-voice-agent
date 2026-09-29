from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import build_v2c5_discovery_corpus as builder

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "data/evals/v2/ml/v2c5_intent_discovery_contract.json"
)
SCRIPT_PATH = ROOT / "scripts/build_v2c5_discovery_corpus.py"


@pytest.fixture(scope="module")
def built_artifacts() -> builder.BuiltArtifacts:
    return builder.build_artifacts()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_row(
    *,
    example_id: str,
    text: str,
    intent: str = builder.PRIMARY_INTENT,
    source_id: str = "sentinelvoice_v2c1_internal",
    source_split: str = "train",
    source_row_index: int = 1,
    data_role: str = "development",
) -> dict[str, Any]:
    external = source_id != "sentinelvoice_v2c1_internal"
    mapping_status = "UNSUPPORTED" if external else "INTERNAL_FROZEN"
    return {
        "data_role": data_role,
        "example_id": example_id,
        "group_id": f"group:{example_id}",
        "intent": intent,
        "mapping_status": mapping_status,
        "normalized_text_sha256": (
            builder.development_builder.normalized_text_sha256(text)
        ),
        "original_example_id": example_id if not external else None,
        "original_split": source_split if not external else None,
        "risk": (
            "ESCALATION_OR_UNCERTAIN"
            if intent == builder.PRIMARY_INTENT
            else "PRIVATE_READ"
        ),
        "source_domain": "sentinelvoice" if not external else "banking",
        "source_id": source_id,
        "source_label": intent if not external else "native_external_label",
        "source_revision": "fixture",
        "source_row_index": source_row_index,
        "source_split": source_split,
        "text": text,
        "text_sha256": builder.development_builder.text_sha256(text),
    }


def test_contract_is_frozen_and_hash_pinned() -> None:
    contract_bytes = CONTRACT_PATH.read_bytes()
    contract = builder.load_json_object_bytes(
        contract_bytes,
        "V2-C5 contract",
    )

    assert sha256_file(CONTRACT_PATH) == builder.CONTRACT_SHA256
    assert contract["phase"] == "V2-C5"
    assert contract["status"] == "frozen"
    assert contract["execution_status"]["contract_frozen"] is True
    builder.validate_contract(contract, contract_bytes)


def test_builder_cannot_write_the_frozen_contract() -> None:
    assert builder.DEFAULT_PATHS.contract == CONTRACT_PATH
    output_paths = {
        builder.DEFAULT_PATHS.corpus_output,
        builder.DEFAULT_PATHS.manifest_output,
    }
    assert builder.DEFAULT_PATHS.contract not in output_paths

    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"write_text", "write_bytes"}
    ]
    assert writes == []


def test_exact_eligible_source_roles_are_frozen() -> None:
    assert builder.EXPECTED_ELIGIBLE_SOURCE_ROLES == [
        {
            "development_assignment_required": True,
            "materialized_path": (
                "data/evals/v2/ml/v2c3_development_dataset.json"
            ),
            "record_id": "internal_train",
            "registry_role": "v2c3_development",
            "source_id": "sentinelvoice_v2c1_internal",
            "source_split": "train",
        },
        {
            "development_assignment_required": True,
            "materialized_path": (
                "data/evals/v2/ml/v2c3_development_dataset.json"
            ),
            "record_id": "internal_validation",
            "registry_role": "v2c3_development",
            "source_id": "sentinelvoice_v2c1_internal",
            "source_split": "validation",
        },
        {
            "development_assignment_required": True,
            "materialized_path": (
                "data/evals/v2/ml/v2c3_development_dataset.json"
            ),
            "record_id": "banking77_train",
            "registry_role": (
                "v2c3_conditional_development_and_fresh_lockbox_source"
            ),
            "source_id": "banking77",
            "source_split": "train",
        },
        {
            "development_assignment_required": True,
            "materialized_path": (
                "data/evals/v2/ml/v2c3_development_dataset.json"
            ),
            "record_id": "clinc_finance_train",
            "registry_role": (
                "v2c3_conditional_development_and_fresh_lockbox_source"
            ),
            "source_id": "clinc150_oos",
            "source_split": "train",
        },
        {
            "development_assignment_required": True,
            "materialized_path": (
                "data/evals/v2/ml/v2c3_development_dataset.json"
            ),
            "record_id": "clinc_finance_val",
            "registry_role": (
                "v2c3_conditional_development_and_fresh_lockbox_source"
            ),
            "source_id": "clinc150_oos",
            "source_split": "val",
        },
    ]


def test_primary_population_contains_only_unsupported_examples() -> None:
    rows = [
        source_row(example_id="unsupported", text="Please help somehow."),
        source_row(
            example_id="supported",
            text="What is my balance?",
            intent="account_balance",
        ),
    ]

    primary = builder.build_primary_population(rows)

    assert len(primary) == 1
    assert primary[0]["current_sentinelvoice_intent"] == (
        "unsupported_or_uncertain"
    )
    assert primary[0]["text"] == "Please help somehow."
    assert primary[0]["current_sentinelvoice_intent_clustering_feature"] is False


@pytest.mark.parametrize(
    ("source_id", "source_split", "data_role"),
    [
        ("sentinelvoice_v2c1_internal", "locked_test", "development"),
        ("banking77", "test", "development"),
        ("clinc150_oos", "test", "development"),
        ("cfpb", "frozen_semantic_holdout_1800", "development"),
        ("v2c3_challenge", "challenge", "development"),
        ("v2c4_selection_probe", "probe", "development"),
        ("v2c4_targeted_augmentation", "augmentation", "development"),
        ("banking77", "train", "fresh_lockbox"),
        ("banking77", "train", "final_evaluation"),
    ],
)
def test_ineligible_and_final_or_test_roles_fail_closed(
    source_id: str,
    source_split: str,
    data_role: str,
) -> None:
    row = source_row(
        example_id="prohibited",
        text="This source is prohibited.",
        source_id=source_id,
        source_split=source_split,
        data_role=data_role,
    )

    with pytest.raises(ValueError):
        builder.validate_source_record(row, builder.source_role_by_key())


def test_sealed_v2c4_holdout_paths_are_rejected_without_access() -> None:
    for filename in builder.FORBIDDEN_INPUT_FILENAMES:
        with pytest.raises(
            ValueError,
            match="sealed V2-C4 holdout input is forbidden",
        ):
            builder.guard_input_path(ROOT / "data/evals/v2/ml" / filename)


def test_exact_normalized_duplicates_collapse_and_preserve_occurrences() -> None:
    internal = source_row(
        example_id="internal",
        text="  Need HELP with this  ",
        source_row_index=4,
    )
    banking = source_row(
        example_id="banking",
        text="need help with this",
        source_id="banking77",
        source_split="train",
        source_row_index=2,
    )

    primary = builder.build_primary_population([banking, internal])

    assert len(primary) == 1
    record = primary[0]
    assert record["source_occurrence_count"] == 2
    assert record["text"] == internal["text"]
    assert record["all_source_datasets"] == [
        "banking77",
        "sentinelvoice_v2c1_internal",
    ]
    assert [item["example_id"] for item in record["occurrences"]] == [
        "internal",
        "banking",
    ]
    assert record["discovery_id"] == (
        "v2c5-discovery:" + internal["normalized_text_sha256"]
    )


def test_canonical_representative_is_input_order_independent() -> None:
    rows = [
        source_row(
            example_id="clinc",
            text="No CURRENT request",
            source_id="clinc150_oos",
            source_split="val",
            source_row_index=8,
        ),
        source_row(
            example_id="banking",
            text="no current   request",
            source_id="banking77",
            source_split="train",
            source_row_index=3,
        ),
        source_row(
            example_id="internal",
            text="no current request",
            source_row_index=9,
        ),
    ]

    forward = builder.build_primary_population(rows)
    reverse = builder.build_primary_population(list(reversed(rows)))

    assert builder.development_builder.stable_json_bytes(forward) == (
        builder.development_builder.stable_json_bytes(reverse)
    )
    assert forward[0]["text"] == "no current request"
    assert forward[0]["occurrences"][0]["source_dataset"] == (
        "sentinelvoice_v2c1_internal"
    )


def test_native_and_current_labels_are_metadata_only() -> None:
    row = source_row(
        example_id="external",
        text="An unsupported external example.",
        source_id="banking77",
        source_split="train",
    )

    record = builder.build_primary_population([row])[0]
    occurrence = record["occurrences"][0]

    assert record["current_sentinelvoice_intent_clustering_feature"] is False
    assert occurrence["native_external_label"] == "native_external_label"
    assert occurrence["native_external_label_clustering_feature"] is False
    corpus = builder.build_corpus_payload(
        [record],
        builder.reference_anchor_summary([]),
    )
    assert corpus["embedding_input_policy"]["fields"] == ["text"]
    assert corpus["embedding_input_policy"]["labels_concatenated_with_text"] is False
    assert (
        corpus["embedding_input_policy"][
            "source_metadata_concatenated_with_text"
        ]
        is False
    )


def test_provenance_uses_null_or_empty_values_for_absent_metadata() -> None:
    row = source_row(
        example_id="external",
        text="Metadata fixture.",
        source_id="clinc150_oos",
        source_split="train",
    )

    occurrence = builder.build_primary_population([row])[0]["occurrences"][0]

    assert occurrence["lineage_id"] is None
    assert occurrence["original_example_id"] is None
    assert occurrence["tags"] == []
    assert occurrence["design_metadata"] == {}
    assert occurrence["group_id"] == "group:external"
    assert occurrence["source_role"] == (
        "v2c3_conditional_development_and_fresh_lockbox_source"
    )


def test_actual_corpus_and_manifest_are_deterministic(
    built_artifacts: builder.BuiltArtifacts,
) -> None:
    contract_before = CONTRACT_PATH.read_bytes()
    rebuilt = builder.build_artifacts()

    assert built_artifacts.corpus_bytes == rebuilt.corpus_bytes
    assert built_artifacts.manifest_bytes == rebuilt.manifest_bytes
    assert CONTRACT_PATH.read_bytes() == contract_before
    assert built_artifacts.manifest_payload["corpus"]["sha256"] == (
        hashlib.sha256(built_artifacts.corpus_bytes).hexdigest()
    )


def test_actual_primary_ids_and_normalized_texts_are_unique(
    built_artifacts: builder.BuiltArtifacts,
) -> None:
    primary = built_artifacts.corpus_payload["primary_cluster_population"]
    discovery_ids = [record["discovery_id"] for record in primary]
    normalized_hashes = [
        record["normalized_text_sha256"] for record in primary
    ]

    assert len(discovery_ids) == len(set(discovery_ids))
    assert len(normalized_hashes) == len(set(normalized_hashes))
    assert {
        record["current_sentinelvoice_intent"] for record in primary
    } == {"unsupported_or_uncertain"}
    assert all(record["source_occurrence_count"] >= 1 for record in primary)


def test_reference_anchors_are_hash_pinned_references_only(
    built_artifacts: builder.BuiltArtifacts,
) -> None:
    anchors = built_artifacts.corpus_payload["reference_anchor_population"]

    assert anchors["data_role"] == "v2c5_reference_anchor_population"
    assert anchors["density_driving_cluster_members"] is False
    assert anchors["records_copied_into_corpus"] is False
    assert anchors["materialization"] == "hash_pinned_source_reference_only"
    assert anchors["source_dataset_sha256"] == (
        builder.DEVELOPMENT_DATASET_SHA256
    )
    assert "records" not in anchors


def test_manifest_records_counts_hashes_and_integrity(
    built_artifacts: builder.BuiltArtifacts,
) -> None:
    manifest = built_artifacts.manifest_payload
    counts = manifest["counts"]
    checks = manifest["exclusion_and_integrity_checks"]

    assert counts["raw_eligible_occurrence_count"] == 8198
    assert counts["primary_unsupported_occurrence_count_before_dedup"] == 6373
    assert counts["primary_unique_normalized_text_count"] <= 6373
    assert counts["duplicate_occurrence_count_removed_from_density"] == (
        counts["primary_unsupported_occurrence_count_before_dedup"]
        - counts["primary_unique_normalized_text_count"]
    )
    assert checks["supported_intents_in_primary_population"] == 0
    assert checks["duplicate_normalized_text_vector_records"] == 0
    assert checks["discovery_id_collisions"] == 0
    assert checks["sealed_v2c4_holdout_accessed"] is False
    assert checks["native_external_labels_are_clustering_features"] is False
    assert checks["current_intent_labels_are_clustering_features"] is False


def test_execution_status_proves_no_ml_or_taxonomy_work(
    built_artifacts: builder.BuiltArtifacts,
) -> None:
    expected = {
        "classifier_training_performed": False,
        "clustering_performed": False,
        "discovery_corpus_built": True,
        "embeddings_generated": False,
        "human_adjudication_performed": False,
        "taxonomy_changed": False,
        "umap_performed": False,
        "v2c5_holdout_created": False,
    }

    assert built_artifacts.corpus_payload["execution_status"] == expected
    assert built_artifacts.manifest_payload["execution_status"] == expected
    runtime = built_artifacts.corpus_payload["runtime_scope"]
    assert runtime["new_intents_created"] is False
    assert runtime["new_runtime_tools_added"] is False
    assert runtime["new_protected_actions_inferred"] is False


def test_write_and_check_are_byte_exact_in_temporary_outputs(
    built_artifacts: builder.BuiltArtifacts,
    tmp_path: Path,
) -> None:
    paths = replace(
        builder.DEFAULT_PATHS,
        corpus_output=tmp_path / "corpus.json",
        manifest_output=tmp_path / "manifest.json",
    )

    builder.write_artifacts(built_artifacts, paths)
    builder.check_artifacts(built_artifacts, paths)
    paths.corpus_output.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="differs from deterministic build"):
        builder.check_artifacts(built_artifacts, paths)


def test_builder_contains_no_ml_embedding_or_clustering_imports_or_calls() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden_imports = {
        "fastembed",
        "hdbscan",
        "umap",
        "sklearn",
    }
    imported: set[str] = set()
    called_attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.add(node.func.attr)

    assert not (imported & forbidden_imports)
    assert not (
        called_attributes
        & {
            "embed",
            "fit",
            "fit_predict",
            "fit_transform",
            "predict",
        }
    )
