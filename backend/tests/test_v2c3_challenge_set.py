import ast
import copy
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import build_v2c3_challenge_set as builder


@pytest.fixture(scope="module")
def artifacts() -> builder.BuiltArtifacts:
    return builder.build_artifacts()


def test_deterministic_output_matches_tracked_artifacts(
    artifacts: builder.BuiltArtifacts,
) -> None:
    repeated = builder.build_artifacts()

    assert artifacts.challenge_bytes == repeated.challenge_bytes
    assert artifacts.manifest_bytes == repeated.manifest_bytes
    assert artifacts.challenge_bytes == (
        builder.DEFAULT_PATHS.challenge_output.read_bytes()
    )
    assert artifacts.manifest_bytes == (
        builder.DEFAULT_PATHS.manifest_output.read_bytes()
    )


def test_exactly_thirty_examples_per_frozen_intent(
    artifacts: builder.BuiltArtifacts,
) -> None:
    contract = json.loads(builder.DEFAULT_PATHS.contract.read_text(encoding="utf-8"))
    examples = artifacts.challenge_payload["examples"]
    counts = Counter(row["intent"] for row in examples)

    assert len(examples) == 270
    assert set(counts) == set(contract["intent_taxonomy"])
    assert set(counts.values()) == {30}
    assert artifacts.manifest_payload["counts_by_intent"] == dict(
        sorted(counts.items())
    )


def test_risk_is_derived_from_frozen_contract(
    artifacts: builder.BuiltArtifacts,
) -> None:
    contract = json.loads(builder.DEFAULT_PATHS.contract.read_text(encoding="utf-8"))

    assert all(
        row["risk"] == contract["risk_by_intent"][row["intent"]]
        for row in artifacts.challenge_payload["examples"]
    )
    assert artifacts.manifest_payload["validation"][
        "risk_derived_from_frozen_contract"
    ] is True


def test_normalized_text_is_unique_and_disjoint_from_development(
    artifacts: builder.BuiltArtifacts,
) -> None:
    examples = artifacts.challenge_payload["examples"]
    challenge_hashes = {row["normalized_text_sha256"] for row in examples}
    development = json.loads(
        builder.DEFAULT_PATHS.development_dataset.read_text(encoding="utf-8")
    )
    development_hashes = {
        row["normalized_text_sha256"] for row in development["examples"]
    }

    assert len(challenge_hashes) == len(examples)
    assert challenge_hashes.isdisjoint(development_hashes)
    assert artifacts.manifest_payload["validation"][
        "development_normalized_hash_overlap"
    ] == 0


def test_duplicate_normalized_text_is_rejected() -> None:
    seed = json.loads(builder.DEFAULT_PATHS.seed.read_text(encoding="utf-8"))
    contract = json.loads(builder.DEFAULT_PATHS.contract.read_text(encoding="utf-8"))
    duplicate_seed = copy.deepcopy(seed)
    duplicate_seed["examples_by_intent"]["account_balance"][1]["text"] = (
        duplicate_seed["examples_by_intent"]["account_balance"][0]["text"]
    )

    with pytest.raises(ValueError, match="normalized-text duplicates"):
        builder._build_records(
            duplicate_seed,
            contract["intent_taxonomy"],
            contract["risk_by_intent"],
        )


def test_development_overlap_is_rejected(tmp_path: Path) -> None:
    seed = json.loads(builder.DEFAULT_PATHS.seed.read_text(encoding="utf-8"))
    development = json.loads(
        builder.DEFAULT_PATHS.development_dataset.read_text(encoding="utf-8")
    )
    overlapping_seed = copy.deepcopy(seed)
    overlapping_seed["examples_by_intent"]["account_balance"][0]["text"] = (
        development["examples"][0]["text"]
    )
    seed_path = tmp_path / "development-overlap-seed.json"
    seed_path.write_text(json.dumps(overlapping_seed), encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps frozen development/evaluation"):
        builder.build_artifacts(replace(builder.DEFAULT_PATHS, seed=seed_path))


def test_internal_locked_test_overlap_is_rejected(tmp_path: Path) -> None:
    seed = json.loads(builder.DEFAULT_PATHS.seed.read_text(encoding="utf-8"))
    internal = json.loads(
        builder.DEFAULT_PATHS.internal_dataset.read_text(encoding="utf-8")
    )
    locked_text = next(
        row["text"] for row in internal["examples"] if row["split"] == "locked_test"
    )
    overlapping_seed = copy.deepcopy(seed)
    overlapping_seed["examples_by_intent"]["account_balance"][0]["text"] = (
        locked_text
    )
    seed_path = tmp_path / "locked-test-overlap-seed.json"
    seed_path.write_text(json.dumps(overlapping_seed), encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps frozen development/evaluation"):
        builder.build_artifacts(replace(builder.DEFAULT_PATHS, seed=seed_path))


def test_disjoint_from_all_frozen_evaluation_references(
    artifacts: builder.BuiltArtifacts,
) -> None:
    validation = artifacts.manifest_payload["validation"]

    assert validation["external_lockbox_normalized_hash_overlap"] == 0
    assert validation["internal_locked_test_normalized_hash_overlap"] == 0
    assert validation["banking77_test_normalized_hash_overlap"] == 0
    assert validation["clinc_test_and_oos_test_normalized_hash_overlap"] == 0
    assert validation["cfpb_used"] is False


def test_every_record_and_dataset_are_final_evaluation_only(
    artifacts: builder.BuiltArtifacts,
) -> None:
    payload = artifacts.challenge_payload
    policy = artifacts.manifest_payload["evaluation_policy"]
    examples = payload["examples"]

    assert payload["data_role"] == builder.DATA_ROLE
    assert policy["data_role"] == builder.DATA_ROLE
    for field in (
        "training_eligible",
        "model_selection_eligible",
        "threshold_selection_eligible",
    ):
        assert payload[field] is False
        assert policy[field] is False
        assert all(row[field] is False for row in examples)
    assert payload["training_or_evaluation_performed"] is False
    assert artifacts.manifest_payload["training_or_evaluation_performed"] is False
    assert {row["source_id"] for row in examples} == {builder.SOURCE_ID}
    assert artifacts.manifest_payload["source_policy"]["cfpb_used"] is False
    assert artifacts.manifest_payload["source_policy"][
        "consumer_authored_text_used"
    ] is False


def test_boundary_pairs_share_lineage_but_contrast_intents(
    artifacts: builder.BuiltArtifacts,
) -> None:
    families: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in artifacts.challenge_payload["examples"]:
        if "boundary_pair" in row["tags"]:
            families[row["lineage_id"]].append(row)

    assert len(families) == 8
    assert all(len(rows) == 2 for rows in families.values())
    assert all(len({row["intent"] for row in rows}) == 2 for rows in families.values())
    assert all(
        len({row["group_id"] for row in rows}) == 1
        for rows in families.values()
    )


def test_protected_write_positives_require_explicit_current_action_tags(
    artifacts: builder.BuiltArtifacts,
) -> None:
    protected = [
        row
        for row in artifacts.challenge_payload["examples"]
        if row["intent"] in builder.PROTECTED_WRITE_INTENTS
    ]

    assert len(protected) == 60
    assert all("explicit_current_action" in row["tags"] for row in protected)
    assert all("protected_write_positive" in row["tags"] for row in protected)
    assert not any(
        "protected_write_positive" in row["tags"]
        for row in artifacts.challenge_payload["examples"]
        if row["intent"] not in builder.PROTECTED_WRITE_INTENTS
    )


def test_required_protected_action_hard_negatives_are_present(
    artifacts: builder.BuiltArtifacts,
) -> None:
    unsupported = [
        row
        for row in artifacts.challenge_payload["examples"]
        if row["intent"] == "unsupported_or_uncertain"
    ]
    tags = {tag for row in unsupported for tag in row["tags"]}

    assert builder.REQUIRED_HARD_NEGATIVE_TAGS <= tags
    assert all("protected_write_positive" not in row["tags"] for row in unsupported)


def test_manifest_hashes_reconcile(artifacts: builder.BuiltArtifacts) -> None:
    manifest = artifacts.manifest_payload
    for name, display_path in manifest["input_paths"].items():
        path = builder.REPOSITORY_ROOT / display_path
        assert builder.development_builder.sha256_bytes(path.read_bytes()) == (
            manifest["input_hashes"][name]
        )
    assert builder.development_builder.sha256_bytes(artifacts.challenge_bytes) == (
        manifest["output_hashes"]["v2c3_challenge_set.json"]
    )


def test_check_detects_drift(
    artifacts: builder.BuiltArtifacts,
    tmp_path: Path,
) -> None:
    paths = replace(
        builder.DEFAULT_PATHS,
        challenge_output=tmp_path / "challenge.json",
        manifest_output=tmp_path / "manifest.json",
    )
    builder.write_artifacts(artifacts, paths)
    builder.check_artifacts(artifacts, paths)

    paths.challenge_output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from deterministic build"):
        builder.check_artifacts(artifacts, paths)


def test_builder_has_no_model_training_evaluation_or_network_imports() -> None:
    tree = ast.parse(Path(builder.__file__).read_text(encoding="utf-8"))
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
            "httpx",
            "joblib",
            "numpy",
            "openai",
            "pandas",
            "requests",
            "sentence_transformers",
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
            "transform",
        }
    )
