import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.evaluation.v2_contracts import IntentRiskDataset
from scripts.build_v2_intent_dataset import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SEED_PATH,
    build_dataset,
    build_files,
    normalize_text,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_seed_examples_are_preserved_unchanged() -> None:
    seed = _load(DEFAULT_SEED_PATH)
    expanded = _load(DEFAULT_DATASET_PATH)
    expanded_by_id = {
        example["example_id"]: example for example in expanded["examples"]
    }

    assert len(seed["examples"]) == 54
    assert all(
        expanded_by_id[example["example_id"]] == example
        for example in seed["examples"]
    )


def test_expansion_is_deterministic_and_matches_checked_in_files(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dataset, first_manifest = build_files(output_dir=first_dir)
    second_dataset, second_manifest = build_files(output_dir=second_dir)

    assert first_dataset.read_bytes() == second_dataset.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first_dataset.read_bytes() == DEFAULT_DATASET_PATH.read_bytes()
    assert first_manifest.read_bytes() == DEFAULT_MANIFEST_PATH.read_bytes()


def test_dataset_manifest_hashes_match_frozen_inputs() -> None:
    manifest = _load(DEFAULT_MANIFEST_PATH)

    assert manifest["hashes"] == {
        "config_sha256": _sha256(DEFAULT_CONFIG_PATH),
        "dataset_sha256": _sha256(DEFAULT_DATASET_PATH),
        "v2_a_seed_sha256": _sha256(DEFAULT_SEED_PATH),
    }
    assert manifest["external_dataset_sources"] == []


def test_all_labels_and_minimum_split_coverage_are_present() -> None:
    dataset = _load(DEFAULT_DATASET_PATH)
    counts = Counter(
        (example["intent"], example["split"])
        for example in dataset["examples"]
    )

    assert len({example["intent"] for example in dataset["examples"]}) == 9
    assert len({example["risk"] for example in dataset["examples"]}) == 4
    for intent in {example["intent"] for example in dataset["examples"]}:
        assert counts[(intent, "train")] >= 45
        assert counts[(intent, "validation")] >= 12
        assert counts[(intent, "locked_test")] >= 10


def test_ids_text_and_groups_are_unique_and_split_safe() -> None:
    dataset = _load(DEFAULT_DATASET_PATH)
    ids = [example["example_id"] for example in dataset["examples"]]
    normalized = [normalize_text(example["text"]) for example in dataset["examples"]]
    splits_by_group: dict[str, set[str]] = {}
    for example in dataset["examples"]:
        splits_by_group.setdefault(example["group_id"], set()).add(
            example["split"]
        )

    assert len(ids) == len(set(ids))
    assert len(normalized) == len(set(normalized))
    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_original_locked_test_examples_remain_unchanged() -> None:
    seed_locked = {
        example["example_id"]: example
        for example in _load(DEFAULT_SEED_PATH)["examples"]
        if example["split"] == "locked_test"
    }
    expanded_locked = {
        example["example_id"]: example
        for example in _load(DEFAULT_DATASET_PATH)["examples"]
        if example["split"] == "locked_test"
    }

    assert len(seed_locked) == 18
    assert all(expanded_locked[key] == value for key, value in seed_locked.items())


def test_manifest_freezes_expanded_locked_test_before_selection() -> None:
    manifest = _load(DEFAULT_MANIFEST_PATH)

    assert manifest["counts"]["locked_test_examples"] == 90
    assert manifest["counts"]["locked_test_groups"] == 81
    assert manifest["counts"]["by_split"] == {
        "locked_test": 90,
        "train": 405,
        "validation": 108,
    }
    assert "never used" in manifest["locked_test_policy"]


def test_normalized_duplicate_text_is_rejected() -> None:
    dataset = _load(DEFAULT_DATASET_PATH)
    payload = deepcopy(dataset)
    payload["examples"][1]["text"] = (
        payload["examples"][0]["text"].upper() + "!!!"
    )

    with pytest.raises(ValidationError, match="normalized example text"):
        IntentRiskDataset.model_validate(payload)


def test_blank_text_is_rejected() -> None:
    seed = _load(DEFAULT_SEED_PATH)
    config = _load(DEFAULT_CONFIG_PATH)
    seed["examples"][0]["text"] = "   "

    with pytest.raises(ValidationError, match="text cannot be blank"):
        build_dataset(seed_payload=seed, config=config)
