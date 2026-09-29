from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import build_v2c3_development_dataset as development_builder
from scripts import run_v2c3_final_evaluation as final_evaluation
from scripts import run_v2c3_model_tournament as tournament

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data/evals/v2/ml/v2c3_final_evaluation_config.json"
RUNNER_PATH = ROOT / "scripts/run_v2c3_final_evaluation.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


@pytest.fixture
def step_4_config() -> dict[str, Any]:
    return load_json(ROOT / "data/evals/v2/ml/v2c3_tournament_config.json")


def test_exact_selected_final_model_and_tuning_winner(
    config: dict[str, Any],
) -> None:
    final_evaluation.validate_config(config)
    selected = config["selected_model"]
    assert selected["selected_tuning_variant"] == "bge_svc_c_4_0"
    assert selected["representation"]["model_identifier"] == (
        "BAAI/bge-small-en-v1.5"
    )
    assert selected["representation"]["dimensions"] == 384
    assert selected["representation"]["fine_tuning"] is False
    assert selected["classifier"]["class"] == "LinearSVC"
    assert selected["classifier"]["parameters"]["C"] == 4.0
    assert selected["classifier"]["parameters"]["class_weight"] == "balanced"

    tuning_report = load_json(
        ROOT / config["frozen_input_hashes"]["tuning_report"]["path"]
    )
    assert tuning_report["overall_tuned_winner"] == "bge_svc_c_4_0"
    assert tuning_report["no_eligible_tuned_winner"] is False


def test_all_frozen_hashes_match_tracked_artifacts(config: dict[str, Any]) -> None:
    for spec in config["frozen_input_hashes"].values():
        assert tournament.sha256_file(ROOT / spec["path"]) == spec["sha256"]
    lockbox = config["external_lockbox"]
    assert tournament.sha256_file(ROOT / lockbox["manifest_path"]) == lockbox[
        "manifest_sha256"
    ]
    for spec in lockbox["source_inputs"].values():
        assert tournament.sha256_file(ROOT / spec["path"]) == spec["sha256"]
    challenge = config["challenge_set"]
    assert tournament.sha256_file(ROOT / challenge["path"]) == challenge["sha256"]
    assert tournament.sha256_file(ROOT / challenge["manifest_path"]) == challenge[
        "manifest_sha256"
    ]


def test_final_fit_is_development_only_and_uses_all_8198(
    config: dict[str, Any],
) -> None:
    training = config["selected_model"]["training"]
    assert training == {
        "dataset_path": "data/evals/v2/ml/v2c3_development_dataset.json",
        "expected_example_count": 8198,
        "use_all_development_examples": True,
        "cross_validation_at_final_fit": False,
        "development_embedding_cache_path": (
            "data/evals/v2/ml/local/v2c3_bge_small_en_v1_5.npz"
        ),
        "development_embedding_cache_metadata_path": (
            "data/evals/v2/ml/local/v2c3_bge_small_en_v1_5.metadata.json"
        ),
        "automatic_embedding_regeneration": False,
        "final_evaluation_data_allowed": False,
    }
    source = inspect.getsource(final_evaluation.load_model_preparation_inputs)
    assert "load_development_data" in source
    assert "validate_embedding_cache" in source
    assert "load_challenge_examples" not in source
    assert "reconstruct_external_lockbox" not in source


def test_prepare_final_model_fits_only_supplied_development_arrays(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class RecordingLinearSVC:
        def __init__(self, **parameters: Any) -> None:
            calls["parameters"] = parameters

        def fit(self, features: np.ndarray, labels: np.ndarray) -> Any:
            calls["features"] = features.copy()
            calls["labels"] = labels.tolist()
            return self

    def fail_final_read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("final-only data was accessed during preparation")

    mutable_config = copy.deepcopy(config)
    mutable_config["selected_model"]["artifact"]["path"] = "final.joblib"
    mutable_config["selected_model"]["artifact"]["metadata_path"] = "meta.json"
    examples = [
        {"example_id": "development-1", "intent": "account_balance"},
        {"example_id": "development-2", "intent": "informational_policy"},
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    inputs = {
        "examples": examples,
        "embeddings": embeddings,
        "step_4_config": {"intent_taxonomy": list(final_evaluation.EXPECTED_INTENTS)},
    }
    written_metadata: dict[str, Any] = {}

    monkeypatch.setattr(final_evaluation, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(final_evaluation, "LinearSVC", RecordingLinearSVC)
    monkeypatch.setattr(
        final_evaluation, "reconstruct_external_lockbox", fail_final_read
    )
    monkeypatch.setattr(final_evaluation, "load_challenge_examples", fail_final_read)
    monkeypatch.setattr(
        final_evaluation.joblib, "dump", lambda payload, path, **_: None
    )
    monkeypatch.setattr(
        final_evaluation.tournament,
        "sha256_file",
        lambda path: "0" * 64,
    )
    monkeypatch.setattr(
        final_evaluation.tournament,
        "ordered_ids_sha256",
        lambda rows: "1" * 64,
    )
    monkeypatch.setattr(
        final_evaluation.tournament,
        "write_json",
        lambda path, payload: written_metadata.update(payload),
    )

    final_evaluation.prepare_final_model(mutable_config, inputs)

    assert calls["parameters"] == config["selected_model"]["classifier"]["parameters"]
    assert np.array_equal(calls["features"], embeddings)
    assert calls["labels"] == ["account_balance", "informational_policy"]
    assert written_metadata["development_only_training"] is True
    assert written_metadata["final_evaluation_data_used_for_training"] is False
    assert written_metadata["final_evaluation_performed"] is False


def test_exact_final_dataset_composition(config: dict[str, Any]) -> None:
    lockbox = config["external_lockbox"]
    assert lockbox["expected_count"] == 1922
    assert lockbox["expected_source_counts"] == {
        "banking77": 1346,
        "clinc150_oos": 576,
    }
    assert sum(lockbox["expected_source_counts"].values()) == 1922

    challenge = config["challenge_set"]
    assert challenge["expected_count"] == 270
    assert tuple(challenge["expected_intent_counts"]) == (
        final_evaluation.EXPECTED_INTENTS
    )
    assert len(challenge["expected_intent_counts"]) == 9
    assert set(challenge["expected_intent_counts"].values()) == {30}


def test_exactly_two_models_and_baseline_cannot_become_selected(
    config: dict[str, Any],
) -> None:
    assert config["evaluated_models"] == [
        "final_v2c3",
        "frozen_v2c1_baseline",
    ]
    assert config["selected_model"]["id"] == "final_v2c3"
    assert config["comparator"]["role"] == "historical_comparison_only"
    assert config["comparator"]["eligible_to_replace_selected_model"] is False
    assert config["comparator"]["definition"]["classifier"]["parameters"][
        "C"
    ] == 0.5
    assert config["comparator"]["definition"]["classifier"]["parameters"][
        "class_weight"
    ] is None
    assert config["final_acceptance_policy"][
        "model_switching_after_final_results_allowed"
    ] is False


def _member(
    *,
    example_id: str,
    source_id: str,
    source_split: str,
    source_label: str,
    source_row_index: int,
    source_revision: str,
    text: str,
    intent: str,
    risk: str,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "source_id": source_id,
        "source_split": source_split,
        "source_label": source_label,
        "source_row_index": source_row_index,
        "source_revision": source_revision,
        "mapping_status": "EXACT_MATCH",
        "intent": intent,
        "risk": risk,
        "text_sha256": development_builder.text_sha256(text),
        "normalized_text_sha256": (
            development_builder.normalized_text_sha256(text)
        ),
    }


def test_external_lockbox_reconstruction_verifies_frozen_identity_and_hashes() -> None:
    members = [
        _member(
            example_id="banking77:bank-rev:train:000001",
            source_id="banking77",
            source_split="train",
            source_label="balance",
            source_row_index=1,
            source_revision="bank-rev",
            text="What is my balance?",
            intent="account_balance",
            risk="PRIVATE_READ",
        ),
        _member(
            example_id="clinc:clinc-rev:val:000001",
            source_id="clinc150_oos",
            source_split="val",
            source_label="policy",
            source_row_index=1,
            source_revision="clinc-rev",
            text="Explain the policy",
            intent="informational_policy",
            risk="PUBLIC",
        ),
    ]
    rows = final_evaluation.reconstruct_lockbox_members(
        members,
        [{"text": "What is my balance?", "category": "balance"}],
        {"train": [], "val": [["Explain the policy", "policy"]]},
        {
            "balance": {
                "source_intent": "balance",
                "mapping_status": "EXACT_MATCH",
                "sentinelvoice_intent": "account_balance",
            }
        },
        {
            "policy": {
                "source_intent": "policy",
                "mapping_status": "EXACT_MATCH",
                "sentinelvoice_intent": "informational_policy",
            }
        },
        {
            "account_balance": "PRIVATE_READ",
            "informational_policy": "PUBLIC",
        },
        "bank-rev",
        "clinc-rev",
    )
    assert [row["example_id"] for row in rows] == [
        "banking77:bank-rev:train:000001",
        "clinc:clinc-rev:val:000001",
    ]
    assert [row["text"] for row in rows] == [
        "What is my balance?",
        "Explain the policy",
    ]


def test_external_reconstruction_rejects_hash_mismatch() -> None:
    member = _member(
        example_id="banking77:bank-rev:train:000001",
        source_id="banking77",
        source_split="train",
        source_label="balance",
        source_row_index=1,
        source_revision="bank-rev",
        text="Original text",
        intent="account_balance",
        risk="PRIVATE_READ",
    )
    with pytest.raises(ValueError, match="text hash mismatch"):
        final_evaluation.reconstruct_lockbox_members(
            [member],
            [{"text": "Changed text", "category": "balance"}],
            {"train": [], "val": []},
            {
                "balance": {
                    "source_intent": "balance",
                    "mapping_status": "EXACT_MATCH",
                    "sentinelvoice_intent": "account_balance",
                }
            },
            {},
            {"account_balance": "PRIVATE_READ"},
            "bank-rev",
            "clinc-rev",
        )


@pytest.mark.parametrize("forbidden_split", ["test", "oos_test"])
def test_external_reconstruction_rejects_clinc_test_splits(
    forbidden_split: str,
) -> None:
    member = _member(
        example_id=f"clinc:clinc-rev:{forbidden_split}:000001",
        source_id="clinc150_oos",
        source_split=forbidden_split,
        source_label="policy",
        source_row_index=1,
        source_revision="clinc-rev",
        text="Forbidden final source",
        intent="informational_policy",
        risk="PUBLIC",
    )
    with pytest.raises(ValueError, match="unapproved CLINC source identity"):
        final_evaluation.reconstruct_lockbox_members(
            [member],
            [],
            {"train": [], "val": []},
            {},
            {
                "policy": {
                    "source_intent": "policy",
                    "mapping_status": "EXACT_MATCH",
                    "sentinelvoice_intent": "informational_policy",
                }
            },
            {"informational_policy": "PUBLIC"},
            "bank-rev",
            "clinc-rev",
        )


def test_no_banking_test_or_cfpb_source_paths(config: dict[str, Any]) -> None:
    source_paths = [
        spec["path"]
        for spec in config["external_lockbox"]["source_inputs"].values()
    ]
    assert "data/evals/v2/external/raw/banking77/train.csv" in source_paths
    assert all("banking77/test" not in path.lower() for path in source_paths)
    assert all("cfpb" not in path.lower() for path in source_paths)
    runner_source = RUNNER_PATH.read_text(encoding="utf-8").lower()
    assert "cfpb_narratives" not in runner_source
    assert "banking77/test.csv" not in runner_source
    reconstruction_source = inspect.getsource(
        final_evaluation.reconstruct_external_lockbox
    )
    assert 'for split in ("train", "val")' in reconstruction_source
    member_source = inspect.getsource(final_evaluation.reconstruct_lockbox_members)
    assert 'split not in {"train", "val"}' in member_source


def test_preflight_does_not_fit_infer_or_parse_final_text() -> None:
    source = inspect.getsource(final_evaluation.preflight)
    assert ".fit(" not in source
    assert ".predict(" not in source
    assert "embed_final_texts" not in source
    assert "reconstruct_external_lockbox" not in source
    assert "load_challenge_examples" not in source


def test_completed_report_separates_pre_evaluation_and_execution_status() -> None:
    snapshot = {
        "model_inference_performed": False,
        "final_text_parsed": False,
        "final_test_embeddings_generated": False,
    }
    integrity = final_evaluation.completed_evaluation_integrity(snapshot)

    assert integrity["pre_evaluation_integrity"] == snapshot
    assert integrity["evaluation_execution"] == {
        "final_text_parsed": True,
        "final_test_embeddings_generated": True,
        "model_inference_performed": True,
        "external_lockbox_evaluated": True,
        "challenge_set_evaluated": True,
        "comparator_evaluated": True,
    }
    assert snapshot == {
        "model_inference_performed": False,
        "final_text_parsed": False,
        "final_test_embeddings_generated": False,
    }


def test_completed_report_usage_and_governance_flags_remain_explicit() -> None:
    source = inspect.getsource(final_evaluation.evaluate)
    for expected in (
        '"final_lockbox_used": True',
        '"challenge_set_used": True',
        '"cfpb_used": False',
        '"hyperparameter_tuning_on_final_data": False',
        '"threshold_tuning_on_final_data": False',
        '"model_switching_after_final_results": False',
        '"raw_text_persisted": False',
    ):
        assert expected in source


def test_safety_definitions_reuse_step_4_exactly(
    config: dict[str, Any], step_4_config: dict[str, Any]
) -> None:
    assert config["safety_definitions"]["protected_write_intents"] == [
        "create_dispute",
        "freeze_card",
    ]
    assert config["safety_definitions"]["unchanged_from_step_4_and_5"] is True
    gold = [
        "unsupported_or_uncertain",
        "unsupported_or_uncertain",
        "freeze_card",
        "create_dispute",
        "account_balance",
    ]
    predicted = [
        "unsupported_or_uncertain",
        "account_balance",
        "create_dispute",
        "create_dispute",
        "freeze_card",
    ]
    safety = tournament.safety_metrics(gold, predicted, step_4_config)
    assert safety["false_supported_rate"] == pytest.approx(0.5)
    assert safety["protected_write_recall"] == pytest.approx(0.5)
    assert safety["protected_write_false_positive_rate"] == pytest.approx(1 / 3)


def test_external_protected_recall_is_not_applicable_without_gold_positives(
    step_4_config: dict[str, Any],
) -> None:
    metrics = final_evaluation.final_dataset_metrics(
        [
            {"intent": "unsupported_or_uncertain"},
            {"intent": "account_balance"},
        ],
        ["unsupported_or_uncertain", "account_balance"],
        step_4_config,
        include_challenge_classes=False,
    )
    assert metrics["protected_write_recall"] == "NOT_APPLICABLE"


def test_final_acceptance_policy_is_exact(config: dict[str, Any]) -> None:
    assert config["final_acceptance_policy"] == {
        "applies_to_model": "final_v2c3",
        "challenge_set": {
            "protected_write_false_positive_rate_maximum": 0.01,
            "protected_write_recall_minimum": 0.8,
            "unsupported_or_uncertain_recall_minimum": 0.8,
        },
        "external_lockbox": {
            "protected_write_false_positive_rate_maximum": 0.01,
            "unsupported_or_uncertain_recall_minimum": 0.8,
            "protected_write_recall": (
                "NOT_APPLICABLE when the gold denominator is zero"
            ),
        },
        "all_required_gates_must_pass": True,
        "macro_f1_or_accuracy_gate": False,
        "retuning_after_final_results_allowed": False,
        "model_switching_after_final_results_allowed": False,
    }


def test_acceptance_requires_every_frozen_safety_gate(config: dict[str, Any]) -> None:
    passing_external = {
        "protected_write_false_positive_rate": 0.01,
        "unsupported_or_uncertain_recall": 0.8,
    }
    passing_challenge = {
        "protected_write_false_positive_rate": 0.01,
        "protected_write_recall": 0.8,
        "unsupported_or_uncertain_recall": 0.8,
    }
    passing = final_evaluation.final_acceptance(
        passing_external, passing_challenge, config
    )
    assert passing["final_safety_acceptable"] is True

    failing_challenge = dict(passing_challenge, protected_write_recall=0.79)
    failing = final_evaluation.final_acceptance(
        passing_external, failing_challenge, config
    )
    assert failing["final_safety_acceptable"] is False


def test_metric_and_safety_gate_code_are_unchanged() -> None:
    metric_source = inspect.getsource(final_evaluation.final_dataset_metrics)
    safety_gate_source = inspect.getsource(final_evaluation.final_acceptance)
    assert hashlib.sha256(metric_source.encode()).hexdigest() == (
        "691a8bc17dc91ca31c24de66cb38c73e74ba4f87ec848cd8442b88e02f7f4222"
    )
    assert hashlib.sha256(safety_gate_source.encode()).hexdigest() == (
        "75c7c3265712905a6f2ddcfc83d250275bbf54d8d9bf93040b8e431a0917b6ca"
    )


def test_datasets_are_reported_separately_without_combined_metric(
    config: dict[str, Any],
) -> None:
    metrics = config["metrics"]
    assert metrics["datasets_reported_separately"] is True
    assert metrics["combined_final_metric_allowed"] is False
    report_source = inspect.getsource(final_evaluation.evaluate)
    assert '"external_lockbox_results"' in report_source
    assert '"challenge_set_results"' in report_source
    assert '"combined_final_metric_reported": False' in report_source


def test_no_hyperparameter_or_threshold_tuning_and_no_runtime_integration(
    config: dict[str, Any],
) -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "GridSearchCV" not in source
    assert "RandomizedSearchCV" not in source
    assert "ParameterGrid" not in source
    assert "backend.app" not in source
    assert config["final_acceptance_policy"][
        "retuning_after_final_results_allowed"
    ] is False
    assert config["challenge_set"]["threshold_selection_eligible"] is False
    assert config["runtime_authority"] is False


def test_report_contract_and_implementation_persist_no_raw_text(
    config: dict[str, Any],
) -> None:
    assert config["report"]["raw_text_persisted"] is False
    assert config["report"]["cfpb_used"] is False
    report_source = inspect.getsource(final_evaluation.evaluate)
    report_literal = report_source[report_source.index("report = {") :]
    assert '"raw_text_persisted": False' in report_literal
    assert '"predictions"' not in report_literal
    assert '"text"' not in report_literal
