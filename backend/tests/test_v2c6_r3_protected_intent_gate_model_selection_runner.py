from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import run_v2c6_r3_protected_intent_gate_model_selection as runner

ROOT = Path(__file__).resolve().parents[2]
ML_PATH = ROOT / "data/evals/v2/ml"
FINAL_HOLDOUT_PATH = ML_PATH / "v2c5_final_holdout.json"
UNSUPPORTED = runner.UNSUPPORTED_INTENT
# Whole-token marker; family IDs such as "..._contextual_boundary" must not collide.
HARD_NEGATIVE_MARKER = "hardnegativemarker"


@pytest.fixture(scope="module")
def real_inputs() -> runner.LoadedInputs:
    return runner.load_and_validate_inputs()


@pytest.fixture()
def contract(real_inputs: runner.LoadedInputs) -> dict[str, Any]:
    return deepcopy(real_inputs.contract)


@pytest.fixture(scope="module")
def real_splits(real_inputs: runner.LoadedInputs) -> runner.EvaluationSplits:
    return runner.build_evaluation_splits(real_inputs)


def _text_fields(text: str) -> dict[str, str]:
    return {
        "text": text,
        "text_sha256": runner.sha256_bytes(text.encode("utf-8")),
        "normalized_text_sha256": runner.sha256_bytes(text.lower().encode("utf-8")),
    }


def synthetic_training() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in runner.TRAINING_FAMILY_IDS:
        for target in runner.PROTECTED_INTENTS:
            for index in range(20):
                for role, intent, marker in (
                    ("protected_positive", target, "genuine"),
                    ("targeted_unsupported_negative", UNSUPPORTED, HARD_NEGATIVE_MARKER),
                ):
                    record_id = f"{family}:{target}:{marker}:{index:02d}"
                    rows.append(
                        {
                            "record_id": record_id,
                            "group_id": f"{record_id}:group",
                            "intent": intent,
                            "protected_boundary_target": target,
                            "verifier_role": role,
                            "source_family_id": family,
                            **_text_fields(f"r3 {marker} {target} {family} {index}"),
                        }
                    )
    return rows


def synthetic_development(
    training: list[dict[str, Any]], labels: list[str]
) -> list[dict[str, Any]]:
    historical: list[dict[str, Any]] = []
    for label in labels:
        for index in range(6):
            example_id = f"historical:{label}:{index}"
            historical.append(
                {
                    "example_id": example_id,
                    "group_id": f"{example_id}:group",
                    "intent": label,
                    "data_role": "development",
                    **_text_fields(f"historical {label} {index}"),
                }
            )
    additions = [
        {
            "example_id": row["record_id"],
            "record_id": row["record_id"],
            "group_id": row["group_id"],
            "intent": row["intent"],
            "protected_boundary_target": row["protected_boundary_target"],
            "verifier_role": row["verifier_role"],
            "source_family_id": row["source_family_id"],
            "v2c6_lineage": runner.R3_LINEAGE,
            "data_role": "development",
            "text": row["text"],
            "text_sha256": row["text_sha256"],
            "normalized_text_sha256": row["normalized_text_sha256"],
        }
        for row in sorted(training, key=lambda item: item["record_id"])
    ]
    return [*historical, *additions]


def synthetic_fresh() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in runner.FRESH_FAMILY_IDS:
        for intent in runner.PRIMARY_INTENTS:
            for index in range(40):
                record_id = f"{family}:{intent}:{index:02d}"
                if intent == UNSUPPORTED:
                    target: str | None = runner.PROTECTED_INTENTS[index % 4]
                    marker = HARD_NEGATIVE_MARKER
                else:
                    target = intent if intent in runner.PROTECTED_INTENTS else None
                    marker = "genuine"
                rows.append(
                    {
                        "record_id": record_id,
                        "group_id": f"{record_id}:group",
                        "intent": intent,
                        "protected_boundary_target": target,
                        "source_family_id": family,
                        "dataset_role": "fresh_source_evaluation",
                        "excluded_from_candidate_fitting": True,
                        "not_training_data": True,
                        **_text_fields(f"fresh {marker} {family} {intent} {index}"),
                    }
                )
    return rows


def synthetic_inputs(real_inputs: runner.LoadedInputs) -> runner.LoadedInputs:
    training = synthetic_training()
    labels = real_inputs.step29g_contract["taxonomy"]["intent_label_order"]
    return runner.LoadedInputs(
        contract=deepcopy(real_inputs.contract),
        prior_contract=deepcopy(real_inputs.prior_contract),
        step29g_contract=deepcopy(real_inputs.step29g_contract),
        training_examples=tuple(training),
        development_examples=tuple(synthetic_development(training, labels)),
        fresh_examples=tuple(synthetic_fresh()),
        source_hashes=dict(real_inputs.source_hashes),
    )


def row_identifier(row: dict[str, Any]) -> str:
    return str(row.get("example_id", row.get("record_id")))


def fake_primary_router(calls: list[dict[str, Any]]) -> Callable[..., list[str]]:
    """Over-route targeted unsupported rows to their protected boundary."""

    def predict(
        train_rows: Any,
        evaluation_rows: Any,
        _step29g_contract: Any,
        *,
        train_bge: np.ndarray,
        evaluation_bge: np.ndarray,
        c_value: float,
    ) -> list[str]:
        assert c_value == 1.0
        assert train_bge.shape[0] == len(train_rows)
        assert evaluation_bge.shape[0] == len(evaluation_rows)
        calls.append(
            {
                "train_ids": [row_identifier(row) for row in train_rows],
                "evaluation_ids": [row_identifier(row) for row in evaluation_rows],
            }
        )
        return [
            str(row["protected_boundary_target"])
            if row["intent"] == UNSUPPORTED and row.get("protected_boundary_target")
            else str(row["intent"])
            for row in evaluation_rows
        ]

    return predict


class FakeVerifier:
    def __init__(self, intent: str, rows: Any, *, accept_all: bool) -> None:
        labels = [runner.verifier_training_label(row, intent) for row in rows]
        self.protected_intent = intent
        self.positive_count = labels.count(intent)
        self.negative_count = labels.count(UNSUPPORTED)
        self.training_ids = [str(row["example_id"]) for row in rows]
        self.accept_all = accept_all
        self.calls: list[list[str]] = []

    def predict(self, texts: Any) -> list[str]:
        self.calls.append(list(texts))
        return [
            self.protected_intent
            if self.accept_all or HARD_NEGATIVE_MARKER not in text.split()
            else UNSUPPORTED
            for text in texts
        ]


def fake_verifier_fit(
    fitted: list[FakeVerifier], *, accept_all: bool = False
) -> Callable[..., FakeVerifier]:
    def fit(intent: str, rows: Any, _step29g_contract: Any, *, c_value: float) -> FakeVerifier:
        assert c_value == 1.0
        verifier = FakeVerifier(intent, rows, accept_all=accept_all)
        fitted.append(verifier)
        return verifier

    return fit


def patch_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ml_path = tmp_path / "ml"
    monkeypatch.setattr(runner, "RESULTS_PATH", ml_path / "results.json")
    monkeypatch.setattr(runner, "RESULTS_MANIFEST_PATH", ml_path / "results.manifest.json")
    monkeypatch.setattr(runner, "BGE_CACHE_PATH", ml_path / "local/bge_cache.npz")
    monkeypatch.setattr(
        runner, "BGE_CACHE_MANIFEST_PATH", ml_path / "local/bge_cache.manifest.json"
    )


def patch_fake_bge(monkeypatch: pytest.MonkeyPatch, inputs: runner.LoadedInputs) -> None:
    def prepare(_inputs: Any, _splits: Any) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
        return (
            {
                "development": np.zeros((len(inputs.development_examples), 384), np.float32),
                "fresh_evaluation": np.zeros((len(inputs.fresh_examples), 384), np.float32),
            },
            {"synthetic_cache": True},
            False,
        )

    monkeypatch.setattr(runner, "prepare_bge_cache", prepare)


def forbid_model_work(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    for name in (
        "load_fastembed_model",
        "embed_texts",
        "fit_classifier",
        "fit_predict_primary_router",
        "fit_protected_verifier",
    ):
        monkeypatch.setattr(
            runner,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(f"{phase} called {_name}"),
        )


def run_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    real_inputs: runner.LoadedInputs,
    *,
    accept_all: bool = False,
) -> tuple[runner.LoadedInputs, list[dict[str, Any]], list[FakeVerifier], dict[str, Any]]:
    inputs = synthetic_inputs(real_inputs)
    patch_outputs(monkeypatch, tmp_path)
    patch_fake_bge(monkeypatch, inputs)
    primary_calls: list[dict[str, Any]] = []
    verifiers: list[FakeVerifier] = []
    monkeypatch.setattr(runner, "fit_predict_primary_router", fake_primary_router(primary_calls))
    monkeypatch.setattr(
        runner, "fit_protected_verifier", fake_verifier_fit(verifiers, accept_all=accept_all)
    )
    runner.run_model_selection(inputs)
    payload = json.loads(runner.RESULTS_PATH.read_text(encoding="utf-8"))
    return inputs, primary_calls, verifiers, payload


def routed_row(gold: str, primary: str, decision: str | None) -> dict[str, Any]:
    final = UNSUPPORTED if decision == runner.VERIFIER_REJECT else primary
    return {
        "gold_intent": gold,
        "primary_predicted_intent": primary,
        "predicted_intent": final,
        "verifier_invoked": decision is not None,
        "verifier_intent": primary if decision is not None else None,
        "verifier_decision": decision,
    }


def safety_values(
    recall: float = 0.9, fpr: float = 0.0, unsupported: float = 0.9
) -> dict[str, float]:
    return {
        "protected_recall": recall,
        "protected_false_positive_rate": fpr,
        "unsupported_recall": unsupported,
    }


def all_scopes(
    overrides: dict[str, dict[str, float]] | None = None,
) -> list[tuple[str, dict[str, float]]]:
    overrides = overrides or {}
    return [(scope, overrides.get(scope, safety_values())) for scope in runner.SAFETY_SCOPES]


def selection_result(
    candidate_id: str,
    *,
    eligible: bool = True,
    worst: float = 0.70,
    pooled: float = 0.75,
    group_macro: float = 0.80,
    protected_fpr: float = 0.005,
    protected_recall: float = 0.90,
    unsupported_recall: float = 0.90,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "eligible": eligible,
        "ineligibility_reasons": [] if eligible else ["failed"],
        "configuration": {"candidate_id": candidate_id},
        "selection_metrics": {
            "worst_fresh_family_primary_8_macro_f1": worst,
            "pooled_fresh_primary_8_macro_f1": pooled,
            "pooled_group_cv_macro_f1_16": group_macro,
            "pooled_fresh_protected_false_positive_rate": protected_fpr,
            "pooled_fresh_protected_recall": protected_recall,
            "pooled_fresh_unsupported_recall": unsupported_recall,
            "model_complexity_rank": runner.model_complexity_rank(candidate_id),
            "candidate_id": candidate_id,
        },
    }


# Contract identity and bounded candidate search.


def test_exact_r3_contract_identity_and_hash(contract: dict[str, Any]) -> None:
    runner.validate_contract(contract)
    assert runner.sha256_file(runner.CONTRACT_PATH) == runner.EXPECTED_CONTRACT_SHA256


def test_exact_two_candidates_in_frozen_order(contract: dict[str, Any]) -> None:
    candidates = contract["candidate_search_space"]["candidates"]
    assert [row["candidate_id"] for row in candidates] == [
        "HYBRID_CONTROL_R3",
        "HYBRID_PROTECTED_VERIFIER_R3",
    ]
    assert runner.EXPECTED_CANDIDATE_IDS == ("HYBRID_CONTROL_R3", "HYBRID_PROTECTED_VERIFIER_R3")
    assert [runner.model_complexity_rank(cid) for cid in runner.EXPECTED_CANDIDATE_IDS] == [1, 2]


def test_candidate_expansion_is_rejected(contract: dict[str, Any]) -> None:
    changed = deepcopy(contract)
    extra = deepcopy(changed["candidate_search_space"]["candidates"][0])
    extra["candidate_id"] = "HYBRID_THIRD_R3"
    changed["candidate_search_space"]["candidates"].append(extra)
    with pytest.raises(ValueError, match="candidate"):
        runner.validate_contract(changed)
    reordered = deepcopy(contract)
    reordered["candidate_search_space"]["candidates"].reverse()
    with pytest.raises(ValueError, match="candidate"):
        runner.validate_contract(reordered)


@pytest.mark.parametrize("value", [0.5, 4.0])
def test_changed_c_is_rejected(contract: dict[str, Any], value: float) -> None:
    primary = deepcopy(contract)
    primary["architecture_decision"]["primary_router"]["C"] = value
    with pytest.raises(ValueError, match="primary-router"):
        runner.validate_contract(primary)
    verifier = deepcopy(contract)
    verifier["verifier_architecture"]["C"] = value
    with pytest.raises(ValueError, match="verifier architecture"):
        runner.validate_contract(verifier)


def test_changed_class_weight_is_rejected(contract: dict[str, Any]) -> None:
    primary = deepcopy(contract)
    primary["architecture_decision"]["primary_router"]["class_weight"] = "balanced"
    with pytest.raises(ValueError, match="primary-router"):
        runner.validate_contract(primary)
    verifier = deepcopy(contract)
    verifier["verifier_architecture"]["class_weight"] = "balanced"
    with pytest.raises(ValueError, match="verifier architecture"):
        runner.validate_contract(verifier)


@pytest.mark.parametrize(
    "field", ["calibration", "threshold_tuning", "confidence_override", "llm_verifier"]
)
def test_calibration_and_threshold_paths_are_prohibited(
    contract: dict[str, Any], field: str
) -> None:
    changed = deepcopy(contract)
    changed["verifier_architecture"][field] = True
    with pytest.raises(ValueError, match="verifier architecture"):
        runner.validate_contract(changed)


def test_prior_hybrid_calibration_or_threshold_change_is_rejected(
    real_inputs: runner.LoadedInputs,
) -> None:
    runner.validate_prior_conventions(real_inputs.prior_contract)
    for field in ("calibration_permitted", "threshold_tuning_permitted"):
        changed = deepcopy(real_inputs.prior_contract)
        changed["candidate_implementation_semantics"][runner.PRIMARY_ROUTER_ID][field] = True
        with pytest.raises(ValueError, match="Hybrid primary-router semantics"):
            runner.validate_prior_conventions(changed)


def test_safety_gate_selection_and_stop_rule_changes_are_rejected(
    contract: dict[str, Any],
) -> None:
    weakened = deepcopy(contract)
    weakened["safety_gates"]["gates"][1]["threshold"] = 0.02
    with pytest.raises(ValueError, match="safety gates"):
        runner.validate_contract(weakened)
    reordered = deepcopy(contract)
    criteria = reordered["selection_rule"]["ordered_lexicographic_criteria"]
    criteria[0], criteria[1] = criteria[1], criteria[0]
    with pytest.raises(ValueError, match="selection rule"):
        runner.validate_contract(reordered)
    r4 = deepcopy(contract)
    r4["stop_rule"]["automatic_r4_classifier_or_data_cycle_authorized"] = True
    with pytest.raises(ValueError, match="stop rule"):
        runner.validate_contract(r4)


def test_reused_r2_runner_is_hash_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = (ROOT / runner.R2_RUNNER_RELATIVE_PATH).read_bytes()
    assert runner.sha256_bytes(source) == runner.EXPECTED_R2_RUNNER_SHA256
    changed = tmp_path / "drifted_runner.py"
    changed.write_bytes(source + b"\n# drift\n")
    monkeypatch.setattr(
        runner.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: pytest.fail("mismatched runner reached module loading"),
    )
    with pytest.raises(ValueError, match="reused frozen R2 runner changed"):
        runner.load_frozen_r2_helpers(changed, module_name="_test_r3_drifted_r2_runner")
    assert "_test_r3_drifted_r2_runner" not in runner.sys.modules


def test_verified_r2_runner_loads_through_import_machinery() -> None:
    module_name = "_test_r3_verified_r2_runner"
    try:
        module = runner.load_frozen_r2_helpers(module_name=module_name)
        assert runner.sys.modules[module_name] is module
        assert module.__spec__ is not None and module.__spec__.name == module_name
        assert Path(module.__file__).resolve() == (ROOT / runner.R2_RUNNER_RELATIVE_PATH)
        assert runner.PRIMARY_ROUTER_ID in module.EXPECTED_CANDIDATE_IDS
        assert callable(module.safety_metrics)
    finally:
        runner.sys.modules.pop(module_name, None)


def test_failed_r2_runner_import_removes_partial_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    broken = tmp_path / "broken_runner.py"
    broken.write_bytes(b"raise RuntimeError('broken frozen runner')\n")
    module_name = "_test_r3_broken_r2_runner"
    with pytest.raises(RuntimeError, match="broken frozen runner"):
        runner.load_frozen_r2_helpers(
            broken,
            expected_sha256=runner.sha256_file(broken),
            module_name=module_name,
        )
    assert module_name not in runner.sys.modules
    monkeypatch.setattr(runner.importlib.util, "spec_from_file_location", lambda *_: None)
    with pytest.raises(ImportError, match="import spec"):
        runner.load_frozen_r2_helpers(
            broken, expected_sha256=runner.sha256_file(broken), module_name=module_name
        )


# Frozen datasets, manifests, and hashes.


def test_exact_r3_dataset_schemas_and_counts(real_inputs: runner.LoadedInputs) -> None:
    assert len(real_inputs.training_examples) == 480
    assert len(real_inputs.development_examples) == 10088
    assert len(real_inputs.fresh_examples) == 640
    for family in runner.FRESH_FAMILY_IDS:
        assert sum(row["source_family_id"] == family for row in real_inputs.fresh_examples) == 320
    additions = real_inputs.development_examples[9608:]
    assert all(row["v2c6_lineage"] == runner.R3_LINEAGE for row in additions)


def test_exact_frozen_hashes_and_manifests(real_inputs: runner.LoadedInputs) -> None:
    expected = {
        runner.TRAINING_DATASET_PATH: runner.EXPECTED_TRAINING_SHA256,
        runner.TRAINING_MANIFEST_PATH: runner.EXPECTED_TRAINING_MANIFEST_SHA256,
        runner.DEVELOPMENT_DATASET_PATH: runner.EXPECTED_DEVELOPMENT_SHA256,
        runner.DEVELOPMENT_MANIFEST_PATH: runner.EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        runner.FRESH_DATASET_PATH: runner.EXPECTED_FRESH_SHA256,
        runner.FRESH_MANIFEST_PATH: runner.EXPECTED_FRESH_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        assert runner.sha256_file(path) == digest
    assert real_inputs.source_hashes["fresh_dataset"] == runner.EXPECTED_FRESH_SHA256


def test_changed_manifest_governance_is_rejected() -> None:
    manifest = runner.read_json(runner.FRESH_MANIFEST_PATH)
    manifest["governance"]["final_holdout_accessed"] = True
    with pytest.raises(ValueError, match="governance changed"):
        runner.validate_dataset_manifest(
            manifest,
            schema_version=runner.FRESH_MANIFEST_SCHEMA_VERSION,
            artifact_relative_path=runner.FRESH_DATASET_RELATIVE_PATH,
            artifact_schema_version=runner.FRESH_SCHEMA_VERSION,
            artifact_sha256=runner.EXPECTED_FRESH_SHA256,
            expected_counts={"record_count": 640},
        )


def test_changed_r3_datasets_are_rejected(real_inputs: runner.LoadedInputs) -> None:
    training = runner.read_json(runner.TRAINING_DATASET_PATH)
    changed_training = deepcopy(training)
    changed_training["examples"][0]["verifier_role"] = "targeted_unsupported_negative"
    with pytest.raises(ValueError, match="verifier metadata"):
        runner.validate_training_dataset(changed_training)

    development = runner.read_json(runner.DEVELOPMENT_DATASET_PATH)
    changed_development = deepcopy(development)
    changed_development["example_count"] = 10087
    with pytest.raises(ValueError, match="population"):
        runner.validate_development_dataset(
            changed_development, real_inputs.step29g_contract, real_inputs.training_examples
        )
    changed_addition = deepcopy(development)
    changed_addition["examples"][-1]["intent"] = "freeze_card"
    with pytest.raises(ValueError, match="addition field changed"):
        runner.validate_development_dataset(
            changed_addition, real_inputs.step29g_contract, real_inputs.training_examples
        )

    fresh = runner.read_json(runner.FRESH_DATASET_PATH)
    changed_fresh = deepcopy(fresh)
    changed_fresh["examples"][0]["source_family_id"] = runner.FRESH_FAMILY_IDS[1]
    with pytest.raises(ValueError, match="per-family intent counts"):
        runner.validate_fresh_dataset(changed_fresh, real_inputs.development_examples)
    unexcluded = deepcopy(fresh)
    unexcluded["examples"][0]["excluded_from_candidate_fitting"] = False
    with pytest.raises(ValueError, match="not excluded from fitting"):
        runner.validate_fresh_dataset(unexcluded, real_inputs.development_examples)


def test_final_holdout_read_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Path, "read_bytes", lambda *_args: pytest.fail("filesystem read attempted")
    )
    with pytest.raises(PermissionError, match="never access"):
        runner.read_json(FINAL_HOLDOUT_PATH)
    with pytest.raises(PermissionError, match="never access"):
        runner.read_bytes(ML_PATH / "local/renamed_final_holdout_copy.json")


def test_final_holdout_hash_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Path, "read_bytes", lambda *_args: pytest.fail("filesystem read attempted")
    )
    with pytest.raises(PermissionError, match="never access"):
        runner.sha256_file(FINAL_HOLDOUT_PATH)


def test_final_holdout_policy_change_is_rejected(contract: dict[str, Any]) -> None:
    changed = deepcopy(contract)
    changed["final_holdout_policy"]["hashing_permitted"] = True
    with pytest.raises(ValueError, match="final-holdout policy"):
        runner.validate_contract(changed)


# Grouped split and verifier fold membership.


def test_group_folds_are_deterministic_five_and_group_atomic(
    real_inputs: runner.LoadedInputs, contract: dict[str, Any]
) -> None:
    first = runner.build_group_folds(real_inputs.development_examples, contract)
    second = runner.build_group_folds(real_inputs.development_examples, contract)
    assert first == second
    assert len(first) == 5
    validated = sorted(index for fold in first for index in fold.validation_indices)
    assert validated == list(range(10088))
    for fold in first:
        train_groups = {
            real_inputs.development_examples[index]["group_id"] for index in fold.train_indices
        }
        validation_groups = {
            real_inputs.development_examples[index]["group_id"]
            for index in fold.validation_indices
        }
        assert not train_groups & validation_groups


def test_group_leakage_and_missing_groups_fail_closed(
    monkeypatch: pytest.MonkeyPatch, contract: dict[str, Any]
) -> None:
    examples = [
        {"example_id": f"row-{index}", "group_id": f"group-{index // 2}", "intent": "x"}
        for index in range(4)
    ]

    class LeakySplitter:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def split(self, *_args: Any) -> Any:
            yield np.asarray([0, 2]), np.asarray([1, 3])

    monkeypatch.setattr(runner, "StratifiedGroupKFold", LeakySplitter)
    with pytest.raises(ValueError, match="group leakage"):
        runner.build_group_folds(examples, contract)
    missing = deepcopy(examples)
    missing[0]["group_id"] = ""
    with pytest.raises(ValueError, match="missing group membership"):
        runner.build_group_folds(missing, contract)


def test_split_audit_excludes_fresh_rows_and_verifier_validation_rows(
    real_splits: runner.EvaluationSplits,
) -> None:
    audit = real_splits.audit
    assert audit["fold_count"] == 5
    assert audit["pooled_validation_count"] == 10088
    assert audit["fresh_records_in_group_cv"] == 0
    assert audit["fresh_record_id_overlap_count"] == 0
    assert audit["fresh_group_id_overlap_count"] == 0
    assert audit["fresh_text_sha256_overlap_count"] == 0
    assert audit["fresh_normalized_text_sha256_overlap_count"] == 0
    assert audit["verifier_validation_leakage_count"] == 0
    for intent in runner.PROTECTED_INTENTS:
        excluded = sum(
            fold["verifier_training"][intent]["validation_excluded_count"]
            for fold in audit["folds"]
        )
        assert excluded == 120
        assert all(
            fold["verifier_training"][intent]["validation_rows_in_verifier_fitting"] == 0
            for fold in audit["folds"]
        )


def test_full_verifier_populations_are_exactly_60_plus_60(
    real_inputs: runner.LoadedInputs, real_splits: runner.EvaluationSplits
) -> None:
    for intent, indices in real_splits.verifier_populations.items():
        labels = [
            runner.verifier_training_label(real_inputs.development_examples[index], intent)
            for index in indices
        ]
        assert labels.count(intent) == 60
        assert labels.count(UNSUPPORTED) == 60
        assert len(indices) == 120


def test_verifier_population_comes_only_from_r3_addendum(
    real_inputs: runner.LoadedInputs,
) -> None:
    inputs = synthetic_inputs(real_inputs)
    populations = runner.build_verifier_populations(
        inputs.training_examples, inputs.development_examples
    )
    selected = {
        inputs.development_examples[index]["example_id"]
        for indices in populations.values()
        for index in indices
    }
    assert selected == {row["record_id"] for row in inputs.training_examples}
    assert all(
        inputs.development_examples[index]["v2c6_lineage"] == runner.R3_LINEAGE
        for indices in populations.values()
        for index in indices
    )


def test_historical_development_rows_cannot_enter_verifier_fitting(
    monkeypatch: pytest.MonkeyPatch, real_inputs: runner.LoadedInputs
) -> None:
    inputs = synthetic_inputs(real_inputs)
    historical = next(
        row for row in inputs.development_examples if row["intent"] == "freeze_card"
    )
    assert "v2c6_lineage" not in historical
    with pytest.raises(ValueError, match="historical development record"):
        runner.verifier_training_label(historical, "freeze_card")
    monkeypatch.setattr(runner, "fit_classifier", lambda *_: pytest.fail("fit attempted"))
    r3_rows = [
        row
        for row in inputs.development_examples
        if row.get("protected_boundary_target") == "freeze_card"
    ]
    with pytest.raises(ValueError, match="historical development record"):
        runner.fit_protected_verifier(
            "freeze_card", [*r3_rows, historical], inputs.step29g_contract, c_value=1.0
        )


def test_verifier_fold_training_inherits_development_fold_membership(
    real_inputs: runner.LoadedInputs,
) -> None:
    inputs = synthetic_inputs(real_inputs)
    splits = runner.build_evaluation_splits(inputs)
    for fold in splits.group_folds:
        plan = splits.verifier_fold_plans[fold.partition_id]
        for intent, population in splits.verifier_populations.items():
            assert set(plan[intent]) == set(population) & set(fold.train_indices)
            assert not set(plan[intent]) & set(fold.validation_indices)
            assert len(plan[intent]) < len(population)


def test_verifier_fold_missing_binary_class_fails_closed(
    real_inputs: runner.LoadedInputs,
) -> None:
    inputs = synthetic_inputs(real_inputs)
    populations = runner.build_verifier_populations(
        inputs.training_examples, inputs.development_examples
    )
    positives = {
        index
        for index in populations["freeze_card"]
        if inputs.development_examples[index]["intent"] == "freeze_card"
    }
    all_indices = set(range(len(inputs.development_examples)))
    fold = runner.Partition(
        partition_id="synthetic_fold",
        train_indices=tuple(sorted(all_indices - positives)),
        validation_indices=tuple(sorted(positives)),
    )
    with pytest.raises(ValueError, match="missing verifier binary class"):
        runner.build_verifier_fold_plan(populations, fold, inputs.development_examples)


# Route-only verifier gate and diagnostics.


class RecordingVerifier:
    def __init__(self, intent: str, *, accept: bool) -> None:
        self.intent = intent
        self.accept = accept
        self.calls: list[list[str]] = []

    def predict(self, texts: Any) -> list[str]:
        self.calls.append(list(texts))
        return [self.intent if self.accept else UNSUPPORTED for _ in texts]


def test_gate_routes_only_protected_predictions_to_their_own_verifier() -> None:
    verifiers = {
        "cancel_transfer": RecordingVerifier("cancel_transfer", accept=True),
        "close_account": RecordingVerifier("close_account", accept=False),
        "create_dispute": RecordingVerifier("create_dispute", accept=True),
        "freeze_card": RecordingVerifier("freeze_card", accept=True),
    }
    primary = ["account_blocked", "freeze_card", "close_account", UNSUPPORTED]
    final, decisions = runner.apply_protected_verifier_gate(
        primary, ["t0", "t1", "t2", "t3"], verifiers
    )
    assert final == ["account_blocked", "freeze_card", UNSUPPORTED, UNSUPPORTED]
    assert verifiers["freeze_card"].calls == [["t1"]]
    assert verifiers["close_account"].calls == [["t2"]]
    assert verifiers["cancel_transfer"].calls == []
    assert verifiers["create_dispute"].calls == []
    assert decisions[0] == runner.NO_VERIFIER_DECISION
    assert decisions[3] == runner.NO_VERIFIER_DECISION
    assert decisions[1] == {
        "verifier_invoked": True,
        "verifier_intent": "freeze_card",
        "verifier_decision": "accept",
    }
    assert decisions[2]["verifier_decision"] == "reject"


def test_verifier_accept_preserves_and_reject_returns_unsupported() -> None:
    accepting = {
        intent: RecordingVerifier(intent, accept=True) for intent in runner.PROTECTED_INTENTS
    }
    rejecting = {
        intent: RecordingVerifier(intent, accept=False) for intent in runner.PROTECTED_INTENTS
    }
    primary = list(runner.PROTECTED_INTENTS)
    texts = [f"t{index}" for index in range(4)]
    assert runner.apply_protected_verifier_gate(primary, texts, accepting)[0] == primary
    assert runner.apply_protected_verifier_gate(primary, texts, rejecting)[0] == [
        UNSUPPORTED
    ] * 4


def test_gate_requires_exactly_four_verifiers() -> None:
    verifiers = {"freeze_card": RecordingVerifier("freeze_card", accept=True)}
    with pytest.raises(ValueError, match="four frozen protected verifiers"):
        runner.apply_protected_verifier_gate(["freeze_card"], ["t"], verifiers)


def diagnostic_rows() -> list[dict[str, Any]]:
    return [
        routed_row("freeze_card", "freeze_card", "accept"),
        routed_row("freeze_card", "freeze_card", "reject"),
        routed_row(UNSUPPORTED, "freeze_card", "reject"),
        routed_row("account_blocked", "close_account", "reject"),
        routed_row(UNSUPPORTED, "cancel_transfer", "accept"),
        routed_row("create_dispute", "freeze_card", "accept"),
        routed_row("close_account", "account_blocked", None),
    ]


def test_verifier_accept_reject_and_prevention_diagnostics() -> None:
    diagnostics = runner.verifier_diagnostics(diagnostic_rows())
    assert diagnostics["primary_protected_predictions_presented_to_verifier"] == 6
    assert diagnostics["verifier_accept_count"] == 3
    assert diagnostics["verifier_reject_to_unsupported_count"] == 3
    assert diagnostics["protected_false_positives_prevented_by_verifier"] == 2
    assert diagnostics["true_protected_requests_rejected_by_verifier"] == 1
    assert diagnostics["unsupported_requests_recovered_by_verifier"] == 1
    assert diagnostics["primary_protected_false_positive_rate"] == 1.0
    assert diagnostics["final_protected_false_positive_rate"] == pytest.approx(1 / 3)
    assert diagnostics["primary_protected_recall"] == 0.75
    assert diagnostics["final_protected_recall"] == 0.5
    assert diagnostics["final_unsupported_recall"] == 0.5


def test_verifier_recall_denominator_is_gold_and_primary_same_intent() -> None:
    per_intent = runner.verifier_diagnostics(diagnostic_rows())[
        "per_protected_intent_verifier_recall"
    ]
    freeze = per_intent["freeze_card"]
    assert freeze["presented_count"] == 4
    assert freeze["presented_gold_same_intent_count"] == 2
    assert freeze["accepted_gold_same_intent_count"] == 1
    assert freeze["presented_gold_other_count"] == 1
    assert freeze["verifier_recall"] == 0.5
    assert per_intent["close_account"]["verifier_recall"] is None
    assert per_intent["cancel_transfer"]["verifier_recall"] is None
    assert per_intent["create_dispute"]["presented_count"] == 0


def test_inconsistent_routing_rows_are_rejected() -> None:
    inconsistent = routed_row(UNSUPPORTED, "freeze_card", "reject")
    inconsistent["predicted_intent"] = "freeze_card"
    with pytest.raises(ValueError, match="rejection"):
        runner.validate_routing_rows([inconsistent], gated=True)
    control = routed_row(UNSUPPORTED, "freeze_card", "accept")
    with pytest.raises(ValueError, match="verifier invoked outside"):
        runner.validate_routing_rows([control], gated=False)


# Mandatory safety gates and eligibility.


def test_all_four_scopes_and_three_gates_are_required(contract: dict[str, Any]) -> None:
    results = runner.apply_mandatory_safety_gates(all_scopes(), contract)
    assert results["all_gates_pass"] is True
    assert len(results["gates"]) == 12
    assert {row["scope"] for row in results["gates"]} == set(runner.SAFETY_SCOPES)
    with pytest.raises(ValueError, match="scope coverage"):
        runner.apply_mandatory_safety_gates(all_scopes()[:-1], contract)


@pytest.mark.parametrize(
    "values",
    [
        safety_values(recall=0.79),
        safety_values(fpr=0.0125),
        safety_values(unsupported=0.79),
    ],
)
@pytest.mark.parametrize("scope", runner.SAFETY_SCOPES)
def test_any_failed_gate_in_any_scope_makes_candidate_ineligible(
    contract: dict[str, Any], scope: str, values: dict[str, float]
) -> None:
    results = runner.apply_mandatory_safety_gates(all_scopes({scope: values}), contract)
    eligible, reasons = runner.determine_candidate_eligibility(
        results,
        completed_group_folds=5,
        group_prediction_count=10,
        expected_group_prediction_count=10,
        fresh_prediction_count=4,
        expected_fresh_prediction_count=4,
        fresh_family_prediction_counts=[2, 2],
        expected_fresh_family_prediction_counts=[2, 2],
    )
    assert eligible is False
    assert len(reasons) == 1 and reasons[0].startswith(f"mandatory_safety_gate_failed:{scope}:")


def fresh_family_predictions(false_positive_count: int) -> tuple[list[str], list[str]]:
    gold = [intent for intent in runner.PRIMARY_INTENTS for _ in range(40)]
    predicted = list(gold)
    flipped = 0
    for index, intent in enumerate(gold):
        if intent == "account_blocked" and flipped < false_positive_count:
            predicted[index] = "freeze_card"
            flipped += 1
    return gold, predicted


@pytest.mark.parametrize(("false_positives", "passes"), [(1, True), (2, False)])
def test_fresh_family_fpr_integer_boundary(
    contract: dict[str, Any], false_positives: int, passes: bool
) -> None:
    gold, predicted = fresh_family_predictions(false_positives)
    metrics = runner.safety_metrics(gold, predicted)
    assert metrics["non_protected_gold_count"] == 160
    assert metrics["protected_gold_count"] == 160
    assert metrics["protected_false_positive_count"] == false_positives
    family = runner.FRESH_FAMILY_IDS[0]
    results = runner.apply_mandatory_safety_gates(all_scopes({family: metrics}), contract)
    fpr_gate = next(
        row
        for row in results["gates"]
        if row["scope"] == family and row["metric"] == "protected_false_positive_rate"
    )
    assert fpr_gate["passed"] is passes
    assert results["all_gates_pass"] is passes


# Prediction coverage.


def test_duplicate_and_incomplete_predictions_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate group-CV OOF prediction"):
        runner.validate_prediction_coverage(["a", "a", "b"], ["a", "b", "c"], "group-CV OOF")
    with pytest.raises(ValueError, match="incomplete group-CV OOF prediction coverage"):
        runner.validate_prediction_coverage(["a"], ["a", "b"], "group-CV OOF")


def test_incomplete_execution_is_an_infrastructure_failure(contract: dict[str, Any]) -> None:
    results = runner.apply_mandatory_safety_gates(all_scopes(), contract)
    with pytest.raises(RuntimeError, match="group_cv_prediction_coverage_exact"):
        runner.determine_candidate_eligibility(
            results,
            completed_group_folds=5,
            group_prediction_count=9,
            expected_group_prediction_count=10,
            fresh_prediction_count=4,
            expected_fresh_prediction_count=4,
            fresh_family_prediction_counts=[2, 2],
            expected_fresh_family_prediction_counts=[2, 2],
        )


def gold_fresh_rows(inputs: runner.LoadedInputs) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row["record_id"],
            "source_family_id": row["source_family_id"],
            "gold_intent": row["intent"],
            "primary_predicted_intent": row["intent"],
            "predicted_intent": row["intent"],
            **runner.NO_VERIFIER_DECISION,
        }
        for row in inputs.fresh_examples
    ]


def test_fresh_family_and_pooled_coverage_are_exact(real_inputs: runner.LoadedInputs) -> None:
    inputs = synthetic_inputs(real_inputs)
    rows = gold_fresh_rows(inputs)
    summary = runner.summarize_fresh(rows, inputs, gated=False)
    assert summary["pooled_prediction_count"] == 640
    assert [row["prediction_count"] for row in summary["families"]] == [320, 320]
    with pytest.raises(ValueError, match="incomplete pooled fresh prediction coverage"):
        runner.summarize_fresh(rows[:-1], inputs, gated=False)
    with pytest.raises(ValueError, match="duplicate pooled fresh prediction"):
        runner.summarize_fresh([*rows[:-1], rows[0]], inputs, gated=False)
    moved = deepcopy(rows)
    moved[0]["source_family_id"] = runner.FRESH_FAMILY_IDS[1]
    with pytest.raises(ValueError, match="differs from the frozen dataset"):
        runner.summarize_fresh(moved, inputs, gated=False)


# Lexicographic selection and stop rule.


def test_worst_family_metric_is_the_first_criterion(contract: dict[str, Any]) -> None:
    control = selection_result("HYBRID_CONTROL_R3", worst=0.70, pooled=0.90)
    gated = selection_result("HYBRID_PROTECTED_VERIFIER_R3", worst=0.71, pooled=0.70)
    selection = runner.select_candidate([control, gated], contract)
    assert selection["selected_candidate_id"] == "HYBRID_PROTECTED_VERIFIER_R3"


@pytest.mark.parametrize(
    ("field", "control_value", "gated_value", "winner"),
    [
        ("pooled", 0.75, 0.76, "HYBRID_PROTECTED_VERIFIER_R3"),
        ("group_macro", 0.81, 0.80, "HYBRID_CONTROL_R3"),
        ("protected_fpr", 0.006, 0.005, "HYBRID_PROTECTED_VERIFIER_R3"),
        ("protected_recall", 0.91, 0.90, "HYBRID_CONTROL_R3"),
        ("unsupported_recall", 0.90, 0.95, "HYBRID_PROTECTED_VERIFIER_R3"),
    ],
)
def test_exact_lexicographic_ordering(
    contract: dict[str, Any],
    field: str,
    control_value: float,
    gated_value: float,
    winner: str,
) -> None:
    control = selection_result("HYBRID_CONTROL_R3", **{field: control_value})
    gated = selection_result("HYBRID_PROTECTED_VERIFIER_R3", **{field: gated_value})
    assert runner.select_candidate([control, gated], contract)["selected_candidate_id"] == winner


def test_complexity_applies_only_after_earlier_criteria(contract: dict[str, Any]) -> None:
    control = selection_result("HYBRID_CONTROL_R3")
    gated = selection_result("HYBRID_PROTECTED_VERIFIER_R3")
    assert (
        runner.select_candidate([control, gated], contract)["selected_candidate_id"]
        == "HYBRID_CONTROL_R3"
    )
    within_tolerance = selection_result("HYBRID_PROTECTED_VERIFIER_R3", worst=0.70 + 1e-13)
    assert (
        runner.select_candidate([control, within_tolerance], contract)["selected_candidate_id"]
        == "HYBRID_CONTROL_R3"
    )
    better = selection_result("HYBRID_PROTECTED_VERIFIER_R3", unsupported_recall=0.91)
    assert (
        runner.select_candidate([control, better], contract)["selected_candidate_id"]
        == "HYBRID_PROTECTED_VERIFIER_R3"
    )


def test_ineligible_candidate_cannot_win(contract: dict[str, Any]) -> None:
    control = selection_result("HYBRID_CONTROL_R3", worst=0.50)
    gated = selection_result("HYBRID_PROTECTED_VERIFIER_R3", eligible=False, worst=0.99)
    selection = runner.select_candidate([control, gated], contract)
    assert selection["selected_candidate_id"] == "HYBRID_CONTROL_R3"
    assert selection["eligible_candidate_count"] == 1
    assert selection["next_required"] == "v2c6_candidate_freeze_before_final_holdout"
    assert selection["step29i_authorized"] is False
    assert selection["final_holdout_access_authorized"] is False


def test_no_eligible_candidate_is_no_acceptable_candidate(contract: dict[str, Any]) -> None:
    selection = runner.select_candidate(
        [
            selection_result("HYBRID_CONTROL_R3", eligible=False),
            selection_result("HYBRID_PROTECTED_VERIFIER_R3", eligible=False, worst=0.99),
        ],
        contract,
    )
    assert selection["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert selection["selected_candidate"] is None
    assert selection["selected_candidate_id"] is None
    assert selection["winner_forced"] is False
    assert selection["gates_weakened"] is False
    assert selection["next_required"] == "routing_architecture_fallback_decision"
    assert selection["step29i_authorized"] is False


# End-to-end synthetic execution with fake routers and verifiers.


def test_synthetic_run_success_path_and_fold_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_inputs: runner.LoadedInputs
) -> None:
    inputs, primary_calls, verifiers, payload = run_synthetic(
        monkeypatch, tmp_path, real_inputs
    )
    splits = runner.build_evaluation_splits(inputs)
    fresh_ids = {row["record_id"] for row in inputs.fresh_examples}
    development_ids = [row["example_id"] for row in inputs.development_examples]

    assert len(primary_calls) == 12
    for control_call, gated_call in zip(primary_calls[:6], primary_calls[6:], strict=True):
        assert control_call == gated_call
    for call, fold in zip(primary_calls[:5], splits.group_folds, strict=True):
        assert call["train_ids"] == [development_ids[index] for index in fold.train_indices]
        assert not set(call["train_ids"]) & set(call["evaluation_ids"])
    for call in primary_calls:
        assert not set(call["train_ids"]) & fresh_ids
    assert primary_calls[5]["train_ids"] == development_ids
    assert set(primary_calls[5]["evaluation_ids"]) == fresh_ids

    assert len(verifiers) == 24
    for fold_index, fold in enumerate(splits.group_folds):
        train_ids = {development_ids[index] for index in fold.train_indices}
        validation_ids = {development_ids[index] for index in fold.validation_indices}
        for verifier in verifiers[fold_index * 4 : fold_index * 4 + 4]:
            assert set(verifier.training_ids) <= train_ids
            assert not set(verifier.training_ids) & validation_ids
            assert not set(verifier.training_ids) & fresh_ids
    for verifier in verifiers[20:]:
        assert (verifier.positive_count, verifier.negative_count) == (60, 60)

    control, gated = payload["candidate_results"]
    assert control["eligible"] is False
    assert any("unsupported_recall" in reason for reason in control["ineligibility_reasons"])
    assert gated["ineligibility_reasons"] == [], gated["ineligibility_reasons"]
    assert gated["eligible"] is True
    gate_rows = gated["safety_gate_results"]["gates"]
    assert {row["scope"] for row in gate_rows} == set(runner.SAFETY_SCOPES)
    for row in gate_rows:
        assert row["passed"] is True, row
    scope_safety = dict(
        runner.safety_scope_metrics(gated["group_aware_cv"], gated["fresh_source_evaluation"])
    )
    for scope in runner.SAFETY_SCOPES:
        metrics = scope_safety[scope]
        assert metrics["protected_recall"] >= 0.80, (scope, metrics)
        assert metrics["protected_false_positive_rate"] <= 0.01, (scope, metrics)
        assert metrics["unsupported_recall"] >= 0.80, (scope, metrics)
    assert gated["fresh_source_evaluation"]["primary_router_fit_count"] == 1
    assert gated["fresh_source_evaluation"]["verifier_fit_count"] == 4
    assert gated["fresh_source_evaluation"]["refit_between_source_families"] is False
    diagnostics = gated["fresh_source_evaluation"]["pooled_metrics"]["verifier_diagnostics"]
    assert diagnostics["protected_false_positives_prevented_by_verifier"] == 80
    assert diagnostics["true_protected_requests_rejected_by_verifier"] == 0
    assert payload["selection"]["selection_status"] == "SELECTED"
    assert payload["selection"]["selected_candidate_id"] == "HYBRID_PROTECTED_VERIFIER_R3"
    assert payload["next_required"] == "v2c6_candidate_freeze_before_final_holdout"
    assert payload["primary_router_identity_audit"][
        "primary_router_predictions_identical_across_candidates"
    ] is True
    governance = payload["governance"]
    assert governance["model_fitting_performed"] is True
    assert governance["model_inference_performed"] is True
    for field in (
        "final_holdout_accessed",
        "threshold_tuning_performed",
        "calibration_performed",
        "runtime_behavior_changed",
        "step29i_authorized",
        "final_model_acceptance_claimed",
        "production_ready_claimed",
        "gates_weakened",
        "winner_forced",
    ):
        assert governance[field] is False


def test_synthetic_run_failure_path_honors_stop_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_inputs: runner.LoadedInputs
) -> None:
    _, _, _, payload = run_synthetic(monkeypatch, tmp_path, real_inputs, accept_all=True)
    assert [row["eligible"] for row in payload["candidate_results"]] == [False, False]
    assert payload["selection"]["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert payload["selection"]["selected_candidate"] is None
    assert payload["selection"]["winner_forced"] is False
    assert payload["selection"]["gates_weakened"] is False
    assert payload["next_required"] == "routing_architecture_fallback_decision"


def test_results_contain_no_raw_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_inputs: runner.LoadedInputs
) -> None:
    inputs, _, _, payload = run_synthetic(monkeypatch, tmp_path, real_inputs)
    content = runner.RESULTS_PATH.read_text(encoding="utf-8")
    for row in (*inputs.development_examples, *inputs.fresh_examples):
        assert row["text"] not in content
    leaked = deepcopy(payload)
    leaked["candidate_results"][0]["note"] = inputs.fresh_examples[0]["text"]
    with pytest.raises(ValueError, match="raw example text"):
        runner.validate_text_free(leaked, inputs.fresh_examples)


# Preflight, check-results, and create-once outputs.


def test_preflight_performs_no_embeddings_fitting_inference_or_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_inputs: runner.LoadedInputs
) -> None:
    inputs = synthetic_inputs(real_inputs)
    patch_outputs(monkeypatch, tmp_path)
    forbid_model_work(monkeypatch, "preflight")
    report = runner.preflight(inputs)
    assert report["embeddings_generated"] is False
    assert report["classifier_fitting_performed"] is False
    assert report["inference_performed"] is False
    assert report["model_selection_performed"] is False
    assert report["files_written"] is False
    assert report["final_holdout_accessed"] is False
    assert report["create_once_outputs_available"] is True
    assert report["expected_primary_router_fits"] == 12
    assert report["expected_verifier_fits"] == 24
    assert list(tmp_path.rglob("*")) == []


def test_check_results_validates_without_model_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_inputs: runner.LoadedInputs
) -> None:
    inputs, _, _, _ = run_synthetic(monkeypatch, tmp_path, real_inputs)
    forbid_model_work(monkeypatch, "check-results")
    report = runner.check_results(inputs)
    assert report["results_valid"] is True
    assert report["selection_status"] == "SELECTED"
    assert report["next_required"] == "v2c6_candidate_freeze_before_final_holdout"
    assert report["embeddings_generated"] is False
    assert report["classifier_fitting_performed"] is False
    assert report["inference_performed"] is False


def test_check_results_rejects_tampered_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_inputs: runner.LoadedInputs
) -> None:
    inputs, _, _, payload = run_synthetic(monkeypatch, tmp_path, real_inputs)
    tampered = deepcopy(payload)
    metrics = tampered["candidate_results"][1]["group_aware_cv"]["metrics"]
    metrics["macro_f1_16"] = metrics["macro_f1_16"] - 0.5
    runner.RESULTS_PATH.write_bytes(runner.stable_json_bytes(tampered))
    with pytest.raises(ValueError):
        runner.check_results(inputs)


@pytest.mark.parametrize("value", [None, 123, ["sha"]])
def test_non_string_bge_cache_lineage_is_a_type_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    real_inputs: runner.LoadedInputs,
    value: Any,
) -> None:
    inputs, _, _, payload = run_synthetic(monkeypatch, tmp_path, real_inputs)
    splits = runner.build_evaluation_splits(inputs)
    runner.validate_results_payload(payload, inputs, splits)
    malformed = deepcopy(payload)
    malformed["bge_cache_manifest_sha256"] = value
    with pytest.raises(TypeError, match="BGE cache lineage"):
        runner.validate_results_payload(malformed, inputs, splits)


def test_result_artifacts_are_create_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_inputs: runner.LoadedInputs
) -> None:
    inputs, _, _, _ = run_synthetic(monkeypatch, tmp_path, real_inputs)
    before = runner.RESULTS_PATH.read_bytes()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        runner.run_model_selection(inputs)
    assert runner.RESULTS_PATH.read_bytes() == before
    with pytest.raises(FileExistsError):
        runner.write_bytes_create_once(runner.RESULTS_PATH, b"{}\n")
    assert runner.RESULTS_PATH.read_bytes() == before


def test_incompatible_existing_outputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_inputs: runner.LoadedInputs
) -> None:
    inputs = synthetic_inputs(real_inputs)
    patch_outputs(monkeypatch, tmp_path)
    forbid_model_work(monkeypatch, "incompatible-output run")
    runner.RESULTS_PATH.parent.mkdir(parents=True)
    runner.RESULTS_PATH.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        runner.run_model_selection(inputs)
    with pytest.raises(FileNotFoundError, match="both R3 result artifacts"):
        runner.check_results(inputs)
    assert runner.preflight(inputs)["status"] == "RESULT_ARTIFACTS_ALREADY_PRESENT"


def test_reserved_paths_and_historical_results_are_distinct() -> None:
    assert runner.RESULTS_RELATIVE_PATH == (
        "data/evals/v2/ml/v2c6_r3_protected_intent_gate_model_selection_results.json"
    )
    assert runner.BGE_CACHE_RELATIVE_PATH.startswith("data/evals/v2/ml/local/v2c6_r3_")
    assert "targeted_remediation" not in runner.RESULTS_RELATIVE_PATH


# Source inspection.


def test_no_grid_search_balanced_weighting_calibration_or_threshold_selection() -> None:
    sources = {
        "r3": inspect.getsource(runner),
        "reused_r2": (ROOT / runner.R2_RUNNER_RELATIVE_PATH).read_text(encoding="utf-8"),
    }
    for name, source in sources.items():
        for prohibited in (
            "GridSearchCV",
            "RandomizedSearchCV",
            "ParameterGrid",
            "CalibratedClassifierCV",
            "predict_proba",
            "decision_function",
            "precision_recall_curve",
            "roc_curve",
            'class_weight="balanced"',
            "class_weight='balanced'",
        ):
            assert prohibited not in source, f"{prohibited} found in {name}"
    r3_source = sources["r3"]
    assert "balanced" not in r3_source
    assert "joblib" not in r3_source
    assert "pickle.dump" not in r3_source


def test_r3_runner_has_no_raw_exec_or_compile_loading_path() -> None:
    source = inspect.getsource(runner)
    assert "exec(" not in source
    assert "compile(" not in source
    assert "spec.loader.exec_module(module)" in source
