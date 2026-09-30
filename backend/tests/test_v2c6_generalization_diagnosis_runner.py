from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import run_v2c6_generalization_diagnosis as runner

ROOT = Path(__file__).resolve().parents[2]


def example(
    example_id: str,
    intent: str,
    text: str,
    group_id: str,
    *,
    risk: str = "PRIVATE_READ",
    source_id: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "example_id": example_id,
        "group_id": group_id,
        "intent": intent,
        "risk": risk,
        "text": text,
    }
    if source_id is not None:
        record["source_id"] = source_id
    return record


@pytest.fixture
def small_examples() -> list[dict[str, Any]]:
    return [
        example("e1", "alpha", "Pay NOW", "g1", source_id="external"),
        example("e2", "alpha", "Pay NOW", "g1", source_id="external"),
        example("e3", "beta", "  PAY   NOW ", "g2", source_id="authored"),
        example("e4", "beta", "different words", "g3", source_id="authored"),
        example("e5", "beta", "last thing", "g4", source_id="authored"),
    ]


def sixteen_labels() -> list[str]:
    return [
        "account_balance",
        "account_blocked",
        "cancel_transfer",
        "card_status",
        "close_account",
        "create_dispute",
        "escalation",
        "freeze_card",
        "informational_policy",
        "lost_or_stolen_phone",
        "passcode_recovery",
        "recent_transactions",
        "transaction_details",
        "transfer_failed_or_declined",
        "transfer_pending",
        "unsupported_or_uncertain",
    ]


def synthetic_oof_candidate() -> tuple[dict[str, Any], list[str], dict[str, int]]:
    labels = sixteen_labels()
    unsupported = "unsupported_or_uncertain"
    counts = {label: 10 for label in labels}
    counts[unsupported] = 8048
    matrix = [[0 for _ in labels] for _ in labels]
    index = {label: position for position, label in enumerate(labels)}
    for label in labels:
        matrix[index[label]][index[label]] = counts[label]

    matrix[index["cancel_transfer"]] = [0 for _ in labels]
    matrix[index["cancel_transfer"]][index["cancel_transfer"]] = 5
    matrix[index["cancel_transfer"]][index["close_account"]] = 2
    matrix[index["cancel_transfer"]][index["account_balance"]] = 2
    matrix[index["cancel_transfer"]][index[unsupported]] = 1

    matrix[index["account_balance"]] = [0 for _ in labels]
    matrix[index["account_balance"]][index["account_balance"]] = 8
    matrix[index["account_balance"]][index["freeze_card"]] = 2

    matrix[index[unsupported]] = [0 for _ in labels]
    matrix[index[unsupported]][index[unsupported]] = 8040
    matrix[index[unsupported]][index["cancel_transfer"]] = 3
    matrix[index[unsupported]][index["account_balance"]] = 5

    per_intent = {
        label: {
            "f1": 0.8,
            "precision": 0.75,
            "recall": 0.9,
            "support": counts[label],
        }
        for label in labels
    }
    candidate = {
        "aggregate_metrics": {
            "accuracy": 0.91,
            "balanced_accuracy": 0.82,
            "confusion_matrix": {"label_order": labels, "values": matrix},
            "historical_nine_label_macro_f1": 0.88,
            "mean_fold_macro_f1": 0.86,
            "new_seven_intent_macro_f1": 0.77,
            "per_intent": per_intent,
            "pooled_oof_macro_f1": 0.84,
            "worst_fold_macro_f1": 0.8,
        },
        "fold_metrics": [
            {
                "accuracy": 0.80 + fold / 100,
                "macro_f1": 0.70 + fold / 100,
                "validation_fold": fold,
            }
            for fold in range(5)
        ],
        "safety_metrics": {
            "exact_protected_write_recall": 0.85,
            "protected_write_false_positive_rate": 0.01,
            "unsupported_or_uncertain_recall": 0.9,
        },
    }
    return candidate, labels, counts


def aggregate_contract() -> dict[str, Any]:
    return {
        "allowed_remediation_recommendations": [
            "improve_training_data_diversity",
            "rebalance_development_data",
            "improve_group_or_source_independence",
            "improve_intent_definitions_or_boundaries",
            "hard_negative_generation",
            "unsupported_class_redesign",
            "reconsider_representation_or_classifier_family",
            "taxonomy_revision",
        ],
        "final_result_context": {
            "development_pooled_oof_macro_f1": 0.8850476526181243,
            "final_exact_protected_write_recall": 0.4,
            "final_macro_f1": 0.5632279946795209,
            "final_new_seven_intent_macro_f1": 0.45253940739110227,
            "final_protected_write_false_positive_rate": 0.014583333333333334,
            "final_unsupported_or_uncertain_recall": 0.85,
        },
    }


def test_real_frozen_lineage_is_8198_records_and_never_reads_consumed_holdout() -> None:
    opened: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        assert path.resolve() != runner.DEFAULT_PATHS.prohibited_holdout.resolve()
        opened.append(path.resolve())
        return path.read_bytes()

    inputs = runner.load_preconditions(
        reader=recording_reader,
        require_outputs_absent=False,
    )

    counts = Counter(record["intent"] for record in inputs["examples"])
    assert len(inputs["examples"]) == 8198
    assert sum(counts.values()) == 8198
    assert dict(counts) == inputs["development_manifest"]["counts"][
        "final_intent_counts"
    ]
    assert runner.DEFAULT_PATHS.prohibited_holdout.resolve() not in opened


def test_preflight_never_opens_consumed_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        assert path.resolve() != runner.DEFAULT_PATHS.prohibited_holdout.resolve()
        opened.append(path.resolve())
        return path.read_bytes()

    monkeypatch.setattr(runner, "ensure_outputs_absent", lambda paths: None)
    report = runner.preflight(reader=recording_reader)

    assert report["development_record_count"] == 8198
    assert report["raw_v2c5_holdout_accessed"] is False
    assert runner.DEFAULT_PATHS.prohibited_holdout.resolve() not in opened


def test_explicit_holdout_guard_runs_before_reader() -> None:
    called = False

    def forbidden_reader(path: Path) -> bytes:
        nonlocal called
        called = True
        return b""

    with pytest.raises(PermissionError, match="holdout access is prohibited"):
        runner.guarded_read_bytes(
            runner.DEFAULT_PATHS.prohibited_holdout,
            runner.DEFAULT_PATHS,
            forbidden_reader,
        )
    assert called is False


def test_run_never_opens_consumed_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = replace(
        runner.DEFAULT_PATHS,
        repository_root=tmp_path,
        prohibited_holdout=tmp_path / "v2c5_final_holdout.json",
        result=tmp_path / "diagnosis.json",
        result_manifest=tmp_path / "diagnosis.manifest.json",
        runner=tmp_path / "runner.py",
    )
    paths.runner.write_text("# runner\n", encoding="utf-8")
    inputs = {"contract": aggregate_contract(), "lineage": {"source": "frozen"}}
    payload = {
        "governance": {**runner.GOVERNANCE_FLAGS, "diagnosis_completed": True},
        "source_lineage": inputs["lineage"],
    }
    opened: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        assert path.resolve() != paths.prohibited_holdout.resolve()
        opened.append(path.resolve())
        return path.read_bytes()

    monkeypatch.setattr(
        runner,
        "load_preconditions",
        lambda supplied_paths, reader: inputs,
    )
    monkeypatch.setattr(runner, "build_diagnosis_payload", lambda value: payload)
    monkeypatch.setattr(runner, "validate_result_payload", lambda value, source: None)

    runner.run_diagnosis(paths, reader=recording_reader)

    assert paths.result.is_file()
    assert paths.result_manifest.is_file()
    assert paths.prohibited_holdout.resolve() not in opened


def test_check_results_never_opens_holdout_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = replace(
        runner.DEFAULT_PATHS,
        repository_root=tmp_path,
        prohibited_holdout=tmp_path / "v2c5_final_holdout.json",
        result=tmp_path / "diagnosis.json",
        result_manifest=tmp_path / "diagnosis.manifest.json",
        runner=tmp_path / "runner.py",
    )
    paths.runner.write_text("# runner\n", encoding="utf-8")
    inputs = {"contract": aggregate_contract(), "lineage": {"source": "frozen"}}
    payload = {
        "governance": {**runner.GOVERNANCE_FLAGS, "diagnosis_completed": True},
        "source_lineage": inputs["lineage"],
    }
    result_bytes = runner.stable_json_bytes(payload)
    manifest = runner.build_manifest(result_bytes, payload, paths, runner.filesystem_reader)
    paths.result.write_bytes(result_bytes)
    paths.result_manifest.write_bytes(runner.stable_json_bytes(manifest))
    before = {
        path: path.read_bytes() for path in (paths.result, paths.result_manifest)
    }
    opened: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        assert path.resolve() != paths.prohibited_holdout.resolve()
        opened.append(path.resolve())
        return path.read_bytes()

    monkeypatch.setattr(
        runner,
        "load_preconditions",
        lambda supplied_paths, reader, require_outputs_absent: inputs,
    )
    monkeypatch.setattr(runner, "build_diagnosis_payload", lambda value: payload)
    monkeypatch.setattr(runner, "validate_result_payload", lambda value, source: None)

    status = runner.check_results(paths, reader=recording_reader)

    assert status["files_written"] is False
    assert before == {
        path: path.read_bytes() for path in (paths.result, paths.result_manifest)
    }
    assert paths.prohibited_holdout.resolve() not in opened

    manifest["result"]["sha256"] = "0" * 64
    paths.result_manifest.write_bytes(runner.stable_json_bytes(manifest))
    with pytest.raises(ValueError, match="manifest hash, lineage, or governance"):
        runner.check_results(paths, reader=recording_reader)


def test_group_structure_and_concentration(small_examples: list[dict[str, Any]]) -> None:
    provenance = runner.metadata_concentration(small_examples, ["alpha", "beta"])
    composition = runner.development_source_composition(
        small_examples, ["alpha", "beta"], provenance
    )
    result = runner.group_structure(small_examples, ["alpha", "beta"])

    assert composition["total_record_count"] == 5
    assert composition["intent_counts"] == {"alpha": 2, "beta": 3}
    assert composition["intent_percentages"] == {"alpha": 0.4, "beta": 0.6}
    assert composition["total_unique_group_count"] == 4
    assert composition["examples_per_group_distribution"]["mean"] == 1.25
    assert result["total_unique_groups"] == 4
    assert result["group_size_distribution"] == {
        "count": 4,
        "max": 2,
        "mean": 1.25,
        "median": 1.0,
        "min": 1,
        "p90": pytest.approx(1.7),
        "p95": pytest.approx(1.85),
        "percentile_method": "linear_interpolation_index_n_minus_1",
    }
    assert result["singleton_group_count"] == 3
    assert result["groups_with_more_than_one_record_count"] == 1
    assert result["multi_intent_group_count"] == 0
    assert result["share_in_top_group_percentages"]["1_percent"]["record_share"] == 0.4
    assert result["share_in_top_group_percentages"]["5_percent"]["record_share"] == 0.4
    assert result["share_in_top_group_percentages"]["10_percent"]["record_share"] == 0.4


def test_multi_intent_group_aggregation() -> None:
    records = [
        example("e1", "alpha", "one", "shared"),
        example("e2", "beta", "two", "shared"),
        example("e3", "beta", "three", "solo"),
    ]
    result = runner.group_structure(records, ["alpha", "beta"])

    assert result["multi_intent_group_count"] == 1
    assert result["records_in_multi_intent_groups"] == 2
    assert result["multi_intent_group_intent_count_distribution"]["mean"] == 2
    assert result["raw_rows_are_not_independent_groups"] is True


def test_normalization_and_duplicate_structure(
    small_examples: list[dict[str, Any]],
) -> None:
    assert runner.normalize_text("  PＡY\tNow  ") == "pay now"
    result = runner.duplicate_structure(small_examples, ["alpha", "beta"])

    assert result["exact_duplicate_record_count"] == 1
    assert result["exact_duplicate_cluster_count"] == 1
    assert result["normalized_duplicate_record_count"] == 2
    assert result["normalized_duplicate_cluster_count"] == 1
    assert result["within_intent_exact_duplicate_cluster_count"] == 1
    assert result["cross_intent_exact_duplicate_conflict_cluster_count"] == 0
    assert result["cross_intent_normalized_duplicate_conflict_cluster_count"] == 1
    assert result["records_in_cross_intent_duplicate_conflicts"] == 3
    assert result["raw_duplicate_text_persisted"] is False
    assert "Pay NOW" not in json.dumps(result)


def test_regex_tokenization_and_lexical_metrics() -> None:
    records = [
        example("e1", "alpha", "Can't pay pay.", "g1"),
        example("e2", "alpha", "Pay now!", "g2"),
    ]
    assert runner.tokenize("  CAN’T pay_2 now! ") == ["can’t", "pay", "2", "now"]

    result = runner.lexical_scope(records)

    assert result["character_length_distribution"]["min"] == 8
    assert result["token_length_distribution"]["min"] == 2
    assert result["token_length_distribution"]["max"] == 3
    assert result["token_statistics"]["total_token_count"] == 5
    assert result["token_statistics"]["unique_token_count"] == 3
    assert result["token_statistics"]["type_token_ratio"] == 0.6
    assert result["token_statistics"]["hapax_token_count"] == 2
    assert result["token_statistics"]["hapax_rate"] == pytest.approx(2 / 3)
    assert result["ngram_concentration"]["unigrams"]["share_top_10"] == 1.0
    assert result["ngram_concentration"]["bigrams"]["total_count"] == 3
    assert result["ngram_concentration"]["trigrams"]["total_count"] == 1


def test_class_imbalance_and_risk_distribution() -> None:
    labels = [
        "informational_policy",
        "unsupported_or_uncertain",
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
        "account_balance",
    ]
    records = [
        example(f"e{index}", label, label, f"g{index}", risk=risk)
        for index, (label, risk) in enumerate(
            [
                ("informational_policy", "PUBLIC"),
                ("unsupported_or_uncertain", "PRIVATE_READ"),
                ("unsupported_or_uncertain", "PRIVATE_READ"),
                ("unsupported_or_uncertain", "PRIVATE_READ"),
                ("cancel_transfer", "PROTECTED_WRITE"),
                ("close_account", "PROTECTED_WRITE"),
                ("create_dispute", "PROTECTED_WRITE"),
                ("freeze_card", "PROTECTED_WRITE"),
                ("account_balance", "PRIVATE_READ"),
            ]
        )
    ]
    protected = ["cancel_transfer", "close_account", "create_dispute", "freeze_card"]
    result = runner.class_imbalance(records, labels, protected)

    assert result["unsupported_or_uncertain"] == {"count": 3, "share": 1 / 3}
    assert result["protected_write_combined"] == {"count": 4, "share": 4 / 9}
    assert result["largest_to_smallest_class_ratio"] == 3
    assert result["risk_levels"]["PROTECTED_WRITE"]["count"] == 4


def test_existing_oof_metrics_and_derived_confusions() -> None:
    candidate, labels, counts = synthetic_oof_candidate()
    protected = ["cancel_transfer", "close_account", "create_dispute", "freeze_card"]
    result = runner.existing_oof_behavior(candidate, labels, protected, counts)

    assert result["pooled_oof_macro_f1"] == 0.84
    assert result["per_intent"] is candidate["aggregate_metrics"]["per_intent"]
    unsupported = result["unsupported_prediction_behavior"]
    assert unsupported["total_unsupported_predictions"] == 8041
    assert unsupported["unsupported_prediction_rate"] == pytest.approx(8041 / 8198)
    assert unsupported["gold_non_unsupported_to_unsupported_count"] == 1
    assert unsupported["gold_unsupported_to_non_unsupported_count"] == 8
    protected_result = result["protected_write_confusion"]
    assert protected_result["gold_protected_predicted_exact_protected_intent"] == 35
    assert protected_result["gold_protected_predicted_another_protected_intent"] == 2
    assert protected_result["gold_protected_predicted_non_protected_intent"] == 2
    assert protected_result["gold_protected_predicted_unsupported_or_uncertain"] == 1
    assert protected_result["non_protected_predicted_any_protected_intent"] == 5


def test_fold_variance_uses_existing_fold_metrics() -> None:
    candidate, _, _ = synthetic_oof_candidate()
    result = runner.fold_variance(candidate)

    assert result["macro_f1"]["values"] == pytest.approx([0.70, 0.71, 0.72, 0.73, 0.74])
    assert result["macro_f1"]["mean"] == pytest.approx(0.72)
    assert result["macro_f1"]["spread"] == pytest.approx(0.04)
    assert result["macro_f1"]["population_standard_deviation"] == pytest.approx(
        0.01414213562373095
    )


def test_provenance_present_and_missing_are_reported(
    small_examples: list[dict[str, Any]],
) -> None:
    result = runner.metadata_concentration(small_examples, ["alpha", "beta"])

    assert result["source_id"]["available"] is True
    assert result["source_id"]["category_count"] == 2
    assert result["source_id"]["largest_category_share"] == 0.6
    assert result["source_id"]["distribution_across_intents"]["alpha"][
        "category_count"
    ] == 1
    assert result["authoring_method"] == {
        "available": False,
        "reason": "metadata_not_present",
    }


def test_intent_boundaries_include_confusion_rates_and_lexical_overlap() -> None:
    candidate, labels, counts = synthetic_oof_candidate()
    protected = ["cancel_transfer", "close_account", "create_dispute", "freeze_card"]
    oof = runner.existing_oof_behavior(candidate, labels, protected, counts)
    focus = [
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
        "account_blocked",
        "transfer_failed_or_declined",
        "transfer_pending",
        "unsupported_or_uncertain",
    ]
    records = [
        example(f"e{index}", label, f"shared token {label}", f"g{index}")
        for index, label in enumerate(focus)
    ]
    result = runner.intent_boundary_diagnostics(records, labels, oof, focus)

    assert result["target_to_unsupported_or_uncertain"]["cancel_transfer"] == {
        "count": 1,
        "denominator": 10,
        "rate": 0.1,
    }
    assert result["unsupported_or_uncertain_to_target"]["cancel_transfer"][
        "count"
    ] == 3
    pair = next(
        item
        for item in result["pairwise_diagnostics"]
        if item["first_intent"] == "cancel_transfer"
        and item["second_intent"] == "close_account"
    )
    assert pair["first_to_second_oof_confusion_count"] == 2
    assert pair["first_to_second_oof_confusion_rate"] == 0.2
    assert pair["token_set_overlap"]["intersection_size"] == 2
    assert pair["token_set_overlap"]["jaccard"] == pytest.approx(1 / 3)


def test_effective_sample_size_is_labeled_as_a_proxy(
    small_examples: list[dict[str, Any]],
) -> None:
    result = runner.effective_sample_size(small_examples, ["alpha", "beta"])

    assert result["global"]["raw_record_count"] == 5
    assert result["global"]["unique_group_count"] == 4
    assert result["global"]["unique_exact_text_count"] == 4
    assert result["global"]["unique_normalized_text_count"] == 3
    assert result["global"]["unique_group_to_raw_record_ratio"] == 0.8
    assert result["section_name"] == "evidence_independence_proxies"
    assert "not formal" in result["limitation"]


def test_historical_context_is_exact_and_aggregate_only() -> None:
    result = runner.historical_aggregate_context(aggregate_contract())

    assert result["development_pooled_oof_macro_f1"] == 0.8850476526181243
    assert result["final_macro_f1"] == 0.5632279946795209
    assert result["development_to_final_macro_f1_gap"] == pytest.approx(
        0.3218196579386034
    )
    assert result["final_exact_protected_write_recall"] == 0.4
    assert result["final_protected_write_false_positive_rate"] == (
        0.014583333333333334
    )
    assert result["final_unsupported_or_uncertain_recall"] == 0.85
    assert result["individual_final_holdout_examples_used"] is False
    assert not (runner.recursive_keys(result) & {"examples", "text", "utterance"})


def test_claim_policy_requires_evidence_and_keeps_causality_empty() -> None:
    payload = {
        "metric": {"value": 1},
        "diagnostic_claims": {
            "causal_claims": [],
            "hypotheses": [
                {
                    "evidence_metric_refs": ["metric.value"],
                    "statement": "This may matter.",
                    "unproven": True,
                }
            ],
            "observations": [
                {
                    "evidence_metric_refs": ["metric.value"],
                    "statement": "The value is one.",
                }
            ],
            "supported_diagnoses": [
                {
                    "confidence": "medium",
                    "evidence_metric_refs": ["metric.value"],
                    "limitation": "This is observational.",
                    "statement": "A measured condition exists.",
                }
            ],
        },
        "remediation_recommendations": [
            {
                "category": "rebalance_development_data",
                "evidence_metric_refs": ["metric.value"],
                "priority": "medium",
                "what_must_be_frozen_before_implementation": "A new contract.",
                "why": "The measured value motivates a controlled experiment.",
            }
        ],
    }
    runner.validate_claims_and_recommendations(payload, aggregate_contract())

    payload["diagnostic_claims"]["supported_diagnoses"][0][
        "evidence_metric_refs"
    ] = []
    with pytest.raises(ValueError, match="lacks required evidence"):
        runner.validate_claims_and_recommendations(payload, aggregate_contract())


def test_result_validation_requires_intent_counts_to_reconcile() -> None:
    inputs = {
        "contract": aggregate_contract(),
        "labels": ["alpha", "beta"],
        "lineage": {"development_dataset_sha256": "d" * 64},
    }
    payload = {
        "development_source_composition": {
            "intent_counts": {"alpha": 4000, "beta": 4198},
            "total_record_count": 8198,
        },
        "diagnosis_completed": True,
        "diagnostic_claims": {
            "causal_claims": [],
            "hypotheses": [],
            "observations": [],
            "supported_diagnoses": [],
        },
        "duplicate_structure": {"raw_duplicate_text_persisted": False},
        "governance": {**runner.GOVERNANCE_FLAGS, "diagnosis_completed": True},
        "historical_aggregate_context": runner.historical_aggregate_context(
            inputs["contract"]
        ),
        "next_required": "v2c6_remediation_design",
        "phase": "V2-C6 Step 29B",
        "remediation_recommendations": [],
        "schema_version": runner.RESULT_SCHEMA_VERSION,
        "selected_candidate_id": runner.SELECTED_CANDIDATE_ID,
        "source_lineage": inputs["lineage"],
    }
    runner.validate_result_payload(payload, inputs)

    payload["development_source_composition"]["intent_counts"]["beta"] = 4197
    with pytest.raises(ValueError, match="counts do not reconcile"):
        runner.validate_result_payload(payload, inputs)


def test_recommendations_are_frozen_categories_and_do_not_implement_remediation() -> None:
    contract = aggregate_contract()
    recommendations = runner.remediation_recommendations(contract)

    assert {item["category"] for item in recommendations} <= set(
        contract["allowed_remediation_recommendations"]
    )
    assert all(item["what_must_be_frozen_before_implementation"] for item in recommendations)
    assert runner.GOVERNANCE_FLAGS["remediation_implemented"] is False


def test_no_model_fit_prediction_or_embedding_api_exists_in_runner() -> None:
    source = (ROOT / runner.SCRIPT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert ".predict(" not in source
    assert ".fit(" not in source
    assert "fastembed" not in source.lower()
    assert "sentence_transform" not in source.lower()


def test_governance_and_cli_are_exact() -> None:
    assert runner.GOVERNANCE_FLAGS == {
        "embeddings_generated": False,
        "model_inference_performed": False,
        "model_selection_performed": False,
        "model_training_performed": False,
        "raw_v2c5_holdout_accessed": False,
        "remediation_implemented": False,
        "runtime_behavior_changed": False,
        "taxonomy_changed": False,
        "threshold_tuning_performed": False,
    }
    parser = runner.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    assert options == {"--preflight", "--run", "--check-results"}


def test_stable_serialization_and_manifest_hash(tmp_path: Path) -> None:
    paths = replace(
        runner.DEFAULT_PATHS,
        repository_root=tmp_path,
        result=tmp_path / "diagnosis.json",
        runner=tmp_path / "runner.py",
    )
    paths.runner.write_text("# deterministic runner\n", encoding="utf-8")
    payload = {
        "governance": {**runner.GOVERNANCE_FLAGS, "diagnosis_completed": True},
        "source_lineage": {"development_dataset_sha256": "d" * 64},
    }

    first = runner.stable_json_bytes({"z": 1, "a": 2})
    second = runner.stable_json_bytes({"a": 2, "z": 1})
    manifest = runner.build_manifest(
        runner.stable_json_bytes(payload),
        payload,
        paths,
        runner.filesystem_reader,
    )

    assert first == second
    assert manifest["result"]["sha256"] == runner.sha256_bytes(
        runner.stable_json_bytes(payload)
    )
    assert manifest["runner"]["sha256"] == runner.sha256_file(
        paths.runner, paths
    )
