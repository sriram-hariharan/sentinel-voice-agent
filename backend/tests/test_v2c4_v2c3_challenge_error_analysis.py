from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import analyze_v2c3_challenge_errors as analysis

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "data/evals/v2/ml/v2c4_experiment_contract.json"
ERROR_CONFIG_PATH = ROOT / "data/evals/v2/ml/v2c4_error_analysis_config.json"
FINAL_REPORT_PATH = ROOT / "data/evals/v2/ml/v2c3_final_evaluation_report.json"
DIAGNOSTIC_REPORT_PATH = (
    ROOT / "data/evals/v2/ml/v2c4_error_analysis_report.json"
)
SCRIPT_PATH = ROOT / "scripts/analyze_v2c3_challenge_errors.py"

RISK_BY_INTENT = {
    "account_balance": "PRIVATE_READ",
    "card_status": "PRIVATE_READ",
    "create_dispute": "PROTECTED_WRITE",
    "escalation": "ESCALATION_OR_UNCERTAIN",
    "freeze_card": "PROTECTED_WRITE",
    "informational_policy": "PUBLIC",
    "recent_transactions": "PRIVATE_READ",
    "transaction_details": "PRIVATE_READ",
    "unsupported_or_uncertain": "ESCALATION_OR_UNCERTAIN",
}
SCORE_CLASS_ORDER = (
    "freeze_card",
    "unsupported_or_uncertain",
    "account_balance",
    "create_dispute",
    "card_status",
    "escalation",
    "informational_policy",
    "recent_transactions",
    "transaction_details",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def example(
    ordinal: int,
    intent: str,
    *,
    text: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "data_role": "final_challenge_evaluation",
        "example_id": f"fixture:{ordinal:03d}",
        "group_id": f"fixture-group:{ordinal:03d}",
        "intent": intent,
        "risk": RISK_BY_INTENT[intent],
        "tags": tags or [],
        "text": text or f"Synthetic fixture {ordinal}",
        "text_sha256": f"{ordinal:064x}",
        "normalized_text_sha256": f"{ordinal + 1000:064x}",
    }


@pytest.fixture
def synthetic_case() -> tuple[list[dict[str, Any]], list[str]]:
    rows = [
        example(
            1,
            "freeze_card",
            tags=["explicit_current_action", "shared"],
        ),
        example(
            2,
            "freeze_card",
            tags=["correction", "explicit_current_action", "shared"],
        ),
        example(3, "freeze_card"),
        example(4, "create_dispute"),
        example(5, "unsupported_or_uncertain"),
        example(6, "unsupported_or_uncertain", tags=["no_current_request"]),
        example(
            7,
            "unsupported_or_uncertain",
            tags=["protected_action_information"],
        ),
        example(8, "account_balance"),
        example(9, "account_balance", tags=["contrast"]),
        example(10, "account_balance"),
    ]
    predictions = [
        "freeze_card",
        "account_balance",
        "create_dispute",
        "create_dispute",
        "unsupported_or_uncertain",
        "account_balance",
        "freeze_card",
        "unsupported_or_uncertain",
        "freeze_card",
        "account_balance",
    ]
    return rows, predictions


@pytest.fixture
def synthetic_analysis(
    synthetic_case: tuple[list[dict[str, Any]], list[str]],
) -> dict[str, Any]:
    rows, predictions = synthetic_case
    return analysis.analyze_predictions(
        rows,
        predictions,
        analysis.EXPECTED_INTENTS,
        RISK_BY_INTENT,
        sorted(analysis.FROZEN_CATEGORY_IDS),
    )


@pytest.fixture
def synthetic_decision_analysis(
    synthetic_case: tuple[list[dict[str, Any]], list[str]],
    synthetic_analysis: dict[str, Any],
) -> dict[str, Any]:
    rows, predictions = synthetic_case
    score_rows = []
    for row, predicted in zip(rows, predictions, strict=True):
        scores = {
            intent: -1.0 - (index / 10)
            for index, intent in enumerate(SCORE_CLASS_ORDER)
        }
        scores[row["intent"]] = 0.25
        scores[predicted] = 1.0
        score_rows.append([scores[intent] for intent in SCORE_CLASS_ORDER])
    return analysis.analyze_decision_boundaries(
        rows,
        predictions,
        score_rows,
        SCORE_CLASS_ORDER,
        synthetic_analysis,
    )


def test_challenge_role_is_consumed_diagnostic_only() -> None:
    contract = load_json(CONTRACT_PATH)
    policy = contract["consumed_data_policy"]["v2c3_challenge_set"]
    assert policy["diagnostic_role"] == "consumed_v2c3_safety_challenge"
    assert policy["v2c4_final_acceptance_eligible"] is False
    assert policy["model_selection_eligible"] is False
    assert policy["threshold_selection_eligible"] is False
    assert policy["may_be_described_as_untouched"] is False


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [
        (
            "unsupported_or_uncertain",
            "account_balance",
            ["unsupported_to_supported"],
        ),
        (
            "account_balance",
            "unsupported_or_uncertain",
            ["supported_to_unsupported"],
        ),
        (
            "freeze_card",
            "account_balance",
            ["protected_to_nonprotected"],
        ),
        (
            "freeze_card",
            "create_dispute",
            [
                "protected_to_wrong_protected",
                "freeze_dispute_cross_confusion",
            ],
        ),
        (
            "account_balance",
            "freeze_card",
            ["nonprotected_to_protected"],
        ),
        (
            "unsupported_or_uncertain",
            "freeze_card",
            ["unsupported_to_supported", "nonprotected_to_protected"],
        ),
        ("account_balance", "account_balance", []),
    ],
)
def test_mechanical_bucket_definitions(
    gold: str, predicted: str, expected: list[str]
) -> None:
    assert analysis.mechanical_buckets(gold, predicted) == expected


def test_risk_mapping_is_applied_and_validated(
    synthetic_case: tuple[list[dict[str, Any]], list[str]],
    synthetic_analysis: dict[str, Any],
) -> None:
    errors = {
        row["example_id"]: row for row in synthetic_analysis["error_records"]
    }
    assert errors["fixture:007"]["gold_risk"] == "ESCALATION_OR_UNCERTAIN"
    assert errors["fixture:007"]["predicted_risk"] == "PROTECTED_WRITE"

    rows, predictions = synthetic_case
    changed = [dict(row) for row in rows]
    changed[0]["risk"] = "PUBLIC"
    with pytest.raises(ValueError, match="gold risk mismatch"):
        analysis.analyze_predictions(
            changed,
            predictions,
            analysis.EXPECTED_INTENTS,
            RISK_BY_INTENT,
            sorted(analysis.FROZEN_CATEGORY_IDS),
        )


def test_protected_write_calculations(
    synthetic_analysis: dict[str, Any],
) -> None:
    protected = synthetic_analysis["protected_safety"]
    assert protected == {
        "protected_total": 4,
        "protected_exact_intent_correct": 2,
        "protected_correctly_recognized_as_protected": 3,
        "protected_miss_count": 2,
        "protected_to_nonprotected_count": 1,
        "protected_to_wrong_protected_intent_count": 1,
        "protected_write_recall": 0.5,
        "nonprotected_total": 6,
        "nonprotected_to_freeze_card": 2,
        "nonprotected_to_create_dispute": 0,
        "protected_false_positive_count": 2,
        "protected_write_false_positive_rate": 2 / 6,
    }


def test_unsupported_calculations(synthetic_analysis: dict[str, Any]) -> None:
    unsupported = synthetic_analysis["unsupported_analysis"]
    assert unsupported["total"] == 3
    assert unsupported["correctly_unsupported"] == 1
    assert unsupported["predicted_supported"] == 2
    assert unsupported["recall"] == 1 / 3
    assert unsupported["predicted_intent_distribution"]["account_balance"] == 1
    assert unsupported["predicted_intent_distribution"]["freeze_card"] == 1
    assert unsupported["predicted_intent_distribution"][
        "unsupported_or_uncertain"
    ] == 1


def test_full_confusion_matrix_uses_frozen_intent_order(
    synthetic_analysis: dict[str, Any],
) -> None:
    matrix = synthetic_analysis["confusion_matrix"]
    assert matrix["label_order"] == list(analysis.EXPECTED_INTENTS)
    assert len(matrix["values"]) == 9
    assert all(len(row) == 9 for row in matrix["values"])
    assert matrix["gold_to_predicted"]["freeze_card"] == {
        "account_balance": 1,
        "card_status": 0,
        "create_dispute": 1,
        "escalation": 0,
        "freeze_card": 1,
        "informational_policy": 0,
        "recent_transactions": 0,
        "transaction_details": 0,
        "unsupported_or_uncertain": 0,
    }


def test_class_score_mapping_rank_top_three_and_margin(
    synthetic_decision_analysis: dict[str, Any],
) -> None:
    assert synthetic_decision_analysis["classifier_class_order"] == list(
        SCORE_CLASS_ORDER
    )
    errors = {
        row["example_id"]: row
        for row in synthetic_decision_analysis["per_error"]
    }
    protected_miss = errors["fixture:002"]
    assert protected_miss["gold_intent"] == "freeze_card"
    assert protected_miss["predicted_intent"] == "account_balance"
    assert protected_miss["gold_class_decision_score"] == 0.25
    assert protected_miss["predicted_class_decision_score"] == 1.0
    assert protected_miss["decision_margin"] == 0.75
    assert protected_miss["gold_class_rank"] == 2
    assert protected_miss["top_3_predictions"] == [
        {"intent": "account_balance", "decision_score": 1.0, "rank": 1},
        {"intent": "freeze_card", "decision_score": 0.25, "rank": 2},
        {
            "intent": "unsupported_or_uncertain",
            "decision_score": -1.1,
            "rank": 3,
        },
    ]


def test_rank_ties_retain_classifier_class_order() -> None:
    assert analysis.rank_class_scores(
        ["freeze_card", "account_balance", "create_dispute"],
        [0.5, 0.5, 0.1],
    ) == [
        {"intent": "freeze_card", "decision_score": 0.5, "rank": 1},
        {"intent": "account_balance", "decision_score": 0.5, "rank": 2},
        {"intent": "create_dispute", "decision_score": 0.1, "rank": 3},
    ]


def test_protected_vs_nonprotected_gap_and_best_intents(
    synthetic_decision_analysis: dict[str, Any],
) -> None:
    protected = {
        row["example_id"]: row
        for row in synthetic_decision_analysis["protected_analysis"]["per_example"]
    }
    miss = protected["fixture:002"]
    assert miss["best_protected_intent"] == "freeze_card"
    assert miss["max_protected_score"] == 0.25
    assert miss["best_nonprotected_intent"] == "account_balance"
    assert miss["max_nonprotected_score"] == 1.0
    assert miss["protected_vs_nonprotected_gap"] == -0.75


def test_tag_analysis_aggregates_all_existing_tags(
    synthetic_decision_analysis: dict[str, Any],
) -> None:
    tags = synthetic_decision_analysis["tag_analysis"]
    assert tags["explicit_current_action"] == {
        "example_count": 2,
        "correct_count": 1,
        "error_count": 1,
        "protected_example_count": 2,
        "protected_miss_count": 1,
        "protected_false_positive_count": 0,
        "unsupported_example_count": 0,
        "unsupported_false_supported_count": 0,
        "error_rate": 0.5,
    }
    assert tags["contrast"]["protected_false_positive_count"] == 1
    assert tags["no_current_request"]["unsupported_example_count"] == 1
    assert tags["no_current_request"][
        "unsupported_false_supported_count"
    ] == 1
    assert set(tags) == {
        "contrast",
        "correction",
        "explicit_current_action",
        "no_current_request",
        "protected_action_information",
        "shared",
    }


def test_existing_diagnostic_preserves_frozen_safety_counts() -> None:
    report = load_json(DIAGNOSTIC_REPORT_PATH)
    metrics = report["diagnostic_regression_metrics"]
    assert metrics["overall"] == {
        "accuracy": 0.7518518518518519,
        "correct_count": 203,
        "error_count": 67,
        "example_count": 270,
    }
    assert metrics["protected_safety"]["protected_miss_count"] == 15
    assert metrics["protected_safety"]["protected_false_positive_count"] == 4
    assert metrics["unsupported_analysis"]["predicted_supported"] == 9


def test_review_queue_deduplicates_overlapping_reasons(
    synthetic_analysis: dict[str, Any],
) -> None:
    queue = synthetic_analysis["safety_critical_review_queue"]
    assert len(queue) == 5
    assert len({row["example_id"] for row in queue}) == 5
    overlap = next(row for row in queue if row["example_id"] == "fixture:007")
    assert overlap["review_reasons"] == [
        "protected_false_positive",
        "unsupported_false_supported",
    ]
    assert synthetic_analysis["review_queue_summary"] == {
        "unique_example_count": 5,
        "reason_counts": {
            "protected_false_positive": 2,
            "protected_miss": 2,
            "unsupported_false_supported": 2,
        },
    }


@pytest.mark.parametrize(
    "filename",
    ["v2c4_safety_holdout.json", "v2c4_safety_holdout_seed.json"],
)
def test_v2c4_holdout_paths_are_rejected(filename: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed V2-C4 final holdout"):
        analysis.guard_input_path(tmp_path / filename)


def test_v2c4_final_data_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="sealed V2-C4 final data role"):
        analysis.guard_data_role(
            {"data_role": "v2c4_final_safety_holdout"},
            label="fixture",
        )


def test_analyzer_contains_no_fitting_or_training_api() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
            "fit",
            "fit_predict",
            "fit_transform",
            "partial_fit",
            "train",
        }
    )
    assert "GridSearchCV" not in source
    assert "RandomizedSearchCV" not in source
    assert "ParameterGrid" not in source


def test_frozen_categories_are_loaded_without_replacement() -> None:
    config = load_json(ERROR_CONFIG_PATH)
    category_ids = analysis.load_frozen_category_ids(config)
    assert set(category_ids) == analysis.FROZEN_CATEGORY_IDS
    assert len(category_ids) == 9
    assert ROOT / config["results_path"] == analysis.DEFAULT_OUTPUT_PATH


def test_semantic_causes_are_not_invented_from_keywords() -> None:
    row = example(
        1,
        "unsupported_or_uncertain",
        text="I heard freeze and dispute but have no request.",
    )
    result = analysis.analyze_predictions(
        [row],
        ["freeze_card"],
        analysis.EXPECTED_INTENTS,
        RISK_BY_INTENT,
        sorted(analysis.FROZEN_CATEGORY_IDS),
    )
    error = result["error_records"][0]
    assert error["automatic_causal_claims"] == []
    assert error["semantic_review_candidates"] == []
    assert error["semantic_review_required"] is True


def test_existing_final_report_has_no_reusable_record_predictions() -> None:
    report = load_json(FINAL_REPORT_PATH)
    example_ids = ["fixture:001", "fixture:002"]
    assert analysis.extract_record_level_predictions(report, example_ids) is None


def test_deterministic_report_generation_from_synthetic_fixture(
    synthetic_analysis: dict[str, Any],
    synthetic_decision_analysis: dict[str, Any],
) -> None:
    kwargs = {
        "analysis": synthetic_analysis,
        "decision_boundary_analysis": synthetic_decision_analysis,
        "prediction_source": "synthetic_fixture",
        "frozen_category_ids": sorted(analysis.FROZEN_CATEGORY_IDS),
        "input_hashes": {"fixture": {"sha256": "0" * 64}},
    }
    first = analysis.build_report_payload(**kwargs)
    second = analysis.build_report_payload(**kwargs)
    assert analysis.stable_json_bytes(first) == analysis.stable_json_bytes(second)
    assert first["analysis_role"] == "consumed_v2c3_diagnostic"
    assert first["final_holdout"] is False
    assert first["model_selection_evidence"] is False
    assert first["v2c4_final_acceptance_evidence"] is False
    assert first["raw_text_persisted"] is False
    assert first["execution_policy"]["model_fitting_performed"] is False
    assert first["execution_policy"]["v2c4_final_holdout_accessed"] is False


def test_report_records_persist_hashes_but_no_raw_text(
    synthetic_analysis: dict[str, Any],
    synthetic_decision_analysis: dict[str, Any],
) -> None:
    report = analysis.build_report_payload(
        analysis=synthetic_analysis,
        decision_boundary_analysis=synthetic_decision_analysis,
        prediction_source="synthetic_fixture",
        frozen_category_ids=sorted(analysis.FROZEN_CATEGORY_IDS),
        input_hashes={"fixture": {"sha256": "0" * 64}},
    )
    analysis.validate_report_contains_no_raw_text(report)
    assert report["raw_text_persisted"] is False
    forbidden = {"text", "raw_text", "utterance"}
    for section in ("error_records", "safety_critical_review_queue"):
        assert report[section]
        for row in report[section]:
            assert forbidden.isdisjoint(row)
            assert len(row["text_sha256"]) == 64
            assert len(row["normalized_text_sha256"]) == 64


def test_print_review_mode_joins_ids_to_synthetic_consumed_challenge(
    synthetic_case: tuple[list[dict[str, Any]], list[str]],
    synthetic_analysis: dict[str, Any],
    synthetic_decision_analysis: dict[str, Any],
) -> None:
    rows, _ = synthetic_case
    report = analysis.build_report_payload(
        analysis=synthetic_analysis,
        decision_boundary_analysis=synthetic_decision_analysis,
        prediction_source="synthetic_fixture",
        frozen_category_ids=sorted(analysis.FROZEN_CATEGORY_IDS),
        input_hashes={"fixture": {"sha256": "0" * 64}},
    )
    joined = analysis.join_review_queue_with_challenge(report, rows)
    assert len(joined) == 5
    assert joined[0]["text"].startswith("Synthetic fixture")
    assert {row["example_id"] for row in joined} == {
        row["example_id"]
        for row in report["safety_critical_review_queue"]
    }


def test_print_review_mode_does_not_infer_or_write(
    synthetic_case: tuple[list[dict[str, Any]], list[str]],
    synthetic_analysis: dict[str, Any],
    synthetic_decision_analysis: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, _ = synthetic_case
    report = analysis.build_report_payload(
        analysis=synthetic_analysis,
        decision_boundary_analysis=synthetic_decision_analysis,
        prediction_source="synthetic_fixture",
        frozen_category_ids=sorted(analysis.FROZEN_CATEGORY_IDS),
        input_hashes={"fixture": {"sha256": "0" * 64}},
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("print-review mode attempted inference or a write")

    monkeypatch.setattr(analysis, "run_diagnostic", fail)
    monkeypatch.setattr(analysis, "write_report", fail)
    monkeypatch.setattr(analysis.final_evaluation, "embed_final_texts", fail)
    output = io.StringIO()
    joined = analysis.print_review_queue(
        report_payload=report,
        challenge_examples=rows,
        stream=output,
    )
    rendered = json.loads(output.getvalue())
    assert rendered["record_count"] == len(joined) == 5
    assert rendered["records"][0]["text"].startswith("Synthetic fixture")


def test_print_review_mode_requires_an_existing_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="diagnostic report is required"):
        analysis.load_review_inputs(tmp_path / "missing-report.json")
