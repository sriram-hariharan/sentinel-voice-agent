from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import run_v2c6_post_selection_failure_analysis as runner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(runner.DEFAULT_PATHS.contract.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_inputs() -> runner.AnalysisInputs:
    return runner.load_preconditions(
        runner.DEFAULT_PATHS,
        require_outputs_absent=False,
    )


@pytest.fixture(scope="module")
def analysis_payload(real_inputs: runner.AnalysisInputs) -> dict[str, Any]:
    return runner.build_results(real_inputs)


def prediction(
    example_id: str,
    gold: str,
    predicted: str,
    *,
    family: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "example_id": example_id,
        "gold_intent": gold,
        "predicted_intent": predicted,
    }
    if family is not None:
        row["held_out_source_family_id"] = family
    return row


def safety_rows() -> list[dict[str, Any]]:
    return [
        prediction("p1", "cancel_transfer", "freeze_card"),
        prediction("p2", "close_account", "account_balance"),
        prediction("n1", "account_balance", "freeze_card"),
        prediction("n2", "card_status", "card_status"),
        prediction(
            "u1",
            "unsupported_or_uncertain",
            "create_dispute",
        ),
        prediction(
            "u2",
            "unsupported_or_uncertain",
            "account_balance",
        ),
        prediction(
            "u3",
            "unsupported_or_uncertain",
            "unsupported_or_uncertain",
        ),
    ]


def test_exact_frozen_contract_is_required(
    real_inputs: runner.AnalysisInputs,
) -> None:
    assert real_inputs.contract_sha256 == runner.EXPECTED_CONTRACT_SHA256
    assert real_inputs.contract["schema_version"] == runner.CONTRACT_SCHEMA_VERSION
    assert real_inputs.contract["phase"] == "V2-C6 Step 29H-A"


def test_exact_step29h_result_identity_is_required(
    real_inputs: runner.AnalysisInputs,
) -> None:
    specification = real_inputs.contract["source_artifacts"]["step29h_results"]

    assert runner.sha256_file(
        runner.DEFAULT_PATHS.step29h_results,
        runner.DEFAULT_PATHS,
    ) == specification["sha256"]
    assert real_inputs.results["schema_version"] == specification["schema_version"]


def test_exact_step29h_manifest_identity_is_required(
    real_inputs: runner.AnalysisInputs,
) -> None:
    specification = real_inputs.contract["source_artifacts"][
        "step29h_results_manifest"
    ]

    assert runner.sha256_file(
        runner.DEFAULT_PATHS.step29h_manifest,
        runner.DEFAULT_PATHS,
    ) == specification["sha256"]
    assert real_inputs.results_manifest["results"]["sha256"] == (
        real_inputs.contract["source_artifacts"]["step29h_results"]["sha256"]
    )


def test_exact_frozen_dataset_identity_is_required(
    real_inputs: runner.AnalysisInputs,
) -> None:
    specification = real_inputs.contract["source_artifacts"][
        "development_dataset"
    ]

    assert len(real_inputs.records_by_id) == 9008
    assert runner.sha256_file(
        runner.DEFAULT_PATHS.development_dataset,
        runner.DEFAULT_PATHS,
    ) == specification["sha256"]


def test_no_acceptable_candidate_state_is_required(
    real_inputs: runner.AnalysisInputs,
) -> None:
    selection = real_inputs.results["selection"]

    assert selection["selection_status"] == "NO_ACCEPTABLE_CANDIDATE"
    assert selection["selected_candidate"] is None
    assert selection["winner_forced"] is False
    assert selection["gates_weakened"] is False


def test_six_completed_ineligible_candidates_are_required(
    real_inputs: runner.AnalysisInputs,
) -> None:
    candidates = real_inputs.results["candidate_results"]

    assert len(candidates) == 6
    assert all(row["execution_status"] == "COMPLETED" for row in candidates)
    assert all(row["eligible"] is False for row in candidates)
    assert real_inputs.results["selection"]["eligible_candidate_count"] == 0


def test_step29i_is_blocked(contract: dict[str, Any]) -> None:
    assert contract["contract_status"]["step29i_authorized"] is False
    assert contract["motivation"]["step29i_blocked"] is True


def test_exact_protected_intents(contract: dict[str, Any]) -> None:
    assert contract["protected_false_positive_analysis"]["protected_intents"] == [
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    ]


def test_exact_three_source_families(contract: dict[str, Any]) -> None:
    assert contract["source_family_analysis"]["required_family_ids"] == [
        "v2c6_sf1_definition_direct",
        "v2c6_sf2_scenario_narrative",
        "v2c6_sf3_boundary_conversational",
    ]


def test_group_cv_prediction_coverage_is_exact(
    real_inputs: runner.AnalysisInputs,
) -> None:
    coverage = real_inputs.coverage

    assert coverage["group_cv_predictions_per_candidate"] == 9008
    assert coverage["group_cv_prediction_count_total"] == 54048
    assert coverage["duplicate_prediction_count"] == 0
    assert coverage["missing_prediction_count"] == 0


def test_source_family_prediction_coverage_is_exact(
    real_inputs: runner.AnalysisInputs,
) -> None:
    coverage = real_inputs.coverage

    assert coverage["source_family_predictions_per_candidate"] == 810
    assert coverage["source_family_prediction_count_total"] == 4860
    assert coverage["source_family_predictions_per_candidate_and_family"] == 270
    assert coverage["candidate_family_scope_count"] == 18


def test_duplicate_predictions_are_rejected() -> None:
    records = {
        "a": {"example_id": "a", "intent": "account_balance"},
        "b": {"example_id": "b", "intent": "card_status"},
    }
    rows = [
        prediction("a", "account_balance", "account_balance"),
        prediction("a", "account_balance", "card_status"),
    ]

    with pytest.raises(ValueError, match="duplicate prediction"):
        runner.validate_prediction_rows(
            rows,
            expected_ids={"a", "b"},
            records=records,
            candidate_id="candidate",
            scope="scope",
        )


def test_missing_predictions_are_rejected() -> None:
    records = {
        "a": {"example_id": "a", "intent": "account_balance"},
        "b": {"example_id": "b", "intent": "card_status"},
    }

    with pytest.raises(ValueError, match="prediction count mismatch"):
        runner.validate_prediction_rows(
            [prediction("a", "account_balance", "account_balance")],
            expected_ids={"a", "b"},
            records=records,
            candidate_id="candidate",
            scope="scope",
        )


def test_protected_false_positive_calculation_is_exact(
    contract: dict[str, Any],
) -> None:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    summary = runner.protected_fp_summary(safety_rows(), protected)

    assert summary["non_protected_gold_count"] == 5
    assert summary["protected_false_positive_count"] == 2
    assert summary["protected_false_positive_rate"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("denominator", "expected"),
    [(1, 0), (99, 0), (100, 1), (150, 1), (450, 4), (8392, 83)],
)
def test_maximum_passing_fp_count_is_integer_derived(
    denominator: int,
    expected: int,
) -> None:
    assert runner.maximum_passing_false_positive_count(denominator) == expected


def test_excess_false_positive_count_is_exact(
    contract: dict[str, Any],
) -> None:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    rows = [
        prediction("p", "cancel_transfer", "cancel_transfer"),
        prediction("u", "unsupported_or_uncertain", "unsupported_or_uncertain"),
    ]
    rows.extend(
        prediction(f"n{index}", "account_balance", "freeze_card")
        if index < 3
        else prediction(f"n{index}", "account_balance", "account_balance")
        for index in range(200)
    )

    summary = runner.protected_fp_summary(rows, protected)

    assert summary["maximum_passing_false_positive_count"] == 2
    assert summary["excess_false_positive_count"] == 1


def test_non_protected_to_protected_confusion_table(
    contract: dict[str, Any],
) -> None:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    rows = runner.protected_fp_confusions(
        safety_rows(),
        protected,
        candidate_id="candidate",
        evaluation_scope="pooled_group_aware_cv",
        source_family_id=None,
    )

    observed = {
        (row["gold_intent"], row["predicted_protected_intent"])
        for row in rows
    }
    assert observed == {
        ("account_balance", "freeze_card"),
        ("unsupported_or_uncertain", "create_dispute"),
    }
    assert all(row["gold_intent_support"] >= row["count"] for row in rows)
    assert all(
        row["rate_among_gold_intent"] == row["rate_within_gold_intent"]
        for row in rows
    )


def test_protected_confusion_sorting_is_deterministic(
    contract: dict[str, Any],
) -> None:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    rows = safety_rows() + [
        prediction("n3", "account_balance", "freeze_card"),
        prediction("n4", "card_status", "create_dispute"),
    ]
    first = runner.protected_fp_confusions(
        rows,
        protected,
        candidate_id="candidate",
        evaluation_scope="scope",
        source_family_id=None,
    )
    second = runner.protected_fp_confusions(
        list(reversed(rows)),
        protected,
        candidate_id="candidate",
        evaluation_scope="scope",
        source_family_id=None,
    )

    assert first == second
    assert first[0]["count"] == 2


def test_protected_to_different_protected_is_not_recall_miss(
    contract: dict[str, Any],
) -> None:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    result = runner.protected_recall_analysis(
        safety_rows(),
        protected,
        candidate_id="candidate",
        evaluation_scope="scope",
        source_family_id=None,
    )

    assert result["protected_recall_miss_count"] == 1
    assert result[
        "protected_to_different_protected_exact_intent_error_count"
    ] == 1
    exact = result["protected_to_different_protected_exact_intent_error_rows"]
    assert exact[0]["protected_recall_failure"] is False


def test_protected_to_non_protected_is_recall_miss(
    contract: dict[str, Any],
) -> None:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    result = runner.protected_recall_analysis(
        safety_rows(),
        protected,
        candidate_id="candidate",
        evaluation_scope="scope",
        source_family_id=None,
    )

    misses = result["protected_recall_miss_rows"]
    assert result["protected_recall"] == pytest.approx(0.5)
    assert misses[0]["gold_protected_intent"] == "close_account"
    assert misses[0]["predicted_non_protected_intent"] == "account_balance"
    assert misses[0]["protected_recall_failure"] is True


def test_unsupported_error_types_are_separated(contract: dict[str, Any]) -> None:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    result = runner.unsupported_analysis(
        safety_rows(),
        protected,
        candidate_id="candidate",
        evaluation_scope="scope",
        source_family_id=None,
    )

    assert result["unsupported_to_protected_count"] == 1
    assert result["unsupported_to_other_supported_count"] == 1
    by_intent = {row["predicted_intent"]: row for row in result["confusion_rows"]}
    assert by_intent["create_dispute"]["protected_prediction"] is True
    assert by_intent["account_balance"]["protected_prediction"] is False


def test_unsupported_recall_is_exact(contract: dict[str, Any]) -> None:
    protected = contract["protected_false_positive_analysis"]["protected_intents"]
    result = runner.unsupported_analysis(
        safety_rows(),
        protected,
        candidate_id="candidate",
        evaluation_scope="scope",
        source_family_id=None,
    )

    assert result["unsupported_denominator"] == 3
    assert result["unsupported_correct_count"] == 1
    assert result["unsupported_recall"] == pytest.approx(1 / 3)


def test_source_family_summary_matches_frozen_metrics(
    real_inputs: runner.AnalysisInputs,
) -> None:
    candidate_id = real_inputs.contract["candidate_coverage"]["candidate_ids"][0]
    family = real_inputs.contract["source_family_analysis"][
        "required_family_ids"
    ][0]
    candidate = real_inputs.candidate_results[candidate_id]
    rows = runner.prediction_scopes(candidate, real_inputs.contract)[family]
    summary = runner.source_family_summary(
        candidate,
        rows,
        real_inputs.contract,
        candidate_id=candidate_id,
        family=family,
    )
    stored = runner.stored_family_metrics(candidate, family)

    assert summary["record_count"] == 270
    assert summary["primary_8_macro_f1"] == stored["primary_8_macro_f1"]
    assert summary["protected_recall"] == stored["safety"]["protected_recall"]
    assert summary["protected_false_positive_rate"] == stored["safety"][
        "protected_false_positive_rate"
    ]


def test_group_source_delta_signs_are_correct(
    real_inputs: runner.AnalysisInputs,
) -> None:
    candidate_id = "WORD_CHAR_TFIDF_LINEAR_SVC__C=1.0__class_weight=none"
    result = runner.group_source_delta(
        real_inputs.candidate_results[candidate_id],
        candidate_id,
        real_inputs.contract,
    )
    deltas = result["signed_safety_deltas"]

    assert deltas["protected_false_positive_rate"] == pytest.approx(
        0.03333333333333333 - 0.0077454718779790275
    )
    assert deltas["protected_recall"] == pytest.approx(0.8444444444444444 - 0.875)
    assert deltas["unsupported_recall"] == pytest.approx(
        0.85 - 0.9062436855930491
    )


def test_macro_f1_comparability_warning_is_preserved(
    real_inputs: runner.AnalysisInputs,
) -> None:
    candidate_id = real_inputs.contract["candidate_coverage"]["candidate_ids"][0]
    result = runner.group_source_delta(
        real_inputs.candidate_results[candidate_id],
        candidate_id,
        real_inputs.contract,
    )

    assert result["macro_f1_comparability_warning"] == real_inputs.contract[
        "group_cv_vs_source_shift_delta_analysis"
    ]["macro_f1_comparison"]["qualification_required"]
    assert result["causal_interpretation_claimed"] is False


def test_exact_three_class_weight_comparisons(
    real_inputs: runner.AnalysisInputs,
) -> None:
    comparisons = runner.class_weight_comparisons(real_inputs)

    assert len(comparisons) == 3
    assert all(
        row["delta_direction"]
        == "balanced_candidate_minus_matched_unweighted_candidate"
        for row in comparisons
    )
    assert all(row["descriptive_only"] is True for row in comparisons)


def test_exact_two_bge_regularization_comparisons(
    real_inputs: runner.AnalysisInputs,
) -> None:
    comparisons = runner.regularization_comparisons(real_inputs)

    assert len(comparisons) == 2
    assert all(
        row["delta_direction"] == "c1_candidate_minus_matched_c4_candidate"
        for row in comparisons
    )
    assert all(row["winner_selected"] is False for row in comparisons)


def test_exact_ten_hard_negative_pairs(contract: dict[str, Any]) -> None:
    pairs = contract["hard_negative_analysis"]["required_pairs"]

    assert len(pairs) == 10
    assert len({tuple(pair) for pair in pairs}) == 10


def test_hard_negative_aggregation_is_exact() -> None:
    rows = [
        {
            "pair": ["a", "b"],
            "candidate_id": "c1",
            "source_family_id": "f1",
            "record_count": 5,
            "correct_count": 3,
            "A_to_B_count": 1,
            "B_to_A_count": 0,
            "other_error_count": 1,
        },
        {
            "pair": ["a", "b"],
            "candidate_id": "c1",
            "source_family_id": "f2",
            "record_count": 5,
            "correct_count": 4,
            "A_to_B_count": 0,
            "B_to_A_count": 1,
            "other_error_count": 0,
        },
    ]

    aggregate = runner.aggregate_hard_negative_rows(
        rows,
        ["candidate_id", "pair"],
    )[0]

    assert aggregate["record_count"] == 10
    assert aggregate["correct_count"] == 7
    assert aggregate["accuracy"] == pytest.approx(0.7)
    assert aggregate["true_a_predicted_b_count"] == 1
    assert aggregate["true_b_predicted_a_count"] == 1


def synthetic_recurrence_inputs(contract: dict[str, Any]) -> runner.AnalysisInputs:
    families = contract["source_family_analysis"]["required_family_ids"]
    candidates: dict[str, dict[str, Any]] = {}
    for candidate_id in contract["candidate_coverage"]["candidate_ids"]:
        group = [prediction("group", "account_balance", "freeze_card")]
        source = [
            prediction(
                f"source-{index}",
                "account_balance",
                "freeze_card",
                family=family,
            )
            for index, family in enumerate(families)
        ]
        candidates[candidate_id] = {
            "group_aware_cv": {"predictions": group},
            "source_family_holdout": {"pooled_predictions": source},
        }
    return runner.AnalysisInputs(
        contract=contract,
        contract_sha256="a" * 64,
        results={},
        results_manifest={},
        step29g_contract={},
        freeze={},
        dataset={},
        records_by_id={},
        remediation_ids=frozenset(),
        candidate_results=candidates,
        coverage={},
    )


def test_cross_candidate_and_representation_recurrence(
    contract: dict[str, Any],
) -> None:
    rows = runner.recurrence_rows(
        synthetic_recurrence_inputs(contract),
        error_kind="protected_false_positive",
    )

    assert len(rows) == 1
    assert rows[0]["candidate_count_with_confusion"] == 6
    assert rows[0]["bge_candidate_count_with_confusion"] == 4
    assert rows[0]["tfidf_candidate_count_with_confusion"] == 2
    assert rows[0]["gold_non_protected_intent"] == "account_balance"
    assert rows[0]["predicted_protected_intent"] == "freeze_card"


def test_cross_family_recurrence(contract: dict[str, Any]) -> None:
    rows = runner.recurrence_rows(
        synthetic_recurrence_inputs(contract),
        error_kind="protected_false_positive",
    )

    assert rows[0]["source_family_count_with_confusion"] == 3
    assert rows[0]["per_source_family_counts"] == {
        "v2c6_sf1_definition_direct": 6,
        "v2c6_sf2_scenario_narrative": 6,
        "v2c6_sf3_boundary_conversational": 6,
    }
    assert rows[0]["observed_in_group_cv"] is True
    assert rows[0]["observed_in_source_family_evaluation"] is True


@pytest.mark.parametrize(
    ("targeted", "representation", "expected"),
    [
        ([], [], "insufficient_evidence"),
        (["boundary"], [], "targeted_data_boundary_evidence"),
        ([], ["comparison"], "representation_change_evidence"),
        (["boundary"], ["comparison"], "both"),
    ],
)
def test_remediation_classifier_follows_frozen_rule(
    contract: dict[str, Any],
    targeted: list[str],
    representation: list[str],
    expected: str,
) -> None:
    result = runner.classify_remediation_evidence(
        contract,
        targeted_data_evidence_links=targeted,
        representation_evidence_links=representation,
    )

    assert result["classification"] == expected
    assert result["automatic_numerical_thresholds_used"] is False
    assert result["remediation_selected"] is False
    assert result["remediation_implemented"] is False


def test_analysis_performs_no_remediation_action(
    analysis_payload: dict[str, Any],
) -> None:
    governance = analysis_payload["governance"]

    assert governance["remediation_selected"] is False
    assert governance["remediation_implemented"] is False
    assert analysis_payload["candidate_coverage"]["winner_selected"] is False


def test_tracked_results_are_text_free(
    analysis_payload: dict[str, Any],
    real_inputs: runner.AnalysisInputs,
) -> None:
    runner.validate_text_free(analysis_payload, real_inputs.records_by_id)
    serialized = json.dumps(analysis_payload, sort_keys=True)

    assert '"text"' not in serialized
    assert '"raw_text"' not in serialized


def test_optional_local_review_is_separate_and_ignored(
    contract: dict[str, Any],
) -> None:
    definition = contract["error_record_schema"]["optional_local_text_review"]
    parser = runner.build_parser()

    assert definition["path"].startswith("data/evals/v2/ml/local/")
    assert definition["tracked"] is False
    assert "data/evals/v2/ml/local/" in (ROOT / ".gitignore").read_text(
        encoding="utf-8"
    )
    assert parser.parse_args(["--write-local-review"]).write_local_review is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--run", "--write-local-review"])


def test_final_holdout_guard_rejects_before_read() -> None:
    with pytest.raises(PermissionError, match="final holdout access is prohibited"):
        runner.read_bytes(
            runner.DEFAULT_PATHS.prohibited_holdout,
            runner.DEFAULT_PATHS,
        )


def test_no_model_or_embedding_imports_or_execution_paths() -> None:
    source = inspect.getsource(runner)

    assert "fastembed" not in source.lower()
    assert "sklearn" not in source.lower()
    assert "LinearSVC" not in source
    assert ".fit(" not in source
    assert ".predict(" not in source
    assert ".transform(" not in source
    assert ".encode(" not in source
    assert ".embed(" not in source


def test_no_threshold_tuning_or_mutation(
    analysis_payload: dict[str, Any],
) -> None:
    governance = analysis_payload["governance"]

    assert governance["threshold_tuning_performed"] is False
    assert governance["dataset_mutated"] is False
    assert governance["label_mutated"] is False
    assert governance["taxonomy_mutated"] is False
    assert governance["runtime_behavior_changed"] is False


def test_consumed_evidence_governance_is_emitted(
    analysis_payload: dict[str, Any],
) -> None:
    governance = analysis_payload["consumed_evidence_governance"]

    assert governance[
        "source_family_evidence_consumed_for_development_model_selection"
    ] is True
    assert governance[
        "diagnostic_error_patterns_consumed_by_this_analysis"
    ] is True
    assert governance["same_families_are_fresh_after_analysis"] is False


def test_same_families_are_nonfresh_for_future_validation(
    analysis_payload: dict[str, Any],
) -> None:
    governance = analysis_payload["consumed_evidence_governance"]

    assert governance[
        "same_source_families_may_be_called_fresh_after_remediation"
    ] is False
    assert governance["post_remediation_reuse_permitted_only_as"] == [
        "diagnostic_regression_evidence",
        "previously_observed_development_evidence",
    ]
    assert governance["future_clean_source_generalization_evidence"][
        "fresh_evidence_required"
    ] is True


def test_stable_serialization_is_deterministic() -> None:
    first = runner.stable_json_bytes({"b": 2, "a": [3, 1]})
    second = runner.stable_json_bytes({"a": [3, 1], "b": 2})

    assert first == second
    assert runner.sha256_bytes(first) == runner.sha256_bytes(second)


def test_create_once_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"

    runner.durable_create(path, b"first\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.durable_create(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_contract_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    changed_contract = tmp_path / "contract.json"
    changed_contract.write_text("{}\n", encoding="utf-8")
    paths = replace(runner.DEFAULT_PATHS, contract=changed_contract)

    with pytest.raises(ValueError, match="contract SHA-256 mismatch"):
        runner.load_preconditions(paths, require_outputs_absent=False)


def test_source_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    paths = replace(runner.DEFAULT_PATHS, repository_root=tmp_path)

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        runner.validate_binding(
            artifact,
            {"path": "artifact.json", "sha256": "0" * 64},
            paths,
            "artifact",
        )


def test_exact_post_analysis_next_required_is_fail_closed(
    analysis_payload: dict[str, Any],
) -> None:
    assert analysis_payload["next_required"] is None
    assert analysis_payload["next_required_contract_ambiguity"]
    assert analysis_payload["governance"]["step29i_authorized"] is False


def test_preflight_reports_no_execution_or_writes(
    real_inputs: runner.AnalysisInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "load_preconditions",
        lambda *_args, **_kwargs: real_inputs,
    )

    result = runner.preflight()

    assert result["failure_analysis_performed"] is False
    assert result["model_fitting_performed"] is False
    assert result["model_inference_performed"] is False
    assert result["embeddings_generated"] is False
    assert result["files_written"] is False
    assert result["step29i_authorized"] is False
