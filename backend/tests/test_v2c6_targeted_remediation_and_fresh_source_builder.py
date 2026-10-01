from __future__ import annotations

import ast
import copy
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import (
    build_v2c6_targeted_remediation_and_fresh_source_data as builder,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sources() -> builder.FrozenSources:
    return builder.load_frozen_sources()


@pytest.fixture
def templates(
    sources: builder.FrozenSources,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return builder.create_authoring_templates(sources.contract)


def temporary_paths(tmp_path: Path) -> builder.BuildPaths:
    return replace(
        builder.DEFAULT_PATHS,
        training_workfile=tmp_path / "local/training.json",
        evaluation_workfile=tmp_path / "local/evaluation.json",
        training_output=tmp_path / "training.json",
        training_manifest=tmp_path / "training.manifest.json",
        combined_output=tmp_path / "combined.json",
        combined_manifest=tmp_path / "combined.manifest.json",
        evaluation_output=tmp_path / "evaluation.json",
        evaluation_manifest=tmp_path / "evaluation.manifest.json",
        builder=ROOT / builder.SCRIPT_RELATIVE_PATH,
    )


def completed_workfile(payload: dict[str, Any], label: str) -> dict[str, Any]:
    completed = copy.deepcopy(payload)
    completed["authoring_complete"] = True
    for record in completed["records"]:
        record["text"] = (
            f"Independent synthetic {label} wording for {record['record_id']}."
        )
        record["authoring_provenance"]["authoring_method"] = "human_authored"
        record["review_method"] = "human_review"
        record["review_notes"] = "Approved synthetic fixture."
        record["review_status"] = "approved"
        record["reviewer"] = "Synthetic test reviewer"
    return completed


def completed_inputs(
    sources: builder.FrozenSources,
    tmp_path: Path,
) -> tuple[builder.BuildPaths, builder.AuthoringInputs]:
    paths = temporary_paths(tmp_path)
    training, evaluation = builder.create_authoring_templates(sources.contract)
    training = completed_workfile(training, "training")
    evaluation = completed_workfile(evaluation, "evaluation")
    paths.training_workfile.parent.mkdir(parents=True)
    paths.training_workfile.write_bytes(builder.stable_json_bytes(training))
    paths.evaluation_workfile.write_bytes(builder.stable_json_bytes(evaluation))
    inputs = builder.load_authoring_inputs(
        sources,
        paths,
        require_complete=True,
    )
    return paths, inputs


def synthetic_historical(text: str, *, consumed: bool = False) -> dict[str, Any]:
    return {
        "normalized_text_sha256": builder.normalized_text_sha256(text),
        "source_family_id": (
            "v2c6_sf1_definition_direct" if consumed else "synthetic_history"
        ),
        "text_sha256": builder.text_sha256(text),
    }


def synthetic_text_record(
    record_id: str,
    text: str,
    intent: str = "account_blocked",
) -> dict[str, Any]:
    return {"intent": intent, "record_id": record_id, "text": text}


def test_frozen_contract_identity() -> None:
    content = builder.DEFAULT_PATHS.contract.read_bytes()
    payload = json.loads(content)

    assert builder.sha256_bytes(content) == builder.CONTRACT_SHA256
    assert payload["phase"] == "V2-C6 Step 29H-C"
    assert payload["status"] == "FROZEN"
    assert payload["next_required"] == (
        "v2c6_targeted_remediation_and_fresh_source_authoring"
    )
    assert payload["contract_status"]["step29i_authorized"] is False


def test_upstream_frozen_9008_dataset_identity() -> None:
    freeze_bytes = builder.DEFAULT_PATHS.freeze.read_bytes()
    development_bytes = builder.DEFAULT_PATHS.frozen_development.read_bytes()
    freeze = json.loads(freeze_bytes)
    development = json.loads(development_bytes)

    assert builder.sha256_bytes(freeze_bytes) == builder.FREEZE_SHA256
    assert builder.sha256_bytes(development_bytes) == (
        builder.FROZEN_DEVELOPMENT_SHA256
    )
    assert freeze["frozen_dataset_record_count"] == 9008
    assert development["example_count"] == 9008
    assert len(development["examples"]) == 9008


def test_preflight_does_not_write_outputs(
    tmp_path: Path,
    sources: builder.FrozenSources,
) -> None:
    paths = temporary_paths(tmp_path)

    result = builder.preflight(paths, sources=sources)

    assert result["files_written"] is False
    assert result["authoring_workfiles_present"] is False
    assert not any(path.exists() for path in builder.output_paths(paths))


def test_prepare_creates_ignored_empty_templates(
    tmp_path: Path,
    sources: builder.FrozenSources,
) -> None:
    paths = temporary_paths(tmp_path)

    result = builder.prepare_authoring_workfiles(paths, sources=sources)
    training = json.loads(paths.training_workfile.read_bytes())
    evaluation = json.loads(paths.evaluation_workfile.read_bytes())

    assert result["training_slot_count"] == 600
    assert result["evaluation_slot_count"] == 640
    assert all(record["text"] == "" for record in training["records"])
    assert all(record["text"] == "" for record in evaluation["records"])
    assert "data/evals/v2/ml/local/" in (ROOT / ".gitignore").read_text()
    assert builder.DEFAULT_PATHS.training_workfile.relative_to(ROOT).as_posix() == (
        "data/evals/v2/ml/local/v2c6_r2_targeted_training_authoring.json"
    )
    assert builder.DEFAULT_PATHS.evaluation_workfile.relative_to(ROOT).as_posix() == (
        "data/evals/v2/ml/local/v2c6_r2_fresh_source_evaluation_authoring.json"
    )
    assert result["tracked_files_written"] is False


def test_exact_template_sizes_distributions_and_intents(
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    training, evaluation = templates
    training_families = Counter(
        record["source_family_id"] for record in training["records"]
    )
    evaluation_families = Counter(
        record["source_family_id"] for record in evaluation["records"]
    )

    assert training["record_count"] == len(training["records"]) == 600
    assert evaluation["record_count"] == len(evaluation["records"]) == 640
    assert training_families == {
        "v2c6_r2_train_sf1_minimal_boundary": 200,
        "v2c6_r2_train_sf2_contextual_scenario": 200,
        "v2c6_r2_train_sf3_conversational_correction": 200,
    }
    assert evaluation_families == {
        "v2c6_r2_eval_sf1_independent_casework": 320,
        "v2c6_r2_eval_sf2_independent_naturalistic": 320,
    }
    for family in training_families:
        counts = Counter(
            record["intent"]
            for record in training["records"]
            if record["source_family_id"] == family
        )
        assert counts == {
            **{intent: 20 for intent in builder.TARGETED_INTENTS[:-1]},
            builder.UNSUPPORTED_INTENT: 60,
        }
    for family in evaluation_families:
        counts = Counter(
            record["intent"]
            for record in evaluation["records"]
            if record["source_family_id"] == family
        )
        assert counts == {intent: 40 for intent in builder.TARGETED_INTENTS}


def test_all_boundaries_and_tier_allocation_are_frozen(
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    for payload in templates:
        report = builder.boundary_coverage_report(payload["records"])

        assert report["required_pair_count"] == 10
        assert report["all_families_cover_all_pairs"] is True
        for family in report["by_source_family"].values():
            assert family["pair_count"] == 10
            assert family["all_ten_pairs_covered_on_both_sides"] is True
            assert family["minimum_tier_1_pair_count"] > (
                family["maximum_tier_2_pair_count"]
            )


def test_unsupported_subtype_vocabulary_is_exact(
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    for payload in templates:
        observed = {
            record["unsupported_subtype"]
            for record in payload["records"]
            if record["intent"] == builder.UNSUPPORTED_INTENT
        }
        assert observed == set(builder.UNSUPPORTED_SUBTYPES)
        assert all(
            record["unsupported_subtype"] is None
            for record in payload["records"]
            if record["intent"] != builder.UNSUPPORTED_INTENT
        )


def test_empty_text_is_rejected_for_completed_authoring(
    sources: builder.FrozenSources,
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    training = completed_workfile(templates[0], "training")
    training["records"][0]["text"] = ""

    with pytest.raises(ValueError, match="authoring text is empty"):
        builder.validate_workfile_structure(
            training,
            sources.contract,
            dataset_role=builder.TRAINING_ROLE,
            require_complete=True,
        )


def test_incomplete_review_is_rejected(
    sources: builder.FrozenSources,
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    training = completed_workfile(templates[0], "training")
    training["records"][0]["review_status"] = "unreviewed"

    with pytest.raises(ValueError, match="all records must be approved"):
        builder.validate_workfile_structure(
            training,
            sources.contract,
            dataset_role=builder.TRAINING_ROLE,
            require_complete=True,
        )


def test_unresolved_required_human_adjudication_is_rejected(
    sources: builder.FrozenSources,
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    training = completed_workfile(templates[0], "training")
    record = training["records"][0]
    record["requires_human_adjudication"] = True
    record["review_flags"]["low_confidence_review"] = True

    with pytest.raises(ValueError, match="human adjudication unresolved"):
        builder.validate_workfile_structure(
            training,
            sources.contract,
            dataset_role=builder.TRAINING_ROLE,
            require_complete=True,
        )


@pytest.mark.parametrize(
    "field",
    (
        "candidate_id",
        "predicted_intent",
        "decision_score",
        "model_score",
        "classifier_score",
        "classifier_confidence",
    ),
)
def test_prediction_informed_metadata_is_rejected(
    sources: builder.FrozenSources,
    templates: tuple[dict[str, Any], dict[str, Any]],
    field: str,
) -> None:
    training = copy.deepcopy(templates[0])
    training["records"][0]["authoring_provenance"][field] = "prohibited"

    with pytest.raises(ValueError, match="prediction-informed metadata"):
        builder.validate_workfile_structure(
            training,
            sources.contract,
            dataset_role=builder.TRAINING_ROLE,
            require_complete=False,
        )


def test_fresh_evaluation_provenance_flags_are_explicit(
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    evaluation = templates[1]
    for record in evaluation["records"]:
        assert record["not_training_data"] is True
        assert record["excluded_from_candidate_fitting"] is True
        assert record["independently_authored"] is True
        assert record["prediction_informed_authoring"] is False
        assert record["candidate_outputs_inspected_during_authoring"] is False
        assert (
            record["consumed_old_source_family_used_as_paraphrase_template"]
            is False
        )


@pytest.mark.parametrize(
    ("second_text", "failure"),
    (
        ("Duplicate text", "within_exact_duplicate_count"),
        ("  DUPLICATE   TEXT  ", "within_normalized_duplicate_count"),
    ),
)
def test_duplicate_and_normalized_duplicate_checks(
    second_text: str,
    failure: str,
) -> None:
    training = [
        synthetic_text_record("train:1", "Duplicate text"),
        synthetic_text_record("train:2", second_text),
    ]
    evaluation = [synthetic_text_record("eval:1", "Independent evaluation")]

    with pytest.raises(ValueError, match=failure):
        builder.duplicate_and_leakage_report(training, evaluation, [])


def test_cross_role_and_historical_overlap_checks() -> None:
    shared = "Text shared across roles"
    history = "Text already in historical development"

    with pytest.raises(ValueError, match="training_evaluation"):
        builder.duplicate_and_leakage_report(
            [synthetic_text_record("train:1", shared)],
            [synthetic_text_record("eval:1", shared)],
            [],
        )
    with pytest.raises(ValueError, match="training_historical"):
        builder.duplicate_and_leakage_report(
            [synthetic_text_record("train:1", history)],
            [synthetic_text_record("eval:1", "Independent evaluation")],
            [synthetic_historical(history)],
        )
    with pytest.raises(ValueError, match="evaluation_consumed"):
        builder.duplicate_and_leakage_report(
            [synthetic_text_record("train:1", "Independent training")],
            [synthetic_text_record("eval:1", history)],
            [synthetic_historical(history, consumed=True)],
        )


def test_cross_intent_duplicate_is_rejected() -> None:
    text = "The same normalized text cannot have two intents"
    training = [synthetic_text_record("train:1", text, "account_blocked")]
    evaluation = [
        synthetic_text_record("eval:1", text.upper(), "cancel_transfer")
    ]

    with pytest.raises(ValueError, match="cross_intent_normalized"):
        builder.duplicate_and_leakage_report(training, evaluation, [])


def test_record_and_group_collisions_are_rejected(
    sources: builder.FrozenSources,
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    training = copy.deepcopy(templates[0]["records"])
    evaluation = copy.deepcopy(templates[1]["records"])
    evaluation[0]["record_id"] = training[0]["record_id"]

    with pytest.raises(ValueError, match="record IDs overlap"):
        builder.validate_record_and_group_isolation(
            training,
            evaluation,
            sources,
        )

    evaluation = copy.deepcopy(templates[1]["records"])
    evaluation[0]["group_id"] = training[0]["group_id"]
    with pytest.raises(ValueError, match="group IDs overlap"):
        builder.validate_record_and_group_isolation(
            training,
            evaluation,
            sources,
        )


def test_historical_record_and_group_collisions_are_rejected(
    sources: builder.FrozenSources,
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    training = copy.deepcopy(templates[0]["records"])
    evaluation = copy.deepcopy(templates[1]["records"])
    training[0]["record_id"] = next(iter(sources.historical_ids))

    with pytest.raises(ValueError, match="record ID collides"):
        builder.validate_record_and_group_isolation(
            training,
            evaluation,
            sources,
        )

    training = copy.deepcopy(templates[0]["records"])
    training[0]["group_id"] = next(iter(sources.historical_group_ids))
    with pytest.raises(ValueError, match="group ID collides"):
        builder.validate_record_and_group_isolation(
            training,
            evaluation,
            sources,
        )


def test_risk_mapping_and_role_isolation(
    sources: builder.FrozenSources,
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    training, evaluation = templates
    assert all(
        record["risk_level"] == builder.RISK_BY_INTENT[record["intent"]]
        for record in [*training["records"], *evaluation["records"]]
    )
    assert all(
        record["dataset_role"] == builder.TRAINING_ROLE
        and record["not_training_data"] is False
        and record["excluded_from_candidate_fitting"] is False
        for record in training["records"]
    )
    assert all(
        record["dataset_role"] == builder.EVALUATION_ROLE
        and record["not_training_data"] is True
        and record["excluded_from_candidate_fitting"] is True
        for record in evaluation["records"]
    )

    changed = copy.deepcopy(training)
    changed["records"][0]["risk_level"] = "PUBLIC"
    with pytest.raises(ValueError, match="frozen slot field changed"):
        builder.validate_workfile_structure(
            changed,
            sources.contract,
            dataset_role=builder.TRAINING_ROLE,
            require_complete=False,
        )


def test_source_family_provenance_mismatch_is_rejected(
    sources: builder.FrozenSources,
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    changed = copy.deepcopy(templates[0])
    changed["records"][0]["authoring_provenance"][
        "source_family_independence_basis"
    ] = "unfrozen provenance"

    with pytest.raises(ValueError, match="frozen authoring provenance changed"):
        builder.validate_workfile_structure(
            changed,
            sources.contract,
            dataset_role=builder.TRAINING_ROLE,
            require_complete=False,
        )


def test_source_family_authoring_method_mismatch_is_rejected(
    sources: builder.FrozenSources,
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    changed = completed_workfile(templates[0], "training")
    changed["records"][1]["authoring_provenance"]["authoring_method"] = (
        "controlled_llm_assisted"
    )

    with pytest.raises(ValueError, match="authoring method is inconsistent"):
        builder.validate_workfile_structure(
            changed,
            sources.contract,
            dataset_role=builder.TRAINING_ROLE,
            require_complete=True,
        )


def test_build_artifacts_have_exact_counts_and_exclude_fresh_evaluation(
    tmp_path: Path,
    sources: builder.FrozenSources,
) -> None:
    paths, inputs = completed_inputs(sources, tmp_path)

    artifacts = builder.build_artifacts(sources, inputs, paths)
    training, _, combined, combined_manifest, evaluation, _ = artifacts.payloads
    evaluation_ids = {record["record_id"] for record in evaluation["examples"]}
    combined_ids = {record["example_id"] for record in combined["examples"]}
    evaluation_texts = {
        record["normalized_text_sha256"] for record in evaluation["examples"]
    }
    combined_texts = {
        record["normalized_text_sha256"] for record in combined["examples"]
    }

    assert training["example_count"] == len(training["examples"]) == 600
    assert combined["example_count"] == len(combined["examples"]) == 9608
    assert evaluation["example_count"] == len(evaluation["examples"]) == 640
    assert evaluation_ids.isdisjoint(combined_ids)
    assert evaluation_texts.isdisjoint(combined_texts)
    assert combined_manifest["counts"]["fresh_evaluation_record_count"] == 0


def test_fresh_evaluation_safety_counts_are_derived_per_family(
    tmp_path: Path,
    sources: builder.FrozenSources,
) -> None:
    _paths, inputs = completed_inputs(sources, tmp_path)
    evaluation = builder.build_evaluation_payload(inputs)

    for counts in evaluation["safety_composition_by_source_family"].values():
        assert counts == {
            "maximum_passing_protected_false_positive_count": 1,
            "non_protected_record_count": 160,
            "protected_false_positive_rate_threshold": 0.01,
            "protected_record_count": 160,
            "record_count": 320,
        }
    assert evaluation["governance"]["protected_fpr_calculated"] is False


def test_manifests_are_deterministic_and_bind_artifact_hashes(
    tmp_path: Path,
    sources: builder.FrozenSources,
) -> None:
    paths, inputs = completed_inputs(sources, tmp_path)

    first = builder.build_artifacts(sources, inputs, paths)
    second = builder.build_artifacts(sources, inputs, paths)

    assert first.contents == second.contents
    for manifest, content in zip(
        (first.payloads[1], first.payloads[3], first.payloads[5]),
        (first.contents[0], first.contents[2], first.contents[4]),
        strict=True,
    ):
        assert manifest["artifact"]["sha256"] == builder.sha256_bytes(content)


def test_tracked_outputs_are_create_once(
    tmp_path: Path,
    sources: builder.FrozenSources,
) -> None:
    paths, inputs = completed_inputs(sources, tmp_path)
    artifacts = builder.build_artifacts(sources, inputs, paths)
    builder.write_artifacts(artifacts, paths)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.write_artifacts(artifacts, paths)


def test_check_results_rejects_corrupted_output(
    tmp_path: Path,
    sources: builder.FrozenSources,
) -> None:
    paths, inputs = completed_inputs(sources, tmp_path)
    artifacts = builder.build_artifacts(sources, inputs, paths)
    builder.write_artifacts(artifacts, paths)
    paths.training_output.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="not the deterministic build"):
        builder.check_results(paths, sources=sources)


def test_script_contains_no_model_or_embedding_execution() -> None:
    tree = ast.parse((ROOT / builder.SCRIPT_RELATIVE_PATH).read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    prohibited_import_roots = {
        "fastembed",
        "hdbscan",
        "numpy",
        "sklearn",
        "torch",
        "umap",
    }
    assert not {
        name.split(".", maxsplit=1)[0] for name in imports
    } & prohibited_import_roots
    assert not called_attributes & {
        "fit",
        "fit_predict",
        "predict",
        "transform",
    }
    source = (ROOT / builder.SCRIPT_RELATIVE_PATH).read_text()
    assert "FastEmbed" not in source
    assert "TextEmbedding" not in source


@pytest.mark.parametrize(
    "prohibited_path",
    (
        builder.DEFAULT_PATHS.prohibited_holdout,
        builder.ML_DIRECTORY / "v2c6_final_holdout.json",
    ),
)
def test_final_holdout_guard_fails_before_reader_is_called(
    prohibited_path: Path,
) -> None:
    reader_called = False

    def forbidden_reader(path: Path) -> bytes:
        nonlocal reader_called
        reader_called = True
        return path.read_bytes()

    with pytest.raises(PermissionError, match="final holdout access is prohibited"):
        builder.guarded_read_bytes(
            prohibited_path,
            builder.DEFAULT_PATHS,
            forbidden_reader,
        )
    assert reader_called is False


def test_step29i_remains_unauthorized(
    templates: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    assert builder.GOVERNANCE_FLAGS["step29i_authorized"] is False
    assert all(
        payload["governance"]["step29i_authorized"] is False
        for payload in templates
    )
