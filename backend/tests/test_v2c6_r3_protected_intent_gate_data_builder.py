from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_v2c6_r3_protected_intent_gate_data as builder

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPOSITORY_ROOT
    / "scripts/build_v2c6_r3_protected_intent_gate_data.py"
)


def contract_payload() -> dict[str, Any]:
    path = (
        REPOSITORY_ROOT
        / "data/evals/v2/ml/"
        "v2c6_protected_intent_gate_remediation_design_contract.json"
    )
    return json.loads(path.read_bytes())


def synthetic_record(identifier: str, text: str) -> dict[str, Any]:
    return {
        "example_id": identifier,
        "group_id": f"{identifier}:group",
        "intent": "account_balance",
        "normalized_text_sha256": builder.normalized_text_sha256(text),
        "text": text,
        "text_sha256": builder.text_sha256(text),
    }


def synthetic_sources(
    contract: dict[str, Any] | None = None,
    *,
    development_count: int = 3,
    consumed_count: int = 2,
) -> Any:
    frozen_contract = contract or contract_payload()
    development = tuple(
        synthetic_record(f"historical:{index}", f"historical text {index}")
        for index in range(development_count)
    )
    consumed = tuple(
        {
            **synthetic_record(
                f"consumed:{index}", f"consumed evaluation text {index}"
            ),
            "record_id": f"consumed:{index}",
        }
        for index in range(consumed_count)
    )
    return builder.FrozenSources(
        contract=frozen_contract,
        contract_sha256=builder.CONTRACT_SHA256,
        development={"examples": list(development)},
        development_records=development,
        consumed_evaluation={"examples": list(consumed)},
        consumed_evaluation_records=consumed,
        source_lineage={
            "consumed_r2_fresh_dataset": {
                "path": "consumed.json",
                "schema_version": "v2c6-fresh-source-evaluation-dataset.v1",
                "sha256": "c" * 64,
            },
            "existing_development_dataset": {
                "path": "development.json",
                "schema_version": (
                    "v2c6-targeted-remediated-development-dataset.v1"
                ),
                "sha256": "d" * 64,
            },
        },
        historical_ids=frozenset(
            str(row.get("example_id", row.get("record_id")))
            for row in (*development, *consumed)
        ),
        historical_group_ids=frozenset(
            str(row["group_id"]) for row in (*development, *consumed)
        ),
    )


def paths(tmp_path: Path) -> Any:
    ml = tmp_path / "data/evals/v2/ml"
    return builder.BuildPaths(
        repository_root=tmp_path,
        contract=ml / "contract.json",
        existing_development=ml / "development.json",
        consumed_r2_evaluation=ml / "consumed.json",
        training_workfile=ml / "local/training.json",
        evaluation_workfile=ml / "local/evaluation.json",
        training_output=ml / "training-output.json",
        training_manifest=ml / "training-output.manifest.json",
        development_output=ml / "development-output.json",
        development_manifest=ml / "development-output.manifest.json",
        evaluation_output=ml / "evaluation-output.json",
        evaluation_manifest=ml / "evaluation-output.manifest.json",
        prohibited_holdout=ml / "v2c5_final_holdout.json",
        builder=MODULE_PATH,
    )


def approve(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    completed = copy.deepcopy(payload)
    completed["authoring_complete"] = True
    for index, record in enumerate(completed["records"]):
        text = f"{prefix} independently authored example {index:04d}"
        record["text"] = text
        record["text_sha256"] = builder.text_sha256(text)
        record["normalized_text_sha256"] = builder.normalized_text_sha256(text)
        record["authoring_provenance"][
            "authoring_method"
        ] = "controlled_llm_assisted"
        record["review_method"] = "ai_assisted_review"
        record["review_notes"] = "Semantics and frozen boundary reviewed."
        record["review_status"] = "approved"
        record["reviewer"] = "synthetic-test-reviewer"
    return completed


def write_workfiles(
    tmp_paths: Any,
    contract: dict[str, Any],
    *,
    complete: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training, evaluation = builder.create_authoring_templates(contract)
    if complete:
        training = approve(training, "training")
        evaluation = approve(evaluation, "evaluation")
    tmp_paths.training_workfile.parent.mkdir(parents=True, exist_ok=True)
    tmp_paths.training_workfile.write_bytes(builder.stable_json_bytes(training))
    tmp_paths.evaluation_workfile.write_bytes(builder.stable_json_bytes(evaluation))
    return training, evaluation


def test_contract_hash_and_identity_are_frozen() -> None:
    content = (
        REPOSITORY_ROOT
        / "data/evals/v2/ml/"
        "v2c6_protected_intent_gate_remediation_design_contract.json"
    ).read_bytes()
    assert builder.sha256_bytes(content) == builder.CONTRACT_SHA256
    builder.validate_contract(json.loads(content))


def test_templates_have_exact_empty_counts_and_deterministic_order() -> None:
    training, evaluation = builder.create_authoring_templates(contract_payload())
    assert training["record_count"] == 480
    assert evaluation["record_count"] == 640
    assert all(record["text"] == "" for record in training["records"])
    assert all(record["text"] == "" for record in evaluation["records"])
    assert training == builder.create_authoring_templates(contract_payload())[0]
    assert evaluation == builder.create_authoring_templates(contract_payload())[1]


def test_training_family_and_verifier_allocations_are_exact() -> None:
    training, _ = builder.create_authoring_templates(contract_payload())
    records = training["records"]
    assert builder.count_by(records, "source_family_id") == {
        "v2c6_r3_train_sf1_minimal_explicitness": 160,
        "v2c6_r3_train_sf2_contextual_boundary": 160,
        "v2c6_r3_train_sf3_conversational_ambiguity": 160,
    }
    assert builder.verifier_composition(records) == {
        intent: {
            "negative_record_count": 60,
            "positive_record_count": 60,
            "record_count": 120,
        }
        for intent in builder.PROTECTED_INTENTS
    }


def test_evaluation_family_intent_and_boundary_allocations_are_exact() -> None:
    _, evaluation = builder.create_authoring_templates(contract_payload())
    for family in (
        "v2c6_r3_eval_sf1_independent_casework",
        "v2c6_r3_eval_sf2_independent_naturalistic",
    ):
        records = [
            row for row in evaluation["records"] if row["source_family_id"] == family
        ]
        assert len(records) == 320
        assert builder.count_by(records, "intent") == {
            intent: 40 for intent in sorted(builder.PRIMARY_INTENTS)
        }
        unsupported = [
            row for row in records if row["intent"] == builder.UNSUPPORTED_INTENT
        ]
        assert builder.count_by(unsupported, "protected_boundary_target") == {
            intent: 10 for intent in builder.PROTECTED_INTENTS
        }


def test_fresh_evaluation_slots_freeze_exclusion_and_independence() -> None:
    _, evaluation = builder.create_authoring_templates(contract_payload())
    for record in evaluation["records"]:
        assert record["excluded_from_candidate_fitting"] is True
        assert record["not_training_data"] is True
        assert record["independently_authored"] is True
        assert record["prediction_informed_authoring"] is False
        assert record["candidate_outputs_inspected_during_authoring"] is False
        assert record["candidate_definitions_revised_from_this_evaluation"] is False
        assert record["other_r3_fresh_family_wording_consulted"] is False
        assert record["r3_training_used_as_paraphrase_source"] is False
        assert record["consumed_r2_evaluation_used_as_paraphrase_source"] is False


def test_protected_positives_and_unsupported_negatives_are_explicitly_tagged() -> None:
    training, _ = builder.create_authoring_templates(contract_payload())
    for record in training["records"]:
        target = record["protected_boundary_target"]
        assert target in builder.PROTECTED_INTENTS
        if record["intent"] == builder.UNSUPPORTED_INTENT:
            assert record["polarity"] == "negative"
            assert record["verifier_role"] == "targeted_unsupported_negative"
        else:
            assert record["intent"] == target
            assert record["polarity"] == "positive"
            assert record["verifier_role"] == "protected_positive"


def test_template_record_and_group_ids_are_unique() -> None:
    training, evaluation = builder.create_authoring_templates(contract_payload())
    all_records = [*training["records"], *evaluation["records"]]
    record_ids = [record["record_id"] for record in all_records]
    group_ids = [record["group_id"] for record in all_records]
    assert len(record_ids) == len(set(record_ids)) == 1120
    assert len(group_ids) == len(set(group_ids)) == 1120


def test_prepare_creates_only_empty_local_workfiles(tmp_path: Path) -> None:
    tmp_paths = paths(tmp_path)
    result = builder.prepare_authoring_workfiles(
        tmp_paths, sources=synthetic_sources()
    )
    assert result["training_slot_count"] == 480
    assert result["evaluation_slot_count"] == 640
    assert result["text_slots_populated"] == 0
    assert result["tracked_files_written"] is False
    assert tmp_paths.training_workfile.exists()
    assert tmp_paths.evaluation_workfile.exists()
    assert not any(path.exists() for path in builder.output_paths(tmp_paths))
    assert "data/evals/v2/ml/local/" in (REPOSITORY_ROOT / ".gitignore").read_text()
    assert builder.DEFAULT_PATHS.training_workfile.relative_to(
        REPOSITORY_ROOT
    ).as_posix() == (
        "data/evals/v2/ml/local/"
        "v2c6_r3_protected_intent_gate_training_authoring.json"
    )
    assert builder.DEFAULT_PATHS.evaluation_workfile.relative_to(
        REPOSITORY_ROOT
    ).as_posix() == (
        "data/evals/v2/ml/local/"
        "v2c6_r3_fresh_source_evaluation_authoring.json"
    )


def test_prepare_is_create_once(tmp_path: Path) -> None:
    tmp_paths = paths(tmp_path)
    sources = synthetic_sources()
    builder.prepare_authoring_workfiles(tmp_paths, sources=sources)
    before = tmp_paths.training_workfile.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.prepare_authoring_workfiles(tmp_paths, sources=sources)
    assert tmp_paths.training_workfile.read_bytes() == before


def test_preflight_is_read_only(tmp_path: Path) -> None:
    tmp_paths = paths(tmp_path)
    write_workfiles(tmp_paths, contract_payload(), complete=False)
    before = {
        tmp_paths.training_workfile: tmp_paths.training_workfile.read_bytes(),
        tmp_paths.evaluation_workfile: tmp_paths.evaluation_workfile.read_bytes(),
    }
    result = builder.preflight(tmp_paths, sources=synthetic_sources())
    assert result["files_written"] is False
    assert result["ready_for_build"] is False
    assert all(path.read_bytes() == content for path, content in before.items())


def test_build_validation_rejects_incomplete_authoring(tmp_path: Path) -> None:
    tmp_paths = paths(tmp_path)
    write_workfiles(tmp_paths, contract_payload(), complete=False)
    with pytest.raises(ValueError, match="not marked complete"):
        builder.load_authoring_inputs(
            synthetic_sources(), tmp_paths, require_complete=True
        )


def test_completed_review_requires_approval_notes_and_reviewer() -> None:
    training, _ = builder.create_authoring_templates(contract_payload())
    completed = approve(training, "training")
    completed["records"][0]["review_notes"] = ""
    with pytest.raises(ValueError, match="requires review_notes"):
        builder.validate_workfile_structure(
            completed,
            contract_payload(),
            dataset_role=builder.TRAINING_ROLE,
            require_complete=True,
        )


def test_ai_assisted_review_does_not_imply_human_adjudication() -> None:
    training, _ = builder.create_authoring_templates(contract_payload())
    completed = approve(training, "training")
    record = completed["records"][0]
    assert record["review_method"] == "ai_assisted_review"
    assert record["requires_human_adjudication"] is False
    assert record["human_adjudication_completed"] is False
    builder.validate_workfile_structure(
        completed,
        contract_payload(),
        dataset_role=builder.TRAINING_ROLE,
        require_complete=True,
    )


def test_review_flag_requires_completed_human_adjudication() -> None:
    training, _ = builder.create_authoring_templates(contract_payload())
    completed = approve(training, "training")
    record = completed["records"][0]
    record["review_flags"]["protected_write_ambiguity"] = True
    record["requires_human_adjudication"] = True
    with pytest.raises(ValueError, match="adjudication unresolved"):
        builder.validate_workfile_structure(
            completed,
            contract_payload(),
            dataset_role=builder.TRAINING_ROLE,
            require_complete=True,
        )


def test_prediction_metadata_is_rejected_recursively() -> None:
    training, _ = builder.create_authoring_templates(contract_payload())
    training["records"][0]["review_context"] = {"predicted_intent": "freeze_card"}
    with pytest.raises(ValueError, match="prediction metadata prohibited"):
        builder.validate_workfile_structure(
            training,
            contract_payload(),
            dataset_role=builder.TRAINING_ROLE,
            require_complete=False,
        )


def test_text_hashes_are_required_and_verified() -> None:
    training, _ = builder.create_authoring_templates(contract_payload())
    completed = approve(training, "training")
    completed["records"][0]["text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="text hash mismatch"):
        builder.validate_workfile_structure(
            completed,
            contract_payload(),
            dataset_role=builder.TRAINING_ROLE,
            require_complete=True,
        )


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("training_within", "training_within"),
        ("evaluation_within", "evaluation_within"),
        ("cross_role", "training_evaluation"),
        ("development", "training_development"),
        ("consumed", "evaluation_consumed"),
    ],
)
def test_duplicate_and_overlap_fail_closed(kind: str, match: str) -> None:
    training = [
        {"intent": "freeze_card", "text": "unique training one"},
        {"intent": "cancel_transfer", "text": "unique training two"},
    ]
    evaluation = [
        {"intent": "freeze_card", "text": "unique evaluation one"},
        {"intent": "unsupported_or_uncertain", "text": "unique evaluation two"},
    ]
    development = [synthetic_record("history", "historical only")]
    consumed = [synthetic_record("consumed", "consumed only")]
    if kind == "training_within":
        training[1]["text"] = " UNIQUE   TRAINING ONE "
    elif kind == "evaluation_within":
        evaluation[1]["text"] = " UNIQUE EVALUATION ONE "
    elif kind == "cross_role":
        evaluation[0]["text"] = training[0]["text"]
    elif kind == "development":
        training[0]["text"] = development[0]["text"]
    else:
        evaluation[0]["text"] = consumed[0]["text"]
    with pytest.raises(ValueError, match=match):
        builder.duplicate_and_leakage_report(
            training, evaluation, development, consumed
        )


def test_cross_intent_normalized_collision_is_rejected() -> None:
    training = [{"intent": "freeze_card", "text": "Same request"}]
    evaluation = [
        {"intent": "unsupported_or_uncertain", "text": " same   request "}
    ]
    with pytest.raises(ValueError, match="cross_intent_normalized_collision"):
        builder.duplicate_and_leakage_report(training, evaluation, [], [])


def test_identity_and_group_collisions_fail_closed() -> None:
    sources = synthetic_sources()
    training = [{"record_id": "new:1", "group_id": "shared"}]
    evaluation = [{"record_id": "new:2", "group_id": "shared"}]
    with pytest.raises(ValueError, match="group IDs overlap"):
        builder.validate_identity_and_groups(training, evaluation, sources)


def test_final_holdout_guard_blocks_before_reader() -> None:
    called = False

    def reader(_: Path) -> bytes:
        nonlocal called
        called = True
        return b"{}"

    with pytest.raises(PermissionError, match="final holdout access is prohibited"):
        builder.guarded_read_bytes(
            builder.DEFAULT_PATHS.prohibited_holdout,
            builder.DEFAULT_PATHS,
            reader,
        )
    assert called is False


def test_build_artifacts_preserve_existing_records_and_exclude_evaluation(
    tmp_path: Path,
) -> None:
    tmp_paths = paths(tmp_path)
    training_payload, evaluation_payload = write_workfiles(
        tmp_paths, contract_payload(), complete=True
    )
    sources = synthetic_sources(development_count=9608)
    inputs = builder.load_authoring_inputs(
        sources, tmp_paths, require_complete=True
    )
    artifacts = builder.build_artifacts(sources, inputs, tmp_paths)
    (
        training,
        training_manifest,
        development,
        development_manifest,
        evaluation,
        evaluation_manifest,
    ) = artifacts.payloads
    assert training["schema_version"] == builder.TRAINING_OUTPUT_SCHEMA
    assert training["example_count"] == 480
    assert development["schema_version"] == builder.DEVELOPMENT_OUTPUT_SCHEMA
    assert development["example_count"] == 10088
    assert development["examples"][:9608] == list(sources.development_records)
    assert evaluation["schema_version"] == builder.EVALUATION_OUTPUT_SCHEMA
    assert evaluation["example_count"] == 640
    assert not (
        {row["record_id"] for row in evaluation_payload["records"]}
        & {
            str(row.get("example_id", row.get("record_id")))
            for row in development["examples"]
        }
    )
    assert training_manifest["contract"]["sha256"] == builder.CONTRACT_SHA256
    assert development_manifest["counts"]["new_training_record_count"] == 480
    assert evaluation_manifest["fresh_evaluation_governance"][
        "excluded_from_candidate_fitting"
    ] is True
    assert len(training_payload["records"]) == 480


def test_manifests_pin_review_leakage_and_consumed_r2_lineage(
    tmp_path: Path,
) -> None:
    tmp_paths = paths(tmp_path)
    write_workfiles(tmp_paths, contract_payload(), complete=True)
    sources = synthetic_sources(development_count=9608)
    inputs = builder.load_authoring_inputs(
        sources, tmp_paths, require_complete=True
    )
    artifacts = builder.build_artifacts(sources, inputs, tmp_paths)
    for manifest in (
        artifacts.payloads[1],
        artifacts.payloads[3],
        artifacts.payloads[5],
    ):
        assert manifest["consumed_r2_lineage"] == sources.source_lineage[
            "consumed_r2_fresh_dataset"
        ]
        assert manifest["validation"]["duplicate_and_leakage"][
            "all_duplicate_and_leakage_checks_passed"
        ] is True
        assert manifest["validation"]["review"]["training"][
            "approved_count"
        ] == 480
        assert manifest["governance"]["step29i_authorized"] is False
        assert manifest["governance"]["final_holdout_accessed"] is False


def test_build_refuses_to_overwrite_any_tracked_output(tmp_path: Path) -> None:
    tmp_paths = paths(tmp_path)
    tmp_paths.training_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_paths.training_output.write_text("sentinel")
    with pytest.raises(FileExistsError, match="tracked outputs"):
        builder.build(tmp_paths, sources=synthetic_sources())
    assert tmp_paths.training_output.read_text() == "sentinel"


def test_check_results_is_read_only(tmp_path: Path, monkeypatch: Any) -> None:
    tmp_paths = paths(tmp_path)
    expected = builder.BuiltArtifacts(
        contents=(b"a", b"b", b"c", b"d", b"e", b"f"),
        payloads=({}, {}, {}, {}, {}, {}),
    )
    write_workfiles(tmp_paths, contract_payload(), complete=False)
    for path, content in zip(
        builder.output_paths(tmp_paths), expected.contents, strict=True
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    before = {path: path.read_bytes() for path in builder.output_paths(tmp_paths)}
    monkeypatch.setattr(
        builder,
        "load_authoring_inputs",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(builder, "build_artifacts", lambda *args, **kwargs: expected)
    result = builder.check_results(tmp_paths, sources=synthetic_sources())
    assert result["results_valid"] is True
    assert result["files_written"] is False
    assert all(path.read_bytes() == content for path, content in before.items())


def test_source_contains_no_ml_framework_imports_or_execution_calls() -> None:
    source = MODULE_PATH.read_text()
    for prohibited in (
        "sentence_transformers",
        "sklearn",
        "torch",
        "transformers",
        "fastembed",
        "TextEmbedding",
        ".fit(",
        ".predict(",
        "passage_embed(",
        "query_embed(",
    ):
        assert prohibited not in source


def test_cli_exposes_only_the_four_frozen_modes() -> None:
    parser = builder.build_parser()
    for flag in (
        "--preflight",
        "--prepare-authoring-workfiles",
        "--build",
        "--check-results",
    ):
        namespace = parser.parse_args([flag])
        assert any(vars(namespace).values())
