from __future__ import annotations

import ast
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts import build_v2c5_final_holdout as builder

EXPECTED_INTENTS = (
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
)
EXPECTED_RISKS = {
    "account_balance": "PRIVATE_READ",
    "account_blocked": "PRIVATE_READ",
    "cancel_transfer": "PROTECTED_WRITE",
    "card_status": "PRIVATE_READ",
    "close_account": "PROTECTED_WRITE",
    "create_dispute": "PROTECTED_WRITE",
    "escalation": "ESCALATION_OR_UNCERTAIN",
    "freeze_card": "PROTECTED_WRITE",
    "informational_policy": "PUBLIC",
    "lost_or_stolen_phone": "ESCALATION_OR_UNCERTAIN",
    "passcode_recovery": "ESCALATION_OR_UNCERTAIN",
    "recent_transactions": "PRIVATE_READ",
    "transaction_details": "PRIVATE_READ",
    "transfer_failed_or_declined": "PRIVATE_READ",
    "transfer_pending": "PRIVATE_READ",
    "unsupported_or_uncertain": "ESCALATION_OR_UNCERTAIN",
}
REQUIRED_TAGS = frozenset(
    {
        "clear_direct",
        "longer_contextual",
        "natural_paraphrase_or_colloquial",
        "neighboring_intent_boundary_where_applicable",
        "short_voice_style",
    }
)


@pytest.fixture(scope="module")
def frozen_sources() -> builder.FrozenSources:
    return builder.load_frozen_sources(probe_paths=())


@pytest.fixture(scope="module")
def authored_seed() -> dict[str, Any]:
    return json.loads(builder.DEFAULT_PATHS.seed.read_text(encoding="utf-8"))


def make_seed() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    required_tags = sorted(REQUIRED_TAGS)
    sequence = 0
    for intent in EXPECTED_INTENTS:
        for intent_index in range(40):
            sequence += 1
            records.append(
                {
                    "boundary_tags": [],
                    "coverage_tags": [
                        required_tags[intent_index % len(required_tags)]
                    ],
                    "intent": intent,
                    "seed_id": f"v2c5-final-holdout-seed:{sequence:04d}",
                    "text": (
                        f"Independent synthetic fixture for {intent} "
                        f"number {intent_index:02d}."
                    ),
                }
            )
    return {
        "authorship": copy.deepcopy(builder.REQUIRED_AUTHORSHIP),
        "records": records,
        "schema_version": builder.SEED_SCHEMA_VERSION,
    }


def make_sources(
    overlap_hashes: dict[str, frozenset[str]] | None = None,
) -> builder.FrozenSources:
    return builder.FrozenSources(
        contract={},
        taxonomy={"taxonomy_version": "v2c5-taxonomy.v1"},
        intent_labels=EXPECTED_INTENTS,
        risk_by_intent=copy.deepcopy(EXPECTED_RISKS),
        required_coverage_tags=REQUIRED_TAGS,
        overlap_hashes=overlap_hashes
        or {
            "all_historical_development_and_training_text_used_in_v2c5": frozenset(),
            "consumed_v2c3_challenge": frozenset(),
            "consumed_v2c3_external_lockbox": frozenset(),
            "preexisting_v2c5_model_selection_probe": frozenset(),
            "v2c5_expanded_development_dataset": frozenset(),
        },
        source_artifacts={},
        probe_record_count=0,
    )


def make_paths(tmp_path: Path, seed: dict[str, Any]) -> builder.BuildPaths:
    seed_path = tmp_path / "v2c5_final_holdout_seed.json"
    seed_path.write_bytes(builder.stable_json_bytes(seed))
    return replace(
        builder.DEFAULT_PATHS,
        seed=seed_path,
        probe_directory=tmp_path,
        dataset_output=tmp_path / "v2c5_final_holdout.json",
        manifest_output=tmp_path / "v2c5_final_holdout.manifest.json",
    )


def test_all_required_frozen_source_hashes_are_pinned() -> None:
    for _, path, expected_hash in builder.source_path_items(builder.DEFAULT_PATHS):
        assert builder.sha256_bytes(path.read_bytes()) == expected_hash


def test_authored_seed_has_exactly_640_records_and_40_per_intent(
    authored_seed: dict[str, Any], frozen_sources: builder.FrozenSources
) -> None:
    validated = builder.validate_seed(
        authored_seed,
        frozen_sources.intent_labels,
        frozen_sources.required_coverage_tags,
    )

    assert len(validated.records) == 640
    assert validated.intent_counts == {intent: 40 for intent in EXPECTED_INTENTS}
    assert tuple(validated.intent_counts) == EXPECTED_INTENTS


def test_authored_seed_has_zero_required_normalized_overlap(
    authored_seed: dict[str, Any], frozen_sources: builder.FrozenSources
) -> None:
    validated = builder.validate_seed(
        authored_seed,
        frozen_sources.intent_labels,
        frozen_sources.required_coverage_tags,
    )

    overlap_counts = builder.validate_no_overlaps(
        validated.records, frozen_sources.overlap_hashes
    )

    assert overlap_counts
    assert set(overlap_counts.values()) == {0}


def test_seed_ids_are_exactly_sequential_and_unique() -> None:
    seed = make_seed()

    validated = builder.validate_seed(seed, EXPECTED_INTENTS, REQUIRED_TAGS)

    assert [record["seed_id"] for record in validated.records] == [
        f"v2c5-final-holdout-seed:{index:04d}" for index in range(1, 641)
    ]


def test_empty_text_is_rejected() -> None:
    seed = make_seed()
    seed["records"][0]["text"] = "  "

    with pytest.raises(ValueError, match="text must be non-empty"):
        builder.validate_seed(seed, EXPECTED_INTENTS, REQUIRED_TAGS)


def test_exact_text_duplicate_is_rejected() -> None:
    seed = make_seed()
    seed["records"][1]["text"] = seed["records"][0]["text"]

    with pytest.raises(ValueError, match="raw-text duplicates"):
        builder.validate_seed(seed, EXPECTED_INTENTS, REQUIRED_TAGS)


def test_normalized_text_duplicate_is_rejected() -> None:
    seed = make_seed()
    seed["records"][1]["text"] = f"  {seed['records'][0]['text'].upper()}  "

    with pytest.raises(ValueError, match="normalized-text duplicates"):
        builder.validate_seed(seed, EXPECTED_INTENTS, REQUIRED_TAGS)


def test_invalid_intent_is_rejected() -> None:
    seed = make_seed()
    seed["records"][0]["intent"] = "invented_intent"

    with pytest.raises(ValueError, match="unknown intent"):
        builder.validate_seed(seed, EXPECTED_INTENTS, REQUIRED_TAGS)


def test_seed_cannot_supply_risk_or_model_evidence() -> None:
    for field in ("risk", "model_prediction", "embedding", "similarity_score"):
        seed = make_seed()
        seed["records"][0][field] = "prohibited"

        with pytest.raises(ValueError, match="prohibited derived metadata"):
            builder.validate_seed(seed, EXPECTED_INTENTS, REQUIRED_TAGS)


def test_step20_risk_is_derived_for_every_output_record() -> None:
    seed = builder.validate_seed(make_seed(), EXPECTED_INTENTS, REQUIRED_TAGS)
    examples = builder.build_examples(seed.records, EXPECTED_RISKS)

    assert all(
        record["risk"] == EXPECTED_RISKS[record["intent"]]
        for record in examples
    )
    assert all("risk" not in source for source in seed.records)


def test_output_preserves_seed_lineage_and_creates_final_ids() -> None:
    seed = builder.validate_seed(make_seed(), EXPECTED_INTENTS, REQUIRED_TAGS)
    examples = builder.build_examples(seed.records, EXPECTED_RISKS)

    for index, (source, output) in enumerate(
        zip(seed.records, examples, strict=True), start=1
    ):
        assert output["source_seed_id"] == source["seed_id"]
        assert output["example_id"] == f"v2c5-final-holdout:{index:04d}"
        assert output["text"] == source["text"]
        assert output["intent"] == source["intent"]


def test_protected_write_set_and_counts_are_exactly_frozen() -> None:
    seed = builder.validate_seed(make_seed(), EXPECTED_INTENTS, REQUIRED_TAGS)
    examples = builder.build_examples(seed.records, EXPECTED_RISKS)
    protected = [
        record
        for record in examples
        if record["intent"] in builder.PROTECTED_WRITE_INTENTS
    ]

    assert builder.PROTECTED_WRITE_INTENTS == (
        "cancel_transfer",
        "close_account",
        "create_dispute",
        "freeze_card",
    )
    assert len(protected) == 160
    assert len(examples) - len(protected) == 480


@pytest.mark.parametrize(
    "source",
    [
        "v2c5_expanded_development_dataset",
        "consumed_v2c3_challenge",
        "consumed_v2c3_external_lockbox",
    ],
)
def test_required_normalized_overlap_is_rejected(source: str) -> None:
    records = [{"text": "Shared normalized text"}]
    overlap_hashes = {
        source: frozenset(
            {builder.normalized_text_sha256("  SHARED normalized   text  ")}
        )
    }

    with pytest.raises(ValueError, match=source):
        builder.validate_no_overlaps(records, overlap_hashes)


def test_preexisting_v2c5_probe_overlap_is_rejected() -> None:
    records = [{"text": "Text already used by the probe"}]
    overlap_hashes = {
        "preexisting_v2c5_model_selection_probe": frozenset(
            {builder.normalized_text_sha256(records[0]["text"])}
        )
    }

    with pytest.raises(ValueError, match="model_selection_probe"):
        builder.validate_no_overlaps(records, overlap_hashes)


def test_overlap_diagnostic_deduplicates_seed_ids_across_sources() -> None:
    record = {
        "intent": "account_balance",
        "seed_id": "v2c5-final-holdout-seed:0001",
        "text": "Local seed collision",
    }
    normalized_hash = builder.normalized_text_sha256(record["text"])

    overlapping = builder.list_overlap_seed_records(
        [record],
        {
            "consumed_v2c3_challenge": frozenset({normalized_hash}),
            "v2c5_expanded_development_dataset": frozenset({normalized_hash}),
        },
    )

    assert overlapping == [
        {
            "intent": "account_balance",
            "overlap_sources": [
                "consumed_v2c3_challenge",
                "v2c5_expanded_development_dataset",
            ],
            "seed_id": "v2c5-final-holdout-seed:0001",
        }
    ]


def test_overlap_diagnostic_emits_no_source_or_seed_text() -> None:
    record = {
        "intent": "account_balance",
        "seed_id": "v2c5-final-holdout-seed:0002",
        "text": "Authored local text that must not be printed",
    }
    normalized_hash = builder.normalized_text_sha256(record["text"])

    overlapping = builder.list_overlap_seed_records(
        [record],
        {"historical_source": frozenset({normalized_hash})},
    )
    rendered = json.dumps(overlapping, sort_keys=True)

    assert set(overlapping[0]) == {"intent", "overlap_sources", "seed_id"}
    assert record["text"] not in rendered
    assert "historical utterance must remain secret" not in rendered


def test_overlap_diagnostic_writes_nothing(tmp_path: Path) -> None:
    seed = make_seed()
    paths = make_paths(tmp_path, seed)
    normalized_hash = builder.normalized_text_sha256(seed["records"][0]["text"])
    sources = make_sources(
        {"v2c5_expanded_development_dataset": frozenset({normalized_hash})}
    )

    diagnostic = builder.build_overlap_diagnostic(paths, sources=sources)

    assert diagnostic["overlapping_seed_count"] == 1
    assert not paths.dataset_output.exists()
    assert not paths.manifest_output.exists()


def test_list_overlap_cli_prints_only_safe_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    diagnostic = {
        "overlapping_seed_count": 1,
        "records": [
            {
                "intent": "account_balance",
                "overlap_sources": ["consumed_v2c3_challenge"],
                "seed_id": "v2c5-final-holdout-seed:0001",
            }
        ],
    }
    monkeypatch.setattr(builder, "build_overlap_diagnostic", lambda: diagnostic)

    assert builder.main(["--list-overlap-seed-ids"]) == 0

    assert json.loads(capsys.readouterr().out) == diagnostic


def test_check_path_still_rejects_overlap(tmp_path: Path) -> None:
    seed = make_seed()
    paths = make_paths(tmp_path, seed)
    normalized_hash = builder.normalized_text_sha256(seed["records"][0]["text"])
    sources = make_sources(
        {"v2c5_expanded_development_dataset": frozenset({normalized_hash})}
    )

    with pytest.raises(ValueError, match="normalized-text overlap detected"):
        builder.build_artifacts(paths, sources=sources)


def test_sealed_v2c4_path_is_prohibited_before_read(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = False
    original = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        nonlocal opened
        if path.name == builder.SEALED_V2C4_HOLDOUT.name:
            opened = True
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)

    with pytest.raises(ValueError, match="prohibited source"):
        builder.read_pinned_json(
            builder.SEALED_V2C4_HOLDOUT,
            "not-used",
            "sealed V2-C4 holdout",
        )

    assert opened is False


def test_normal_source_loading_never_opens_sealed_v2c4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_names: list[str] = []
    original = Path.read_bytes

    def recording_read(path: Path) -> bytes:
        opened_names.append(path.name)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", recording_read)

    builder.load_frozen_sources(probe_paths=())

    assert builder.SEALED_V2C4_HOLDOUT.name not in opened_names


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("contract", "frozen contract SHA-256 changed"),
        ("taxonomy", "frozen taxonomy SHA-256 changed"),
        ("development", "frozen development SHA-256 changed"),
    ],
)
def test_wrong_frozen_hash_is_rejected(
    tmp_path: Path, field: str, error: str
) -> None:
    original = getattr(builder.DEFAULT_PATHS, field)
    changed = tmp_path / original.name
    changed.write_bytes(original.read_bytes() + b"\n")
    paths = replace(builder.DEFAULT_PATHS, **{field: changed})

    with pytest.raises(ValueError, match=error):
        builder.load_frozen_sources(paths, probe_paths=())


def test_output_bytes_are_deterministic(tmp_path: Path) -> None:
    paths = make_paths(tmp_path, make_seed())
    sources = make_sources()

    first = builder.build_artifacts(paths, sources=sources)
    second = builder.build_artifacts(paths, sources=sources)

    assert first.dataset_bytes == second.dataset_bytes
    assert first.manifest_bytes == second.manifest_bytes
    assert first.dataset_bytes.endswith(b"\n")
    assert first.manifest_bytes.endswith(b"\n")


def test_check_writes_nothing(tmp_path: Path) -> None:
    paths = make_paths(tmp_path, make_seed())
    artifacts = builder.build_artifacts(paths, sources=make_sources())

    builder.check_artifacts(artifacts, paths)

    assert not paths.dataset_output.exists()
    assert not paths.manifest_output.exists()


def test_frozen_manifest_completes_step21_and_permits_step22(
    tmp_path: Path,
) -> None:
    paths = make_paths(tmp_path, make_seed())
    artifacts = builder.build_artifacts(paths, sources=make_sources())
    status = artifacts.manifest_payload["execution_status"]

    assert status["final_holdout_contract_frozen"] is True
    assert status["final_holdout_examples_created"] is True
    assert status["final_holdout_frozen"] is True
    assert status["final_holdout_evaluated"] is False
    assert status["step21_complete"] is True
    assert status["step22_permitted"] is True
    assert artifacts.manifest_payload["next_required"] == (
        "v2c5_expanded_taxonomy_model_selection"
    )


def test_dataset_governance_prohibits_development_uses(tmp_path: Path) -> None:
    paths = make_paths(tmp_path, make_seed())
    artifacts = builder.build_artifacts(paths, sources=make_sources())
    governance = artifacts.dataset_payload["evaluation_governance"]

    assert governance == {
        "augmentation_source_eligible": False,
        "error_analysis_eligible_before_final_evaluation": False,
        "final_evaluation_step": "V2-C5 Step 23",
        "model_selection_eligible": False,
        "single_final_evaluation_only": True,
        "step22_may_inspect_individual_holdout_examples": False,
        "step22_may_run_holdout_inference": False,
        "step22_may_tune_from_holdout": False,
        "step22_may_use_holdout_errors": False,
        "taxonomy_discovery_eligible": False,
        "threshold_selection_eligible": False,
        "training_eligible": False,
    }


def test_manifest_records_no_model_or_vector_evidence(tmp_path: Path) -> None:
    paths = make_paths(tmp_path, make_seed())
    artifacts = builder.build_artifacts(paths, sources=make_sources())
    checks = artifacts.manifest_payload["integrity_and_overlap_checks"]

    assert checks["embeddings_used"] is False
    assert checks["semantic_similarity_used"] is False
    assert checks["vector_similarity_used"] is False
    assert checks["model_predictions_used"] is False
    assert checks["sealed_v2c4_holdout_accessed"] is False
    assert set(checks["prohibited_source_counts"].values()) == {0}


def test_builder_uses_only_python_standard_library() -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        node.names[0].name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
    } | {
        str(node.module).split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    }

    assert imported_roots == {
        "argparse",
        "collections",
        "copy",
        "dataclasses",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "tempfile",
        "typing",
        "unicodedata",
    }
    for prohibited in (
        "fastembed",
        "sentence_transformers",
        "sklearn",
        "numpy",
        "pandas",
        "cosine_similarity",
        "predict_proba(",
        "fit_predict(",
    ):
        assert prohibited not in source
