import json
from pathlib import Path

import pytest

from scripts import build_cfpb_annotation_workfile as contract
from scripts import review_cfpb_semantic_annotations as reviewer


def synthetic_row(
    complaint_id: str,
    narrative: str,
    *,
    mapping_status: str = "NEAR_MATCH",
    candidates: list[str] | None = None,
) -> dict[str, object]:
    return {
        "complaint_id": complaint_id,
        "narrative": narrative,
        "narrative_sha256": contract.sha256_bytes(narrative.encode("utf-8")),
        "mapping_status": mapping_status,
        "candidate_sentinelvoice_intents": candidates or ["account_balance"],
        "product": "Synthetic product",
        "issue": "Synthetic issue",
        "sub_issue": "Synthetic sub-issue",
        "narrative_length": len(narrative),
        "synthetic_extra": {"preserve": True},
        **contract.initialized_annotation_fields(),
    }


def annotation(
    *,
    category: str = "SINGLE_SUPPORTED_INTENT",
    intents: list[str] | None = None,
    confidence: str = "HIGH",
    status: str = "REVIEWED",
    secondary: bool = False,
) -> dict[str, object]:
    return reviewer.build_annotation(
        review_category=category,
        supported_intents=["account_balance"] if intents is None else intents,
        annotation_confidence=confidence,
        annotation_note="Synthetic reviewer rationale.",
        reviewer_id="reviewer-synthetic",
        adjudication_status=status,
        secondary_review_required=secondary,
    )


def reviewed_row(row: dict[str, object], **kwargs: object) -> dict[str, object]:
    return {**row, **annotation(**kwargs)}


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_bytes(reviewer.serialize_records(records))


def test_next_unreviewed_resume_wraps_deterministically() -> None:
    records = [
        reviewed_row(synthetic_row("1", "Synthetic reviewed narrative one.")),
        synthetic_row("2", "Synthetic unreviewed narrative two."),
        reviewed_row(synthetic_row("3", "Synthetic reviewed narrative three.")),
    ]

    assert reviewer.next_unreviewed_index(records) == 1
    assert reviewer.next_unreviewed_index(records, start_index=2) == 1
    assert reviewer.next_unreviewed_index(
        [reviewed_row(record) for record in records]
    ) is None


def test_save_reload_and_resume_preserve_other_annotations(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    first = synthetic_row("1", "Synthetic first narrative.")
    second = reviewed_row(
        synthetic_row("2", "Synthetic second narrative."),
        category="UNSUPPORTED",
        intents=[],
        confidence="MEDIUM",
    )
    third = synthetic_row("3", "Synthetic third narrative.")
    write_records(path, [first, second, third])

    records = reviewer.load_records(path)
    saved = reviewer.save_annotation(path, records, 0, annotation())
    reloaded = reviewer.load_records(path)

    assert reloaded == saved
    assert reloaded[0]["review_category"] == "SINGLE_SUPPORTED_INTENT"
    assert reloaded[1] == second
    assert reviewer.next_unreviewed_index(reloaded) == 2


@pytest.mark.parametrize(
    ("category", "intents", "confidence", "message"),
    [
        ("SINGLE_SUPPORTED_INTENT", [], "HIGH", "exactly one"),
        (
            "SINGLE_SUPPORTED_INTENT",
            ["account_balance", "card_status"],
            "HIGH",
            "exactly one",
        ),
        ("MULTI_SUPPORTED_INTENT", ["account_balance"], "HIGH", "at least two"),
        ("UNSUPPORTED", ["account_balance"], "HIGH", "empty supported_intents"),
        (
            "SINGLE_SUPPORTED_INTENT",
            ["unsupported_or_uncertain"],
            "HIGH",
            "unsupported supported_intents",
        ),
        ("SINGLE_SUPPORTED_INTENT", ["account_balance"], "CERTAIN", "confidence"),
    ],
)
def test_invalid_annotation_combinations_are_rejected(
    category: str,
    intents: list[str],
    confidence: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        annotation(category=category, intents=intents, confidence=confidence)


def test_skip_navigation_does_not_modify_unreviewed_or_reviewed_rows() -> None:
    unreviewed = synthetic_row("1", "Synthetic unreviewed narrative.")
    reviewed = reviewed_row(synthetic_row("2", "Synthetic reviewed narrative."))
    records = [unreviewed, reviewed]
    before = json.loads(json.dumps(records))

    destination = reviewer.next_index(0, len(records))

    assert destination == 1
    assert records == before
    assert records[0]["adjudication_status"] == "UNREVIEWED"
    assert records[1]["adjudication_status"] == "REVIEWED"


def test_load_display_and_navigation_create_no_automatic_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "review.jsonl"
    row = synthetic_row(
        "1",
        "My card was stolen and an earlier charge was disputed.",
        mapping_status="AMBIGUOUS",
        candidates=["freeze_card", "create_dispute"],
    )
    write_records(path, [row])

    loaded = reviewer.load_records(path)
    displayed = reviewer.display_safe_metadata(
        loaded[reviewer.next_index(0, len(loaded))],
        index=0,
        progress=reviewer.progress_summary(loaded),
    )

    assert displayed["candidate_intents_hints_only"] == [
        "freeze_card",
        "create_dispute",
    ]
    assert loaded[0]["review_category"] is None
    assert loaded[0]["supported_intents"] == []
    assert loaded[0]["annotation_confidence"] is None
    assert loaded[0]["annotation_note"] == ""


def test_display_allowlist_hides_classifier_like_fields() -> None:
    row = {
        **synthetic_row("1", "Synthetic narrative with hidden model metadata."),
        "classifier_prediction": "SECRET_PREDICTION",
        "svm_margin": "SECRET_MARGIN",
        "logistic_top_1_probability": "SECRET_PROBABILITY",
        "abstention_decision": "SECRET_ABSTENTION",
        "model_output_custom": "SECRET_CUSTOM_OUTPUT",
    }
    records = [row]
    metadata = reviewer.display_safe_metadata(
        row,
        index=0,
        progress=reviewer.progress_summary(records),
    )
    output: list[str] = []

    reviewer.display_record(
        row,
        index=0,
        progress=reviewer.progress_summary(records),
        output=output.append,
    )

    rendered = "\n".join(output)
    assert not any(reviewer.is_model_related_field(key) for key in metadata)
    assert "Candidate intents (HINTS only)" in rendered
    assert "SECRET_" not in rendered


def test_atomic_write_replaces_with_complete_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    path.write_text("old content\n", encoding="utf-8")
    records = [
        synthetic_row("1", "Synthetic first complete row."),
        synthetic_row("2", "Synthetic second complete row."),
    ]

    reviewer.atomic_write_records(path, records)

    assert reviewer.load_records(path) == records
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_atomic_write_failure_preserves_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "review.jsonl"
    original = b"original bytes\n"
    path.write_bytes(original)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"synthetic replace failure: {source} -> {destination}")

    monkeypatch.setattr(reviewer.os, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        reviewer.atomic_write_records(
            path,
            [synthetic_row("1", "Synthetic interrupted write row.")],
        )

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_progress_summary_counts_initial_review_and_adjudication() -> None:
    records = [
        synthetic_row("1", "Synthetic unreviewed narrative."),
        reviewed_row(synthetic_row("2", "Synthetic reviewed narrative.")),
        reviewed_row(
            synthetic_row("3", "Synthetic adjudication narrative."),
            status="NEEDS_ADJUDICATION",
            secondary=True,
        ),
        reviewed_row(
            synthetic_row("4", "Synthetic adjudicated narrative."),
            status="ADJUDICATED",
        ),
    ]

    assert reviewer.progress_summary(records) == {
        "total": 4,
        "reviewed": 3,
        "unreviewed": 1,
        "needs_adjudication": 1,
        "adjudicated": 1,
    }


def test_protected_intents_appear_only_when_explicitly_selected() -> None:
    row = synthetic_row(
        "1",
        "A synthetic report mentions theft, fraud, and an old dispute.",
        mapping_status="AMBIGUOUS",
        candidates=["freeze_card", "create_dispute"],
    )

    unchanged = reviewer.apply_annotation(
        [row],
        0,
        annotation(category="UNSUPPORTED", intents=[]),
    )
    explicit = reviewer.apply_annotation(
        [row],
        0,
        annotation(
            category="MULTI_SUPPORTED_INTENT",
            intents=["freeze_card", "create_dispute"],
        ),
    )

    assert unchanged[0]["supported_intents"] == []
    assert explicit[0]["supported_intents"] == [
        "create_dispute",
        "freeze_card",
    ]


def test_save_preserves_unknown_fields_and_record_order(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    records = [
        synthetic_row("1", "Synthetic first narrative."),
        synthetic_row("2", "Synthetic second narrative."),
    ]
    records[0]["future_schema_field"] = ["untouched", {"value": 7}]
    write_records(path, records)

    saved = reviewer.save_annotation(path, records, 0, annotation())

    assert [row["complaint_id"] for row in saved] == ["1", "2"]
    assert saved[0]["future_schema_field"] == ["untouched", {"value": 7}]
    assert saved[1] == records[1]


def test_navigation_helpers_wrap_and_validate_jumps() -> None:
    assert reviewer.next_index(2, 3) == 0
    assert reviewer.previous_index(0, 3) == 2
    assert reviewer.jump_index(2, 3) == 1
    with pytest.raises(IndexError, match="between 1 and 3"):
        reviewer.jump_index(0, 3)
