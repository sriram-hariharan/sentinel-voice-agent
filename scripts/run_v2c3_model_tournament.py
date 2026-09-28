"""Prepare and run the frozen SentinelVoice V2-C3 Step 4 tournament.

Only ``v2c3_development_dataset.json`` is accepted as model-development data.
The final lockbox, challenge set, historical tests, and CFPB data are never
opened by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_DIRECTORY = REPOSITORY_ROOT / "data/evals/v2/ml"
CONFIG_PATH = ML_DIRECTORY / "v2c3_tournament_config.json"
REPORT_SCHEMA_VERSION = "v2c3-model-tournament-report.v1"
FOLD_SCHEMA_VERSION = "v2c3-cv-folds.v1"
CACHE_SCHEMA_VERSION = "v2c3-bge-cache.v1"
UNSUPPORTED = "unsupported_or_uncertain"


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def repository_path(relative_path: str) -> Path:
    path = (REPOSITORY_ROOT / relative_path).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(payload))


def ordered_ids_sha256(examples: Sequence[dict[str, Any]]) -> str:
    return sha256_bytes(stable_json_bytes([row["example_id"] for row in examples]))


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["schema_version"] != "v2c3-model-tournament-config.v1":
        raise ValueError("unexpected tournament config schema")
    if config["training_or_evaluation_performed"] is not False:
        raise ValueError("the frozen config cannot contain experiment results")
    if config["random_seed"] != 20260928:
        raise ValueError("tournament seed changed")
    cv = config["cv"]
    if (
        cv["method"] != "StratifiedGroupKFold"
        or cv["n_splits"] != 5
        or cv["shuffle"] is not True
        or cv["random_state"] != 20260928
        or cv["fallback_allowed"] is not False
    ):
        raise ValueError("the frozen five-fold decision changed")
    candidates = config["candidates"]
    if len(candidates) != 14 or len({row["id"] for row in candidates}) != 14:
        raise ValueError("the tournament must contain exactly 14 unique candidates")
    for candidate in candidates:
        if candidate["architecture"] not in config["architectures"]:
            raise ValueError(f"unknown architecture for {candidate['id']}")
        if candidate["representation"] not in config["representations"]:
            raise ValueError(f"unknown representation for {candidate['id']}")
        if candidate["classifier"] not in config["classifiers"]:
            raise ValueError(f"unknown classifier for {candidate['id']}")
    expected_candidate_ids = [
        "direct_v2c1_tfidf_v2c1_svc_anchor",
        "direct_v2c1_tfidf_balanced_svc",
        "direct_v2c1_tfidf_balanced_logreg",
        "direct_v2c1_tfidf_balanced_sgd",
        "direct_v2c1_tfidf_balanced_ridge",
        "direct_word_tfidf_balanced_svc",
        "direct_char_tfidf_balanced_svc",
        "direct_alt_tfidf_balanced_svc",
        "direct_lsa_balanced_svc",
        "direct_lsa_balanced_logreg",
        "direct_bge_balanced_svc",
        "direct_bge_balanced_logreg",
        "hierarchical_v2c1_tfidf_balanced_svc",
        "hierarchical_bge_balanced_logreg",
    ]
    if [row["id"] for row in candidates] != expected_candidate_ids:
        raise ValueError("the exact frozen candidate matrix or ordering changed")
    anchor = candidates[0]
    if (
        anchor["architecture"] != "direct_9_way"
        or anchor["representation"] != "v2c1_word_char_tfidf"
        or anchor["classifier"] != "v2c1_linear_svc_anchor"
    ):
        raise ValueError("the V2-C1 anchor candidate changed")


def validate_frozen_input(path_spec: dict[str, Any]) -> Path:
    path = repository_path(path_spec["path"])
    actual = sha256_file(path)
    if actual != path_spec["sha256"]:
        raise ValueError(f"frozen input hash mismatch for {path_spec['path']}: {actual}")
    return path


def load_development_data(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    spec = config["development_dataset"]
    path = validate_frozen_input(spec)
    dataset = read_json(path)
    if dataset["schema_version"] != spec["schema_version"]:
        raise ValueError("development dataset schema changed")
    examples = dataset["examples"]
    if len(examples) != spec["example_count"] or dataset["example_count"] != len(examples):
        raise ValueError("development example count changed")
    if len({row["example_id"] for row in examples}) != len(examples):
        raise ValueError("development example IDs are not unique")
    allowed_sources = set(spec["allowed_source_ids"])
    taxonomy = set(config["intent_taxonomy"])
    for row in examples:
        if row["data_role"] != spec["allowed_data_role"]:
            raise ValueError(f"non-development row encountered: {row['example_id']}")
        if row["source_id"] not in allowed_sources:
            raise ValueError(f"unapproved source encountered: {row['source_id']}")
        if row["intent"] not in taxonomy:
            raise ValueError(f"unknown intent encountered: {row['intent']}")
        if not row["text"].strip() or not row["group_id"]:
            raise ValueError(f"invalid development row: {row['example_id']}")
    if len({row["group_id"] for row in examples}) != spec["group_count"]:
        raise ValueError("development group count changed")
    return dataset, examples, path


def group_counts_by_intent(examples: Sequence[dict[str, Any]]) -> dict[str, int]:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        groups[row["intent"]].add(row["group_id"])
    return dict(sorted((label, len(values)) for label, values in groups.items()))


def validate_five_fold_feasibility(
    examples: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, int]:
    targets_by_group: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        targets_by_group[row["group_id"]].add(row["intent"])
    cross_target_groups = sorted(
        group_id for group_id, targets in targets_by_group.items() if len(targets) != 1
    )
    if cross_target_groups:
        raise ValueError(
            "development groups cannot span intent targets: "
            f"{cross_target_groups[:5]}"
        )
    counts = group_counts_by_intent(examples)
    taxonomy = config["intent_taxonomy"]
    if set(counts) != set(taxonomy):
        raise ValueError("development data does not contain the frozen taxonomy")
    insufficient = {label: count for label, count in counts.items() if count < 5}
    if insufficient:
        raise ValueError(f"five group-aware folds are infeasible: {insufficient}")
    return counts


def fold_assignments_payload(
    examples: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Build text-free assignments. This performs no feature or model fitting."""
    validate_five_fold_feasibility(examples, config)
    labels = np.asarray([row["intent"] for row in examples])
    groups = np.asarray([row["group_id"] for row in examples])
    placeholder = np.zeros(len(examples), dtype=np.int8)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config["cv"]["random_state"],
    )
    fold_by_index = np.full(len(examples), -1, dtype=np.int8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(placeholder, labels, groups)
    ):
        fold_by_index[validation_indices] = fold
    if np.any(fold_by_index < 0):
        raise ValueError("at least one example received no validation fold")
    assignments = [
        {"example_id": row["example_id"], "validation_fold": int(fold_by_index[index])}
        for index, row in enumerate(examples)
    ]
    taxonomy = config["intent_taxonomy"]
    fold_summaries = _fold_summaries(examples, fold_by_index, taxonomy)
    payload = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "development_dataset_path": config["development_dataset"]["path"],
        "development_dataset_sha256": config["development_dataset"]["sha256"],
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": config["cv"]["random_state"],
        "five_fold_feasibility_decided_before_model_scores": True,
        "model_scores_available_when_prepared": False,
        "training_or_evaluation_performed": False,
        "example_count": len(examples),
        "group_count": len(set(groups.tolist())),
        "independent_group_counts_by_intent": group_counts_by_intent(examples),
        "fold_summaries": fold_summaries,
        "assignments": assignments,
    }
    payload["assignment_sha256"] = sha256_bytes(stable_json_bytes(assignments))
    return payload


def _fold_summaries(
    examples: Sequence[dict[str, Any]], folds: np.ndarray, taxonomy: Sequence[str]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for fold in range(5):
        rows = [row for index, row in enumerate(examples) if folds[index] == fold]
        intent_counts = Counter(row["intent"] for row in rows)
        if set(intent_counts) != set(taxonomy):
            raise ValueError(f"fold {fold} does not contain every intent")
        summaries.append(
            {
                "validation_fold": fold,
                "example_count": len(rows),
                "group_count": len({row["group_id"] for row in rows}),
                "counts_by_intent": {label: intent_counts[label] for label in taxonomy},
                "counts_by_source": dict(
                    sorted(Counter(row["source_id"] for row in rows).items())
                ),
                "counts_by_source_and_intent": {
                    source_id: dict(
                        sorted(
                            Counter(
                                row["intent"]
                                for row in rows
                                if row["source_id"] == source_id
                            ).items()
                        )
                    )
                    for source_id in sorted({row["source_id"] for row in rows})
                },
            }
        )
    return summaries


def validate_fold_artifact(
    artifact: dict[str, Any],
    examples: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> np.ndarray:
    expected_header = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "development_dataset_sha256": config["development_dataset"]["sha256"],
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": config["cv"]["random_state"],
        "five_fold_feasibility_decided_before_model_scores": True,
        "model_scores_available_when_prepared": False,
        "training_or_evaluation_performed": False,
    }
    for field, expected in expected_header.items():
        if artifact.get(field) != expected:
            raise ValueError(f"frozen fold artifact field changed: {field}")
    assignments = artifact["assignments"]
    if sha256_bytes(stable_json_bytes(assignments)) != artifact["assignment_sha256"]:
        raise ValueError("fold assignment hash mismatch")
    fold_by_id = {row["example_id"]: row["validation_fold"] for row in assignments}
    expected_ids = [row["example_id"] for row in examples]
    if len(fold_by_id) != len(assignments) or set(fold_by_id) != set(expected_ids):
        raise ValueError("fold assignments do not cover each development example exactly once")
    folds = np.asarray([fold_by_id[example_id] for example_id in expected_ids], dtype=np.int8)
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("fold assignments must use exactly folds 0 through 4")
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row, fold in zip(examples, folds, strict=True):
        group_folds[row["group_id"]].add(int(fold))
    if any(len(values) != 1 for values in group_folds.values()):
        raise ValueError("a development group crosses validation folds")
    if artifact.get("example_count") != len(examples):
        raise ValueError("fold artifact example count changed")
    group_count = len(group_folds)
    if artifact.get("group_count") != group_count:
        raise ValueError("fold artifact group count changed")
    expected_group_counts = group_counts_by_intent(examples)
    if artifact.get("independent_group_counts_by_intent") != expected_group_counts:
        raise ValueError("fold artifact independent-group counts changed")
    expected_summaries = _fold_summaries(examples, folds, config["intent_taxonomy"])
    if artifact.get("fold_summaries") != expected_summaries:
        raise ValueError("fold artifact summaries do not match its assignments")
    return folds


def prepare_folds(config: dict[str, Any], examples: Sequence[dict[str, Any]]) -> Path:
    payload = fold_assignments_payload(examples, config)
    path = repository_path(config["cv"]["artifact_path"])
    write_json(path, payload)
    return path


def _tfidf(spec: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        input=spec["input"],
        encoding=spec["encoding"],
        decode_error=spec["decode_error"],
        strip_accents=spec["strip_accents"],
        preprocessor=spec["preprocessor"],
        tokenizer=spec["tokenizer"],
        analyzer=spec["analyzer"],
        stop_words=spec["stop_words"],
        token_pattern=spec["token_pattern"],
        ngram_range=tuple(spec["ngram_range"]),
        sublinear_tf=spec["sublinear_tf"],
        min_df=spec["min_df"],
        max_df=spec["max_df"],
        max_features=spec["max_features"],
        vocabulary=spec["vocabulary"],
        binary=spec["binary"],
        norm=spec["norm"],
        use_idf=spec["use_idf"],
        smooth_idf=spec["smooth_idf"],
        lowercase=spec["lowercase"],
        dtype=np.float64,
    )


def build_representation(name: str, config: dict[str, Any]) -> BaseEstimator:
    spec = config["representations"][name]
    if spec["family"] == "frozen_local_sentence_embedding":
        raise ValueError("BGE uses the separately validated frozen feature cache")
    if name == "word_tfidf":
        return _tfidf(spec["word_tfidf"])
    if name == "char_tfidf":
        return _tfidf(spec["character_tfidf"])
    base_name = spec.get("tfidf_base", name)
    base = config["representations"][base_name]
    union = FeatureUnion(
        [
            ("word", _tfidf(base["word_tfidf"])),
            ("character", _tfidf(base["character_tfidf"])),
        ]
    )
    if name != "tfidf_lsa":
        return union
    svd = spec["truncated_svd"]
    normalizer = spec["normalizer"]
    return Pipeline(
        [
            ("tfidf", union),
            (
                "truncated_svd",
                TruncatedSVD(
                    n_components=svd["n_components"],
                    algorithm=svd["algorithm"],
                    n_iter=svd["n_iter"],
                    random_state=svd["random_state"],
                ),
            ),
            ("normalize", Normalizer(norm=normalizer["norm"], copy=normalizer["copy"])),
        ]
    )


def build_classifier(name: str, config: dict[str, Any]) -> BaseEstimator:
    spec = config["classifiers"][name]
    parameters = dict(spec["parameters"])
    classes: dict[str, type[BaseEstimator]] = {
        "LinearSVC": LinearSVC,
        "LogisticRegression": LogisticRegression,
        "SGDClassifier": SGDClassifier,
        "RidgeClassifier": RidgeClassifier,
    }
    return classes[spec["class"]](**parameters)


class HierarchicalClassifier:
    """Two-stage classifier with deterministic nine-intent output."""

    def __init__(self, stage_1: BaseEstimator, stage_2: BaseEstimator) -> None:
        self.stage_1 = stage_1
        self.stage_2 = stage_2

    def predict(self, features: Any) -> np.ndarray:
        stage_1_predictions = np.asarray(self.stage_1.predict(features))
        predictions = np.full(len(stage_1_predictions), UNSUPPORTED, dtype=object)
        supported_indices = np.flatnonzero(stage_1_predictions == "supported")
        if supported_indices.size:
            subset = _subset_features(features, supported_indices)
            predictions[supported_indices] = self.stage_2.predict(subset)
        return predictions


def _subset_features(features: Any, indices: np.ndarray) -> Any:
    if isinstance(features, np.ndarray):
        return features[indices]
    return [features[int(index)] for index in indices]


def fit_candidate(
    candidate: dict[str, Any],
    train_features: Any,
    train_labels: np.ndarray,
    config: dict[str, Any],
) -> BaseEstimator | HierarchicalClassifier:
    """Fit fresh fold-local transforms and estimators for one candidate."""
    is_bge = candidate["representation"] == "bge_small_en_v1_5"

    def new_model() -> BaseEstimator:
        classifier = build_classifier(candidate["classifier"], config)
        if is_bge:
            return classifier
        return Pipeline(
            [
                ("representation", build_representation(candidate["representation"], config)),
                ("classifier", classifier),
            ]
        )

    if candidate["architecture"] == "direct_9_way":
        model = new_model()
        model.fit(train_features, train_labels)
        return model
    if candidate["architecture"] != "hierarchical_supported_then_intent":
        raise ValueError(f"unsupported architecture: {candidate['architecture']}")
    stage_1_labels = np.where(train_labels == UNSUPPORTED, UNSUPPORTED, "supported")
    stage_1 = new_model()
    stage_1.fit(train_features, stage_1_labels)
    supported_indices = np.flatnonzero(train_labels != UNSUPPORTED)
    if not supported_indices.size:
        raise ValueError("hierarchical Stage 2 has no supported training examples")
    stage_2 = new_model()
    stage_2.fit(
        _subset_features(train_features, supported_indices),
        train_labels[supported_indices],
    )
    return HierarchicalClassifier(stage_1, stage_2)


def classification_metrics(
    expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    report = classification_report(
        expected,
        predicted,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(expected, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": {
            "label_order": list(labels),
            "values": confusion_matrix(expected, predicted, labels=list(labels)).tolist(),
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def safety_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    protected = set(config["protected_write_intents"])
    supported = set(config["supported_intents"])
    pairs = list(zip(expected, predicted, strict=True))
    unsupported_total = sum(gold == UNSUPPORTED for gold, _ in pairs)
    false_supported = sum(gold == UNSUPPORTED and guess != UNSUPPORTED for gold, guess in pairs)
    protected_total = sum(gold in protected for gold, _ in pairs)
    protected_exact = sum(gold in protected and guess == gold for gold, guess in pairs)
    non_protected_total = sum(gold not in protected for gold, _ in pairs)
    protected_false_positives = sum(
        gold not in protected and guess in protected for gold, guess in pairs
    )
    supported_pairs = [(gold, guess) for gold, guess in pairs if gold in supported]
    return {
        "unsupported_or_uncertain_recall": _ratio(
            sum(gold == UNSUPPORTED and guess == UNSUPPORTED for gold, guess in pairs),
            unsupported_total,
        ),
        "false_supported_count": false_supported,
        "false_supported_rate": _ratio(false_supported, unsupported_total),
        "supported_intent_accuracy": _ratio(
            sum(gold == guess for gold, guess in supported_pairs), len(supported_pairs)
        ),
        "protected_write_recall": _ratio(protected_exact, protected_total),
        "protected_write_false_positive_count": protected_false_positives,
        "protected_write_false_positive_rate": _ratio(
            protected_false_positives, non_protected_total
        ),
    }


def pooled_metrics(
    expected: Sequence[str], predicted: Sequence[str], config: dict[str, Any]
) -> dict[str, Any]:
    result = classification_metrics(expected, predicted, config["intent_taxonomy"])
    result.update(safety_metrics(expected, predicted, config))
    result["freeze_card_metrics"] = result["per_class"]["freeze_card"]
    result["create_dispute_metrics"] = result["per_class"]["create_dispute"]
    return result


def source_breakdowns(
    examples: Sequence[dict[str, Any]],
    predicted: Sequence[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    for source_id in config["metrics"]["source_breakdown_source_ids"]:
        indices = [index for index, row in enumerate(examples) if row["source_id"] == source_id]
        expected = [examples[index]["intent"] for index in indices]
        guesses = [predicted[index] for index in indices]
        present_labels = [label for label in config["intent_taxonomy"] if label in set(expected)]
        result = {
            "count": len(indices),
            "accuracy": float(accuracy_score(expected, guesses)),
            "macro_f1_over_classes_present": float(
                f1_score(expected, guesses, labels=present_labels, average="macro", zero_division=0)
            ),
        }
        safety = safety_metrics(expected, guesses, config)
        result["unsupported_or_uncertain_recall"] = safety["unsupported_or_uncertain_recall"]
        result["false_supported_rate"] = safety["false_supported_rate"]
        breakdown[source_id] = result
    return breakdown


def apply_safety_gates(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["safety_gates"]
    results = {
        "protected_write_false_positive_rate": {
            "value": metrics["protected_write_false_positive_rate"],
            "operator": "<=",
            "threshold": gates["protected_write_false_positive_rate_maximum"],
            "passed": metrics["protected_write_false_positive_rate"]
            <= gates["protected_write_false_positive_rate_maximum"],
        },
        "protected_write_recall": {
            "value": metrics["protected_write_recall"],
            "operator": ">=",
            "threshold": gates["protected_write_recall_minimum"],
            "passed": metrics["protected_write_recall"]
            >= gates["protected_write_recall_minimum"],
        },
        "unsupported_or_uncertain_recall": {
            "value": metrics["unsupported_or_uncertain_recall"],
            "operator": ">=",
            "threshold": gates["unsupported_or_uncertain_recall_minimum"],
            "passed": metrics["unsupported_or_uncertain_recall"]
            >= gates["unsupported_or_uncertain_recall_minimum"],
        },
    }
    return {"eligible": all(row["passed"] for row in results.values()), "gates": results}


def cache_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    spec = config["representations"]["bge_small_en_v1_5"]
    return repository_path(spec["cache_path"]), repository_path(spec["cache_metadata_path"])


def validate_embedding_cache(
    config: dict[str, Any], examples: Sequence[dict[str, Any]]
) -> np.ndarray:
    cache_path, metadata_path = cache_paths(config)
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "validated BGE development cache is required; run prepare-embeddings first"
        )
    metadata = read_json(metadata_path)
    representation = config["representations"]["bge_small_en_v1_5"]
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_identifier": representation["model_identifier"],
        "development_dataset_sha256": config["development_dataset"]["sha256"],
        "ordered_example_ids_sha256": ordered_ids_sha256(examples),
        "example_count": len(examples),
        "dimensions": representation["dimensions"],
        "label_data_used": False,
        "fine_tuning_performed": False,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"BGE cache metadata mismatch: {field}")
    if sha256_file(cache_path) != metadata["cache_file_sha256"]:
        raise ValueError("BGE cache file hash mismatch")
    with np.load(cache_path, allow_pickle=False) as cache:
        if set(cache.files) != {"embeddings", "example_ids"}:
            raise ValueError("unexpected arrays in BGE cache")
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        example_ids = cache["example_ids"].astype(str).tolist()
    if example_ids != [row["example_id"] for row in examples]:
        raise ValueError("BGE cache example order changed")
    if embeddings.shape != (len(examples), representation["dimensions"]):
        raise ValueError("BGE cache shape mismatch")
    if not np.isfinite(embeddings).all():
        raise ValueError("BGE cache contains non-finite values")
    return embeddings


def prepare_embeddings(
    config: dict[str, Any], examples: Sequence[dict[str, Any]]
) -> tuple[Path, Path]:
    """Generate label-free frozen features for development examples only."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for prepare-embeddings") from exc
    spec = config["representations"]["bge_small_en_v1_5"]
    try:
        model = TextEmbedding(model_name=spec["model_identifier"])
    except Exception as exc:
        raise RuntimeError(
            f"FastEmbed could not load {spec['model_identifier']}; no substitute is allowed"
        ) from exc
    texts = [row["text"] for row in examples]
    try:
        vectors = list(model.passage_embed(texts, batch_size=spec["batch_size"]))
    except Exception as exc:
        raise RuntimeError("FastEmbed failed while generating the development cache") from exc
    embeddings = np.asarray(vectors, dtype=np.float32)
    expected_shape = (len(examples), spec["dimensions"])
    if embeddings.shape != expected_shape or not np.isfinite(embeddings).all():
        raise ValueError(f"unexpected BGE embedding matrix: {embeddings.shape}")
    cache_path, metadata_path = cache_paths(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_path,
            embeddings=embeddings,
            example_ids=np.asarray([row["example_id"] for row in examples]),
        )
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_identifier": spec["model_identifier"],
        "development_dataset_sha256": config["development_dataset"]["sha256"],
        "ordered_example_ids_sha256": ordered_ids_sha256(examples),
        "example_count": len(examples),
        "dimensions": spec["dimensions"],
        "dtype": "float32",
        "cache_file_sha256": sha256_file(cache_path),
        "label_data_used": False,
        "fine_tuning_performed": False,
        "source_scope": "v2c3_development_dataset_only",
        "training_or_evaluation_performed": False,
        "fastembed_version": importlib.metadata.version("fastembed"),
    }
    write_json(metadata_path, metadata)
    validate_embedding_cache(config, examples)
    return cache_path, metadata_path


def latency_sample_indices(
    examples: Sequence[dict[str, Any]], config: dict[str, Any]
) -> np.ndarray:
    seed = str(config["random_seed"])
    ranked = sorted(
        range(len(examples)),
        key=lambda index: (
            hashlib.sha256(f"{seed}:{examples[index]['example_id']}".encode()).hexdigest(),
            examples[index]["example_id"],
        ),
    )
    count = config["latency_measurement_policy"]["sample_count"]
    return np.asarray(ranked[:count], dtype=np.int64)


def _latency_summary(durations: Sequence[float]) -> dict[str, float]:
    milliseconds = np.asarray(durations) * 1000.0
    return {
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p90_ms": float(np.percentile(milliseconds, 90)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
    }


def _measure_calls(calls: Sequence[Callable[[], Any]], warmups: int) -> dict[str, float]:
    if not calls:
        raise ValueError("latency sample cannot be empty")
    for index in range(warmups):
        calls[index % len(calls)]()
    durations: list[float] = []
    for call in calls:
        started = time.perf_counter()
        call()
        durations.append(time.perf_counter() - started)
    return _latency_summary(durations)


def serialized_size_bytes(model: Any) -> int:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        path = Path(handle.name)
    try:
        joblib.dump(model, path)
        return path.stat().st_size
    finally:
        path.unlink(missing_ok=True)


def measure_full_development_candidate(
    candidate: dict[str, Any],
    examples: Sequence[dict[str, Any]],
    labels: np.ndarray,
    config: dict[str, Any],
    embeddings: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    is_bge = candidate["representation"] == "bge_small_en_v1_5"
    features: Any = embeddings if is_bge else [row["text"] for row in examples]
    if is_bge and embeddings is None:
        raise ValueError("BGE cache was not loaded")
    model = fit_candidate(candidate, features, labels, config)
    sample_indices = latency_sample_indices(examples, config)
    policy = config["latency_measurement_policy"]
    warmups = policy["warmup_calls"]
    if not is_bge:
        calls = [
            (lambda index=int(index): model.predict([examples[index]["text"]]))
            for index in sample_indices
        ]
        latency = {
            "primary_scope": "representation_and_classifier",
            "representation_and_classifier": _measure_calls(calls, warmups),
        }
        size = {
            "serialized_fitted_candidate_bytes": serialized_size_bytes(model),
            "scope": "fold-independent full-development fitted representation and classifier",
        }
        return latency, size

    downstream_calls = [
        (lambda index=int(index): model.predict(embeddings[index : index + 1]))
        for index in sample_indices
    ]
    downstream = _measure_calls(downstream_calls, warmups)
    try:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(
            model_name=config["representations"]["bge_small_en_v1_5"]["model_identifier"]
        )
    except Exception as exc:
        raise RuntimeError("the frozen BGE model is unavailable for end-to-end latency") from exc

    def end_to_end(index: int) -> np.ndarray:
        vector = np.asarray(
            list(embedding_model.passage_embed([examples[index]["text"]])), dtype=np.float32
        )
        return model.predict(vector)

    end_to_end_calls = [lambda index=int(index): end_to_end(index) for index in sample_indices]
    latency = {
        "primary_scope": "end_to_end_local_embedding_and_classifier",
        "downstream_classifier_on_cached_embedding": downstream,
        "end_to_end_local_embedding_and_classifier": _measure_calls(end_to_end_calls, warmups),
    }
    size = {
        "serialized_downstream_classifier_bytes": serialized_size_bytes(model),
        "frozen_embedding_model_weight_cache_bytes": None,
        "embedding_model_weight_cache_measurement": (
            "not reliably exposed by FastEmbed; not guessed"
        ),
        "total_runtime_bytes": None,
        "scope": "classifier bytes are not the total BGE runtime model size",
    }
    return latency, size


def run_candidate_cv(
    candidate: dict[str, Any],
    examples: Sequence[dict[str, Any]],
    folds: np.ndarray,
    config: dict[str, Any],
    embeddings: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    labels = np.asarray([row["intent"] for row in examples])
    texts = [row["text"] for row in examples]
    oof = np.full(len(examples), "", dtype=object)
    fold_metrics: list[dict[str, Any]] = []
    is_bge = candidate["representation"] == "bge_small_en_v1_5"
    if is_bge and embeddings is None:
        raise ValueError("BGE candidate requires the validated cache")
    for fold in range(5):
        validation_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        train_features = (
            embeddings[train_indices]
            if is_bge
            else [texts[int(index)] for index in train_indices]
        )
        validation_features = (
            embeddings[validation_indices]
            if is_bge
            else [texts[int(index)] for index in validation_indices]
        )
        model = fit_candidate(candidate, train_features, labels[train_indices], config)
        predictions = np.asarray(model.predict(validation_features), dtype=object)
        oof[validation_indices] = predictions
        metrics = classification_metrics(
            labels[validation_indices], predictions, config["intent_taxonomy"]
        )
        fold_metrics.append(
            {
                "validation_fold": fold,
                "example_count": len(validation_indices),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    if np.any(oof == ""):
        raise ValueError("candidate did not produce exactly one OOF prediction per example")
    pooled = pooled_metrics(labels, oof, config)
    result = {
        "candidate_id": candidate["id"],
        "definition": expanded_candidate_definition(candidate, config),
        "fold_metrics": fold_metrics,
        "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_metrics])),
        "macro_f1_standard_deviation": float(
            np.std([row["macro_f1"] for row in fold_metrics], ddof=0)
        ),
        "macro_f1_variance": float(
            np.var([row["macro_f1"] for row in fold_metrics], ddof=0)
        ),
        "pooled_oof_metrics": pooled,
        "source_breakdowns": source_breakdowns(examples, oof, config),
        "safety_gate_result": apply_safety_gates(pooled, config),
        "oof_prediction_count": len(oof),
        "raw_text_persisted": False,
        "oof_predictions_persisted": False,
    }
    return result, oof


def expanded_candidate_definition(
    candidate: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    return {
        **candidate,
        "architecture_definition": config["architectures"][candidate["architecture"]],
        "representation_definition": config["representations"][candidate["representation"]],
        "classifier_definition": config["classifiers"][candidate["classifier"]],
    }


def _tie_key(result: dict[str, Any]) -> tuple[Any, ...]:
    pooled = result["pooled_oof_metrics"]
    latency = result["latency"]
    primary_latency = (
        latency["end_to_end_local_embedding_and_classifier"]["p95_ms"]
        if latency["primary_scope"] == "end_to_end_local_embedding_and_classifier"
        else latency["representation_and_classifier"]["p95_ms"]
    )
    size = result["model_size"].get(
        "serialized_fitted_candidate_bytes",
        result["model_size"].get("serialized_downstream_classifier_bytes"),
    )
    architecture_complexity = (
        0 if result["definition"]["architecture"] == "direct_9_way" else 1
    )
    return (
        pooled["false_supported_rate"],
        -pooled["unsupported_or_uncertain_recall"],
        pooled["protected_write_false_positive_rate"],
        result["macro_f1_standard_deviation"],
        primary_latency,
        size,
        architecture_complexity,
        result["candidate_id"],
    )


def rank_eligible_candidates(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = [row for row in results if row["safety_gate_result"]["eligible"]]
    ranked: list[dict[str, Any]] = []
    while remaining:
        best_macro = max(row["macro_f1_mean"] for row in remaining)
        tie_pool = [row for row in remaining if best_macro - row["macro_f1_mean"] < 0.005]
        selected = min(tie_pool, key=_tie_key)
        ranked.append(selected)
        remaining.remove(selected)
    return ranked


def select_finalists(
    ranked: Sequence[dict[str, Any]], config: dict[str, Any]
) -> list[str]:
    maximum = config["finalist_policy"]["maximum_count"]
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    for result in ranked:
        representation = result["definition"]["representation"]
        family = config["representations"][representation]["family"]
        if family not in used_families:
            selected.append(result)
            used_families.add(family)
        if len(selected) == maximum:
            break
    if len(selected) < maximum:
        for result in ranked:
            if result not in selected:
                selected.append(result)
            if len(selected) == maximum:
                break
    return [row["candidate_id"] for row in selected]


def preflight(config: dict[str, Any], examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    contract_path = validate_frozen_input(config["experiment_contract"])
    group_counts = validate_five_fold_feasibility(examples, config)
    fold_path = repository_path(config["cv"]["artifact_path"])
    folds_prepared = fold_path.is_file()
    if folds_prepared:
        validate_fold_artifact(read_json(fold_path), examples, config)
    packages: dict[str, str | None] = {}
    for package in ("numpy", "scikit-learn", "fastembed", "joblib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "status": "ready" if packages["fastembed"] else "missing_fastembed",
        "training_or_evaluation_performed": False,
        "development_dataset_sha256": config["development_dataset"]["sha256"],
        "experiment_contract_sha256": sha256_file(contract_path),
        "candidate_count": len(config["candidates"]),
        "five_fold_feasible": True,
        "independent_group_counts_by_intent": group_counts,
        "frozen_fold_artifact_present": folds_prepared,
        "packages": packages,
    }


def run_tournament(
    config: dict[str, Any], examples: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], Path]:
    contract_path = validate_frozen_input(config["experiment_contract"])
    fold_path = repository_path(config["cv"]["artifact_path"])
    if not fold_path.is_file():
        raise FileNotFoundError("frozen fold artifact is required; run prepare-folds first")
    fold_artifact = read_json(fold_path)
    folds = validate_fold_artifact(fold_artifact, examples, config)
    needs_bge = any(
        candidate["representation"] == "bge_small_en_v1_5"
        for candidate in config["candidates"]
    )
    embeddings = validate_embedding_cache(config, examples) if needs_bge else None
    labels = np.asarray([row["intent"] for row in examples])
    results: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        result, _ = run_candidate_cv(candidate, examples, folds, config, embeddings)
        results.append(result)
    for candidate, result in zip(config["candidates"], results, strict=True):
        latency, size = measure_full_development_candidate(
            candidate, examples, labels, config, embeddings
        )
        result["latency"] = latency
        result["model_size"] = size
    ranked = rank_eligible_candidates(results)
    rank_by_id = {row["candidate_id"]: rank for rank, row in enumerate(ranked, start=1)}
    for result in results:
        result["eligible_rank"] = rank_by_id.get(result["candidate_id"])
    winner = ranked[0]["candidate_id"] if ranked else None
    finalist_ids = select_finalists(ranked, config)
    sample_indices = latency_sample_indices(examples, config)
    sample_ids = [examples[int(index)]["example_id"] for index in sample_indices]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_phase": "V2-C3-Step-4",
        "development_training_and_evaluation_performed": True,
        "development_dataset_sha256": config["development_dataset"]["sha256"],
        "tournament_config_sha256": sha256_file(CONFIG_PATH),
        "experiment_contract_sha256": sha256_file(contract_path),
        "fold_artifact_sha256": sha256_file(fold_path),
        "fold_assignment_sha256": fold_artifact["assignment_sha256"],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "fastembed": importlib.metadata.version("fastembed"),
            "joblib": importlib.metadata.version("joblib"),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "latency_measurement_policy": config["latency_measurement_policy"],
        "latency_sample_example_ids_sha256": sha256_bytes(stable_json_bytes(sample_ids)),
        "model_size_measurement_policy": config["model_size_measurement_policy"],
        "safety_gates": config["safety_gates"],
        "ranking_rules": config["ranking"],
        "finalist_policy": config["finalist_policy"],
        "candidate_definitions": [
            expanded_candidate_definition(candidate, config)
            for candidate in config["candidates"]
        ],
        "fold_summaries": fold_artifact["fold_summaries"],
        "candidate_results": results,
        "deterministic_eligible_ranking": [row["candidate_id"] for row in ranked],
        "eligible_winner": winner,
        "no_eligible_winner": winner is None,
        "finalist_shortlist": finalist_ids,
        "safety_gate_failed_candidates": [
            row["candidate_id"]
            for row in results
            if not row["safety_gate_result"]["eligible"]
        ],
        "final_lockbox_used": False,
        "challenge_set_used": False,
        "cfpb_used": False,
        "hyperparameter_search_performed": False,
        "raw_text_persisted": False,
        "final_test_metric_reported": False,
        "runtime_integration_performed": False,
    }
    report_path = repository_path(config["report"]["path"])
    write_json(report_path, report)
    return report, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "prepare-folds", "prepare-embeddings", "run")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    _, examples, _ = load_development_data(config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config, examples), indent=2, sort_keys=True))
        return
    if args.mode == "prepare-folds":
        path = prepare_folds(config, examples)
        print(f"wrote text-free frozen folds: {path.relative_to(REPOSITORY_ROOT)}")
        return
    if args.mode == "prepare-embeddings":
        cache_path, metadata_path = prepare_embeddings(config, examples)
        print(f"wrote local BGE cache: {cache_path.relative_to(REPOSITORY_ROOT)}")
        print(f"wrote cache metadata: {metadata_path.relative_to(REPOSITORY_ROOT)}")
        return
    _, report_path = run_tournament(config, examples)
    print(f"wrote tournament report: {report_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
