"""Run the frozen V2-C5 unsupervised intent-discovery experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score

try:
    from scripts import build_v2c5_discovery_corpus as corpus_builder
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import build_v2c5_discovery_corpus as corpus_builder


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPOSITORY_ROOT / "data/evals/v2/ml"
LOCAL_ROOT = ML_ROOT / "local"

CONTRACT_SHA256 = (
    "acc48bace76a5ae93ff9a1818856e3b2dc0ad3585c80e8ba7a3f582208a71427"
)
CORPUS_SHA256 = (
    "0a873fe5797fdd93689e7d5054b029cd7b9068017b0ecff54af000641e282e01"
)
CORPUS_MANIFEST_SHA256 = (
    "8917c6b27b59c672286672ea3ad915b0195d9ef937605662a01e4209bca4162d"
)
ORDERED_NORMALIZED_TEXT_SHA256 = (
    "6b3fd2f4e2ac3806860b6c8de76e514c756579a9d5a22a4552c2dad8c5017953"
)
ORDERED_DISCOVERY_ID_SHA256 = (
    "8d1087625ea9429df5c43e4820d88f6a17c7419e5b3a6fa606706b651640ca99"
)

EXPECTED_COUNT = 6372
DIMENSIONS = 384
MODEL_IDENTIFIER = "BAAI/bge-small-en-v1.5"
EMBEDDING_PROVIDER = "FastEmbed"
EMBEDDING_METHOD = "passage_embed"
PRIMARY_INTENT = "unsupported_or_uncertain"
UMAP_RANDOM_STATE = 20260928

REPORT_SCHEMA_VERSION = "v2c5-intent-discovery-report.v1"
ASSIGNMENTS_SCHEMA_VERSION = "v2c5-intent-discovery-assignments.v1"
MANIFEST_SCHEMA_VERSION = "v2c5-intent-discovery-manifest.v1"
CACHE_SCHEMA_VERSION = "v2c5-intent-discovery-bge-cache.v1"
UMAP_SCHEMA_VERSION = "v2c5-intent-discovery-umap.v1"

PRIMARY_CONFIG = {
    "allow_single_cluster": False,
    "cluster_selection_epsilon": 0.0,
    "cluster_selection_method": "eom",
    "metric": "euclidean",
    "min_cluster_size": 30,
    "min_samples": 10,
}
SENSITIVITY_CONFIGS = [
    {
        "allow_single_cluster": False,
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_method": "eom",
        "metric": "euclidean",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }
    for min_cluster_size in (15, 30, 60)
    for min_samples in (5, 10, 15)
]
PRIMARY_CONFIG_ID = "mcs30_ms10"
FORBIDDEN_INPUT_FILENAMES = corpus_builder.FORBIDDEN_INPUT_FILENAMES


@dataclass(frozen=True)
class ExperimentPaths:
    contract: Path
    corpus: Path
    corpus_manifest: Path
    embedding_cache: Path
    embedding_cache_metadata: Path
    report_output: Path
    assignments_output: Path
    manifest_output: Path
    umap_output: Path


DEFAULT_PATHS = ExperimentPaths(
    contract=ML_ROOT / "v2c5_intent_discovery_contract.json",
    corpus=ML_ROOT / "v2c5_discovery_corpus.json",
    corpus_manifest=ML_ROOT / "v2c5_discovery_corpus.manifest.json",
    embedding_cache=LOCAL_ROOT / "v2c5_bge_small_en_v1_5_l2.npz",
    embedding_cache_metadata=(
        LOCAL_ROOT / "v2c5_bge_small_en_v1_5_l2.metadata.json"
    ),
    report_output=ML_ROOT / "v2c5_intent_discovery_report.json",
    assignments_output=ML_ROOT / "v2c5_intent_discovery_assignments.json",
    manifest_output=ML_ROOT / "v2c5_intent_discovery.manifest.json",
    umap_output=ML_ROOT / "v2c5_intent_discovery_umap.json",
)


@dataclass(frozen=True)
class FrozenInputs:
    contract: dict[str, Any]
    corpus: dict[str, Any]
    corpus_manifest: dict[str, Any]
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class ClusterRun:
    config_id: str
    parameters: dict[str, Any]
    raw_labels: np.ndarray
    membership_values: np.ndarray | None
    canonical_labels: list[str]
    cluster_definitions: list[dict[str, Any]]


@dataclass(frozen=True)
class ResultArtifacts:
    report_bytes: bytes
    assignments_bytes: bytes
    manifest_bytes: bytes
    umap_bytes: bytes | None
    report_payload: dict[str, Any]
    assignments_payload: dict[str, Any]
    manifest_payload: dict[str, Any]


def stable_json_bytes(payload: Any) -> bytes:
    return corpus_builder.development_builder.stable_json_bytes(payload)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def guard_input_path(path: Path) -> None:
    if path.name in FORBIDDEN_INPUT_FILENAMES:
        raise ValueError(f"sealed V2-C4 holdout input is forbidden: {path.name}")


def read_input_bytes(path: Path) -> bytes:
    guard_input_path(path)
    return path.read_bytes()


def load_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    payload = json.loads(value.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def config_id(parameters: Mapping[str, Any]) -> str:
    return (
        f"mcs{parameters['min_cluster_size']}_"
        f"ms{parameters['min_samples']}"
    )


def ordered_lines_sha256(values: Sequence[str]) -> str:
    return sha256_bytes("".join(f"{value}\n" for value in values).encode())


def validate_contract(contract: Mapping[str, Any], raw: bytes) -> None:
    if sha256_bytes(raw) != CONTRACT_SHA256:
        raise ValueError("frozen V2-C5 contract SHA-256 changed")
    if contract.get("phase") != "V2-C5" or contract.get("status") != "frozen":
        raise ValueError("V2-C5 contract is not frozen")
    execution = contract.get("execution_status")
    if not isinstance(execution, dict) or execution.get("contract_frozen") is not True:
        raise ValueError("V2-C5 contract frozen flag is not true")
    representation = contract.get("representation")
    expected_representation = {
        "dimensions": DIMENSIONS,
        "external_api": False,
        "fine_tuning": False,
        "input_space": "original_384_dimensional_embedding_space",
        "l2_normalization_required": True,
        "method": EMBEDDING_METHOD,
        "model": MODEL_IDENTIFIER,
        "provider": EMBEDDING_PROVIDER,
        "umap_reduced_vectors_used_for_clustering": False,
    }
    if not isinstance(representation, dict):
        raise TypeError("frozen representation must be an object")
    for field, expected in expected_representation.items():
        if representation.get(field) != expected:
            raise ValueError(f"frozen representation changed: {field}")
    method = contract.get("primary_discovery_method")
    if not isinstance(method, dict) or method.get("algorithm") != "HDBSCAN":
        raise ValueError("primary discovery method changed")
    if method.get("parameters") != PRIMARY_CONFIG:
        raise ValueError("primary HDBSCAN configuration changed")
    sensitivity = contract.get("sensitivity_analysis")
    if not isinstance(sensitivity, dict):
        raise TypeError("sensitivity analysis must be an object")
    if sensitivity.get("configuration_count") != 9:
        raise ValueError("sensitivity configuration count changed")
    if sensitivity.get("grid") != {
        "min_cluster_size": [15, 30, 60],
        "min_samples": [5, 10, 15],
    }:
        raise ValueError("sensitivity grid changed")
    if sensitivity.get("primary_30_10_result_remains_primary") is not True:
        raise ValueError("primary sensitivity policy changed")
    if (
        sensitivity.get(
            "post_hoc_best_looking_configuration_selection_allowed"
        )
        is not False
    ):
        raise ValueError("post-hoc HDBSCAN selection became allowed")


def validate_corpus(
    corpus: Mapping[str, Any],
    raw: bytes,
    corpus_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if sha256_bytes(raw) != CORPUS_SHA256:
        raise ValueError("frozen V2-C5 discovery corpus SHA-256 changed")
    if corpus.get("schema_version") != "v2c5-discovery-corpus.v1":
        raise ValueError("unexpected V2-C5 corpus schema")
    if corpus.get("phase") != "V2-C5":
        raise ValueError("V2-C5 corpus phase changed")
    if corpus.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("V2-C5 corpus contract hash changed")
    policy = corpus.get("embedding_input_policy")
    if policy != {
        "fields": ["text"],
        "labels_concatenated_with_text": False,
        "native_labels_are_features": False,
        "source_metadata_concatenated_with_text": False,
    }:
        raise ValueError("embedding input is not raw text only")
    records_value = corpus.get("primary_cluster_population")
    if not isinstance(records_value, list) or len(records_value) != EXPECTED_COUNT:
        raise ValueError("V2-C5 primary population count changed")
    records: list[dict[str, Any]] = []
    discovery_ids: list[str] = []
    normalized_hashes: list[str] = []
    for row_value in records_value:
        if not isinstance(row_value, dict):
            raise TypeError("V2-C5 discovery record must be an object")
        row = dict(row_value)
        if row.get("current_sentinelvoice_intent") != PRIMARY_INTENT:
            raise ValueError("non-unsupported record entered primary population")
        if row.get("current_sentinelvoice_intent_clustering_feature") is not False:
            raise ValueError("current intent became a clustering feature")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("discovery record text is missing")
        if row.get("text_sha256") != corpus_builder.development_builder.text_sha256(
            text
        ):
            raise ValueError("discovery text SHA-256 mismatch")
        normalized_hash = corpus_builder.development_builder.normalized_text_sha256(
            text
        )
        if row.get("normalized_text_sha256") != normalized_hash:
            raise ValueError("discovery normalized-text SHA-256 mismatch")
        discovery_id = row.get("discovery_id")
        if discovery_id != f"v2c5-discovery:{normalized_hash}":
            raise ValueError("discovery ID is not derived from normalized text")
        occurrences = row.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError("discovery provenance occurrences are missing")
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                raise TypeError("discovery occurrence must be an object")
            if (
                occurrence.get("native_external_label_clustering_feature")
                is not False
            ):
                raise ValueError("native label became a clustering feature")
            if occurrence.get("data_role") != "development":
                raise ValueError("final or test role entered discovery corpus")
            source = occurrence.get("source_dataset")
            split = occurrence.get("source_split")
            if (source, split) not in {
                ("sentinelvoice_v2c1_internal", "train"),
                ("sentinelvoice_v2c1_internal", "validation"),
                ("banking77", "train"),
                ("clinc150_oos", "train"),
                ("clinc150_oos", "val"),
            }:
                raise ValueError("prohibited source entered discovery corpus")
        discovery_ids.append(str(discovery_id))
        normalized_hashes.append(normalized_hash)
        records.append(row)
    if len(set(discovery_ids)) != EXPECTED_COUNT:
        raise ValueError("discovery IDs are not unique")
    if len(set(normalized_hashes)) != EXPECTED_COUNT:
        raise ValueError("normalized discovery texts are not unique")
    if ordered_lines_sha256(discovery_ids) != ORDERED_DISCOVERY_ID_SHA256:
        raise ValueError("ordered discovery-ID SHA-256 changed")
    if ordered_lines_sha256(normalized_hashes) != ORDERED_NORMALIZED_TEXT_SHA256:
        raise ValueError("ordered normalized-text SHA-256 changed")
    counts = corpus_manifest.get("counts")
    if not isinstance(counts, dict):
        raise TypeError("corpus-manifest counts must be an object")
    if counts.get("primary_unique_normalized_text_count") != EXPECTED_COUNT:
        raise ValueError("corpus-manifest primary count changed")
    ordered = corpus_manifest.get("ordered_hashes")
    if not isinstance(ordered, dict):
        raise TypeError("corpus-manifest ordered hashes must be an object")
    if (
        ordered.get("normalized_text_ordered_sha256")
        != ORDERED_NORMALIZED_TEXT_SHA256
        or ordered.get("canonical_discovery_id_ordered_sha256")
        != ORDERED_DISCOVERY_ID_SHA256
    ):
        raise ValueError("corpus-manifest ordered hashes changed")
    return records


def load_frozen_inputs(paths: ExperimentPaths = DEFAULT_PATHS) -> FrozenInputs:
    contract_bytes = read_input_bytes(paths.contract)
    corpus_bytes = read_input_bytes(paths.corpus)
    corpus_manifest_bytes = read_input_bytes(paths.corpus_manifest)
    if sha256_bytes(corpus_manifest_bytes) != CORPUS_MANIFEST_SHA256:
        raise ValueError("frozen V2-C5 corpus manifest SHA-256 changed")
    contract = load_json_bytes(contract_bytes, "V2-C5 contract")
    corpus = load_json_bytes(corpus_bytes, "V2-C5 discovery corpus")
    corpus_manifest = load_json_bytes(
        corpus_manifest_bytes,
        "V2-C5 discovery corpus manifest",
    )
    validate_contract(contract, contract_bytes)
    manifest_contract = corpus_manifest.get("contract")
    manifest_corpus = corpus_manifest.get("corpus")
    if not isinstance(manifest_contract, dict) or not isinstance(
        manifest_corpus,
        dict,
    ):
        raise TypeError("corpus manifest frozen inputs must be objects")
    if manifest_contract.get("sha256") != CONTRACT_SHA256:
        raise ValueError("corpus manifest contract SHA-256 changed")
    if manifest_corpus.get("sha256") != CORPUS_SHA256:
        raise ValueError("corpus manifest corpus SHA-256 changed")
    records = validate_corpus(corpus, corpus_bytes, corpus_manifest)
    return FrozenInputs(
        contract=contract,
        corpus=corpus,
        corpus_manifest=corpus_manifest,
        records=records,
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def package_versions(*, include_umap: bool) -> dict[str, str]:
    versions = {
        "fastembed": package_version("fastembed"),
        "numpy": np.__version__,
        "onnx_runtime": package_version("onnxruntime"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }
    if versions["onnx_runtime"] == "not_installed":
        versions["onnx_runtime"] = package_version("onnxruntime-gpu")
    if include_umap:
        versions["umap_learn"] = package_version("umap-learn")
    return versions


def dependency_status() -> dict[str, Any]:
    hdbscan_available = False
    hdbscan_error: str | None = None
    try:
        from sklearn.cluster import HDBSCAN

        del HDBSCAN
        hdbscan_available = True
    except ImportError as exc:
        hdbscan_error = str(exc)
    return {
        "fastembed_available": importlib.util.find_spec("fastembed") is not None,
        "hdbscan_error": hdbscan_error,
        "sklearn_cluster_hdbscan_available": hdbscan_available,
        "umap_learn_available": importlib.util.find_spec("umap") is not None,
        "umap_required_for_primary_results": False,
    }


def matrix_content_sha256(
    matrix: np.ndarray,
    discovery_ids: Sequence[str],
    normalized_hashes: Sequence[str],
) -> str:
    contiguous = np.ascontiguousarray(matrix, dtype="<f4")
    header = stable_json_bytes(
        {
            "dtype": "float32-little-endian",
            "ordered_discovery_ids": list(discovery_ids),
            "ordered_normalized_text_sha256": list(normalized_hashes),
            "shape": list(contiguous.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def validate_embedding_matrix(
    matrix: np.ndarray,
    *,
    expected_count: int = EXPECTED_COUNT,
    expected_dimensions: int = DIMENSIONS,
) -> np.ndarray:
    if matrix.dtype != np.float32:
        raise ValueError("embedding cache dtype must be float32")
    if matrix.shape != (expected_count, expected_dimensions):
        raise ValueError(
            "embedding matrix shape differs from the required count and "
            "dimensions"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("embedding matrix rows are not L2 normalized")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def cache_metadata_header(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "contract_sha256": CONTRACT_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "dimensions": DIMENSIONS,
        "dtype": "float32",
        "embedding_input_fields": ["text"],
        "example_count": EXPECTED_COUNT,
        "external_api": False,
        "fastembed_method": EMBEDDING_METHOD,
        "fine_tuning": False,
        "l2_normalized": True,
        "labels_or_metadata_used_as_embedding_features": False,
        "model_identifier": MODEL_IDENTIFIER,
        "ordered_discovery_ids_sha256": ordered_lines_sha256(
            [str(row["discovery_id"]) for row in records]
        ),
        "ordered_normalized_text_sha256": ordered_lines_sha256(
            [str(row["normalized_text_sha256"]) for row in records]
        ),
        "provider": EMBEDDING_PROVIDER,
        "schema_version": CACHE_SCHEMA_VERSION,
    }


def write_embedding_cache(
    matrix: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    paths: ExperimentPaths,
) -> dict[str, Any]:
    paths.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=paths.embedding_cache.parent,
        prefix=f".{paths.embedding_cache.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    discovery_ids = [str(row["discovery_id"]) for row in records]
    normalized_hashes = [
        str(row["normalized_text_sha256"]) for row in records
    ]
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                discovery_ids=np.asarray(discovery_ids),
                embeddings=matrix,
                normalized_text_sha256=np.asarray(normalized_hashes),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, paths.embedding_cache)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    metadata = {
        **cache_metadata_header(records),
        "cache_file_sha256": sha256_file(paths.embedding_cache),
        "embedding_matrix_content_sha256": matrix_content_sha256(
            matrix,
            discovery_ids,
            normalized_hashes,
        ),
        "generation_package_versions": package_versions(include_umap=False),
    }
    corpus_builder.development_builder.atomic_write_bytes(
        paths.embedding_cache_metadata,
        stable_json_bytes(metadata),
    )
    return metadata


def load_embedding_cache(
    inputs: FrozenInputs,
    paths: ExperimentPaths = DEFAULT_PATHS,
) -> tuple[np.ndarray, dict[str, Any]]:
    if paths.embedding_cache.exists() != paths.embedding_cache_metadata.exists():
        raise FileNotFoundError(
            "embedding cache and metadata must exist together"
        )
    if not paths.embedding_cache.exists():
        raise FileNotFoundError("verified V2-C5 embedding cache is missing")
    metadata_bytes = read_input_bytes(paths.embedding_cache_metadata)
    metadata = load_json_bytes(metadata_bytes, "V2-C5 embedding metadata")
    expected_header = cache_metadata_header(inputs.records)
    for field, expected in expected_header.items():
        if metadata.get(field) != expected:
            raise ValueError(f"embedding cache metadata changed: {field}")
    if metadata.get("cache_file_sha256") != sha256_file(paths.embedding_cache):
        raise ValueError("embedding cache container SHA-256 mismatch")
    with np.load(paths.embedding_cache, allow_pickle=False) as cache:
        raw_matrix = cache["embeddings"]
        discovery_ids = cache["discovery_ids"].astype(str).tolist()
        normalized_hashes = (
            cache["normalized_text_sha256"].astype(str).tolist()
        )
    expected_ids = [str(row["discovery_id"]) for row in inputs.records]
    expected_hashes = [
        str(row["normalized_text_sha256"]) for row in inputs.records
    ]
    if discovery_ids != expected_ids:
        raise ValueError("embedding cache discovery-ID ordering changed")
    if normalized_hashes != expected_hashes:
        raise ValueError("embedding cache normalized-text ordering changed")
    matrix = validate_embedding_matrix(raw_matrix)
    content_hash = matrix_content_sha256(
        matrix,
        discovery_ids,
        normalized_hashes,
    )
    if metadata.get("embedding_matrix_content_sha256") != content_hash:
        raise ValueError("embedding matrix content SHA-256 mismatch")
    return matrix, metadata


def generate_embeddings(
    records: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("fastembed is required for Step 18 embeddings") from exc
    model = TextEmbedding(model_name=MODEL_IDENTIFIER)
    texts = [str(row["text"]) for row in records]
    vectors = list(model.passage_embed(texts, batch_size=256))
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (EXPECTED_COUNT, DIMENSIONS):
        raise ValueError("raw FastEmbed matrix shape changed")
    if not np.isfinite(matrix).all():
        raise ValueError("raw FastEmbed matrix contains non-finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms <= 0):
        raise ValueError("raw FastEmbed matrix contains a zero vector")
    normalized = (matrix / norms[:, None]).astype(np.float32)
    return validate_embedding_matrix(normalized)


def load_or_create_embedding_cache(
    inputs: FrozenInputs,
    paths: ExperimentPaths = DEFAULT_PATHS,
) -> tuple[np.ndarray, dict[str, Any]]:
    if paths.embedding_cache.exists() or paths.embedding_cache_metadata.exists():
        return load_embedding_cache(inputs, paths)
    matrix = generate_embeddings(inputs.records)
    write_embedding_cache(matrix, inputs.records, paths)
    return load_embedding_cache(inputs, paths)


def canonicalize_cluster_labels(
    raw_labels: Sequence[int],
    discovery_ids: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    if len(raw_labels) != len(discovery_ids):
        raise ValueError("cluster-label count differs from discovery IDs")
    members: dict[int, list[str]] = {}
    for raw_label, discovery_id in zip(raw_labels, discovery_ids, strict=True):
        label = int(raw_label)
        if label < -1:
            raise ValueError("HDBSCAN label must be noise or non-negative")
        if label != -1:
            members.setdefault(label, []).append(discovery_id)
    mapping: dict[int, str] = {-1: "noise"}
    definitions: list[dict[str, Any]] = []
    seen_canonical: set[str] = set()
    for raw_label, member_ids in sorted(members.items()):
        ordered_members = sorted(member_ids)
        signature = ordered_lines_sha256(ordered_members)
        canonical_id = f"cluster-{signature[:16]}"
        if canonical_id in seen_canonical:
            raise ValueError("canonical cluster-ID collision")
        seen_canonical.add(canonical_id)
        mapping[raw_label] = canonical_id
        definitions.append(
            {
                "canonical_cluster_id": canonical_id,
                "cluster_signature_sha256": signature,
                "member_count": len(ordered_members),
                "raw_hdbscan_label": raw_label,
            }
        )
    return [mapping[int(label)] for label in raw_labels], definitions


def default_hdbscan_factory(parameters: Mapping[str, Any]) -> Any:
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "the supported scikit-learn version does not provide HDBSCAN"
        ) from exc
    return HDBSCAN(copy=False, **dict(parameters))


def run_hdbscan_grid(
    matrix: np.ndarray,
    discovery_ids: Sequence[str],
    clusterer_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    *,
    expected_count: int = EXPECTED_COUNT,
    expected_dimensions: int = DIMENSIONS,
) -> list[ClusterRun]:
    validated = validate_embedding_matrix(
        matrix,
        expected_count=expected_count,
        expected_dimensions=expected_dimensions,
    )
    if len(discovery_ids) != expected_count:
        raise ValueError("HDBSCAN discovery-ID count changed")
    factory = clusterer_factory or default_hdbscan_factory
    runs: list[ClusterRun] = []
    for parameters in SENSITIVITY_CONFIGS:
        model = factory(parameters)
        raw_labels = np.asarray(model.fit_predict(validated), dtype=np.int64)
        if raw_labels.shape != (expected_count,):
            raise ValueError("HDBSCAN assignment shape changed")
        probabilities = getattr(model, "probabilities_", None)
        membership_values: np.ndarray | None = None
        if probabilities is not None:
            membership_values = np.asarray(probabilities, dtype=np.float64)
            if membership_values.shape != (expected_count,):
                raise ValueError("HDBSCAN membership-value shape changed")
            if (
                not np.isfinite(membership_values).all()
                or np.any(membership_values < 0)
                or np.any(membership_values > 1)
            ):
                raise ValueError("invalid HDBSCAN membership values")
        canonical_labels, definitions = canonicalize_cluster_labels(
            raw_labels.tolist(),
            discovery_ids,
        )
        runs.append(
            ClusterRun(
                config_id=config_id(parameters),
                parameters=dict(parameters),
                raw_labels=raw_labels,
                membership_values=membership_values,
                canonical_labels=canonical_labels,
                cluster_definitions=definitions,
            )
        )
    if len(runs) != 9:
        raise ValueError("exactly nine HDBSCAN configurations are required")
    if sum(run.config_id == PRIMARY_CONFIG_ID for run in runs) != 1:
        raise ValueError("primary 30/10 configuration must occur exactly once")
    return runs


def primary_run(runs: Sequence[ClusterRun]) -> ClusterRun:
    matches = [run for run in runs if run.config_id == PRIMARY_CONFIG_ID]
    if len(matches) != 1:
        raise ValueError("primary HDBSCAN result is not unique")
    return matches[0]


def size_summary(raw_labels: np.ndarray) -> dict[str, Any]:
    sizes = sorted(Counter(int(value) for value in raw_labels if value >= 0).values())
    if not sizes:
        return {
            "cluster_size_distribution": [],
            "max": None,
            "mean": None,
            "median": None,
            "min": None,
        }
    return {
        "cluster_size_distribution": sizes,
        "max": max(sizes),
        "mean": float(statistics.fmean(sizes)),
        "median": float(statistics.median(sizes)),
        "min": min(sizes),
    }


def assignment_summary(run: ClusterRun) -> dict[str, Any]:
    example_count = len(run.raw_labels)
    if example_count == 0:
        raise ValueError("cluster assignment cannot be empty")
    noise_count = int(np.sum(run.raw_labels == -1))
    cluster_count = len({int(value) for value in run.raw_labels if value >= 0})
    return {
        "cluster_count_excluding_noise": cluster_count,
        "cluster_size_summary": size_summary(run.raw_labels),
        "clustered_coverage": (example_count - noise_count) / example_count,
        "example_count": example_count,
        "noise_count": noise_count,
        "noise_fraction": noise_count / example_count,
    }


def count_proportions(values: Sequence[str]) -> dict[str, Any]:
    counts = dict(sorted(Counter(values).items()))
    total = sum(counts.values())
    proportions = {
        key: count / total for key, count in counts.items()
    } if total else {}
    dominant = (
        min(counts, key=lambda key: (-counts[key], key)) if counts else None
    )
    return {
        "counts": counts,
        "dominant": dominant,
        "dominant_proportion": proportions.get(dominant) if dominant else None,
        "proportions": proportions,
        "unit": "provenance_occurrences",
    }


def representative_and_boundary_ids(
    member_indices: Sequence[int],
    matrix: np.ndarray,
    discovery_ids: Sequence[str],
    membership_values: np.ndarray | None,
) -> tuple[list[str], list[str]]:
    index_array = np.asarray(member_indices, dtype=np.int64)
    cluster_matrix = matrix[index_array].astype(np.float64)
    centroid = np.mean(cluster_matrix, axis=0)
    distances = np.linalg.norm(cluster_matrix - centroid, axis=1)
    representative_order = sorted(
        range(len(index_array)),
        key=lambda offset: (
            float(distances[offset]),
            discovery_ids[int(index_array[offset])],
        ),
    )
    representatives = [
        discovery_ids[int(index_array[offset])]
        for offset in representative_order[:5]
    ]
    boundaries: list[str] = []
    if membership_values is not None:
        boundary_order = sorted(
            range(len(index_array)),
            key=lambda offset: (
                float(membership_values[int(index_array[offset])]),
                discovery_ids[int(index_array[offset])],
            ),
        )
        boundaries = [
            discovery_ids[int(index_array[offset])]
            for offset in boundary_order[:5]
        ]
    return representatives, boundaries


def cluster_diagnostics(
    run: ClusterRun,
    matrix: np.ndarray,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    discovery_ids = [str(row["discovery_id"]) for row in records]
    definition_by_raw = {
        int(row["raw_hdbscan_label"]): row
        for row in run.cluster_definitions
    }
    diagnostics: list[dict[str, Any]] = []
    ordered_raw_labels = sorted(
        definition_by_raw,
        key=lambda raw_label: definition_by_raw[raw_label][
            "canonical_cluster_id"
        ],
    )
    for raw_label in ordered_raw_labels:
        indices = np.flatnonzero(run.raw_labels == raw_label).tolist()
        representatives, boundaries = representative_and_boundary_ids(
            indices,
            matrix,
            discovery_ids,
            run.membership_values,
        )
        sources: list[str] = []
        native_labels: list[str] = []
        risks: list[str] = []
        occurrence_counts: list[str] = []
        for index in indices:
            record = records[index]
            risk = record.get("current_risk")
            if risk is not None:
                risks.append(str(risk))
            occurrence_counts.append(str(record["source_occurrence_count"]))
            for occurrence in record["occurrences"]:
                sources.append(str(occurrence["source_dataset"]))
                native = occurrence.get("native_external_label")
                if native is not None:
                    native_labels.append(str(native))
        membership_summary = {
            "available": run.membership_values is not None,
            "mean": None,
            "median": None,
        }
        # HDBSCAN membership probability is not a semantic-correctness probability.
        if run.membership_values is not None:
            values = run.membership_values[np.asarray(indices)]
            membership_summary["mean"] = float(np.mean(values))
            membership_summary["median"] = float(np.median(values))
        definition = definition_by_raw[raw_label]
        diagnostics.append(
            {
                "boundary_discovery_ids": boundaries,
                "canonical_cluster_id": definition["canonical_cluster_id"],
                "cluster_signature_sha256": definition[
                    "cluster_signature_sha256"
                ],
                "current_risk_distribution": dict(
                    sorted(Counter(risks).items())
                ),
                "member_count": len(indices),
                "membership_value_summary": membership_summary,
                "native_external_label_concentration": count_proportions(
                    native_labels
                ),
                "occurrence_count_distribution": dict(
                    sorted(Counter(occurrence_counts).items())
                ),
                "raw_hdbscan_label": raw_label,
                "representative_discovery_ids": representatives,
                "source_dataset_concentration": count_proportions(sources),
            }
        )
    return diagnostics


def sensitivity_diagnostics(
    runs: Sequence[ClusterRun],
) -> list[dict[str, Any]]:
    primary = primary_run(runs)
    results: list[dict[str, Any]] = []
    for run in runs:
        both_clustered = (primary.raw_labels != -1) & (run.raw_labels != -1)
        both_count = int(np.sum(both_clustered))
        restricted_ari: float | None = None
        if both_count >= 2:
            restricted_ari = float(
                adjusted_rand_score(
                    primary.raw_labels[both_clustered],
                    run.raw_labels[both_clustered],
                )
            )
        results.append(
            {
                **assignment_summary(run),
                "adjusted_rand_index_vs_primary_all_records": float(
                    adjusted_rand_score(
                        primary.raw_labels,
                        run.raw_labels,
                    )
                ),
                "adjusted_rand_index_vs_primary_both_clustered": (
                    restricted_ari
                ),
                "both_clustered_record_count": both_count,
                "comparison_role": "diagnostic_only_no_selection_threshold",
                "config_id": run.config_id,
                "is_primary": run.config_id == PRIMARY_CONFIG_ID,
                "parameters": run.parameters,
            }
        )
    return results


def build_assignments_payload(
    runs: Sequence[ClusterRun],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary = primary_run(runs)
    other_runs = [run for run in runs if run.config_id != PRIMARY_CONFIG_ID]
    assignments: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        assignments.append(
            {
                "discovery_id": record["discovery_id"],
                "normalized_text_sha256": record[
                    "normalized_text_sha256"
                ],
                "primary_canonical_cluster_id": primary.canonical_labels[index],
                "primary_is_noise": int(primary.raw_labels[index]) == -1,
                "primary_membership_value": (
                    float(primary.membership_values[index])
                    if primary.membership_values is not None
                    else None
                ),
                "primary_raw_hdbscan_label": int(
                    primary.raw_labels[index]
                ),
                "sensitivity_assignments": {
                    run.config_id: {
                        "canonical_cluster_id": run.canonical_labels[index],
                        "is_noise": int(run.raw_labels[index]) == -1,
                        "membership_value": (
                            float(run.membership_values[index])
                            if run.membership_values is not None
                            else None
                        ),
                        "raw_hdbscan_label": int(run.raw_labels[index]),
                    }
                    for run in other_runs
                },
            }
        )
    return {
        "assignment_count": len(assignments),
        "assignments": assignments,
        "contract_sha256": CONTRACT_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "primary_config_id": PRIMARY_CONFIG_ID,
        "raw_text_persisted": False,
        "schema_version": ASSIGNMENTS_SCHEMA_VERSION,
        "semantic_intent_names_assigned": False,
    }


def governance_payload() -> dict[str, Any]:
    return {
        "analysis_role": "unsupervised_taxonomy_discovery_evidence",
        "classifier_output_role": "advisory_only",
        "final_acceptance_evidence": False,
        "human_adjudication_required": True,
        "new_intents_created": False,
        "runtime_behavior_changed": False,
        "supervised_training_performed": False,
        "taxonomy_changed": False,
    }


def build_report_payload(
    runs: Sequence[ClusterRun],
    matrix: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    cache_metadata: Mapping[str, Any],
    umap_status: Mapping[str, Any],
) -> dict[str, Any]:
    primary = primary_run(runs)
    return {
        **governance_payload(),
        "canonical_cluster_id_scheme": {
            "format": "cluster-{first_16_hex_of_signature}",
            "noise_value": "noise",
            "raw_hdbscan_labels_preserved": True,
            "signature": (
                "SHA-256 over sorted member discovery IDs, one UTF-8 ID per "
                "line with a trailing newline"
            ),
        },
        "cluster_persistence_available": False,
        "contract_sha256": CONTRACT_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "dataset_concentration_interpretation": (
            "Source and native-label concentration are post-hoc metadata and "
            "do not establish a valid SentinelVoice intent."
        ),
        "execution_status": {
            "clustering_performed": True,
            "embeddings_available_and_verified": True,
            "supervised_training_performed": False,
            "taxonomy_changed": False,
            "umap_performed": bool(umap_status["performed"]),
        },
        "membership_value_semantics": (
            "HDBSCAN membership probability is not a semantic-correctness "
            "probability."
        ),
        "package_versions": package_versions(
            include_umap=bool(umap_status["performed"])
        ),
        "primary": {
            **assignment_summary(primary),
            "cluster_diagnostics": cluster_diagnostics(
                primary,
                matrix,
                records,
            ),
            "cluster_persistence_available": False,
            "config_id": PRIMARY_CONFIG_ID,
            "membership_values_available": (
                primary.membership_values is not None
            ),
            "parameters": PRIMARY_CONFIG,
            "primary_result_replaceable_post_hoc": False,
        },
        "raw_text_persisted": False,
        "representation": {
            "cache_metadata": dict(cache_metadata),
            "clustering_input_shape": [EXPECTED_COUNT, DIMENSIONS],
            "dimensions": DIMENSIONS,
            "embedding_input_fields": ["text"],
            "l2_normalized": True,
            "method": EMBEDDING_METHOD,
            "model_identifier": MODEL_IDENTIFIER,
            "provider": EMBEDDING_PROVIDER,
            "umap_vectors_used_for_clustering": False,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "sensitivity": {
            "configuration_count": len(runs),
            "diagnostics": sensitivity_diagnostics(runs),
            "primary_config_id": PRIMARY_CONFIG_ID,
            "selection_from_sensitivity_allowed": False,
        },
        "umap": dict(umap_status),
    }


def validate_no_raw_text(payload: Any) -> None:
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"text", "normalized_text"}:
                    raise ValueError("raw text field cannot be persisted")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)


def run_umap_visualization(
    matrix: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    primary: ClusterRun,
    output_path: Path,
) -> tuple[dict[str, Any], bytes | None]:
    if importlib.util.find_spec("umap") is None:
        return {
            "performed": False,
            "reason": "optional_dependency_unavailable",
            "requested": True,
            "required_for_primary_results": False,
        }, None
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_jobs=1,
            random_state=UMAP_RANDOM_STATE,
        )
        coordinates = np.asarray(
            reducer.fit_transform(matrix),
            dtype=np.float64,
        )
        if coordinates.shape != (EXPECTED_COUNT, 2):
            raise ValueError("UMAP coordinate shape changed")
        if not np.isfinite(coordinates).all():
            raise ValueError("UMAP coordinates contain non-finite values")
    except (ImportError, ValueError, RuntimeError) as exc:
        return {
            "error_type": type(exc).__name__,
            "performed": False,
            "reason": "optional_visualization_failed",
            "requested": True,
            "required_for_primary_results": False,
        }, None
    payload = {
        "coordinates": [
            {
                "canonical_cluster_id": primary.canonical_labels[index],
                "discovery_id": record["discovery_id"],
                "is_noise": int(primary.raw_labels[index]) == -1,
                "x": float(coordinates[index, 0]),
                "y": float(coordinates[index, 1]),
            }
            for index, record in enumerate(records)
        ],
        "random_state": UMAP_RANDOM_STATE,
        "role": "visualization_only",
        "schema_version": UMAP_SCHEMA_VERSION,
    }
    return {
        "artifact_path": display_path(output_path),
        "performed": True,
        "random_state": UMAP_RANDOM_STATE,
        "reason": None,
        "requested": True,
        "required_for_primary_results": False,
    }, stable_json_bytes(payload)


def build_manifest_payload(
    *,
    report_bytes: bytes,
    assignments_bytes: bytes,
    cache_metadata: Mapping[str, Any],
    umap_status: Mapping[str, Any],
    umap_bytes: bytes | None,
    script_path: Path,
    paths: ExperimentPaths,
) -> dict[str, Any]:
    umap_artifact: dict[str, Any] | None = None
    if umap_bytes is not None:
        umap_artifact = {
            "path": display_path(paths.umap_output),
            "sha256": sha256_bytes(umap_bytes),
        }
    return {
        "assignments": {
            "path": display_path(paths.assignments_output),
            "sha256": sha256_bytes(assignments_bytes),
        },
        "embedding_cache": {
            "container_path": display_path(paths.embedding_cache),
            "container_sha256": cache_metadata["cache_file_sha256"],
            "local_generated_artifact": True,
            "matrix_content_sha256": cache_metadata[
                "embedding_matrix_content_sha256"
            ],
            "metadata_path": display_path(paths.embedding_cache_metadata),
            "metadata_sha256": sha256_file(paths.embedding_cache_metadata),
            "tracked_by_git": False,
        },
        "execution_status": {
            "clustering_performed": True,
            "embeddings_generated_or_exact_cache_verified": True,
            "supervised_training_performed": False,
            "taxonomy_changed": False,
            "umap_performed": bool(umap_status["performed"]),
        },
        "frozen_inputs": {
            "contract": {
                "path": display_path(paths.contract),
                "sha256": CONTRACT_SHA256,
            },
            "corpus": {
                "path": display_path(paths.corpus),
                "sha256": CORPUS_SHA256,
            },
            "corpus_manifest": {
                "path": display_path(paths.corpus_manifest),
                "sha256": CORPUS_MANIFEST_SHA256,
            },
            "ordered_discovery_id_sha256": ORDERED_DISCOVERY_ID_SHA256,
            "ordered_normalized_text_sha256": (
                ORDERED_NORMALIZED_TEXT_SHA256
            ),
            "record_count": EXPECTED_COUNT,
        },
        "governance": governance_payload(),
        "hdbscan": {
            "implementation": "sklearn.cluster.HDBSCAN",
            "primary_config_id": PRIMARY_CONFIG_ID,
            "primary_parameters": PRIMARY_CONFIG,
            "sensitivity_configurations": [
                {
                    "config_id": config_id(parameters),
                    "parameters": parameters,
                }
                for parameters in SENSITIVITY_CONFIGS
            ],
            "total_fit_count": 9,
        },
        "package_versions": package_versions(
            include_umap=bool(umap_status["performed"])
        ),
        "report": {
            "path": display_path(paths.report_output),
            "sha256": sha256_bytes(report_bytes),
        },
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "script": {
            "path": display_path(script_path),
            "sha256": sha256_file(script_path),
        },
        "umap": {
            "artifact": umap_artifact,
            **dict(umap_status),
        },
    }


def build_result_artifacts(
    *,
    inputs: FrozenInputs,
    matrix: np.ndarray,
    cache_metadata: Mapping[str, Any],
    runs: Sequence[ClusterRun],
    umap_status: Mapping[str, Any],
    umap_bytes: bytes | None,
    paths: ExperimentPaths = DEFAULT_PATHS,
    script_path: Path | None = None,
) -> ResultArtifacts:
    assignments_payload = build_assignments_payload(runs, inputs.records)
    report_payload = build_report_payload(
        runs,
        matrix,
        inputs.records,
        cache_metadata,
        umap_status,
    )
    validate_no_raw_text(assignments_payload)
    validate_no_raw_text(report_payload)
    assignments_bytes = stable_json_bytes(assignments_payload)
    report_bytes = stable_json_bytes(report_payload)
    actual_script_path = script_path or Path(__file__).resolve()
    manifest_payload = build_manifest_payload(
        report_bytes=report_bytes,
        assignments_bytes=assignments_bytes,
        cache_metadata=cache_metadata,
        umap_status=umap_status,
        umap_bytes=umap_bytes,
        script_path=actual_script_path,
        paths=paths,
    )
    validate_no_raw_text(manifest_payload)
    return ResultArtifacts(
        report_bytes=report_bytes,
        assignments_bytes=assignments_bytes,
        manifest_bytes=stable_json_bytes(manifest_payload),
        umap_bytes=umap_bytes,
        report_payload=report_payload,
        assignments_payload=assignments_payload,
        manifest_payload=manifest_payload,
    )


def write_result_artifacts(
    artifacts: ResultArtifacts,
    paths: ExperimentPaths = DEFAULT_PATHS,
) -> None:
    outputs = {
        paths.assignments_output: artifacts.assignments_bytes,
        paths.report_output: artifacts.report_bytes,
        paths.manifest_output: artifacts.manifest_bytes,
    }
    if artifacts.umap_bytes is not None:
        outputs[paths.umap_output] = artifacts.umap_bytes
    elif paths.umap_output.exists():
        raise FileExistsError(
            "stale UMAP artifact exists but this run did not produce UMAP"
        )
    for path, payload in outputs.items():
        corpus_builder.development_builder.atomic_write_bytes(path, payload)


def preflight(paths: ExperimentPaths = DEFAULT_PATHS) -> dict[str, Any]:
    inputs = load_frozen_inputs(paths)
    dependencies = dependency_status()
    if not dependencies["sklearn_cluster_hdbscan_available"]:
        raise RuntimeError(
            "supported scikit-learn does not provide sklearn.cluster.HDBSCAN"
        )
    if not dependencies["fastembed_available"] and not (
        paths.embedding_cache.exists()
        and paths.embedding_cache_metadata.exists()
    ):
        raise RuntimeError(
            "fastembed is unavailable and no exact embedding cache exists"
        )
    cache_exists = paths.embedding_cache.exists()
    metadata_exists = paths.embedding_cache_metadata.exists()
    if cache_exists != metadata_exists:
        raise FileNotFoundError(
            "embedding cache and metadata must exist together"
        )
    cache_status = (
        "present_verification_deferred_to_run_or_check"
        if cache_exists
        else "missing_will_generate_on_run"
    )
    return {
        "cache_status": cache_status,
        "dependencies": dependencies,
        "embedding_input_fields": ["text"],
        "frozen_input_status": "passed",
        "hdbscan_fit_count_on_preflight": 0,
        "output_paths": {
            "assignments": display_path(paths.assignments_output),
            "manifest": display_path(paths.manifest_output),
            "report": display_path(paths.report_output),
            "umap_optional": display_path(paths.umap_output),
        },
        "primary_count": len(inputs.records),
    }


def run_experiment(
    *,
    with_umap: bool,
    paths: ExperimentPaths = DEFAULT_PATHS,
) -> ResultArtifacts:
    inputs = load_frozen_inputs(paths)
    dependencies = dependency_status()
    if not dependencies["sklearn_cluster_hdbscan_available"]:
        raise RuntimeError(
            "supported scikit-learn does not provide sklearn.cluster.HDBSCAN"
        )
    matrix, cache_metadata = load_or_create_embedding_cache(inputs, paths)
    discovery_ids = [str(row["discovery_id"]) for row in inputs.records]
    runs = run_hdbscan_grid(matrix, discovery_ids)
    primary = primary_run(runs)
    if with_umap:
        umap_status, umap_bytes = run_umap_visualization(
            matrix,
            inputs.records,
            primary,
            paths.umap_output,
        )
    else:
        umap_status = {
            "performed": False,
            "reason": "not_requested",
            "requested": False,
            "required_for_primary_results": False,
        }
        umap_bytes = None
    artifacts = build_result_artifacts(
        inputs=inputs,
        matrix=matrix,
        cache_metadata=cache_metadata,
        runs=runs,
        umap_status=umap_status,
        umap_bytes=umap_bytes,
        paths=paths,
    )
    write_result_artifacts(artifacts, paths)
    return artifacts


def load_result_manifest(
    paths: ExperimentPaths = DEFAULT_PATHS,
) -> dict[str, Any]:
    return load_json_bytes(
        read_input_bytes(paths.manifest_output),
        "V2-C5 intent-discovery manifest",
    )


def check_existing_output(
    path: Path,
    expected_bytes: bytes,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.read_bytes() != expected_bytes:
        raise ValueError(f"{label} differs from deterministic reproduction")


def check_experiment(
    paths: ExperimentPaths = DEFAULT_PATHS,
) -> ResultArtifacts:
    inputs = load_frozen_inputs(paths)
    matrix, cache_metadata = load_embedding_cache(inputs, paths)
    manifest = load_result_manifest(paths)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unexpected Step 18 manifest schema")
    recorded_versions = manifest.get("package_versions")
    if not isinstance(recorded_versions, dict):
        raise TypeError("Step 18 package versions must be an object")
    umap_section = manifest.get("umap")
    if not isinstance(umap_section, dict):
        raise TypeError("Step 18 UMAP status must be an object")
    performed = umap_section.get("performed") is True
    if recorded_versions != package_versions(include_umap=performed):
        raise ValueError("Step 18 package versions changed")
    umap_bytes: bytes | None = None
    if performed:
        artifact = umap_section.get("artifact")
        if not isinstance(artifact, dict):
            raise TypeError("UMAP artifact metadata must be an object")
        umap_bytes = read_input_bytes(paths.umap_output)
        if sha256_bytes(umap_bytes) != artifact.get("sha256"):
            raise ValueError("UMAP artifact SHA-256 mismatch")
    elif paths.umap_output.exists():
        raise FileExistsError("unexpected UMAP artifact exists")
    umap_status = {
        key: value for key, value in umap_section.items() if key != "artifact"
    }
    discovery_ids = [str(row["discovery_id"]) for row in inputs.records]
    runs = run_hdbscan_grid(matrix, discovery_ids)
    artifacts = build_result_artifacts(
        inputs=inputs,
        matrix=matrix,
        cache_metadata=cache_metadata,
        runs=runs,
        umap_status=umap_status,
        umap_bytes=umap_bytes,
        paths=paths,
    )
    check_existing_output(
        paths.assignments_output,
        artifacts.assignments_bytes,
        "assignments",
    )
    check_existing_output(
        paths.report_output,
        artifacts.report_bytes,
        "report",
    )
    check_existing_output(
        paths.manifest_output,
        artifacts.manifest_bytes,
        "manifest",
    )
    return artifacts


def summary(paths: ExperimentPaths = DEFAULT_PATHS) -> dict[str, Any]:
    load_frozen_inputs(paths)
    report = load_json_bytes(
        read_input_bytes(paths.report_output),
        "V2-C5 intent-discovery report",
    )
    primary = report["primary"]
    return {
        "analysis_role": report["analysis_role"],
        "cluster_count_excluding_noise": primary[
            "cluster_count_excluding_noise"
        ],
        "clustered_coverage": primary["clustered_coverage"],
        "noise_count": primary["noise_count"],
        "primary_config_id": primary["config_id"],
        "sensitivity_configuration_count": report["sensitivity"][
            "configuration_count"
        ],
        "taxonomy_changed": report["taxonomy_changed"],
        "umap_performed": report["umap"]["performed"],
    }


def bounded_limit(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 100:
        raise argparse.ArgumentTypeError("limit must be between 1 and 100")
    return parsed


def print_cluster(
    cluster_id: str,
    limit: int,
    paths: ExperimentPaths = DEFAULT_PATHS,
) -> list[dict[str, Any]]:
    inputs = load_frozen_inputs(paths)
    report = load_json_bytes(
        read_input_bytes(paths.report_output),
        "V2-C5 intent-discovery report",
    )
    assignments = load_json_bytes(
        read_input_bytes(paths.assignments_output),
        "V2-C5 intent-discovery assignments",
    )
    clusters = report["primary"]["cluster_diagnostics"]
    cluster = next(
        (
            row
            for row in clusters
            if row["canonical_cluster_id"] == cluster_id
        ),
        None,
    )
    if cluster is None:
        raise ValueError(f"unknown canonical primary cluster ID: {cluster_id}")
    assignment_by_id = {
        row["discovery_id"]: row for row in assignments["assignments"]
    }
    record_by_id = {
        row["discovery_id"]: row for row in inputs.records
    }
    ordered_ids: list[str] = []
    for key in ("representative_discovery_ids", "boundary_discovery_ids"):
        for discovery_id in cluster[key]:
            if discovery_id not in ordered_ids:
                ordered_ids.append(discovery_id)
    for row in assignments["assignments"]:
        if (
            row["primary_canonical_cluster_id"] == cluster_id
            and row["discovery_id"] not in ordered_ids
        ):
            ordered_ids.append(row["discovery_id"])
    review: list[dict[str, Any]] = []
    for discovery_id in ordered_ids[:limit]:
        record = record_by_id[discovery_id]
        assignment = assignment_by_id[discovery_id]
        sources = sorted(
            {
                str(item["source_dataset"])
                for item in record["occurrences"]
            }
        )
        native_labels = sorted(
            {
                str(item["native_external_label"])
                for item in record["occurrences"]
                if item.get("native_external_label") is not None
            }
        )
        review.append(
            {
                "canonical_cluster_id": cluster_id,
                "discovery_id": discovery_id,
                "membership_value": assignment[
                    "primary_membership_value"
                ],
                "native_external_labels": native_labels,
                "source_datasets": sources,
                "text": record["text"],
            }
        )
    return review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--with-umap", action="store_true")
    subparsers.add_parser("check")
    subparsers.add_parser("summary")
    cluster_parser = subparsers.add_parser("print-cluster")
    cluster_parser.add_argument("--cluster-id", required=True)
    cluster_parser.add_argument("--limit", type=bounded_limit, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight(), sort_keys=True))
    elif args.command == "run":
        artifacts = run_experiment(with_umap=args.with_umap)
        print(json.dumps(summary(), sort_keys=True))
        if artifacts.umap_bytes is None and args.with_umap:
            reason = artifacts.report_payload["umap"]["reason"]
            print(f"UMAP not performed ({reason}); primary results completed.")
    elif args.command == "check":
        check_experiment()
        print(json.dumps(summary(), sort_keys=True))
    elif args.command == "summary":
        print(json.dumps(summary(), sort_keys=True))
    elif args.command == "print-cluster":
        for row in print_cluster(args.cluster_id, args.limit):
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
