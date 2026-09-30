from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import run_v2c5_intent_discovery as discovery

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/run_v2c5_intent_discovery.py"


@pytest.fixture(scope="module")
def frozen_inputs() -> discovery.FrozenInputs:
    return discovery.load_frozen_inputs()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_matrix() -> np.ndarray:
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (matrix / np.linalg.norm(matrix, axis=1)[:, None]).astype(
        np.float32
    )


class MockHDBSCAN:
    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters = parameters
        self.probabilities_ = np.asarray(
            [0.95, 0.8, 0.6, 0.9, 0.7, 0.0],
            dtype=np.float64,
        )

    def fit_predict(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.shape == (6, 3)
        return np.asarray([4, 4, 4, 9, 9, -1], dtype=np.int64)


def synthetic_runs() -> list[discovery.ClusterRun]:
    return discovery.run_hdbscan_grid(
        normalized_matrix(),
        [f"id-{index}" for index in range(6)],
        lambda parameters: MockHDBSCAN(dict(parameters)),
        expected_count=6,
        expected_dimensions=3,
    )


def test_frozen_input_hashes_population_and_ordered_hashes(
    frozen_inputs: discovery.FrozenInputs,
) -> None:
    assert sha256_file(discovery.DEFAULT_PATHS.contract) == (
        discovery.CONTRACT_SHA256
    )
    assert sha256_file(discovery.DEFAULT_PATHS.corpus) == (
        discovery.CORPUS_SHA256
    )
    assert sha256_file(discovery.DEFAULT_PATHS.corpus_manifest) == (
        discovery.CORPUS_MANIFEST_SHA256
    )
    assert len(frozen_inputs.records) == 6372
    assert discovery.ordered_lines_sha256(
        [str(row["discovery_id"]) for row in frozen_inputs.records]
    ) == discovery.ORDERED_DISCOVERY_ID_SHA256
    assert discovery.ordered_lines_sha256(
        [str(row["normalized_text_sha256"]) for row in frozen_inputs.records]
    ) == discovery.ORDERED_NORMALIZED_TEXT_SHA256


def test_primary_population_is_unsupported_and_development_only(
    frozen_inputs: discovery.FrozenInputs,
) -> None:
    allowed = {
        ("sentinelvoice_v2c1_internal", "train"),
        ("sentinelvoice_v2c1_internal", "validation"),
        ("banking77", "train"),
        ("clinc150_oos", "train"),
        ("clinc150_oos", "val"),
    }
    assert all(
        row["current_sentinelvoice_intent"]
        == discovery.PRIMARY_INTENT
        for row in frozen_inputs.records
    )
    occurrences = [
        occurrence
        for row in frozen_inputs.records
        for occurrence in row["occurrences"]
    ]
    assert all(occurrence["data_role"] == "development" for occurrence in occurrences)
    assert all(
        (occurrence["source_dataset"], occurrence["source_split"])
        in allowed
        for occurrence in occurrences
    )


def test_sealed_v2c4_holdout_paths_fail_before_access() -> None:
    for filename in discovery.FORBIDDEN_INPUT_FILENAMES:
        with pytest.raises(ValueError, match="sealed V2-C4 holdout"):
            discovery.read_input_bytes(discovery.ML_ROOT / filename)


def test_representation_is_exact_and_text_only(
    frozen_inputs: discovery.FrozenInputs,
) -> None:
    representation = frozen_inputs.contract["representation"]
    policy = frozen_inputs.corpus["embedding_input_policy"]

    assert discovery.EMBEDDING_PROVIDER == "FastEmbed"
    assert discovery.MODEL_IDENTIFIER == "BAAI/bge-small-en-v1.5"
    assert discovery.EMBEDDING_METHOD == "passage_embed"
    assert discovery.DIMENSIONS == 384
    assert representation["fine_tuning"] is False
    assert representation["external_api"] is False
    assert representation["l2_normalization_required"] is True
    assert representation["input_space"] == (
        "original_384_dimensional_embedding_space"
    )
    assert representation["umap_reduced_vectors_used_for_clustering"] is False
    assert policy == {
        "fields": ["text"],
        "labels_concatenated_with_text": False,
        "native_labels_are_features": False,
        "source_metadata_concatenated_with_text": False,
    }


def test_embedding_validation_requires_float32_finite_l2_rows() -> None:
    matrix = normalized_matrix()
    validated = discovery.validate_embedding_matrix(
        matrix,
        expected_count=6,
        expected_dimensions=3,
    )
    assert validated.dtype == np.float32
    assert np.allclose(np.linalg.norm(validated, axis=1), 1.0)

    invalid = matrix.copy()
    invalid[0] *= 2
    with pytest.raises(ValueError, match="not L2 normalized"):
        discovery.validate_embedding_matrix(
            invalid,
            expected_count=6,
            expected_dimensions=3,
        )


def test_cache_content_hash_covers_float_data_ids_and_text_hashes() -> None:
    matrix = normalized_matrix()
    ids = [f"id-{index}" for index in range(6)]
    hashes = [f"{index:064x}" for index in range(6)]
    original = discovery.matrix_content_sha256(matrix, ids, hashes)

    changed = matrix.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(0.0))
    assert discovery.matrix_content_sha256(changed, ids, hashes) != original
    assert discovery.matrix_content_sha256(
        matrix,
        list(reversed(ids)),
        hashes,
    ) != original
    assert discovery.matrix_content_sha256(
        matrix,
        ids,
        list(reversed(hashes)),
    ) != original


def test_hdbscan_primary_and_exact_nine_run_grid() -> None:
    recorded: list[dict[str, Any]] = []

    def factory(parameters: Any) -> MockHDBSCAN:
        copied = dict(parameters)
        recorded.append(copied)
        return MockHDBSCAN(copied)

    runs = discovery.run_hdbscan_grid(
        normalized_matrix(),
        [f"id-{index}" for index in range(6)],
        factory,
        expected_count=6,
        expected_dimensions=3,
    )

    assert len(runs) == 9
    assert recorded == discovery.SENSITIVITY_CONFIGS
    assert sum(run.config_id == discovery.PRIMARY_CONFIG_ID for run in runs) == 1
    assert discovery.primary_run(runs).parameters == {
        "allow_single_cluster": False,
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_method": "eom",
        "metric": "euclidean",
        "min_cluster_size": 30,
        "min_samples": 10,
    }


def test_hdbscan_constructor_explicitly_preserves_copy_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_hdbscan(**parameters: Any) -> object:
        captured.update(parameters)
        return sentinel

    monkeypatch.setattr("sklearn.cluster.HDBSCAN", fake_hdbscan)

    result = discovery.default_hdbscan_factory(discovery.PRIMARY_CONFIG)

    assert result is sentinel
    assert captured == {**discovery.PRIMARY_CONFIG, "copy": False}


def test_primary_cannot_be_replaced_post_hoc(
    frozen_inputs: discovery.FrozenInputs,
) -> None:
    sensitivity = frozen_inputs.contract["sensitivity_analysis"]
    assert sensitivity["primary_30_10_result_remains_primary"] is True
    assert (
        sensitivity["post_hoc_best_looking_configuration_selection_allowed"]
        is False
    )


def test_canonical_cluster_ids_are_membership_based_and_deterministic() -> None:
    ids = ["delta", "alpha", "noise", "charlie"]
    first, _ = discovery.canonicalize_cluster_labels([7, 7, -1, 3], ids)
    second, _ = discovery.canonicalize_cluster_labels([2, 2, -1, 9], ids)

    assert first == second
    assert first[2] == "noise"
    assert first[0].startswith("cluster-")
    assert first[3].startswith("cluster-")


def test_representatives_use_geometry_and_boundaries_use_membership() -> None:
    matrix = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.6, 0.6], [0.5, 0.5]],
        dtype=np.float32,
    )
    representatives, boundaries = discovery.representative_and_boundary_ids(
        [0, 1, 2, 3],
        matrix,
        ["far-a", "far-b", "near-b", "near-a"],
        np.asarray([0.9, 0.8, 0.2, 0.1]),
    )

    assert representatives[:2] == ["near-a", "near-b"]
    assert boundaries[:2] == ["near-a", "near-b"]


def test_source_and_native_label_concentration_are_post_hoc() -> None:
    run = discovery.ClusterRun(
        config_id=discovery.PRIMARY_CONFIG_ID,
        parameters=discovery.PRIMARY_CONFIG,
        raw_labels=np.asarray([0, 0, 0]),
        membership_values=np.asarray([0.9, 0.8, 0.7]),
        canonical_labels=["cluster-test"] * 3,
        cluster_definitions=[
            {
                "canonical_cluster_id": "cluster-test",
                "cluster_signature_sha256": "a" * 64,
                "member_count": 3,
                "raw_hdbscan_label": 0,
            }
        ],
    )
    records = [
        {
            "discovery_id": f"id-{index}",
            "current_risk": "ESCALATION_OR_UNCERTAIN",
            "source_occurrence_count": 1,
            "occurrences": [
                {
                    "source_dataset": "banking77" if index < 2 else "clinc150_oos",
                    "native_external_label": "native-a" if index < 2 else "native-b",
                }
            ],
        }
        for index in range(3)
    ]
    diagnostics = discovery.cluster_diagnostics(
        run,
        normalized_matrix()[:3],
        records,
    )[0]

    assert diagnostics["source_dataset_concentration"]["counts"] == {
        "banking77": 2,
        "clinc150_oos": 1,
    }
    assert diagnostics["source_dataset_concentration"]["dominant"] == "banking77"
    assert diagnostics["native_external_label_concentration"]["counts"] == {
        "native-a": 2,
        "native-b": 1,
    }


def test_membership_and_sensitivity_metrics_are_diagnostic_only() -> None:
    diagnostics = discovery.sensitivity_diagnostics(synthetic_runs())

    assert len(diagnostics) == 9
    assert all(
        item["comparison_role"] == "diagnostic_only_no_selection_threshold"
        for item in diagnostics
    )
    assert all(
        item["adjusted_rand_index_vs_primary_all_records"] == pytest.approx(1.0)
        for item in diagnostics
    )
    assert "not a semantic-correctness probability" in SCRIPT_PATH.read_text(
        encoding="utf-8"
    )


def test_assignments_omit_raw_text_and_semantic_intent_names() -> None:
    records = [
        {
            "discovery_id": f"id-{index}",
            "normalized_text_sha256": f"{index:064x}",
        }
        for index in range(6)
    ]
    payload = discovery.build_assignments_payload(synthetic_runs(), records)
    serialized = discovery.stable_json_bytes(payload)

    discovery.validate_no_raw_text(payload)
    assert b'"text"' not in serialized
    assert payload["semantic_intent_names_assigned"] is False
    with pytest.raises(ValueError, match="raw text"):
        discovery.validate_no_raw_text({"text": "must not persist"})


def test_report_governance_is_non_authoritative() -> None:
    assert discovery.governance_payload() == {
        "analysis_role": "unsupervised_taxonomy_discovery_evidence",
        "classifier_output_role": "advisory_only",
        "final_acceptance_evidence": False,
        "human_adjudication_required": True,
        "new_intents_created": False,
        "runtime_behavior_changed": False,
        "supervised_training_performed": False,
        "taxonomy_changed": False,
    }


def test_preflight_performs_no_embedding_or_clustering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = discovery.ExperimentPaths(
        contract=tmp_path / "contract.json",
        corpus=tmp_path / "corpus.json",
        corpus_manifest=tmp_path / "corpus.manifest.json",
        embedding_cache=tmp_path / "local/cache.npz",
        embedding_cache_metadata=tmp_path / "local/cache.json",
        report_output=tmp_path / "report.json",
        assignments_output=tmp_path / "assignments.json",
        manifest_output=tmp_path / "manifest.json",
        umap_output=tmp_path / "umap.json",
    )
    monkeypatch.setattr(
        discovery,
        "load_frozen_inputs",
        lambda unused_paths: discovery.FrozenInputs({}, {}, {}, [{}]),
    )
    monkeypatch.setattr(
        discovery,
        "dependency_status",
        lambda: {
            "fastembed_available": True,
            "sklearn_cluster_hdbscan_available": True,
        },
    )
    monkeypatch.setattr(
        discovery,
        "generate_embeddings",
        lambda unused_records: pytest.fail("preflight generated embeddings"),
    )
    monkeypatch.setattr(
        discovery,
        "load_embedding_cache",
        lambda *args, **kwargs: pytest.fail("preflight loaded embeddings"),
    )
    monkeypatch.setattr(
        discovery,
        "run_hdbscan_grid",
        lambda *args, **kwargs: pytest.fail("preflight ran clustering"),
    )

    status = discovery.preflight(paths)

    assert status["hdbscan_fit_count_on_preflight"] == 0
    assert status["cache_status"] == "missing_will_generate_on_run"


def test_check_uses_verified_cache_and_reproduces_clustering() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "check_experiment"
    )
    calls = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "load_embedding_cache" in calls
    assert "run_hdbscan_grid" in calls
    assert "generate_embeddings" not in calls
    assert "load_or_create_embedding_cache" not in calls


def test_umap_is_optional_visualization_and_never_clustering_input() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    grid_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_hdbscan_grid"
    )
    names = {node.id for node in ast.walk(grid_function) if isinstance(node, ast.Name)}

    assert "umap" not in names
    assert discovery.UMAP_RANDOM_STATE == 20260928


def test_print_cluster_joins_text_for_review_without_persisting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = discovery.FrozenInputs(
        {},
        {},
        {},
        [
            {
                "discovery_id": "id-1",
                "text": "Human review text",
                "occurrences": [
                    {
                        "source_dataset": "banking77",
                        "native_external_label": "native-label",
                    }
                ],
            }
        ],
    )
    report = {
        "primary": {
            "cluster_diagnostics": [
                {
                    "canonical_cluster_id": "cluster-test",
                    "representative_discovery_ids": ["id-1"],
                    "boundary_discovery_ids": [],
                }
            ]
        }
    }
    assignments = {
        "assignments": [
            {
                "discovery_id": "id-1",
                "primary_canonical_cluster_id": "cluster-test",
                "primary_membership_value": 0.75,
            }
        ]
    }
    monkeypatch.setattr(discovery, "load_frozen_inputs", lambda unused: inputs)
    monkeypatch.setattr(
        discovery,
        "read_input_bytes",
        lambda path: b"report" if path.name.endswith("report.json") else b"assignments",
    )
    monkeypatch.setattr(
        discovery,
        "load_json_bytes",
        lambda value, unused_label: report if value == b"report" else assignments,
    )

    review = discovery.print_cluster("cluster-test", 1)

    assert review == [
        {
            "canonical_cluster_id": "cluster-test",
            "discovery_id": "id-1",
            "membership_value": 0.75,
            "native_external_labels": ["native-label"],
            "source_datasets": ["banking77"],
            "text": "Human review text",
        }
    ]


def test_print_cluster_cli_emits_review_text_only_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        discovery,
        "print_cluster",
        lambda cluster_id, limit: [
            {"cluster": cluster_id, "limit": limit, "text": "review-only"}
        ],
    )

    assert discovery.main(
        ["print-cluster", "--cluster-id", "cluster-test", "--limit", "3"]
    ) == 0
    output = capsys.readouterr().out
    assert "review-only" in output
