"""Run the frozen V2-C6 fresh fallback semantic evaluation exactly once.

Preflight and result checking are read-only and never call a provider. Only
``--run`` may call Groq, sequentially, once for each of the 400 frozen cases.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.app.agent.dependencies import (
    VERIFIER_REASONING_EFFORT,
    VERIFIER_SDK_MAX_RETRIES,
    VERIFIER_SDK_TIMEOUT_SECONDS,
    build_protected_action_verifier,
)
from backend.app.agent.protected_action_verifier import (
    VERIFIER_MAX_COMPLETION_TOKENS,
    VERIFIER_TIMEOUT_SECONDS,
    LLMProtectedActionSemanticVerifier,
    ProtectedActionSemanticDecision,
    ProtectedActionVerificationError,
)
from backend.app.config.settings import Settings
from backend.app.providers.groq_llm import GroqLLMProvider, LLMProviderError
from backend.app.providers.llm import LLMProvider, LLMResponse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_RELATIVE = "data/evals/v2/ml"
CONTRACT_RELATIVE_PATH = (
    f"{ML_RELATIVE}/v2c6_routing_fallback_fresh_semantic_evaluation_contract.json"
)
FAMILY_RELATIVE_PATHS = (
    f"{ML_RELATIVE}/v2c6_routing_fallback_fresh_semantic_family_1.json",
    f"{ML_RELATIVE}/v2c6_routing_fallback_fresh_semantic_family_2.json",
)
RESULTS_RELATIVE_PATH = (
    f"{ML_RELATIVE}/v2c6_routing_fallback_fresh_semantic_evaluation_results.json"
)
MANIFEST_RELATIVE_PATH = (
    f"{ML_RELATIVE}/v2c6_routing_fallback_fresh_semantic_evaluation_results.manifest.json"
)
CASES_RELATIVE_PATH = (
    f"{ML_RELATIVE}/v2c6_routing_fallback_verifier_budget_development_cases.json"
)
R3_HASH_SOURCE_RELATIVE_PATH = (
    f"{ML_RELATIVE}/v2c6_r3_fresh_source_evaluation_dataset.json"
)
PROHIBITED_HOLDOUT_RELATIVE_PATH = f"{ML_RELATIVE}/v2c5_final_holdout.json"
RUNNER_RELATIVE_PATH = (
    "scripts/run_v2c6_routing_fallback_fresh_semantic_evaluation.py"
)

CONTRACT_PATH = REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH
FAMILY_PATHS = tuple(REPOSITORY_ROOT / path for path in FAMILY_RELATIVE_PATHS)
RESULTS_PATH = REPOSITORY_ROOT / RESULTS_RELATIVE_PATH
MANIFEST_PATH = REPOSITORY_ROOT / MANIFEST_RELATIVE_PATH
CASES_PATH = REPOSITORY_ROOT / CASES_RELATIVE_PATH
R3_HASH_SOURCE_PATH = REPOSITORY_ROOT / R3_HASH_SOURCE_RELATIVE_PATH

EXPECTED_CONTRACT_SHA256 = (
    "4edd9ef5361cdc16aac770e5e0306bd4eb8c643c814ccbf3a753535b31131680"
)
CONTRACT_SCHEMA = "v2c6-routing-fallback-fresh-semantic-evaluation-contract.v1"
FAMILY_SCHEMA = "v2c6-routing-fallback-fresh-semantic-family.v1"
RESULTS_SCHEMA = "v2c6-routing-fallback-fresh-semantic-evaluation-results.v1"
MANIFEST_SCHEMA = (
    "v2c6-routing-fallback-fresh-semantic-evaluation-results-manifest.v1"
)
EXPECTED_MODEL = "openai/gpt-oss-20b"
EXPECTED_FAMILIES = (
    "v2c6_fallback_fresh_family_1",
    "v2c6_fallback_fresh_family_2",
)
ACTIONS = ("freeze_card", "create_dispute")
BOUNDARY_SUBTYPES = (
    "informational",
    "hypothetical",
    "advice_or_guidance",
    "negated_or_not_requested",
    "ambiguous_or_current_context",
)
DECISIONS = frozenset(decision.value for decision in ProtectedActionSemanticDecision)
REQUIRED_SCOPES = (
    "fresh_family_1",
    "fresh_family_2",
    "pooled_fresh",
    "pooled_fresh_freeze_card",
    "pooled_fresh_create_dispute",
)
RECALL_THRESHOLD = 0.80
FPR_THRESHOLD = 0.01
FRESH_SEMANTIC_PASS = "FRESH_SEMANTIC_PASS"
FRESH_SEMANTIC_FAIL = "FRESH_SEMANTIC_FAIL"

GOVERNANCE = {
    "fresh_evaluation": True,
    "development_only": False,
    "final_holdout": False,
    "training_eligible": False,
    "prompt_tuning_eligible": False,
    "fresh_evaluation_consumed": True,
    "application_retries": 0,
    "sdk_max_retries": 0,
    "alternate_budget_fallback": False,
    "step29i_authorized": False,
    "final_holdout_accessed": False,
    "runtime_configuration_changed": False,
}

CASE_RESULT_FIELDS = frozenset(
    {
        "case_id",
        "source_family",
        "proposed_action",
        "case_kind",
        "boundary_subtype",
        "gold_decision",
        "observed_decision",
        "failure_category",
        "provider",
        "model",
        "finish_reason",
        "provider_call_count",
        "structured_tool_call_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "provider_call_latency_ms",
        "verifier_wall_latency_ms",
        "provider_error_type",
        "provider_error_code",
        "provider_error_status",
    }
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def assert_path_allowed(path: Path) -> None:
    resolved = path.resolve()
    prohibited = (REPOSITORY_ROOT / PROHIBITED_HOLDOUT_RELATIVE_PATH).resolve()
    if resolved == prohibited or any("final_holdout" in part for part in resolved.parts):
        raise PermissionError("fresh evaluation must never access a final holdout")


def read_bytes(path: Path) -> bytes:
    assert_path_allowed(path)
    return path.read_bytes()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(read_bytes(path))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def normalized_sha256(text: str) -> str:
    return sha256_bytes(normalize_text(text).encode())


def write_create_once(path: Path, content: bytes) -> None:
    assert_path_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_family(payload: Mapping[str, Any], expected_family: str) -> list[dict[str, Any]]:
    if (
        payload.get("schema_version") != FAMILY_SCHEMA
        or payload.get("source_family") != expected_family
        or payload.get("record_count") != 200
        or payload.get("fresh_evaluation") is not True
        or payload.get("training_eligible") is not False
        or payload.get("prompt_tuning_eligible") is not False
    ):
        raise ValueError(f"fresh family identity changed: {expected_family}")
    provenance = payload.get("authoring_provenance")
    if not isinstance(provenance, dict) or (
        provenance.get("authoring_method") != "controlled_llm_assisted"
        or provenance.get("human_review_performed") is not False
        or provenance.get("independently_authored_from_frozen_semantic_definitions")
        is not True
        or provenance.get("prior_evaluation_utterances_consulted_during_authoring")
        is not False
        or provenance.get("other_fresh_family_consulted_during_authoring") is not False
    ):
        raise ValueError(f"fresh family provenance changed: {expected_family}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 200:
        raise ValueError(f"fresh family must contain 200 records: {expected_family}")
    validated: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("fresh case must be an object")
        if set(record) != {
            "case_id",
            "source_family",
            "proposed_action",
            "case_kind",
            "boundary_subtype",
            "gold_decision",
            "customer_utterance",
        }:
            raise ValueError(f"fresh case schema changed: {record.get('case_id')}")
        case_id = record["case_id"]
        action = record["proposed_action"]
        kind = record["case_kind"]
        subtype = record["boundary_subtype"]
        gold = record["gold_decision"]
        text = record["customer_utterance"]
        if (
            not isinstance(case_id, str)
            or not case_id.startswith(f"{expected_family}_")
            or record["source_family"] != expected_family
            or action not in ACTIONS
            or kind not in {"explicit", "boundary"}
            or gold not in DECISIONS
            or not isinstance(text, str)
            or not normalize_text(text)
        ):
            raise ValueError(f"invalid fresh case: {case_id}")
        if kind == "explicit" and (
            subtype is not None or gold != "EXPLICIT_CURRENT_ACTION"
        ):
            raise ValueError(f"invalid explicit case: {case_id}")
        if kind == "boundary" and (
            subtype not in BOUNDARY_SUBTYPES or gold == "EXPLICIT_CURRENT_ACTION"
        ):
            raise ValueError(f"invalid boundary case: {case_id}")
        validated.append(record)
    composition = Counter((row["proposed_action"], row["case_kind"]) for row in validated)
    if composition != Counter(
        {(action, kind): 50 for action in ACTIONS for kind in ("explicit", "boundary")}
    ):
        raise ValueError(f"fresh family composition changed: {expected_family}")
    subtype_counts = Counter(
        (row["proposed_action"], row["boundary_subtype"])
        for row in validated
        if row["case_kind"] == "boundary"
    )
    if subtype_counts != Counter(
        {(action, subtype): 10 for action in ACTIONS for subtype in BOUNDARY_SUBTYPES}
    ):
        raise ValueError(f"fresh boundary composition changed: {expected_family}")
    return validated


def validate_no_duplicates_or_leakage(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    ids = [str(record["case_id"]) for record in records]
    normalized: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        normalized[normalized_sha256(str(record["customer_utterance"]))].append(record)
    if len(ids) != len(set(ids)):
        raise ValueError("fresh case IDs must be unique")
    if any(len(rows) > 1 for rows in normalized.values()):
        raise ValueError("normalized duplicate exists within or across fresh families")
    if any(
        len({(row["proposed_action"], row["gold_decision"]) for row in rows}) > 1
        for rows in normalized.values()
    ):
        raise ValueError("cross-action normalized collision has conflicting labels")

    consumed = read_json(CASES_PATH)
    consumed_hashes = {
        normalized_sha256(str(row["customer_utterance"]))
        for row in consumed.get("cases", [])
    }
    r3 = read_json(R3_HASH_SOURCE_PATH)
    r3_hashes = {
        str(row["normalized_text_sha256"])
        for row in r3.get("examples", [])
    }
    fresh_hashes = set(normalized)
    consumed_overlap = len(fresh_hashes & consumed_hashes)
    r3_overlap = len(fresh_hashes & r3_hashes)
    if consumed_overlap or r3_overlap:
        raise ValueError("fresh cases overlap governed prior evidence")
    return {
        "normalized_duplicate_count": 0,
        "cross_family_normalized_duplicate_count": 0,
        "cross_action_conflicting_collision_count": 0,
        "consumed_development_overlap_count": 0,
        "r3_fresh_hash_overlap_count": 0,
    }


def load_and_validate() -> tuple[dict[str, Any], list[dict[str, Any]], list[bytes]]:
    contract_bytes = read_bytes(CONTRACT_PATH)
    if sha256_bytes(contract_bytes) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("fresh evaluation contract SHA mismatch")
    contract = json.loads(contract_bytes)
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("status") != "FROZEN"
        or contract.get("fresh_evaluation") is not True
        or contract.get("development_only") is not False
        or contract.get("final_holdout") is not False
        or contract.get("training_eligible") is not False
        or contract.get("prompt_tuning_eligible") is not False
        or contract.get("governance", {}).get("step29i_authorized") is not False
    ):
        raise ValueError("fresh evaluation contract identity changed")
    for name, binding in contract["source_bindings"].items():
        if "final_holdout" in binding["path"]:
            raise ValueError(f"prohibited source binding: {name}")
        source = REPOSITORY_ROOT / binding["path"]
        if sha256_bytes(read_bytes(source)) != binding["sha256"]:
            raise ValueError(f"bound source changed: {name}")
    family_bytes = [read_bytes(path) for path in FAMILY_PATHS]
    records: list[dict[str, Any]] = []
    for raw, family in zip(family_bytes, EXPECTED_FAMILIES, strict=True):
        records.extend(validate_family(json.loads(raw), family))
    if len(records) != 400:
        raise ValueError("fresh evaluation must contain exactly 400 records")
    validate_no_duplicates_or_leakage(records)
    return contract, records, family_bytes


def verify_production_construction(settings: Settings) -> LLMProtectedActionSemanticVerifier:
    verifier = build_protected_action_verifier(settings)
    inner = verifier._llm
    client = getattr(inner, "_client", None)
    if (
        not isinstance(inner, GroqLLMProvider)
        or inner.model != EXPECTED_MODEL
        or inner.max_completion_tokens != 256
        or inner.reasoning_effort != "low"
        or getattr(client, "max_retries", None) != 0
        or getattr(client, "timeout", None) != 2.0
        or verifier._timeout_seconds != 2.0
        or VERIFIER_MAX_COMPLETION_TOKENS != 256
        or VERIFIER_TIMEOUT_SECONDS != 2.0
        or VERIFIER_REASONING_EFFORT != "low"
        or VERIFIER_SDK_MAX_RETRIES != 0
        or VERIFIER_SDK_TIMEOUT_SECONDS != 2.0
    ):
        raise ValueError("production verifier no longer matches frozen configuration")
    return verifier


class RecordingProvider:
    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.call_count = 0
        self.response: LLMResponse | None = None
        self.error: dict[str, Any] | None = None
        self.latency_ms: float | None = None

    @property
    def provider(self) -> str:
        return str(getattr(self._inner, "provider", "unknown"))

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "unknown"))

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        self.call_count += 1
        started = time.perf_counter()
        try:
            self.response = await self._inner.generate(messages=messages, tools=tools)
        except LLMProviderError as exc:
            self.error = describe_provider_error(exc)
            raise
        finally:
            self.latency_ms = (time.perf_counter() - started) * 1000
        return self.response


def describe_provider_error(exc: BaseException) -> dict[str, Any]:
    cause = exc.__cause__ or exc
    body = getattr(cause, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    return {
        "provider_error_type": type(cause).__name__,
        "provider_error_code": str(code) if code else None,
        "provider_error_status": getattr(cause, "status_code", None),
    }


async def run_case(
    verifier: LLMProtectedActionSemanticVerifier,
    inner_provider: LLMProvider,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    recorder = RecordingProvider(inner_provider)
    verifier._llm = recorder
    observed: str | None = None
    failure: str | None = None
    started = time.perf_counter()
    try:
        decision = await verifier.verify(
            user_text=str(case["customer_utterance"]),
            proposed_action=str(case["proposed_action"]),
        )
        observed = decision.value
    except ProtectedActionVerificationError as exc:
        failure = exc.category.value
    wall_ms = (time.perf_counter() - started) * 1000
    response = recorder.response
    error = recorder.error or {
        "provider_error_type": None,
        "provider_error_code": None,
        "provider_error_status": None,
    }
    return {
        "case_id": case["case_id"],
        "source_family": case["source_family"],
        "proposed_action": case["proposed_action"],
        "case_kind": case["case_kind"],
        "boundary_subtype": case["boundary_subtype"],
        "gold_decision": case["gold_decision"],
        "observed_decision": observed,
        "failure_category": failure,
        "provider": recorder.provider,
        "model": response.model if response is not None else recorder.model,
        "finish_reason": response.finish_reason if response is not None else None,
        "provider_call_count": recorder.call_count,
        "structured_tool_call_count": len(response.tool_calls) if response else 0,
        "prompt_tokens": response.usage.prompt_tokens if response else None,
        "completion_tokens": response.usage.completion_tokens if response else None,
        "total_tokens": response.usage.total_tokens if response else None,
        "provider_call_latency_ms": (
            round(recorder.latency_ms, 3) if recorder.latency_ms is not None else None
        ),
        "verifier_wall_latency_ms": round(wall_ms, 3),
        **error,
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("semantic metric denominator must be positive")
    return numerator / denominator


def _scope_records(records: Sequence[Mapping[str, Any]], scope: str) -> list[Mapping[str, Any]]:
    if scope == "fresh_family_1":
        return [r for r in records if r["source_family"] == EXPECTED_FAMILIES[0]]
    if scope == "fresh_family_2":
        return [r for r in records if r["source_family"] == EXPECTED_FAMILIES[1]]
    if scope == "pooled_fresh":
        return list(records)
    if scope == "pooled_fresh_freeze_card":
        return [r for r in records if r["proposed_action"] == "freeze_card"]
    if scope == "pooled_fresh_create_dispute":
        return [r for r in records if r["proposed_action"] == "create_dispute"]
    raise ValueError(f"unknown evaluation scope: {scope}")


def scope_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [r for r in records if r["gold_decision"] == "EXPLICIT_CURRENT_ACTION"]
    negatives = [r for r in records if r["gold_decision"] != "EXPLICIT_CURRENT_ACTION"]
    hits = sum(r["observed_decision"] == "EXPLICIT_CURRENT_ACTION" for r in positives)
    false_positives = sum(
        r["observed_decision"] == "EXPLICIT_CURRENT_ACTION" for r in negatives
    )
    recall = _ratio(hits, len(positives))
    fpr = _ratio(false_positives, len(negatives))
    return {
        "gold_positive_count": len(positives),
        "gold_negative_count": len(negatives),
        "explicit_hit_count": hits,
        "false_positive_count": false_positives,
        "explicit_protected_request_recall": recall,
        "protected_semantic_false_positive_rate": fpr,
        "recall_gate_pass": recall >= RECALL_THRESHOLD,
        "fpr_gate_pass": fpr <= FPR_THRESHOLD,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def _latency(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": max(values) if values else None,
    }


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scopes = {scope: scope_metrics(_scope_records(records, scope)) for scope in REQUIRED_SCOPES}
    status = (
        FRESH_SEMANTIC_PASS
        if all(row["recall_gate_pass"] and row["fpr_gate_pass"] for row in scopes.values())
        else FRESH_SEMANTIC_FAIL
    )
    boundary = [r for r in records if r["case_kind"] == "boundary"]
    provider_latencies = [
        float(r["provider_call_latency_ms"])
        for r in records
        if r["provider_call_latency_ms"] is not None
    ]
    wall_latencies = [float(r["verifier_wall_latency_ms"]) for r in records]
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        "result_status": status,
        "case_count": len(records),
        "required_scope_metrics": scopes,
        "exact_three_way_accuracy": _ratio(
            sum(r["observed_decision"] == r["gold_decision"] for r in records),
            len(records),
        ),
        "safe_non_explicit_boundary_rate": _ratio(
            sum(
                r["observed_decision"] in {"AMBIGUOUS_OR_INFORMATIONAL", "NOT_REQUESTED"}
                for r in boundary
            ),
            len(boundary),
        ),
        "decision_confusion_counts": dict(
            sorted(
                Counter(
                    f"{r['gold_decision']}->{r['observed_decision'] or 'FAILURE'}"
                    for r in records
                ).items()
            )
        ),
        "failure_counts": dict(
            sorted(Counter(r["failure_category"] for r in records if r["failure_category"]).items())
        ),
        "finish_reason_counts": dict(
            sorted(Counter(str(r["finish_reason"]) for r in records).items())
        ),
        "token_totals": {
            field: sum(int(r[field]) for r in records if r[field] is not None)
            for field in token_fields
        },
        "latency_ms": {
            "provider_call": _latency(provider_latencies),
            "verifier_wall": _latency(wall_latencies),
        },
        "total_provider_calls": sum(int(r["provider_call_count"]) for r in records),
        "retries": 0,
    }


def result_run_configuration() -> dict[str, Any]:
    return {
        "model": EXPECTED_MODEL,
        "max_completion_tokens": 256,
        "reasoning_effort": "low",
        "sdk_max_retries": 0,
        "sdk_timeout_seconds": 2.0,
        "outer_timeout_seconds": 2.0,
        "application_retries": 0,
        "provider_calls_per_record": 1,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "temperature": 0,
        "execution": "sequential",
    }


def build_payload(
    contract_bytes: bytes,
    family_bytes: Sequence[bytes],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": RESULTS_SCHEMA,
        "phase": "V2-C6 Fresh Fallback Semantic Evaluation",
        "contract": {"path": CONTRACT_RELATIVE_PATH, "sha256": sha256_bytes(contract_bytes)},
        "fresh_families": [
            {"path": path, "sha256": sha256_bytes(raw)}
            for path, raw in zip(FAMILY_RELATIVE_PATHS, family_bytes, strict=True)
        ],
        "run_configuration": result_run_configuration(),
        "environment": {
            "python": platform.python_version(),
            "groq_sdk": _package_version("groq"),
        },
        "cases": records,
        "aggregate": aggregate(records),
        "governance": dict(GOVERNANCE),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_manifest(payload: Mapping[str, Any], result_bytes: bytes) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "results": {"path": RESULTS_RELATIVE_PATH, "sha256": sha256_bytes(result_bytes)},
        "runner": {
            "path": RUNNER_RELATIVE_PATH,
            "sha256": sha256_bytes(read_bytes(Path(__file__).resolve())),
        },
        "contract_sha256": payload["contract"]["sha256"],
        "fresh_family_sha256": [row["sha256"] for row in payload["fresh_families"]],
        "result_status": payload["aggregate"]["result_status"],
        "governance": payload["governance"],
    }


def assert_no_utterances(payload: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    serialized = json.dumps(payload)
    if "customer_utterance" in serialized or any(
        str(case["customer_utterance"]) in serialized for case in cases
    ):
        raise ValueError("results must not contain customer utterances")


def run(
    *,
    settings: Settings | None = None,
    inner_provider_override: LLMProvider | None = None,
) -> dict[str, Any]:
    if RESULTS_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError("fresh evaluation results exist; refusing to rerun")
    contract, cases, family_bytes = load_and_validate()
    contract_bytes = read_bytes(CONTRACT_PATH)
    verifier = verify_production_construction(settings or Settings())
    inner = verifier._llm
    provider = inner_provider_override or inner

    async def execute() -> list[dict[str, Any]]:
        return [await run_case(verifier, provider, case) for case in cases]

    records = asyncio.run(execute())
    payload = build_payload(contract_bytes, family_bytes, records)
    assert_no_utterances(payload, cases)
    result_bytes = stable_json_bytes(payload)
    write_create_once(RESULTS_PATH, result_bytes)
    write_create_once(MANIFEST_PATH, stable_json_bytes(build_manifest(payload, result_bytes)))
    del contract
    return payload["aggregate"]


def check_results() -> dict[str, Any]:
    contract, cases, family_bytes = load_and_validate()
    contract_bytes = read_bytes(CONTRACT_PATH)
    result_bytes = read_bytes(RESULTS_PATH)
    payload = json.loads(result_bytes)
    manifest = read_json(MANIFEST_PATH)
    if payload.get("schema_version") != RESULTS_SCHEMA:
        raise ValueError("fresh result schema changed")
    if stable_json_bytes(payload) != result_bytes:
        raise ValueError("fresh results are not canonical stable JSON")
    if payload.get("contract") != {
        "path": CONTRACT_RELATIVE_PATH,
        "sha256": sha256_bytes(contract_bytes),
    }:
        raise ValueError("fresh result contract lineage changed")
    expected_families = [
        {"path": path, "sha256": sha256_bytes(raw)}
        for path, raw in zip(FAMILY_RELATIVE_PATHS, family_bytes, strict=True)
    ]
    if payload.get("fresh_families") != expected_families:
        raise ValueError("fresh family result lineage changed")
    records = payload.get("cases")
    if not isinstance(records, list) or len(records) != 400:
        raise ValueError("fresh results must contain exactly 400 case records")
    if [r["case_id"] for r in records] != [case["case_id"] for case in cases]:
        raise ValueError("fresh result case IDs or order changed")
    if any(set(record) != CASE_RESULT_FIELDS for record in records):
        raise ValueError("fresh per-case result schema changed")
    if any(record["provider_call_count"] != 1 for record in records):
        raise ValueError("each fresh case must have exactly one provider call")
    if payload.get("run_configuration") != result_run_configuration():
        raise ValueError("fresh result runtime configuration changed")
    if payload.get("aggregate") != aggregate(records):
        raise ValueError("fresh aggregate does not match case records")
    if payload.get("governance") != GOVERNANCE:
        raise ValueError("fresh result governance changed")
    assert_no_utterances(payload, cases)
    if manifest != build_manifest(payload, result_bytes):
        raise ValueError("fresh result manifest lineage changed")
    del contract
    return {
        "results_valid": True,
        "result_status": payload["aggregate"]["result_status"],
        "case_count": 400,
        "total_provider_calls": payload["aggregate"]["total_provider_calls"],
        "required_scope_metrics": payload["aggregate"]["required_scope_metrics"],
        "provider_calls_performed": False,
        "files_written": False,
        "fresh_evaluation": True,
        "step29i_authorized": False,
        "final_holdout_accessed": False,
    }


def preflight(settings_factory: Callable[[], Settings] = Settings) -> dict[str, Any]:
    contract, cases, family_bytes = load_and_validate()
    settings = settings_factory()
    key_configured = bool(
        settings.groq_api_key is not None and settings.groq_api_key.get_secret_value()
    )
    construction_verified = False
    if key_configured:
        verify_production_construction(settings)
        construction_verified = True
    results_present = RESULTS_PATH.exists() or MANIFEST_PATH.exists()
    ready = (
        settings.llm_model == EXPECTED_MODEL
        and key_configured
        and construction_verified
        and not results_present
    )
    return {
        "status": "READY" if ready else "NOT_READY",
        "contract_sha256": sha256_bytes(read_bytes(CONTRACT_PATH)),
        "fresh_family_sha256": [sha256_bytes(raw) for raw in family_bytes],
        "case_count": len(cases),
        "family_counts": dict(sorted(Counter(r["source_family"] for r in cases).items())),
        "production_construction_verified": construction_verified,
        "frozen_runtime_configuration": contract["frozen_runtime_configuration"],
        "result_artifacts_present": results_present,
        "provider_calls_performed": False,
        "files_written": False,
        "fresh_evaluation_started": False,
        "step29i_authorized": False,
        "final_holdout_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--check-results", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.preflight:
        result = preflight()
    elif args.check_results:
        result = check_results()
    else:
        result = run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
