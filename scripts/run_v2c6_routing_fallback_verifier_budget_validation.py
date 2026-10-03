"""Development-only check of the 64-token protected-action verifier budget.

Answers one question: does the real production verifier path
(openai/gpt-oss-20b, max_completion_tokens 64, 2.0-second timeout, the
existing tool_choice "auto") reliably return one valid structured decision?
It is not fresh fallback evaluation, final semantic acceptance, or model
selection. Real Groq calls happen only under ``--run``.
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
import re
import tempfile
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from backend.app.agent.dependencies import build_protected_action_verifier
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
    f"{ML_RELATIVE}/v2c6_routing_fallback_verifier_budget_validation_contract.json"
)
CASES_RELATIVE_PATH = (
    f"{ML_RELATIVE}/v2c6_routing_fallback_verifier_budget_development_cases.json"
)
RESULTS_RELATIVE_PATH = (
    f"{ML_RELATIVE}/v2c6_routing_fallback_verifier_budget_validation_results.json"
)
MANIFEST_RELATIVE_PATH = (
    f"{ML_RELATIVE}/v2c6_routing_fallback_verifier_budget_validation_results.manifest.json"
)
R3_FRESH_RELATIVE_PATH = f"{ML_RELATIVE}/v2c6_r3_fresh_source_evaluation_dataset.json"
PROHIBITED_HOLDOUT_RELATIVE_PATH = f"{ML_RELATIVE}/v2c5_final_holdout.json"
RUNNER_RELATIVE_PATH = "scripts/run_v2c6_routing_fallback_verifier_budget_validation.py"

CONTRACT_PATH = REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH
CASES_PATH = REPOSITORY_ROOT / CASES_RELATIVE_PATH
RESULTS_PATH = REPOSITORY_ROOT / RESULTS_RELATIVE_PATH
MANIFEST_PATH = REPOSITORY_ROOT / MANIFEST_RELATIVE_PATH
R3_FRESH_PATH = REPOSITORY_ROOT / R3_FRESH_RELATIVE_PATH

EXPECTED_CONTRACT_SHA256 = (
    "ddb87a2620315a18cbda114187d86f2f7975b13ade4ec13bb3cbccb94e745e5a"
)
CONTRACT_SCHEMA = "v2c6-routing-fallback-verifier-budget-validation-contract.v1"
CASES_SCHEMA = "v2c6-routing-fallback-verifier-budget-development-cases.v1"
RESULTS_SCHEMA = "v2c6-routing-fallback-verifier-budget-validation-results.v1"
MANIFEST_SCHEMA = "v2c6-routing-fallback-verifier-budget-validation-results-manifest.v1"
EXPECTED_MODEL = "openai/gpt-oss-20b"
EXPECTED_CASE_COUNT = 20
PROTECTED_ACTIONS = ("freeze_card", "create_dispute")

KEEP_64 = "KEEP_64"
TOKEN_BUDGET_AMENDMENT_REQUIRED = "TOKEN_BUDGET_AMENDMENT_REQUIRED"
STRUCTURED_OUTPUT_POLICY_ISSUE = "STRUCTURED_OUTPUT_POLICY_ISSUE"
INCONCLUSIVE_PROVIDER_FAILURE = "INCONCLUSIVE_PROVIDER_FAILURE"

TOKEN_EXHAUSTION_FINISH_REASONS = frozenset({"length"})
STRUCTURED_OUTPUT_FAILURES = frozenset(
    {
        "zero_structured_calls",
        "multiple_structured_calls",
        "wrong_structured_tool_name",
        "invalid_arguments",
        "invalid_enum",
        "malformed_structured_output",
    }
)
STRUCTURED_OUTPUT_PROVIDER_ERROR_CODES = frozenset({"tool_use_failed"})
PROVIDER_FAILURES = frozenset(
    {"timeout", "provider_error", "unexpected_verifier_exception"}
)
_TOKEN_LIMIT_TEXT = re.compile(
    r"max[_ ]?(?:completion[_ ])?tokens|token limit|length limit|maximum (?:output )?tokens",
    re.IGNORECASE,
)

GOVERNANCE = {
    "development_only": True,
    "final_acceptance_evidence": False,
    "fresh_fallback_evaluation_started": False,
    "fresh_fallback_evaluation_authored": False,
    "cases_consumed": True,
    "automatic_retries": 0,
    "alternate_budgets_tested": False,
    "verifier_prompt_changed": False,
    "tool_choice_changed": False,
    "step29i_authorized": False,
    "final_holdout_accessed": False,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def assert_path_allowed(path: Path) -> None:
    resolved = path.resolve()
    prohibited = (REPOSITORY_ROOT / PROHIBITED_HOLDOUT_RELATIVE_PATH).resolve()
    if resolved == prohibited or any("final_holdout" in part for part in resolved.parts):
        raise PermissionError("budget validation must never access a final-holdout artifact")


def read_bytes(path: Path) -> bytes:
    assert_path_allowed(path)
    return path.read_bytes()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(read_bytes(path))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def normalized_sha256(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return sha256_bytes(normalized.encode())


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


def validate_cases(cases_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        cases_artifact.get("schema_version") != CASES_SCHEMA
        or cases_artifact.get("development_only") is not True
        or cases_artifact.get("eligible_for_future_fresh_evaluation") is not False
        or cases_artifact.get("eligible_for_final_acceptance") is not False
    ):
        raise ValueError("development case artifact identity or eligibility changed")
    cases = cases_artifact.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("exactly 20 development cases are required")
    composition = Counter((case["proposed_action"], case["case_kind"]) for case in cases)
    if composition != Counter(
        {(action, kind): 5 for action in PROTECTED_ACTIONS for kind in ("explicit", "boundary")}
    ):
        raise ValueError("development case composition changed")
    if len({case["case_id"] for case in cases}) != EXPECTED_CASE_COUNT:
        raise ValueError("duplicate development case_id")
    for case in cases:
        explicit = case["case_kind"] == "explicit"
        expected = case["expected_decision"]
        if explicit != (expected == "EXPLICIT_CURRENT_ACTION"):
            raise ValueError(f"expected decision inconsistent: {case['case_id']}")
        if expected not in ProtectedActionSemanticDecision.__members__:
            raise ValueError(f"invalid expected decision: {case['case_id']}")
        if not str(case.get("customer_utterance", "")).strip():
            raise ValueError(f"empty utterance: {case['case_id']}")
    return cases


def prohibited_evidence_audit(cases: Sequence[dict[str, Any]], raw: bytes) -> dict[str, Any]:
    """Reference and stored-hash overlap checks; never reads prohibited content."""
    authoring = json.loads(raw)["authoring"]
    if any(
        authoring[flag] is not False
        for flag in (
            "derived_from_r3_fresh_records",
            "derived_from_future_fresh_fallback_evaluation",
            "derived_from_v2c5_final_holdout",
        )
    ):
        raise ValueError("development cases declare prohibited provenance")
    text = json.dumps(list(cases))
    case_hashes = {normalized_sha256(case["customer_utterance"]) for case in cases}
    r3_hashes = {
        str(row.get("normalized_text_sha256"))
        for row in read_json(R3_FRESH_PATH).get("examples", [])
    }
    audit = {
        "final_holdout_path_referenced": "v2c5_final_holdout" in text,
        "r3_record_ids_referenced": "v2c6_r3_eval_" in text,
        "r3_fresh_normalized_hash_overlap_count": len(case_hashes & r3_hashes),
    }
    if any(audit.values()):
        raise ValueError("development cases reference or overlap prohibited evidence")
    return audit


def load_and_validate() -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    contract_bytes = read_bytes(CONTRACT_PATH)
    if sha256_bytes(contract_bytes) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("budget validation contract SHA mismatch")
    contract = json.loads(contract_bytes)
    configuration = contract["frozen_run_configuration"]
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("status") != "FROZEN"
        or configuration["max_completion_tokens"] != 64
        or configuration["timeout_seconds"] != 2.0
        or configuration["automatic_retries"] != 0
        or configuration["cases"] != EXPECTED_CASE_COUNT
        or configuration["model"] != EXPECTED_MODEL
        or configuration["fresh_evaluation"] is not False
    ):
        raise ValueError("frozen budget validation configuration changed")
    for name, binding in contract["source_bindings"].items():
        if sha256_bytes(read_bytes(REPOSITORY_ROOT / binding["path"])) != binding["sha256"]:
            raise ValueError(f"bound source changed: {name}")
    if VERIFIER_MAX_COMPLETION_TOKENS != 64 or VERIFIER_TIMEOUT_SECONDS != 2.0:
        raise ValueError("production verifier budget or timeout changed")
    cases_bytes = read_bytes(CASES_PATH)
    cases = validate_cases(json.loads(cases_bytes))
    return contract, cases, cases_bytes


class RecordingProvider:
    """Runner-local wrapper; the production provider protocol is unchanged."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.call_count = 0
        self.response: LLMResponse | None = None
        self.error: dict[str, Any] | None = None

    @property
    def provider(self) -> str:
        return str(getattr(self._inner, "provider", "unknown"))

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "unknown"))

    async def generate(self, *, messages, tools=None) -> LLMResponse:
        self.call_count += 1
        try:
            self.response = await self._inner.generate(messages=messages, tools=tools)
        except LLMProviderError as exc:
            self.error = describe_provider_error(exc)
            raise
        return self.response


def describe_provider_error(exc: BaseException) -> dict[str, Any]:
    cause = exc.__cause__ or exc
    body = getattr(cause, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    message = " ".join(
        str(part)
        for part in (cause, error.get("message") if isinstance(error, dict) else "")
        if part
    )
    return {
        "provider_error_type": type(cause).__name__,
        "provider_error_code": str(code) if code else None,
        "provider_error_status": getattr(cause, "status_code", None),
        "provider_error_mentions_token_limit": bool(_TOKEN_LIMIT_TEXT.search(message)),
    }


def is_structural_success(record: dict[str, Any]) -> bool:
    return (
        record["failure_category"] is None
        and record["observed_decision"] in ProtectedActionSemanticDecision.__members__
        and record["provider_call_count"] == 1
        and record["structured_tool_call_count"] == 1
        and record["finish_reason"] not in TOKEN_EXHAUSTION_FINISH_REASONS
    )


def classify(records: Sequence[dict[str, Any]]) -> str:
    """Predeclared precedence; semantic errors never imply token exhaustion."""
    if any(
        record["finish_reason"] in TOKEN_EXHAUSTION_FINISH_REASONS
        or record["provider_error_mentions_token_limit"]
        for record in records
    ):
        return TOKEN_BUDGET_AMENDMENT_REQUIRED
    if any(
        record["failure_category"] in STRUCTURED_OUTPUT_FAILURES
        or record["provider_error_code"] in STRUCTURED_OUTPUT_PROVIDER_ERROR_CODES
        for record in records
    ):
        return STRUCTURED_OUTPUT_POLICY_ISSUE
    if any(record["failure_category"] in PROVIDER_FAILURES for record in records):
        return INCONCLUSIVE_PROVIDER_FAILURE
    if len(records) == EXPECTED_CASE_COUNT and all(map(is_structural_success, records)):
        return KEEP_64
    return STRUCTURED_OUTPUT_POLICY_ISSUE


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def _summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": _percentile(ordered, 50),
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    explicit = [r for r in records if r["expected_decision"] == "EXPLICIT_CURRENT_ACTION"]
    boundary = [r for r in records if r["expected_decision"] != "EXPLICIT_CURRENT_ACTION"]
    latencies = [r["latency_ms"] for r in records]
    return {
        "result_status": classify(records),
        "case_count": len(records),
        "structural_success_count": sum(map(is_structural_success, records)),
        "total_provider_calls": sum(r["provider_call_count"] for r in records),
        "retries": 0,
        "failure_counts": dict(
            sorted(Counter(r["failure_category"] for r in records if r["failure_category"]).items())
        ),
        "finish_reason_counts": dict(
            sorted(Counter(str(r["finish_reason"]) for r in records).items())
        ),
        "semantic_diagnostics": {
            "role": "diagnostic_only_not_acceptance",
            "exact_correct_of_20": sum(
                r["observed_decision"] == r["expected_decision"] for r in records
            ),
            "explicit_correct_of_10": sum(
                r["observed_decision"] == r["expected_decision"] for r in explicit
            ),
            "boundary_exact_correct_of_10": sum(
                r["observed_decision"] == r["expected_decision"] for r in boundary
            ),
            "boundary_safe_non_explicit_of_10": sum(
                r["observed_decision"] in {"AMBIGUOUS_OR_INFORMATIONAL", "NOT_REQUESTED"}
                for r in boundary
            ),
            "decision_confusion_counts": dict(
                sorted(
                    Counter(
                        f"{r['expected_decision']}->{r['observed_decision'] or 'FAILURE'}"
                        for r in records
                    ).items()
                )
            ),
        },
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p90": _percentile(latencies, 90),
            "p95": _percentile(latencies, 95),
        },
        "tokens": {
            "prompt": _summary([r["prompt_tokens"] for r in records if r["prompt_tokens"] is not None]),
            "completion": _summary(
                [r["completion_tokens"] for r in records if r["completion_tokens"] is not None]
            ),
            "total": _summary([r["total_tokens"] for r in records if r["total_tokens"] is not None]),
        },
    }


async def run_case(
    verifier: LLMProtectedActionSemanticVerifier,
    inner_provider: LLMProvider,
    case: dict[str, Any],
) -> dict[str, Any]:
    recorder = RecordingProvider(inner_provider)
    verifier._llm = recorder  # runner-local instrumentation of the production verifier
    observed: str | None = None
    failure: str | None = None
    started = time.perf_counter()
    try:
        decision = await verifier.verify(
            user_text=case["customer_utterance"],
            proposed_action=case["proposed_action"],
        )
        observed = decision.value
    except ProtectedActionVerificationError as exc:
        failure = exc.category.value
    latency_ms = (time.perf_counter() - started) * 1000
    response = recorder.response
    error = recorder.error or {
        "provider_error_type": None,
        "provider_error_code": None,
        "provider_error_status": None,
        "provider_error_mentions_token_limit": False,
    }
    return {
        "case_id": case["case_id"],
        "proposed_action": case["proposed_action"],
        "expected_decision": case["expected_decision"],
        "observed_decision": observed,
        "failure_category": failure,
        "provider": recorder.provider,
        "model": response.model if response is not None else recorder.model,
        "finish_reason": response.finish_reason if response is not None else None,
        "provider_call_count": recorder.call_count,
        "structured_tool_call_count": len(response.tool_calls) if response is not None else 0,
        "prompt_tokens": response.usage.prompt_tokens if response is not None else None,
        "completion_tokens": response.usage.completion_tokens if response is not None else None,
        "total_tokens": response.usage.total_tokens if response is not None else None,
        "latency_ms": round(latency_ms, 3),
        **error,
    }


def build_payload(
    contract_bytes: bytes,
    cases_bytes: bytes,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": RESULTS_SCHEMA,
        "phase": "V2-C6 Routing Fallback Development Verifier Budget Validation",
        "contract": {"path": CONTRACT_RELATIVE_PATH, "sha256": sha256_bytes(contract_bytes)},
        "development_cases": {"path": CASES_RELATIVE_PATH, "sha256": sha256_bytes(cases_bytes)},
        "run_configuration": {
            "model": EXPECTED_MODEL,
            "max_completion_tokens": VERIFIER_MAX_COMPLETION_TOKENS,
            "timeout_seconds": VERIFIER_TIMEOUT_SECONDS,
            "automatic_retries": 0,
            "verifier_calls_per_case": 1,
            "execution": "sequential",
        },
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


def build_manifest(payload: dict[str, Any], result_bytes: bytes) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "results": {"path": RESULTS_RELATIVE_PATH, "sha256": sha256_bytes(result_bytes)},
        "runner": {
            "path": RUNNER_RELATIVE_PATH,
            "sha256": sha256_bytes(read_bytes(Path(__file__).resolve())),
        },
        "contract_sha256": payload["contract"]["sha256"],
        "development_cases_sha256": payload["development_cases"]["sha256"],
        "result_status": payload["aggregate"]["result_status"],
        "governance": payload["governance"],
    }


def assert_no_utterances(payload: dict[str, Any], cases: Sequence[dict[str, Any]]) -> None:
    serialized = json.dumps(payload)
    if any(case["customer_utterance"] in serialized for case in cases):
        raise ValueError("results must not contain customer utterances")


def run(
    *,
    settings: Settings | None = None,
    inner_provider_override: LLMProvider | None = None,
) -> dict[str, Any]:
    if RESULTS_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError("budget validation results exist; refusing to rerun")
    contract, cases, cases_bytes = load_and_validate()
    prohibited_evidence_audit(cases, cases_bytes)
    contract_bytes = read_bytes(CONTRACT_PATH)
    verifier = build_protected_action_verifier(settings or Settings())
    inner = verifier._llm
    if not isinstance(inner, GroqLLMProvider) or inner.max_completion_tokens != 64:
        raise ValueError("production verifier must use a 64-token Groq provider")
    if inner.model != EXPECTED_MODEL:
        raise ValueError(f"unexpected verifier model: {inner.model}")
    provider = inner_provider_override or inner

    async def execute() -> list[dict[str, Any]]:
        return [await run_case(verifier, provider, case) for case in cases]

    records = asyncio.run(execute())
    payload = build_payload(contract_bytes, cases_bytes, records)
    assert_no_utterances(payload, cases)
    result_bytes = stable_json_bytes(payload)
    write_create_once(RESULTS_PATH, result_bytes)
    write_create_once(MANIFEST_PATH, stable_json_bytes(build_manifest(payload, result_bytes)))
    del contract
    return payload["aggregate"]


def check_results() -> dict[str, Any]:
    """Revalidate persisted results; never calls a provider or writes."""
    contract, cases, cases_bytes = load_and_validate()
    contract_bytes = read_bytes(CONTRACT_PATH)
    result_bytes = read_bytes(RESULTS_PATH)
    payload = json.loads(result_bytes)
    manifest = read_json(MANIFEST_PATH)
    if payload.get("schema_version") != RESULTS_SCHEMA:
        raise ValueError("result schema changed")
    if stable_json_bytes(payload) != result_bytes:
        raise ValueError("results are not canonical stable JSON")
    if payload["contract"]["sha256"] != sha256_bytes(contract_bytes):
        raise ValueError("result contract lineage changed")
    if payload["development_cases"]["sha256"] != sha256_bytes(cases_bytes):
        raise ValueError("results were not produced from the frozen development set")
    records = payload["cases"]
    if [r["case_id"] for r in records] != [case["case_id"] for case in cases]:
        raise ValueError("result case IDs or order changed")
    if any(r["provider_call_count"] != 1 for r in records):
        raise ValueError("each case must have exactly one provider call and no retry")
    if payload["aggregate"] != aggregate(records):
        raise ValueError("recorded aggregate does not match per-case results")
    if payload["governance"] != GOVERNANCE:
        raise ValueError("result governance changed")
    assert_no_utterances(payload, cases)
    if manifest != build_manifest(payload, result_bytes):
        raise ValueError("result manifest lineage mismatch")
    del contract
    return {
        "results_valid": True,
        "result_status": payload["aggregate"]["result_status"],
        "structural_success_count": payload["aggregate"]["structural_success_count"],
        "total_provider_calls": payload["aggregate"]["total_provider_calls"],
        "semantic_diagnostics": payload["aggregate"]["semantic_diagnostics"],
        "provider_calls_performed": False,
        "files_written": False,
        "development_only": True,
        "final_acceptance_evidence": False,
    }


def preflight(settings_factory: Callable[[], Settings] = Settings) -> dict[str, Any]:
    """Validate everything for --run; no provider call and no writes."""
    contract, cases, cases_bytes = load_and_validate()
    audit = prohibited_evidence_audit(cases, cases_bytes)
    settings = settings_factory()
    key_configured = bool(
        settings.groq_api_key is not None and settings.groq_api_key.get_secret_value()
    )
    construction_verified = False
    if key_configured:
        verifier = build_protected_action_verifier(settings)
        inner = verifier._llm
        construction_verified = (
            isinstance(inner, GroqLLMProvider)
            and inner.max_completion_tokens == 64
            and inner.model == EXPECTED_MODEL
        )
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
        "development_cases_sha256": sha256_bytes(cases_bytes),
        "case_count": len(cases),
        "composition": {
            f"{action}:{kind}": count
            for (action, kind), count in sorted(
                Counter((case["proposed_action"], case["case_kind"]) for case in cases).items()
            )
        },
        "verifier_max_completion_tokens": VERIFIER_MAX_COMPLETION_TOKENS,
        "verifier_timeout_seconds": VERIFIER_TIMEOUT_SECONDS,
        "settings_llm_model": settings.llm_model,
        "expected_model": EXPECTED_MODEL,
        "groq_api_key_configured": key_configured,
        "production_construction_verified": construction_verified,
        "prohibited_evidence_audit": audit,
        "result_artifacts_present": results_present,
        "frozen_run_configuration": contract["frozen_run_configuration"],
        "provider_calls_performed": False,
        "files_written": False,
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
