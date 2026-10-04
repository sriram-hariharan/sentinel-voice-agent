"""Run the deterministic V2-C6 protected-action runtime safety closeout."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE_PATH = (
    "data/evals/v2/ml/"
    "v2c6_routing_fallback_deterministic_runtime_safety_contract.json"
)
RESULT_RELATIVE_PATH = (
    "data/evals/v2/ml/"
    "v2c6_routing_fallback_deterministic_runtime_safety_result.json"
)
PROHIBITED_HOLDOUT_RELATIVE_PATH = "data/evals/v2/ml/v2c5_final_holdout.json"
CONTRACT_PATH = ROOT / CONTRACT_RELATIVE_PATH
RESULT_PATH = ROOT / RESULT_RELATIVE_PATH
PROHIBITED_HOLDOUT_PATH = ROOT / PROHIBITED_HOLDOUT_RELATIVE_PATH
EXPECTED_CONTRACT_SHA256 = (
    "50ec51a82f5b1eb76310779fd4bfb38e4eb08d260aa189873f6afc921b194675"
)
EXPECTED_INVARIANT_IDS = tuple("ABCDEFGHIJKL")
PASS_STATUS = "RUNTIME_SAFETY_PASS"
FAIL_STATUS = "RUNTIME_SAFETY_FAIL"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_bytes(path: Path) -> bytes:
    resolved = path.resolve()
    if resolved == PROHIBITED_HOLDOUT_PATH.resolve():
        raise PermissionError("the prohibited V2-C5 final holdout cannot be accessed")
    return resolved.read_bytes()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(read_bytes(path))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _test_exists(node_id: str) -> bool:
    path_text, separator, test_name = node_id.partition("::")
    if not separator or not path_text or not test_name:
        return False
    path = ROOT / path_text
    if not path.is_file():
        return False
    tree = ast.parse(read_bytes(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == test_name
        for node in tree.body
    )


def _validate_contract() -> dict[str, Any]:
    contract_bytes = read_bytes(CONTRACT_PATH)
    if sha256_bytes(contract_bytes) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("closeout contract hash mismatch")
    contract = json.loads(contract_bytes)
    if not isinstance(contract, dict):
        raise TypeError("closeout contract must be a JSON object")
    if contract.get("phase") != "V2-C6 Deterministic Runtime Safety Closeout":
        raise ValueError("unexpected closeout phase")
    if contract.get("semantic_evaluation_status") != "FRESH_SEMANTIC_PASS":
        raise ValueError("fresh semantic prerequisite is not frozen as passing")

    source_bindings = contract.get("source_bindings")
    if not isinstance(source_bindings, dict) or not source_bindings:
        raise ValueError("source bindings are required")
    for name, binding in source_bindings.items():
        if not isinstance(binding, dict):
            raise TypeError(f"source binding must be an object: {name}")
        path = ROOT / str(binding.get("path", ""))
        expected_hash = binding.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"source binding hash is invalid: {name}")
        if sha256_bytes(read_bytes(path)) != expected_hash:
            raise ValueError(f"source binding hash mismatch: {name}")

    fresh_result = load_json(
        ROOT / source_bindings["fresh_semantic_result"]["path"]
    )
    if fresh_result.get("aggregate", {}).get("result_status") != (
        "FRESH_SEMANTIC_PASS"
    ):
        raise ValueError("fresh semantic result prerequisite is not passing")
    if fresh_result.get("governance", {}).get("fresh_evaluation_consumed") is not True:
        raise ValueError("fresh semantic evidence is not marked consumed")

    invariants = contract.get("invariants")
    if not isinstance(invariants, list):
        raise TypeError("invariants must be a list")
    invariant_ids = tuple(row.get("id") for row in invariants)
    if invariant_ids != EXPECTED_INVARIANT_IDS:
        raise ValueError("closeout must declare every invariant A-L exactly once")
    for invariant in invariants:
        if invariant.get("coverage_after") != "SUFFICIENT":
            raise ValueError(f"invariant coverage is incomplete: {invariant['id']}")
        evidence = [
            *invariant.get("existing_test_evidence", []),
            *invariant.get("new_test_evidence", []),
        ]
        if not evidence:
            raise ValueError(f"invariant has no test evidence: {invariant['id']}")
        missing = [node_id for node_id in evidence if not _test_exists(node_id)]
        if missing:
            raise ValueError(
                f"invariant {invariant['id']} has missing tests: {missing}"
            )

    governance = contract.get("governance", {})
    if governance.get("requires_real_provider") is not False:
        raise ValueError("closeout must not require a real provider")
    if governance.get("runtime_configuration_change_authorized") is not False:
        raise ValueError("runtime configuration changes are not authorized")
    if governance.get("step29i_authorized") is not False:
        raise ValueError("Step 29I must remain blocked")
    if contract.get("final_holdout_policy", {}).get(
        "final_holdout_access_permitted"
    ) is not False:
        raise ValueError("final holdout access must remain prohibited")
    return contract


def preflight() -> dict[str, Any]:
    contract = _validate_contract()
    return {
        "status": "READY",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "invariant_count": len(contract["invariants"]),
        "semantic_evaluation_status": "FRESH_SEMANTIC_PASS",
        "provider_calls_performed": False,
        "files_written": False,
        "final_holdout_accessed": False,
        "step29i_authorized": False,
    }


def _run_invariant_tests(invariant: dict[str, Any]) -> dict[str, Any]:
    evidence = [
        *invariant["existing_test_evidence"],
        *invariant["new_test_evidence"],
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *evidence],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "id": invariant["id"],
        "name": invariant["name"],
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "test_evidence": evidence,
        "pytest_exit_code": completed.returncode,
    }


def run() -> dict[str, Any]:
    contract = _validate_contract()
    if RESULT_PATH.exists():
        raise FileExistsError("closeout result already exists; refusing to overwrite")

    invariant_results = [
        _run_invariant_tests(invariant) for invariant in contract["invariants"]
    ]
    passed = sum(row["status"] == "PASS" for row in invariant_results)
    failed = len(invariant_results) - passed
    final_status = PASS_STATUS if failed == 0 else FAIL_STATUS
    result = {
        "schema_version": (
            "v2c6-routing-fallback-deterministic-runtime-safety-result.v1"
        ),
        "phase": "V2-C6 Deterministic Runtime Safety Closeout",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH,
            "sha256": EXPECTED_CONTRACT_SHA256,
        },
        "semantic_prerequisite": "FRESH_SEMANTIC_PASS",
        "source_runtime_hashes": {
            name: binding["sha256"]
            for name, binding in contract["source_bindings"].items()
        },
        "invariants": invariant_results,
        "total_required_invariants": len(invariant_results),
        "passed_invariants": passed,
        "failed_invariants": failed,
        "final_status": final_status,
        "governance": {
            "provider_calls_performed": False,
            "fresh_semantic_evaluation_rerun": False,
            "runtime_changed": False,
            "final_holdout_accessed": False,
            "step29i_authorized": False,
        },
    }
    result_bytes = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    with RESULT_PATH.open("xb") as handle:
        handle.write(result_bytes)
    return {
        "final_status": final_status,
        "total_required_invariants": len(invariant_results),
        "passed_invariants": passed,
        "failed_invariants": failed,
        "provider_calls_performed": False,
        "files_written": True,
        "result_sha256": sha256_bytes(result_bytes),
        "final_holdout_accessed": False,
        "step29i_authorized": False,
    }


def check_results() -> dict[str, Any]:
    contract = _validate_contract()
    result = load_json(RESULT_PATH)
    if result.get("contract") != {
        "path": CONTRACT_RELATIVE_PATH,
        "sha256": EXPECTED_CONTRACT_SHA256,
    }:
        raise ValueError("result contract binding mismatch")
    if result.get("semantic_prerequisite") != "FRESH_SEMANTIC_PASS":
        raise ValueError("result semantic prerequisite mismatch")
    expected_hashes = {
        name: binding["sha256"]
        for name, binding in contract["source_bindings"].items()
    }
    if result.get("source_runtime_hashes") != expected_hashes:
        raise ValueError("result source/runtime hashes mismatch")
    invariants = result.get("invariants")
    if not isinstance(invariants, list):
        raise TypeError("result invariants must be a list")
    if tuple(row.get("id") for row in invariants) != EXPECTED_INVARIANT_IDS:
        raise ValueError("result invariant set mismatch")
    passed = sum(row.get("status") == "PASS" for row in invariants)
    failed = len(invariants) - passed
    recomputed_status = PASS_STATUS if failed == 0 else FAIL_STATUS
    if result.get("total_required_invariants") != len(EXPECTED_INVARIANT_IDS):
        raise ValueError("result required invariant count mismatch")
    if result.get("passed_invariants") != passed:
        raise ValueError("result passed invariant count mismatch")
    if result.get("failed_invariants") != failed:
        raise ValueError("result failed invariant count mismatch")
    if result.get("final_status") != recomputed_status:
        raise ValueError("result final status does not match invariant evidence")
    expected_governance = {
        "provider_calls_performed": False,
        "fresh_semantic_evaluation_rerun": False,
        "runtime_changed": False,
        "final_holdout_accessed": False,
        "step29i_authorized": False,
    }
    if result.get("governance") != expected_governance:
        raise ValueError("result governance mismatch")
    return {
        "results_valid": True,
        "final_status": recomputed_status,
        "total_required_invariants": len(invariants),
        "passed_invariants": passed,
        "failed_invariants": failed,
        "provider_calls_performed": False,
        "files_written": False,
        "final_holdout_accessed": False,
        "step29i_authorized": False,
        "result_sha256": sha256_bytes(read_bytes(RESULT_PATH)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--check-results", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        report = preflight()
    elif args.run:
        report = run()
    else:
        report = check_results()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
