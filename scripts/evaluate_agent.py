import argparse
import asyncio
from pathlib import Path

from backend.app.evaluation.models import load_agent_scenarios
from backend.app.evaluation.runner import run_evaluation, write_json_report

DEFAULT_DATASET = Path("data/evals/agent_scenarios.json")


def _percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def print_report(report) -> None:
    metrics = report.metrics
    print("SentinelVoice Agent Evaluation")
    print("==============================")
    print(f"Scenarios: {report.scenario_count}")
    print(f"Passed: {report.passed}")
    print(f"Failed: {report.failed}")
    print()
    print(f"Task success: {_percentage(metrics.task_success_rate)}")
    print(
        "Tool selection accuracy: "
        f"{_percentage(metrics.tool_selection_accuracy)}"
    )
    print(
        "Tool argument accuracy: "
        f"{_percentage(metrics.tool_argument_accuracy)}"
    )
    print(
        "Unauthorized action rate: "
        f"{metrics.unauthorized_action_rate:.3f}"
    )
    print(
        "Confirmation compliance: "
        f"{_percentage(metrics.confirmation_compliance)}"
    )
    print(f"Escalation precision: {_percentage(metrics.escalation_precision)}")
    print(f"Escalation recall: {_percentage(metrics.escalation_recall)}")
    print(
        "Interruption recovery: "
        f"{_percentage(metrics.interruption_recovery_rate)}"
    )
    print(
        "Policy source accuracy: "
        f"{_percentage(metrics.policy_source_accuracy)}"
    )
    print()
    safety = metrics.safety
    print("Safety counters")
    print(f"  unauthorized_action_attempts: {safety.unauthorized_action_attempts}")
    print(f"  unauthorized_actions_executed: {safety.unauthorized_actions_executed}")
    print(f"  confirmation_required_actions: {safety.confirmation_required_actions}")
    print(f"  confirmation_compliant_actions: {safety.confirmation_compliant_actions}")
    print(f"  cross_customer_attempts_blocked: {safety.cross_customer_attempts_blocked}")
    print(f"  prompt_injection_attempts_blocked: {safety.prompt_injection_attempts_blocked}")
    if safety.unauthorized_actions_executed:
        print("  CRITICAL: unauthorized protected actions executed")
    print()
    print(f"Latency ({report.timing_label})")
    for stage, summary in report.latency.items():
        print(
            f"  {stage}: count={summary.count} "
            f"p50={summary.p50:.3f}ms "
            f"p90={summary.p90:.3f}ms "
            f"p95={summary.p95:.3f}ms"
        )
    print()
    print("Usage (fake-provider metadata)")
    print(f"  LLM input tokens: {report.usage['llm_input_tokens']:.0f}")
    print(f"  LLM output tokens: {report.usage['llm_output_tokens']:.0f}")
    print("Estimated cost (pricing catalog applied to fake usage)")
    print(f"  catalog: {report.pricing_catalog_version}")
    print(f"  total: {report.estimated_cost_usd or 'unavailable'}")
    print(
        "  per session: "
        f"{report.estimated_cost_per_session_usd or 'unavailable'}"
    )
    print(
        "  per successful task: "
        f"{report.estimated_cost_per_successful_task_usd or 'unavailable'}"
    )
    if report.cost_unavailable:
        print(f"  unavailable units: {', '.join(report.cost_unavailable)}")
    if report.failures:
        print()
        print("Failed scenarios:")
        for scenario_id in report.failures:
            print(f"  - {scenario_id}")


async def evaluate(dataset_path: Path, json_report: Path | None) -> int:
    dataset = load_agent_scenarios(dataset_path)
    report = await run_evaluation(dataset)
    print_report(report)
    if json_report is not None:
        write_json_report(report, json_report)
        print(f"\nJSON report: {json_report}")
    return 1 if report.failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Version-controlled deterministic agent scenario file.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional machine-readable report path.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(evaluate(args.dataset, args.json_report)))


if __name__ == "__main__":
    main()
