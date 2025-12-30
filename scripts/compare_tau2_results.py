#!/usr/bin/env python3
"""Compare tau2-bench results between original implementation and MASEval.

This script compares results from two sources:
1. Original tau2-bench (from ~/Repositories/tau2-bench/data/simulations/)
2. MASEval implementation (from examples/tau2_benchmark/results/)

It auto-discovers result files based on model and domain, or you can provide
explicit paths. The script provides helpful feedback when files are missing
or contain insufficient data.

Usage:
    # Auto-discover files for a model/domain
    uv run python scripts/compare_tau2_results.py --model gpt-5-mini --domain retail

    # Use explicit file paths
    uv run python scripts/compare_tau2_results.py --original /path/to/original.json --maseval /path/to/maseval.jsonl

    # Require minimum number of tasks
    uv run python scripts/compare_tau2_results.py --model gpt-5-mini --domain retail --min-tasks 10
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Paths
DEFAULT_TAU2_BENCH_PATH = Path.home() / "Repositories" / "tau2-bench"
DEFAULT_MASEVAL_RESULTS_PATH = Path(__file__).parent.parent / "examples" / "tau2_benchmark" / "results"


@dataclass
class ResultFile:
    """Information about a result file."""

    path: Path
    exists: bool
    task_count: int
    implementation: str


@dataclass
class ParsedResults:
    """Parsed results from a benchmark run."""

    implementation: str
    task_results: List[Dict[str, Any]]
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    error_tasks: int
    success_rate: float
    mean_reward: float
    source_file: Path


def get_litellm_model_id(model_id: str) -> str:
    """Convert model ID to LiteLLM format if needed."""
    model_lower = model_id.lower()
    if any(x in model_lower for x in ["gemini", "palm", "bard"]) and not model_id.startswith("gemini/"):
        return f"gemini/{model_id}"
    return model_id


def find_original_results(model_id: str, domain: str, tau2_bench_path: Path) -> Optional[Path]:
    """Find existing tau2-bench results for the given model and domain."""
    sim_dir = tau2_bench_path / "data" / "simulations"
    if not sim_dir.exists():
        return None

    # Try different model name variants
    model_variants = [
        model_id,
        model_id.replace(".", "-"),
        model_id.replace("-", "."),
    ]

    for model in model_variants:
        pattern = f"*_{domain}_llm_agent_{model}_user_simulator_{model}.json"
        matches = list(sim_dir.glob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)

    return None


def find_maseval_results(model_id: str, domain: str, maseval_results_path: Path) -> Optional[Path]:
    """Find existing MASEval results for the given model and domain."""
    if not maseval_results_path.exists():
        return None

    # MASEval files: tau2_{domain}_default_{timestamp}.jsonl
    pattern = f"tau2_{domain}_default_*.jsonl"
    matches = list(maseval_results_path.glob(pattern))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)

    return None


def get_original_run_command(
    model_id: str,
    domain: str,
    tau2_bench_path: Path,
    num_tasks: Optional[int] = None,
    seed: int = 300,
    temperature: float = 0.0,
) -> str:
    """Generate the command to run the original tau2-bench."""
    litellm_model_id = get_litellm_model_id(model_id)

    cmd_parts = [
        "cd",
        str(tau2_bench_path),
        "&&",
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        domain,
        "--agent-llm",
        litellm_model_id,
        "--user-llm",
        litellm_model_id,
        "--agent-llm-args",
        f"'{json.dumps({'temperature': temperature})}'",
        "--user-llm-args",
        f"'{json.dumps({'temperature': temperature})}'",
        "--seed",
        str(seed),
        "--max-concurrency",
        "1",
    ]

    if num_tasks:
        cmd_parts.extend(["--num-tasks", str(num_tasks)])

    return " ".join(cmd_parts)


def get_maseval_run_command(
    model_id: str,
    domain: str,
    num_tasks: Optional[int] = None,
) -> str:
    """Generate the command to run MASEval."""
    cmd = f"uv run python examples/tau2_benchmark/tau2_benchmark.py --framework default --model {model_id} --domain {domain}"
    if num_tasks:
        cmd += f" --limit {num_tasks}"
    return cmd


def parse_original_results(file_path: Path) -> ParsedResults:
    """Parse results from original tau2-bench output."""
    with open(file_path) as f:
        data = json.load(f)

    simulations = data.get("simulations", [])
    task_results = []
    passed = 0
    failed = 0
    errors = 0

    for sim in simulations:
        reward_info = sim.get("reward_info", {})
        reward = reward_info.get("reward", 0.0) if reward_info else 0.0
        passed_task = reward == 1.0

        task_results.append(
            {
                "task_id": sim.get("task_id"),
                "passed": passed_task,
                "reward": reward,
                "termination_reason": sim.get("termination_reason"),
            }
        )

        if reward_info is None:
            errors += 1
        elif passed_task:
            passed += 1
        else:
            failed += 1

    total = len(simulations)
    evaluated = passed + failed

    return ParsedResults(
        implementation="original",
        task_results=task_results,
        total_tasks=total,
        passed_tasks=passed,
        failed_tasks=failed,
        error_tasks=errors,
        success_rate=passed / evaluated if evaluated > 0 else 0.0,
        mean_reward=sum(r["reward"] for r in task_results) / total if total > 0 else 0.0,
        source_file=file_path,
    )


def parse_maseval_results(file_path: Path) -> ParsedResults:
    """Parse results from MASEval output."""
    task_results = []
    passed = 0
    failed = 0
    errors = 0

    with open(file_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                eval_list = data.get("eval")
                task_id = data.get("task_id", "unknown")

                if eval_list is None or len(eval_list) == 0:
                    errors += 1
                    task_results.append({"task_id": task_id, "passed": None, "reward": None, "error": True})
                else:
                    eval_data = eval_list[0]
                    reward = eval_data.get("reward", 0.0)
                    passed_task = eval_data.get("passed", False)
                    task_results.append({"task_id": task_id, "passed": passed_task, "reward": reward, "error": False})
                    if passed_task:
                        passed += 1
                    else:
                        failed += 1

    total = len(task_results)
    evaluated = passed + failed

    return ParsedResults(
        implementation="maseval",
        task_results=task_results,
        total_tasks=total,
        passed_tasks=passed,
        failed_tasks=failed,
        error_tasks=errors,
        success_rate=passed / evaluated if evaluated > 0 else 0.0,
        mean_reward=sum(r["reward"] or 0 for r in task_results) / total if total > 0 else 0.0,
        source_file=file_path,
    )


def count_tasks_in_file(file_path: Path, is_original: bool) -> int:
    """Count the number of tasks in a result file."""
    try:
        if is_original:
            with open(file_path) as f:
                data = json.load(f)
            return len(data.get("simulations", []))
        else:
            count = 0
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
    except Exception:
        return 0


def aggregate_task_results(task_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate multiple runs per task_id into a single result."""
    grouped = defaultdict(list)
    for r in task_results:
        grouped[r["task_id"]].append(r)

    aggregated = {}
    for task_id, runs in grouped.items():
        valid_runs = [r for r in runs if not r.get("error")]

        if not valid_runs:
            aggregated[task_id] = {"task_id": task_id, "error": True, "n_runs": len(runs)}
        else:
            rewards = [r.get("reward", 0.0) or 0.0 for r in valid_runs]
            passed = [r.get("passed", False) for r in valid_runs]
            aggregated[task_id] = {
                "task_id": task_id,
                "mean_reward": sum(rewards) / len(rewards),
                "any_pass": any(passed),
                "pass_rate": sum(passed) / len(passed),
                "n_runs": len(runs),
                "n_valid_runs": len(valid_runs),
                "error": False,
            }

    return aggregated


def print_results_summary(results: ParsedResults, label: str) -> None:
    """Print a summary of results."""
    print(f"\n{label} RESULTS:")
    print(f"  Source:       {results.source_file}")
    print(f"  Total Tasks:  {results.total_tasks}")
    print(f"  Passed:       {results.passed_tasks}")
    print(f"  Failed:       {results.failed_tasks}")
    print(f"  Errors:       {results.error_tasks}")
    print(f"  Success Rate: {results.success_rate:.2%}")
    print(f"  Mean Reward:  {results.mean_reward:.4f}")


def compare_results(original: ParsedResults, maseval: ParsedResults) -> None:
    """Compare results from both implementations."""
    orig_tasks = aggregate_task_results(original.task_results)
    maseval_tasks = aggregate_task_results(maseval.task_results)

    # Check for task count mismatch
    orig_count = len(orig_tasks)
    maseval_count = len(maseval_tasks)

    if orig_count != maseval_count:
        print("\n" + "!" * 60)
        print("!" + " " * 58 + "!")
        print("!" + "  WARNING: TASK COUNT MISMATCH".center(58) + "!")
        print("!" + " " * 58 + "!")
        print("!" * 60)
        print(f"""
  Original tau2-bench: {orig_count} tasks
  MASEval:             {maseval_count} tasks

  The two result sets have different numbers of tasks!
  Only the INTERSECTION of tasks will be compared.
""")
        print("!" * 60)

    print("\n" + "=" * 60)
    print("AGREEMENT ANALYSIS")
    print("=" * 60)

    # Find common tasks (intersection)
    orig_task_ids = set(orig_tasks.keys())
    maseval_task_ids = set(maseval_tasks.keys())
    common_task_ids = orig_task_ids & maseval_task_ids

    # Report on non-overlapping tasks
    only_in_original = orig_task_ids - maseval_task_ids
    only_in_maseval = maseval_task_ids - orig_task_ids

    if only_in_original or only_in_maseval:
        print("\nTask ID overlap:")
        print(f"  Common to both:      {len(common_task_ids)}")
        if only_in_original:
            print(f"  Only in Original:    {len(only_in_original)} (excluded from comparison)")
        if only_in_maseval:
            print(f"  Only in MASEval:     {len(only_in_maseval)} (excluded from comparison)")

    # Build common tasks list
    common_tasks = []
    for task_id in sorted(common_task_ids, key=lambda x: (int(x) if x.isdigit() else float("inf"), x)):
        orig = orig_tasks[task_id]
        mas = maseval_tasks[task_id]
        if not orig.get("error") and not mas.get("error"):
            common_tasks.append((task_id, orig, mas))

    if not common_tasks:
        print("\nNo common completed tasks to compare!")
        print(f"  Original task IDs: {sorted(orig_tasks.keys())[:10]}...")
        print(f"  MASEval task IDs:  {sorted(maseval_tasks.keys())[:10]}...")
        return

    sample_task = common_tasks[0][1]
    has_repeats = sample_task.get("n_runs", 1) > 1

    print(f"\nComparing {len(common_tasks)} common tasks")

    # Extract rewards and pass rates
    orig_rewards = []
    mas_rewards = []
    orig_passed = []
    mas_passed = []

    for task_id, orig, mas in common_tasks:
        orig_rewards.append(orig.get("mean_reward", orig.get("reward", 0.0)) or 0.0)
        mas_rewards.append(mas.get("mean_reward", mas.get("reward", 0.0)) or 0.0)
        orig_passed.append(orig.get("pass_rate", 1.0 if orig.get("passed") else 0.0))
        mas_passed.append(mas.get("pass_rate", 1.0 if mas.get("passed") else 0.0))

    # Reward metrics
    print("\n--- Reward Agreement ---")
    mse = sum((o - m) ** 2 for o, m in zip(orig_rewards, mas_rewards)) / len(common_tasks)
    mae = sum(abs(o - m) for o, m in zip(orig_rewards, mas_rewards)) / len(common_tasks)
    rmse = mse**0.5
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    if len(common_tasks) > 1:
        mean_orig = sum(orig_rewards) / len(orig_rewards)
        mean_mas = sum(mas_rewards) / len(mas_rewards)
        numerator = sum((o - mean_orig) * (m - mean_mas) for o, m in zip(orig_rewards, mas_rewards))
        denom_orig = sum((o - mean_orig) ** 2 for o in orig_rewards) ** 0.5
        denom_mas = sum((m - mean_mas) ** 2 for m in mas_rewards) ** 0.5
        if denom_orig > 0 and denom_mas > 0:
            correlation = numerator / (denom_orig * denom_mas)
            print(f"  Pearson Correlation: {correlation:.4f}")

    # Confusion matrix
    print("\n--- Pass/Fail Confusion Matrix ---")
    if has_repeats:
        print("  (Using pass_rate >= 0.5 as threshold)")
    print("  (Original \\ MASEval)")

    orig_binary = [1 if p >= 0.5 else 0 for p in orig_passed]
    mas_binary = [1 if p >= 0.5 else 0 for p in mas_passed]

    tp = sum(1 for o, m in zip(orig_binary, mas_binary) if o == 1 and m == 1)
    tn = sum(1 for o, m in zip(orig_binary, mas_binary) if o == 0 and m == 0)
    fp = sum(1 for o, m in zip(orig_binary, mas_binary) if o == 0 and m == 1)
    fn = sum(1 for o, m in zip(orig_binary, mas_binary) if o == 1 and m == 0)

    print("                    MASEval")
    print("                  Pass    Fail")
    print(f"  Original Pass   {tp:4d}    {fn:4d}")
    print(f"  Original Fail   {fp:4d}    {tn:4d}")

    agreement = (tp + tn) / len(common_tasks) if common_tasks else 0
    print(f"\n  Agreement Rate: {agreement:.2%} ({tp + tn}/{len(common_tasks)})")

    # Cohen's Kappa
    p_orig_pass = sum(orig_binary) / len(orig_binary) if orig_binary else 0
    p_mas_pass = sum(mas_binary) / len(mas_binary) if mas_binary else 0
    p_expected = p_orig_pass * p_mas_pass + (1 - p_orig_pass) * (1 - p_mas_pass)
    if p_expected < 1:
        kappa = (agreement - p_expected) / (1 - p_expected)
        print(f"  Cohen's Kappa:  {kappa:.4f}")

    # Divergent tasks
    divergent_tasks = []
    for i, (task_id, orig, mas) in enumerate(common_tasks):
        if orig_binary[i] != mas_binary[i]:
            divergent_tasks.append(
                {
                    "task_id": task_id,
                    "original_pass_rate": orig_passed[i],
                    "maseval_pass_rate": mas_passed[i],
                    "reward_diff": mas_rewards[i] - orig_rewards[i],
                }
            )

    print("\n--- Divergent Tasks ---")
    if not divergent_tasks:
        print(f"  No divergent tasks! All {len(common_tasks)} tasks agree.")
    else:
        print(f"  {len(divergent_tasks)} tasks diverged:")
        print()
        print(f"  {'Task ID':<40} {'Original':<12} {'MASEval':<12} {'Reward Diff':<12}")
        print(f"  {'-' * 40} {'-' * 12} {'-' * 12} {'-' * 12}")
        for dt in divergent_tasks[:20]:  # Limit output
            orig_status = "PASS" if dt["original_pass_rate"] >= 0.5 else "FAIL"
            mas_status = "PASS" if dt["maseval_pass_rate"] >= 0.5 else "FAIL"
            print(f"  {dt['task_id']:<40} {orig_status:<12} {mas_status:<12} {dt['reward_diff']:+.4f}")
        if len(divergent_tasks) > 20:
            print(f"  ... and {len(divergent_tasks) - 20} more")

    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY:")
    if agreement >= 0.9:
        print(f"  High agreement ({agreement:.1%}) - implementations are consistent")
    elif agreement >= 0.7:
        print(f"  Moderate agreement ({agreement:.1%}) - some differences exist")
    else:
        print(f"  Low agreement ({agreement:.1%}) - significant differences")


def main():
    parser = argparse.ArgumentParser(
        description="Compare tau2-bench results between original and MASEval implementations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Auto-discovery options
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model ID for auto-discovery (e.g., gpt-5-mini)",
    )

    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        default="retail",
        choices=["airline", "retail", "telecom"],
        help="Domain for auto-discovery (default: retail)",
    )

    # Explicit file paths
    parser.add_argument(
        "--original",
        type=Path,
        help="Explicit path to original tau2-bench results file",
    )

    parser.add_argument(
        "--maseval",
        type=Path,
        help="Explicit path to MASEval results file",
    )

    # Configuration
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=1,
        help="Minimum number of tasks required in each file (default: 1)",
    )

    parser.add_argument(
        "--tau2-bench-path",
        type=Path,
        default=DEFAULT_TAU2_BENCH_PATH,
        help="Path to original tau2-bench repository",
    )

    parser.add_argument(
        "--maseval-results-path",
        type=Path,
        default=DEFAULT_MASEVAL_RESULTS_PATH,
        help="Path to MASEval results directory",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.original and not args.model:
        parser.error("Either --model (for auto-discovery) or --original (explicit path) is required")

    print("=" * 60)
    print("TAU2-BENCH RESULTS COMPARISON")
    print("=" * 60)

    # Resolve file paths
    original_path = args.original
    maseval_path = args.maseval

    if not original_path and args.model:
        original_path = find_original_results(args.model, args.domain, args.tau2_bench_path)

    if not maseval_path and args.model:
        maseval_path = find_maseval_results(args.model, args.domain, args.maseval_results_path)

    # Check what we have
    issues = []

    # Check original
    if original_path and original_path.exists():
        original_count = count_tasks_in_file(original_path, is_original=True)
        print(f"\nOriginal tau2-bench: {original_path}")
        print(f"  Tasks: {original_count}")

        if original_count < args.min_tasks:
            issues.append(("original", "insufficient_tasks", original_count))
    else:
        print("\nOriginal tau2-bench: NOT FOUND")
        issues.append(("original", "missing", 0))

    # Check MASEval
    if maseval_path and maseval_path.exists():
        maseval_count = count_tasks_in_file(maseval_path, is_original=False)
        print(f"\nMASEval results: {maseval_path}")
        print(f"  Tasks: {maseval_count}")

        if maseval_count < args.min_tasks:
            issues.append(("maseval", "insufficient_tasks", maseval_count))
    else:
        print("\nMASEval results: NOT FOUND")
        issues.append(("maseval", "missing", 0))

    # Handle issues
    if issues:
        print("\n" + "=" * 60)
        print("ACTION REQUIRED")
        print("=" * 60)

        for impl, issue_type, count in issues:
            if impl == "original":
                if issue_type == "missing":
                    print("\nOriginal tau2-bench results not found.")
                    print("To generate them, run:")
                    print()
                    print("-" * 60)
                    print(
                        get_original_run_command(
                            args.model or "MODEL",
                            args.domain,
                            args.tau2_bench_path,
                            num_tasks=args.min_tasks if args.min_tasks > 1 else None,
                        )
                    )
                    print("-" * 60)
                else:
                    print(f"\nOriginal tau2-bench has only {count} task(s), need at least {args.min_tasks}.")
                    print("To run with more tasks:")
                    print()
                    print("-" * 60)
                    print(
                        get_original_run_command(
                            args.model or "MODEL",
                            args.domain,
                            args.tau2_bench_path,
                            num_tasks=args.min_tasks,
                        )
                    )
                    print("-" * 60)

            elif impl == "maseval":
                if issue_type == "missing":
                    print("\nMASEval results not found.")
                    print("To generate them, run:")
                    print()
                    print("-" * 60)
                    print(
                        get_maseval_run_command(
                            args.model or "MODEL",
                            args.domain,
                            num_tasks=args.min_tasks if args.min_tasks > 1 else None,
                        )
                    )
                    print("-" * 60)
                else:
                    print(f"\nMASEval results have only {count} task(s), need at least {args.min_tasks}.")
                    print("To run with more tasks:")
                    print()
                    print("-" * 60)
                    print(
                        get_maseval_run_command(
                            args.model or "MODEL",
                            args.domain,
                            num_tasks=args.min_tasks,
                        )
                    )
                    print("-" * 60)

        # If only one is missing/insufficient, still show what we have
        can_compare = not any(issue[0] == "original" for issue in issues) and not any(issue[0] == "maseval" for issue in issues)
        if not can_compare:
            print("\nCannot compare until both result sets are available.")
            sys.exit(1)

    # Parse and compare
    print("\n" + "=" * 60)
    print("PARSING RESULTS")
    print("=" * 60)

    original_results = parse_original_results(original_path)  # type: ignore[arg-type]
    maseval_results = parse_maseval_results(maseval_path)  # type: ignore[arg-type]

    print_results_summary(original_results, "ORIGINAL")
    print_results_summary(maseval_results, "MASEVAL")

    compare_results(original_results, maseval_results)


if __name__ == "__main__":
    main()
