#!/usr/bin/env python3
"""Compare tau2-bench results between original implementation and MASEval.

This script checks AGREEMENT between the two implementations by running the same
tasks on both and comparing their results. The goal is to verify that MASEval's
tau2 implementation produces consistent results with the original tau2-bench.

What is compared:
    - Success rate agreement: Do both implementations achieve similar pass rates?
    - Per-task agreement: For each task, did both implementations pass/fail?
    - Mean reward agreement: Are the reward scores consistent?

Note: This is NOT comparing which implementation performs better - it's checking
whether they produce the same results on the same tasks, which validates that
MASEval correctly implements the tau2 benchmark.

Supported models:
    - OpenAI: gpt-4.1, gpt-4o, gpt-5, gpt-5-mini, o1-*, o3-*
    - Google: gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro
    - Anthropic: claude-3-opus, claude-3-sonnet (original tau2-bench only)

Usage:
    # Run with GPT-4.1
    uv run python scripts/compare_tau2_implementations.py --model gpt-4.1 --limit 5

    # Run with Gemini (default)
    uv run python scripts/compare_tau2_implementations.py --model gemini-2.5-flash --limit 5

    # Run with different Gemini version
    uv run python scripts/compare_tau2_implementations.py --model gemini-2.5-pro --limit 5

    # Run specific domain
    uv run python scripts/compare_tau2_implementations.py --model gpt-4o --domain airline --limit 3

    # Run with multiple repeats per task for more robust estimates
    uv run python scripts/compare_tau2_implementations.py --model gemini-2.5-flash --limit 5 --n-repeat 3

    # Skip original tau2-bench (only run MASEval)
    uv run python scripts/compare_tau2_implementations.py --model gemini-2.5-flash --maseval-only --limit 5

    # Skip MASEval (only run original tau2-bench)
    uv run python scripts/compare_tau2_implementations.py --model gpt-4.1 --original-only --limit 5
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Add maseval to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ComparisonConfig:
    """Configuration for the comparison run."""

    domain: Literal["airline", "retail", "telecom"]
    model_id: str
    limit: Optional[int]
    n_repeat: int
    seed: int
    temperature: float
    output_dir: Path
    tau2_bench_path: Path
    maseval_only: bool
    original_only: bool


@dataclass
class RunResult:
    """Result from a single implementation run."""

    implementation: str
    success_rate: float
    mean_reward: float
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    error_tasks: int
    task_results: List[Dict[str, Any]]
    duration_seconds: float
    output_file: Optional[Path]


def get_provider_from_model(model_id: str) -> Literal["openai", "google", "anthropic"]:
    """Determine the provider from a model ID.

    Args:
        model_id: The model identifier (e.g., "gpt-4.1", "gemini-2.5-flash", "claude-3-opus")

    Returns:
        The provider name.
    """
    model_lower = model_id.lower()

    # OpenAI models
    if any(x in model_lower for x in ["gpt-", "o1-", "o3-", "chatgpt"]):
        return "openai"

    # Google models
    if any(x in model_lower for x in ["gemini", "palm", "bard"]):
        return "google"

    # Anthropic models
    if any(x in model_lower for x in ["claude"]):
        return "anthropic"

    # Default to OpenAI for unknown models (LiteLLM will handle routing)
    return "openai"


def get_litellm_model_id(model_id: str) -> str:
    """Convert model ID to LiteLLM format if needed.

    Args:
        model_id: The model identifier.

    Returns:
        LiteLLM-compatible model ID.
    """
    provider = get_provider_from_model(model_id)

    # Google models need gemini/ prefix for LiteLLM
    if provider == "google" and not model_id.startswith("gemini/"):
        return f"gemini/{model_id}"

    return model_id


def run_original_tau2_bench(config: ComparisonConfig) -> Optional[RunResult]:
    """Run the original tau2-bench implementation."""
    import time

    print("\n" + "=" * 60)
    print("RUNNING ORIGINAL TAU2-BENCH")
    print("=" * 60)

    # Check if tau2-bench exists
    if not config.tau2_bench_path.exists():
        print(f"Warning: tau2-bench not found at {config.tau2_bench_path}")
        return None

    # Build output filename - save to tau2-bench's data folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use a safe filename (replace special chars)
    safe_model_name = config.model_id.replace("/", "-").replace(".", "-")
    output_filename = f"comparison_{config.domain}_{safe_model_name}_{timestamp}"
    output_file = config.tau2_bench_path / "data" / "tau2" / "simulations" / f"{output_filename}.json"

    # Build command - convert model ID to LiteLLM format
    litellm_model_id = get_litellm_model_id(config.model_id)
    cmd = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        config.domain,
        "--agent-llm",
        litellm_model_id,
        "--user-llm",
        litellm_model_id,
        "--agent-llm-args",
        json.dumps({"temperature": config.temperature}),
        "--user-llm-args",
        json.dumps({"temperature": config.temperature}),
        "--seed",
        str(config.seed),
        "--save-to",
        output_filename,  # Just the filename, not full path
        "--max-concurrency",
        "1",  # Sequential for reproducibility
    ]

    if config.limit:
        cmd.extend(["--num-tasks", str(config.limit)])

    if config.n_repeat > 1:
        cmd.extend(["--num-trials", str(config.n_repeat)])

    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {config.tau2_bench_path}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=config.tau2_bench_path,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            print("Error running tau2-bench:")
            print(result.stderr)
            return None

        print(result.stdout)

        # Parse results - check the expected output location
        if output_file.exists():
            return parse_original_results(output_file, duration)
        else:
            # Try to find the file
            sim_dir = config.tau2_bench_path / "data" / "tau2" / "simulations"
            if sim_dir.exists():
                matching = list(sim_dir.glob(f"{output_filename}*.json"))
                if matching:
                    output_file = matching[0]
                    return parse_original_results(output_file, duration)
            print(f"Output file not found: {output_file}")
            return None

    except subprocess.TimeoutExpired:
        print("Timeout running tau2-bench")
        return None
    except Exception as e:
        print(f"Error running tau2-bench: {e}")
        return None


def parse_original_results(output_file: Path, duration: float) -> RunResult:
    """Parse results from original tau2-bench output."""
    with open(output_file) as f:
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
                "duration": sim.get("duration", 0),
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

    return RunResult(
        implementation="original",
        success_rate=passed / evaluated if evaluated > 0 else 0.0,
        mean_reward=sum(r["reward"] for r in task_results) / total if total > 0 else 0.0,
        total_tasks=total,
        passed_tasks=passed,
        failed_tasks=failed,
        error_tasks=errors,
        task_results=task_results,
        duration_seconds=duration,
        output_file=output_file,
    )


def run_maseval_tau2_bench(config: ComparisonConfig) -> Optional[RunResult]:
    """Run the MASEval tau2-bench implementation."""
    import time

    print("\n" + "=" * 60)
    print("RUNNING MASEVAL TAU2-BENCH")
    print("=" * 60)

    provider = get_provider_from_model(config.model_id)

    # Check for supported providers
    if provider not in ("google", "openai"):
        print("Warning: MASEval example currently only supports 'google' and 'openai' providers.")
        print(f"Model '{config.model_id}' uses provider '{provider}'.")
        print("Skipping MASEval. To add support, implement a benchmark class for this provider.")
        return None

    start_time = time.time()

    try:
        # Import the run_benchmark function from the example
        sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "tau2_benchmark"))
        from tau2_default_agent_benchmark import run_benchmark  # type: ignore[unresolved-import]

        # Setup output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = config.model_id.replace("/", "-").replace(".", "-")
        output_file = config.output_dir / f"maseval_{config.domain}_{safe_model_name}_{timestamp}.jsonl"

        print(f"Running MASEval benchmark on {config.domain} domain...")
        print(f"Model: {config.model_id}")
        print(f"Limit: {config.limit}")

        # Run the benchmark (returns summary metrics)
        summary = run_benchmark(
            domain=config.domain,
            model_id=config.model_id,
            limit=config.limit,
            n_task_repeats=config.n_repeat,
            output_dir=config.output_dir,
        )

        duration = time.time() - start_time

        # Find the actual output file (pattern: tau2_{domain}_default_{timestamp}.jsonl)
        result_files = sorted(config.output_dir.glob(f"tau2_{config.domain}_default_*.jsonl"), reverse=True)
        if result_files:
            output_file = result_files[0]

        # Parse results from file
        return parse_maseval_results_from_file(output_file, duration, summary)

    except Exception as e:
        print(f"Error running MASEval: {e}")
        import traceback

        traceback.print_exc()
        return None


def parse_maseval_results_from_file(output_file: Path, duration: float, summary: Dict[str, Any]) -> RunResult:
    """Parse results from MASEval output file."""
    task_results = []
    passed = 0
    failed = 0
    errors = 0

    if output_file.exists():
        with open(output_file) as f:
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

    return RunResult(
        implementation="maseval",
        success_rate=summary.get("success_rate", passed / evaluated if evaluated > 0 else 0.0),
        mean_reward=summary.get("mean_reward", sum(r["reward"] or 0 for r in task_results) / total if total > 0 else 0.0),
        total_tasks=total,
        passed_tasks=passed,
        failed_tasks=failed,
        error_tasks=errors,
        task_results=task_results,
        duration_seconds=duration,
        output_file=output_file,
    )


def aggregate_task_results(task_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate multiple runs per task_id into a single result.

    For n_repeat > 1, each task_id may have multiple entries.
    This function aggregates them by computing:
    - mean_reward: average reward across runs
    - any_pass: True if any run passed
    - pass_rate: fraction of runs that passed
    - n_runs: number of runs for this task

    Args:
        task_results: List of per-run results

    Returns:
        Dict mapping task_id to aggregated result
    """
    from collections import defaultdict

    # Group by task_id
    grouped = defaultdict(list)
    for r in task_results:
        grouped[r["task_id"]].append(r)

    # Aggregate
    aggregated = {}
    for task_id, runs in grouped.items():
        # Filter out error runs
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


def compare_results(original: Optional[RunResult], maseval: Optional[RunResult]) -> None:
    """Compare results from both implementations with comprehensive metrics."""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    if original is None and maseval is None:
        print("No results to compare!")
        return

    # Print individual results
    for result in [original, maseval]:
        if result is None:
            continue

        print(f"\n{result.implementation.upper()} RESULTS:")
        print(f"  Total Runs:     {result.total_tasks}")
        print(f"  Passed:         {result.passed_tasks}")
        print(f"  Failed:         {result.failed_tasks}")
        print(f"  Errors:         {result.error_tasks}")
        print(f"  Success Rate:   {result.success_rate:.2%}")
        print(f"  Mean Reward:    {result.mean_reward:.4f}")
        print(f"  Duration:       {result.duration_seconds:.1f}s")
        if result.output_file:
            print(f"  Output File:    {result.output_file}")

    # Compare if both available
    if original and maseval:
        print("\n" + "=" * 60)
        print("AGREEMENT ANALYSIS")
        print("=" * 60)

        # Aggregate multiple runs per task_id (for n_repeat > 1)
        orig_tasks = aggregate_task_results(original.task_results)
        maseval_tasks = aggregate_task_results(maseval.task_results)

        # Find common tasks (excluding errors)
        common_tasks = []
        for task_id in orig_tasks:
            if task_id in maseval_tasks:
                orig = orig_tasks[task_id]
                mas = maseval_tasks[task_id]
                # Only compare tasks that completed in both (no errors)
                if not orig.get("error") and not mas.get("error"):
                    common_tasks.append((task_id, orig, mas))

        if not common_tasks:
            print("\nNo common completed tasks to compare!")
            return

        # Check if we have multiple runs per task
        sample_task = common_tasks[0][1]
        has_repeats = sample_task.get("n_runs", 1) > 1

        unique_tasks = len(common_tasks)
        total_orig_runs = sum(orig.get("n_valid_runs", 1) for _, orig, _ in common_tasks)
        total_mas_runs = sum(mas.get("n_valid_runs", 1) for _, _, mas in common_tasks)

        print(f"\nUnique Comparable Tasks: {unique_tasks}")
        if has_repeats:
            print(f"  Original Runs: {total_orig_runs}")
            print(f"  MASEval Runs:  {total_mas_runs}")

        # Extract mean rewards and pass rates for comparison
        # For n_repeat=1, mean_reward=reward and pass_rate=passed
        # For n_repeat>1, these are aggregated values
        orig_rewards = []
        mas_rewards = []
        orig_passed = []
        mas_passed = []

        for task_id, orig, mas in common_tasks:
            # Use mean_reward if available (aggregated), else reward
            orig_rewards.append(orig.get("mean_reward", orig.get("reward", 0.0)) or 0.0)
            mas_rewards.append(mas.get("mean_reward", mas.get("reward", 0.0)) or 0.0)
            # Use pass_rate if available (aggregated), else passed as 0/1
            orig_passed.append(orig.get("pass_rate", 1.0 if orig.get("passed") else 0.0))
            mas_passed.append(mas.get("pass_rate", 1.0 if mas.get("passed") else 0.0))

        # --- Reward Metrics ---
        print("\n--- Reward Agreement ---")

        # MSE (Mean Squared Error)
        mse = sum((o - m) ** 2 for o, m in zip(orig_rewards, mas_rewards)) / len(common_tasks)
        print(f"  MSE:  {mse:.6f}")

        # MAE (Mean Absolute Error)
        mae = sum(abs(o - m) for o, m in zip(orig_rewards, mas_rewards)) / len(common_tasks)
        print(f"  MAE:  {mae:.6f}")

        # RMSE
        rmse = mse**0.5
        print(f"  RMSE: {rmse:.6f}")

        # Correlation (Pearson)
        if len(common_tasks) > 1:
            mean_orig = sum(orig_rewards) / len(orig_rewards)
            mean_mas = sum(mas_rewards) / len(mas_rewards)
            numerator = sum((o - mean_orig) * (m - mean_mas) for o, m in zip(orig_rewards, mas_rewards))
            denom_orig = sum((o - mean_orig) ** 2 for o in orig_rewards) ** 0.5
            denom_mas = sum((m - mean_mas) ** 2 for m in mas_rewards) ** 0.5
            if denom_orig > 0 and denom_mas > 0:
                correlation = numerator / (denom_orig * denom_mas)
                print(f"  Pearson Correlation: {correlation:.4f}")

        # --- Confusion Matrix ---
        print("\n--- Pass/Fail Confusion Matrix ---")
        if has_repeats:
            print("  (Using pass_rate >= 0.5 as threshold)")
        print("  (Original \\ MASEval)")

        # Binarize pass rates: >= 0.5 = pass, < 0.5 = fail
        orig_binary = [1 if p >= 0.5 else 0 for p in orig_passed]
        mas_binary = [1 if p >= 0.5 else 0 for p in mas_passed]

        # Calculate confusion matrix
        # TP: both pass, TN: both fail, FP: orig fail but mas pass, FN: orig pass but mas fail
        tp = sum(1 for o, m in zip(orig_binary, mas_binary) if o == 1 and m == 1)
        tn = sum(1 for o, m in zip(orig_binary, mas_binary) if o == 0 and m == 0)
        fp = sum(1 for o, m in zip(orig_binary, mas_binary) if o == 0 and m == 1)
        fn = sum(1 for o, m in zip(orig_binary, mas_binary) if o == 1 and m == 0)

        print("                    MASEval")
        print("                  Pass    Fail")
        print(f"  Original Pass   {tp:4d}    {fn:4d}")
        print(f"  Original Fail   {fp:4d}    {tn:4d}")

        # Agreement metrics
        agreement = (tp + tn) / len(common_tasks) if common_tasks else 0
        print(f"\n  Agreement Rate: {agreement:.2%} ({tp + tn}/{len(common_tasks)})")

        # Cohen's Kappa
        p_orig_pass = sum(orig_binary) / len(orig_binary) if orig_binary else 0
        p_mas_pass = sum(mas_binary) / len(mas_binary) if mas_binary else 0
        p_expected = p_orig_pass * p_mas_pass + (1 - p_orig_pass) * (1 - p_mas_pass)
        if p_expected < 1:
            kappa = (agreement - p_expected) / (1 - p_expected)
            print(f"  Cohen's Kappa:  {kappa:.4f}")

        # --- Divergent Tasks ---
        divergent_tasks = []
        for i, (task_id, orig, mas) in enumerate(common_tasks):
            # Compare binary pass/fail decisions
            if orig_binary[i] != mas_binary[i]:
                divergent_tasks.append(
                    {
                        "task_id": task_id,
                        "original_pass_rate": orig_passed[i],
                        "original_reward": orig_rewards[i],
                        "maseval_pass_rate": mas_passed[i],
                        "maseval_reward": mas_rewards[i],
                        "n_orig_runs": orig.get("n_valid_runs", 1),
                        "n_mas_runs": mas.get("n_valid_runs", 1),
                    }
                )

        print("\n--- Divergent Tasks ---")
        if not divergent_tasks:
            print(f"  No divergent tasks! All {len(common_tasks)} tasks agree.")
        else:
            print(f"  {len(divergent_tasks)} tasks diverged:")
            print()
            if has_repeats:
                print(f"  {'Task ID':<40} {'Orig Pass%':<12} {'MAS Pass%':<12} {'Reward Diff':<12}")
                print(f"  {'-' * 40} {'-' * 12} {'-' * 12} {'-' * 12}")
                for dt in divergent_tasks:
                    diff = dt["maseval_reward"] - dt["original_reward"]
                    print(
                        f"  {dt['task_id']:<40} {dt['original_pass_rate'] * 100:>5.1f}%     {dt['maseval_pass_rate'] * 100:>5.1f}%     {diff:+.4f}"
                    )
            else:
                print(f"  {'Task ID':<40} {'Original':<12} {'MASEval':<12} {'Reward Diff':<12}")
                print(f"  {'-' * 40} {'-' * 12} {'-' * 12} {'-' * 12}")
                for dt in divergent_tasks:
                    orig_status = "PASS" if dt["original_pass_rate"] >= 0.5 else "FAIL"
                    mas_status = "PASS" if dt["maseval_pass_rate"] >= 0.5 else "FAIL"
                    diff = dt["maseval_reward"] - dt["original_reward"]
                    print(f"  {dt['task_id']:<40} {orig_status:<12} {mas_status:<12} {diff:+.4f}")

        # --- Summary ---
        print("\n" + "-" * 40)
        print("SUMMARY:")
        if agreement >= 0.9:
            print(f"  ✓ High agreement ({agreement:.1%}) - implementations are consistent")
        elif agreement >= 0.7:
            print(f"  ~ Moderate agreement ({agreement:.1%}) - some differences exist")
        else:
            print(f"  ✗ Low agreement ({agreement:.1%}) - significant differences")


def main():
    parser = argparse.ArgumentParser(
        description="Compare tau2-bench results between original and MASEval implementations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with GPT-4.1
    uv run python scripts/compare_tau2_implementations.py --model gpt-4.1 --limit 5

    # Run with Gemini
    uv run python scripts/compare_tau2_implementations.py --model gemini-2.5-flash --limit 5

    # Run only original tau2-bench
    uv run python scripts/compare_tau2_implementations.py --model gpt-4o --limit 5 --original-only
""",
    )

    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        default="retail",
        choices=["airline", "retail", "telecom"],
        help="Domain to benchmark (default: retail)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-2.5-flash",
        help="Model ID to use (e.g., gpt-4.1, gemini-2.5-flash, claude-3-opus). Default: gemini-2.5-flash",
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of tasks (default: all)",
    )

    parser.add_argument(
        "--n-repeat",
        "-n",
        type=int,
        default=1,
        help="Number of times to repeat each task for more robust estimates (default: 1)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=300,
        help="Random seed for reproducibility (default: 300)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(__file__).parent.parent / "examples" / "tau2_benchmark" / "results",
        help="Output directory for results (default: examples/tau2_benchmark/results)",
    )

    parser.add_argument(
        "--tau2-bench-path",
        type=Path,
        default=Path.home() / "Repositories" / "tau2-bench",
        help="Path to original tau2-bench repository",
    )

    parser.add_argument(
        "--maseval-only",
        action="store_true",
        help="Only run MASEval implementation",
    )

    parser.add_argument(
        "--original-only",
        action="store_true",
        help="Only run original tau2-bench implementation",
    )

    args = parser.parse_args()

    # Determine provider from model
    provider = get_provider_from_model(args.model)

    # Create config
    config = ComparisonConfig(
        domain=args.domain,
        model_id=args.model,
        limit=args.limit,
        n_repeat=args.n_repeat,
        seed=args.seed,
        temperature=args.temperature,
        output_dir=args.output_dir,
        tau2_bench_path=args.tau2_bench_path,
        maseval_only=args.maseval_only,
        original_only=args.original_only,
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TAU2-BENCH IMPLEMENTATION COMPARISON")
    print("=" * 60)
    print(f"Domain:      {config.domain}")
    print(f"Model:       {config.model_id}")
    print(f"Provider:    {provider}")
    print(f"Limit:       {config.limit or 'all'}")
    print(f"N-Repeat:    {config.n_repeat}")
    print(f"Seed:        {config.seed}")
    print(f"Temperature: {config.temperature}")
    print(f"Output:      {config.output_dir}")

    # Run implementations
    original_result = None
    maseval_result = None

    if not config.maseval_only:
        original_result = run_original_tau2_bench(config)

    if not config.original_only:
        maseval_result = run_maseval_tau2_bench(config)

    # Compare results
    compare_results(original_result, maseval_result)


if __name__ == "__main__":
    main()
