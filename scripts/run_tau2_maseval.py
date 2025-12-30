#!/usr/bin/env python3
"""Run the MASEval tau2-bench implementation.

This script runs the tau2 benchmark using MASEval's implementation and saves
results to a JSONL file for later comparison with the original tau2-bench.

Usage:
    # Run on retail domain with gpt-5-mini
    uv run python scripts/run_tau2_maseval.py --model gpt-5-mini --domain retail

    # Limit to 5 tasks for testing
    uv run python scripts/run_tau2_maseval.py --model gpt-5-mini --domain retail --limit 5

    # Run with multiple repeats per task
    uv run python scripts/run_tau2_maseval.py --model gemini-2.5-flash --domain airline --n-repeat 3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Literal, Optional

# Add maseval to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_provider_from_model(model_id: str) -> Literal["openai", "google", "anthropic"]:
    """Determine the provider from a model ID."""
    model_lower = model_id.lower()

    if any(x in model_lower for x in ["gpt-", "o1-", "o3-", "o4-", "chatgpt"]):
        return "openai"
    if any(x in model_lower for x in ["gemini", "palm", "bard"]):
        return "google"
    if any(x in model_lower for x in ["claude"]):
        return "anthropic"

    return "openai"


def run_maseval_benchmark(
    domain: str,
    model_id: str,
    limit: Optional[int],
    n_repeat: int,
    output_dir: Path,
) -> Optional[Path]:
    """Run the MASEval tau2-bench implementation.

    Returns:
        Path to the output file, or None if failed
    """
    provider = get_provider_from_model(model_id)

    if provider not in ("google", "openai"):
        print("Error: MASEval currently only supports 'google' and 'openai' providers.")
        print(f"Model '{model_id}' uses provider '{provider}'.")
        return None

    try:
        # Import the run_benchmark function from the example
        sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "tau2_benchmark"))
        from tau2_default_agent_benchmark import run_benchmark  # type: ignore[unresolved-import]

        print("Running MASEval benchmark...")
        print(f"  Domain: {domain}")
        print(f"  Model:  {model_id}")
        print(f"  Limit:  {limit or 'all'}")
        print(f"  Repeat: {n_repeat}")
        print()

        start_time = time.time()

        # Run the benchmark
        summary = run_benchmark(
            domain=domain,
            model_id=model_id,
            limit=limit,
            n_task_repeats=n_repeat,
            output_dir=output_dir,
        )

        duration = time.time() - start_time

        # Find the output file
        result_files = sorted(output_dir.glob(f"tau2_{domain}_default_*.jsonl"), reverse=True)
        if result_files:
            output_file = result_files[0]
            print()
            print("=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"  Success Rate: {summary.get('success_rate', 0):.2%}")
            print(f"  Mean Reward:  {summary.get('mean_reward', 0):.4f}")
            print(f"  Duration:     {duration:.1f}s")
            print(f"  Output File:  {output_file}")
            return output_file

        print("Warning: Output file not found after benchmark run")
        return None

    except Exception as e:
        print(f"Error running MASEval: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run tau2-bench using MASEval implementation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        required=True,
        help="Model ID to use (e.g., gpt-5-mini, gemini-2.5-flash)",
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
        help="Number of times to repeat each task (default: 1)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(__file__).parent.parent / "examples" / "tau2_benchmark" / "results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MASEVAL TAU2-BENCH RUNNER")
    print("=" * 60)

    output_file = run_maseval_benchmark(
        domain=args.domain,
        model_id=args.model,
        limit=args.limit,
        n_repeat=args.n_repeat,
        output_dir=args.output_dir,
    )

    if output_file:
        print()
        print("To compare with original tau2-bench, run:")
        print(f"  uv run python scripts/compare_tau2_results.py --model {args.model} --domain {args.domain}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
