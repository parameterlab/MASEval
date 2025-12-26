"""Tau 2 Benchmark with Default Agent Example.

This example demonstrates running the Tau 2 benchmark using MASEval's built-in
DefaultAgentTau2Benchmark, which implements the original tau2-bench agent logic
without requiring external agent frameworks (like smolagents or langgraph).

This is useful for:
- Reproducing original tau2-bench results for comparison
- Running benchmarks without additional dependencies
- Understanding how the tau2 agent works internally

The default agent mirrors the original tau2-bench LLMAgent:
- System prompt: <instructions>...</instructions><policy>...</policy>
- ReAct-style tool calling loop
- Supports any LLM via ModelAdapter

Reference:
    Paper: https://arxiv.org/abs/2506.07982
    Original: https://github.com/sierra-research/tau2-bench

Usage:
    # Run on retail domain with 5 tasks
    uv run python examples/tau2_benchmark/tau2_default_agent.py --domain retail --limit 5

    # Run on airline domain
    uv run python examples/tau2_benchmark/tau2_default_agent.py --domain airline --limit 5

    # Run with multiple repeats for Pass@k computation
    uv run python examples/tau2_benchmark/tau2_default_agent.py --domain telecom --limit 10 --repeats 4
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

# MASEval imports
from maseval import Task
from maseval.core.callbacks.result_logger import FileResultLogger

from maseval.benchmark.tau2 import (
    DefaultAgentTau2Benchmark,
    Tau2Environment,
    Tau2User,
    compute_benchmark_metrics,
    compute_pass_at_k,
    configure_model_ids,
    ensure_data_exists,
    load_tasks,
)

# Import a ModelAdapter - using Google GenAI adapter (has built-in tool calling support)
# You can substitute with OpenAI, Anthropic, LiteLLM, or any other ModelAdapter
from google.genai import Client as GoogleGenAIClient
from maseval.interface.inference import GoogleGenAIModelAdapter


# =============================================================================
# Model Setup
# =============================================================================

_google_client: Optional[GoogleGenAIClient] = None


def get_google_client() -> GoogleGenAIClient:
    """Get or create the shared Google GenAI client."""
    global _google_client
    if _google_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        _google_client = GoogleGenAIClient(api_key=api_key)
    return _google_client


# =============================================================================
# Default Agent User (with user tool support)
# =============================================================================


class DefaultTau2User(Tau2User):
    """Tau2 User that provides a simple callable for the default agent.

    The default agent uses Dict[str, Callable] for tools, so we provide
    a get_tool() that returns a callable which invokes simulate_response().
    """

    def get_tool(self) -> Dict[str, Any]:
        """Return user tool info for the default agent.

        Returns a dict with tool function that can be added to agent's tools.
        The agent can call this to ask the user questions mid-conversation.
        """

        def ask_user(question: str) -> str:
            """Ask the customer a question to clarify their request or get additional information.

            Args:
                question: The question to ask the customer.

            Returns:
                The customer's response.
            """
            return self.simulate_response(question)

        return {"ask_user": ask_user}


# =============================================================================
# Concrete Benchmark Implementation
# =============================================================================


class GoogleGenAITau2Benchmark(DefaultAgentTau2Benchmark):
    """Tau2 Benchmark using Google GenAI for the default agent.

    This is a complete, runnable implementation that:
    - Uses DefaultTau2Agent (mirrors original tau2-bench LLMAgent)
    - Uses Google GenAI for LLM calls
    - Supports user simulation with user tools

    You can create similar classes for other providers (OpenAI, Anthropic, etc.)
    by implementing get_model_adapter() with your preferred ModelAdapter.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        **kwargs: Any,
    ):
        """Initialize the benchmark.

        Args:
            model_id: Google GenAI model to use (default: gemini-2.5-flash)
            **kwargs: Additional arguments passed to parent
        """
        # Set model_id in agent_data
        agent_data = kwargs.pop("agent_data", {})
        agent_data["model_id"] = model_id
        super().__init__(agent_data=agent_data, **kwargs)
        self._model_id = model_id

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> GoogleGenAIModelAdapter:
        """Create a Google GenAI model adapter with tool calling support.

        Args:
            model_id: Model identifier
            **kwargs: Additional arguments (e.g., register_name for tracing)

        Returns:
            Configured GoogleGenAIModelAdapter
        """
        adapter = GoogleGenAIModelAdapter(get_google_client(), model_id=model_id)

        # Register for tracing if requested
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)

        return adapter

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
    ) -> DefaultTau2User:
        """Create user simulator with tool support for default agent.

        Args:
            agent_data: Agent configuration
            environment: Task environment
            task: Current task

        Returns:
            DefaultTau2User instance with callable user tool
        """
        user_data = task.user_data
        instructions = user_data.get("instructions", {})

        # Build scenario from instructions
        if isinstance(instructions, str):
            scenario = instructions
        elif isinstance(instructions, dict):
            parts = []
            if instructions.get("reason_for_call"):
                parts.append(f"Reason for call: {instructions['reason_for_call']}")
            if instructions.get("known_info"):
                parts.append(f"Known info: {instructions['known_info']}")
            if instructions.get("task_instructions"):
                parts.append(f"Task: {instructions['task_instructions']}")
            scenario = "\n".join(parts)
        else:
            scenario = ""

        # Add persona if available
        persona = user_data.get("persona")
        if persona:
            scenario = f"Persona: {persona}\n\n{scenario}"

        user_model_id = self._get_user_model_id(task)
        user_model = self.get_model_adapter(user_model_id, register_name="user_simulator")

        return DefaultTau2User(
            model=user_model,
            scenario=scenario,
            initial_query=task.query,
            tools=environment.create_user_tools(),
        )

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[DefaultTau2User],
    ):
        """Create the default agent with user tool support.

        Extends parent to add the user's ask_user tool to the agent's toolset.
        """
        # Get base agent setup
        agents_to_run, agents_dict = super().setup_agents(agent_data, environment, task, user)

        # Add user tool to agent if user is available
        if user is not None:
            agent = agents_dict["default_agent"]._agent
            user_tools = user.get_tool()
            agent.tools.update(user_tools)

        return agents_to_run, agents_dict


# =============================================================================
# Main Entry Point
# =============================================================================


def run_benchmark(
    domain: Literal["airline", "retail", "telecom"],
    model_id: str = "gemini-2.5-flash",
    limit: Optional[int] = None,
    n_task_repeats: int = 1,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the Tau2 benchmark with the default agent.

    Args:
        domain: Tau2 domain (airline, retail, or telecom)
        model_id: LLM model to use
        limit: Maximum number of tasks to run
        n_task_repeats: Number of times to repeat each task (for Pass@k)
        output_dir: Directory for results

    Returns:
        Summary metrics from the benchmark run
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure data is downloaded
    print("Ensuring Tau2 data is available...")
    ensure_data_exists(domain=domain)

    # Load tasks
    print(f"Loading {domain} domain tasks...")
    tasks = load_tasks(domain=domain, split="base", limit=limit)
    print(f"Loaded {len(tasks)} tasks")

    # Configure model IDs for user simulator and evaluator
    configure_model_ids(
        tasks,
        user_model_id=model_id,
        evaluator_model_id=model_id,
    )

    # Setup callback for logging results
    logger = FileResultLogger(
        output_dir=output_dir,
        filename_pattern=f"tau2_{domain}_default_{{timestamp}}.jsonl",
    )

    # Create benchmark
    benchmark = GoogleGenAITau2Benchmark(
        model_id=model_id,
        callbacks=[logger],
        n_task_repeats=n_task_repeats,
        fail_on_setup_error=True,
        fail_on_task_error=False,  # Continue on task errors
        fail_on_evaluation_error=True,
    )

    # Run benchmark
    print(f"\nRunning default agent benchmark on {domain} domain...")
    print(f"Model: {model_id}")
    results = benchmark.run(tasks=tasks)

    # Compute summary metrics
    summary = compute_benchmark_metrics(results)

    # Compute Pass@k if we have multiple repeats
    if n_task_repeats > 1:
        k_values = list(range(1, min(n_task_repeats + 1, 5)))
        pass_at_k = compute_pass_at_k(results, k_values=k_values)
        summary["pass_at_k"] = pass_at_k

    # Print summary
    print("\n" + "=" * 60)
    print("TAU2 BENCHMARK SUMMARY (Default Agent)")
    print("=" * 60)
    print(f"Domain: {domain}")
    print(f"Model: {model_id}")
    total = summary.get("total_tasks", len(results))
    print(f"Total Tasks: {total}")
    print(f"Success Rate: {summary.get('success_rate', 0):.2%}")
    print(f"Mean Reward: {summary.get('mean_reward', 0):.4f}")

    # Show Pass@k if computed
    if "pass_at_k" in summary:
        print("\nPass@k Metrics:")
        for k, score in summary["pass_at_k"].items():
            print(f"  {k}: {score:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)

    return summary


def main():
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run the Tau2 benchmark with MASEval's default agent implementation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on retail domain with 5 tasks
    uv run python examples/tau2_benchmark/tau2_default_agent.py --domain retail --limit 5

    # Run on airline domain with a different model
    uv run python examples/tau2_benchmark/tau2_default_agent.py --domain airline --limit 5 --model gemini-2.0-flash

    # Run with 4 repetitions per task for Pass@k
    uv run python examples/tau2_benchmark/tau2_default_agent.py --domain telecom --limit 10 --repeats 4
        """,
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["airline", "retail", "telecom"],
        help="Tau2 domain to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model ID to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to run (default: all)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each task for Pass@k (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    run_benchmark(
        domain=args.domain,
        model_id=args.model,
        limit=args.limit,
        n_task_repeats=args.repeats,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
