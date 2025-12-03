"""MACS Benchmark Example.

This example demonstrates running the AWS Multi-Agent Collaboration Scenarios (MACS)
benchmark with either smolagents or langgraph frameworks.

The MACS benchmark evaluates multi-agent collaboration in three enterprise domains:
- Travel: 10 agents, 52 tools (flight booking, hotels, weather, etc.)
- Mortgage: 6 agents, 35 tools (loan processing, document handling, etc.)
- Software: 8 agents, 4 tools (code review, issue tracking, etc.)

Reference:
    Paper: https://arxiv.org/abs/2412.05449
    Data: https://github.com/aws-samples/multiagent-collab-scenario-benchmark

Usage:
    # Run with smolagents
    python examples/macs_benchmark.py --framework smolagents --domain travel --limit 5

    # Run with langgraph
    python examples/macs_benchmark.py --framework langgraph --domain travel --limit 5
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from google.genai import Client as GoogleGenAIClient

from maseval import AgentAdapter, Environment, Task, User
from maseval.core.callbacks.result_logger import FileResultLogger
from maseval.core.config import ConfigurableMixin
from maseval.core.tracing import TraceableMixin
from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

from maseval.benchmark.macs import (
    MACSBenchmark,
    MACSGenericTool,
    compute_benchmark_metrics,
    ensure_data_exists,
    load_agent_config,
    load_tasks,
)


# =============================================================================
# Model Setup
# =============================================================================


def create_model(model_id: str = "gemini-2.5-flash") -> GoogleGenAIModelAdapter:
    """Create a Google GenAI model adapter.

    Args:
        model_id: Model identifier (default: gemini-2.5-flash)

    Returns:
        Configured GoogleGenAIModelAdapter
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    client = GoogleGenAIClient(api_key=api_key)
    return GoogleGenAIModelAdapter(client, model_id=model_id)


# =============================================================================
# Smolagents Implementation
# =============================================================================


def _create_smolagents_benchmark():
    """Create smolagents-specific benchmark class."""
    from smolagents import Tool as SmolagentsTool, ToolCallingAgent, OpenAIServerModel
    from maseval.interface.agents.smolagents import SmolAgentAdapter

    class SmolagentsToolWrapper(SmolagentsTool, ConfigurableMixin, TraceableMixin):
        """Smolagents wrapper for MACSGenericTool."""

        skip_forward_signature_validation = True

        def __init__(self, generic_tool: MACSGenericTool):
            self.generic_tool = generic_tool
            self.name = generic_tool.name
            self.description = generic_tool.description
            self.inputs = generic_tool.inputs
            self.output_type = generic_tool.output_type
            super().__init__()

        def forward(self, **kwargs) -> str:
            return self.generic_tool(**kwargs)

        def gather_traces(self) -> Dict[str, Any]:
            return self.generic_tool.gather_traces()

        def gather_config(self) -> Dict[str, Any]:
            return self.generic_tool.gather_config()

    class SmolagentsMACSBenchmark(MACSBenchmark):
        """MACS Benchmark implementation for smolagents."""

        def setup_agents(
            self,
            agent_data: Dict[str, Any],
            environment: Environment,
            task: Task,
            user: Optional[User],
        ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
            """Create smolagents agents."""
            # Get tools from environment
            generic_tools = environment.create_tools()
            wrapped_tools = [SmolagentsToolWrapper(t) for t in generic_tools]

            # Create smolagents model
            # Use OpenAI-compatible API with Gemini via AI Studio
            smol_model = OpenAIServerModel(
                model_id="gemini-2.5-flash",
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.getenv("GOOGLE_API_KEY"),
            )

            # Get primary agent config
            primary_agent_id = agent_data.get("primary_agent_id", "supervisor")
            agents_config = agent_data.get("agents", [])
            primary_config = next(
                (a for a in agents_config if a.get("agent_id") == primary_agent_id), agents_config[0] if agents_config else {}
            )

            # Create agent
            agent = ToolCallingAgent(
                tools=list(wrapped_tools),  # type: ignore[arg-type]
                model=smol_model,
                max_steps=10,
                name=primary_config.get("agent_name", "MACS Agent"),
                description=primary_config.get("agent_instruction", "Multi-agent collaboration agent"),
            )

            # Wrap with adapter
            adapter = SmolAgentAdapter(agent, name=primary_agent_id)

            return [adapter], {primary_agent_id: adapter}

    return SmolagentsMACSBenchmark


# =============================================================================
# LangGraph Implementation
# =============================================================================


def _create_langgraph_benchmark():
    """Create langgraph-specific benchmark class."""
    from langchain_core.tools import StructuredTool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict, Annotated
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter

    class LanggraphToolWrapper(ConfigurableMixin, TraceableMixin):
        """LangGraph wrapper for MACSGenericTool."""

        def __init__(self, generic_tool: MACSGenericTool):
            self.generic_tool = generic_tool
            self.name = generic_tool.name
            self.tool = StructuredTool.from_function(
                func=generic_tool.__call__,
                name=generic_tool.name,
                description=generic_tool.description,
            )

        def __call__(self, *args, **kwargs):
            return self.tool(*args, **kwargs)

        def gather_traces(self) -> Dict[str, Any]:
            return self.generic_tool.gather_traces()

        def gather_config(self) -> Dict[str, Any]:
            return self.generic_tool.gather_config()

    class LanggraphMACSBenchmark(MACSBenchmark):
        """MACS Benchmark implementation for langgraph."""

        def setup_agents(
            self,
            agent_data: Dict[str, Any],
            environment: Environment,
            task: Task,
            user: Optional[User],
        ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
            """Create langgraph agents."""
            # Get tools from environment
            generic_tools = environment.create_tools()
            wrapped_tools = [LanggraphToolWrapper(t) for t in generic_tools]
            langchain_tools = [w.tool for w in wrapped_tools]

            # Create LangChain model with tools
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            llm_with_tools = llm.bind_tools(langchain_tools)

            # Define state
            class State(TypedDict):
                messages: Annotated[list, add_messages]

            # Build graph
            def chatbot(state: State):
                return {"messages": [llm_with_tools.invoke(state["messages"])]}

            graph = StateGraph(State)
            graph.add_node("chatbot", chatbot)
            graph.add_node("tools", ToolNode(tools=langchain_tools))

            graph.add_conditional_edges("chatbot", tools_condition)
            graph.add_edge("tools", "chatbot")
            graph.set_entry_point("chatbot")

            compiled_graph = graph.compile()

            # Get primary agent config
            primary_agent_id = agent_data.get("primary_agent_id", "supervisor")

            # Wrap with adapter
            adapter = LangGraphAgentAdapter(compiled_graph, name=primary_agent_id)

            return [adapter], {primary_agent_id: adapter}

    return LanggraphMACSBenchmark


# =============================================================================
# Main Entry Point
# =============================================================================


def get_benchmark_class(framework: Literal["smolagents", "langgraph"]) -> type:
    """Get the benchmark class for the specified framework.

    Args:
        framework: Either "smolagents" or "langgraph"

    Returns:
        The appropriate MACSBenchmark subclass
    """
    if framework == "smolagents":
        return _create_smolagents_benchmark()
    elif framework == "langgraph":
        return _create_langgraph_benchmark()
    else:
        raise ValueError(f"Unsupported framework: {framework}. Choose 'smolagents' or 'langgraph'.")


def run_benchmark(
    framework: Literal["smolagents", "langgraph"],
    domain: Literal["travel", "mortgage", "software"],
    limit: Optional[int] = None,
    n_task_repeats: int = 1,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the MACS benchmark.

    Args:
        framework: Agent framework to use
        domain: MACS domain (travel, mortgage, or software)
        limit: Maximum number of tasks to run (None for all)
        n_task_repeats: Number of times to repeat each task
        output_dir: Directory for results (default: examples/results/)

    Returns:
        Summary metrics from the benchmark run
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure data is downloaded
    print("Ensuring MACS data is available...")
    ensure_data_exists(verbose=1)

    # Create model for tool simulation and evaluation
    model = create_model("gemini-2.5-flash")

    # Load data
    print(f"Loading {domain} domain tasks...")
    tasks = load_tasks(domain, limit=limit)
    agent_config = load_agent_config(domain)
    print(f"Loaded {len(tasks)} tasks")

    # Setup callback for logging results
    logger = FileResultLogger(
        output_dir=output_dir,
        filename_pattern=f"{domain}_{framework}_{{timestamp}}.jsonl",
    )

    # Get benchmark class and instantiate
    BenchmarkClass = get_benchmark_class(framework)
    benchmark = BenchmarkClass(
        agent_data=agent_config,
        model=model,
        callbacks=[logger],
        n_task_repeats=n_task_repeats,
    )

    # Run benchmark
    print(f"\nRunning {framework} benchmark on {domain} domain...")
    results = benchmark.run(tasks=tasks)

    # Compute summary metrics
    summary = compute_benchmark_metrics(results)

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Framework: {framework}")
    print(f"Domain: {domain}")
    print(f"Total Tasks: {summary['total_tasks']}")
    print(f"Successful Tasks (Overall GSR=1.0): {summary['successful_tasks']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")

    print("\nMean Metrics:")
    for metric, value in summary["mean_metrics"].items():
        print(f"  {metric:<25} {value:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 50)

    return summary


def main():
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run the MACS benchmark with smolagents or langgraph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with smolagents on travel domain
    python examples/macs_benchmark.py --framework smolagents --domain travel

    # Run with langgraph on mortgage domain, limited to 5 tasks
    python examples/macs_benchmark.py --framework langgraph --domain mortgage --limit 5

    # Run with 3 repetitions per task
    python examples/macs_benchmark.py --framework smolagents --domain software --repeats 3
        """,
    )

    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=["smolagents", "langgraph"],
        help="Agent framework to use",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["travel", "mortgage", "software"],
        help="MACS domain to evaluate",
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
        help="Number of times to repeat each task (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: examples/results/)",
    )

    args = parser.parse_args()

    run_benchmark(
        framework=args.framework,
        domain=args.domain,
        limit=args.limit,
        n_task_repeats=args.repeats,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
