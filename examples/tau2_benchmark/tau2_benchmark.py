"""Tau 2 Benchmark Example.

This example demonstrates running the Tau 2 benchmark with either
smolagents or langgraph frameworks.

The Tau 2 benchmark evaluates single-agent customer service tasks across three domains:
- airline: Flight reservation management (50 base tasks)
- retail: E-commerce order management (114 base tasks)
- telecom: Telecom customer service (114 base tasks)

Key Differences from MACS:
- Real tool implementations that modify database state (not LLM-simulated)
- Deterministic evaluation via database state comparison
- Single-agent tasks (customer service representative)
- Pass@k metrics recommended for evaluation

Reference:
    Paper: https://arxiv.org/abs/2506.07982
    Data: https://github.com/sierra-research/tau2-bench

Usage:
    # Run with smolagents on retail domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework smolagents --domain retail --limit 5

    # Run with langgraph on airline domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework langgraph --domain airline --limit 5

    # Run with multiple repeats for Pass@k computation
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework smolagents --domain telecom --repeats 4
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Third-party imports
from google.genai import Client as GoogleGenAIClient

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, FinalAnswerTool
from smolagents import Tool as SmolagentsTool

# langgraph imports
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict, Annotated

# MASEval imports
from maseval import AgentAdapter, Task, User
from maseval.core.callbacks.result_logger import FileResultLogger
from maseval.core.config import ConfigurableMixin
from maseval.core.tracing import TraceableMixin
from maseval.interface.agents.smolagents import SmolAgentAdapter
from maseval.interface.agents.langgraph import LangGraphAgentAdapter
from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

from maseval.benchmark.tau2 import (
    Tau2Benchmark,
    Tau2Environment,
    Tau2User,
    compute_benchmark_metrics,
    compute_pass_at_k,
    configure_model_ids,
    ensure_data_exists,
    load_tasks,
)


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


def create_model(
    model_id: str = "gemini-2.5-flash",
) -> GoogleGenAIModelAdapter:
    """Create a Google GenAI model adapter.

    Args:
        model_id: Model identifier (default: gemini-2.5-flash)

    Returns:
        Configured GoogleGenAIModelAdapter
    """
    return GoogleGenAIModelAdapter(get_google_client(), model_id=model_id)


# =============================================================================
# Tool Wrappers
# =============================================================================


class SmolagentsToolWrapper(SmolagentsTool, ConfigurableMixin, TraceableMixin):
    """Smolagents wrapper for Tau2 real tools."""

    skip_forward_signature_validation = True

    def __init__(self, name: str, func: callable, description: str, inputs: Dict[str, Any]):
        self._tool_func = func
        self.name = name
        self.description = description
        self.inputs = inputs
        self.output_type = "string"
        self._call_count = 0
        super().__init__()

    def forward(self, **kwargs) -> str:
        self._call_count += 1
        try:
            result = self._tool_func(**kwargs)
            # Convert Pydantic models to string representation
            if hasattr(result, "model_dump"):
                return str(result.model_dump())
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def gather_traces(self) -> Dict[str, Any]:
        return {"tool_name": self.name, "call_count": self._call_count}

    def gather_config(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description}


class SmolagentsTau2User(Tau2User):
    """Tau2 User with smolagents tool integration."""

    def get_tool(self):
        """Return a smolagents-compatible user input tool."""
        user = self

        class UserInputTool(SmolagentsTool):
            name = "user_input"
            description = "Ask the customer a question to clarify their request or get additional information."
            inputs = {"question": {"type": "string", "description": "The question to ask the customer."}}
            output_type = "string"

            def forward(self, question: str) -> str:
                return user.simulate_response(question)

        return UserInputTool()


# =============================================================================
# Smolagents Implementation
# =============================================================================


class SmolagentsTau2Benchmark(Tau2Benchmark):
    """Tau2 Benchmark implementation for smolagents."""

    def get_model_adapter(self, model_id: str, **kwargs):
        """Create a model adapter for the given model ID."""
        adapter = create_model(model_id=model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,  # type: ignore[override]
        task: Task,
    ) -> SmolagentsTau2User:
        """Create smolagents-compatible user simulator."""
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

        return SmolagentsTau2User(
            model=user_model,
            scenario=scenario,
            initial_query=task.query,
        )

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,  # type: ignore[override]
        task: Task,
        user: Optional[User],
    ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create smolagents customer service agent."""
        # Create smolagents model
        smol_model = OpenAIServerModel(
            model_id="gemini-2.5-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # Wrap real tools for smolagents
        tools: List[SmolagentsTool] = []
        for name, func in environment.tools.items():
            # Get tool metadata from toolkit
            tool_meta = environment.toolkit.get_tool_metadata(name)
            wrapper = SmolagentsToolWrapper(
                name=name,
                func=func,
                description=tool_meta.get("description", f"Execute {name}"),
                inputs=tool_meta.get("inputs", {}),
            )
            tools.append(wrapper)
            self.register("tools", name, wrapper)

        tools.append(FinalAnswerTool())

        # Add user tool if available
        if user and hasattr(user, "get_tool"):
            user_tool = user.get_tool()
            if user_tool:
                tools.append(user_tool)

        # Create agent with domain policy as system prompt
        system_prompt = f"""You are a customer service agent. Follow these policies:

{environment.policy}

Be helpful, accurate, and follow all policies strictly."""

        agent = ToolCallingAgent(
            model=smol_model,
            tools=tools,
            name="customer_service_agent",
            system_prompt=system_prompt,
            max_steps=25,
            verbosity_level=0,
        )

        adapter = SmolAgentAdapter(agent, name="customer_service_agent")
        return [adapter], {"customer_service_agent": adapter}


# =============================================================================
# LangGraph Implementation
# =============================================================================


def _create_langgraph_tool(name: str, func: callable, description: str) -> StructuredTool:
    """Create a LangGraph StructuredTool from a Tau2 tool."""

    def tool_func(**kwargs) -> str:
        try:
            result = func(**kwargs)
            if hasattr(result, "model_dump"):
                return str(result.model_dump())
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    tool_func.__name__ = name
    tool_func.__doc__ = description

    return StructuredTool.from_function(
        func=tool_func,
        name=name,
        description=description,
    )


class LangGraphTau2User(Tau2User):
    """Tau2 User with LangGraph tool integration."""

    def get_tool(self):
        """Return a LangGraph-compatible user input tool."""

        def user_input(question: str) -> str:
            """Ask the customer a question to clarify their request."""
            return self.simulate_response(question)

        return StructuredTool.from_function(
            func=user_input,
            name="user_input",
            description="Ask the customer a question to clarify requirements or get additional information.",
        )


class AgentState(TypedDict):
    """LangGraph agent state."""

    messages: Annotated[list, add_messages]


class LangGraphTau2Benchmark(Tau2Benchmark):
    """Tau2 Benchmark implementation for langgraph."""

    def get_model_adapter(self, model_id: str, **kwargs):
        """Create a model adapter for the given model ID."""
        adapter = create_model(model_id=model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,  # type: ignore[override]
        task: Task,
    ) -> LangGraphTau2User:
        """Create langgraph-compatible user simulator."""
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

        return LangGraphTau2User(
            model=user_model,
            scenario=scenario,
            initial_query=task.query,
        )

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,  # type: ignore[override]
        task: Task,
        user: Optional[User],
    ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create langgraph customer service agent."""
        # Create LangChain model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # Wrap real tools for LangGraph
        tools: List[StructuredTool] = []
        for name, func in environment.tools.items():
            tool_meta = environment.toolkit.get_tool_metadata(name)
            lg_tool = _create_langgraph_tool(
                name=name,
                func=func,
                description=tool_meta.get("description", f"Execute {name}"),
            )
            tools.append(lg_tool)

        # Add user tool if available
        if user and hasattr(user, "get_tool"):
            user_tool = user.get_tool()
            if user_tool:
                tools.append(user_tool)

        # Bind tools to model
        llm_with_tools = llm.bind_tools(tools)

        # Build graph
        system_prompt = f"""You are a customer service agent. Follow these policies:

{environment.policy}

Be helpful, accurate, and follow all policies strictly."""

        def call_agent(state: AgentState):
            messages = state["messages"]
            has_system = any(isinstance(m, SystemMessage) for m in messages)
            if not has_system:
                system_msg = SystemMessage(content=system_prompt)
                messages = [system_msg] + list(messages)
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("chatbot", call_agent)
        graph.add_node("tools", ToolNode(tools=tools))
        graph.add_conditional_edges("chatbot", tools_condition)
        graph.add_edge("tools", "chatbot")
        graph.set_entry_point("chatbot")

        compiled_graph = graph.compile()
        adapter = LangGraphAgentAdapter(compiled_graph, name="customer_service_agent")

        return [adapter], {"customer_service_agent": adapter}


# =============================================================================
# Main Entry Point
# =============================================================================


def get_benchmark_class(framework: Literal["smolagents", "langgraph"]) -> type:
    """Get the benchmark class for the specified framework."""
    if framework == "smolagents":
        return SmolagentsTau2Benchmark
    elif framework == "langgraph":
        return LangGraphTau2Benchmark
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def run_benchmark(
    framework: Literal["smolagents", "langgraph"],
    domain: Literal["airline", "retail", "telecom"],
    limit: Optional[int] = None,
    n_task_repeats: int = 1,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the Tau2 benchmark.

    Args:
        framework: Agent framework to use
        domain: Tau2 domain (airline, retail, or telecom)
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

    # Configure model IDs
    configure_model_ids(
        tasks,
        user_model_id="gemini-2.5-flash",
        evaluator_model_id="gemini-2.5-flash",
    )

    # Setup callback for logging results
    logger = FileResultLogger(
        output_dir=output_dir,
        filename_pattern=f"tau2_{domain}_{framework}_{{timestamp}}.jsonl",
    )

    # Get benchmark class and instantiate
    BenchmarkClass = get_benchmark_class(framework)
    benchmark = BenchmarkClass(
        agent_data={},
        callbacks=[logger],
        n_task_repeats=n_task_repeats,
        fail_on_setup_error=True,
        fail_on_task_error=False,  # Continue on task errors
        fail_on_evaluation_error=True,
    )

    # Run benchmark
    print(f"\nRunning {framework} benchmark on {domain} domain...")
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
    print("TAU2 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Framework: {framework}")
    print(f"Domain: {domain}")
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
        description="Run the Tau2 benchmark with smolagents or langgraph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with smolagents on retail domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework smolagents --domain retail --limit 5

    # Run with langgraph on airline domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework langgraph --domain airline --limit 5

    # Run with 4 repetitions per task for Pass@k
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework smolagents --domain telecom --repeats 4
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
        choices=["airline", "retail", "telecom"],
        help="Tau2 domain to evaluate",
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
        framework=args.framework,
        domain=args.domain,
        limit=args.limit,
        n_task_repeats=args.repeats,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
