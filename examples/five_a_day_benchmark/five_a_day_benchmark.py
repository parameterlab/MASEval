"""
5-A-Day Benchmark - Framework-Agnostic Agent Evaluation

Educational example showcasing the maseval library through 5 diverse evaluation tasks.
Demonstrates framework-agnostic tools, multiple evaluation types, and single vs multi-agent patterns.

Key Features:
- Framework-agnostic BaseTools with conversion methods for smolagents, langgraph, llamaindex
- 7 evaluation types: unit tests, LLM judges, optimization, pattern matching, static analysis
- Environment loaded from tasks.json with automatic tool instantiation
- Both single-agent and orchestrator+specialist multi-agent configurations

Tasks:
0. Email & Banking: Verify payment and draft email (financial accuracy, privacy)
1. Finance Calculation: Calculate stock inheritance split (arithmetic, tool validation)
2. Code Generation: Implement DP algorithm (unit tests, complexity analysis, quality)
3. Calendar Scheduling: Find meeting overlaps using MCP (slot matching, logic validation)
4. Hotel Optimization: Select best hotel by criteria (ranking, search strategy, reasoning)
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

from utils import derive_seed, sanitize_name

from maseval import Benchmark, Environment, Evaluator, Task, TaskCollection, AgentAdapter, MessageHistory
from maseval.core.callbacks.result_logger import FileResultLogger

# Import tool implementations
from tools import (
    EmailToolCollection,
    BankingToolCollection,
    CalculatorToolCollection,
    CodeExecutionToolCollection,
    FamilyInfoToolCollection,
    StockPriceToolCollection,
    CalendarToolCollection,
    RunningAppToolCollection,
    GymTrackerToolCollection,
    HotelSearchToolCollection,
    MCPCalendarToolCollection,
)

# Import all evaluators dynamically
import evaluators

# ============================================================================
# Parse command-line arguments
# ============================================================================
parser = argparse.ArgumentParser(
    description="5-A-Day Benchmark - Framework-Agnostic Agent Evaluation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--framework", type=str, default="smolagents", choices=["smolagents", "langgraph", "llamaindex"], help="Agent framework to use"
)
parser.add_argument(
    "--config-type", type=str, default="single", choices=["single", "multi"], help="Agent configuration type (single-agent or multi-agent)"
)
parser.add_argument("--limit", type=int, default=None, help="Number of tasks to run (default: all tasks)")
parser.add_argument("--task", type=int, default=None, help="Run only a specific task by index (default: all tasks)")
parser.add_argument("--model-id", type=str, default="gemini-2.5-flash", help="Model identifier to use")
parser.add_argument("--temperature", type=float, default=0.7, help="Model temperature")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None for non-deterministic)")


# ============================================================================
# Global Model Factory
# ============================================================================


def get_model(model_id: str, framework: str, temperature: float = 0.7, seed: Optional[int] = None) -> Any:
    """Get a model instance for the specified framework.

    Args:
        model_id: Model identifier (e.g., 'gemini-2.5-flash')
        framework: Target framework ('smolagents', 'langgraph', 'llamaindex')
        temperature: Model temperature (default: 0.7)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Framework-specific model instance
    """
    if framework == "smolagents":
        from smolagents import LiteLLMModel

        return LiteLLMModel(
            model_id=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            seed=seed,
        )

    elif framework == "langgraph":
        from langchain_litellm import ChatLiteLLM

        return ChatLiteLLM(
            model=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            seed=seed,
        )

    elif framework == "llamaindex":
        from llama_index.llms.litellm import LiteLLM

        return LiteLLM(
            model=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            seed=seed,
        )

    else:
        raise ValueError(f"Unsupported framework: {framework}")


# ============================================================================
# Environment
# ============================================================================


class FiveADayEnvironment(Environment):
    """Environment that creates tools from task data and converts them to framework-specific types.

    Loads tool specifications from tasks.json, instantiates the appropriate tool classes
    with task-specific data, and converts them to the target framework (smolagents, langgraph, llamaindex).
    """

    def __init__(self, task_data: Dict[str, Any], framework: str = "smolagents", callbacks=None):
        """Initialize environment with framework specification.

        Args:
            task_data: Task data containing environment_data with tool specifications
            framework: Target framework ('smolagents', 'langgraph', 'llamaindex')
            callbacks: Optional callbacks
        """
        self.framework = framework
        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize environment state from task data."""
        return task_data["environment_data"]

    def _get_mcp_calendar_data(self, calendar_key: str) -> Dict[str, Any]:
        """Convert availability format to MCP events format.

        Task data: {"2025-12-02": [{"start": "09:00", "end": "11:00"}]}
        MCP format: {"events": [{"date": "2025-12-02", "start_time": "09:00", ...}]}
        """
        availability = self.state[calendar_key]

        events = [
            {
                "date": date,
                "start_time": slot["start"],
                "end_time": slot["end"],
                "title": "Busy",
            }
            for date, slots in availability.items()
            for slot in slots
        ]
        return {"events": events}

    def create_tools(self) -> list:
        """Create tool instances from environment_data and convert to framework-specific types.

        The base Environment class stores tools in self._tools_dict for tracing.

        Returns:
            List of framework-specific tool objects (smolagents Tool, LangChain StructuredTool, etc.)
        """
        tools_list = []

        # Map tool names to tool collection classes and their initialization data
        tool_mapping = {
            "email": (EmailToolCollection, lambda: (self.state["email_inbox"],)),
            "banking": (
                BankingToolCollection,
                lambda: (
                    self.state["banking"]["bank_transactions"],
                    self.state["banking"]["current_balance"],
                    self.state["banking"]["assets"],
                ),
            ),
            "calculator": (CalculatorToolCollection, lambda: ()),
            "python_executor": (CodeExecutionToolCollection, lambda: (self.state["test_cases"],)),
            "family_info": (FamilyInfoToolCollection, lambda: (self.state["family_info"],)),
            "stock_price": (StockPriceToolCollection, lambda: (self.state["stock_price_lookup"],)),
            "websearch": (StockPriceToolCollection, lambda: (self.state["stock_price_lookup"],)),
            "calendar": (CalendarToolCollection, lambda: ("my_calendar", self.state["my_calendar_availability"])),
            "running_app": (RunningAppToolCollection, lambda: (self.state["running_activities"],)),
            "gym_tracker": (GymTrackerToolCollection, lambda: (self.state["gym_activities"],)),
            "hotel_search": (HotelSearchToolCollection, lambda: (self.state["hotels"],)),
            "my_calendar_mcp": (
                MCPCalendarToolCollection,
                lambda: ("my_calendar_mcp", self._get_mcp_calendar_data("my_calendar_mcp_availability")),
            ),
            "other_calendar_mcp": (
                MCPCalendarToolCollection,
                lambda: ("other_calendar_mcp", self._get_mcp_calendar_data("other_calendar_mcp_availability")),
            ),
        }

        for tool_name in self.state["tools"]:
            if tool_name in tool_mapping:
                ToolClass, get_init_args = tool_mapping[tool_name]
                init_args = get_init_args()

                assert isinstance(init_args, tuple), f"Tool {tool_name} init_args must be a tuple"
                tool_instance = ToolClass(*init_args)

                # Check if it's a collection with get_sub_tools method
                if hasattr(tool_instance, "get_sub_tools"):
                    # Get all sub-tools from the collection
                    base_tools = tool_instance.get_sub_tools()
                else:
                    # Legacy single tool
                    base_tools = [tool_instance]

                # Convert each base tool to framework-specific tool
                for base_tool in base_tools:
                    framework_tool = self._convert_tool(base_tool)
                    tools_list.append(framework_tool)

        return tools_list

    def _convert_tool(self, base_tool):
        """Convert BaseTool to framework-specific tool adapter.

        Args:
            base_tool: BaseTool instance

        Returns:
            Tool adapter with gather_traces() for tracing support.
            For smolagents: SmolagentsToolAdapter (has .tool attribute for raw Tool)
            For langgraph: LangGraphToolAdapter (has .tool attribute for StructuredTool)
            For llamaindex: LlamaIndexToolAdapter (has .tool attribute for FunctionTool)
        """
        if self.framework == "smolagents":
            return base_tool.to_smolagents()
        elif self.framework == "langgraph":
            return base_tool.to_langgraph()
        elif self.framework == "llamaindex":
            return base_tool.to_llamaindex()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")


# ============================================================================
# Framework-Specific Agent Builders
# ============================================================================


def build_smolagents_single_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: List[Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    filter_tools_fn,
) -> Any:
    """Build a single smolagents agent.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters
        primary_spec: Primary agent specification
        specialist_specs: Empty list for single-agent (ignored)
        filter_tools_fn: Function to filter tools by name

    Returns:
        SmolAgentAdapter wrapping the created agent
    """
    from smolagents import ToolCallingAgent
    from maseval.interface.agents.smolagents import SmolAgentAdapter

    seed = primary_spec.get("seed")
    model = get_model(model_id, "smolagents", temperature, seed)
    tool_adapters = filter_tools_fn(all_tool_adapters, primary_spec["tools"])
    tools = [adapter.tool for adapter in tool_adapters]
    sanitized_name = sanitize_name(primary_spec["agent_name"])

    agent = ToolCallingAgent(
        model=model,
        tools=tools,
        name=sanitized_name,
        instructions=primary_spec["agent_instruction"],
        verbosity_level=2,
    )

    return SmolAgentAdapter(agent, primary_spec["agent_id"])


def build_langgraph_single_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: List[Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    filter_tools_fn,
) -> Any:
    """Build a single langgraph agent.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters
        primary_spec: Primary agent specification
        specialist_specs: Empty list for single-agent (ignored)
        filter_tools_fn: Function to filter tools by name

    Returns:
        LangGraphAgentAdapter wrapping the created graph
    """
    from langchain_core.messages import SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict, Annotated
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter

    seed = primary_spec.get("seed")
    model = get_model(model_id, "langgraph", temperature, seed)
    tool_adapters = filter_tools_fn(all_tool_adapters, primary_spec["tools"])
    tools = [adapter.tool for adapter in tool_adapters]

    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]

    llm_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState):
        messages = state["messages"]
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system and primary_spec["agent_instruction"]:
            system_message = SystemMessage(content=primary_spec["agent_instruction"])
            messages = [system_message] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    graph = workflow.compile()
    return LangGraphAgentAdapter(graph, primary_spec["agent_id"])


def build_llamaindex_single_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: List[Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    filter_tools_fn,
) -> Any:
    """Build a single llamaindex agent.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters
        primary_spec: Primary agent specification
        specialist_specs: Empty list for single-agent (ignored)
        filter_tools_fn: Function to filter tools by name

    Returns:
        LlamaIndexAgentAdapter wrapping the created agent
    """
    from llama_index.core.agent.workflow.react_agent import ReActAgent
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter

    seed = primary_spec.get("seed")
    model = get_model(model_id, "llamaindex", temperature, seed)
    tool_adapters = filter_tools_fn(all_tool_adapters, primary_spec["tools"])
    tools = [adapter.tool for adapter in tool_adapters]

    agent = ReActAgent(
        tools=tools,
        llm=model,
        verbose=True,
        max_iterations=10,
    )

    if hasattr(agent, "update_prompts") and primary_spec["agent_instruction"]:
        try:
            agent.update_prompts({"agent_worker:system_prompt": primary_spec["agent_instruction"]})
        except Exception:
            pass

    return LlamaIndexAgentAdapter(agent, primary_spec["agent_id"])


def build_smolagents_multi_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: List[Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    filter_tools_fn,
) -> Any:
    """Build smolagents multi-agent setup with orchestrator and specialists.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters
        primary_spec: Primary agent specification
        specialist_specs: List of specialist agent specifications
        filter_tools_fn: Function to filter tools by name

    Returns:
        SmolAgentAdapter wrapping the orchestrator agent
    """
    from smolagents import ToolCallingAgent, FinalAnswerTool
    from maseval.interface.agents.smolagents import SmolAgentAdapter

    specialist_agents = []
    for agent_spec in specialist_specs:
        specialist_seed = agent_spec.get("seed")
        specialist_model = get_model(model_id, "smolagents", temperature, specialist_seed)
        specialist_adapters = filter_tools_fn(all_tool_adapters, agent_spec["tools"])
        specialist_tools = [adapter.tool for adapter in specialist_adapters]
        specialist_tools.append(FinalAnswerTool())
        sanitized_name = sanitize_name(agent_spec["agent_name"])

        specialist = ToolCallingAgent(
            model=specialist_model,
            tools=specialist_tools,
            name=sanitized_name,
            description=agent_spec["agent_instruction"],
            instructions=agent_spec["agent_instruction"],
            verbosity_level=2,
        )
        specialist_agents.append(specialist)

    primary_adapters = filter_tools_fn(all_tool_adapters, primary_spec["tools"])
    primary_tools = [adapter.tool for adapter in primary_adapters]
    primary_tools.append(FinalAnswerTool())
    sanitized_primary_name = sanitize_name(primary_spec["agent_name"])
    primary_seed = primary_spec.get("seed")
    primary_model = get_model(model_id, "smolagents", temperature, primary_seed)

    agent = ToolCallingAgent(
        model=primary_model,
        tools=primary_tools,
        managed_agents=specialist_agents if specialist_agents else None,
        name=sanitized_primary_name,
        instructions=primary_spec["agent_instruction"],
        verbosity_level=2,
    )

    return SmolAgentAdapter(agent, primary_spec["agent_id"])


def build_langgraph_multi_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: List[Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    filter_tools_fn,
) -> Any:
    """Build langgraph multi-agent setup with orchestrator and specialists.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters
        primary_spec: Primary agent specification
        specialist_specs: List of specialist agent specifications
        filter_tools_fn: Function to filter tools by name

    Returns:
        LangGraphAgentAdapter wrapping the multi-agent graph
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.tools import tool as create_tool
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from typing_extensions import TypedDict, Annotated
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter

    class MultiAgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]

    # Create specialist subgraphs
    specialist_subgraphs = {}
    for agent_spec in specialist_specs:
        agent_id = agent_spec["agent_id"]
        agent_instruction = agent_spec["agent_instruction"]
        specialist_seed = agent_spec.get("seed")
        specialist_model = get_model(model_id, "langgraph", temperature, specialist_seed)
        specialist_adapters = filter_tools_fn(all_tool_adapters, agent_spec["tools"])
        specialist_tools = [adapter.tool for adapter in specialist_adapters]

        def make_specialist_node(spec_instruction, spec_tools, spec_model):
            def specialist_node(state: MultiAgentState):
                messages = state["messages"]

                # Extract delegated query from orchestrator's tool call
                delegated_query = None
                if messages and isinstance(messages[-1], AIMessage):
                    last_msg = messages[-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        tool_call = last_msg.tool_calls[0]
                        if "args" in tool_call and "query" in tool_call["args"]:
                            delegated_query = tool_call["args"]["query"]

                if not delegated_query:
                    for msg in reversed(messages):
                        if hasattr(msg, "type") and msg.type == "human":
                            delegated_query = msg.content
                            break

                if not delegated_query:
                    delegated_query = "Please help with the task."

                specialist_messages = []
                if spec_instruction:
                    specialist_messages.append(SystemMessage(content=spec_instruction))
                specialist_messages.append(HumanMessage(content=delegated_query))

                if spec_tools:
                    specialist_model_with_tools = spec_model.bind_tools(spec_tools)
                    current_messages = specialist_messages

                    for _ in range(5):
                        response = specialist_model_with_tools.invoke(current_messages)
                        current_messages = current_messages + [response]

                        if hasattr(response, "tool_calls") and response.tool_calls:
                            tool_node = ToolNode(spec_tools)
                            tool_results = tool_node.invoke({"messages": [response]})
                            current_messages = current_messages + tool_results["messages"]
                        else:
                            return {"messages": [response]}

                    return {"messages": [current_messages[-1]]}
                else:
                    response = spec_model.invoke(specialist_messages)
                    return {"messages": [response]}

            return specialist_node

        specialist_subgraphs[agent_id] = make_specialist_node(agent_instruction, specialist_tools, specialist_model)

    # Create handoff tools
    handoff_tools = []
    for agent_spec in specialist_specs:
        agent_id = agent_spec["agent_id"]
        agent_name = sanitize_name(agent_spec["agent_name"])
        agent_description = agent_spec["agent_instruction"]

        def make_handoff_tool(spec_id, spec_name, spec_description):
            @create_tool
            def handoff_tool(query: str) -> str:
                """Delegate query to specialist."""
                return f"Delegating to {spec_name}: {query}"

            handoff_tool.name = f"ask_{spec_id}"
            handoff_tool.description = f"Delegate to {spec_name}: {spec_description}"
            handoff_tool._target_agent = spec_id  # type: ignore[attr-defined]
            return handoff_tool

        handoff_tools.append(make_handoff_tool(agent_id, agent_name, agent_description))

    # Create orchestrator
    primary_instruction = primary_spec["agent_instruction"]
    primary_seed = primary_spec.get("seed")
    primary_model = get_model(model_id, "langgraph", temperature, primary_seed)
    orchestrator_model = primary_model.bind_tools(handoff_tools)

    def orchestrator_node(state: MultiAgentState):
        messages = state["messages"]
        if primary_instruction and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=primary_instruction)] + messages
        response = orchestrator_model.invoke(messages)
        return {"messages": [response]}

    def route_after_orchestrator(state: MultiAgentState):
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            tool_name = tool_call["name"]
            if tool_name.startswith("ask_"):
                specialist_id = tool_name[4:]
                return specialist_id

        return END

    # Build graph
    workflow = StateGraph(MultiAgentState)
    workflow.add_node("orchestrator", orchestrator_node)

    for agent_id, node_fn in specialist_subgraphs.items():
        workflow.add_node(agent_id, node_fn)

    workflow.set_entry_point("orchestrator")

    specialist_routes = {agent_id: agent_id for agent_id in specialist_subgraphs.keys()}
    specialist_routes[END] = END
    workflow.add_conditional_edges("orchestrator", route_after_orchestrator, specialist_routes)

    for agent_id in specialist_subgraphs.keys():
        workflow.add_edge(agent_id, "orchestrator")

    graph = workflow.compile()
    return LangGraphAgentAdapter(graph, primary_spec["agent_id"])


def build_llamaindex_multi_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: List[Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    filter_tools_fn,
) -> Any:
    """Build llamaindex multi-agent setup with orchestrator and specialists.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters
        primary_spec: Primary agent specification
        specialist_specs: List of specialist agent specifications
        filter_tools_fn: Function to filter tools by name

    Returns:
        LlamaIndexAgentAdapter wrapping the orchestrator agent
    """
    from llama_index.core.agent.workflow.react_agent import ReActAgent
    from llama_index.core.tools import FunctionTool
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    import asyncio

    specialist_agents_dict = {}
    for agent_spec in specialist_specs:
        agent_id = agent_spec["agent_id"]
        agent_name = sanitize_name(agent_spec["agent_name"])
        agent_instruction = agent_spec["agent_instruction"]
        specialist_seed = agent_spec.get("seed")
        specialist_model = get_model(model_id, "llamaindex", temperature, specialist_seed)
        specialist_adapters = filter_tools_fn(all_tool_adapters, agent_spec["tools"])
        specialist_tools = [adapter.tool for adapter in specialist_adapters]

        specialist_agent = ReActAgent(
            tools=specialist_tools,
            llm=specialist_model,
            verbose=True,
            max_iterations=10,
        )

        if hasattr(specialist_agent, "update_prompts") and agent_instruction:
            try:
                specialist_agent.update_prompts({"agent_worker:system_prompt": agent_instruction})
            except Exception:
                pass

        specialist_agents_dict[agent_id] = {
            "agent": specialist_agent,
            "name": agent_name,
            "description": agent_instruction,
        }

    def make_handoff_tool(specialist_id: str, specialist_info: dict):
        def handoff_to_specialist(task: str) -> str:
            """Delegate task to specialist agent."""
            specialist_agent = specialist_info["agent"]

            async def run_specialist():
                handler = specialist_agent.run(user_msg=task)
                result = await handler
                return result

            result = asyncio.run(run_specialist())
            if hasattr(result, "response"):
                return str(result.response)
            return str(result)

        return FunctionTool.from_defaults(
            fn=handoff_to_specialist,
            name=f"ask_{specialist_id}",
            description=f"Delegate to {specialist_info['name']}: {specialist_info['description']}",
        )

    orchestrator_tools = [make_handoff_tool(spec_id, spec_info) for spec_id, spec_info in specialist_agents_dict.items()]
    primary_adapters = filter_tools_fn(all_tool_adapters, primary_spec["tools"])
    primary_tools = [adapter.tool for adapter in primary_adapters]
    orchestrator_tools.extend(primary_tools)

    primary_seed = primary_spec.get("seed")
    primary_model = get_model(model_id, "llamaindex", temperature, primary_seed)

    orchestrator = ReActAgent(
        tools=orchestrator_tools,
        llm=primary_model,
        verbose=True,
        max_iterations=15,
    )

    primary_instruction = primary_spec["agent_instruction"]
    if hasattr(orchestrator, "update_prompts") and primary_instruction:
        try:
            orchestrator.update_prompts({"agent_worker:system_prompt": primary_instruction})
        except Exception:
            pass

    return LlamaIndexAgentAdapter(orchestrator, primary_spec["agent_id"])


def get_agent_builder(framework: str, agent_type: str):
    """Get the appropriate agent builder function based on framework and agent type."""
    if agent_type == "single":
        if framework == "smolagents":
            return build_smolagents_single_agent
        elif framework == "langgraph":
            return build_langgraph_single_agent
        elif framework == "llamaindex":
            return build_llamaindex_single_agent
    elif agent_type == "multi":
        if framework == "smolagents":
            return build_smolagents_multi_agent
        elif framework == "langgraph":
            return build_langgraph_multi_agent
        elif framework == "llamaindex":
            return build_llamaindex_multi_agent

    raise ValueError(f"Unsupported combination of framework '{framework}' and agent_type '{agent_type}'")


# ============================================================================
# Benchmark
# ============================================================================


class FiveADayBenchmark(Benchmark):
    """5-A-Day benchmark with framework-agnostic tools.

    Demonstrates the maseval library through 5 diverse tasks with different evaluation approaches.
    Supports single-agent and multi-agent (orchestrator+specialist) configurations.
    """

    @staticmethod
    def _filter_tool_adapters(adapters: List[Any], tool_names: List[str]) -> List[Any]:
        """Filter tool adapters by exact name or collection prefix.

        Supports two matching modes (both used):
        1. Exact match: "calculator" → adapter.name="calculator"
        2. Collection prefix: "banking" → adapter.name="banking.get_balance"

        Args:
            adapters: List of tool adapters to filter
            tool_names: Tool/collection names to match (e.g., ["banking", "calculator"])

        Returns:
            Filtered list of adapters matching the specified names
        """
        if not tool_names:
            return []

        filtered = []
        for adapter in adapters:
            if not hasattr(adapter, "name"):
                continue

            for tool_name in tool_names:
                if adapter.name == tool_name or adapter.name.startswith(f"{tool_name}."):
                    filtered.append(adapter)
                    break

        return filtered

    def setup_environment(self, agent_data: Dict[str, Any], task: Task) -> Environment:
        """Create environment from task data."""
        # Pass full task data to environment
        task_data = {
            "environment_data": task.environment_data,
            "query": task.query,
            "evaluation_data": task.evaluation_data,
            "metadata": task.metadata,
        }

        environment = FiveADayEnvironment(task_data, framework=agent_data["framework"])

        # Register all tools with the benchmark for tracing
        for tool_adapter in environment.get_tools():
            tool_name = getattr(tool_adapter, "name", None) or str(type(tool_adapter).__name__)
            self.register("tools", tool_name, tool_adapter)

        return environment

    def setup_agents(
        self, agent_data: Dict[str, Any], environment: Environment, task: Task, user=None
    ) -> tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create framework-specific agent with tools from environment."""
        framework = agent_data["framework"]
        agent_type = agent_data["agent_type"]
        model_id = agent_data["model_config"]["model_id"]
        temperature = agent_data["model_config"]["temperature"]
        primary_agent_id = agent_data["primary_agent_id"]
        agents_specs = agent_data["agents"]
        all_tool_adapters = environment.get_tools()

        # Extract primary and specialist specs
        primary_spec = next(a for a in agents_specs if a["agent_id"] == primary_agent_id)
        specialist_specs = [a for a in agents_specs if a["agent_id"] != primary_agent_id]

        # Build agent using unified interface
        builder = get_agent_builder(framework, agent_type)
        agent_adapter = builder(model_id, temperature, all_tool_adapters, primary_spec, specialist_specs, self._filter_tool_adapters)

        # Use .visualize() when smolagents # TODO remove
        if framework == "smolagents":
            agent_adapter.agent.visualize()

        return [agent_adapter], {primary_agent_id: agent_adapter}

    def setup_evaluators(self, environment, task, agents, user) -> Sequence[Evaluator]:
        """Create evaluators based on task's evaluation_data.evaluators list."""
        if not task.evaluation_data["evaluators"]:
            return []

        # Dynamically instantiate evaluators
        evaluator_instances = []
        for name in task.evaluation_data["evaluators"]:
            # Get evaluator class from evaluators module
            evaluator_class = getattr(evaluators, name)

            # Instantiate with standard arguments
            evaluator_instances.append(evaluator_class(task, environment, user))

        return evaluator_instances

    def run_agents(self, agents: Sequence[AgentAdapter], task: Task, environment: Environment) -> Sequence[Any]:
        """Execute agents and return their final answers."""
        answers = [agent.run(task.query) for agent in agents]
        return answers

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate agent performance."""
        results = []

        for evaluator in evaluators:
            # ensure the evaluator is only called with relevant traces
            filtered_traces = evaluator.filter_traces(traces)
            results.append(evaluator(filtered_traces))

        return results


# ============================================================================
# Data Loading
# ============================================================================


def load_benchmark_data(
    config_type: str,
    framework: str,
    model_id: str,
    temperature: float,
    limit: Optional[int] = None,
    specific_task: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[TaskCollection, List[Dict[str, Any]]]:
    """Load tasks and agent configurations with validation.

    Args:
        config_type: Agent configuration type ('single' or 'multi')
        framework: Target framework ('smolagents', 'langgraph', 'llamaindex')
        limit: Optional limit on number of tasks/configs to load
        specific_task: Optional index to load only a specific task
        model_id: Model identifier
        temperature: Model temperature
        seed: Base random seed for reproducibility (None for non-deterministic)

    Returns:
        Tuple of (TaskCollection, agent_configs_list)
    """
    if limit is not None and specific_task is not None:
        raise ValueError("Cannot specify both limit and specific_task")

    data_dir = Path(__file__).parent / "data"

    # Load raw data from JSON files
    with open(data_dir / "tasks.json", "r") as f:
        tasks_raw = json.load(f)
    with open(data_dir / f"{config_type}agent.json", "r") as f:
        configs_raw = json.load(f)

    # Validate alignment
    if len(tasks_raw) != len(configs_raw):
        raise ValueError(f"Mismatch: {len(tasks_raw)} tasks vs {len(configs_raw)} configs")

    # Determine indices to load
    if specific_task is not None:
        indices = [specific_task]
    elif limit is not None:
        indices = list(range(limit))
    else:
        indices = list(range(len(tasks_raw)))

    # Build tasks and configs in parallel
    tasks_data = []
    configs_data = []

    for idx in indices:
        task_dict = tasks_raw[idx]
        config = configs_raw[idx]
        task_id = task_dict["metadata"]["task_id"]

        # Create task
        tasks_data.append(
            Task(
                query=task_dict["query"],
                environment_data=task_dict["environment_data"],
                evaluation_data=task_dict["evaluation_data"],
                metadata=task_dict["metadata"],
            )
        )

        # Enrich config with framework and model info
        config["framework"] = framework
        config["model_config"] = {"model_id": model_id, "temperature": temperature}

        # Derive seeds for all agents in this config
        if seed is not None:
            for agent_spec in config["agents"]:
                agent_spec["seed"] = derive_seed(seed, task_id, agent_spec["agent_id"])

        configs_data.append(config)

    print(f"Loaded {len(tasks_data)} tasks and {len(configs_data)} agent configs\n")

    return TaskCollection(tasks_data), configs_data


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    from langfuse import get_client  # TODO remove
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor

    SmolagentsInstrumentor().instrument()

    langfuse = get_client()

    # Verify connection
    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
    else:
        print("Authentication failed. Please check your credentials and host.")
        exit(1)

    args = parser.parse_args()

    print("Running 5-A-Day Benchmark")
    print(f"Framework: {args.framework}")
    print(f"Config: {args.config_type}agent")
    print(f"Model: {args.model_id}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed if args.seed is not None else 'None (non-deterministic)'}")
    print(f"Task limit: {args.limit or 'all'}")
    print(f"Specific task: {args.task if args.task is not None else 'all'}\n")

    # Load benchmark data
    tasks, agent_configs = load_benchmark_data(
        config_type=args.config_type,
        framework=args.framework,
        model_id=args.model_id,
        temperature=args.temperature,
        limit=args.limit,
        specific_task=args.task,
        seed=args.seed,
    )

    logger = FileResultLogger(
        output_dir=str(Path(__file__).parent / "results"),
        filename_pattern=f"{args.framework}_{args.config_type}agent_{{timestamp}}.jsonl",
        validate_on_completion=False,
    )

    benchmark = FiveADayBenchmark(
        agent_data=agent_configs,
        callbacks=[logger],
        fail_on_task_error=True,
        fail_on_evaluation_error=True,
    )
    results = benchmark.run(tasks=tasks)

    print("\n--- Benchmark Complete ---")
    print(f"Total tasks: {len(tasks)}")
    print(f"Results saved to: {logger.output_dir}")
