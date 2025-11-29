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

from maseval import Benchmark, Environment, Evaluator, Task, TaskCollection, AgentAdapter, MessageHistory
from maseval.core.callbacks.result_logger import FileResultLogger

# Import tool implementations
from tools import (
    EmailTool,
    BankingTool,
    CalculatorTool,
    CodeExecutionTool,
    FamilyInfoTool,
    StockPriceTool,
    CalendarTool,
    RunningAppTool,
    GymTrackerTool,
    HotelSearchTool,
    MCPCalendarTool,
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


# ============================================================================
# Global Model Factory
# ============================================================================


def get_model(model_id: str, framework: str, temperature: float = 0.7):
    """Get a model instance for the specified framework.

    Args:
        model_id: Model identifier (e.g., 'gemini-2.5-flash')
        framework: Target framework ('smolagents', 'langgraph', 'llamaindex')
        temperature: Model temperature (default: 0.7)

    Returns:
        Framework-specific model instance
    """
    if framework == "smolagents":
        from smolagents import LiteLLMModel

        return LiteLLMModel(
            model_id=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )

    elif framework == "langgraph":
        from langchain_litellm import ChatLiteLLM

        return ChatLiteLLM(
            model=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )

    elif framework == "llamaindex":
        from llama_index.llms.litellm import LiteLLM

        return LiteLLM(
            model=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
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
        # For other_calendar, check person-specific availability
        if calendar_key == "other_calendar":
            person = self.state["other_person_name"]
            availability = self.state[f"{person}_availability"]
        else:
            availability = self.state[calendar_key]

        # Convert to events format
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

        # Map tool names to tool classes and their initialization data
        tool_mapping = {
            "email": (EmailTool, lambda: self.state["email_inbox"]),
            "banking": (BankingTool, lambda: (self.state["bank_transactions"], self.state["current_balance"])),
            "calculator": (CalculatorTool, lambda: ()),
            "python_executor": (CodeExecutionTool, lambda: (self.state["test_cases"],)),
            "family_info": (FamilyInfoTool, lambda: (self.state["family_info"],)),
            "stock_price": (StockPriceTool, lambda: (self.state["stock_price_lookup"],)),
            "calendar": (CalendarTool, lambda: ("my_calendar", self.state["my_calendar_availability"])),
            "running_app": (RunningAppTool, lambda: (self.state["running_activities"],)),
            "gym_tracker": (GymTrackerTool, lambda: (self.state["gym_activities"],)),
            "hotel_search": (HotelSearchTool, lambda: (self.state["hotels"],)),
            # MCP calendar tools
            "my_calendar_mcp": (
                MCPCalendarTool,
                lambda: ("my_calendar_mcp", self._get_mcp_calendar_data("my_calendar_availability")),
            ),
            "other_calendar_mcp": (
                MCPCalendarTool,
                lambda: ("other_calendar_mcp", self._get_mcp_calendar_data("other_calendar")),
            ),
        }

        for tool_name in self.state["tools"]:
            if tool_name in tool_mapping:
                ToolClass, get_init_args = tool_mapping[tool_name]
                init_args = get_init_args()

                # Create base tool instance
                if isinstance(init_args, tuple):
                    base_tool = ToolClass(*init_args)
                else:
                    base_tool = ToolClass(init_args)

                # Convert to framework-specific tool
                framework_tool = self._convert_tool(base_tool)
                tools_list.append(framework_tool)

        return tools_list

    def _convert_tool(self, base_tool):
        """Convert BaseTool to framework-specific tool type.

        Args:
            base_tool: BaseTool instance

        Returns:
            Framework-specific tool object with tracing preserved
        """
        if self.framework == "smolagents":
            adapter = base_tool.to_smolagents()
            # Return the actual tool, not the adapter
            # The adapter has a .tool attribute with the smolagents Tool instance
            return adapter.tool if hasattr(adapter, "tool") else adapter
        elif self.framework == "langgraph":
            adapter = base_tool.to_langgraph()
            # LangGraph adapter already returns the StructuredTool directly
            return adapter
        elif self.framework == "llamaindex":
            return base_tool.to_llamaindex()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")


# ============================================================================
# Benchmark
# ============================================================================


class FiveADayBenchmark(Benchmark):
    """5-A-Day benchmark with framework-agnostic tools.

    Demonstrates the maseval library through 5 diverse tasks with different evaluation approaches.
    Supports single-agent and multi-agent (orchestrator+specialist) configurations.
    """

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize name to be a valid Python identifier (for smolagents)."""
        sanitized = name.replace(" ", "_").replace("-", "_")
        if not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized
        return sanitized

    @staticmethod
    def _filter_tools(all_tools: List[Any], tool_names: List[str]) -> List[Any]:
        """Filter tools by name."""
        if not tool_names:
            return []
        return [t for t in all_tools if hasattr(t, "name") and t.name in tool_names]

    def setup_environment(self, agent_data: Dict[str, Any], task: Task) -> Environment:
        """Create environment from task data.

        Loads tools based on task.environment_data and converts them to the
        framework specified in agent_data.

        Args:
            agent_data: Agent configuration including framework specification
            task: Task containing environment_data, query, and evaluation_data

        Returns:
            FiveADayEnvironment with framework-specific tools
        """
        # Pass full task data to environment
        task_data = {
            "environment_data": task.environment_data,
            "query": task.query,
            "evaluation_data": task.evaluation_data,
            "metadata": task.metadata,
        }

        return FiveADayEnvironment(task_data, framework=agent_data["framework"])

    def setup_agents(
        self, agent_data: Dict[str, Any], environment: Environment, task: Task, user=None
    ) -> tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create framework-specific agent with tools from environment."""
        if agent_data["agent_type"] == "single":
            setup_fn = self._setup_single_agent
        elif agent_data["agent_type"] == "multi":
            setup_fn = self._setup_multi_agent
        else:
            raise ValueError(f"Unsupported agent_type: {agent_data['agent_type']}")

        return setup_fn(
            agent_data,
            environment,
            agent_data["framework"],
            agent_data["model_config"]["model_id"],
            agent_data["model_config"]["temperature"],
        )

    def _setup_single_agent(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        framework: str,
        model_id: str,
        temperature: float,
    ) -> tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Setup a single agent with all tools."""
        tools = environment.get_tools()
        agent_spec = agent_data["agent"]

        # Sanitize agent name to be a valid Python identifier for smolagents
        sanitized_name = self._sanitize_name(agent_spec["agent_name"])

        if framework == "smolagents":
            from smolagents import ToolCallingAgent

            # Create smolagents model
            model = get_model(model_id, framework, temperature)

            # Create agent
            agent = ToolCallingAgent(
                model=model,
                tools=tools,
                name=sanitized_name,
                instructions=agent_spec["agent_instruction"],
                verbosity_level=2,
            )

            # Wrap in adapter
            from maseval.interface.agents.smolagents import SmolAgentAdapter

            wrapper = SmolAgentAdapter(agent, agent_spec["agent_id"])

        elif framework == "langgraph":
            from langchain_core.messages import SystemMessage
            from langgraph.graph import StateGraph, END
            from langgraph.graph.message import add_messages
            from langgraph.prebuilt import ToolNode, tools_condition
            from typing_extensions import TypedDict, Annotated

            # Create LangChain model
            model = get_model(model_id, framework, temperature)

            # Create agent graph
            class AgentState(TypedDict):
                messages: Annotated[List[Any], add_messages]

            llm_with_tools = model.bind_tools(tools)

            def call_model(state: AgentState):
                messages = state["messages"]
                # Add system message with agent instruction if not present
                has_system = any(isinstance(m, SystemMessage) for m in messages)
                if not has_system and agent_spec["agent_instruction"]:
                    system_message = SystemMessage(content=agent_spec["agent_instruction"])
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

            # Wrap in adapter
            from maseval.interface.agents.langgraph import LangGraphAgentAdapter

            wrapper = LangGraphAgentAdapter(graph, agent_spec["agent_id"])

        elif framework == "llamaindex":
            from llama_index.core.agent import ReActAgent

            # Create LlamaIndex LiteLLM model (supports Gemini via 'gemini/' prefix)
            model = get_model(model_id, framework, temperature)

            # Create ReActAgent with tools
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=model,
                verbose=True,
                max_iterations=10,
            )

            # Set system prompt if provided
            if agent_spec["agent_instruction"]:
                agent.update_prompts({"agent_worker:system_prompt": agent_spec["agent_instruction"]})

            # Wrap in adapter
            from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter

            wrapper = LlamaIndexAgentAdapter(agent, agent_spec["agent_id"])

        else:
            raise ValueError(f"Unsupported framework: {framework}")

        return [wrapper], {agent_spec["agent_id"]: wrapper}

    def _setup_multi_agent(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        framework: str,
        model_id: str,
        temperature: float,
    ) -> tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Setup multiple agents with orchestrator pattern."""
        primary_agent_id = agent_data["primary_agent_id"]
        agents_specs = agent_data["agents"]

        if not primary_agent_id:
            raise ValueError("Multi-agent setup requires primary_agent_id")

        primary_spec = next((a for a in agents_specs if a["agent_id"] == primary_agent_id), None)
        if not primary_spec:
            raise ValueError(f"Primary agent {primary_agent_id} not found in agents list")

        specialist_specs = [a for a in agents_specs if a["agent_id"] != primary_agent_id]

        if framework == "smolagents":
            from smolagents import ToolCallingAgent, FinalAnswerTool

            # Create smolagents model
            model = get_model(model_id, framework, temperature)

            # Create specialist agents
            specialist_agents = []
            all_tools = environment.get_tools()

            for agent_spec in specialist_specs:
                # Get tools for this specialist
                specialist_tools = self._filter_tools(all_tools, agent_spec["tools"])
                specialist_tools.append(FinalAnswerTool())

                # Sanitize agent name to be a valid Python identifier
                sanitized_name = self._sanitize_name(agent_spec["agent_name"])

                # Create specialist agent
                specialist = ToolCallingAgent(
                    model=model,
                    tools=specialist_tools,
                    name=sanitized_name,
                    description=agent_spec["agent_instruction"],
                    instructions=agent_spec["agent_instruction"],
                    verbosity_level=2,
                )
                specialist_agents.append(specialist)

            # Get primary agent tools (usually empty for orchestrators)
            primary_tools = self._filter_tools(all_tools, primary_spec["tools"])
            primary_tools.append(FinalAnswerTool())

            # Sanitize agent name to be a valid Python identifier
            sanitized_primary_name = self._sanitize_name(primary_spec["agent_name"])

            # Create primary orchestrator agent with managed_agents
            agent = ToolCallingAgent(
                model=model,
                tools=primary_tools,
                managed_agents=specialist_agents if specialist_agents else None,
                name=sanitized_primary_name,
                instructions=primary_spec["agent_instruction"],
                verbosity_level=2,
            )

            # Wrap in adapter
            from maseval.interface.agents.smolagents import SmolAgentAdapter

            wrapper = SmolAgentAdapter(agent, primary_agent_id)

        elif framework == "langgraph":
            from langchain_core.messages import SystemMessage
            from langgraph.graph import StateGraph, END
            from langgraph.graph.message import add_messages
            from typing_extensions import TypedDict, Annotated
            from typing import Literal

            # Create LangChain model
            model = get_model(model_id, framework, temperature)

            # Define state for multi-agent graph
            class MultiAgentState(TypedDict):
                messages: Annotated[List[Any], add_messages]
                next_agent: str

            all_tools = environment.get_tools()

            # Create specialist agent nodes
            specialist_nodes = {}
            for agent_spec in specialist_specs:
                agent_id = agent_spec["agent_id"]
                agent_instruction = agent_spec["agent_instruction"]

                # Get tools for this specialist
                specialist_tools = self._filter_tools(all_tools, agent_spec["tools"])

                # Create specialist node function
                specialist_model = model.bind_tools(specialist_tools) if specialist_tools else model

                def make_specialist_node(spec_model, spec_instruction, spec_tools, spec_name):
                    def specialist_node(state: MultiAgentState):
                        messages = state["messages"]
                        # Add system message with specialist instruction
                        if spec_instruction and not any(isinstance(m, SystemMessage) for m in messages):
                            messages = [SystemMessage(content=spec_instruction)] + messages

                        # If there are tools, use tool calling workflow
                        if spec_tools:
                            from langgraph.prebuilt import ToolNode

                            response = spec_model.invoke(messages)
                            # Check if tools were called
                            if hasattr(response, "tool_calls") and response.tool_calls:
                                # Execute tools
                                tool_node = ToolNode(spec_tools)
                                tool_results = tool_node.invoke({"messages": [response]})
                                # Get final response after tool execution
                                final_messages = messages + [response] + tool_results["messages"]
                                final_response = spec_model.invoke(final_messages)
                                return {"messages": [final_response], "next_agent": "orchestrator"}
                            return {"messages": [response], "next_agent": "orchestrator"}
                        else:
                            response = spec_model.invoke(messages)
                            return {"messages": [response], "next_agent": "orchestrator"}

                    return specialist_node

                specialist_nodes[agent_id] = make_specialist_node(
                    specialist_model, agent_instruction, specialist_tools, agent_spec["agent_name"]
                )

            # Create orchestrator node with routing logic
            primary_instruction = primary_spec["agent_instruction"]

            def orchestrator_node(state: MultiAgentState):
                messages = state["messages"]

                # Add system message with orchestrator instruction
                if primary_instruction and not any(isinstance(m, SystemMessage) for m in messages):
                    orchestrator_prompt = (
                        f"{primary_instruction}\n\n"
                        f"Available specialists: {', '.join(specialist_nodes.keys())}\n"
                        "Analyze the query and decide which specialist to call next, or provide a final answer."
                    )
                    messages = [SystemMessage(content=orchestrator_prompt)] + messages

                response = model.invoke(messages)
                return {"messages": [response], "next_agent": "__end__"}

            # Router function to decide next agent
            def route_to_specialist(state: MultiAgentState) -> Literal["orchestrator", "__end__"] | str:
                next_agent = state.get("next_agent", "__end__")
                if next_agent == "__end__":
                    return END
                return next_agent

            # Build the multi-agent graph
            workflow = StateGraph(MultiAgentState)

            # Add orchestrator node
            workflow.add_node("orchestrator", orchestrator_node)

            # Add specialist nodes
            for agent_id, node_fn in specialist_nodes.items():
                workflow.add_node(agent_id, node_fn)

            # Set orchestrator as entry point
            workflow.set_entry_point("orchestrator")

            # Add conditional edges from orchestrator to specialists
            workflow.add_conditional_edges(
                "orchestrator", route_to_specialist, {agent_id: agent_id for agent_id in specialist_nodes.keys()} | {"__end__": END}
            )

            # Add edges from specialists back to orchestrator
            for agent_id in specialist_nodes.keys():
                workflow.add_edge(agent_id, "orchestrator")

            graph = workflow.compile()

            # Wrap in adapter
            from maseval.interface.agents.langgraph import LangGraphAgentAdapter

            wrapper = LangGraphAgentAdapter(graph, primary_agent_id)

        elif framework == "llamaindex":
            from llama_index.core.agent import ReActAgent
            from llama_index.core.tools import FunctionTool

            # Create LlamaIndex LiteLLM model (supports Gemini via 'gemini/' prefix)
            model = get_model(model_id, framework, temperature)

            all_tools = environment.get_tools()

            # Create specialist agents
            specialist_agents_dict = {}
            for agent_spec in specialist_specs:
                agent_id = agent_spec["agent_id"]
                agent_name = agent_spec["agent_name"]
                agent_instruction = agent_spec["agent_instruction"]

                # Get tools for this specialist
                specialist_tools = self._filter_tools(all_tools, agent_spec["tools"])

                # Create specialist agent
                specialist_agent = ReActAgent.from_tools(
                    tools=specialist_tools,
                    llm=model,
                    verbose=True,
                    max_iterations=10,
                )

                # Set system prompt if provided
                if agent_instruction:
                    specialist_agent.update_prompts({"agent_worker:system_prompt": agent_instruction})

                specialist_agents_dict[agent_id] = {
                    "agent": specialist_agent,
                    "name": agent_name,
                    "description": agent_instruction,
                }

            # Create handoff tools that delegate to specialist agents
            def make_handoff_tool(specialist_id: str, specialist_info: dict):
                def handoff_to_specialist(task: str) -> str:
                    f"""Delegate task to {specialist_info["name"]}.
                    
                    Args:
                        task: The task description for the specialist
                    
                    Returns:
                        The specialist's response
                    """
                    specialist_agent = specialist_info["agent"]
                    response = specialist_agent.chat(task)
                    return str(response)

                return FunctionTool.from_defaults(
                    fn=handoff_to_specialist,
                    name=f"ask_{specialist_id}",
                    description=f"Delegate to {specialist_info['name']}: {specialist_info['description']}",
                )

            # Create handoff tools for orchestrator
            orchestrator_tools = [make_handoff_tool(spec_id, spec_info) for spec_id, spec_info in specialist_agents_dict.items()]

            # Get primary agent tools (if any)
            primary_tools = self._filter_tools(all_tools, primary_spec["tools"])
            orchestrator_tools.extend(primary_tools)

            # Create orchestrator agent with handoff tools
            orchestrator = ReActAgent.from_tools(
                tools=orchestrator_tools,
                llm=model,
                verbose=True,
                max_iterations=15,
            )

            # Set orchestrator system prompt
            primary_instruction = primary_spec["agent_instruction"]
            if primary_instruction:
                orchestrator.update_prompts({"agent_worker:system_prompt": primary_instruction})

            # Wrap orchestrator in adapter
            from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter

            wrapper = LlamaIndexAgentAdapter(orchestrator, primary_agent_id)

        else:
            raise ValueError(f"Unsupported framework: {framework}")

        return [wrapper], {primary_agent_id: wrapper}

    def setup_user(self, agent_data: Dict[str, Any], environment: Environment, task: Task):
        """No user simulation for this benchmark."""
        return None

    def setup_evaluators(self, environment, task, agents, user) -> Sequence[Evaluator]:
        """Create evaluators based on task's evaluation_data.evaluators list."""
        if not task.evaluation_data["evaluators"]:
            return []

        # Dynamically instantiate evaluators
        evaluator_instances = []
        for name in task.evaluation_data["evaluators"]:
            # Get evaluator class from evaluators module
            evaluator_class = getattr(evaluators, name, None)
            if evaluator_class is None:
                raise ValueError(f"Evaluator '{name}' not found in evaluators module")

            # Instantiate with standard arguments
            evaluator_instances.append(evaluator_class(task, environment, user))

        return evaluator_instances

    def run_agents(self, agents: Sequence[AgentAdapter], task: Task, environment: Environment) -> Any:
        """Execute agents and return their final answers."""
        answers = [agent.run(task.query) for agent in agents]
        return answers[0] if len(answers) == 1 else answers

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate agent performance."""
        results = []

        # Get the main agent's trace from the agents dict in traces
        agent_traces = traces.get("agents", {})
        if not agent_traces:
            # No agent traces found
            return [{"evaluator": "system", "error": "No agent traces found", "success": False}]

        # Get the first agent's trace (for single agent) or orchestrator trace (for multi-agent)
        main_trace_dict = agent_traces.get("orchestrator") or next(iter(agent_traces.values()))

        # Extract messages and wrap in MessageHistory
        messages = main_trace_dict["messages"]
        message_history = MessageHistory(messages)

        # Run each evaluator
        for evaluator in evaluators:
            try:
                eval_result = evaluator(message_history)
                eval_result["evaluator"] = evaluator.__class__.__name__
                results.append(eval_result)
            except Exception as e:
                results.append({"evaluator": evaluator.__class__.__name__, "error": str(e), "success": False})

        return results


# ============================================================================
# Data Loading
# ============================================================================


def load_tasks(data_file: str = "data/tasks.json", limit: Optional[int] = None, specific_task_only: Optional[int] = None) -> TaskCollection:
    """Load tasks from JSON file.

    Args:
        data_file: Path to tasks.json file
        limit: Optional limit on number of tasks to load

    Returns:
        TaskCollection containing loaded tasks
    """
    data_path = Path(__file__).parent / data_file

    with open(data_path, "r") as f:
        tasks_list = json.load(f)

    if limit is not None and specific_task_only is not None:
        raise ValueError("Cannot specify both limit and specific_task_only")

    tasks_data = []
    for task_idx, task_dict in enumerate(tasks_list[:limit] if limit else tasks_list):
        if (specific_task_only is not None) and not (task_idx == specific_task_only):
            continue
        tasks_data.append(
            Task(
                query=task_dict["query"],
                environment_data=task_dict["environment_data"],
                evaluation_data=task_dict["evaluation_data"],
                metadata=task_dict["metadata"],
            )
        )

    print(f"Loaded {len(tasks_data)} tasks\n")

    return TaskCollection(tasks_data)


def load_agent_configs(
    config_file: str = "data/singleagent.json",
    framework: str = "smolagents",
    limit: Optional[int] = None,
    specific_task_only: Optional[int] = None,
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Load agent configurations from JSON file.

    Args:
        config_file: Path to agent configuration JSON file (singleagent.json or multiagent.json)
        framework: Target framework ('smolagents', 'langgraph', 'llamaindex')
        limit: Optional limit on number of configs to load (must match number of tasks)
        specific_task_only: Optional index to load only a specific config (must match specific task)
        model_id: Model identifier to inject into config
        temperature: Model temperature to inject into config

    Returns:
        List of agent configuration dictionaries with framework and model_config added
    """
    config_path = Path(__file__).parent / config_file

    with open(config_path, "r") as f:
        configs = json.load(f)

    if limit is not None and specific_task_only is not None:
        raise ValueError("Cannot specify both limit and specific_task_only")

    # Filter by specific task if specified
    configs_data = []
    for config_idx, config in enumerate(configs[:limit] if limit else configs):
        if (specific_task_only is not None) and not (config_idx == specific_task_only):
            continue
        # Add framework to config
        config["framework"] = framework
        # Create model_config from argparse arguments
        config["model_config"] = {
            "model_id": model_id,
            "temperature": temperature,
        }
        configs_data.append(config)

    return configs_data


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":

    from langfuse import get_client
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
    print(f"Task limit: {args.limit or 'all'}")
    print(f"Specific task: {args.task if args.task is not None else 'all'}\n")

    # Load tasks and agent configs
    tasks = load_tasks(limit=args.limit, specific_task_only=args.task)
    agent_configs = load_agent_configs(
        config_file=f"data/{args.config_type}agent.json",
        framework=args.framework,
        limit=args.limit,
        specific_task_only=args.task,
        model_id=args.model_id,
        temperature=args.temperature,
    )

    output_dir = Path(__file__).parent / "results"
    logger = FileResultLogger(
        output_dir=str(output_dir),
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
    print(f"Results saved to: {output_dir}")
