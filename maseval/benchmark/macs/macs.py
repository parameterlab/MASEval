"""MACS Benchmark - Multi-Agent Collaboration Scenarios.

Framework-agnostic implementation of the AWS MACS benchmark for evaluating
multi-agent collaboration in enterprise applications.

Reference: https://arxiv.org/abs/2412.05449
Dataset: https://github.com/aws-samples/multiagent-collab-scenario-benchmark

Usage:
    from maseval.benchmark.macs import (
        MACSBenchmark, MACSEnvironment, MACSEvaluator, MACSGenericTool,
        load_tasks, load_agent_config,
    )

    # Load data
    tasks = load_tasks("travel", limit=5)
    agent_config = load_agent_config("travel")

    # Create your framework-specific benchmark subclass
    class MyMACSBenchmark(MACSBenchmark):
        def setup_agents(self, agent_data, environment, task, user):
            # Your framework-specific agent creation
            ...

    # Run
    benchmark = MyMACSBenchmark(agent_data=agent_config, model=my_model)
    results = benchmark.run(tasks)
"""

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from maseval import (
    AgentAdapter,
    Benchmark,
    Environment,
    Evaluator,
    MessageHistory,
    ModelAdapter,
    Task,
    ToolInvocationHistory,
    ToolLLMSimulator,
    User,
)
from maseval.core.config import ConfigurableMixin
from maseval.core.tracing import TraceableMixin


# =============================================================================
# Tool
# =============================================================================


class MACSGenericTool(TraceableMixin, ConfigurableMixin):
    """Framework-agnostic tool with LLM-based response simulation.

    This tool does not inherit from any framework-specific Tool class.
    Users wrap it for their framework using composition. Example for smolagents:

        class MySmolagentsTool(smolagents.Tool):
            skip_forward_signature_validation = True

            def __init__(self, generic_tool: MACSGenericTool):
                self.generic_tool = generic_tool
                self.name = generic_tool.name
                self.description = generic_tool.description
                self.inputs = generic_tool.inputs
                self.output_type = "string"
                super().__init__()

            def forward(self, **kwargs) -> str:
                return self.generic_tool(**kwargs)
    """

    def __init__(self, spec: Dict[str, Any], model: ModelAdapter):
        """Initialize tool from specification.

        Args:
            spec: Tool specification with 'name', 'description', 'input_schema'
            model: ModelAdapter for LLM-based response simulation
        """
        super().__init__()
        self.name = spec["name"]
        self.description = spec.get("description", "")
        self.input_schema = spec.get("input_schema", {})
        self.output_type = "string"
        self.history = ToolInvocationHistory()

        # Convert schema to inputs format
        self.inputs = self._schema_to_inputs(self.input_schema)

        # Create simulator
        self.simulator = ToolLLMSimulator(
            model=model,
            tool_name=self.name,
            tool_description=self.description,
            tool_inputs=self.inputs,
        )

    @staticmethod
    def _schema_to_inputs(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON schema to inputs format."""
        inputs = {}
        for k, prop in schema.get("properties", {}).items():
            dtype = prop.get("data_type") or prop.get("type", "string")
            inputs[k] = {
                "type": dtype if isinstance(dtype, str) else "string",
                "description": prop.get("description", ""),
            }
        return inputs

    def __call__(self, **kwargs) -> str:
        """Execute the tool with simulated response."""
        response, details = self.simulator(actual_inputs=kwargs)
        self.history.add_invocation(
            inputs=kwargs,
            outputs=response,
            status="success",
            meta=details,
        )
        return response

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces."""
        return {
            **super().gather_traces(),
            "name": self.name,
            "invocations": self.history.to_list(),
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration."""
        return {
            **super().gather_config(),
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        args = ", ".join(f"{k}: {v['type']}" for k, v in self.inputs.items())
        return f"{self.__class__.__name__}({self.name}({args}) -> {self.output_type})"


# =============================================================================
# Evaluator
# =============================================================================


class MACSEvaluator(Evaluator):
    """LLM-based assertion evaluator for GSR metrics.

    Follows AWS paper methodology for Goal Success Rate (GSR) evaluation:
    - user: Evaluates user-observable behaviors (conversation only)
    - system: Evaluates internal behaviors (tool calls, agent actions)
    """

    DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"

    def __init__(
        self,
        model: ModelAdapter,
        task: Task,
        gsr_type: Literal["user", "system"] = "user",
        template: Optional[str] = None,
    ):
        """Initialize the evaluator.

        Args:
            model: ModelAdapter for LLM evaluation
            task: Task being evaluated (contains assertions)
            gsr_type: Either "user" or "system"
            template: Optional custom prompt template (uses default if None)
        """
        # Note: base Evaluator.__init__ does nothing, so we skip calling it
        # to avoid needing a real Environment instance
        self.model = model
        self.task = task
        self.gsr_type = gsr_type

        # Load template
        if template is None:
            template_file = "user.txt" if gsr_type == "user" else "system.txt"
            template_path = self.DEFAULT_TEMPLATES_DIR / template_file
            self.template = template_path.read_text()
        else:
            self.template = template

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Filter traces based on gsr_type.

        For user evaluation: only user-observable messages
        For system evaluation: full traces including tool invocations
        """
        if self.gsr_type == "user":
            user_trace = traces.get("user", {})
            return {"messages": MessageHistory(user_trace.get("history", []))}
        else:
            # System gets everything
            return traces

    def __call__(
        self,
        traces: Dict[str, Any],
        final_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate the trace against assertions.

        Args:
            traces: Filtered traces dict containing 'messages' and optionally 'tool_traces'
            final_answer: Final answer from agents (unused in MACS evaluation)

        Returns:
            Dict with: gsr, partial_gsr, report (list of assertion judgments)
        """
        # Extract message history and tool traces from the traces dict
        trace = traces.get("messages", MessageHistory())
        tool_traces = traces.get("tool_traces")

        # Parse assertions for this evaluation type
        all_assertions = self.task.evaluation_data.get("assertions", [])
        assertions = self._parse_assertions(all_assertions)

        if not assertions:
            return {"gsr": 1.0, "partial_gsr": 1.0, "report": []}

        # Format conversation history
        history = self._format_conversation_history(trace)

        # Get scenario description
        scenario = self.task.metadata.get("scenario", "")
        if not scenario:
            raise ValueError("Task metadata must include 'scenario' for GSR evaluation")

        # Build prompt
        if self.gsr_type == "user":
            prompt = (
                self.template.replace("{{scenario}}", scenario).replace("{{history}}", history).replace("{{assertions}}", "\n".join(assertions))
            )
        else:
            invocations = self._format_tool_invocations(tool_traces or {})
            prompt = (
                self.template.replace("{{scenario}}", scenario)
                .replace("{{history}}", history)
                .replace("{{invocations}}", invocations)
                .replace("{{assertions}}", "\n".join(assertions))
            )

        # Get LLM judgment
        response = self.model.generate(prompt).strip()
        response = response.strip("```").strip("json").strip()

        try:
            report = json.loads(response)

            # Handle wrapped responses
            for key in ["assertions", "results"]:
                if isinstance(report, dict) and key in report:
                    report = report[key]
                    break

            if isinstance(report, dict):
                report = [report]

            gsr, partial_gsr = self._compute_gsr(report)

            for item in report:
                item["assertion_type"] = self.gsr_type

            return {"gsr": gsr, "partial_gsr": partial_gsr, "report": report}

        except json.JSONDecodeError as e:
            return {
                "gsr": 0.0,
                "partial_gsr": 0.0,
                "report": [],
                "error": f"JSON decode error: {e}",
                "raw_response": response,
            }

    def _parse_assertions(self, assertions: List[str]) -> List[str]:
        """Parse assertions and filter by type."""
        parsed = []
        user_prefix, system_prefix = "user:", "agent:"

        for assertion in assertions:
            assertion = assertion.strip()

            if self.gsr_type == "user":
                if assertion.lower().startswith(user_prefix):
                    parsed.append(assertion[len(user_prefix) :].strip())
                elif not assertion.lower().startswith(system_prefix):
                    # No prefix means user assertion (AWS default)
                    parsed.append(assertion)
            else:
                if assertion.lower().startswith(system_prefix):
                    parsed.append(assertion[len(system_prefix) :].strip())

        return parsed

    def _format_conversation_history(self, trace: MessageHistory) -> str:
        """Format conversation history for the prompt."""
        lines = []
        for msg in trace:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, list):
                content = " ".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)

            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _format_tool_invocations(self, tool_traces: Dict[str, Any]) -> str:
        """Format tool invocations for system-side evaluation."""
        lines = []

        for tool_name, tool_data in tool_traces.items():
            invocations = tool_data.get("invocations", [])
            for inv in invocations:
                lines.append(
                    f"Tool: {tool_name}\n"
                    f"  Inputs: {inv.get('inputs', {})}\n"
                    f"  Outputs: {inv.get('outputs', '')}\n"
                    f"  Status: {inv.get('status', 'Unknown')}"
                )

        return "\n".join(lines) if lines else "No tool invocations recorded"

    def _compute_gsr(self, report: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Compute GSR metrics.

        Returns:
            (gsr, partial_gsr) where:
            - gsr: 1.0 if all assertions True, else 0.0
            - partial_gsr: Percentage of True assertions
        """
        if not report:
            return 1.0, 1.0

        true_count = sum(1 for item in report if str(item.get("answer", "")).lower() == "true")
        total = len(report)

        gsr = 1.0 if true_count == total else 0.0
        partial_gsr = true_count / total if total > 0 else 1.0

        return gsr, partial_gsr


# =============================================================================
# User Simulator
# =============================================================================


class MACSUser(User):
    """MACS-specific user simulator with conversation limits.

    Extends the base User class with MACS-specific behavior:
    - Maximum 5 turns of interaction (as per MACS paper)
    - </stop> token detection for natural conversation ending
    - User profile and scenario-aware responses

    The simulator maintains a conversation history and uses an LLM to generate
    responses that are consistent with the user's profile and scenario.

    Note: This is a base class. Framework-specific subclasses should override
    get_tool() to return a compatible tool (e.g., SmolAgentUserSimulationInputTool).
    """

    DEFAULT_MAX_TURNS = 5
    STOP_TOKEN = "</stop>"

    def __init__(
        self,
        model: ModelAdapter,
        scenario: str,
        initial_prompt: str,
        name: str = "Simulated User",
        template: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ):
        """Initialize MACS user simulator.

        Args:
            model: ModelAdapter for LLM-based response generation
            scenario: Full scenario text (contains goals and user background)
            initial_prompt: The initial query to the agent
            name: User name for identification (default: "Simulated User")
            template: Optional custom prompt template (uses base User's default)
            max_turns: Maximum conversation turns (default: 5, per MACS paper)
        """
        # Extract user profile from scenario text
        user_profile = self._extract_user_profile(scenario)

        super().__init__(
            name=name,
            model=model,
            user_profile=user_profile,
            scenario=scenario,
            initial_prompt=initial_prompt,
            template=template,
        )
        self.max_turns = max_turns
        self._turn_count = 0
        self._stopped = False

    def get_tool(self) -> Any:
        """Return a tool for agent interaction.

        This base implementation raises NotImplementedError.
        Framework-specific subclasses should override this method.

        For smolagents, use SmolAgentMACSUser which provides a smolagents-compatible tool.
        For langgraph, use LangGraphMACSUser which provides a langchain-compatible tool.

        Raises:
            NotImplementedError: Always, as this must be implemented by subclass.
        """
        raise NotImplementedError(
            "MACSUser.get_tool() must be overridden by framework-specific subclass. "
            "Use SmolAgentMACSUser for smolagents or LangGraphMACSUser for langgraph."
        )

    @property
    def is_done(self) -> bool:
        """Check if the conversation should end.

        Returns True if:
        - Maximum turns reached
        - User responded with </stop> token
        """
        return self._stopped or self._turn_count >= self.max_turns

    def simulate_response(self, question: str) -> str:
        """Simulate a user response, respecting turn limits.

        Args:
            question: The assistant's question/message

        Returns:
            The simulated user response, or empty string if done
        """
        if self.is_done:
            return ""

        # Use parent's simulate_response which handles LLM generation
        response = super().simulate_response(question)

        # Check for stop token
        if self.STOP_TOKEN in response.lower():
            self._stopped = True
            # Clean up the response
            response = response.replace(self.STOP_TOKEN, "").strip()
            if not response:
                response = "Thank you, that's all I needed!"

        self._turn_count += 1
        return response

    def reset(self) -> None:
        """Reset the conversation state for a new interaction."""
        self._turn_count = 0
        self._stopped = False
        # Keep only the initial user message
        if len(self.messages) > 0:
            initial = self.messages[0]
            self.messages = MessageHistory([initial])

    @staticmethod
    def _extract_user_profile(scenario: str) -> Dict[str, Any]:
        """Extract user profile from scenario text.

        The MACS scenarios contain user background info after "Background:" marker.

        Args:
            scenario: Full scenario text with goals and background

        Returns:
            Dict with user profile fields
        """
        profile: Dict[str, Any] = {}

        # Find the Background section
        if "Background:" in scenario:
            background_section = scenario.split("Background:")[-1].strip()

            # Parse bullet points (* User's name is ...)
            for line in background_section.split("\n"):
                line = line.strip().lstrip("*").strip()
                if line.lower().startswith("user"):
                    # Try to extract key-value pairs
                    if " is " in line.lower():
                        key_part, value_part = line.split(" is ", 1)
                        key = key_part.lower().replace("user's ", "").replace("user ", "").strip()
                        profile[key] = value_part.strip().rstrip(".")
                    elif " has " in line.lower():
                        key_part, value_part = line.split(" has ", 1)
                        key = key_part.lower().replace("user's ", "").replace("user ", "").strip()
                        profile[key] = value_part.strip().rstrip(".")

        # Include full scenario as fallback context
        profile["full_scenario"] = scenario

        return profile

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces with MACS-specific information."""
        traces = super().gather_traces()
        traces.update(
            {
                "max_turns": self.max_turns,
                "turns_used": self._turn_count,
                "stopped_by_user": self._stopped,
            }
        )
        return traces


# =============================================================================
# Environment
# =============================================================================


class MACSEnvironment(Environment):
    """Unified environment for all MACS domains.

    Creates MACSGenericTool instances from task's environment_data.
    Users can override to convert tools to their framework format.
    """

    def __init__(
        self,
        task_data: Dict[str, Any],
        model: ModelAdapter,
        callbacks: Optional[List[Any]] = None,
    ):
        """Initialize environment.

        Args:
            task_data: Task data containing environment_data with tool specs
            model: ModelAdapter for tool simulation
            callbacks: Optional callbacks
        """
        self._model = model
        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state from task data."""
        return {
            "tool_specs": task_data.get("environment_data", {}).get("tools", []),
        }

    def create_tools(self) -> List[MACSGenericTool]:
        """Create framework-agnostic tools from specifications."""
        tools = []
        seen = set()

        for tool_group in self.state["tool_specs"]:
            for action in tool_group.get("actions", []):
                name = action.get("name")
                if name and name not in seen:
                    tools.append(MACSGenericTool(action, self._model))
                    seen.add(name)

        return tools

    def get_tools_by_group(self, group_names: List[str]) -> List[MACSGenericTool]:
        """Get tools belonging to specified tool groups.

        Args:
            group_names: List of tool group names (e.g., ["Weather", "BookFlight"])

        Returns:
            List of tools from those groups
        """
        result = []
        for tool_group in self.state["tool_specs"]:
            if tool_group.get("tool_name") in group_names:
                for action in tool_group.get("actions", []):
                    name = action.get("name")
                    if name and name in self._tools_dict:
                        result.append(self._tools_dict[name])
        return result

    def get_tools_for_agent(self, agent_spec: Dict[str, Any]) -> List[MACSGenericTool]:
        """Get tools for a specific agent based on its configuration.

        Args:
            agent_spec: Agent specification dict with 'tools' key containing tool group names

        Returns:
            List of MACSGenericTool instances assigned to this agent
        """
        tool_groups = agent_spec.get("tools", [])
        return self.get_tools_by_group(tool_groups)


# =============================================================================
# Benchmark
# =============================================================================


class MACSBenchmark(Benchmark):
    """MACS Benchmark - Framework-agnostic base class.

    This base class handles:
    - Environment setup with MACSEnvironment
    - Dual evaluator setup (user-side + system-side)
    - GSR metric aggregation

    Users must subclass and implement setup_agents() for their framework.
    """

    def __init__(
        self,
        agent_data: Dict[str, Any],
        model: ModelAdapter,
        callbacks: Optional[List[Any]] = None,
        n_task_repeats: int = 1,
        **kwargs: Any,
    ):
        """Initialize benchmark.

        Args:
            agent_data: Agent configuration from load_agent_config()
            model: ModelAdapter for tool simulation and evaluation
            callbacks: Benchmark callbacks
            n_task_repeats: Repetitions per task
        """
        self._model = model
        super().__init__(agent_data, callbacks, n_task_repeats, **kwargs)

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
    ) -> MACSEnvironment:
        """Create environment for a task."""
        return MACSEnvironment(
            task_data={"environment_data": task.environment_data},
            model=self._model,
        )

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
    ) -> MACSUser:
        """Create MACS user simulator.

        Creates a MACSUser with scenario and query from the task.
        The user profile is automatically extracted from the scenario text.

        Note: MACSUser.get_tool() raises NotImplementedError.
        Framework-specific subclasses in examples should wrap this user
        or override setup_user() to return a user with get_tool() implemented.

        Args:
            agent_data: Agent configuration
            environment: The task environment
            task: Current task with scenario and user profile

        Returns:
            MACSUser instance
        """
        scenario = task.metadata.get("scenario", "")
        return MACSUser(
            model=self._model,
            scenario=scenario,
            initial_prompt=task.query,
        )

    @abstractmethod
    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for this task. Must be implemented by subclass.

        Args:
            agent_data: Agent configuration with hierarchy spec
            environment: MACSEnvironment with tools
            task: Current task
            user: Optional user simulator

        Returns:
            Tuple of (ordered agent list, agent dict keyed by ID)
        """
        pass

    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
    ) -> Sequence[Evaluator]:
        """Create user-side and system-side evaluators."""
        return [
            MACSEvaluator(self._model, task, gsr_type="user"),
            MACSEvaluator(self._model, task, gsr_type="system"),
        ]

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
    ) -> Any:
        """Execute agents and return final answer."""
        answers = [agent.run(task.query) for agent in agents]
        return answers[0] if len(answers) == 1 else answers

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate using both evaluators and aggregate GSR metrics.

        Returns AWS paper format:
        - user_gsr, system_gsr, overall_gsr, supervisor_gsr
        - user_partial_gsr, system_partial_gsr, overall_partial_gsr
        - report: Combined assertion judgments
        """
        # Get agent traces - primary agent's messages
        primary_agent_id = list(agents.keys())[0]
        agent_trace = traces.get("agents", {}).get(primary_agent_id, {})
        all_messages = MessageHistory(agent_trace.get("messages", []))

        # For user-side evaluation: filter to user-observable messages only
        # (user queries and assistant responses - not tool calls)
        user_messages = MessageHistory([msg for msg in all_messages if msg.get("role") in ("user", "assistant")])

        tool_traces = traces.get("tools", {})

        # Run evaluators with properly structured traces dict
        results = []
        for evaluator in evaluators:
            if isinstance(evaluator, MACSEvaluator) and evaluator.gsr_type == "system":
                eval_traces = {"messages": all_messages, "tool_traces": tool_traces}
            else:
                eval_traces = {"messages": user_messages}
            result = evaluator(eval_traces, final_answer)
            results.append(result)

        # Combine results
        user_result = results[0] if results else {"gsr": 0.0, "partial_gsr": 0.0, "report": []}
        system_result = results[1] if len(results) > 1 else {"gsr": 0.0, "partial_gsr": 0.0, "report": []}

        combined_report = user_result.get("report", []) + system_result.get("report", [])

        # Compute overall metrics per AWS paper
        overall_gsr = 1.0 if (user_result.get("gsr", 0.0) == 1.0 and system_result.get("gsr", 0.0) == 1.0) else 0.0

        # Supervisor GSR: success if overall passes OR user-side passes
        supervisor_gsr = 1.0 if (overall_gsr == 1.0 or user_result.get("gsr", 0.0) == 1.0) else 0.0

        # Overall partial GSR
        if combined_report:
            total_true = sum(1 for item in combined_report if str(item.get("answer", "")).lower() == "true")
            overall_partial_gsr = total_true / len(combined_report)
        else:
            overall_partial_gsr = 1.0

        return [
            {
                "user_gsr": user_result.get("gsr", 0.0),
                "user_partial_gsr": user_result.get("partial_gsr", 0.0),
                "system_gsr": system_result.get("gsr", 0.0),
                "system_partial_gsr": system_result.get("partial_gsr", 0.0),
                "overall_gsr": overall_gsr,
                "overall_partial_gsr": overall_partial_gsr,
                "supervisor_gsr": supervisor_gsr,
                "report": combined_report,
            }
        ]


# =============================================================================
# Utility Functions
# =============================================================================


def compute_benchmark_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics across all benchmark results.

    Args:
        results: List of result dicts from benchmark.run()

    Returns:
        Dict with total_tasks, successful_tasks, success_rate, mean_metrics
    """
    if not results:
        return {
            "total_tasks": 0,
            "successful_tasks": 0,
            "success_rate": 0.0,
            "mean_metrics": {},
        }

    total_tasks = len(results)
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    successful_tasks = 0

    for res in results:
        evals = res.get("eval") or []
        found_success = False

        for entry in evals:
            for k, v in entry.items():
                if isinstance(v, (int, float)):
                    metric_sums[k] = metric_sums.get(k, 0.0) + v
                    metric_counts[k] = metric_counts.get(k, 0) + 1

            if not found_success and entry.get("overall_gsr", 0.0) == 1.0:
                found_success = True

        if found_success:
            successful_tasks += 1

    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
    mean_metrics = {k: metric_sums[k] / metric_counts[k] if metric_counts[k] else 0.0 for k in metric_sums}

    return {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "success_rate": success_rate,
        "mean_metrics": mean_metrics,
    }
