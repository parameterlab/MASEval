from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import os
import inspect
import time
from datetime import datetime

from maseval.core.user import User
from maseval.core.simulator import LLMSimulator, UserSimulatorError
from maseval.core.model import ModelAdapter


class AgenticUserLLMSimulator(LLMSimulator):
    """
    A simulator that uses an LLM to act as an agentic user (capable of using tools).
    """

    _component_name = "user_simulator"

    def __init__(
        self,
        model: ModelAdapter,
        user_profile: Dict[str, str],
        scenario: str,
        template: Optional[str] = None,
        max_try: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
        stop_token: Optional[str] = None,
        early_stopping_condition: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        # Validate early stopping configuration
        if (stop_token is None) != (early_stopping_condition is None):
            raise ValueError(
                "stop_token and early_stopping_condition must both be set or both be None. "
                f"Got stop_token={stop_token!r}, early_stopping_condition={early_stopping_condition!r}"
            )

        if template is None:
            template_path = os.path.join(os.path.dirname(__file__), "utils", "templates", "agentic_user_llm_simulator_template.txt")
            with open(template_path, "r") as f:
                template = f.read()

        super().__init__(
            model=model,
            template=template,
            max_try=max_try,
            generation_params=generation_params,
        )
        self.user_profile = user_profile
        self.scenario = scenario
        self.stop_token = stop_token
        self.early_stopping_condition = early_stopping_condition
        self.tools = tools or []

    def _create_error(
        self,
        message: str,
        attempts: int,
        last_error: Optional[str],
        logs: List[Dict[str, Any]],
    ) -> UserSimulatorError:
        """Create UserSimulatorError for user simulation failures."""
        return UserSimulatorError(
            message=message,
            attempts=attempts,
            last_error=last_error,
            logs=logs,
            component="user_simulator",
        )

    def __call__(
        self,
        conversation_history: List[Dict[str, str]],
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:  # type: ignore[override]
        """
        Generates a simulated user response with potential tool calls.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: (text_response, list_of_tool_calls)
        """
        return super().__call__(generation_params=generation_params, conversation_history=conversation_history)

    def _parse_output(self, output: str) -> Tuple[str, List[Dict[str, Any]]]:  # type: ignore[override]
        """
        Parses the raw JSON output from the model.
        """
        text_stripped = output.strip()
        if text_stripped.strip().startswith("```") and text_stripped.strip().endswith("```"):
            text_stripped = text_stripped.strip()[3:-3].strip()
            if text_stripped.startswith("json"):
                text_stripped = text_stripped[4:].strip()

        try:
            output_data = json.loads(text_stripped)
        except json.JSONDecodeError:
            # If strictly requiring JSON, raise. Or fallback to text only.
            # For agentic user, we expect JSON.
            raise

        text = output_data.get("text", "")
        tool_calls = output_data.get("tool_calls", [])
        return text, tool_calls

    def _fill_prompt_template(self, **kwargs) -> str:
        """
        Fills the prompt template with the message history, user profile, and tools.
        """
        conversation_history = kwargs.get("conversation_history", [])
        assert self.template is not None, "Template must be set"
        prompt = self.template

        # Format history into a string
        formatted_history = ""
        for message in conversation_history:
            formatted_history += f"{message['role']}: {message['content']}\n"

        # Build early stopping instructions
        early_stopping_instructions = ""
        if self.stop_token and self.early_stopping_condition:
            early_stopping_instructions = (
                f"\n### EARLY STOPPING\n"
                f"If the following condition is satisfied: {self.early_stopping_condition}\n"
                f"Then end your response with the token `{self.stop_token}` to signal that the conversation should end.\n"
            )

        # Build tool instructions
        tool_instructions = ""
        if self.tools:
            tool_instructions = "\n### TOOLS\nYou have access to the following tools to interact with your environment:\n"
            for tool in self.tools:
                tool_instructions += f"- {tool['name']}: {tool.get('description', '')}\n"
                if "inputs" in tool:
                    tool_instructions += f"  Inputs: {json.dumps(tool['inputs'])}\n"

            tool_instructions += (
                "\nTo use a tool, include a `tool_calls` field in your JSON response with a list of tool invocations.\n"
                'Example: {"text": "I\'ll check the signal.", "tool_calls": [{"name": "check_status", "arguments": {}}]}\n'
            )

        replacements = {
            "user_profile": json.dumps(self.user_profile, indent=2),
            "scenario": self.scenario,
            "conversation_history": formatted_history,
            "early_stopping_instructions": early_stopping_instructions,
            "tool_instructions": tool_instructions,
        }
        for k, v in replacements.items():
            prompt = prompt.replace("{{" + k + "}}", str(v))
        return prompt


class AgenticUser(User):
    """A user that can use tools to interact with the environment."""

    def __init__(
        self,
        name: str,
        model: ModelAdapter,
        user_profile: Dict[str, Any],
        scenario: str,
        tools: Optional[Dict[str, Callable]] = None,
        max_internal_steps: int = 5,
        **kwargs,
    ):
        """
        Args:
            tools: Dictionary of tools available to the user.
            max_internal_steps: Maximum number of tool execution loops per turn.
            **kwargs: Arguments passed to User.__init__
        """
        self.tools = tools or {}
        self.max_internal_steps = max_internal_steps

        # We initialize the parent, but we will replace the simulator
        super().__init__(name=name, model=model, user_profile=user_profile, scenario=scenario, **kwargs)

        # Generate tool definitions
        tool_definitions = self._generate_tool_definitions() if self.tools else None

        # Replace the standard simulator with AgenticUserLLMSimulator
        self.simulator = AgenticUserLLMSimulator(
            model=self.model,
            user_profile=self.user_profile,
            scenario=self.scenario,
            template=kwargs.get("template"),
            max_try=kwargs.get("max_try", 3),
            stop_token=kwargs.get("stop_token"),
            early_stopping_condition=kwargs.get("early_stopping_condition"),
            tools=tool_definitions,
        )

    def _generate_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generate tool definitions from the tools dictionary."""
        definitions = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            doc = func.__doc__ or ""

            inputs = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                param_type = "string"  # Default
                if param.annotation is not inspect.Parameter.empty:
                    if param.annotation is int:
                        param_type = "integer"
                    elif param.annotation is float:
                        param_type = "number"
                    elif param.annotation is bool:
                        param_type = "boolean"

                inputs[param_name] = {"type": param_type, "description": f"Parameter {param_name}"}

            definitions.append({"name": name, "description": doc.strip(), "inputs": inputs})
        return definitions

    def simulate_response(self, question: str) -> str:
        """Simulates a user response, potentially executing tools in a loop."""
        if self.is_done():
            return ""

        # Start with the current shared history
        self.messages.add_message("assistant", question)

        # Internal history for the ReAct loop (scratchpad)
        # We start with a copy of the shared conversation history
        internal_history = list(self.messages.to_list())

        start_time = time.time()
        log_entry: Dict[str, Any] = {"timestamp": datetime.now().isoformat(), "question": question, "status": "success", "internal_steps": []}

        final_response = ""
        steps = 0

        try:
            while steps < self.max_internal_steps:
                steps += 1

                # Call simulator
                # Note: AgenticUserLLMSimulator returns (text, tool_calls)
                text, tool_calls = self.simulator(conversation_history=internal_history)

                # Log the step
                step_log = {"step": steps, "thought": text, "tool_calls": tool_calls}

                if not tool_calls:
                    # No tools called, this is the final response
                    final_response = text
                    log_entry["internal_steps"].append(step_log)
                    break

                # Tools called - execute them
                # Append the thought to internal history
                # We use 'user' role for thoughts as it's the user speaking to themselves/environment
                internal_history.append({"role": "user", "content": text})

                tool_outputs = []
                for call in tool_calls:
                    name = call.get("name")  # type: ignore[union-attr]
                    args = call.get("arguments", {})  # type: ignore[union-attr]
                    result_str = ""

                    if name in self.tools:
                        try:
                            result = self.tools[name](**args)
                            result_str = str(result)
                            status = "success"
                        except Exception as e:
                            result_str = f"Error: {str(e)}"
                            status = "error"
                    else:
                        result_str = f"Error: Tool '{name}' not found"
                        status = "error"

                    tool_outputs.append({"name": name, "arguments": args, "result": result_str, "status": status})

                    # Append observation to internal history
                    # Format: Tool Output [name]: result
                    internal_history.append(
                        {
                            "role": "user",  # Using user role for simplicity, or we could use 'system'
                            "content": f"Tool Output [{name}]: {result_str}",
                        }
                    )

                step_log["tool_outputs"] = tool_outputs
                log_entry["internal_steps"].append(step_log)

            if not final_response and steps >= self.max_internal_steps:
                # Forced termination of loop
                final_response = "I need to stop checking things now."
                log_entry["status"] = "max_internal_steps_reached"

        except Exception as exc:
            log_entry["duration_seconds"] = time.time() - start_time
            log_entry["status"] = "error"
            log_entry["error"] = str(exc)
            self.logs.append(log_entry)
            raise

        log_entry["duration_seconds"] = time.time() - start_time
        log_entry["final_response"] = final_response[:200]
        self.logs.append(log_entry)

        # Clean stop token from final response
        _, clean_response = self._check_stop_token(final_response)

        # Update the shared history with the final text only
        self.messages.add_message("user", clean_response)
        self.increment_turn()

        return clean_response
