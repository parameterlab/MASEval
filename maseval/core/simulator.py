from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime
from .model import ModelAdapter
from .tracing import TraceableMixin
import uuid
from enum import Enum


class LLMSimulator(ABC, TraceableMixin):
    """
    A base class for simulators that use an LLM.
    """

    def __init__(
        self,
        model: ModelAdapter,
        template: Optional[str] = None,
        max_try: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the LLMSimulator.

        Args:
            model (ModelAdapter): The language model to use for generation.
            template (str, optional): A prompt template.
            max_try (int, optional): Maximum number of model calls to attempt. Defaults to 3.
            generation_params (Dict[str, Any], optional): Default generation parameters for the model. This overwrites the ModelAdapter's defaults if provided.
                Both can be overridden at call time. Defaults to None.
        """
        self.model = model
        self.template = template
        self.max_try = max_try
        self.generation_params = generation_params or {}
        # canonical structured trace of model invocation attempts and results
        # Each simulator call is assigned a request id; each individual
        # attempt (successful or failed) is appended as a separate entry.
        # Entry schema: {id, timestamp, input, raw_output, parsed_output, status}
        self.logs: list[dict[str, Any]] = []

    def __call__(self, generation_params: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Generates a simulated output.
        """
        prompt = self._fill_prompt_template(**kwargs)

        request_id = str(uuid.uuid4())
        attempts = 0
        parsed_result = None

        # merging of LLM default and call-time generation params done here, so subclasses
        # can just call super().__call__(generation_params=...) and have it handled
        generation_params = self.generation_params | (generation_params or {})

        # For each attempt, append a separate history entry with the request id.
        while attempts < self.max_try and parsed_result is None:
            attempts += 1
            raw_output = None
            entry = {
                "id": request_id,
                "timestamp": datetime.now().isoformat(),
                "input": kwargs,
                "prompt": prompt,
                "generation_params": generation_params,
                "raw_output": None,
                "parsed_output": None,
                "status": None,
                "error": None,
            }

            try:
                raw_output = self.model.generate(prompt, generation_params=generation_params)
                entry["raw_output"] = raw_output
            except Exception as e:
                # record model call error attempt by updating the pre-created entry
                entry["raw_output"] = None
                entry["status"] = SimulatorCallStatus.ModelCallError.value
                entry["error"] = str(e)
                # rich.print(
                #     f"[yellow]Warning:[/yellow] Attempt {attempts} failed to call model: {e}"
                #     + (" Retrying..." if attempts < self.max_try else "")
                # )
                self.logs.append(entry)
                continue

            # try parsing the raw output
            try:
                parsed_result = self._parse_output(raw_output)
                # update the existing entry with successful result
                entry["parsed_output"] = parsed_result
                entry["status"] = SimulatorCallStatus.Successful.value

            except (json.JSONDecodeError, AttributeError) as e:
                # update the existing entry with parsing error info
                entry["status"] = SimulatorCallStatus.ModelParsingError.value
                entry["error"] = str(e)
                # rich.print(
                #     f"[yellow]Warning:[/yellow] Attempt {attempts} failed to parse LLM output: {e}"
                #     + (" Retrying..." if attempts < self.max_try else "")
                # )
            self.logs.append(entry)

        return parsed_result if parsed_result is not None else self._get_error_result()

    def _call_model_and_parse(self, prompt: str) -> Any:
        """
        Calls the model with the given prompt and attempts to parse the output.
        """
        raw_output = self.model.generate(prompt)
        return self._parse_output(raw_output)

    @abstractmethod
    def _fill_prompt_template(self, **kwargs) -> str:
        """
        Fills the prompt template with the provided arguments.
        """
        pass

    @abstractmethod
    def _parse_output(self, output: str) -> Any:
        """
        Parses the raw output from the model.
        """
        pass

    @abstractmethod
    def _get_error_result(self) -> Any:
        """
        Returns the error result when parsing fails.
        """
        pass

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this simulator.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - simulator_type: The specific simulator class
            - total_calls: Number of simulation attempts
            - successful_calls: Number of successful simulations
            - failed_calls: Number of failed attempts
            - history: Complete history of all simulation attempts with timestamps,
                      inputs, outputs, status, and error messages
        """
        total_calls = len(self.logs)
        successful = sum(1 for entry in self.logs if entry.get("status") == SimulatorCallStatus.Successful.value)
        failed = total_calls - successful

        return {
            **super().gather_traces(),
            "simulator_type": self.__class__.__name__,
            "total_calls": total_calls,
            "successful_calls": successful,
            "failed_calls": failed,
            "logs": self.logs,
        }


class SimulatorCallStatus(Enum):
    ModelCallError = "ModelCallError"
    ModelParsingError = "ModelParsingError"
    Successful = "Successful"


class ToolLLMSimulator(LLMSimulator):
    """
    A simulator that uses an LLM to generate plausible tool outputs.
    """

    def __init__(
        self,
        model: ModelAdapter,
        tool_name: str,
        tool_description: str,
        tool_inputs: Dict[str, Any],
        template: Optional[str] = None,
        max_try: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the ToolLLMSimulator.

        Args:
            model (ModelAdapter): The language model to use for generation (must have a `generate` method).
            tool_name (str): The name of the tool.
            tool_description (str): The description of the tool.
            tool_inputs (Dict[str, Any]): The schema for the tool's arguments.
            template (str, optional): a prompt template. Defaults to the one in the library. See `maseval.utils.templates.tool_llm_simulator_template.txt`.
                The template should use double curly braces for placeholders. Should contain placeholders for `name`, `description`, `inputs`, and `input_value_dict`.
            max_try (int, optional): Maximum number of model calls to attempt if output parsing json output fails. Defaults to 3.
            generation_params (Dict[str, Any], optional): Default generation parameters for the model. This overwrites the ModelAdapter's defaults if provided.
                Both can be overridden at call time. Defaults to None.
        """
        if template is None:
            template_path = os.path.join(os.path.dirname(__file__), "utils", "templates", "tool_llm_simulator_template.txt")
            with open(template_path, "r") as f:
                template = f.read()
        super().__init__(model, template, max_try)
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_inputs = tool_inputs
        self.generation_params = generation_params or {}

    def __call__(self, generation_params: Optional[Dict[str, Any]] = None, **actual_inputs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return super().__call__(generation_params=generation_params, **actual_inputs)

    def _parse_output(self, output: str) -> tuple[str, Dict[str, Any]]:
        text_stripped = output.strip()
        # if starts and stop with ```json ... ``` or ``` ... ```, strip those
        if text_stripped.strip().startswith("```") and text_stripped.strip().endswith("```"):
            text_stripped = text_stripped.strip()[3:-3].strip()
            if text_stripped.startswith("json"):
                text_stripped = text_stripped[4:].strip()
        # The model is expected to return a single JSON object string.
        output_data = json.loads(text_stripped)
        return output_data.get("text", ""), output_data.get("details", {})

    def _fill_prompt_template(self, **kwargs) -> str:
        """
        Fills the prompt template with the tool's data and input values.
        """
        assert self.template is not None, "Template must be set"
        prompt = self.template
        replacements = {
            "name": str(self.tool_name),
            "description": str(self.tool_description),
            "inputs": json.dumps(self.tool_inputs, indent=2),
            "input_value_dict": json.dumps(kwargs, indent=2),
        }
        for k, v in replacements.items():
            prompt = prompt.replace("{{" + k + "}}", v)
        return prompt

    def _get_error_result(self) -> tuple[str, Dict[str, Any]]:
        return "Error: Failed to decode LLM output after multiple attempts.", {"raw_output": None}


class UserLLMSimulator(LLMSimulator):
    """
    A simulator that uses an LLM to act as the user.
    """

    def __init__(
        self,
        model: ModelAdapter,
        user_profile: Dict[str, str],
        scenario: str,
        template: Optional[str] = None,
        max_try: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the UserLLMSimulator.

        Args:
            model (ModelAdapter): The language model to use for generation.
            user_profile (Dict[str, str]): A dictionary containing the user's profile.
            scenario (str): The scenario for the user.
            template (str, optional): A prompt template. Defaults to the one in the library. See `maseval.utils.templates.user_llm_simulator_template.txt`.
            max_try (int, optional): Maximum number of model calls to attempt. Defaults to 3.
            generation_params (Dict[str, Any], optional): Default generation parameters for the model. This overwrites the ModelAdapter's defaults if provided.
                Both can be overridden at call time. Defaults to None.
        """
        if template is None:
            template_path = os.path.join(os.path.dirname(__file__), "utils", "templates", "user_llm_simulator_template.txt")
            with open(template_path, "r") as f:
                template = f.read()
        super().__init__(model, template, max_try)
        self.user_profile = user_profile
        self.scenario = scenario
        self.generation_params = generation_params or {}

    def __call__(
        self,
        conversation_history: List[Dict[str, str]],
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generates a simulated user response.

        Args:
            conversation_history: The history of the conversation.
            generation_params: Optional generation parameters for LLM to override the defaults.

        Returns:
            The simulated user response string.
        """
        return super().__call__(generation_params=generation_params, conversation_history=conversation_history)

    def _parse_output(self, output: str) -> str:
        """
        Parses the raw JSON output from the model.
        """
        text_stripped = output.strip()
        if text_stripped.strip().startswith("```") and text_stripped.strip().endswith("```"):
            text_stripped = text_stripped.strip()[3:-3].strip()
            if text_stripped.startswith("json"):
                text_stripped = text_stripped[4:].strip()

        output_data = json.loads(text_stripped)
        return output_data.get("text", "")

    def _fill_prompt_template(self, **kwargs) -> str:
        """
        Fills the prompt template with the message history and user profile.
        """
        conversation_history = kwargs.get("conversation_history", [])
        assert self.template is not None, "Template must be set"
        prompt = self.template

        # Format history into a string
        formatted_history = ""
        for message in conversation_history:
            formatted_history += f"{message['role']}: {message['content']}\n"

        replacements = {
            "user_profile": json.dumps(self.user_profile, indent=2),
            "scenario": self.scenario,
            "conversation_history": formatted_history,
        }
        for k, v in replacements.items():
            prompt = prompt.replace("{{" + k + "}}", str(v))
        return prompt

    def _get_error_result(self) -> str:
        return "Error: Failed to get a response from the user simulator."
