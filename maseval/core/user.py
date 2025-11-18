from .model import ModelAdapter
from .simulator import UserLLMSimulator
from .tracing import TraceableMixin
from .config import ConfigurableMixin
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import time
from .history import MessageHistory


class User(ABC, TraceableMixin, ConfigurableMixin):
    """A class representing a simulated user that can interact with a multi-agent system (MAS).

    The User class is designed to simulate human-like interaction by generating responses
    to questions from the MAS. It maintains a conversation history and uses a
    UserLLMSimulator to generate responses based on a predefined user profile and scenario.

    This class is abstract and requires a concrete implementation of the `get_tool` method,
    which should provide a tool specific to the MAS framework being used.

    The user only has access to the conversation history and does not see the full environment state,
    ensuring partial observability of environment and MAS.

    Attributes:
        name (str): The name of the user.
        model (ModelAdapter): The language model used for generating responses.
        user_profile (Dict[str, Any]): A dictionary describing the user's persona.
        scenario (str): A description of the task the user is trying to accomplish.
        simulator (UserLLMSimulator): The simulator instance used to generate responses.
        history (List[Dict[str, str]]): The conversation history between the user and the MAS.
    """

    def __init__(
        self,
        name: str,
        model: ModelAdapter,
        user_profile: Dict[str, Any],
        scenario: str,
        initial_prompt: str,
        template: Optional[str] = None,
        max_try: int = 3,
    ):
        """Initializes the User.

        Args:
            name (str): The name of the user.
            model (ModelAdapter): The language model to be used for generating responses.
            user_profile (Dict[str, Any]): A dictionary describing the user's persona,
                preferences, and other relevant information.
            scenario (str): A description of the situation or task the user is trying to
                accomplish.
            initial_prompt (str): The initial message or prompt that starts the conversation.
            template (Optional[str], optional): A custom prompt template for the user
                simulator. Defaults to None.
            max_try (int, optional): The maximum number of attempts for the simulator to
                generate a valid response. Defaults to 3.
        """
        self.name = name
        self.model = model
        self.user_profile = user_profile
        self.scenario = scenario
        self.simulator = UserLLMSimulator(
            model=self.model,
            user_profile=self.user_profile,
            scenario=self.scenario,
            template=template,
            max_try=max_try,
        )
        self.messages = MessageHistory([{"role": "user", "content": initial_prompt}])
        self.logs: list[Dict[str, Any]] = []

    def simulate_response(self, question: str) -> str:
        """Simulates a user response to a given question from the MAS.

        This method appends the agent's question to the conversation history,
        generates a response using the UserLLMSimulator, appends the simulated
        response to the history, and returns the response.

        Args:
            question (str): The question or message from the MAS to which the user should respond.

        Returns:
            str: The simulated user's response.
        """
        # Record the assistant prompt and ask simulator. MessageHistory is iterable
        # and can be converted to a list for the simulator.
        self.messages.add_message("assistant", question)
        start_time = time.time()
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "status": "success",
        }

        try:
            response = self.simulator(conversation_history=self.messages.to_list())
        except Exception as exc:  # pragma: no cover - propagate but log failure
            log_entry["duration_seconds"] = time.time() - start_time
            log_entry["status"] = "error"
            log_entry["error"] = str(exc)
            log_entry["error_type"] = type(exc).__name__
            self.logs.append(log_entry)
            raise

        log_entry["duration_seconds"] = time.time() - start_time
        log_entry["response_preview"] = self._summarize_response(response)
        self.logs.append(log_entry)

        self.messages.add_message("user", response)
        return response

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this user simulator.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - name: User name
            - profile: User profile data
            - message_count: Number of messages in conversation history
            - history: Full conversation history as list of messages
        """
        return {
            **super().gather_traces(),
            "name": self.name,
            "profile": self.user_profile,
            "message_count": len(self.messages),
            "messages": self.messages.to_list(),
            "logs": self.logs,
        }

    @staticmethod
    def _summarize_response(response: str) -> str:
        return response[:2000]

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this user simulator.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - name: User name
            - profile: User profile data
            - scenario: Task scenario description
        """
        return {
            **super().gather_config(),
            "name": self.name,
            "profile": self.user_profile,
            "scenario": self.scenario,
        }

    @abstractmethod
    def get_tool(self):
        """Returns a tool that can be used by the MAS to interact with the user.

        This method must be implemented by a concrete subclass to provide a tool
        that is compatible with the specific MAS framework being used. The tool
        should wrap the `simulate_response` method to allow the MAS to get
        user input.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
