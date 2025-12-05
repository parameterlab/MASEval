from .model import ModelAdapter
from .simulator import UserLLMSimulator
from .tracing import TraceableMixin
from .config import ConfigurableMixin
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import time
from .history import MessageHistory


class TerminationReason(Enum):
    """Reason why user interaction terminated."""

    NOT_TERMINATED = "not_terminated"
    MAX_TURNS = "max_turns"
    USER_TERMINATED = "user_terminated"  # stop token detected
    MAX_TURNS_AND_USER_TERMINATED = "max_turns_and_user_terminated"  # both conditions met


class User(ABC, TraceableMixin, ConfigurableMixin):
    """A class representing a simulated user that can interact with a multi-agent system (MAS).

    The User class is designed to simulate human-like interaction by generating responses
    to questions from the MAS. It maintains a conversation history and uses a
    UserLLMSimulator to generate responses based on a predefined user profile and scenario.

    This class is abstract and requires a concrete implementation of the `get_tool` method,
    which should provide a tool specific to the MAS framework being used.

    The user only has access to the conversation history and does not see the full environment state,
    ensuring partial observability of environment and MAS.

    Multi-Turn Interaction:
        By default, users support single-turn interaction (max_turns=1). For benchmarks
        that require multiple agent-user exchanges, set max_turns > 1.

    Early Stopping (User Satisfaction):
        For benchmarks where the termination criterion is "user satisfaction" rather than
        a fixed number of turns, configure a stop_token. When the LLM-generated user
        response contains this token, is_done() returns True, ending the interaction early.

        Example: MACS benchmark uses "</stop>" to signal the user is satisfied with the
        agent's response, allowing natural conversation endings before max_turns.

    Attributes:
        name (str): The name of the user.
        model (ModelAdapter): The language model used for generating responses.
        user_profile (Dict[str, Any]): A dictionary describing the user's persona.
        scenario (str): A description of the task the user is trying to accomplish.
        simulator (UserLLMSimulator): The simulator instance used to generate responses.
        messages (MessageHistory): The conversation history between the user and the MAS.
        max_turns (int): Maximum number of user response turns.
        stop_token (Optional[str]): Token that triggers early stopping when detected.
        early_stopping_condition (Optional[str]): Description of when to emit the stop token.
    """

    def __init__(
        self,
        name: str,
        model: ModelAdapter,
        user_profile: Dict[str, Any],
        scenario: str,
        initial_query: Optional[str] = None,
        template: Optional[str] = None,
        max_try: int = 3,
        max_turns: int = 1,
        stop_token: Optional[str] = None,
        early_stopping_condition: Optional[str] = None,
    ):
        """Initializes the User.

        Args:
            name (str): The name of the user.
            model (ModelAdapter): The language model to be used for generating responses.
            user_profile (Dict[str, Any]): A dictionary describing the user's persona,
                preferences, and other relevant information.
            scenario (str): A description of the situation or task the user is trying to
                accomplish.
            initial_query (Optional[str], optional): A pre-set query to start the
                conversation. If provided, it becomes the first user message. If None,
                call get_initial_query() to generate one from the model based on the
                user profile and scenario. Defaults to None.
            template (Optional[str], optional): A custom prompt template for the user
                simulator. Defaults to None.
            max_try (int, optional): The maximum number of attempts for the simulator to
                generate a valid response. Defaults to 3.
            max_turns (int, optional): Maximum number of user messages in the
                conversation. Each user message counts as one turn, including the
                initial_query. Use max_turns=1 for single-turn benchmarks, or higher
                values for multi-turn interaction. Defaults to 1.
            stop_token (Optional[str], optional): Token that signals user satisfaction,
                enabling early termination. When the user's LLM-generated response contains
                this token, is_done() returns True regardless of remaining turns. Use this
                for benchmarks where termination is based on user satisfaction rather than
                a fixed turn count. The token is stripped from the response. Defaults to
                None (early stopping disabled).
            early_stopping_condition (Optional[str], optional): A description of when the
                user should stop the conversation (e.g., "all goals have been accomplished").
                Used with stop_token to instruct the LLM when to emit the stop token.
                Must be provided if stop_token is set. Defaults to None.

        Raises:
            ValueError: If only one of stop_token or early_stopping_condition is provided.
        """
        # Validate early stopping configuration
        if (stop_token is None) != (early_stopping_condition is None):
            raise ValueError(
                "stop_token and early_stopping_condition must both be set or both be None. "
                f"Got stop_token={stop_token!r}, early_stopping_condition={early_stopping_condition!r}"
            )

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
            stop_token=stop_token,
            early_stopping_condition=early_stopping_condition,
        )
        # Initialize message history - empty or with initial query
        if initial_query is not None:
            self.messages = MessageHistory([{"role": "user", "content": initial_query}])
            self._initial_turn_count = 1  # Initial query counts as first turn
        else:
            self.messages = MessageHistory()
            self._initial_turn_count = 0
        self.logs: list[Dict[str, Any]] = []

        # Multi-turn configuration
        self.max_turns = max_turns
        self.stop_token = stop_token
        self.early_stopping_condition = early_stopping_condition
        self._turn_count = self._initial_turn_count
        self._stopped = False

    def simulate_response(self, question: str) -> str:
        """Simulates a user response to a given question from the MAS.

        This method appends the agent's question to the conversation history,
        generates a response using the UserLLMSimulator, appends the simulated
        response to the history, and returns the response.

        If the user is already done (max_turns reached or stop_token detected),
        returns an empty string without making an LLM call. If a stop_token is
        detected in the response, triggers early stopping.

        Args:
            question (str): The question or message from the MAS to which the user should respond.

        Returns:
            str: The simulated user's response, or empty string if done.
        """
        # Check if already done - saves LLM call
        if self.is_done():
            return ""

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

        # Check for stop token and clean response if needed
        _, clean_response = self._check_stop_token(response)

        self.messages.add_message("user", clean_response)
        self.increment_turn()
        return clean_response

    def get_initial_query(self) -> str:
        """Get the initial query for the conversation.

        If an initial_query was provided at construction, returns it.
        Otherwise, generates one using the LLM simulator based on the user's
        profile and scenario.

        This method:
        - Returns the existing initial query if one was provided
        - Or calls the LLM simulator to generate one
        - Ensures the query is in the message history
        - Counts the initial query as the first turn

        Returns:
            str: The initial query (either pre-set or LLM-generated).

        Raises:
            RuntimeError: If called after conversation has progressed beyond
                the initial message.
        """
        # If we already have an initial query in messages, return it
        if len(self.messages) > 0:
            first_message = self.messages[0]
            if first_message.get("role") == "user":
                return first_message.get("content", "")
            raise RuntimeError("Cannot get initial query: conversation has progressed. Use simulate_response() for subsequent turns.")

        # Generate initial query via LLM
        start_time = time.time()
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "question": "[initial_query]",
            "status": "success",
        }

        try:
            response = self.simulator(conversation_history=[])
        except Exception as exc:  # pragma: no cover
            log_entry["duration_seconds"] = time.time() - start_time
            log_entry["status"] = "error"
            log_entry["error"] = str(exc)
            log_entry["error_type"] = type(exc).__name__
            self.logs.append(log_entry)
            raise

        log_entry["duration_seconds"] = time.time() - start_time
        log_entry["response_preview"] = self._summarize_response(response)
        self.logs.append(log_entry)

        # Check for stop token (user might be immediately satisfied with scenario)
        _, clean_response = self._check_stop_token(response)

        # Add as initial user message and count as first turn
        self.messages.add_message("user", clean_response)
        self.increment_turn()
        return clean_response

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
            "termination_reason": self.termination_reason.value,
        }

    @staticmethod
    def _summarize_response(response: str) -> str:
        return response[:2000]

    @property
    def termination_reason(self) -> TerminationReason:
        """Get the reason why the user interaction terminated.

        Returns:
            TerminationReason indicating why is_done() returns True,
            or NOT_TERMINATED if the interaction is still ongoing.
        """
        max_turns_reached = self._turn_count >= self.max_turns
        user_terminated = self._stopped

        if max_turns_reached and user_terminated:
            return TerminationReason.MAX_TURNS_AND_USER_TERMINATED
        elif max_turns_reached:
            return TerminationReason.MAX_TURNS
        elif user_terminated:
            return TerminationReason.USER_TERMINATED
        else:
            return TerminationReason.NOT_TERMINATED

    def is_done(self) -> bool:
        """Check if the user interaction should end.

        The base implementation checks:
        1. If max_turns has been reached
        2. If the user previously indicated termination (via stop_token)

        Subclasses can override to add custom termination logic (e.g., LLM-based
        satisfaction checks) by calling super().is_done() first.

        Returns:
            True if the user is done interacting, False to continue.
        """
        return self.termination_reason != TerminationReason.NOT_TERMINATED

    def _check_stop_token(self, response: str) -> tuple[bool, str]:
        """Check if response contains stop token and clean it up.

        Args:
            response: The user's response to check.

        Returns:
            Tuple of (should_stop, cleaned_response).
        """
        if self.stop_token and self.stop_token.lower() in response.lower():
            self._stopped = True
            # Remove the stop token from the response
            cleaned = response.replace(self.stop_token, "").replace(self.stop_token.lower(), "").strip()
            return True, cleaned if cleaned else "Thank you, that's all I needed!"
        return False, response

    def increment_turn(self) -> None:
        """Increment the turn counter.

        Call this after recording a user response in the message history.
        """
        self._turn_count += 1

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this user simulator.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - name: User name
            - profile: User profile data
            - scenario: Task scenario description
            - max_turns: Maximum interaction turns
            - stop_token: Early stopping token (if configured)
        """
        return {
            **super().gather_config(),
            "name": self.name,
            "profile": self.user_profile,
            "scenario": self.scenario,
            "max_turns": self.max_turns,
            "stop_token": self.stop_token,
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
