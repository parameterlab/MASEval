"""Core model adapter abstractions.

Concrete implementations for specific inference providers are in:
    maseval.interface.inference
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from datetime import datetime
import time

from .tracing import TraceableMixin
from .config import ConfigurableMixin


class ModelAdapter(ABC, TraceableMixin, ConfigurableMixin):
    """Abstract base class for model adapters.

    Concrete implementations must provide a `generate` method that accepts a
    prompt string and returns the model's text output. They should also expose
    a `model_id` property identifying the underlying model.

    This class automatically tracks all generation calls for tracing and evaluation.

    See maseval.interface.inference for concrete implementations:
    - GoogleGenAIModelAdapter
    - OpenAIModelAdapter
    - HuggingFaceModelAdapter
    """

    def __init__(self):
        """Initialize the model adapter with call tracing."""
        super().__init__()
        self.logs: list[dict[str, Any]] = []

    @property
    @abstractmethod
    def model_id(self) -> str:
        """A string identifier for the underlying model."""

    def generate(self, prompt: str, generation_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        """Generate text from the model with automatic tracing.

        This method wraps the actual generation logic to track timing,
        parameters, and errors for later evaluation.

        Args:
            prompt: The input prompt
            generation_params: Optional generation parameters
            **kwargs: Additional provider-specific arguments

        Returns:
            The model output as a string

        Raises:
            Exception: Any exception from the underlying model is logged and re-raised
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        try:
            result = self._generate_impl(prompt, generation_params, **kwargs)
            duration = time.time() - start_time

            self.logs.append(
                {
                    "timestamp": timestamp,
                    "prompt_length": len(prompt),
                    "response_length": len(result) if result else 0,
                    "duration_seconds": duration,
                    "status": "success",
                    "generation_params": generation_params or {},
                    "kwargs": {k: str(v) for k, v in kwargs.items()},  # Serialize for JSON
                }
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            self.logs.append(
                {
                    "timestamp": timestamp,
                    "prompt_length": len(prompt),
                    "duration_seconds": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "generation_params": generation_params or {},
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                }
            )

            raise

    @abstractmethod
    def _generate_impl(self, prompt: str, generation_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        """Internal generation implementation to be overridden by subclasses.

        Args:
            prompt: The input prompt
            generation_params: Optional generation parameters
            **kwargs: Additional provider-specific arguments

        Returns:
            The model output as a string
        """

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this model adapter.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - model_id: Model identifier
            - total_calls: Number of generation calls
            - successful_calls: Number of successful calls
            - failed_calls: Number of failed calls
            - total_duration_seconds: Total time spent generating
            - average_duration_seconds: Average time per call
            - logs: List of all individual call records with timestamps, durations, and parameters
        """
        total_calls = len(self.logs)
        successful_calls = sum(1 for call in self.logs if call["status"] == "success")
        failed_calls = total_calls - successful_calls
        total_duration = sum(call["duration_seconds"] for call in self.logs)
        avg_duration = total_duration / total_calls if total_calls > 0 else 0

        return {
            **super().gather_traces(),
            "model_id": self.model_id,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": avg_duration,
            "logs": self.logs,
        }

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this model adapter.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - model_id: Model identifier
            - adapter_type: The specific adapter class (e.g., OpenAIModelAdapter)
        """
        return {
            **super().gather_config(),
            "model_id": self.model_id,
            "adapter_type": type(self).__name__,
        }
