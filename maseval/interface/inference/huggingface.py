"""HuggingFace model adapter.

Requires transformers to be installed:
    pip install maseval[transformers]
"""

from typing import Any, Callable, Optional, Dict

from maseval.core.model import ModelAdapter


class HuggingFaceModelAdapter(ModelAdapter):
    """Adapter for HuggingFace-style generation.

    This adapter accepts either a `callable` that takes `prompt` and returns
    text, or a thin `pipeline`-like object with a `__call__`.

    Requires transformers to be installed.
    """

    def __init__(
        self,
        model: Callable[[str], str],
        model_id: Optional[str] = None,
        default_generation_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._model = model
        self._model_id = model_id or getattr(model, "name_or_path", "huggingface:unknown")
        self._default_generation_params = default_generation_params or {}

    @property
    def model_id(self) -> str:
        return self._model_id

    def _generate_impl(self, prompt: str, generation_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        # Merge default params and call-time params; forward to underlying callable
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        # allow explicit kwargs to override
        params.update(kwargs)
        try:
            return self._model(prompt, **params)
        except TypeError:
            # fall back to calling without kwargs
            return self._model(prompt)

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this HuggingFace model adapter.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - model_id: Model identifier
            - adapter_type: HuggingFaceModelAdapter
            - default_generation_params: Default parameters used for generation (temperature, top_p, max_length, etc.)
            - callable_type: Type name of the underlying callable
            - pipeline_config: Pipeline configuration affecting model behavior:
                - task: Pipeline task type (e.g., text-generation, text-classification)
                - device: Device (cpu, cuda, etc.)
                - framework: Framework (pt for PyTorch, tf for TensorFlow)
        """
        base_config = super().gather_config()
        base_config.update(
            {
                "default_generation_params": self._default_generation_params,
                "callable_type": type(self._model).__name__,
            }
        )

        # Extract pipeline configuration that affects model behavior
        pipeline_config = {}

        # Core pipeline attributes
        if hasattr(self._model, "task"):
            pipeline_config["task"] = self._model.task

        if hasattr(self._model, "device"):
            device = self._model.device
            # Convert device to string representation
            pipeline_config["device"] = str(device) if device is not None else None

        if hasattr(self._model, "framework"):
            pipeline_config["framework"] = self._model.framework

        if pipeline_config:
            base_config["pipeline_config"] = pipeline_config

        return base_config
