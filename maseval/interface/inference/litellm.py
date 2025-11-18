"""LiteLLM model adapter.

LiteLLM provides a unified interface for 100+ LLM APIs.

Requires litellm to be installed:
    pip install maseval[litellm]
"""

from typing import Any, Optional, Dict

from maseval.core.model import ModelAdapter


class LiteLLMModelAdapter(ModelAdapter):
    """Adapter for LiteLLM unified interface.

    LiteLLM provides a consistent API for calling multiple LLM providers
    (OpenAI, Anthropic, Cohere, Azure, AWS Bedrock, etc.) using the same
    interface.

    Requires litellm to be installed.

    Example:
        ```python
        from maseval.interface.inference import LiteLLMModelAdapter

        # OpenAI
        model = LiteLLMModelAdapter(model_id="gpt-4")

        # Anthropic
        model = LiteLLMModelAdapter(model_id="claude-3-opus-20240229")

        # Azure OpenAI
        model = LiteLLMModelAdapter(
            model_id="azure/gpt-4",
            default_generation_params={"api_base": "..."}
        )

        # AWS Bedrock
        model = LiteLLMModelAdapter(model_id="bedrock/anthropic.claude-v2")
        ```
    """

    def __init__(
        self,
        model_id: str,
        default_generation_params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """Initialize LiteLLM model adapter.

        Args:
            model_id: The model identifier in LiteLLM format (e.g., "gpt-4",
                "claude-3-opus-20240229", "azure/gpt-4", "bedrock/...").
                See: https://docs.litellm.ai/docs/providers
            default_generation_params: Default parameters passed to litellm.completion()
                (e.g., temperature, max_tokens, top_p, etc.)
            api_key: Optional API key. If not provided, LiteLLM will use environment
                variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
            api_base: Optional API base URL for custom endpoints
        """
        super().__init__()
        self._model_id = model_id
        self._default_generation_params = default_generation_params or {}
        self._api_key = api_key
        self._api_base = api_base

    @property
    def model_id(self) -> str:
        return self._model_id

    def _generate_impl(self, prompt: str, generation_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        """Generate text using LiteLLM.

        Args:
            prompt: The input prompt
            generation_params: Optional generation parameters (temperature, max_tokens, etc.)
            **kwargs: Additional LiteLLM-specific parameters

        Returns:
            Generated text string
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError("LiteLLM is not installed. Install it with: pip install maseval[litellm] or pip install litellm") from e

        # Merge parameters
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        # Add API credentials if provided
        if self._api_key:
            params["api_key"] = self._api_key
        if self._api_base:
            params["api_base"] = self._api_base

        # LiteLLM expects messages format
        messages = [{"role": "user", "content": prompt}]

        # Call LiteLLM
        response = litellm.completion(model=self._model_id, messages=messages, **params)

        # Extract text from response
        # LiteLLM returns a ModelResponse object similar to OpenAI's format
        content = response.choices[0].message.content
        return content if content is not None else ""

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this LiteLLM model adapter.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - model_id: Model identifier
            - adapter_type: LiteLLMModelAdapter
            - default_generation_params: Default parameters used for generation (temperature, top_p, etc.)
            - litellm_global_config: LiteLLM global configuration affecting model behavior:
                - num_retries: Number of retry attempts (affects reliability)
                - drop_params: Whether to drop unsupported params (affects behavior)
                - verbose: Debug logging enabled (affects observability)
        """
        base_config = super().gather_config()
        base_config.update(
            {
                "default_generation_params": self._default_generation_params,
            }
        )

        # Extract LiteLLM global configuration that affects model behavior
        try:
            import litellm

            litellm_config = {}

            # Retry configuration (affects reliability and latency)
            if hasattr(litellm, "num_retries"):
                litellm_config["num_retries"] = litellm.num_retries

            # Drop params (affects model behavior with unsupported parameters)
            if hasattr(litellm, "drop_params"):
                litellm_config["drop_params"] = litellm.drop_params

            # Verbose mode (affects logging and debugging)
            if hasattr(litellm, "verbose"):
                litellm_config["verbose"] = litellm.verbose

            if litellm_config:
                base_config["litellm_global_config"] = litellm_config

        except ImportError:
            pass

        return base_config
