"""Google Generative AI model adapter.

Requires google-genai to be installed:
    pip install maseval[google-genai]
"""

from typing import Any, Optional, Dict
import json

from maseval.core.model import ModelAdapter


class GoogleGenAIModelAdapter(ModelAdapter):
    """Adapter for Google Generative AI.

    The `client` may be a callable that accepts the prompt and returns a dict-like
    response, or a client object with a `generate` method. The adapter will try
    to normalize the response to a text string.

    Requires google-genai to be installed.
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        default_generation_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._client = client
        self._model_id = model_id
        self._default_generation_params = default_generation_params or {}

    @property
    def model_id(self) -> str:
        return self._model_id

    def _extract_text(self, response: Any) -> str:
        # Normalize a few common shapes
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            # google generative responses often have `candidates` or `output` fields
            if "candidates" in response and response["candidates"]:
                return response["candidates"][0].get("content", "")
            if "output" in response and isinstance(response["output"], list) and response["output"]:
                # some wrappers return a list of text chunks
                first = response["output"][0]
                if isinstance(first, dict):
                    return first.get("content", "")
                return str(first)
            # fallback to stringifying
            return json.dumps(response)
        return str(response)

    def _generate_impl(self, prompt: str, generation_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        from google import genai  # Lazy import

        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        generation_config = genai.types.GenerateContentConfig(**params) if params else None

        # Call client
        response = self._client.models.generate_content(model=self.model_id, contents=prompt, config=generation_config)
        return response.text

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this Google GenAI model adapter.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - model_id: Model identifier
            - adapter_type: GoogleGenAIModelAdapter
            - default_generation_params: Default parameters used for generation (temperature, top_p, etc.)
            - client_type: Type name of the underlying client
        """
        base_config = super().gather_config()
        base_config.update(
            {
                "default_generation_params": self._default_generation_params,
                "client_type": type(self._client).__name__,
            }
        )

        return base_config
