"""OpenAI and OpenAI-compatible model adapter.

Requires openai to be installed:
    pip install maseval[openai]
"""

from typing import Any, Optional, Dict
import json

from maseval.core.model import ModelAdapter


class OpenAIModelAdapter(ModelAdapter):
    """Adapter for OpenAI-compatible models (openai or OpenAI-compatible servers).

    The `client` can be a callable returning a string, or an object with a
    `complete`/`chat`/`create` method. This adapter tries common method names.

    Requires openai to be installed.
    """

    def __init__(
        self,
        client: Any,
        model_id: Optional[str] = None,
        default_generation_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._client = client
        self._model_id = model_id or getattr(client, "model_id", "openai:unknown")
        self._default_generation_params = default_generation_params or {}

    @property
    def model_id(self) -> str:
        return self._model_id

    def _extract_text(self, resp: Any) -> str:
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            # common OpenAI shapes
            if "choices" in resp and resp["choices"]:
                choice = resp["choices"][0]
                # chat-like
                if "message" in choice and isinstance(choice["message"], dict):
                    return choice["message"].get("content", "")
                # completion-like
                return choice.get("text", "")
            # fallback
            return json.dumps(resp)
        return str(resp)

    def _generate_impl(self, prompt: str, generation_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        # try common call patterns
        # 1) client(prompt)
        try:
            resp = self._client(prompt, **params)
        except TypeError:
            # 2) client.create / client.complete / client.chat
            for meth in ("create", "complete", "chat", "generate"):
                if hasattr(self._client, meth):
                    func = getattr(self._client, meth)
                    try:
                        resp = func(prompt, **params)
                        break
                    except TypeError:
                        resp = func(prompt)
                        break
            else:
                # last resort: call without kwargs
                resp = self._client(prompt)

        return self._extract_text(resp)

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this OpenAI model adapter.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - model_id: Model identifier
            - adapter_type: OpenAIModelAdapter
            - default_generation_params: Default parameters used for generation (temperature, top_p, etc.)
            - client_type: Type name of the underlying client
            - client_config: OpenAI client configuration affecting model behavior:
                - timeout: Request timeout settings (affects latency)
                - max_retries: Maximum number of retry attempts (affects reliability)
        """
        base_config = super().gather_config()
        base_config.update(
            {
                "default_generation_params": self._default_generation_params,
                "client_type": type(self._client).__name__,
            }
        )

        # Extract OpenAI client configuration that affects model behavior
        client_config = {}

        # Timeout configuration (affects latency and reliability)
        if hasattr(self._client, "timeout"):
            timeout = self._client.timeout
            # Handle both httpx.Timeout objects and simple floats
            if hasattr(timeout, "connect"):
                client_config["timeout"] = {
                    "connect": timeout.connect,
                    "read": timeout.read,
                    "write": timeout.write,
                    "pool": timeout.pool,
                }
            else:
                client_config["timeout"] = timeout

        # Max retries (affects reliability and latency)
        if hasattr(self._client, "max_retries"):
            client_config["max_retries"] = self._client.max_retries

        if client_config:
            base_config["client_config"] = client_config

        return base_config
