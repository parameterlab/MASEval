"""Inference model adapters for various providers.

This package contains concrete implementations of ModelAdapter for different
inference providers. Each adapter requires the corresponding optional dependency.
"""

__all__ = []

# Conditionally import google-genai adapter
try:
    from .google_genai import GoogleGenAIModelAdapter  # noqa: F401

    __all__.append("GoogleGenAIModelAdapter")
except ImportError:
    pass

# Conditionally import OpenAI adapter
try:
    from .openai import OpenAIModelAdapter  # noqa: F401

    __all__.append("OpenAIModelAdapter")
except ImportError:
    pass

# Conditionally import HuggingFace adapter
try:
    from .huggingface import HuggingFaceModelAdapter  # noqa: F401

    __all__.append("HuggingFaceModelAdapter")
except ImportError:
    pass

# Conditionally import LiteLLM adapter
try:
    from .litellm import LiteLLMModelAdapter  # noqa: F401

    __all__.append("LiteLLMModelAdapter")
except ImportError:
    pass
