"""Agent framework adapters.

This package contains adapters for different agent frameworks.
"""

__all__ = []

# Conditionally import smolagents
try:
    from .smolagents import SmolAgentAdapter, SmolAgentUser  # noqa: F401

    __all__.extend(["SmolAgentAdapter", "SmolAgentUser"])
except ImportError:
    pass

# Conditionally import langgraph
try:
    from .langgraph import LangGraphAgentAdapter, LangGraphUser  # noqa: F401

    __all__.extend(["LangGraphAgentAdapter", "LangGraphUser"])
except ImportError:
    pass
