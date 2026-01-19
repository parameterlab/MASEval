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

# Conditionally import llamaindex
try:
    from .llamaindex import LlamaIndexAgentAdapter, LlamaIndexUser  # noqa: F401

    __all__.extend(["LlamaIndexAgentAdapter", "LlamaIndexUser"])
except ImportError:
    pass

# Conditionally import camel
try:
    from .camel import (  # noqa: F401
        CamelAgentAdapter,
        CamelUser,
        CamelAgentUser,
        camel_role_playing_execution_loop,
        CamelRolePlayingTracer,
        CamelWorkforceTracer,
    )

    __all__.extend([
        "CamelAgentAdapter",
        "CamelUser",
        "CamelAgentUser",
        "camel_role_playing_execution_loop",
        "CamelRolePlayingTracer",
        "CamelWorkforceTracer",
    ])
except ImportError:
    pass
