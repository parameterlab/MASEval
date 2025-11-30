"""Utility functions for the 5-A-Day Benchmark.

This module contains helper functions that may be promoted to the main maseval library
if they prove useful across multiple benchmarks.
"""

import hashlib


def derive_seed(base_seed: int, *components: str | int) -> int:
    """Derive a unique seed from base seed and component identifiers.

    Uses SHA256 hashing to create deterministic, unique seeds for different components
    (tasks, agents, etc.) from a single base seed. This ensures reproducibility while
    giving each component its own random stream.

    The derivation is order-dependent: derive_seed(42, "a", "b") != derive_seed(42, "b", "a")
    This is intentional to allow hierarchical seed structures.

    Args:
        base_seed: Base seed from CLI (e.g., --seed 42)
        *components: Component identifiers (task_idx, agent_id, etc.)
                     Can be strings or integers

    Returns:
        Derived seed as 31-bit positive integer (compatible with most RNG implementations)

    Example:
        >>> derive_seed(42, 0, "orchestrator")  # Task 0, orchestrator agent
        1234567890  # deterministic output
        >>> derive_seed(42, 0, "specialist_finance")  # Task 0, finance specialist
        987654321  # different but deterministic output
        >>> derive_seed(42, 1, "orchestrator")  # Task 1, orchestrator agent
        567891234  # different from task 0
    """
    # Combine base seed with all components
    seed_string = f"{base_seed}:" + ":".join(str(c) for c in components)

    # SHA256 hash for cryptographically sound mixing
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()

    # Convert first 4 bytes to int, use 31 bits to ensure positive int
    # (some RNG implementations expect positive seeds)
    return int.from_bytes(hash_bytes[:4], "big") & 0x7FFFFFFF


def sanitize_name(name: str) -> str:
    """Sanitize name to be a valid Python identifier.

    Required for smolagents framework which uses names as Python identifiers.

    Args:
        name: Name to sanitize (may contain spaces, hyphens, etc.)

    Returns:
        Valid Python identifier (underscores instead of spaces/hyphens,
        starts with letter or underscore)

    Example:
        >>> sanitize_name("Email Assistant")
        "Email_Assistant"
        >>> sanitize_name("123-agent")
        "_123_agent"
    """
    sanitized = name.replace(" ", "_").replace("-", "_")
    if not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = "_" + sanitized
    return sanitized
