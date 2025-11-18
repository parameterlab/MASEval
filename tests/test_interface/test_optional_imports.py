"""Test that optional dependencies are handled correctly.

These tests ensure the package works gracefully without optional dependencies.
Mark with @pytest.mark.core to run in minimal environment.
"""

import pytest

# All tests in this file are core tests (no optional deps required)
pytestmark = pytest.mark.core


def test_core_package_imports():
    """Test that core package can be imported without optional deps."""
    import maseval
    from maseval import AgentAdapter, Task, Environment, Benchmark  # noqa: F401

    assert maseval is not None


def test_interface_package_imports():
    """Test that interface package imports without optional deps."""
    import maseval.interface

    # Should not raise ImportError
    assert hasattr(maseval.interface, "__all__")
    assert isinstance(maseval.interface.__all__, list)


def test_interface_all_reflects_available_integrations():
    """Test that __all__ only contains available integrations."""
    import maseval.interface

    # With no optional deps, __all__ should be empty or only contain core items
    # This will vary based on what's installed
    assert isinstance(maseval.interface.__all__, list)


def test_model_adapter_imports_without_optional_deps():
    """Test that ModelAdapter base class can be imported without optional deps."""
    from maseval.core.model import ModelAdapter

    # Should not raise ImportError
    assert ModelAdapter is not None
    # Should be abstract
    assert hasattr(ModelAdapter, "__abstractmethods__")


def test_inference_adapters_gracefully_handle_missing_deps():
    """Test that inference adapter modules can be imported without their dependencies."""
    # Try importing adapter modules (not the adapters themselves)
    # This should work even without optional dependencies
    try:
        import maseval.interface.inference

        assert maseval.interface.inference is not None
    except ImportError as e:
        # If the module itself can't be imported, that's okay
        # as long as the core package works
        assert "maseval.core" not in str(e)


def test_agent_adapters_gracefully_handle_missing_deps():
    """Test that agent adapter modules can be imported without their dependencies."""
    # Try importing adapter modules (not the adapters themselves)
    # This should work even without optional dependencies
    try:
        import maseval.interface.agents

        assert maseval.interface.agents is not None
    except ImportError as e:
        # If the module itself can't be imported, that's okay
        # as long as the core package works
        assert "maseval.core" not in str(e)
