"""Test ModelAdapter functionality.

These tests verify the core contracts of the ModelAdapter base class:
- Call logging and tracing infrastructure
- Error handling and logging
- Timing capture
- Trace and config structure
- JSON serializability
- Base mixin functionality (TraceableMixin, ConfigurableMixin)

All tests use DummyModelAdapter from conftest.py to avoid external dependencies.
"""

import pytest
import json
import time
from datetime import datetime
from conftest import DummyModelAdapter


@pytest.mark.core
class TestModelAdapterBaseContract:
    """Test fundamental ModelAdapter base class behavior."""

    def test_model_adapter_has_abstract_methods(self):
        """ModelAdapter requires subclasses to implement model_id and _generate_impl."""
        from maseval.core.model import ModelAdapter

        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            ModelAdapter()  # type: ignore

    def test_model_adapter_requires_model_id_property(self):
        """Subclasses must implement model_id property."""
        from maseval.core.model import ModelAdapter

        class IncompleteAdapter(ModelAdapter):
            def _generate_impl(self, prompt, generation_params=None, **kwargs):
                return "test"

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_model_adapter_requires_generate_impl(self):
        """Subclasses must implement _generate_impl method."""
        from maseval.core.model import ModelAdapter

        class IncompleteAdapter(ModelAdapter):
            @property
            def model_id(self):
                return "test"

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_model_adapter_initializes_logs(self):
        """ModelAdapter initializes empty logs on construction."""
        model = DummyModelAdapter()
        assert hasattr(model, "logs")
        assert isinstance(model.logs, list)
        assert len(model.logs) == 0

    def test_model_adapter_model_id_accessible(self):
        """model_id property is accessible."""
        model = DummyModelAdapter(model_id="test-model-id")
        assert model.model_id == "test-model-id"


@pytest.mark.core
class TestModelAdapterGenerationContract:
    """Test generation method behavior and logging."""

    def test_generate_returns_string(self):
        """generate() returns string output."""
        model = DummyModelAdapter(responses=["Test response"])
        result = model.generate("Test prompt")

        assert isinstance(result, str)
        assert result == "Test response"

    def test_generate_logs_successful_calls(self, dummy_model):
        """generate() logs successful calls with metadata."""
        dummy_model.generate("Test prompt 1")
        dummy_model.generate("Test prompt 2")

        # Check internal logs
        assert len(dummy_model.logs) == 2
        assert dummy_model.logs[0]["status"] == "success"
        assert dummy_model.logs[1]["status"] == "success"

        # Verify required fields
        call = dummy_model.logs[0]
        assert "timestamp" in call
        assert "prompt_length" in call
        assert "response_length" in call
        assert "duration_seconds" in call
        assert "status" in call
        assert "generation_params" in call
        assert "kwargs" in call

    def test_generate_logs_generation_params(self):
        """generate() logs generation parameters."""
        model = DummyModelAdapter()
        params = {"temperature": 0.7, "max_tokens": 100}
        model.generate("Test", generation_params=params)

        call = model.logs[0]
        assert call["generation_params"] == params

    def test_generate_logs_kwargs(self):
        """generate() logs additional kwargs."""
        model = DummyModelAdapter()
        model.generate("Test", custom_arg="value", another_arg=123)

        call = model.logs[0]
        assert "custom_arg" in call["kwargs"]
        assert "another_arg" in call["kwargs"]
        # kwargs are stringified for JSON serialization
        assert call["kwargs"]["custom_arg"] == "value"
        assert call["kwargs"]["another_arg"] == "123"

    def test_model_adapter_log_accumulation(self, dummy_model):
        """Test that logs accumulate across multiple calls."""
        for i in range(5):
            dummy_model.generate(f"Prompt {i}")

        traces = dummy_model.gather_traces()
        assert traces["total_calls"] == 5
        assert traces["successful_calls"] == 5
        assert len(traces["logs"]) == 5

    def test_generate_captures_timing(self):
        """generate() captures execution duration."""
        model = DummyModelAdapter()
        model.generate("Test")

        call = model.logs[0]
        assert "duration_seconds" in call
        assert isinstance(call["duration_seconds"], (int, float))
        assert call["duration_seconds"] >= 0

    def test_generate_with_empty_prompt(self):
        """generate() handles empty prompt."""
        model = DummyModelAdapter(responses=["Response"])
        result = model.generate("")

        assert isinstance(result, str)
        assert len(model.logs) == 1
        assert model.logs[0]["prompt_length"] == 0


@pytest.mark.core
class TestModelAdapterErrorHandling:
    """Test error handling and error logging."""

    def test_model_adapter_error_handling(self, dummy_model):
        """Test that errors are logged correctly."""

        class FailingModel(DummyModelAdapter):
            def _generate_impl(self, prompt, generation_params=None, **kwargs):
                raise ValueError("Test error")

        model = FailingModel()

        with pytest.raises(ValueError):
            model.generate("Test")

        # Error should be logged
        assert len(model.logs) == 1
        assert model.logs[0]["status"] == "error"
        assert "Test error" in model.logs[0]["error"]

    def test_generate_logs_error_timing(self):
        """generate() logs duration even when errors occur."""

        class FailingModel(DummyModelAdapter):
            def _generate_impl(self, prompt, generation_params=None, **kwargs):
                time.sleep(0.01)  # Small delay
                raise RuntimeError("Fail")

        model = FailingModel()

        with pytest.raises(RuntimeError):
            model.generate("Test")

        call = model.logs[0]
        assert call["duration_seconds"] >= 0.01

    def test_generate_logs_error_metadata(self):
        """generate() logs prompt length and params even on error."""

        class FailingModel(DummyModelAdapter):
            def _generate_impl(self, prompt, generation_params=None, **kwargs):
                raise Exception("Fail")

        model = FailingModel()
        params = {"temperature": 0.9}

        with pytest.raises(Exception):
            model.generate("Test prompt", generation_params=params, custom="arg")

        call = model.logs[0]
        assert call["prompt_length"] == len("Test prompt")
        assert call["generation_params"] == params
        assert "custom" in call["kwargs"]

    def test_generate_reraises_original_exception(self):
        """generate() re-raises the original exception type."""

        class CustomError(Exception):
            pass

        class FailingModel(DummyModelAdapter):
            def _generate_impl(self, prompt, generation_params=None, **kwargs):
                raise CustomError("Original error")

        model = FailingModel()

        with pytest.raises(CustomError, match="Original error"):
            model.generate("Test")


@pytest.mark.core
class TestModelAdapterTracing:
    """Test gather_traces() contract."""

    def test_gather_traces_returns_dict(self):
        """gather_traces() returns a dictionary."""
        model = DummyModelAdapter()
        traces = model.gather_traces()

        assert isinstance(traces, dict)

    def test_model_adapter_gather_traces_structure(self, dummy_model):
        """Test that gather_traces() has correct structure."""
        dummy_model.generate("Test prompt")

        traces = dummy_model.gather_traces()

        # Base fields from TraceableMixin
        assert "type" in traces
        assert "gathered_at" in traces
        # ModelAdapter-specific fields
        assert "model_id" in traces
        assert "total_calls" in traces
        assert "successful_calls" in traces
        assert "failed_calls" in traces
        assert "logs" in traces

    def test_gather_traces_has_base_fields(self):
        """gather_traces() includes base TraceableMixin fields."""
        model = DummyModelAdapter()
        traces = model.gather_traces()

        # Base fields from TraceableMixin
        assert "type" in traces
        assert "gathered_at" in traces

        assert traces["type"] == "DummyModelAdapter"
        assert isinstance(traces["gathered_at"], str)

        # Validate timestamp format
        datetime.fromisoformat(traces["gathered_at"])

    def test_gather_traces_aggregates_statistics(self):
        """gather_traces() correctly aggregates call statistics."""
        model = DummyModelAdapter(responses=["R1", "R2", "R3"])

        for i in range(3):
            model.generate(f"Prompt {i}")

        traces = model.gather_traces()

        assert traces["total_calls"] == 3
        assert traces["successful_calls"] == 3
        assert traces["failed_calls"] == 0
        assert traces["total_duration_seconds"] >= 0
        assert traces["average_duration_seconds"] >= 0

    def test_gather_traces_counts_failures(self):
        """gather_traces() counts failed calls correctly."""

        class SometimesFailingModel(DummyModelAdapter):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def _generate_impl(self, prompt, generation_params=None, **kwargs):
                self.call_count += 1
                if self.call_count % 2 == 0:
                    raise ValueError("Fail")
                return "Success"

        model = SometimesFailingModel()

        # Call 1: success
        model.generate("Test 1")
        # Call 2: fail
        with pytest.raises(ValueError):
            model.generate("Test 2")
        # Call 3: success
        model.generate("Test 3")

        traces = model.gather_traces()

        assert traces["total_calls"] == 3
        assert traces["successful_calls"] == 2
        assert traces["failed_calls"] == 1

    def test_model_adapter_timing_captured(self, dummy_model):
        """Test that timing information is captured."""
        dummy_model.generate("Test")

        traces = dummy_model.gather_traces()
        assert "total_duration_seconds" in traces
        assert "average_duration_seconds" in traces
        assert traces["total_duration_seconds"] >= 0

    def test_gather_traces_handles_zero_calls(self):
        """gather_traces() handles models with no calls."""
        model = DummyModelAdapter()
        traces = model.gather_traces()

        assert traces["total_calls"] == 0
        assert traces["successful_calls"] == 0
        assert traces["failed_calls"] == 0
        assert traces["total_duration_seconds"] == 0
        assert traces["average_duration_seconds"] == 0
        assert traces["logs"] == []

    def test_gather_traces_is_json_serializable(self):
        """gather_traces() returns JSON-serializable data."""
        model = DummyModelAdapter()
        model.generate("Test", generation_params={"temp": 0.5}, custom_kwarg="value")

        traces = model.gather_traces()

        # Should not raise
        json_str = json.dumps(traces)
        assert isinstance(json_str, str)

        # Should round-trip
        recovered = json.loads(json_str)
        assert recovered["total_calls"] == traces["total_calls"]

    def test_gather_traces_is_idempotent(self):
        """gather_traces() can be called multiple times."""
        model = DummyModelAdapter()
        model.generate("Test")

        traces1 = model.gather_traces()
        traces2 = model.gather_traces()

        # Type and model_id should be identical
        assert traces1["type"] == traces2["type"]
        assert traces1["model_id"] == traces2["model_id"]
        assert traces1["total_calls"] == traces2["total_calls"]

        # Timestamps might differ
        assert "gathered_at" in traces1
        assert "gathered_at" in traces2


@pytest.mark.core
class TestModelAdapterConfiguration:
    """Test gather_config() contract."""

    def test_gather_config_returns_dict(self):
        """gather_config() returns a dictionary."""
        model = DummyModelAdapter()
        config = model.gather_config()

        assert isinstance(config, dict)

    def test_model_adapter_gather_config(self, dummy_model):
        """Test that gather_config() returns configuration."""
        config = dummy_model.gather_config()

        assert "type" in config
        assert "gathered_at" in config
        assert config["type"] == "DummyModelAdapter"

    def test_gather_config_has_base_fields(self):
        """gather_config() includes base ConfigurableMixin fields."""
        model = DummyModelAdapter()
        config = model.gather_config()

        # Base fields from ConfigurableMixin
        assert "type" in config
        assert "gathered_at" in config

        assert config["type"] == "DummyModelAdapter"
        assert isinstance(config["gathered_at"], str)

        # Validate timestamp format
        datetime.fromisoformat(config["gathered_at"])

    def test_gather_config_has_model_fields(self):
        """gather_config() includes ModelAdapter-specific fields."""
        model = DummyModelAdapter(model_id="test-model")
        config = model.gather_config()

        # ModelAdapter-specific fields
        assert "model_id" in config
        assert "adapter_type" in config

        assert config["model_id"] == "test-model"
        assert config["adapter_type"] == "DummyModelAdapter"

    def test_gather_config_is_static(self):
        """gather_config() returns same data before and after execution."""
        model = DummyModelAdapter()

        config_before = model.gather_config()
        model.generate("Test")
        config_after = model.gather_config()

        # Remove timestamps for comparison
        def remove_timestamp(d):
            return {k: v for k, v in d.items() if k != "gathered_at"}

        assert remove_timestamp(config_before) == remove_timestamp(config_after)

    def test_gather_config_is_json_serializable(self):
        """gather_config() returns JSON-serializable data."""
        model = DummyModelAdapter(model_id="test-model")
        config = model.gather_config()

        # Should not raise
        json_str = json.dumps(config)
        assert isinstance(json_str, str)

        # Should round-trip
        recovered = json.loads(json_str)
        assert recovered["model_id"] == config["model_id"]

    def test_gather_config_is_idempotent(self):
        """gather_config() can be called multiple times."""
        model = DummyModelAdapter()

        config1 = model.gather_config()
        config2 = model.gather_config()

        # Type and model_id should be identical
        assert config1["type"] == config2["type"]
        assert config1["model_id"] == config2["model_id"]

        # Timestamps might differ
        assert "gathered_at" in config1
        assert "gathered_at" in config2


@pytest.mark.core
class TestModelAdapterMixinIntegration:
    """Test that ModelAdapter correctly inherits from TraceableMixin and ConfigurableMixin."""

    def test_model_adapter_is_traceable(self):
        """ModelAdapter inherits from TraceableMixin."""
        from maseval.core.model import ModelAdapter
        from maseval.core.tracing import TraceableMixin

        assert issubclass(ModelAdapter, TraceableMixin)

        model = DummyModelAdapter()
        assert isinstance(model, TraceableMixin)

    def test_model_adapter_is_configurable(self):
        """ModelAdapter inherits from ConfigurableMixin."""
        from maseval.core.model import ModelAdapter
        from maseval.core.config import ConfigurableMixin

        assert issubclass(ModelAdapter, ConfigurableMixin)

        model = DummyModelAdapter()
        assert isinstance(model, ConfigurableMixin)

    def test_gather_traces_includes_mixin_fields(self):
        """gather_traces() includes fields from TraceableMixin."""
        model = DummyModelAdapter()
        traces = model.gather_traces()

        # Fields from TraceableMixin
        assert "type" in traces
        assert "gathered_at" in traces

    def test_gather_config_includes_mixin_fields(self):
        """gather_config() includes fields from ConfigurableMixin."""
        model = DummyModelAdapter()
        config = model.gather_config()

        # Fields from ConfigurableMixin
        assert "type" in config
        assert "gathered_at" in config
