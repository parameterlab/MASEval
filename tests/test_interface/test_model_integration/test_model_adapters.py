"""Integration tests for ModelAdapter interface implementations.

Tests specific behavior and integration for each ModelAdapter implementation:
- OpenAIModelAdapter
- GoogleGenAIModelAdapter
- HuggingFaceModelAdapter
- LiteLLMModelAdapter

These tests verify that each adapter correctly wraps its underlying client
and provides adapter-specific configuration.
"""

import pytest


# ==================== OpenAI Tests ====================


@pytest.mark.interface
class TestOpenAIModelAdapterIntegration:
    """Test OpenAIModelAdapter specific behavior."""

    def test_openai_adapter_initialization(self):
        """OpenAIModelAdapter initializes with client and model_id."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        # Mock client with chat.completions.create interface
        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "Response"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")

        assert adapter.model_id == "gpt-4"

    def test_openai_adapter_generate_with_modern_client(self):
        """OpenAIModelAdapter works with modern client interface."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        # Extract user message content
                        user_msg = next((m for m in messages if m["role"] == "user"), {})
                        content = user_msg.get("content", "")
                        return {"choices": [{"message": {"content": f"Response to: {content}"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        result = adapter.generate("Test prompt")

        assert isinstance(result, str)
        assert "Response to: Test prompt" in result

    def test_openai_adapter_extract_text_from_dict(self):
        """OpenAIModelAdapter extracts text from various response formats."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        # Chat completion format
        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "Chat response"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        result = adapter.generate("Test")
        assert result == "Chat response"

    def test_openai_adapter_default_generation_params(self):
        """OpenAIModelAdapter uses default generation parameters."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        captured_params = {}

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        captured_params.update(kwargs)
                        return {"choices": [{"message": {"content": "Response"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(
            client=MockClient(),
            model_id="gpt-4",
            default_generation_params={"temperature": 0.7, "max_tokens": 100},
        )

        adapter.generate("Test")

        assert "temperature" in captured_params
        assert captured_params["temperature"] == 0.7
        assert captured_params["max_tokens"] == 100

    def test_openai_adapter_gather_config_includes_params(self):
        """OpenAIModelAdapter config includes generation parameters."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "Response"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(
            client=MockClient(),
            model_id="gpt-4",
            default_generation_params={"temperature": 0.9},
        )

        config = adapter.gather_config()

        assert "default_generation_params" in config
        assert config["default_generation_params"]["temperature"] == 0.9
        assert "client_type" in config

    def test_openai_adapter_gather_config_includes_client_config(self):
        """OpenAIModelAdapter config includes client configuration."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        # Create a mock client with configuration attributes
        class MockOpenAIClient:
            def __init__(self):
                self.timeout = 60
                self.max_retries = 3

            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "Response"}}]}

                completions = Completions()

            chat = Chat()

        client = MockOpenAIClient()
        adapter = OpenAIModelAdapter(client=client, model_id="gpt-4")

        config = adapter.gather_config()

        # Should include client configuration that affects behavior
        assert "client_type" in config
        assert config["client_type"] == "MockOpenAIClient"


# ==================== Google GenAI Tests ====================


@pytest.mark.interface
class TestGoogleGenAIModelAdapterIntegration:
    """Test GoogleGenAIModelAdapter specific behavior."""

    def test_google_genai_adapter_initialization(self):
        """GoogleGenAIModelAdapter initializes with client and model_id."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        # Mock client
        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "Response"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        client = MockClient()
        adapter = GoogleGenAIModelAdapter(client=client, model_id="gemini-pro")

        assert adapter.model_id == "gemini-pro"

    def test_google_genai_adapter_generate(self):
        """GoogleGenAIModelAdapter generates text."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    # Extract text from contents (first user message)
                    text = ""
                    if contents:
                        for content in contents:
                            if content.get("role") == "user":
                                parts = content.get("parts", [])
                                if parts:
                                    text = parts[0].get("text", "")
                                    break

                    class Response:
                        pass

                    resp = Response()
                    resp.text = f"Response to: {text}"
                    return resp

            def __init__(self):
                self.models = self.Models()

        client = MockClient()
        adapter = GoogleGenAIModelAdapter(client=client, model_id="gemini-pro")
        result = adapter.generate("Test prompt")

        assert isinstance(result, str)
        assert "Response to: Test prompt" in result

    def test_google_genai_adapter_default_generation_params(self):
        """GoogleGenAIModelAdapter uses default generation parameters."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        captured_config = None

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    nonlocal captured_config
                    captured_config = config

                    class Response:
                        text = "Response"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        client = MockClient()
        adapter = GoogleGenAIModelAdapter(
            client=client,
            model_id="gemini-pro",
            default_generation_params={"temperature": 0.8},
        )

        adapter.generate("Test")

        # Should pass config to generate_content
        assert captured_config is not None

    def test_google_genai_adapter_gather_config(self):
        """GoogleGenAIModelAdapter config includes parameters."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "Response"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        client = MockClient()
        adapter = GoogleGenAIModelAdapter(
            client=client,
            model_id="gemini-pro",
            default_generation_params={"temperature": 0.9},
        )

        config = adapter.gather_config()

        assert "default_generation_params" in config
        assert config["default_generation_params"]["temperature"] == 0.9
        assert "client_type" in config


# ==================== HuggingFace Tests ====================


@pytest.mark.interface
class TestHuggingFaceModelAdapterIntegration:
    """Test HuggingFaceModelAdapter specific behavior."""

    def test_huggingface_adapter_initialization(self):
        """HuggingFaceModelAdapter initializes with callable."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt, **kwargs):
            return f"Response to: {prompt}"

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="gpt2")

        assert adapter.model_id == "gpt2"

    def test_huggingface_adapter_generate(self):
        """HuggingFaceModelAdapter generates text with message formatting."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt, **kwargs):
            return f"Generated: {prompt}"

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="gpt2")
        result = adapter.generate("Test prompt")

        assert isinstance(result, str)
        # Without a tokenizer, the adapter formats messages as "user: content\nassistant:"
        assert "Generated:" in result

    def test_huggingface_adapter_default_generation_params(self):
        """HuggingFaceModelAdapter uses default generation parameters."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        captured_params = {}

        def mock_model(prompt, **kwargs):
            captured_params.update(kwargs)
            return "Response"

        adapter = HuggingFaceModelAdapter(
            model=mock_model,
            model_id="gpt2",
            default_generation_params={"max_length": 50, "temperature": 0.7},
        )

        adapter.generate("Test")

        assert "max_length" in captured_params
        assert captured_params["max_length"] == 50
        assert captured_params["temperature"] == 0.7

    def test_huggingface_adapter_fallback_without_kwargs(self):
        """HuggingFaceModelAdapter falls back to calling without kwargs."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt):
            # Only accepts prompt
            return f"Response: {prompt}"

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="gpt2")
        result = adapter.generate("Test")

        # Should still work, just formats the prompt as messages
        assert "Response:" in result

    def test_huggingface_adapter_gather_config(self):
        """HuggingFaceModelAdapter config includes parameters."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt):
            return "Response"

        adapter = HuggingFaceModelAdapter(
            model=mock_model,
            model_id="gpt2",
            default_generation_params={"max_length": 100},
        )

        config = adapter.gather_config()

        assert "default_generation_params" in config
        assert config["default_generation_params"]["max_length"] == 100
        assert "callable_type" in config

    def test_huggingface_adapter_gather_config_with_pipeline(self):
        """HuggingFaceModelAdapter config includes pipeline configuration."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        # Mock pipeline object with attributes
        class MockPipeline:
            def __init__(self):
                self.task = "text-generation"
                self.device = "cpu"
                self.framework = "pt"

            def __call__(self, prompt, **kwargs):
                return "Response"

        pipeline = MockPipeline()
        adapter = HuggingFaceModelAdapter(model=pipeline, model_id="gpt2")

        config = adapter.gather_config()

        assert "pipeline_config" in config
        assert config["pipeline_config"]["task"] == "text-generation"
        assert "cpu" in str(config["pipeline_config"]["device"])
        assert config["pipeline_config"]["framework"] == "pt"


# ==================== LiteLLM Tests ====================


@pytest.mark.interface
class TestLiteLLMModelAdapterIntegration:
    """Test LiteLLMModelAdapter specific behavior."""

    def test_litellm_adapter_initialization(self):
        """LiteLLMModelAdapter initializes with model_id."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(model_id="gpt-3.5-turbo")

        assert adapter.model_id == "gpt-3.5-turbo"

    def test_litellm_adapter_initialization_with_params(self):
        """LiteLLMModelAdapter initializes with parameters."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(
            model_id="gpt-4",
            default_generation_params={"temperature": 0.7, "max_tokens": 100},
            api_key="test-key",
            api_base="https://test.com",
        )

        assert adapter.model_id == "gpt-4"
        assert adapter._default_generation_params["temperature"] == 0.7
        assert adapter._api_key == "test-key"
        assert adapter._api_base == "https://test.com"

    def test_litellm_adapter_gather_config(self):
        """LiteLLMModelAdapter config includes parameters."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(
            model_id="gpt-4",
            default_generation_params={"temperature": 0.9, "max_tokens": 200},
        )

        config = adapter.gather_config()

        assert "default_generation_params" in config
        assert config["default_generation_params"]["temperature"] == 0.9
        assert config["default_generation_params"]["max_tokens"] == 200
        assert config["model_id"] == "gpt-4"


# ==================== Cross-Adapter Tests ====================


@pytest.mark.interface
class TestCrossAdapterConsistency:
    """Test that all adapters follow consistent patterns."""

    def test_all_adapters_expose_model_id(self):
        """All adapters expose model_id property."""
        pytest.importorskip("openai")
        pytest.importorskip("google.genai")
        pytest.importorskip("transformers")
        pytest.importorskip("litellm")

        from maseval.interface.inference.openai import OpenAIModelAdapter
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        # OpenAI - mock with modern interface
        class MockOpenAIClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "R"}}]}

                completions = Completions()

            chat = Chat()

        openai_adapter = OpenAIModelAdapter(client=MockOpenAIClient(), model_id="gpt-4")
        assert openai_adapter.model_id == "gpt-4"

        # Google GenAI
        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "R"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        google_adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro")
        assert google_adapter.model_id == "gemini-pro"

        # HuggingFace
        hf_adapter = HuggingFaceModelAdapter(model=lambda p: "R", model_id="gpt2")
        assert hf_adapter.model_id == "gpt2"

        # LiteLLM
        litellm_adapter = LiteLLMModelAdapter(model_id="gpt-3.5-turbo")
        assert litellm_adapter.model_id == "gpt-3.5-turbo"

    def test_all_adapters_include_default_params_in_config(self):
        """All adapters include default_generation_params in config."""
        pytest.importorskip("openai")
        pytest.importorskip("google.genai")
        pytest.importorskip("transformers")
        pytest.importorskip("litellm")

        from maseval.interface.inference.openai import OpenAIModelAdapter
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        params = {"temperature": 0.7}

        # OpenAI
        class MockOpenAIClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "R"}}]}

                completions = Completions()

            chat = Chat()

        openai_config = OpenAIModelAdapter(
            client=MockOpenAIClient(),
            model_id="gpt-4",
            default_generation_params=params,
        ).gather_config()
        assert "default_generation_params" in openai_config

        # Google GenAI
        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "R"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        google_config = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro", default_generation_params=params).gather_config()
        assert "default_generation_params" in google_config

        # HuggingFace
        hf_config = HuggingFaceModelAdapter(model=lambda p: "R", model_id="gpt2", default_generation_params=params).gather_config()
        assert "default_generation_params" in hf_config

        # LiteLLM
        litellm_config = LiteLLMModelAdapter(model_id="gpt-3.5-turbo", default_generation_params=params).gather_config()
        assert "default_generation_params" in litellm_config
