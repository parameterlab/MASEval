# Model Adapters

Model Adapters provide a uniform runtime interface to heterogeneous model providers (HuggingFace, OpenAI, Google GenAI, or simple callables). Each adapter exposes a stable API so that `maseval` can call models without handling provider-specific shapes.

!!! note

    `Benchmark` expects `AgentAdapter` instances; it does not consume model adapters directly. ModelAdapters are used by agents, simulators, others that directly invoke models.

::: maseval.core.model.ModelAdapter

## Interfaces

The following adapter classes implement the ModelAdapter interface for specific providers. Each requires their own dependencies.

::: maseval.interface.inference.openai.OpenAIModelAdapter

::: maseval.interface.inference.huggingface.HuggingFaceModelAdapter

::: maseval.interface.inference.google_genai.GoogleGenAIModelAdapter
