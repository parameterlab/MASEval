# LlamaIndex

Adapter implementing commonly used functions for LlamaIndex's workflow-based agent system. The API will be rendered when the optional dependency is installed in the build environment.

## Overview

LlamaIndex provides a powerful workflow-based agent framework that enables building sophisticated multi-agent systems. MASEval's LlamaIndex adapter allows you to evaluate these agents using standardized benchmarks.

## Installation

```bash
pip install maseval[llamaindex]
```

This installs `llama-index-core`, which contains the workflow-based agent system.

## Quick Start

### Basic Agent Workflow

```python
from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.llms import OpenAI

# Create a LlamaIndex workflow agent
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[...],
    llm=OpenAI(model="gpt-4"),
    system_prompt="You are a helpful assistant"
)

# Wrap with MASEval adapter
agent_adapter = LlamaIndexAgentAdapter(workflow, "my_agent")

# Run with a query
result = agent_adapter.run("What's the weather in New York?")

# Access message history in OpenAI format
for msg in agent_adapter.get_messages():
    print(f"{msg['role']}: {msg['content']}")
```

### User Simulation

```python
from maseval.interface.agents.llamaindex import LlamaIndexUser
from maseval import ModelAdapter

# Create a user simulator
user = LlamaIndexUser(
    name="customer",
    model=your_model_adapter,
    user_profile={"role": "customer", "preferences": "..."},
    scenario="Shopping for a laptop",
    initial_prompt="I need help finding a laptop"
)

# Get LlamaIndex-compatible tool
user_tool = user.get_tool()

# Add tool to agent
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[user_tool, ...],
    llm=llm
)
```

### Running Benchmarks

```python
from maseval import Benchmark
from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter

# Create your LlamaIndex agent
workflow = AgentWorkflow.from_tools_or_functions(...)

# Wrap with adapter
agent_adapter = LlamaIndexAgentAdapter(workflow, "agent")

# Run benchmark
benchmark = MyBenchmark(tasks=tasks, agent_data={"agent": agent_adapter})
results = benchmark.run()
```

## Supported Agent Types

The adapter supports LlamaIndex's workflow-based agents:

- **AgentWorkflow**: Multi-agent workflow orchestrator
- **FunctionAgent**: Function-calling based agent (for LLMs with tool calling)
- **ReActAgent**: ReAct prompting pattern agent
- **CodeActAgent**: Code execution based agent

## Message Format

LlamaIndex uses `ChatMessage` objects with `MessageRole` enums. The adapter automatically converts these to OpenAI-compatible format:

| LlamaIndex Role | OpenAI Role |
| --------------- | ----------- |
| USER            | user        |
| ASSISTANT       | assistant   |
| SYSTEM          | system      |
| TOOL            | tool        |

Tool calls are preserved in the `additional_kwargs` field and converted to OpenAI format.

## Async Handling

LlamaIndex agents are async-first. The adapter handles async execution automatically:

- If the agent has `run_sync()`, it's preferred
- Otherwise, `asyncio.run()` is used to run the async `run()` method
- This works seamlessly in synchronous benchmarking contexts

## Configuration and Tracing

The adapter provides comprehensive configuration and trace collection:

```python
# Gather configuration
config = agent_adapter.gather_config()
# Returns: agent name, description, system prompt, tools, etc.

# Gather execution traces
traces = agent_adapter.gather_traces()
# Returns: messages, execution logs, timing, token usage (if available)
```

## Limitations

- **Streaming**: Not yet supported (planned for future release)
- **Multi-agent workflows**: Basic support; per-agent tracking coming soon
- **Token usage**: Only available if LLM responses include usage metadata

## API Reference

::: maseval.interface.agents.llamaindex
