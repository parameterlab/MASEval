# LlamaIndex Interface Implementation Plan

## Executive Summary

This document outlines the plan for implementing a LlamaIndex interface for MASEval, following the same architectural principles used for `smolagents` and `langgraph` integrations. The implementation will focus on the **workflow-based agent system** introduced in LlamaIndex, specifically the `AgentWorkflow` and `BaseWorkflowAgent` classes.

## 1. Understanding MASEval's Architecture

### 1.1 Core Principles

- **Evaluation, Not Implementation**: MASEval provides infrastructure for evaluating existing agent implementations
- **Framework Agnostic**: Simple adapters translate between MASEval abstractions and external frameworks
- **Strict Separation**: `maseval/core` (minimal dependencies) vs `maseval/interface` (optional integrations)
- **Core NEVER imports from interface**

### 1.2 Key Abstractions

1. **AgentAdapter**: Wraps framework-specific agents with unified interface

   - `run(query: str) -> Any`: Execute agent and return final answer
   - `_run_agent(query: str) -> Any`: Framework-specific execution (abstract)
   - `get_messages() -> MessageHistory`: Return OpenAI-compatible message history
   - `gather_traces() -> dict`: Collect execution traces (timing, tokens, steps)
   - `gather_config() -> dict`: Collect configuration metadata

2. **User**: Simulates user interactions

   - `get_tool()`: Returns framework-specific tool for user input
   - `simulate_response(question: str) -> str`: Generate simulated user response

3. **MessageHistory**: OpenAI-compatible message format
   - Iterable, indexable collection of messages
   - Each message: `{"role": "user|assistant|system|tool", "content": "...", ...}`
   - Supports tool calls, multi-modal content

### 1.3 Adapter Pattern (Critical Rule)

**Framework's native storage is the single source of truth:**

- **Pattern 1: Persistent State** (smolagents)

  - Framework stores messages internally (e.g., `agent.memory.steps`)
  - `get_messages()` dynamically fetches and converts from framework storage
  - NO caching of converted messages
  - Conversion is cheap, framework state is authoritative

- **Pattern 2: Stateless/Result-based** (langgraph)
  - Framework may not persist state between runs
  - Cache last result from `invoke()` call
  - Prefer fetching from persistent state if available (checkpointer + thread_id)
  - Fall back to cached result if no persistent state

## 2. Understanding LlamaIndex Agents

### 2.1 LlamaIndex Agent Architecture

LlamaIndex uses a **workflow-based** agent system built on async workflows:

**Core Components:**

- `AgentWorkflow`: Multi-agent workflow orchestrator
- `BaseWorkflowAgent`: Base class for individual agents
  - `FunctionAgent`: Function-calling based agent (for LLMs with tool calling)
  - `ReActAgent`: ReAct prompting pattern agent
  - `CodeActAgent`: Code execution based agent

**Key Characteristics:**

1. **Async-first**: All agent methods are async (`async def take_step`, `async def run_agent_step`)
2. **Workflow-based**: Uses LlamaIndex's Workflow system with events
3. **Memory management**: Uses `BaseMemory` for conversation history
4. **ChatMessage format**: Uses `ChatMessage` objects (similar to LangChain)
5. **Stateful execution**: Workflow context maintains state across steps

### 2.2 Message Format

LlamaIndex uses `ChatMessage` objects with roles:

```python
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Message roles: USER, ASSISTANT, SYSTEM, TOOL
message = ChatMessage(role=MessageRole.USER, content="Hello")
```

### 2.3 Execution Model

**Single Agent Workflow:**

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

# Create agent
agent = FunctionAgent(
    tools=[...],
    llm=llm,
    system_prompt="You are a helpful assistant"
)

# Create workflow
workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[...],
    llm=llm,
    system_prompt="..."
)

# Run workflow (async)
result = await workflow.run(input="What's the weather?")
# Or synchronous wrapper
result = workflow.run_sync(input="What's the weather?")
```

**Key Methods:**

- `run(input: str) -> Any`: Async execution
- `run_sync(input: str) -> Any`: Sync wrapper
- Result contains: `response: ChatMessage`, `structured_response: Dict`, `current_agent_name: str`

### 2.4 State Management

**Memory Access:**

- Workflow uses `Context` to store state
- Memory is managed via `BaseMemory` interface
- Messages stored in memory during execution
- Need to extract from workflow context after execution

**Challenge**: Unlike smolagents (which exposes `agent.memory.steps`) or langgraph (which returns state in result), LlamaIndex workflows:

- Store state in workflow context (not directly accessible)
- Return `AgentOutput` event with final response
- May not expose full message history directly

**Solution**: We'll need to:

1. Check if workflow/agent exposes memory directly
2. If not, cache messages from execution events
3. Potentially use workflow streaming to capture intermediate messages

## 3. LlamaIndex Interface Design

### 3.1 Adapter Architecture

**Proposed Adapters:**

1. **`LlamaIndexAgentAdapter`** - For `AgentWorkflow` or `BaseWorkflowAgent`

   - Wraps both `AgentWorkflow` and individual `BaseWorkflowAgent` instances
   - Handles async/sync execution
   - Converts `ChatMessage` to `MessageHistory`
   - Tracks execution via workflow events

2. **`LlamaIndexUser`** - For user simulation
   - Provides LlamaIndex-compatible tool for user input
   - Returns `AsyncBaseTool` that can be used with agents

### 3.2 State Management Strategy

**Based on LlamaIndex architecture, we'll use Pattern 2 (Stateless/Result-based):**

```python
class LlamaIndexAgentAdapter(AgentAdapter):
    def __init__(self, agent_instance, name: str, callbacks=None):
        super().__init__(agent_instance, name, callbacks)
        self._last_result = None
        self._message_cache = []  # Cache messages from execution

    def get_messages(self) -> MessageHistory:
        """Get message history from cached execution."""
        # Try to extract from agent/workflow if possible
        if hasattr(self.agent, 'memory'):
            messages = self.agent.memory.get_messages()
            return self._convert_llamaindex_messages(messages)

        # Fall back to cached messages
        return MessageHistory(self._message_cache)

    def _run_agent(self, query: str) -> Any:
        """Run agent and cache execution state."""
        # Check if agent/workflow is async
        if asyncio.iscoroutinefunction(self.agent.run):
            # Run async in sync context
            result = asyncio.run(self.agent.run(input=query))
        else:
            # Use sync wrapper if available
            result = self.agent.run_sync(input=query)

        # Cache result
        self._last_result = result

        # Extract messages from result and cache them
        self._extract_and_cache_messages(result)

        # Return final answer
        return self._extract_final_answer(result)
```

### 3.3 Message Conversion

**Convert LlamaIndex `ChatMessage` to OpenAI format:**

```python
def _convert_llamaindex_messages(self, messages: List[ChatMessage]) -> MessageHistory:
    """Convert LlamaIndex ChatMessage to MASEval MessageHistory."""
    converted = []

    for msg in messages:
        # Extract role (MessageRole enum to string)
        role = msg.role.value.lower()  # USER -> "user", ASSISTANT -> "assistant"

        message_dict = {
            "role": role,
            "content": msg.content or "",
        }

        # Handle tool calls if present
        if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
            if 'tool_calls' in msg.additional_kwargs:
                message_dict['tool_calls'] = msg.additional_kwargs['tool_calls']

        converted.append(message_dict)

    return MessageHistory(converted)
```

### 3.4 Async Handling

**Key Challenge**: LlamaIndex agents are async-first, but MASEval's `run()` is synchronous.

**Solutions:**

1. **Use asyncio.run()**: Run async code in sync context
2. **Provide both sync/async versions**: If workflow has `run_sync()`, prefer that
3. **Event loop management**: Handle existing event loops gracefully

```python
def _run_agent_sync(self, query: str) -> Any:
    """Run async agent in sync context."""
    # Check for existing event loop
    try:
        loop = asyncio.get_running_loop()
        # We're already in an async context, need to use run_in_executor
        return loop.run_until_complete(self._run_agent_async(query))
    except RuntimeError:
        # No event loop running, create new one
        return asyncio.run(self._run_agent_async(query))

async def _run_agent_async(self, query: str) -> Any:
    """Async implementation of agent execution."""
    result = await self.agent.run(input=query)
    return result
```

## 4. Implementation Details

### 4.1 File Structure

```
maseval/interface/agents/
├── __init__.py           # Export LlamaIndexAgentAdapter, LlamaIndexUser
├── smolagents.py         # Existing
├── langgraph.py          # Existing
└── llamaindex.py         # NEW - LlamaIndex integration
```

### 4.2 Dependencies

**Add to `pyproject.toml`:**

```toml
[project.optional-dependencies]
# Agent frameworks
llamaindex = ["llama-index-core>=0.13.0"]  # Core is sufficient for agents
```

**Note**:

- `llama-index-core` is the minimal package containing agent workflows
- Full `llama-index` includes many integrations (not needed)
- Version `>=0.13.0` ensures workflow-based agents are available

### 4.3 Lazy Imports

**Follow existing pattern:**

```python
"""LlamaIndex integration for MASEval.

This module requires llama-index-core to be installed:
    pip install maseval[llamaindex]
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.agent.workflow import AgentWorkflow, BaseWorkflowAgent
    from llama_index.core.base.llms.types import ChatMessage
else:
    AgentWorkflow = None
    BaseWorkflowAgent = None
    ChatMessage = None

def _check_llamaindex_installed():
    """Check if llama-index-core is installed."""
    try:
        import llama_index.core  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "llama-index-core is not installed. "
            "Install it with: pip install maseval[llamaindex]"
        ) from e
```

### 4.4 Key Methods to Implement

**LlamaIndexAgentAdapter:**

1. `__init__(agent_instance, name, callbacks=None)`

   - Accept both `AgentWorkflow` and `BaseWorkflowAgent`
   - Initialize message cache
   - Detect if agent is async

2. `_run_agent(query: str) -> Any`

   - Handle async execution
   - Run workflow/agent with query
   - Cache result and messages
   - Extract final answer
   - Log execution details (timing, token usage if available)

3. `get_messages() -> MessageHistory`

   - Try to extract from agent memory first
   - Fall back to cached messages
   - Convert `ChatMessage` to OpenAI format

4. `gather_traces() -> dict`

   - Extend base implementation
   - Include workflow execution details
   - Token usage (if available in result)
   - Step count, agent names (for multi-agent)
   - Timing information

5. `gather_config() -> dict`

   - Extend base implementation
   - Include workflow/agent configuration
   - System prompts, tool names
   - LLM configuration (if accessible)

6. `_convert_llamaindex_messages(messages: List[ChatMessage]) -> MessageHistory`
   - Convert `MessageRole` enum to string literals
   - Handle tool calls in `additional_kwargs`
   - Preserve message metadata

**LlamaIndexUser:**

1. `get_tool()`
   - Return LlamaIndex `AsyncBaseTool` or `BaseTool`
   - Tool wraps `simulate_response()`
   - Handle both async and sync tool execution

## 5. Testing Strategy

### 5.1 Test Structure

```
tests/test_interface/test_agent_integration/
├── test_llamaindex_agent.py  # NEW - LlamaIndex-specific tests
├── test_smolagents_agent.py  # Existing
└── test_langgraph_agent.py   # Existing
```

### 5.2 Test Coverage

**Contract Tests** (in `test_contract/test_agent_adapter_contract.py`):

- Add `"llamaindex"` to parametrized framework list
- Verify LlamaIndexAgentAdapter honors same contract as smolagents/langgraph
- All contract tests should pass without modification

**Integration Tests** (in `test_interface/test_agent_integration/test_llamaindex_agent.py`):

1. **Basic execution**

   - Create simple workflow with FunctionAgent
   - Run with query
   - Verify result and message history

2. **Message conversion**

   - Test ChatMessage -> MessageHistory conversion
   - Verify role mapping (USER -> user, ASSISTANT -> assistant)
   - Test tool calls preservation

3. **Async handling**

   - Test async workflow execution in sync context
   - Verify no event loop conflicts
   - Test both `run()` and `run_sync()` if available

4. **Multi-turn conversations**

   - Run multiple queries
   - Verify message history accumulates
   - Check message ordering

5. **Tool usage**

   - Create agent with tools
   - Verify tool calls in message history
   - Check tool results captured

6. **User simulation**

   - Create LlamaIndexUser
   - Get user tool
   - Test tool integration with agent

7. **Callbacks**

   - Test on_run_start/on_run_end
   - Verify callback order
   - Test multiple callbacks

8. **Traces and config**
   - Test gather_traces() structure
   - Test gather_config() structure
   - Verify framework-specific data included

### 5.3 Pytest Markers

```python
pytestmark = [pytest.mark.interface, pytest.mark.llamaindex]
```

**To run tests:**

```bash
# All LlamaIndex tests
pytest -m llamaindex -v

# Contract tests including LlamaIndex
pytest -m contract -v

# All interface tests
pytest -m interface -v
```

## 6. Documentation Plan

### 6.1 API Documentation

**Create `docs/interface/agents/llamaindex.md`:**

````markdown
# LlamaIndex Integration

## Overview

The LlamaIndex integration provides adapters for evaluating agents built with
LlamaIndex's workflow-based agent system.

## Installation

```bash
pip install maseval[llamaindex]
```
````

## Quick Start

[Example code showing basic usage]

## Supported Agent Types

- AgentWorkflow
- FunctionAgent
- ReActAgent
- CodeActAgent

## Message History

[Explain ChatMessage -> MessageHistory conversion]

## Async Handling

[Explain how async workflows are handled]

## User Simulation

[Show LlamaIndexUser usage]

## API Reference

::: maseval.interface.agents.llamaindex.LlamaIndexAgentAdapter
::: maseval.interface.agents.llamaindex.LlamaIndexUser

````

### 6.2 Update Existing Docs

**Update `docs/interface/agents/index.md`** (or create if doesn't exist):
- Add LlamaIndex to list of supported frameworks
- Link to llamaindex.md

**Update `mkdocs.yml`**:
```yaml
nav:
  - Interface:
    - Agents:
      - smolagents: interface/agents/smolagents.md
      - langgraph: interface/agents/langgraph.md
      - llamaindex: interface/agents/llamaindex.md  # NEW
````

**Update main `README.md`**:

- Add LlamaIndex to supported frameworks list
- Update installation instructions

## 7. Edge Cases and Challenges

### 7.1 Async Execution in Sync Context

**Challenge**: LlamaIndex workflows are async, MASEval is sync.

**Solutions**:

1. Use `asyncio.run()` carefully to avoid nested event loop issues
2. Provide sync wrapper if workflow has `run_sync()`
3. Handle Jupyter/IPython environments (which have running event loops)

```python
def _run_in_sync_context(self, coro):
    """Run async coroutine in sync context, handling edge cases."""
    try:
        # Try to get running loop (raises RuntimeError if none)
        loop = asyncio.get_running_loop()
        # We're in async context - use nest_asyncio or run_in_executor
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(coro)
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)
```

### 7.2 Message History Access

**Challenge**: LlamaIndex workflows may not expose full message history directly.

**Solutions**:

1. Check if workflow/agent has accessible memory attribute
2. Use event streaming to capture messages during execution
3. Cache messages from execution result
4. Document limitations if full history unavailable

### 7.3 Multi-Agent Workflows

**Challenge**: `AgentWorkflow` can manage multiple agents with handoffs.

**Considerations**:

1. Initial implementation focuses on single-agent workflows
2. For multi-agent, track which agent generated which message
3. Include agent names in traces
4. Document multi-agent support status

### 7.4 Streaming Support

**Challenge**: LlamaIndex workflows support streaming responses.

**Decision**:

- Initial implementation: non-streaming only
- Future enhancement: support streaming mode
- Document streaming limitations

### 7.5 Memory Persistence

**Challenge**: LlamaIndex workflows can have persistent memory across runs.

**Solutions**:

1. Support both stateless and stateful modes
2. Allow optional memory instance in constructor
3. Document memory management patterns
4. Test both fresh and persistent memory scenarios

## 8. Implementation Checklist

### 8.1 Core Implementation

- [ ] Create `maseval/interface/agents/llamaindex.py`
- [ ] Implement `LlamaIndexAgentAdapter`
  - [ ] `__init__` with agent type detection
  - [ ] `_run_agent` with async handling
  - [ ] `get_messages` with ChatMessage conversion
  - [ ] `gather_traces` with LlamaIndex-specific data
  - [ ] `gather_config` with workflow/agent config
  - [ ] `_convert_llamaindex_messages` helper
- [ ] Implement `LlamaIndexUser`
  - [ ] `get_tool` returning AsyncBaseTool
- [ ] Add lazy import checking
- [ ] Add proper error messages

### 8.2 Dependencies

- [ ] Add `llamaindex` to `pyproject.toml` optional dependencies
- [ ] Update `all` extras to include llamaindex
- [ ] Update `typing` dev group for type checking

### 8.3 Testing

- [ ] Create `tests/test_interface/test_agent_integration/test_llamaindex_agent.py`
- [ ] Write integration tests (basic execution, message conversion, async handling, etc.)
- [ ] Add `"llamaindex"` to contract test parametrization
- [ ] Verify all contract tests pass
- [ ] Add pytest marker configuration
- [ ] Test with Python 3.10, 3.11, 3.12

### 8.4 Documentation

- [ ] Create `docs/interface/agents/llamaindex.md`
- [ ] Add docstrings to all classes and methods
- [ ] Update `mkdocs.yml` navigation
- [ ] Update main `README.md`
- [ ] Add code examples
- [ ] Document limitations (streaming, multi-agent)

### 8.5 CI/CD

- [ ] Ensure CI installs llama-index-core for relevant tests
- [ ] Update GitHub Actions workflows if needed
- [ ] Verify test markers work in CI

### 8.6 Quality Checks

- [ ] Run `ruff format .`
- [ ] Run `ruff check . --fix`
- [ ] Run full test suite: `pytest -v`
- [ ] Run contract tests: `pytest -m contract -v`
- [ ] Run llamaindex tests: `pytest -m llamaindex -v`
- [ ] Build docs: `mkdocs build --strict`
- [ ] Manual testing with real LlamaIndex agents

## 9. Open Questions

### 9.1 Resolved Through Research

1. ✅ **Which LlamaIndex agent API to support?**

   - Focus on workflow-based agents (`AgentWorkflow`, `BaseWorkflowAgent`)
   - These are the modern, recommended API

2. ✅ **How to handle async/sync?**

   - Use `asyncio.run()` with proper event loop detection
   - Prefer `run_sync()` wrapper if available

3. ✅ **Message history access?**
   - Cache messages from execution result
   - Check for memory attribute if available
   - Pattern 2: Stateless/Result-based

### 9.2 To Be Determined During Implementation

1. **Exact message history extraction method**

   - Need to inspect `AgentOutput` structure
   - May need to use workflow events/streaming
   - Will determine best approach through experimentation

2. **Token usage tracking**

   - Check if `AgentOutput` includes token counts
   - May need to extract from LLM response metadata
   - Could be unavailable (document limitation)

3. **Tool call format**

   - Verify how tool calls appear in ChatMessage
   - May be in `additional_kwargs` or separate attribute
   - Ensure proper conversion to OpenAI format

4. **Multi-agent workflow support**
   - Start with single-agent workflows
   - Assess complexity of multi-agent support
   - Document as future enhancement if complex

## 10. Success Criteria

### 10.1 Functional Requirements

- ✅ LlamaIndexAgentAdapter can wrap AgentWorkflow and BaseWorkflowAgent
- ✅ `run()` executes agent and returns final answer
- ✅ `get_messages()` returns OpenAI-compatible message history
- ✅ Callbacks (on_run_start, on_run_end) work correctly
- ✅ gather_traces() includes workflow execution details
- ✅ gather_config() includes agent/workflow configuration
- ✅ LlamaIndexUser provides tool for user simulation

### 10.2 Contract Compliance

- ✅ All contract tests pass with `framework="llamaindex"`
- ✅ Behavior identical to smolagents/langgraph adapters
- ✅ Message history format matches OpenAI spec
- ✅ Callback lifecycle matches other frameworks

### 10.3 Quality Standards

- ✅ Code passes ruff formatting and linting
- ✅ All tests pass (unit, integration, contract)
- ✅ Documentation complete and accurate
- ✅ Proper error messages for missing dependencies
- ✅ Type hints correct and comprehensive

### 10.4 Integration

- ✅ Works with existing MASEval benchmarks without modification
- ✅ Can be used interchangeably with smolagents/langgraph
- ✅ No breaking changes to core library
- ✅ CI/CD pipeline passes

## 11. Timeline Estimate

### 11.1 Development Phases

1. **Core Implementation** (4-6 hours)

   - LlamaIndexAgentAdapter basic structure
   - Async handling
   - Message conversion
   - LlamaIndexUser

2. **Testing** (3-4 hours)

   - Integration tests
   - Contract test integration
   - Edge case handling

3. **Documentation** (2-3 hours)

   - API docs
   - Examples
   - Update existing docs

4. **Quality & Integration** (2-3 hours)
   - Code review
   - CI/CD updates
   - Final testing

**Total Estimate**: 11-16 hours

### 11.2 Risks and Mitigation

- **Risk**: Async handling complexity

  - **Mitigation**: Use well-tested patterns, handle edge cases early

- **Risk**: Message history access limitations

  - **Mitigation**: Cache from result, document limitations clearly

- **Risk**: Undocumented LlamaIndex behavior
  - **Mitigation**: Experimentation, community consultation if needed

## 12. Future Enhancements

### 12.1 Potential Improvements

1. **Streaming support**

   - Capture streaming responses
   - Provide streaming callback hooks

2. **Multi-agent workflows**

   - Full support for agent handoffs
   - Track per-agent messages and traces

3. **Memory persistence**

   - Support custom memory implementations
   - Integrate with MASEval's environment state

4. **Advanced tracing**

   - Capture workflow graph structure
   - Track agent decision points
   - Integrate with LlamaIndex observability

5. **Performance optimization**
   - Minimize async overhead
   - Optimize message conversion
   - Cache compiled workflows

### 12.2 Out of Scope (Initial Implementation)

- Legacy LlamaIndex agent APIs (pre-workflow)
- RAG/query engine integration (focus on agents)
- Custom workflow creation utilities
- LlamaIndex-specific benchmarks
- Integration with LlamaIndex observability tools (future)

## 13. References

### 13.1 LlamaIndex Documentation

- [LlamaAgents Overview](https://developers.llamaindex.ai/python/llamaagents/overview/)
- [Agent API Reference](https://developers.llamaindex.ai/python/framework-api-reference/agent/)
- [Workflows Documentation](https://developers.llamaindex.ai/python/llamaagents/workflows/)
- [Memory API](https://developers.llamaindex.ai/python/framework-api-reference/memory/)

### 13.2 MASEval Documentation

- `AGENTS.md`: Development guidelines
- `maseval/interface/README.md`: Interface architecture
- `maseval/interface/agents/smolagents.py`: Reference implementation
- `maseval/interface/agents/langgraph.py`: Stateless pattern reference
- `tests/test_contract/test_agent_adapter_contract.py`: Contract requirements

### 13.3 Related Issues/PRs

- None yet (first LlamaIndex integration)

## 14. Review and Approval

This plan should be reviewed for:

- ✅ Alignment with MASEval architecture principles
- ✅ Technical feasibility of proposed solutions
- ✅ Completeness of edge case handling
- ✅ Adequacy of testing strategy
- ✅ Documentation coverage

**Reviewer**: @cemde

---

**Plan Status**: Draft - Awaiting Review
**Created**: 2025-11-24
**Author**: GitHub Copilot
