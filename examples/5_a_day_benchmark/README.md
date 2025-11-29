# 5-A-Day Benchmark

An educational example demonstrating **maseval** through 5 diverse agent evaluation tasks. Shows how the library handles different tool types, evaluation approaches, and agent configurations.

## What This Example Shows

See maseval's key features in action:

1. **Environment** - Loads task data and instantiates framework-agnostic tools
2. **Framework Adapters** - Same tools work with smolagents, langgraph, llamaindex
3. **Task Structure** - JSON-based task definitions with queries, test data, and evaluation criteria
4. **Evaluation Types** - 7 different evaluator approaches (unit tests, LLM judges, optimization, etc.)
5. **Agent Patterns** - Single-agent vs multi-agent orchestrator configurations
6. **Message Tracing** - Full conversation history captured for evaluation

## Running the Example

```bash
# Set your Google API key (required for Gemini models)
export GOOGLE_API_KEY="your-api-key-here"

# Run with smolagents (default)
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py

# Run with LangGraph
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --framework langgraph

# Run with LlamaIndex
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --framework llamaindex

# Run specific task only (e.g., task 0)
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --task 0

# Run with multi-agent orchestrator pattern
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --config-type multi

# Specify model (default: gemini-2.5-flash)
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --model-id gemini-2.5-flash
```

Results saved to `results/` as JSONL with traces and evaluation scores.

## The 5 Tasks

### Task 0: Email & Banking

Agent verifies tenant payment and drafts confirmation email.

**Tools:** Email, Banking  
**Evaluators:** Financial accuracy (assertion), email quality (LLM judge), privacy leakage (pattern matching)  
**Shows:** Multi-tool coordination, numerical validation, privacy checking

### Task 1: Financial Calculations

Agent calculates stock inheritance split among children.

**Tools:** Family info, Stock prices, Calculator  
**Evaluators:** Arithmetic accuracy (numerical), information retrieval (tool validation)  
**Shows:** Multi-step reasoning, data lookup, computation verification

### Task 2: Code Generation

Agent implements house robber dynamic programming algorithm.

**Tools:** Python code executor  
**Evaluators:** Unit tests (execution), complexity analysis (static), code quality (heuristics)  
**Shows:** Generated code evaluation, test harness integration, AST analysis

### Task 3: Calendar Scheduling

Agent finds overlapping free time slots across two calendars.

**Tools:** MCP Calendar (Model Context Protocol pattern)  
**Evaluators:** Slot matching, MCP integration, constraint logic  
**Shows:** Constraint solving, modern protocol patterns, logical validation

### Task 4: Hotel Optimization

Agent selects best hotel based on weighted criteria (distance, WiFi, price).

**Tools:** Hotel search, Calculator  
**Evaluators:** Optimization quality (ranking), search strategy, reasoning transparency  
**Shows:** Multi-criteria optimization, scoring, explanation evaluation

## How It Works

### 1. Environment Creates Tools

`FiveADayEnvironment` reads `tasks.json` and instantiates tools with test data:

```python
class FiveADayEnvironment(Environment):
    def create_tools(self) -> list:
        # Read tool names from task: ["email", "banking"]
        # Instantiate: EmailTool(inbox_data), BankingTool(transactions)
        # Convert to framework: base_tool.to_smolagents()
        return framework_specific_tools
```

### 2. Framework-Agnostic Tools

Tools inherit from `BaseTool` with automatic framework conversion:

```python
class EmailTool(BaseTool):
    def execute(self, action: str, **kwargs) -> ToolResult:
        if action == "get_inbox":
            return ToolResult(success=True, data=self.inbox)
        # ...

# Works with any framework
smolagents_tool = email_tool.to_smolagents()
langgraph_tool = email_tool.to_langgraph()
```

### 3. Task Definitions

Tasks in `tasks.json` specify everything needed:

```json
{
  "query": "Check my transactions and draft email to Sarah...",
  "environment_data": {
    "tools": ["email", "banking"],
    "email_inbox": [...],
    "bank_transactions": [...]
  },
  "evaluation_data": {
    "expected_deposit_amount": 2000,
    "evaluators": ["FinancialAccuracyEvaluator", "EmailQualityEvaluator"]
  }
}
```

### 4. Evaluation Approaches

Seven evaluator types demonstrated:

- **Assertion**: Check exact values (amounts, counts)
- **Unit Tests**: Execute generated code against test cases
- **LLM Judge**: Assess quality, reasoning, explanations
- **Optimization**: Compare to mathematically optimal solution
- **Pattern Matching**: Detect unwanted patterns (privacy leaks)
- **Static Analysis**: Analyze code structure (complexity, patterns)
- **Tool Validation**: Verify correct tool usage and data retrieval

Each evaluator receives the full message trace and returns metrics.

### 5. Agent Adapters

Unified interface across frameworks:

```python
# Wrap framework-specific agents
adapter = SmolAgentAdapter(smolagents_agent, "agent_id")
adapter = LangGraphAgentAdapter(langgraph_graph, "agent_id")

# Unified execution
result = adapter.run(task.query)
trace = adapter.get_messages()  # Full conversation history
```

### 6. Single vs Multi-Agent

**Single**: One agent with all tools

**Multi**: Orchestrator + specialists

- Orchestrator coordinates task
- Specialists have tool subsets
- Shows delegation patterns across all frameworks:
  - **smolagents**: Uses `managed_agents`
  - **langgraph**: Uses supervisor pattern with conditional routing
  - **llamaindex**: Uses handoff pattern with tool delegation

## Project Structure

```
5_a_day_benchmark/
├── five_a_day_benchmark.py       # Benchmark implementation
├── tools/                        # Framework-agnostic tools
│   ├── base.py                   # BaseTool + framework converters
│   ├── email.py                  # Email inbox/sending
│   ├── banking.py                # Transaction lookup
│   ├── calculator.py             # Safe math evaluation
│   ├── code_execution.py         # Python runner with tests
│   └── ...
├── evaluators/                   # Task-specific evaluators
│   ├── email_banking.py          # Task 0 evaluators
│   ├── finance_calc.py           # Task 1 evaluators
│   ├── code_generation.py        # Task 2 evaluators
│   └── ...
└── data/
    ├── tasks.json                # 5 task definitions
    ├── singleagent.json          # Single-agent configs
    └── multiagent.json           # Multi-agent configs
```

## Key Takeaways

**For maseval users**, this example shows:

- How to structure benchmarks with Environment, Task, Evaluator
- How to make tools work across agent frameworks
- Different evaluation approaches for different task types
- How message tracing enables diverse evaluation metrics
- Pattern for single vs multi-agent comparison

**Requirements:** maseval, RestrictedPython, smolagents, langgraph, langchain-google-genai, llama-index-core, llama-index-llms-litellm, litellm  
**Environment:** Set `GOOGLE_API_KEY` for Gemini models
