# 5-A-Day Benchmark

A framework-agnostic benchmark for testing LLM agents on diverse tool usage tasks. Tests agents across 5 different scenarios requiring various combinations of tool interactions.

## Quick Start

```bash
# Run the benchmark
python five_a_day_benchmark.py

# Try the MCP demo
python demo_mcp.py
```

## Configuration

Edit variables at the top of `five_a_day_benchmark.py`:

```python
framework = "smolagents"    # or "langgraph", "llamaindex"
config_type = "single"      # or "multi" for multi-agent
limit = None               # or integer to limit number of tasks
```

## Features

- **Framework-Agnostic Tools**: All tools are implemented as `BaseTool` classes that can be converted to any supported framework
- **Multiple Frameworks**: Supports smolagents, langgraph, and llamaindex (via conversion methods)
- **MCP Integration**: Demonstrates Model Context Protocol (MCP) for standardized tool access - see `demo_mcp.py` and `tools/MCP_README.md`
- **Task-Specific Configurations**: Each of the 5 tasks has dedicated single-agent and multi-agent configurations
- **Diverse Tool Types**: Email, Banking, Calculator, Code Execution, Search, Calendar, Fitness tracking, and more
- **Multi-Agent Support**: Orchestrator + specialist pattern for complex task decomposition (smolagents)

## Architecture

### Tool System

All tools inherit from `BaseTool` (in `tools/base.py`) which provides:

- Framework-agnostic `execute()` method
- Conversion methods: `to_smolagents()`, `to_langgraph()`, `to_llamaindex()`
- Tracing and configuration support via maseval mixins

**Available Tools:**

1. **EmailTool** - Send emails and check inbox
2. **BankingTool** - Check balance and transaction history
3. **CalculatorTool** - Safe expression evaluation (RestrictedPython)
4. **CodeExecutionTool** - Execute Python code with test cases
5. **FamilyInfoTool** - Query family member information
6. **StockPriceTool** - Look up stock prices
7. **CalendarTool** - Check calendar availability
8. **RunningAppTool** - Track running activities
9. **GymTrackerTool** - Track gym workouts
10. **HotelSearchTool** - Search and score hotels
11. **MCPCalendarTool** - Calendar access using Model Context Protocol pattern

**MCP Integration:** Task 3 showcases calendar coordination via MCP (Model Context Protocol), demonstrating how maseval works with standardized protocols for external tool access. Run `python demo_mcp.py` to see it in action, or see `tools/MCP_README.md` for implementation details.

### Task Configurations

Each task (0-4) has two configuration files:

- `data/singleagent.json` - Single agent with all tools
- `data/multiagent.json` - Orchestrator + specialist agents

**Task 0: Email & Banking**

- Tools: email, banking
- Single: "Email and Banking Assistant"
- Multi: orchestrator + banking_specialist + email_specialist

**Task 1: Financial Calculations**

- Tools: family_info, stock_price, calculator
- Single: "Financial Planning Assistant"
- Multi: orchestrator + data_specialist + calc_specialist

**Task 2: Code Generation**

- Tools: python_executor
- Single: "Python Code Generator"
- Multi: orchestrator + code_specialist

**Task 3: Calendar Scheduling (MCP Demo)**

- Tools: my_calendar_mcp, other_calendar_mcp
- Single: "Scheduling Assistant"
- Multi: orchestrator + calendar_specialist
- Demonstrates Model Context Protocol for multi-calendar coordination

**Task 4: Hotel Optimization**

- Tools: hotel_search, calculator
- Single: "Hotel Search Optimizer"
- Multi: orchestrator + search_specialist + optimization_specialist

## Usage

### Basic Usage

```bash
# Run with smolagents (default), single-agent config, 1 task
python five_a_day_benchmark.py smolagents 1

# Run with langgraph, single-agent config, all tasks
python five_a_day_benchmark.py langgraph

# Run with multi-agent config
python five_a_day_benchmark.py smolagents 1 --config multiagent

# Run all tasks with multi-agent config
python five_a_day_benchmark.py smolagents --config multiagent
```

### Command Line Arguments

```
python five_a_day_benchmark.py <framework> [limit] [--config <type>]

framework: smolagents | langgraph | llamaindex
limit: Number of tasks to run (1-5), omit for all
--config: singleagent | multiagent (default: singleagent)
```

### Examples

```bash
# Compare single vs multi-agent on task 0
python five_a_day_benchmark.py smolagents 1 --config singleagent
python five_a_day_benchmark.py smolagents 1 --config multiagent

# Test langgraph with all tasks
python five_a_day_benchmark.py langgraph --config singleagent

# Run first 3 tasks with multi-agent setup
python five_a_day_benchmark.py smolagents 3 --config multiagent
```

## File Structure

```
5_a_day_benchmark/
├── five_a_day_benchmark.py       # Main benchmark orchestration
├── tools/
│   ├── __init__.py               # Exports all tools
│   ├── base.py                   # BaseTool + framework adapters
│   ├── email.py                  # EmailTool implementation
│   ├── banking.py                # BankingTool implementation
│   ├── calculator.py             # CalculatorTool (RestrictedPython)
│   ├── code_execution.py         # CodeExecutionTool
│   ├── family_info.py            # FamilyInfoTool
│   ├── stock_price.py            # StockPriceTool
│   ├── calendar.py               # CalendarTool
│   ├── running_app.py            # RunningAppTool
│   ├── gym_tracker.py            # GymTrackerTool
│   └── hotel_search.py           # HotelSearchTool
├── data/
│   ├── tasks.json                # 5 benchmark tasks
│   ├── singleagent.json          # Single-agent configs (5 tasks)
│   └── multiagent.json           # Multi-agent configs (5 tasks)
├── results/                      # Benchmark outputs (auto-created)
└── README.md                     # This file
```

## Implementation Notes

### Framework Conversion Pattern

Tools use composition-based adapters to avoid `__call__` conflicts:

```python
class BaseTool(ABC, TraceableMixin, ConfigurableMixin):
    def to_smolagents(self):
        return SmolagentsToolAdapter(self)

    def to_langgraph(self):
        return LangGraphToolAdapter(self)

    def to_llamaindex(self):
        return LlamaIndexToolAdapter(self)
```

### Multi-Agent Architecture

Smolagents supports multi-agent via `managed_agents`:

- Orchestrator agent coordinates specialists
- Each specialist has subset of tools
- Orchestrator delegates tasks to specialists

LangGraph and LlamaIndex multi-agent support is TODO.

### Environment Loading

`FiveADayEnvironment` loads tools from `tasks.json`:

1. Reads `environment_data.tools` list
2. Instantiates tool classes with task-specific data
3. Converts to framework-specific types
4. Returns tools for agent setup

### Task-Specific Configs

Each task has tailored agent instructions and tool assignments:

- Configs are arrays (5 items, one per task)
- `load_agent_config(task_id)` retrieves task-specific config
- Framework selection is separate from agent config

## Requirements

**Core Dependencies:**

- maseval
- RestrictedPython (8.1+)

**Framework Dependencies:**

- smolagents (for smolagents support)
- langgraph, langchain-google-genai (for langgraph support)
- llama-index-core, llama-index-llms-openai-like (for llamaindex support - TODO)

**Environment Variables:**

- `GOOGLE_API_KEY` - Required for Gemini models

## Development

### Adding New Tools

1. Create tool class inheriting from `BaseTool`
2. Implement `execute()` method
3. Add to `tools/__init__.py` exports
4. Add to tool_mapping in `FiveADayEnvironment.create_tools()`
5. Update task configurations as needed

### Adding New Tasks

1. Add task to `data/tasks.json` with environment_data and evaluation_data
2. Add single-agent config to `data/singleagent.json`
3. Add multi-agent config to `data/multiagent.json`
4. Ensure task_id matches array index

### Testing

```bash
# Check linting
ruff check examples/5_a_day_benchmark/

# Auto-fix linting issues
ruff check --fix examples/5_a_day_benchmark/

# Format code
ruff format examples/5_a_day_benchmark/

# Test tool instantiation
python -c "from tools import EmailTool; print(EmailTool([]).execute())"

# Test config loading
python -c "from five_a_day_benchmark import load_agent_config; print(load_agent_config(0))"
```

## Future Work

- [ ] Implement evaluators based on evaluation_data
- [ ] Add LlamaIndex support (adapter + multi-agent)
- [ ] Add LangGraph multi-agent support
- [ ] Add more diverse tool types
- [ ] Expand to 10+ tasks
- [ ] Add visualization of tool usage patterns
- [ ] Benchmark latency and token usage

## License

See main repository LICENSE file.

# TODO

- run entire benchmark
- change pattern of loading environment. more automatic.
- implement langgraph and llamaindex
- update evaluators with existing evaluators
- add more unit tests
-
