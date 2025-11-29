#! /bin/bash

# Navigate to the project root
cd ../../


# Run benchmarks for SmolAgents
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --framework smolagents --config-type single
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --framework smolagents --config-type multi

# Run benchmarks for Langgraph
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --framework langgraph --config-type single
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --framework langgraph --config-type multi

# Run benchmarks for LLamaIndex
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --framework llamaindex --config-type single
uv run python examples/5_a_day_benchmark/five_a_day_benchmark.py --framework llamaindex --config-type multi 

# Navigate back to the example directory
cd examples/5_a_day_benchmark/

echo "Benchmarking completed."