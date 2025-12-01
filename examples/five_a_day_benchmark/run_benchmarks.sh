#! /bin/bash

# Navigate to the project root
cd ../../


# # Run benchmarks for SmolAgents
# uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py --framework smolagents --config-type single --seed 0
# uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py --framework smolagents --config-type multi --seed 0

# Run benchmarks for Langgraph
uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py --framework langgraph --config-type single --seed 0
uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py --framework langgraph --config-type multi --seed 0

# Run benchmarks for LLamaIndex
uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py --framework llamaindex --config-type single --seed 0
uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py --framework llamaindex --config-type multi --seed 0

# Navigate back to the example directory
cd examples/five_a_day_benchmark/

echo "Benchmarking completed."