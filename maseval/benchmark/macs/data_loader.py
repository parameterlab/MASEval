"""Data loading utilities for MACS benchmark.

This module provides functions to:
1. Download original data from AWS GitHub to data/original/
2. Restructure data into data/restructured/ (tasks.json, agents.json per domain)
3. Load restructured data for use in benchmarks

No side effects on import. Data download/processing must be explicitly called.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from maseval import Task, TaskCollection


# =============================================================================
# Constants
# =============================================================================

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
VALID_DOMAINS = ("travel", "mortgage", "software")

# AWS Multi-Agent Collaboration Scenarios benchmark data
# Source: https://github.com/aws-samples/multiagent-collab-scenario-benchmark
URLS = {
    "data": {
        "software": {
            "agents": "https://raw.githubusercontent.com/aws-samples/multiagent-collab-scenario-benchmark/refs/heads/main/datasets/software/agents.json",
            "scenarios": "https://raw.githubusercontent.com/aws-samples/multiagent-collab-scenario-benchmark/refs/heads/main/datasets/software/scenarios_30.json",
        },
        "travel": {
            "agents": "https://raw.githubusercontent.com/aws-samples/multiagent-collab-scenario-benchmark/refs/heads/main/datasets/travel/agents.json",
            "scenarios": "https://raw.githubusercontent.com/aws-samples/multiagent-collab-scenario-benchmark/refs/heads/main/datasets/travel/scenarios_30.json",
        },
        "mortgage": {
            "agents": "https://raw.githubusercontent.com/aws-samples/multiagent-collab-scenario-benchmark/refs/heads/main/datasets/mortgage/agents.json",
            "scenarios": "https://raw.githubusercontent.com/aws-samples/multiagent-collab-scenario-benchmark/refs/heads/main/datasets/mortgage/scenarios_30.json",
        },
    },
    "evaluation": {
        "prompt_templates": "https://raw.githubusercontent.com/aws-samples/multiagent-collab-scenario-benchmark/refs/heads/main/src/prompt_templates.py",
    },
}


# =============================================================================
# Download Functions
# =============================================================================


def download_file(url: str, timeout: int = 15) -> str:
    """Download a file from URL and return as text."""
    try:
        with urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
            return raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to download from {url}: {e}") from e


def download_json(url: str) -> object:
    """Download and parse JSON from URL."""
    text = download_file(url)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to decode JSON from {url}: {e}") from e


def download_original_data(
    data_dir: Optional[Path] = None,
    domain: Optional[str] = None,
    verbose: int = 1,
) -> Path:
    """Download original data from AWS GitHub to data/original/.

    Args:
        data_dir: Base data directory (default: module's data/)
        domain: Specific domain to download, or None for all
        verbose: 0=silent, 1=summary, 2=detailed

    Returns:
        Path to the original data directory
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    original_dir = data_dir / "original"

    domains = [domain] if domain else list(URLS["data"].keys())

    for d in domains:
        if d not in URLS["data"]:
            raise ValueError(f"Unknown domain: {d}")

        domain_dir = original_dir / d
        domain_dir.mkdir(parents=True, exist_ok=True)

        for name, url in URLS["data"][d].items():
            content = download_json(url)
            out_path = domain_dir / f"{name}.json"
            with out_path.open("w") as f:
                json.dump(content, f, indent=2)
            if verbose >= 2:
                print(f"Downloaded {url} -> {out_path}")

    if verbose >= 1:
        print(f"Downloaded original data to {original_dir}")

    return original_dir


def download_prompt_templates(
    data_dir: Optional[Path] = None,
    verbose: int = 1,
) -> Path:
    """Download prompt templates from AWS GitHub.

    Args:
        data_dir: Base data directory (default: module's data/)
        verbose: 0=silent, 1=summary, 2=detailed

    Returns:
        Path to the prompt_templates directory
    """
    import ast

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    templates_dir = data_dir.parent / "prompt_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    url = URLS["evaluation"]["prompt_templates"]
    text = download_file(url)

    # Parse Python file to extract prompt constants
    tree = ast.parse(text)
    values: Dict[str, str] = {}

    VARS = {
        "USER_GSR_PROMPT": "user",
        "SYSTEM_GSR_PROMPT": "system",
        "ISSUES_PROMPT": "issues",
    }

    def _const_str(node) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts = []
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    parts.append(part.value)
                else:
                    return None
            return "".join(parts)
        return None

    def _eval_node(node):
        s = _const_str(node)
        if s is not None:
            return s
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left_val = _eval_node(node.left)
            right_val = _eval_node(node.right)
            if left_val is None or right_val is None:
                return None
            return left_val + right_val
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            val = _eval_node(node.value)
            if val is not None:
                values[target.id] = val

    # Escape placeholders for template use
    def _escape_placeholders(s: str) -> str:
        if not s:
            return ""
        s = s.replace("{{", "__DBL_OPEN__").replace("}}", "__DBL_CLOSE__")
        s = s.replace("{", "{{").replace("}", "}}")
        s = s.replace("__DBL_OPEN__", "{{").replace("__DBL_CLOSE__", "}}")
        return s

    # Write template files
    for var_name, file_key in VARS.items():
        content = _escape_placeholders(values.get(var_name, ""))
        out_path = templates_dir / f"{file_key}.txt"
        with out_path.open("w") as f:
            f.write(content)
        if verbose >= 2:
            print(f"Wrote {out_path}")

    if verbose >= 1:
        print(f"Downloaded prompt templates to {templates_dir}")

    return templates_dir


# =============================================================================
# Restructuring Functions
# =============================================================================


def _dedupe_tools_by_name(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate tools by tool_name, raising on conflicts."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    no_name: List[Dict[str, Any]] = []

    for t in tools:
        if not isinstance(t, dict):
            no_name.append(t)
            continue
        name = t.get("tool_name")
        if not name:
            no_name.append(t)
        else:
            grouped[name].append(t)

    deduped = list(no_name)
    for name, items in grouped.items():
        if len(items) == 1:
            deduped.append(items[0])
        else:
            first = items[0]
            for other in items[1:]:
                if first != other:
                    raise ValueError(f"Conflicting tools for tool_name='{name}'")
            deduped.append(first)

    return deduped


def _create_tools_list(agents_obj: object) -> List[Dict[str, Any]]:
    """Extract and deduplicate tools from agents data."""
    tools: List[Dict[str, Any]] = []

    if isinstance(agents_obj, dict) and isinstance(agents_obj.get("agents"), list):
        agents_list = agents_obj["agents"]
    elif isinstance(agents_obj, list):
        agents_list = agents_obj
    else:
        return tools

    for agent in agents_list:
        if not isinstance(agent, dict):
            continue
        for t in agent.get("tools", []):
            if isinstance(t, dict):
                tools.append(t)

    return _dedupe_tools_by_name(tools)


def _create_agents_list(agents_obj: object) -> Dict[str, Any]:
    """Create agents config with tool names only (not full tool dicts)."""

    def _process_agent(agent: Dict[str, Any]) -> Dict[str, Any]:
        a_copy = {k: v for k, v in agent.items() if k != "tools"}
        tool_names = [t.get("tool_name") for t in agent.get("tools", []) if isinstance(t, dict) and t.get("tool_name")]
        a_copy["tools"] = tool_names
        return a_copy

    if isinstance(agents_obj, dict) and isinstance(agents_obj.get("agents"), list):
        processed = [_process_agent(a) for a in agents_obj["agents"] if isinstance(a, dict)]
        out: Dict[str, Any] = {"agents": processed}
        if "primary_agent_id" in agents_obj:
            out["primary_agent_id"] = agents_obj["primary_agent_id"]
        if "human_id" in agents_obj:
            out["human_id"] = agents_obj["human_id"]
        return out

    return {}


def _create_tasks_list(scenarios_obj: object, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert scenarios to task format with sequential IDs."""
    tasks: List[Dict[str, Any]] = []

    if isinstance(scenarios_obj, dict) and isinstance(scenarios_obj.get("scenarios"), list):
        scenarios_list = scenarios_obj["scenarios"]
    elif isinstance(scenarios_obj, list):
        scenarios_list = scenarios_obj
    else:
        return tasks

    for idx, scen in enumerate(scenarios_list, start=1):
        if not isinstance(scen, dict):
            continue

        query = scen.get("input_problem") or scen.get("query") or ""
        # Always generate sequential task ID
        tid = f"task-{idx:06d}"

        task = {
            "id": tid,
            "query": query,
            "environment_data": {"tools": tools},
            "evaluation_data": {"assertions": scen.get("assertions", [])},
            "metadata": {k: v for k, v in scen.items() if k not in ("input_problem", "query", "assertions")},
        }
        tasks.append(task)

    return tasks


def restructure_data(
    data_dir: Optional[Path] = None,
    domain: Optional[str] = None,
    verbose: int = 1,
) -> Path:
    """Restructure original data into tasks.json and agents.json per domain.

    Reads from data/original/, writes to data/restructured/.

    Args:
        data_dir: Base data directory (default: module's data/)
        domain: Specific domain to restructure, or None for all
        verbose: 0=silent, 1=summary, 2=detailed

    Returns:
        Path to the restructured data directory
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    original_dir = data_dir / "original"
    restructured_dir = data_dir / "restructured"

    domains = [domain] if domain else list(VALID_DOMAINS)

    for d in domains:
        orig_domain_dir = original_dir / d
        if not orig_domain_dir.exists():
            raise FileNotFoundError(f"Original data not found: {orig_domain_dir}")

        # Load original data
        with (orig_domain_dir / "agents.json").open() as f:
            agents_data = json.load(f)
        with (orig_domain_dir / "scenarios.json").open() as f:
            scenarios_data = json.load(f)

        # Restructure
        tools = _create_tools_list(agents_data)
        agents = _create_agents_list(agents_data)
        tasks = _create_tasks_list(scenarios_data, tools)

        # Save restructured data
        out_domain_dir = restructured_dir / d
        out_domain_dir.mkdir(parents=True, exist_ok=True)

        with (out_domain_dir / "agents.json").open("w") as f:
            json.dump(agents, f, indent=2)
        with (out_domain_dir / "tasks.json").open("w") as f:
            json.dump(tasks, f, indent=2)

        if verbose >= 2:
            print(f"Restructured {d}: {len(agents.get('agents', []))} agents, {len(tasks)} tasks")

    if verbose >= 1:
        print(f"Restructured data to {restructured_dir}")

    return restructured_dir


def ensure_data_exists(
    data_dir: Optional[Path] = None,
    force_download: bool = False,
    verbose: int = 1,
) -> Path:
    """Ensure restructured data exists, downloading if needed.

    Args:
        data_dir: Base data directory (default: module's data/)
        force_download: If True, re-download even if data exists
        verbose: 0=silent, 1=summary, 2=detailed

    Returns:
        Path to the restructured data directory
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    restructured_dir = data_dir / "restructured"

    # Check if all domains exist
    all_exist = all((restructured_dir / d / "tasks.json").exists() and (restructured_dir / d / "agents.json").exists() for d in VALID_DOMAINS)

    if all_exist and not force_download:
        return restructured_dir

    # Download and restructure
    download_original_data(data_dir, verbose=verbose)
    download_prompt_templates(data_dir, verbose=verbose)
    restructure_data(data_dir, verbose=verbose)

    return restructured_dir


def process_data(verbose: int = 1) -> Path:
    """Download and process all MACS data. Convenience wrapper.

    Args:
        verbose: 0=silent, 1=summary, 2=detailed

    Returns:
        Path to the restructured data directory
    """
    return ensure_data_exists(force_download=True, verbose=verbose)


# =============================================================================
# Data Loading Functions (for use in benchmarks)
# =============================================================================


def load_tasks(
    domain: str,
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> TaskCollection:
    """Load tasks for a MACS domain.

    Args:
        domain: One of "travel", "mortgage", or "software"
        data_dir: Base data directory (default: module's data/).
                  Tasks are loaded from data_dir/restructured/{domain}/tasks.json
        limit: Maximum number of tasks to load

    Returns:
        TaskCollection containing Task objects

    Raises:
        ValueError: If domain is not valid
        FileNotFoundError: If tasks.json doesn't exist
    """
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {VALID_DOMAINS}")

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    tasks_path = data_dir / "restructured" / domain / "tasks.json"

    with tasks_path.open() as f:
        tasks_list: List[Dict[str, Any]] = json.load(f)

    if limit:
        tasks_list = tasks_list[:limit]

    tasks = []
    for t in tasks_list:
        task_kwargs: Dict[str, Any] = {
            "query": t["query"],
            "environment_data": t.get("environment_data", {}),
            "evaluation_data": t.get("evaluation_data", {}),
            "metadata": t.get("metadata", {}),
        }
        # Store task ID in metadata (format: task-NNNNNN)
        if t.get("id"):
            task_kwargs["metadata"]["task_id"] = t["id"]
        tasks.append(Task(**task_kwargs))

    return TaskCollection(tasks)


def load_agent_config(
    domain: str,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load agent hierarchy configuration for a domain.

    The returned configuration contains:
    - agents: List of agent specifications with agent_id, agent_name,
              agent_instruction, reachable_agents, and tools (as names)
    - primary_agent_id: ID of the supervisor/orchestrator agent
    - human_id: ID used for human/user in trajectories

    Args:
        domain: One of "travel", "mortgage", or "software"
        data_dir: Base data directory (default: module's data/).
                  Config is loaded from data_dir/restructured/{domain}/agents.json

    Returns:
        Dict with agent configuration

    Raises:
        ValueError: If domain is not valid
        FileNotFoundError: If agents.json doesn't exist
    """
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {VALID_DOMAINS}")

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    config_path = data_dir / "restructured" / domain / "agents.json"

    with config_path.open() as f:
        return json.load(f)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    process_data(verbose=2)
