"""Shared utility functions for evaluators."""

import re
import os
from typing import List, Optional

from litellm import completion
from maseval import MessageHistory


def extract_assistant_response(trace: MessageHistory) -> str:
    """Extract all assistant messages from trace as a single string.

    Handles multiple message formats:
    - Standard: role="assistant"
    - Smolagents: role="tool-response" (final_answer tool)
    - LangGraph: role="ai" or role="assistant"

    Args:
        trace: The message history trace

    Returns:
        Combined assistant response text
    """
    response_parts = []

    # Collect messages from different roles
    for msg in trace:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Standard assistant messages
        if role == "assistant":
            response_parts.append(str(content))

        # Smolagents tool responses (especially final_answer)
        elif role == "tool-response":
            # Extract text from content array if present
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        # Remove "Observation:" prefix if present
                        text = text.replace("Observation:\n", "").strip()
                        response_parts.append(text)
            else:
                text = str(content).replace("Observation:\n", "").strip()
                response_parts.append(text)

        # LangGraph AI messages
        elif role == "ai":
            response_parts.append(str(content))

    return " ".join([part for part in response_parts if part])


def extract_tool_calls(trace: MessageHistory) -> List[str]:
    """Extract list of tool names that were called.

    Args:
        trace: The message history trace

    Returns:
        List of tool names used
    """
    tools_used = set()

    # From tool responses
    tool_messages = [msg for msg in trace if msg.get("role") == "tool"]
    for msg in tool_messages:
        if "name" in msg:
            tools_used.add(msg["name"])

    # From tool calls
    tool_call_messages = [msg for msg in trace if msg.get("tool_calls")]
    for msg in tool_call_messages:
        for tool_call in msg.get("tool_calls", []):
            if "function" in tool_call:
                tools_used.add(tool_call["function"]["name"])

    return list(tools_used)


def extract_python_code(trace: MessageHistory) -> Optional[str]:
    """Extract Python code from trace.

    Looks for:
    1. Code blocks marked with ```python
    2. Function definitions starting with 'def'

    Args:
        trace: The message history trace

    Returns:
        Extracted Python code or None if not found
    """
    full_trace = extract_assistant_response(trace)

    # Look for code blocks with markdown
    code_block_pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(code_block_pattern, full_trace, re.DOTALL)

    if matches:
        return matches[-1]  # Return last code block

    # Look for function definitions without markdown
    code_pattern = r"def\s+\w+\s*\([^)]*\):"
    if re.search(code_pattern, full_trace):
        lines = full_trace.split("\n")
        code_lines = []
        in_function = False

        for line in lines:
            if re.match(r"def\s+\w+", line):
                in_function = True
            if in_function:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines)

    return None


def check_amount_in_text(amount: float, text: str) -> bool:
    """Check if a monetary amount appears in text (various formats).

    Args:
        amount: The amount to look for
        text: The text to search in

    Returns:
        True if amount found in any common format
    """
    return str(int(amount)) in text or f"${int(amount)}" in text or f"{int(amount):,}" in text or f"${int(amount):,}" in text


def call_llm_judge(prompt: str, model: str = "gemini/gemini-2.5-flash") -> str:
    """Call LLM to evaluate a response.

    Args:
        prompt: The evaluation prompt
        model: The model to use (default: gemini/gemini-2.5-flash)

    Returns:
        The LLM's response
    """
    try:
        response = completion(model=model, messages=[{"role": "user", "content": prompt}], api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.0)
        content = response.choices[0].message.content
        return content if content else ""
    except Exception as e:
        print(f"Error calling LLM judge: {e}")
        return ""
