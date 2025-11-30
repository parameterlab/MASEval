"""Evaluators for Task 1: Finance & Stock Calculation.

Task: User asks to split Apple stock value equally among children.
Success criteria: Agent correctly calculates total value and per-child amount.
"""

from typing import Any, Dict, Optional
import json

from maseval import Evaluator, Environment, Task, User, MessageHistory
from .utils import extract_assistant_response, call_llm_judge


class ArithmeticAccuracyEvaluator(Evaluator):
    """Evaluates arithmetic correctness of stock value calculations.

    Evaluation type: LLM-as-judge
    Measures: Did the agent correctly calculate the total stock value and per-child split?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.ground_truth = task.evaluation_data

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to agent messages only."""
        return {"agents": traces.get("agents", {})}

    def __call__(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate arithmetic accuracy."""
        # Extract messages from agent traces
        agents = traces.get("agents", {})
        message_history = MessageHistory()
        if agents:
            agent_trace = next(iter(agents.values()))
            message_history._messages = agent_trace.get("messages", [])

        full_response = extract_assistant_response(message_history)

        if not full_response:
            return {
                "total_value_correct": False,
                "per_child_value_correct": False,
                "arithmetic_score": 0.0,
                "error": "No assistant response found",
            }

        expected_total = self.ground_truth["total_value"]
        expected_per_child = self.ground_truth["value_per_child"]

        prompt = f"""
        You are evaluating an AI agent's financial calculation.
        
        Expected Values:
        - Total Value: {expected_total}
        - Value Per Child: {expected_per_child}
        
        Agent Response:
        "{full_response}"
        
        Did the agent correctly calculate and state these values?
        Allow for minor formatting differences (e.g., $1,000 vs 1000).
        
        Return a JSON object with:
        - total_value_correct: boolean
        - per_child_value_correct: boolean
        - arithmetic_score: float (0.0 to 1.0)
        """

        try:
            response = call_llm_judge(prompt)
            response = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(response)

            return {
                "total_value_correct": result.get("total_value_correct", False),
                "per_child_value_correct": result.get("per_child_value_correct", False),
                "arithmetic_score": result.get("arithmetic_score", 0.0),
                "expected_total_value": expected_total,
                "expected_per_child_value": expected_per_child,
            }
        except Exception as e:
            return {
                "total_value_correct": False,
                "per_child_value_correct": False,
                "arithmetic_score": 0.0,
                "error": f"LLM evaluation failed: {str(e)}",
            }


class InformationRetrievalEvaluator(Evaluator):
    """Evaluates if agent retrieved necessary information from tools.

    Evaluation type: Tool validation (Deterministic)
    Measures: Did the agent use the appropriate tools to gather required data?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.ground_truth = task.evaluation_data

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to tool traces only."""
        return {"tools": traces.get("tools", {})}

    def __call__(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate information retrieval from tools."""
        # Get tool invocations directly from traces
        tools = traces.get("tools", {})

        # Check for required tools by checking if they have invocations
        stock_price_retrieved = bool(tools.get("stock_price.get", {}).get("invocations", []))
        family_info_retrieved = bool(tools.get("family_info.get_children", {}).get("invocations", []))

        # Check if correct stock symbol was used (AAPL) by checking banking.get_asset invocations
        asset_invocations = tools.get("banking.get_asset", {}).get("invocations", [])
        used_aapl = any(inv.get("inputs", {}).get("asset_name", "").upper() == "AAPL" for inv in asset_invocations)

        # Create tools_used list for backwards compatibility
        tools_used = [name for name, data in tools.items() if data.get("invocations", [])]

        # Calculate completeness
        required_retrievals = [stock_price_retrieved, family_info_retrieved, used_aapl]
        retrieval_completeness = sum(required_retrievals) / len(required_retrievals)

        return {
            "stock_price_retrieved": stock_price_retrieved,
            "family_info_retrieved": family_info_retrieved,
            "used_correct_stock_symbol": used_aapl,
            "retrieval_completeness": retrieval_completeness,
            "tools_used": tools_used,
        }
