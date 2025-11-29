"""Evaluators for Task 1: Finance & Stock Calculation.

Task: User asks to split Apple stock value equally among children.
Success criteria: Agent correctly calculates total value and per-child amount.
"""

from typing import Any, Dict, Optional
import json

from maseval import Evaluator, Environment, Task, User, MessageHistory
from .utils import extract_assistant_response, extract_tool_calls, call_llm_judge


class ArithmeticAccuracyEvaluator(Evaluator):
    """Evaluates arithmetic correctness of stock value calculations.

    Evaluation type: LLM-as-judge
    Measures: Did the agent correctly calculate the total stock value and per-child split?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.ground_truth = task.evaluation_data

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate arithmetic accuracy."""
        full_response = extract_assistant_response(trace)

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

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate information retrieval from tools."""
        tools_used = extract_tool_calls(trace)

        # Check for required tools
        stock_price_retrieved = any("stock_price" in str(t).lower() or "StockPriceTool" in str(t) for t in tools_used)
        family_info_retrieved = any("family_info" in str(t).lower() or "FamilyInfoTool" in str(t) for t in tools_used)

        # Check if correct stock symbol was used (AAPL)
        full_trace_str = str(trace.to_list())
        used_aapl = "AAPL" in full_trace_str or "Apple" in full_trace_str

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
