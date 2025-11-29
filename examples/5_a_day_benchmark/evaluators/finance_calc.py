"""Evaluators for Task 1: Finance & Stock Calculation.

Task: User asks to split Apple stock value equally among children.
Success criteria: Agent correctly calculates total value and per-child amount.
"""

import re
from typing import Any, Dict, Optional

from maseval import Evaluator, Environment, Task, User, MessageHistory
from .utils import extract_assistant_response, extract_tool_calls, check_amount_in_text


class ArithmeticAccuracyEvaluator(Evaluator):
    """Evaluates arithmetic correctness of stock value calculations.
    
    Evaluation type: Numerical comparison with tolerance
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
                "error": "No assistant response found"
            }
        
        expected_total = self.ground_truth["total_value"]
        expected_per_child = self.ground_truth["value_per_child"]
        
        # Extract numbers from response
        numbers = re.findall(r'\$?[\d,]+\.?\d*', full_response)
        numbers = [float(n.replace('$', '').replace(',', '')) for n in numbers]
        
        # Check if expected values appear (with small tolerance for floating point)
        tolerance = 1.0
        
        total_found = any(abs(n - expected_total) < tolerance for n in numbers)
        per_child_found = any(abs(n - expected_per_child) < tolerance for n in numbers)
        
        # Also check string matching for exact values
        total_str_found = check_amount_in_text(expected_total, full_response)
        per_child_str_found = check_amount_in_text(expected_per_child, full_response)
        
        total_correct = total_found or total_str_found
        per_child_correct = per_child_found or per_child_str_found
        
        score = (int(total_correct) + int(per_child_correct)) / 2.0
        
        return {
            "total_value_correct": total_correct,
            "per_child_value_correct": per_child_correct,
            "arithmetic_score": score,
            "expected_total_value": expected_total,
            "expected_per_child_value": expected_per_child
        }


class InformationRetrievalEvaluator(Evaluator):
    """Evaluates if agent retrieved necessary information from tools.
    
    Evaluation type: Tool validation
    Measures: Did the agent use the appropriate tools to gather required data?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.ground_truth = task.evaluation_data

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate information retrieval from tools."""
        tools_used = extract_tool_calls(trace)
        
        # Check for required tools
        stock_price_retrieved = any(
            "stock_price" in str(t).lower() or "StockPriceTool" in str(t) 
            for t in tools_used
        )
        
        family_info_retrieved = any(
            "family_info" in str(t).lower() or "FamilyInfoTool" in str(t)
            for t in tools_used
        )
        
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
            "tools_used": tools_used
        }
