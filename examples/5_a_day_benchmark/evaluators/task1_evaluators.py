"""Task 1 Evaluators: Stock Inheritance Calculation.

Task: User asks to split Apple stock value equally among children.
Success criteria: Agent correctly calculates total value and per-child amount.
"""

import re
from typing import Any, Dict, Optional

from maseval import Evaluator, Environment, Task, User, MessageHistory
from .utils import extract_assistant_response, extract_tool_calls, check_amount_in_text


class ArithmeticAccuracyEvaluator(Evaluator):
    """Evaluates arithmetic correctness of stock value calculations.
    
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
        
        # Look for total value (with tolerance for formatting)
        # Extract numbers from response
        numbers = re.findall(r'\$?[\d,]+\.?\d*', full_response)
        numbers = [float(n.replace('$', '').replace(',', '')) for n in numbers]
        
        # Check if expected values appear (with small tolerance)
        tolerance = 1.0  # $1 tolerance for floating point
        
        total_found = any(abs(n - expected_total) < tolerance for n in numbers)
        per_child_found = any(abs(n - expected_per_child) < tolerance for n in numbers)
        
        # Also check string matching for exact values
        total_str_found = check_amount_in_text(expected_total, full_response)
        per_child_str_found = check_amount_in_text(expected_per_child, full_response)
        
        total_correct = total_found or total_str_found
        per_child_correct = per_child_found or per_child_str_found
        
        # Calculate error magnitude
        if numbers:
            # Find closest number to expected total
            total_candidates = [n for n in numbers if 30000 < n < 50000]  # Reasonable range
            per_child_candidates = [n for n in numbers if 10000 < n < 15000]
            
            total_error = min([abs(n - expected_total) for n in total_candidates]) if total_candidates else float('inf')
            per_child_error = min([abs(n - expected_per_child) for n in per_child_candidates]) if per_child_candidates else float('inf')
        else:
            total_error = float('inf')
            per_child_error = float('inf')
        
        score = (int(total_correct) + int(per_child_correct)) / 2.0
        
        return {
            "total_value_correct": total_correct,
            "per_child_value_correct": per_child_correct,
            "total_value_error_magnitude": total_error if total_error != float('inf') else None,
            "per_child_error_magnitude": per_child_error if per_child_error != float('inf') else None,
            "arithmetic_score": score,
            "expected_total_value": expected_total,
            "expected_per_child_value": expected_per_child
        }


class InformationRetrievalEvaluator(Evaluator):
    """Evaluates if agent retrieved necessary information from tools.
    
    Measures: Did the agent use the appropriate tools to gather required data?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.ground_truth = task.evaluation_data

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate information retrieval from tools."""
        tools_used = extract_tool_calls(trace)
        
        # Check for required tools
        stock_price_retrieved = (
            "stock_price" in tools_used or 
            "websearch" in tools_used or
            any("StockPriceTool" in str(t) for t in tools_used)
        )
        
        family_info_retrieved = (
            "family_info" in tools_used or
            any("FamilyInfoTool" in str(t) for t in tools_used)
        )
        
        # Check if correct stock symbol was used (AAPL)
        full_trace_str = str(trace.to_list())
        used_aapl = "AAPL" in full_trace_str or "Apple" in full_trace_str
        used_wrong_symbol = "TSLA" in full_trace_str or "Tesla" in full_trace_str
        
        # Check if shares count was identified
        shares_identified = "150" in full_trace_str
        
        # Calculate completeness
        required_retrievals = [
            stock_price_retrieved,
            family_info_retrieved,
            used_aapl,
            shares_identified
        ]
        retrieval_completeness = sum(required_retrievals) / len(required_retrievals)
        
        return {
            "stock_price_retrieved": stock_price_retrieved,
            "family_info_retrieved": family_info_retrieved,
            "used_correct_stock_symbol": used_aapl and not used_wrong_symbol,
            "shares_count_identified": shares_identified,
            "retrieval_completeness": retrieval_completeness,
            "tools_used": tools_used
        }
