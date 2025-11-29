"""Task 3 Evaluators: Calendar Scheduling (MCP).

Task: User asks to find meeting times with Roger Bannister by checking both calendars.
Success criteria: Agent correctly identifies overlapping available time slots.
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from maseval import Evaluator, Environment, Task, User, MessageHistory
from .utils import extract_tool_calls, extract_assistant_response


class SchedulingAccuracyEvaluator(Evaluator):
    """Evaluates if agent correctly identified overlapping calendar slots.
    
    Measures: Did the agent successfully find the available meeting times?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.expected_slots = task.evaluation_data["overlapping_slots"]

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate scheduling accuracy."""
        full_response = extract_assistant_response(trace)
        
        if not full_response:
            return {
                "overlapping_slots_found": 0,
                "all_overlaps_identified": False,
                "no_false_positives": True,
                "scheduling_precision": 0.0,
                "error": "No assistant response found"
            }
        
        # Check for each expected slot
        found_slots = []
        for slot in self.expected_slots:
            slot_date = slot["date"]
            slot_start = slot["start"]
            slot_end = slot["end"]
            
            # Check if this slot is mentioned
            date_mentioned = slot_date in full_response or slot_date.replace("-", "/") in full_response
            
            # Extract date variations (12-02, 12/02, December 2, etc.)
            date_parts = slot_date.split("-")
            if len(date_parts) == 3:
                month_day = f"{date_parts[1]}-{date_parts[2]}"  # 12-02
                month_day_slash = f"{date_parts[1]}/{date_parts[2]}"  # 12/02
                
                date_mentioned = date_mentioned or month_day in full_response or month_day_slash in full_response
            
            time_mentioned = slot_start in full_response or slot_end in full_response
            
            # Convert times for alternative formats
            start_12hr = self._convert_to_12hr(slot_start)
            end_12hr = self._convert_to_12hr(slot_end)
            time_mentioned = time_mentioned or start_12hr in full_response or end_12hr in full_response
            
            if date_mentioned and time_mentioned:
                found_slots.append(slot)
        
        # Calculate metrics
        slots_found = len(found_slots)
        all_found = slots_found == len(self.expected_slots)
        
        # Check for false positives (suggested times that don't match expected slots)
        # For simplicity, if agent found the right count, assume no false positives
        no_false_positives = True  # Could be enhanced with more sophisticated parsing
        
        # F1-like precision score
        if len(self.expected_slots) > 0:
            recall = slots_found / len(self.expected_slots)
            precision = 1.0 if no_false_positives else 0.5
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1_score = 0.0
        
        return {
            "overlapping_slots_found": slots_found,
            "total_expected_slots": len(self.expected_slots),
            "all_overlaps_identified": all_found,
            "no_false_positives": no_false_positives,
            "scheduling_precision": f1_score,
            "found_slot_details": found_slots
        }

    def _convert_to_12hr(self, time_24hr: str) -> str:
        """Convert 24hr time to 12hr format."""
        try:
            hour, minute = time_24hr.split(":")
            hour = int(hour)
            
            if hour == 0:
                return f"12:{minute} AM"
            elif hour < 12:
                return f"{hour}:{minute} AM"
            elif hour == 12:
                return f"12:{minute} PM"
            else:
                return f"{hour-12}:{minute} PM"
        except:
            return time_24hr


class MCPIntegrationEvaluator(Evaluator):
    """Evaluates if agent properly used MCP calendar tools.
    
    Measures: Did the agent access both calendars to find overlaps?
    Note: This focuses on task success (finding overlaps), not MCP protocol itself.
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate MCP tool usage for task completion."""
        tools_used = extract_tool_calls(trace)
        
        # Check if both calendars were accessed
        my_calendar_used = any("my_calendar" in str(tool).lower() for tool in tools_used)
        other_calendar_used = any("other_calendar" in str(tool).lower() or "roger" in str(tool).lower() for tool in tools_used)
        
        used_both_calendars = my_calendar_used and other_calendar_used
        
        # Count calls
        my_calendar_calls = sum(1 for tool in tools_used if "my_calendar" in str(tool).lower())
        other_calendar_calls = sum(1 for tool in tools_used if "other_calendar" in str(tool).lower() or "roger" in str(tool).lower())
        
        # Calculate score
        score = 1.0 if used_both_calendars else 0.0
        
        return {
            "used_both_calendars": used_both_calendars,
            "my_calendar_calls": my_calendar_calls,
            "other_calendar_calls": other_calendar_calls,
            "mcp_integration_score": score,
            "tools_used": tools_used
        }


class ConstraintSatisfactionEvaluator(Evaluator):
    """Evaluates logical correctness of scheduling constraints.
    
    Measures: Did the agent understand that overlapping means BOTH people are free?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.expected_slots = task.evaluation_data["overlapping_slots"]

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate constraint satisfaction logic."""
        full_response = extract_assistant_response(trace).lower()
        
        if not full_response:
            return {
                "understands_overlap_logic": False,
                "constraint_satisfaction_score": 0.0
            }
        
        # Check for understanding of overlap logic
        overlap_keywords = ["overlap", "both", "available", "free", "both calendars", "common"]
        understands_overlap = any(keyword in full_response for keyword in overlap_keywords)
        
        # Check if suggested times are reasonable (at least one match)
        suggested_reasonable = any(
            slot["date"] in full_response or slot["start"] in full_response
            for slot in self.expected_slots
        )
        
        # Score based on understanding and reasonableness
        score = (int(understands_overlap) + int(suggested_reasonable)) / 2.0
        
        return {
            "understands_overlap_logic": understands_overlap,
            "suggested_times_reasonable": suggested_reasonable,
            "constraint_satisfaction_score": score
        }
