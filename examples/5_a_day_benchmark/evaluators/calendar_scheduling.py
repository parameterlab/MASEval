"""Evaluators for Task 3: Calendar Scheduling (MCP).

Task: User asks to find meeting times with Roger Bannister by checking both calendars.
Success criteria: Agent correctly identifies overlapping available time slots.
"""

from typing import Any, Dict, Optional

from maseval import Evaluator, Environment, Task, User, MessageHistory
from .utils import extract_tool_calls, extract_assistant_response


class SchedulingAccuracyEvaluator(Evaluator):
    """Evaluates if agent correctly identified overlapping calendar slots.
    
    Evaluation type: Slot matching
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
                "scheduling_precision": 0.0,
                "error": "No assistant response found"
            }
        
        # Check for each expected slot
        found_slots = []
        for slot in self.expected_slots:
            if self._slot_mentioned(slot, full_response):
                found_slots.append(slot)
        
        slots_found = len(found_slots)
        all_found = slots_found == len(self.expected_slots)
        precision = slots_found / len(self.expected_slots) if self.expected_slots else 0.0
        
        return {
            "overlapping_slots_found": slots_found,
            "total_expected_slots": len(self.expected_slots),
            "all_overlaps_identified": all_found,
            "scheduling_precision": precision,
            "found_slot_details": found_slots
        }

    def _slot_mentioned(self, slot: Dict[str, str], response: str) -> bool:
        """Check if a time slot is mentioned in the response."""
        slot_date = slot["date"]
        slot_start = slot["start"]
        slot_end = slot["end"]
        
        # Check for date in various formats
        date_mentioned = (
            slot_date in response or 
            slot_date.replace("-", "/") in response
        )
        
        # Extract date variations (12-02, 12/02, December 2, etc.)
        date_parts = slot_date.split("-")
        if len(date_parts) == 3:
            month_day = f"{date_parts[1]}-{date_parts[2]}"
            month_day_slash = f"{date_parts[1]}/{date_parts[2]}"
            date_mentioned = date_mentioned or month_day in response or month_day_slash in response
        
        # Check for times in 24hr or 12hr formats
        time_mentioned = slot_start in response or slot_end in response
        start_12hr = self._convert_to_12hr(slot_start)
        end_12hr = self._convert_to_12hr(slot_end)
        time_mentioned = time_mentioned or start_12hr in response or end_12hr in response
        
        return date_mentioned and time_mentioned

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
    
    Evaluation type: Tool validation
    Measures: Did the agent access both calendars to find overlaps?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate MCP tool usage for task completion."""
        tools_used = extract_tool_calls(trace)
        
        # Check if both calendars were accessed
        my_calendar_used = any("my_calendar" in str(tool).lower() for tool in tools_used)
        other_calendar_used = any(
            "other_calendar" in str(tool).lower() or "roger" in str(tool).lower() 
            for tool in tools_used
        )
        
        used_both_calendars = my_calendar_used and other_calendar_used
        score = 1.0 if used_both_calendars else 0.0
        
        return {
            "used_both_calendars": used_both_calendars,
            "mcp_integration_score": score,
            "tools_used": tools_used
        }


class ConstraintSatisfactionEvaluator(Evaluator):
    """Evaluates logical correctness of scheduling constraints.
    
    Evaluation type: Logic validation
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
        
        score = (int(understands_overlap) + int(suggested_reasonable)) / 2.0
        
        return {
            "understands_overlap_logic": understands_overlap,
            "suggested_times_reasonable": suggested_reasonable,
            "constraint_satisfaction_score": score
        }
