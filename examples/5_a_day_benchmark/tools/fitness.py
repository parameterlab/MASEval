"""Fitness tracking tools (running app and gym tracker)."""

from dataclasses import dataclass
from typing import Any

from .base import BaseTool, ToolResult


@dataclass
class Activity:
    """Fitness activity representation."""

    id: str
    date: str
    activity: str
    duration_minutes: int
    type: str
    distance_km: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "date": self.date,
            "activity": self.activity,
            "duration_minutes": self.duration_minutes,
            "type": self.type,
        }
        if self.distance_km is not None:
            result["distance_km"] = self.distance_km
        return result


class RunningAppTool(BaseTool):
    """Running and cardio activity tracker."""

    def __init__(self, activities_data: list[dict[str, Any]]):
        super().__init__(
            "running_app",
            "Access running and cardio activities",
            tool_args=["action", "start_date", "end_date", "activity_id"],
        )
        self.activities: list[Activity] = []

        # Load activities
        for i, activity_data in enumerate(activities_data):
            self.activities.append(
                Activity(
                    id=str(i + 1),
                    date=activity_data["date"],
                    activity=activity_data["activity"],
                    duration_minutes=activity_data["duration_minutes"],
                    distance_km=activity_data.get("distance_km"),
                    type=activity_data["type"],
                )
            )

    def execute(self, **kwargs) -> ToolResult:
        """Execute running app action."""
        action = kwargs.get("action")

        if action == "get_activities":
            return self._get_activities(
                start_date=kwargs.get("start_date"),
                end_date=kwargs.get("end_date"),
            )
        elif action == "get_activity":
            return self._get_activity(kwargs.get("activity_id"))
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _get_activities(self, start_date: str | None = None, end_date: str | None = None) -> ToolResult:
        """Get activities in date range."""
        filtered = self.activities

        if start_date:
            filtered = [a for a in filtered if a.date >= start_date]
        if end_date:
            filtered = [a for a in filtered if a.date <= end_date]

        return ToolResult(
            success=True,
            data={
                "activities": [a.to_dict() for a in filtered],
                "count": len(filtered),
            },
        )

    def _get_activity(self, activity_id: str | None) -> ToolResult:
        """Get specific activity by ID."""
        if not activity_id:
            return ToolResult(success=False, data=None, error="activity_id is required")

        for activity in self.activities:
            if activity.id == activity_id:
                return ToolResult(success=True, data=activity.to_dict())

        return ToolResult(success=False, data=None, error=f"Activity {activity_id} not found")


class GymTrackerTool(BaseTool):
    """Gym and strength workout tracker."""

    def __init__(self, workouts_data: list[dict[str, Any]]):
        super().__init__(
            "gym_tracker",
            "Access gym and strength workouts",
            tool_args=["action", "start_date", "end_date", "workout_id"],
        )
        self.workouts: list[Activity] = []

        # Load workouts
        for i, workout_data in enumerate(workouts_data):
            self.workouts.append(
                Activity(
                    id=str(i + 1),
                    date=workout_data["date"],
                    activity=workout_data["activity"],
                    duration_minutes=workout_data["duration_minutes"],
                    type=workout_data["type"],
                    distance_km=None,
                )
            )

    def execute(self, **kwargs) -> ToolResult:
        """Execute gym tracker action."""
        action = kwargs.get("action")

        if action == "get_workouts":
            return self._get_workouts(
                start_date=kwargs.get("start_date"),
                end_date=kwargs.get("end_date"),
            )
        elif action == "get_workout":
            return self._get_workout(kwargs.get("workout_id"))
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _get_workouts(self, start_date: str | None = None, end_date: str | None = None) -> ToolResult:
        """Get workouts in date range."""
        filtered = self.workouts

        if start_date:
            filtered = [w for w in filtered if w.date >= start_date]
        if end_date:
            filtered = [w for w in filtered if w.date <= end_date]

        return ToolResult(
            success=True,
            data={
                "workouts": [w.to_dict() for w in filtered],
                "count": len(filtered),
            },
        )

    def _get_workout(self, workout_id: str | None) -> ToolResult:
        """Get specific workout by ID."""
        if not workout_id:
            return ToolResult(success=False, data=None, error="workout_id is required")

        for workout in self.workouts:
            if workout.id == workout_id:
                return ToolResult(success=True, data=workout.to_dict())

        return ToolResult(success=False, data=None, error=f"Workout {workout_id} not found")
