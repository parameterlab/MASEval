"""Task 4 Evaluators: Hotel Optimization.

Task: User asks to find the best hotel based on distance, WiFi, and price priorities.
Success criteria: Agent selects hotel close to the mathematical optimum.
"""

import re
from typing import Any, Dict, List, Optional

from maseval import Evaluator, Environment, Task, User, MessageHistory
from .utils import extract_assistant_response, extract_tool_calls


class OptimizationQualityEvaluator(Evaluator):
    """Evaluates how close agent's hotel choice is to the optimal solution.
    
    Measures: Did the agent find the best (or near-best) hotel?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.ground_truth = task.evaluation_data
        self.hotels = task.environment_data.get("hotels", [])
        self.weights = task.environment_data.get("user_priorities", {
            "distance_weight": 0.4,
            "wifi_weight": 0.35,
            "price_weight": 0.25
        })
        
        # Pre-calculate scores for all hotels
        self.hotel_scores = self._calculate_all_hotel_scores()

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate optimization quality."""
        full_response = extract_assistant_response(trace)
        
        if not full_response:
            return {
                "found_optimal_hotel": False,
                "optimization_score": 0.0,
                "error": "No assistant response found"
            }
        
        # Extract hotel choice from response
        chosen_hotel_id = self._extract_hotel_choice(full_response)
        
        if not chosen_hotel_id:
            return {
                "found_optimal_hotel": False,
                "chosen_hotel": None,
                "optimization_score": 0.0,
                "error": "No hotel choice found in response"
            }
        
        # Get scores sorted by quality
        sorted_hotels = sorted(self.hotel_scores.items(), key=lambda x: x[1], reverse=True)
        optimal_hotel_id = sorted_hotels[0][0]
        
        # Calculate rank distance (1 = optimal, 2 = second best, etc.)
        chosen_rank = self._get_hotel_rank(chosen_hotel_id, sorted_hotels)
        
        # Score based on how close to optimal (exponential decay)
        # Rank 1 = 1.0, Rank 2 = 0.7, Rank 3 = 0.5, Rank 4+ = 0.3 decreasing
        if chosen_rank == 1:
            score = 1.0
        elif chosen_rank == 2:
            score = 0.7
        elif chosen_rank == 3:
            score = 0.5
        else:
            score = max(0.1, 0.5 - (chosen_rank - 3) * 0.1)
        
        return {
            "found_optimal_hotel": chosen_rank == 1,
            "chosen_hotel": chosen_hotel_id,
            "chosen_hotel_rank": chosen_rank,
            "optimal_hotel": optimal_hotel_id,
            "optimization_score": score,
            "hotel_scores": sorted_hotels[:3]  # Top 3 for reference
        }

    def _calculate_all_hotel_scores(self) -> Dict[str, float]:
        """Calculate optimization scores for all hotels."""
        max_price = max(hotel["price_per_night"] for hotel in self.hotels)
        
        scores = {}
        for hotel in self.hotels:
            score = self._calculate_hotel_score(hotel, max_price)
            scores[hotel["id"]] = score
        
        return scores

    def _calculate_hotel_score(self, hotel: Dict[str, Any], max_price: float) -> float:
        """Calculate score using the weighted formula from task data."""
        # Normalize distance (lower is better, max 5km)
        distance_score = max(0, 1.0 - hotel["distance_to_venue_km"] / 5.0)
        
        # Normalize WiFi (higher is better, max 500 Mbps)
        wifi_score = min(1.0, hotel["wifi_speed_mbps"] / 500.0)
        
        # Normalize price (lower is better)
        price_score = max(0, 1.0 - hotel["price_per_night"] / max_price)
        
        # Weighted sum
        score = (
            distance_score * self.weights["distance_weight"] +
            wifi_score * self.weights["wifi_weight"] +
            price_score * self.weights["price_weight"]
        )
        
        return score

    def _extract_hotel_choice(self, response: str) -> Optional[str]:
        """Extract hotel ID from response."""
        # Look for hotel IDs (H001, H002, etc.)
        import re
        
        # Try to find hotel ID pattern
        hotel_id_pattern = r'\bH0*(\d{1,2})\b'
        matches = re.findall(hotel_id_pattern, response.upper())
        
        if matches:
            # Convert to standard format (H001, H002, etc.)
            hotel_num = int(matches[-1])  # Take last mentioned
            return f"H{hotel_num:03d}"
        
        # Try to find hotel name
        for hotel in self.hotels:
            if hotel["name"] in response:
                return hotel["id"]
        
        return None

    def _get_hotel_rank(self, hotel_id: str, sorted_hotels: List[tuple]) -> int:
        """Get rank of hotel (1-indexed)."""
        for i, (hid, _) in enumerate(sorted_hotels, 1):
            if hid == hotel_id:
                return i
        return len(sorted_hotels) + 1


class SearchStrategyEvaluator(Evaluator):
    """Evaluates the search strategy employed by the agent.
    
    Measures: Did the agent examine hotels systematically?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.total_hotels = len(task.environment_data.get("hotels", []))

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate search strategy."""
        tools_used = extract_tool_calls(trace)
        
        # Check if hotel_search was used
        used_hotel_search = any("hotel" in str(tool).lower() for tool in tools_used)
        
        # Check if calculator was used (for scoring)
        used_calculator = any("calculator" in str(tool).lower() for tool in tools_used)
        
        # Estimate hotels examined (from tool responses)
        tool_messages = [msg for msg in trace if msg.get("role") == "tool"]
        hotels_examined = 0
        for msg in tool_messages:
            content = str(msg.get("content", ""))
            # Look for hotel data patterns
            if "H0" in content or "hotel" in content.lower():
                # Count hotel mentions
                import re
                hotel_mentions = len(re.findall(r'H\d{3}', content))
                hotels_examined = max(hotels_examined, hotel_mentions)
        
        # Search completeness
        search_completeness = min(1.0, hotels_examined / self.total_hotels) if self.total_hotels > 0 else 0.0
        
        # Calculate strategy score
        score_components = [
            int(used_hotel_search),
            int(used_calculator),
            search_completeness
        ]
        strategy_score = sum(score_components) / len(score_components)
        
        return {
            "hotels_retrieved": hotels_examined,
            "total_available_hotels": self.total_hotels,
            "used_hotel_search": used_hotel_search,
            "used_calculator": used_calculator,
            "search_completeness": round(search_completeness, 2),
            "search_strategy_score": round(strategy_score, 2)
        }


class ReasoningTransparencyEvaluator(Evaluator):
    """Evaluates explanation quality and transparency of decision-making.
    
    Measures: Did the agent explain its reasoning and the tradeoffs?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate reasoning transparency."""
        full_response = extract_assistant_response(trace).lower()
        
        if not full_response:
            return {
                "explained_priorities": False,
                "showed_calculation": False,
                "justified_choice": False,
                "mentioned_tradeoffs": False,
                "transparency_score": 0.0
            }
        
        # Check for priority explanation
        priority_keywords = ["distance", "wifi", "price", "close", "proximity", "speed", "cost"]
        explained_priorities = sum(1 for kw in priority_keywords if kw in full_response) >= 3
        
        # Check for calculation/scoring
        calc_keywords = ["score", "calculate", "formula", "weight", "multiply", "add"]
        showed_calculation = any(kw in full_response for kw in calc_keywords)
        
        # Check for justification
        justification_keywords = ["because", "since", "as", "due to", "reason", "best"]
        justified_choice = any(kw in full_response for kw in justification_keywords)
        
        # Check for tradeoff discussion
        tradeoff_keywords = ["however", "but", "although", "tradeoff", "trade-off", "balance", "alternative"]
        mentioned_tradeoffs = any(kw in full_response for kw in tradeoff_keywords)
        
        # Calculate transparency score
        checks = [explained_priorities, showed_calculation, justified_choice, mentioned_tradeoffs]
        transparency_score = sum(checks) / len(checks)
        
        return {
            "explained_priorities": explained_priorities,
            "showed_calculation": showed_calculation,
            "justified_choice": justified_choice,
            "mentioned_tradeoffs": mentioned_tradeoffs,
            "transparency_score": round(transparency_score, 2)
        }
