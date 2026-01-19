"""Tests for MultiAgentBench evaluator."""

import pytest

from conftest import DummyModelAdapter
from maseval.benchmark.multiagentbench.evaluator import (
    MultiAgentBenchEvaluator,
    MultiAgentBenchMetrics,
)


class TestMultiAgentBenchMetrics:
    """Tests for MultiAgentBenchMetrics dataclass."""

    def test_default_values(self):
        """Metrics should have sensible defaults."""
        metrics = MultiAgentBenchMetrics()

        assert metrics.task_completion is False
        assert metrics.token_consumption == 0
        assert metrics.planning_score == -1.0
        assert metrics.communication_score == -1.0
        assert metrics.task_evaluation == {}
        assert metrics.agent_kpis == {}
        assert metrics.total_milestones == 0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        metrics = MultiAgentBenchMetrics(
            task_completion=True,
            token_consumption=1000,
            planning_score=4.0,
            communication_score=5.0,
        )
        d = metrics.to_dict()

        assert d["task_completion"] is True
        assert d["token_consumption"] == 1000
        assert d["planning_score"] == 4.0
        assert d["communication_score"] == 5.0


class TestMultiAgentBenchEvaluator:
    """Tests for MultiAgentBenchEvaluator class."""

    @pytest.fixture
    def mock_model_adapter(self):
        """Create a mock model adapter."""
        return DummyModelAdapter(
            model_id="test-model",
            responses=['{"rating": 4}'],
        )

    @pytest.fixture
    def research_evaluator(self, mock_model_adapter):
        """Create a research domain evaluator."""
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=mock_model_adapter,
        )

    @pytest.fixture
    def bargaining_evaluator(self, mock_model_adapter):
        """Create a bargaining domain evaluator."""
        return MultiAgentBenchEvaluator(
            domain="bargaining",
            model_adapter=mock_model_adapter,
        )

    def test_init_normalizes_domain(self, mock_model_adapter):
        """Evaluator should normalize domain to lowercase."""
        evaluator = MultiAgentBenchEvaluator(
            domain="RESEARCH",
            model_adapter=mock_model_adapter,
        )
        assert evaluator.domain == "research"

    def test_filter_traces_extracts_agents(self, research_evaluator):
        """filter_traces should extract agent traces."""
        traces = {
            "agents": {"agent1": {"action_log": [{"task": "test", "result": "done"}]}},
            "environment": {},
        }
        filtered = research_evaluator.filter_traces(traces)

        assert "agents" in filtered
        assert "agent1" in filtered["agents"]

    def test_extract_communications_formats_correctly(self, research_evaluator):
        """_extract_communications should format communication logs."""
        traces = {
            "agents": {
                "agent1": {"communication_log": [{"communication": "Hello from agent1"}]},
                "agent2": {"communication_log": [{"communication": "Hello from agent2"}]},
            }
        }
        comms = research_evaluator._extract_communications(traces)

        assert "[agent1]: Hello from agent1" in comms
        assert "[agent2]: Hello from agent2" in comms

    def test_extract_communications_empty(self, research_evaluator):
        """_extract_communications should handle empty logs."""
        traces = {"agents": {}}
        comms = research_evaluator._extract_communications(traces)

        assert comms == "No communications recorded."

    def test_extract_results_formats_correctly(self, research_evaluator):
        """_extract_results should format action logs."""
        traces = {"agents": {"agent1": {"action_log": [{"result": "Completed task A"}]}}}
        results = research_evaluator._extract_results(traces)

        assert "[agent1]: Completed task A" in results

    def test_calculate_token_consumption(self, research_evaluator):
        """_calculate_token_consumption should sum agent token usage."""
        traces = {
            "agents": {
                "agent1": {"token_usage": 500},
                "agent2": {"token_usage": 300},
            }
        }
        total = research_evaluator._calculate_token_consumption(traces)

        assert total == 800

    def test_parse_score_valid_json(self, research_evaluator):
        """_parse_score should parse valid JSON response."""
        response = '{"rating": 4}'
        score = research_evaluator._parse_score(response)

        assert score == 4.0

    def test_parse_score_with_markdown(self, research_evaluator):
        """_parse_score should handle markdown code blocks."""
        response = '```json\n{"rating": 5}\n```'
        score = research_evaluator._parse_score(response)

        assert score == 5.0

    def test_parse_score_fallback(self, research_evaluator):
        """_parse_score should fallback to finding digit."""
        response = "The rating is 4 out of 5"
        score = research_evaluator._parse_score(response)

        assert score == 4.0

    def test_parse_score_default(self, research_evaluator):
        """_parse_score should return 3 as default."""
        response = "No score here"
        score = research_evaluator._parse_score(response)

        assert score == 3.0

    def test_parse_research_ratings_valid(self, research_evaluator):
        """_parse_research_ratings should parse valid ratings."""
        response = '{"innovation": 4, "safety": 3, "feasibility": 5}'
        ratings = research_evaluator._parse_research_ratings(response)

        assert ratings["innovation"] == 4
        assert ratings["safety"] == 3
        assert ratings["feasibility"] == 5

    def test_parse_research_ratings_invalid(self, research_evaluator):
        """_parse_research_ratings should return -1 for invalid."""
        response = "Invalid response"
        ratings = research_evaluator._parse_research_ratings(response)

        assert ratings["innovation"] == -1
        assert ratings["safety"] == -1
        assert ratings["feasibility"] == -1

    def test_determine_completion_research_positive(self, research_evaluator):
        """_determine_completion should return True for positive research scores."""
        metrics = MultiAgentBenchMetrics(task_evaluation={"innovation": 4, "safety": 3, "feasibility": 5})
        assert research_evaluator._determine_completion(metrics) is True

    def test_determine_completion_research_negative(self, research_evaluator):
        """_determine_completion should return False for negative scores."""
        metrics = MultiAgentBenchMetrics(task_evaluation={"innovation": -1, "safety": 3, "feasibility": 5})
        assert research_evaluator._determine_completion(metrics) is False

    def test_call_returns_expected_structure(self, research_evaluator):
        """__call__ should return expected result structure."""
        traces = {
            "agents": {"agent1": {"token_usage": 100, "action_log": [], "communication_log": []}},
            "environment": {},
        }
        final_answer = [{"agent_id": "agent1", "result": "Done"}]

        # Mock model adapter to return research ratings
        research_evaluator.model_adapter = DummyModelAdapter(
            model_id="test",
            responses=['{"innovation": 4, "safety": 4, "feasibility": 4}'],
        )

        result = research_evaluator(traces, final_answer)

        assert "passed" in result
        assert "metrics" in result
        assert "domain" in result
        assert result["domain"] == "research"

    def test_format_final_answer_dict(self, research_evaluator):
        """_format_final_answer should format dict input."""
        final_answer = {
            "agent_results": [
                {"agent_id": "agent1", "result": "Result 1"},
                {"agent_id": "agent2", "result": "Result 2"},
            ]
        }
        formatted = research_evaluator._format_final_answer(final_answer)

        assert "[agent1]: Result 1" in formatted
        assert "[agent2]: Result 2" in formatted

    def test_format_final_answer_list(self, research_evaluator):
        """_format_final_answer should format list input."""
        final_answer = [
            {"agent_id": "agent1", "result": "Result 1"},
        ]
        formatted = research_evaluator._format_final_answer(final_answer)

        assert "[agent1]: Result 1" in formatted

    def test_format_final_answer_string(self, research_evaluator):
        """_format_final_answer should pass through string."""
        formatted = research_evaluator._format_final_answer("Simple result")
        assert formatted == "Simple result"

    def test_format_final_answer_none(self, research_evaluator):
        """_format_final_answer should handle None."""
        formatted = research_evaluator._format_final_answer(None)
        assert formatted == ""


class TestBargainingEvaluation:
    """Tests for bargaining-specific evaluation."""

    @pytest.fixture
    def bargaining_evaluator(self):
        """Create evaluator with bargaining-specific responses."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=[
                '{"effectiveness_of_strategies": 4, "progress_and_outcome": 3, "interaction_dynamics": 5}',
                '{"effectiveness_of_strategies": 3, "progress_and_outcome": 4, "interaction_dynamics": 4}',
            ],
        )
        return MultiAgentBenchEvaluator(
            domain="bargaining",
            model_adapter=adapter,
        )

    def test_evaluate_bargaining_returns_buyer_seller(self, bargaining_evaluator):
        """_evaluate_bargaining should return buyer and seller ratings."""
        ratings = bargaining_evaluator._evaluate_bargaining("Task", "Result")

        assert "buyer" in ratings
        assert "seller" in ratings
        assert "effectiveness_of_strategies" in ratings["buyer"]

    def test_determine_completion_bargaining_positive(self, bargaining_evaluator):
        """_determine_completion should work for bargaining domain."""
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "buyer": {
                    "effectiveness_of_strategies": 4,
                    "progress_and_outcome": 3,
                    "interaction_dynamics": 5,
                },
                "seller": {
                    "effectiveness_of_strategies": 3,
                    "progress_and_outcome": 4,
                    "interaction_dynamics": 4,
                },
            }
        )
        assert bargaining_evaluator._determine_completion(metrics) is True


class TestCodingEvaluation:
    """Tests for coding-specific evaluation."""

    @pytest.fixture
    def coding_evaluator(self):
        """Create evaluator with coding-specific responses."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=[
                '{"instruction_following": 5, "executability": 4, "consistency": 4, "quality": 3}',
            ],
        )
        return MultiAgentBenchEvaluator(
            domain="coding",
            model_adapter=adapter,
        )

    def test_evaluate_coding_returns_all_metrics(self, coding_evaluator):
        """_evaluate_coding should return all coding metrics."""
        ratings = coding_evaluator._evaluate_coding("Task", "Solution")

        assert "instruction_following" in ratings
        assert "executability" in ratings
        assert "consistency" in ratings
        assert "quality" in ratings

    def test_determine_completion_coding_positive(self, coding_evaluator):
        """_determine_completion should work for coding domain."""
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "instruction_following": 5,
                "executability": 4,
                "consistency": 4,
                "quality": 3,
            }
        )
        assert coding_evaluator._determine_completion(metrics) is True
