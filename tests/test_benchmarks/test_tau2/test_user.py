"""Unit tests for Tau2User."""

import pytest
from unittest.mock import MagicMock
from maseval.benchmark.tau2 import Tau2User


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.mark.benchmark
def test_extract_user_profile():
    """Test extracting persona and profile from scenario."""
    scenario = "Persona: Angry Customer\n\nTask: Return item."
    
    profile = Tau2User._extract_user_profile(scenario)
    
    assert profile["persona"] == "Angry Customer"
    assert profile["full_scenario"] == scenario


@pytest.mark.benchmark
def test_extract_user_profile_no_persona():
    """Test extracting profile when no persona is present."""
    scenario = "Just a task."
    
    profile = Tau2User._extract_user_profile(scenario)
    
    assert "persona" not in profile
    assert profile["full_scenario"] == scenario


@pytest.mark.benchmark
def test_get_tool_raises(mock_model):
    """Test that base Tau2User raises NotImplementedError for get_tool."""
    user = Tau2User(
        model=mock_model,
        scenario="test",
        initial_query="hi"
    )
    
    with pytest.raises(NotImplementedError):
        user.get_tool()


@pytest.mark.benchmark
def test_init_passes_tools(mock_model):
    """Test that tools are correctly passed to parent AgenticUser."""
    tools = {"test_tool": lambda x: x}
    
    user = Tau2User(
        model=mock_model,
        scenario="test",
        initial_query="hi",
        tools=tools
    )
    
    assert user.tools == tools
