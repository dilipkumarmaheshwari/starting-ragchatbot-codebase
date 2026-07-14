"""Shared fixtures for the backend test suite.

Fake Anthropic response types (FakeTextBlock / FakeToolUseBlock / FakeResponse)
stand in for the SDK's pydantic objects so AIGenerator tests never touch the
network - they only need `.type`, `.text` / `.name` / `.input` / `.id`,
`.content` and `.stop_reason`.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure `backend/` is importable even if pythonpath isn't picked up
# (e.g. running `pytest` directly from backend/tests).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from vector_store import SearchResults, VectorStore  # noqa: E402
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager  # noqa: E402


# ---------------------------------------------------------------------------
# Fake Anthropic SDK objects
# ---------------------------------------------------------------------------

class FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeToolUseBlock:
    def __init__(self, name: str, input_: dict, id_: str = "tool_use_1"):
        self.type = "tool_use"
        self.name = name
        self.input = input_
        self.id = id_


class FakeResponse:
    def __init__(self, content, stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason


# ---------------------------------------------------------------------------
# VectorStore / search fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store():
    """A MagicMock honoring VectorStore's public interface."""
    store = MagicMock(spec=VectorStore)
    store.get_lesson_link.return_value = "https://example.com/lesson"
    store.get_course_link.return_value = "https://example.com/course"
    return store


@pytest.fixture
def sample_search_results():
    return SearchResults(
        documents=[
            "Tools let a model call external functions.",
            "A tool has a name, description, and input schema.",
        ],
        metadata=[
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 3},
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 5},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    return SearchResults.empty("Search error: connection refused")


@pytest.fixture
def search_tool(mock_vector_store):
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def outline_tool(mock_vector_store):
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager_with_search_tool(search_tool):
    manager = ToolManager()
    manager.register_tool(search_tool)
    return manager
