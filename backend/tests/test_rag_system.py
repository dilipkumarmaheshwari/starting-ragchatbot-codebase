"""Tests for RAGSystem.query() handling content-related questions,
in backend/rag_system.py.

Unlike the other two test files, these use a REAL VectorStore backed by a
temp ChromaDB (seeded with a fake course) so the full chain --
RAGSystem -> AIGenerator -> ToolManager -> CourseSearchTool -> VectorStore
-> embeddings/Chroma -- is actually exercised. Only the Anthropic network
call is mocked (deterministic, no API key needed) except in the final
opt-in live test, which hits the real API and the real persisted
backend/chroma_db to check the system exactly as a user would experience it.
"""
import dataclasses
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config import Config
from rag_system import RAGSystem
from tests.conftest import FakeResponse, FakeTextBlock, FakeToolUseBlock

BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def test_config(tmp_path):
    return Config(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="claude-sonnet-4-5",
        CHROMA_PATH=str(tmp_path / "chroma_db"),
        MAX_RESULTS=5,
        MAX_HISTORY=2,
    )


@pytest.fixture
def seeded_rag_system(test_config):
    """A RAGSystem with a real (temp, empty-until-seeded) vector store and a
    mocked Anthropic client, pre-loaded with one fake course."""
    rag = RAGSystem(test_config)
    rag.ai_generator.client.messages.create = MagicMock()

    from models import Course, Lesson, CourseChunk

    course = Course(
        title="MCP: Build Rich-Context AI Apps",
        course_link="https://example.com/mcp-course",
        instructor="Someone",
        lessons=[
            Lesson(lesson_number=3, title="Chatbot Example", lesson_link="https://example.com/l3"),
            Lesson(lesson_number=5, title="Creating An MCP Client", lesson_link="https://example.com/l5"),
        ],
    )
    rag.vector_store.add_course_metadata(course)
    rag.vector_store.add_course_content([
        CourseChunk(
            content="A tool has a name, a description, and an input schema. "
                     "The model detects when a tool should be used but never executes it directly.",
            course_title=course.title,
            lesson_number=3,
            chunk_index=0,
        ),
        CourseChunk(
            content="The MCP client connects to a server and exposes its tools to the chatbot loop.",
            course_title=course.title,
            lesson_number=5,
            chunk_index=1,
        ),
    ])
    return rag


class TestContentQueryHandling:
    def test_content_query_triggers_search_and_returns_grounded_answer(self, seeded_rag_system):
        rag = seeded_rag_system
        tool_use = FakeResponse(
            [FakeToolUseBlock(
                "search_course_content",
                {"query": "what is a tool", "course_name": "MCP"},
            )],
            "tool_use",
        )
        final = FakeResponse(
            [FakeTextBlock("A tool has a name, description, and input schema.")],
            "end_turn",
        )
        rag.ai_generator.client.messages.create.side_effect = [tool_use, final]

        answer, sources = rag.query("What is a tool in the MCP course?", session_id=None)

        assert answer == "A tool has a name, description, and input schema."
        assert len(sources) >= 1
        assert sources[0]["text"].startswith("MCP: Build Rich-Context AI Apps")

    def test_general_knowledge_query_skips_tool_and_returns_no_sources(self, seeded_rag_system):
        rag = seeded_rag_system
        rag.ai_generator.client.messages.create.return_value = FakeResponse(
            [FakeTextBlock("Paris is the capital of France.")], "end_turn"
        )

        answer, sources = rag.query("What is the capital of France?", session_id=None)

        assert answer == "Paris is the capital of France."
        assert sources == []

    def test_content_query_for_unknown_course_does_not_raise(self, seeded_rag_system):
        """The real VectorStore.search() catches course-resolution misses and
        returns an error string rather than raising, so this must complete
        normally -- if it raised, that would itself explain a 500."""
        rag = seeded_rag_system
        tool_use = FakeResponse(
            [FakeToolUseBlock(
                "search_course_content",
                {"query": "anything", "course_name": "Some Course That Does Not Exist"},
            )],
            "tool_use",
        )
        final = FakeResponse(
            [FakeTextBlock("I couldn't find that course.")], "end_turn"
        )
        rag.ai_generator.client.messages.create.side_effect = [tool_use, final]

        answer, sources = rag.query("Tell me about a course that isn't loaded", session_id=None)

        assert answer == "I couldn't find that course."

    def test_session_history_recorded_after_content_query(self, seeded_rag_system):
        rag = seeded_rag_system
        tool_use = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "what is a tool"})], "tool_use"
        )
        final = FakeResponse([FakeTextBlock("A tool has a name.")], "end_turn")
        rag.ai_generator.client.messages.create.side_effect = [tool_use, final]

        session_id = rag.session_manager.create_session()
        rag.query("What is a tool?", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert "What is a tool?" in history
        assert "A tool has a name." in history

    def test_sources_reset_between_successive_queries(self, seeded_rag_system):
        rag = seeded_rag_system
        tool_use = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "what is a tool"})], "tool_use"
        )
        final = FakeResponse([FakeTextBlock("answer 1")], "end_turn")
        rag.ai_generator.client.messages.create.side_effect = [tool_use, final]
        _, sources_1 = rag.query("What is a tool?", session_id=None)
        assert len(sources_1) >= 1

        rag.ai_generator.client.messages.create.side_effect = None
        rag.ai_generator.client.messages.create.return_value = FakeResponse(
            [FakeTextBlock("answer 2")], "end_turn"
        )
        _, sources_2 = rag.query("What is the capital of France?", session_id=None)
        assert sources_2 == []


class TestContentQueryResilience:
    def test_transient_api_failure_degrades_gracefully_instead_of_500(self, seeded_rag_system):
        """AIGenerator now catches Anthropic API errors and returns a friendly
        message, so rag_system.query() completes normally instead of raising
        -- the frontend no longer shows a hard 'Query failed' for a transient
        network/rate-limit blip."""
        rag = seeded_rag_system
        rag.ai_generator.client.messages.create.side_effect = ConnectionError(
            "upstream connection reset"
        )

        answer, sources = rag.query("What is a tool in the MCP course?", session_id=None)

        assert "trouble reaching the AI service" in answer
        assert sources == []

    def test_malformed_tool_call_degrades_gracefully_instead_of_500(self, seeded_rag_system):
        """ToolManager.execute_tool now catches exceptions from the underlying
        tool (e.g. a TypeError from a malformed/hallucinated argument) and
        returns an error string as the tool result, letting the flow complete
        instead of raising out of rag_system.query()."""
        rag = seeded_rag_system
        tool_use = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q", "course_title": "MCP"})],
            "tool_use",
        )
        final = FakeResponse(
            [FakeTextBlock("I ran into an issue searching, please rephrase your question.")],
            "end_turn",
        )
        rag.ai_generator.client.messages.create.side_effect = [tool_use, final]

        answer, sources = rag.query("What is a tool in the MCP course?", session_id=None)

        assert answer == "I ran into an issue searching, please rephrase your question."


ANTHROPIC_KEY_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY")) or bool(
    Config().ANTHROPIC_API_KEY
)


@pytest.mark.skipif(
    not ANTHROPIC_KEY_AVAILABLE, reason="No ANTHROPIC_API_KEY available for a live API check"
)
class TestLiveSystemContentQuery:
    """Opt-in end-to-end check against the REAL Anthropic API and the REAL
    persisted backend/chroma_db, exactly as the running app would see it.
    This is the closest thing to reproducing 'the chatbot returns Query
    failed for content questions' inside the test suite."""

    def test_real_content_query_against_persisted_store_succeeds(self):
        real_chroma_path = BACKEND_DIR / "chroma_db"
        if not real_chroma_path.exists():
            pytest.skip("backend/chroma_db has not been populated yet")

        live_config = dataclasses.replace(Config(), CHROMA_PATH=str(real_chroma_path))
        rag = RAGSystem(live_config)

        course_titles = rag.vector_store.get_existing_course_titles()
        if not course_titles:
            pytest.skip("No courses loaded in backend/chroma_db")

        answer, sources = rag.query(
            f"According to the {course_titles[0]} course, what is covered in the first lesson?",
            session_id=None,
        )

        assert isinstance(answer, str) and len(answer) > 0
