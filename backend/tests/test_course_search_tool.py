"""Tests for CourseSearchTool.execute() in backend/search_tools.py.

These isolate CourseSearchTool from VectorStore by mocking the store, so
failures here point specifically at the tool's own logic (query dispatch,
error/empty handling, result formatting, source tracking) rather than at
ChromaDB, embeddings, or the Anthropic API.
"""
from vector_store import SearchResults


class TestExecuteHappyPath:
    def test_returns_formatted_results_with_headers_and_content(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        mock_vector_store.search.return_value = sample_search_results

        result = search_tool.execute(query="what is a tool")

        assert "[MCP: Build Rich-Context AI Apps - Lesson 3]" in result
        assert "Tools let a model call external functions." in result
        assert "[MCP: Build Rich-Context AI Apps - Lesson 5]" in result
        assert "A tool has a name, description, and input schema." in result

    def test_passes_query_course_name_and_lesson_number_to_store(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        mock_vector_store.search.return_value = sample_search_results

        search_tool.execute(query="what is a tool", course_name="MCP", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="what is a tool", course_name="MCP", lesson_number=3
        )

    def test_defaults_course_name_and_lesson_number_to_none(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        mock_vector_store.search.return_value = sample_search_results

        search_tool.execute(query="what is a tool")

        mock_vector_store.search.assert_called_once_with(
            query="what is a tool", course_name=None, lesson_number=None
        )

    def test_populates_last_sources_with_text_and_link(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson-3"

        search_tool.execute(query="what is a tool")

        assert search_tool.last_sources == [
            {"text": "MCP: Build Rich-Context AI Apps - Lesson 3", "link": "https://example.com/lesson-3"},
            {"text": "MCP: Build Rich-Context AI Apps - Lesson 5", "link": "https://example.com/lesson-3"},
        ]

    def test_uses_course_link_when_result_has_no_lesson_number(
        self, search_tool, mock_vector_store
    ):
        results = SearchResults(
            documents=["Course-level overview text."],
            metadata=[{"course_title": "MCP Course", "lesson_number": None}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = results
        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        search_tool.execute(query="overview")

        mock_vector_store.get_course_link.assert_called_once_with("MCP Course")
        mock_vector_store.get_lesson_link.assert_not_called()
        assert search_tool.last_sources == [
            {"text": "MCP Course", "link": "https://example.com/course"}
        ]

    def test_last_sources_cleared_after_a_subsequent_empty_result(
        self, search_tool, mock_vector_store, sample_search_results, empty_search_results
    ):
        """execute() now clears last_sources up front, so a later empty/error
        result doesn't leak the previous query's sources into the UI."""
        mock_vector_store.search.return_value = sample_search_results
        search_tool.execute(query="first query")
        assert len(search_tool.last_sources) == 2

        mock_vector_store.search.return_value = empty_search_results
        search_tool.execute(query="second query")
        assert search_tool.last_sources == []


class TestExecuteEmptyAndErrorHandling:
    def test_returns_error_string_when_store_reports_error(
        self, search_tool, mock_vector_store, error_search_results
    ):
        mock_vector_store.search.return_value = error_search_results

        result = search_tool.execute(query="anything")

        assert result == "Search error: connection refused"

    def test_returns_no_results_message_when_empty(
        self, search_tool, mock_vector_store, empty_search_results
    ):
        mock_vector_store.search.return_value = empty_search_results

        result = search_tool.execute(query="nonexistent topic")

        assert result == "No relevant content found."

    def test_no_results_message_includes_course_and_lesson_filter_info(
        self, search_tool, mock_vector_store, empty_search_results
    ):
        mock_vector_store.search.return_value = empty_search_results

        result = search_tool.execute(
            query="nonexistent topic", course_name="MCP", lesson_number=99
        )

        assert result == "No relevant content found in course 'MCP' in lesson 99."

    def test_does_not_raise_when_store_raises(self, search_tool, mock_vector_store):
        """CourseSearchTool has no try/except of its own -- if VectorStore.search
        raises instead of returning SearchResults.empty(...), execute() propagates
        the exception uncaught. Documents the gap rather than asserting it away."""
        mock_vector_store.search.side_effect = RuntimeError("boom")

        try:
            search_tool.execute(query="anything")
            raised = False
        except RuntimeError:
            raised = True

        assert raised, (
            "CourseSearchTool.execute propagates unhandled exceptions from the "
            "vector store; callers (ToolManager/AIGenerator) must guard against this."
        )


class TestToolManagerGuardsToolExecution:
    def test_execute_tool_catches_tool_exceptions_and_returns_error_string(
        self, tool_manager_with_search_tool, mock_vector_store
    ):
        mock_vector_store.search.side_effect = RuntimeError("boom")

        result = tool_manager_with_search_tool.execute_tool(
            "search_course_content", query="anything"
        )

        assert "Error executing tool 'search_course_content'" in result
        assert "boom" in result

    def test_execute_tool_catches_mismatched_arguments(self, tool_manager_with_search_tool):
        """Claude passing an argument name that doesn't match the tool's
        signature (e.g. a hallucinated param) used to raise a raw TypeError
        straight out of ToolManager; it's now caught and returned as a
        tool-result string instead."""
        result = tool_manager_with_search_tool.execute_tool(
            "search_course_content", query="q", course_title="MCP"
        )

        assert "Error executing tool 'search_course_content'" in result


class TestGetToolDefinition:
    def test_tool_definition_has_expected_shape(self, search_tool):
        definition = search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert set(definition["input_schema"]["properties"]) == {
            "query", "course_name", "lesson_number"
        }
        assert definition["input_schema"]["required"] == ["query"]
