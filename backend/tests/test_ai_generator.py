"""Tests for AIGenerator in backend/ai_generator.py.

The Anthropic client is patched out entirely (no network calls) so these
tests isolate AIGenerator's own orchestration logic: does it correctly wire
up tools, detect a tool_use stop_reason, call ToolManager with the right
arguments, and feed results back for a second turn.
"""
from unittest.mock import MagicMock, patch

import pytest

from ai_generator import AIGenerator
from tests.conftest import FakeResponse, FakeTextBlock, FakeToolUseBlock


@pytest.fixture
def generator():
    with patch("ai_generator.anthropic.Anthropic"):
        gen = AIGenerator(api_key="test-key", model="claude-sonnet-4-5")
    return gen


@pytest.fixture
def search_tool_def():
    return {
        "name": "search_course_content",
        "description": "Search course materials",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "course_name": {"type": "string"},
                "lesson_number": {"type": "integer"},
            },
            "required": ["query"],
        },
    }


class TestNoToolUse:
    def test_direct_answer_returned_when_no_tools_supplied(self, generator):
        generator.client.messages.create = MagicMock(
            return_value=FakeResponse([FakeTextBlock("Paris is the capital of France.")], "end_turn")
        )

        result = generator.generate_response(query="What is the capital of France?")

        assert result == "Paris is the capital of France."
        generator.client.messages.create.assert_called_once()

    def test_tools_and_tool_choice_omitted_when_no_tools_supplied(self, generator):
        generator.client.messages.create = MagicMock(
            return_value=FakeResponse([FakeTextBlock("answer")], "end_turn")
        )

        generator.generate_response(query="general knowledge question")

        call_kwargs = generator.client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_tools_and_tool_choice_included_when_tools_supplied(self, generator, search_tool_def):
        generator.client.messages.create = MagicMock(
            return_value=FakeResponse([FakeTextBlock("answer")], "end_turn")
        )

        generator.generate_response(query="q", tools=[search_tool_def], tool_manager=MagicMock())

        call_kwargs = generator.client.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == [search_tool_def]
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_tool_manager_not_invoked_when_claude_answers_directly(self, generator, search_tool_def):
        tool_manager = MagicMock()
        generator.client.messages.create = MagicMock(
            return_value=FakeResponse([FakeTextBlock("no search needed")], "end_turn")
        )

        generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

        tool_manager.execute_tool.assert_not_called()

    def test_conversation_history_folded_into_system_prompt(self, generator):
        generator.client.messages.create = MagicMock(
            return_value=FakeResponse([FakeTextBlock("answer")], "end_turn")
        )

        generator.generate_response(query="q", conversation_history="User: hi\nAssistant: hello")

        system = generator.client.messages.create.call_args.kwargs["system"]
        assert "Previous conversation:" in system
        assert "User: hi\nAssistant: hello" in system


class TestToolUseFlow:
    def test_search_course_content_tool_called_with_claudes_arguments(
        self, generator, search_tool_def
    ):
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "[MCP Course - Lesson 3]\nA tool has a name."

        first = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "what is a tool", "course_name": "MCP"})],
            "tool_use",
        )
        second = FakeResponse([FakeTextBlock("A tool has a name, description, and schema.")], "end_turn")
        generator.client.messages.create = MagicMock(side_effect=[first, second])

        result = generator.generate_response(
            query="What is a tool in the MCP course?",
            tools=[search_tool_def],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="what is a tool", course_name="MCP"
        )
        assert result == "A tool has a name, description, and schema."
        assert generator.client.messages.create.call_count == 2

    def test_second_api_call_still_offers_tools_within_round_budget(self, generator, search_tool_def):
        """With the default 2-round budget, round 2 is NOT the forced final
        call -- Claude is still within budget and may choose a second,
        distinct tool call, so tools must still be offered."""
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "some result"

        first = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q"})], "tool_use"
        )
        second = FakeResponse([FakeTextBlock("final answer")], "end_turn")
        generator.client.messages.create = MagicMock(side_effect=[first, second])

        generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

        second_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
        assert second_call_kwargs["tools"] == [search_tool_def]
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    def test_tool_result_message_references_correct_tool_use_id(self, generator, search_tool_def):
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "formatted results"

        first = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q"}, id_="toolu_abc123")],
            "tool_use",
        )
        second = FakeResponse([FakeTextBlock("final answer")], "end_turn")
        generator.client.messages.create = MagicMock(side_effect=[first, second])

        generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

        second_call_messages = generator.client.messages.create.call_args_list[1].kwargs["messages"]
        tool_result_message = second_call_messages[-1]
        assert tool_result_message["role"] == "user"
        assert tool_result_message["content"][0]["tool_use_id"] == "toolu_abc123"
        assert tool_result_message["content"][0]["content"] == "formatted results"

    def test_multiple_tool_calls_in_single_turn_all_executed(self, generator, search_tool_def):
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["result 1", "result 2"]

        first = FakeResponse(
            [
                FakeToolUseBlock("search_course_content", {"query": "q1"}, id_="id1"),
                FakeToolUseBlock("search_course_content", {"query": "q2"}, id_="id2"),
            ],
            "tool_use",
        )
        second = FakeResponse([FakeTextBlock("combined answer")], "end_turn")
        generator.client.messages.create = MagicMock(side_effect=[first, second])

        generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

        assert tool_manager.execute_tool.call_count == 2

    def test_tool_use_with_no_tool_manager_returns_raw_first_response_text(self, generator, search_tool_def):
        """If tool_use is signaled but no tool_manager is supplied, generate_response
        falls through to `response.content[0].text` -- but content[0] is a
        tool_use block, which has no `.text` attribute. This should surface,
        not silently swallow, the mismatch."""
        first = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q"})], "tool_use"
        )
        generator.client.messages.create = MagicMock(return_value=first)

        with pytest.raises(AttributeError):
            generator.generate_response(query="q", tools=[search_tool_def], tool_manager=None)


class TestSequentialToolRounds:
    def test_two_sequential_tool_rounds_then_final_answer(self, generator, search_tool_def):
        """Mirrors the task's example flow: round 1 resolves a course outline,
        round 2 uses info from round 1 to search, round 2's own answer is final."""
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = [
            "Course Title: Course X\nLessons:\n  4. Retrieval Basics",
            "[Course Y - Lesson 2]\nRetrieval basics content...",
        ]

        round_1 = FakeResponse(
            [FakeToolUseBlock("get_course_outline", {"course_name": "Course X"}, id_="id1")],
            "tool_use",
        )
        round_2 = FakeResponse(
            [FakeToolUseBlock(
                "search_course_content",
                {"query": "Retrieval Basics"},
                id_="id2",
            )],
            "tool_use",
        )
        final = FakeResponse([FakeTextBlock("Course Y covers the same topic.")], "end_turn")
        generator.client.messages.create = MagicMock(side_effect=[round_1, round_2, final])

        result = generator.generate_response(
            query="Find a course covering the same topic as lesson 4 of Course X",
            tools=[search_tool_def],
            tool_manager=tool_manager,
        )

        assert generator.client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Course X")
        tool_manager.execute_tool.assert_any_call("search_course_content", query="Retrieval Basics")
        assert result == "Course Y covers the same topic."

    def test_round_2_offers_tools_and_completes_without_a_third_call(self, generator, search_tool_def):
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["result 1", "result 2"]

        round_1 = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q1"}, id_="id1")], "tool_use"
        )
        round_2 = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q2"}, id_="id2")], "tool_use"
        )
        final = FakeResponse([FakeTextBlock("combined answer")], "end_turn")
        generator.client.messages.create = MagicMock(side_effect=[round_1, round_2, final])

        result = generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

        assert generator.client.messages.create.call_count == 3
        second_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
        assert second_call_kwargs["tools"] == [search_tool_def]
        assert result == "combined answer"

    def test_round_budget_exhausted_forces_final_call_without_tools(self, generator, search_tool_def):
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["result 1", "result 2"]

        round_1 = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q1"}, id_="id1")], "tool_use"
        )
        round_2 = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q2"}, id_="id2")], "tool_use"
        )
        # Round 3 is offered no tools (budget spent), so it can only answer.
        forced_final = FakeResponse([FakeTextBlock("final synthesis")], "end_turn")
        generator.client.messages.create = MagicMock(side_effect=[round_1, round_2, forced_final])

        result = generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

        assert generator.client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2
        third_call_kwargs = generator.client.messages.create.call_args_list[2].kwargs
        assert "tools" not in third_call_kwargs
        assert "tool_choice" not in third_call_kwargs
        assert result == "final synthesis"

    def test_tool_error_string_flows_to_claude_as_content_not_a_short_circuit(
        self, generator, search_tool_def
    ):
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Error executing tool 'search_course_content': boom"

        round_1 = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q"}, id_="id1")], "tool_use"
        )
        round_2 = FakeResponse(
            [FakeTextBlock("I couldn't complete that search due to an error.")], "end_turn"
        )
        generator.client.messages.create = MagicMock(side_effect=[round_1, round_2])

        result = generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

        assert generator.client.messages.create.call_count == 2
        second_call_messages = generator.client.messages.create.call_args_list[1].kwargs["messages"]
        tool_result_message = second_call_messages[-1]
        assert tool_result_message["content"][0]["content"] == (
            "Error executing tool 'search_course_content': boom"
        )
        assert result == "I couldn't complete that search due to an error."

    def test_api_failure_on_forced_final_round_returns_graceful_message(self, generator, search_tool_def):
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["result 1", "result 2"]

        round_1 = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q1"}, id_="id1")], "tool_use"
        )
        round_2 = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q2"}, id_="id2")], "tool_use"
        )
        generator.client.messages.create = MagicMock(
            side_effect=[round_1, round_2, ConnectionError("upstream connection reset")]
        )

        result = generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

        assert "trouble reaching the AI service" in result
        assert generator.client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2

    def test_max_tool_rounds_override_reproduces_old_single_round_behavior(self, generator, search_tool_def):
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "some result"

        first = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q"})], "tool_use"
        )
        second = FakeResponse([FakeTextBlock("final answer")], "end_turn")
        generator.client.messages.create = MagicMock(side_effect=[first, second])

        result = generator.generate_response(
            query="q", tools=[search_tool_def], tool_manager=tool_manager, max_tool_rounds=1
        )

        assert generator.client.messages.create.call_count == 2
        second_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
        assert "tools" not in second_call_kwargs
        assert "tool_choice" not in second_call_kwargs
        assert result == "final answer"


class TestSystemPromptSupportsMultipleRounds:
    def test_system_prompt_no_longer_caps_at_one_tool_call(self):
        assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_describes_sequential_rounds(self):
        assert "2 sequential tool-calling rounds" in AIGenerator.SYSTEM_PROMPT


class TestToolExecutionErrorHandling:
    def test_exception_from_a_raw_tool_manager_still_propagates(self, generator, search_tool_def):
        """AIGenerator itself adds no guard around tool_manager.execute_tool() --
        that guard now lives in ToolManager.execute_tool (see
        test_course_search_tool.py::TestToolManagerGuardsToolExecution). A caller
        that hands AIGenerator something other than a real ToolManager (e.g. this
        raw MagicMock) gets no protection, by design -- the guard isn't duplicated
        here."""
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = TypeError(
            "execute() got an unexpected keyword argument 'course_title'"
        )

        first = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q", "course_title": "MCP"})],
            "tool_use",
        )
        generator.client.messages.create = MagicMock(return_value=first)

        with pytest.raises(TypeError):
            generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

    def test_initial_api_call_failure_returns_graceful_message(self, generator, search_tool_def):
        """A transient/auth/rate-limit error on the FIRST Claude call is now
        caught and turned into a user-facing message instead of raising."""
        generator.client.messages.create = MagicMock(
            side_effect=ConnectionError("upstream connection reset")
        )

        result = generator.generate_response(query="q", tools=[search_tool_def], tool_manager=MagicMock())

        assert "trouble reaching the AI service" in result

    def test_second_api_call_failure_returns_graceful_message(self, generator, search_tool_def):
        """A transient/auth/rate-limit error on the SECOND (post-tool) Claude
        call is now caught too. Content-related queries make two API calls
        (initial + follow-up) vs. one for general-knowledge answers, so before
        this fix they were twice as likely to hit a transient failure like this."""
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "some result"

        first = FakeResponse(
            [FakeToolUseBlock("search_course_content", {"query": "q"})], "tool_use"
        )
        generator.client.messages.create = MagicMock(
            side_effect=[first, ConnectionError("upstream connection reset")]
        )

        result = generator.generate_response(query="q", tools=[search_tool_def], tool_manager=tool_manager)

        assert "trouble reaching the AI service" in result
