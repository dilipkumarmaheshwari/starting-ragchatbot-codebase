import anthropic
from typing import List, Optional, Dict, Any

DEFAULT_MAX_TOOL_ROUNDS = 2
API_ERROR_MESSAGE = "I'm having trouble reaching the AI service right now. Please try again in a moment."

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to two tools for course information: a content search tool and a course outline tool.

Tool Usage:
- **search_course_content**: Use for questions about specific course content, concepts, or detailed educational materials covered inside lessons.
- **get_course_outline**: Use for questions about course structure — course title, course link, lesson list, "what lessons are in X", "show me the outline/syllabus of X", or how many lessons a course has.
- **Up to 2 sequential tool-calling rounds per query.** After reviewing a tool's results, you may call a tool again — the same tool or a different one — if you still need more information to answer fully. Use a second round for cases like: resolving a lesson title from a course outline, then searching for that topic in another course; comparing content across two different courses or lessons; or answering a multi-part question where each part needs its own lookup.
- Don't make a second tool call just to re-confirm something you already know — stop and answer as soon as you have enough information. Most questions only need one tool call, and many need none.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Outline Responses:
- When answering outline/structure questions, always include: the course title, the course link, and the complete lesson list (each lesson's number and title)

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Search content first, then answer
- **Course structure/outline questions**: Get the outline first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the course outline"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.

        Claude may call a tool, see the result, and call another tool in a
        follow-up round (up to max_tool_rounds total rounds) before answering.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool-calling rounds

        Returns:
            Generated response as string
        """
        system_content = self._build_system_content(conversation_history)
        messages = [{"role": "user", "content": query}]
        current_tools = tools
        round_number = 1

        while True:
            response = self._call_claude(messages, system_content, current_tools)
            if response is None:
                return API_ERROR_MESSAGE

            # Terminate: no tool_use blocks, or no tool_manager to execute them
            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text(response)

            messages = messages + [{"role": "assistant", "content": response.content}]
            tool_results = self._execute_tool_calls(response, tool_manager)
            if tool_results:
                messages = messages + [{"role": "user", "content": tool_results}]

            # Once the round budget is spent, the next call gets no tools --
            # Claude can then only answer, which terminates via stop_reason above.
            current_tools = tools if round_number < max_tool_rounds else None
            round_number += 1

    def _call_claude(self, messages: List[Dict[str, Any]], system: str, tools: Optional[List]):
        """
        Single choke point for every Anthropic API call this method makes,
        across all rounds. Returns None on any transport/API failure so the
        caller can degrade gracefully instead of raising.
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system
        }
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        try:
            return self.client.messages.create(**api_params)
        except Exception:
            return None

    def _execute_tool_calls(self, response, tool_manager) -> List[Dict[str, Any]]:
        """Execute every tool_use block in a response and build tool_result content."""
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name,
                    **content_block.input
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        return tool_results

    def _extract_text(self, response) -> str:
        return response.content[0].text

    def _build_system_content(self, conversation_history: Optional[str]) -> str:
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
