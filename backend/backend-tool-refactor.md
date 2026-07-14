Refactor @backend/ai_generator.py to support sequential tool calling where Claude can make up to 2 tool call in seprate API rounds.

Current behaviour:
- Claude make 1 tool call -> tools are removed from API params -> final response
- If claude wants another tool call after seeing results, it can't (gets empty response)


Desired behaviour:
- Each tool call should be a seprate API request where Claude can reason about previous results
- Support for complex queries requiring multiple searches for comparisons, multi-part  questions, or when information from   different courses/lessons is needed

Example flow:
1. User: "Search for a course that discusses the same topic as lesson 4 of course X"
2. Claude: get course outline for Course X -> gets title of lesson 4
3. Claude: Uses the title to search for a course that discusses the same topic -> returns course information 
4. Claude: provides complete answer

Requirements:
- Maximum 2 sequential rounds per user query 
- Terminate when: (a) rounds completed, (b) Claude's response has no tool_use blocks or  (c) tool call fails
- Preserve conversion contex between rounds
- Handle tool execution error gracefully

Notes:
- Update the system prompt in @backend/ai_generator.py
- update the tests @backend/tests/test_ai_generator.py
- write tests that verify the external behaviour (API Calls made, tools executed, result returned) rather than internal state details.


Use two parallel subagents to brainstrom possible plans. Do not implement any code.