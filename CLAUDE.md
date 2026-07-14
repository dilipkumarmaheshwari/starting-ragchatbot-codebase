# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Always use `uv` for running the server and managing dependencies — never invoke `pip` or bare `python`/`python3` directly in this repo.

```bash
uv sync                                              # install/sync dependencies
uv add <package>                                     # add a dependency (not pip install)
uv run <script>                                      # run any Python script/command
```

Requires a `.env` file in the repo root with `ANTHROPIC_API_KEY=...` (see `.env.example`).

Run the app (the only real entry point — `main.py` at the repo root is an unrelated placeholder script, not part of the app):

```bash
./run.sh                                             # cd backend && uv run uvicorn app:app --reload --port 8000
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

- Web UI: http://localhost:8000
- Swagger/API docs: http://localhost:8000/docs

There are no linter, formatter, or test configs/suites in this repo currently.

## Architecture

This is a tool-based RAG chatbot: FastAPI backend (`backend/`) + static vanilla JS/HTML/CSS frontend (`frontend/`, served directly by FastAPI via `StaticFiles`). ChromaDB is the vector store, Anthropic's Claude does generation, `sentence-transformers` (`all-MiniLM-L6-v2`) does embeddings.

**Request flow**: `frontend/script.js` → `POST /api/query` (`backend/app.py`) → `RAGSystem.query()` (`backend/rag_system.py`), which is the central orchestrator wiring together document processing, vector storage, AI generation, tools, and session history.

**Tool-based retrieval, not naive RAG**: Claude is *not* always given retrieved context up front. Instead `AIGenerator` (`backend/ai_generator.py`) calls Claude with a `search_course_content` tool available (defined in `backend/search_tools.py`), and Claude's system prompt instructs it to only search for course-specific questions (one search max) and answer general-knowledge questions directly. When Claude issues a `tool_use`, `AIGenerator._handle_tool_execution` runs the tool via `ToolManager`, feeds the formatted results back as a `tool_result` message, and makes a second Claude call (tools disabled) to synthesize the final answer. Sources surfaced during a search are stashed on `CourseSearchTool.last_sources` and pulled/reset by `RAGSystem` after each query — this is how citations reach the API response without threading them through the generation call.

**Vector store has two ChromaDB collections** (`backend/vector_store.py`):
- `course_catalog` — one entry per course (title as ID), used only to fuzzy-resolve a user/Claude-supplied course name (e.g. "MCP") to an exact title via a semantic query before filtering.
- `course_content` — the actual chunked lesson text, filtered by `course_title`/`lesson_number` and queried semantically.

**Document ingestion** (`backend/document_processor.py`): on startup, `app.py` loads every file in `docs/` via `RAGSystem.add_course_folder`, skipping courses whose title already exists in the store (so restarts don't re-embed). Course documents are plain text with a required header format:
```
Course Title: ...
Course Link: ...
Course Instructor: ...

Lesson 0: <title>
Lesson Link: ...
<lesson content...>

Lesson 1: <title>
...
```
Each lesson's text is chunked sentence-aware (`chunk_text`) with configurable size/overlap, and the first chunk of a lesson is prefixed with lesson context (e.g. `"Lesson 0 content: ..."`) so retrieval works without relying on metadata alone.

**Session history** (`backend/session_manager.py`) is in-memory only — keyed by `session_N` IDs, capped to the last `MAX_HISTORY` exchanges, not persisted across restarts.

**Config** (`backend/config.py`) is a single dataclass instance (`config`) covering the Anthropic model, embedding model, chunk size/overlap, max search results, history length, and the ChromaDB path — read by `RAGSystem.__init__` to wire up every component.
