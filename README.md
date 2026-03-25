# Chatbot — Conversational Agent with Persistent Memory

A Python conversational agent that remembers information across sessions using hybrid search (vector + BM25) over SQLite. Uses Gemini API with function calling for autonomous memory operations.

## Stack

- **Python 3.11+** with **FastAPI** and **Uvicorn**
- **SQLite + FTS5** for persistence and full-text search
- **sentence-transformers** (`all-MiniLM-L6-v2`) for embeddings
- **google-generativeai** (Gemini 2.0 Flash) for chat + tool calling

## Project Structure

```
database.py    — SQLite tables, FTS5 index, vector/BM25/hybrid search
embeddings.py  — Sentence-transformers wrapper (384-dim vectors)
tools.py       — Gemini function declarations + tool dispatcher
agent.py       — Chat session management + function-call loop
main.py        — FastAPI HTTP endpoints
```

## Setup

```bash
# Install dependencies
uv init
uv add fastapi uvicorn google-generativeai sentence-transformers python-dotenv

# Configure API key
cp .env.example .env  # or create .env manually
# Set GOOGLE_API_KEY=your_key_here in .env
```

## Running

```bash
uv run uvicorn main:app --reload --port 8000
```

## API Endpoints

| Method | Path          | Body                                          | Description          |
|--------|---------------|-----------------------------------------------|----------------------|
| GET    | `/health`     | —                                             | Health check         |
| POST   | `/chat`       | `{user_id, message, session_id?}`             | Send a chat message  |
| POST   | `/chat/reset` | `{session_id}`                                | Reset a chat session |

## Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Start a conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "message": "Hi! I prefer Python and work at Acme Corp.", "session_id": "s1"}'

# New session — agent recalls memories from previous conversations
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "message": "What framework should I use?", "session_id": "s2"}'
```

## How Memory Works

1. **Save**: When the user shares important info, Gemini calls `memory_save` — the content is embedded and stored in SQLite with an FTS5 index.
2. **Search**: When context is needed, Gemini calls `memory_search` — a hybrid search combines cosine similarity (vector, 70% weight) with BM25 (keyword, 30% weight) to find relevant memories.
3. **Recall**: Results are fed back to Gemini, which uses them to personalize responses without the user repeating themselves.
