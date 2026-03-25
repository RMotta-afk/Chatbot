import sqlite3
import struct
import math
import re
from datetime import datetime, timezone

DB_PATH = "chatbot.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            embedding BLOB,
            created_at TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT
        )
    """)
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE memories_fts
            USING fts5(content, category, content_rowid='id', tokenize='porter')
        """)
    except sqlite3.OperationalError:
        pass  # Already exists
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai
        AFTER INSERT ON memories
        BEGIN
            INSERT INTO memories_fts(rowid, content, category)
            VALUES (NEW.id, NEW.content, NEW.category);
        END
    """)
    conn.commit()
    conn.close()


def save_memory(user_id: str, content: str, category: str, embedding: list[float]) -> dict:
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    blob = struct.pack(f"{len(embedding)}f", *embedding)
    cur = conn.execute(
        "INSERT INTO memories (user_id, content, category, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, content, category, blob, now),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return {"id": row_id, "content": content, "category": category, "created_at": now}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_by_vector(user_id: str, query_embedding: list[float], limit: int = 5) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, content, category, embedding, created_at FROM memories WHERE user_id = ?",
        (user_id,),
    ).fetchall()

    results = []
    for row in rows:
        if row["embedding"] is None:
            continue
        blob = row["embedding"]
        dim = len(blob) // 4
        emb = list(struct.unpack(f"{dim}f", blob))
        score = _cosine_similarity(query_embedding, emb)
        results.append({
            "id": row["id"],
            "content": row["content"],
            "category": row["category"],
            "created_at": row["created_at"],
            "vector_score": score,
        })

    results.sort(key=lambda x: x["vector_score"], reverse=True)
    results = results[:limit]

    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, r["id"]),
        )
    conn.commit()
    conn.close()
    return results


def search_by_bm25(user_id: str, query: str, limit: int = 5) -> list[dict]:
    conn = _get_conn()
    sanitized = re.sub(r"[^\w\s]", "", query).strip()
    if not sanitized:
        conn.close()
        return []
    rows = conn.execute(
        """
        SELECT m.id, m.content, m.category, m.created_at, -fts.rank AS bm25_score
        FROM memories_fts fts
        JOIN memories m ON m.id = fts.rowid
        WHERE fts.memories_fts MATCH ? AND m.user_id = ?
        ORDER BY fts.rank
        LIMIT ?
        """,
        (sanitized, user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def hybrid_search(
    user_id: str,
    query: str,
    query_embedding: list[float],
    limit: int = 5,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    vec_results = search_by_vector(user_id, query_embedding, limit=20)
    bm25_results = search_by_bm25(user_id, query, limit=20)

    def _normalize(items, key):
        if not items:
            return
        scores = [it[key] for it in items]
        mn, mx = min(scores), max(scores)
        for it in items:
            it[key] = 1.0 if mx == mn else (it[key] - mn) / (mx - mn)

    _normalize(vec_results, "vector_score")
    _normalize(bm25_results, "bm25_score")

    candidates: dict[int, dict] = {}
    for r in vec_results:
        candidates[r["id"]] = {
            "id": r["id"],
            "content": r["content"],
            "category": r["category"],
            "created_at": r["created_at"],
            "vector_score": r["vector_score"],
            "bm25_score": 0.0,
        }
    for r in bm25_results:
        if r["id"] in candidates:
            candidates[r["id"]]["bm25_score"] = r["bm25_score"]
        else:
            candidates[r["id"]] = {
                "id": r["id"],
                "content": r["content"],
                "category": r["category"],
                "created_at": r["created_at"],
                "vector_score": 0.0,
                "bm25_score": r["bm25_score"],
            }

    for c in candidates.values():
        c["hybrid_score"] = vector_weight * c["vector_score"] + bm25_weight * c["bm25_score"]

    ranked = sorted(candidates.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return ranked[:limit]
