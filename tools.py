import google.generativeai as genai

from database import hybrid_search, save_memory
from embeddings import get_embedding

MEMORY_TOOLS = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="memory_search",
            description=(
                "Search the user's stored memories. Use when the user references past conversations, "
                "asks about something they previously mentioned, or when prior context would help "
                "provide a better response."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "query": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The search query to find relevant memories.",
                    ),
                    "limit": genai.protos.Schema(
                        type=genai.protos.Type.INTEGER,
                        description="Maximum number of results to return (default 5).",
                    ),
                },
                required=["query"],
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="memory_save",
            description=(
                "Save important information the user shares to long-term memory. Use when the user "
                "shares preferences, personal facts, decisions, or important context. "
                "Categories: preference, fact, decision, context, general."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "content": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The information to save.",
                    ),
                    "category": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Category: preference, fact, decision, context, or general.",
                    ),
                },
                required=["content", "category"],
            ),
        ),
    ]
)


def execute_memory_tool(name: str, args: dict, user_id: str) -> str:
    if name == "memory_search":
        query = args["query"]
        limit = int(args.get("limit", 5))
        query_embedding = get_embedding(query)
        results = hybrid_search(user_id, query, query_embedding, limit=limit)
        if not results:
            return "No memories found."
        lines = []
        for r in results:
            lines.append(
                f"[Score: {r['hybrid_score']:.2f}] ({r['category']}) {r['content']} (saved: {r['created_at']})"
            )
        return "\n".join(lines)

    elif name == "memory_save":
        content = args["content"]
        category = args.get("category", "general")
        embedding = get_embedding(content)
        result = save_memory(user_id, content, category, embedding)
        return f"Memory saved (id={result['id']}): {result['content']}"

    else:
        return f"Unknown tool: {name}"
