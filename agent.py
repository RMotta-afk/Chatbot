import os

import google.generativeai as genai
from dotenv import load_dotenv

from tools import MEMORY_TOOLS, execute_memory_tool

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SYSTEM_PROMPT = """\
You are a helpful conversational assistant with persistent memory.

Guidelines:
- At the start of a conversation, search memory for relevant context about the user.
- When the user shares important information (preferences, facts, decisions, context), save it to memory.
- When the user references past interactions or previously shared info, search memory.
- Do NOT save trivial or redundant information.
- Be natural — do not narrate every memory operation to the user.
- Use memory results to personalize your responses without explicitly saying "I found in my memory..."
"""

_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[MEMORY_TOOLS],
    system_instruction=SYSTEM_PROMPT,
)

_sessions: dict[str, genai.ChatSession] = {}


def _get_or_create_session(session_id: str) -> genai.ChatSession:
    if session_id not in _sessions:
        _sessions[session_id] = _model.start_chat()
    return _sessions[session_id]


def reset_session(session_id: str) -> bool:
    return _sessions.pop(session_id, None) is not None


def chat(user_id: str, user_message: str, session_id: str) -> str:
    session = _get_or_create_session(session_id)
    response = session.send_message(user_message)

    for _ in range(10):
        # Collect function calls from response parts
        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call.name
        ]

        if not function_calls:
            # Extract text response
            texts = [
                part.text
                for part in response.candidates[0].content.parts
                if part.text
            ]
            return " ".join(texts) if texts else "I'm not sure how to respond to that."

        # Execute each function call and build responses
        function_responses = []
        for fc in function_calls:
            result = execute_memory_tool(fc.name, dict(fc.args), user_id)
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )

        response = session.send_message(function_responses)

    return "I'm having trouble processing your request. Please try again."
