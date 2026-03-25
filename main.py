from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import chat, reset_session
from database import init_db

app = FastAPI(title="Conversational Agent with Persistent Memory")


class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or f"{req.user_id}_{uuid4().hex[:8]}"
    try:
        response = chat(req.user_id, req.message, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(response=response, session_id=session_id)


@app.post("/chat/reset")
def reset_endpoint(req: ResetRequest):
    reset_session(req.session_id)
    return {"status": "session reset"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
