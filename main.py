"""FastAPI app exposing /documents (ingest) and /chat (streaming RAG)."""
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from ingest import ingest_text
from retrieval import retrieve
from chat import chat_stream

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    yield
    await app.state.pool.close()


app = FastAPI(title="RAG demo", lifespan=lifespan)


@app.get("/health")
async def health():
    async with app.state.pool.acquire() as conn:
        n = await conn.fetchval("SELECT count(*) FROM chunks")
    return {"status": "ok", "chunks": n}


@app.post("/documents")
async def upload_document(
    title: str = Form(...),
    file: UploadFile = File(...),
    x_tenant_id: str = Header(..., alias="x-tenant-id"),
):
    text = (await file.read()).decode("utf-8", errors="replace")
    if not text.strip():
        raise HTTPException(400, "Empty document")
    result = await ingest_text(
        app.state.pool, x_tenant_id, title, file.filename, text,
    )
    return result


class ChatRequest(BaseModel):
    question: str
    top_k: int = 8


@app.post("/chat")
async def chat(
    body: ChatRequest,
    x_tenant_id: str = Header(..., alias="x-tenant-id"),
):
    chunks = await retrieve(app.state.pool, x_tenant_id, body.question, limit=body.top_k)
    if not chunks:
        raise HTTPException(404, "No documents indexed for this tenant")

    async def event_stream():
        # Send the citation list first as a JSON line, then the streamed answer
        citations = [
            {
                "chunk_index": c.chunk_index,
                "document_id": c.document_id,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "score": round(c.score, 4),
            }
            for c in chunks
        ]
        import json
        yield f"data: {json.dumps({'citations': citations})}\n\n"

        async for text in chat_stream(chunks, body.question):
            yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
