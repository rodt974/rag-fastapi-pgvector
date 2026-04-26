"""Hybrid retrieval: vector cosine + BM25 fused via Reciprocal Rank Fusion.

The fusion logic lives in the SQL function `hybrid_search` defined in schema.sql.
This module is a thin async wrapper that embeds the query and calls the function.
"""
import os
from dataclasses import dataclass

import asyncpg
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")


@dataclass
class RetrievedChunk:
    id: str
    document_id: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    score: float


async def retrieve(
    pool: asyncpg.Pool,
    tenant_id: str,
    query: str,
    limit: int = 8,
) -> list[RetrievedChunk]:
    """Embed the query, then call the SQL hybrid_search function. Returns the top chunks."""
    resp = await oai.embeddings.create(model=EMBED_MODEL, input=[query])
    embedding = resp.data[0].embedding
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, document_id, chunk_index, char_start, char_end, text, score "
            "FROM hybrid_search($1, $2, $3::vector, $4)",
            tenant_id, query, embedding_str, limit,
        )

    return [
        RetrievedChunk(
            id=str(r["id"]),
            document_id=str(r["document_id"]),
            chunk_index=r["chunk_index"],
            char_start=r["char_start"],
            char_end=r["char_end"],
            text=r["text"],
            score=float(r["score"]),
        )
        for r in rows
    ]
