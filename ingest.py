"""Document ingestion: chunk with LlamaIndex, embed with OpenAI, insert into pgvector."""
import os
import asyncio
import sys
from uuid import uuid4
from pathlib import Path

import asyncpg
from openai import AsyncOpenAI
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
oai = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenAI. Returns one vector per input."""
    resp = await oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def chunk_document(text: str) -> list[tuple[int, int, str]]:
    """Split text into chunks. Returns list of (char_start, char_end, chunk_text)."""
    nodes = splitter.get_nodes_from_documents(
        [type("Doc", (), {"text": text, "metadata": {}, "id_": "x", "embedding": None,
                          "extra_info": {}, "excluded_embed_metadata_keys": [],
                          "excluded_llm_metadata_keys": [], "relationships": {}})()]
    )
    out = []
    cursor = 0
    for n in nodes:
        chunk_text = n.text
        # Find chunk position in original text (approximate, via search from cursor)
        idx = text.find(chunk_text, cursor)
        if idx < 0:
            idx = cursor
        out.append((idx, idx + len(chunk_text), chunk_text))
        cursor = idx + 1
    return out


async def ingest_text(
    pool: asyncpg.Pool,
    tenant_id: str,
    title: str,
    source: str | None,
    text: str,
) -> dict:
    """Ingest a single document: chunk, embed in batches of 50, insert all chunks in one transaction."""
    chunks = chunk_document(text)
    if not chunks:
        return {"document_id": None, "chunk_count": 0}

    # Embed in batches of 50 to fit OpenAI request limits
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), 50):
        batch = [c[2] for c in chunks[i : i + 50]]
        embeddings.extend(await embed_batch(batch))

    document_id = uuid4()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO documents (id, tenant_id, title, source, char_count) VALUES ($1, $2, $3, $4, $5)",
                document_id, tenant_id, title, source, len(text),
            )
            rows = [
                (
                    uuid4(), tenant_id, document_id, idx,
                    char_start, char_end, chunk_text,
                    "[" + ",".join(str(x) for x in embedding) + "]",
                )
                for idx, ((char_start, char_end, chunk_text), embedding) in enumerate(zip(chunks, embeddings))
            ]
            await conn.executemany(
                """
                INSERT INTO chunks (id, tenant_id, document_id, chunk_index, char_start, char_end, text, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
                """,
                rows,
            )

    return {"document_id": str(document_id), "chunk_count": len(chunks)}


async def main(corpus_dir: str = "samples") -> None:
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    try:
        for path in Path(corpus_dir).glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            result = await ingest_text(pool, "demo", path.stem, str(path), text)
            print(f"Ingested {path.name}: doc_id={result['document_id']} chunks={result['chunk_count']}")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "samples"))
