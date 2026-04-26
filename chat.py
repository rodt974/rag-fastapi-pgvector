"""Streaming chat over retrieved chunks using Anthropic Claude with prompt caching.

The system prompt is marked as cacheable (cache_control=ephemeral) so subsequent
requests within 5 minutes get a 90% discount on those tokens. The retrieved
chunks change per query so they go in the user message, not cached.
"""
import os
from typing import AsyncIterator

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from retrieval import RetrievedChunk

load_dotenv()

anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You answer questions strictly based on the retrieved context provided below.

Rules:
- Cite each claim with [chunk_index] from the context
- If the answer is not in the context, say so explicitly. Do not make up answers.
- Be concise. Bullet points when listing.
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.chunk_index}] (doc {c.document_id[:8]}) {c.text}")
    return "\n\n".join(parts)


async def chat_stream(
    chunks: list[RetrievedChunk],
    question: str,
) -> AsyncIterator[str]:
    """Yields text chunks as Claude streams the response."""
    context = format_context(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    async with anthropic.messages.stream(
        model=CHAT_MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
