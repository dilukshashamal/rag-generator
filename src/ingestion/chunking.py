import hashlib

import tiktoken

from src.config import Settings
from src.models.schemas import ChunkData, Section


def chunk_sections(sections: list[Section], settings: Settings) -> list[ChunkData]:
    """Token windows never cross a page/heading; prefer paragraph endings after 450 tokens."""
    encoding = tiktoken.get_encoding(settings.tokenizer_encoding)
    chunks: list[ChunkData] = []
    for section in sections:
        tokens = encoding.encode(section.text, disallowed_special=())
        start = 0
        while start < len(tokens):
            end = min(start + settings.chunk_tokens, len(tokens))
            content = encoding.decode(tokens[start:end])
            if end < len(tokens):
                boundary = content.rfind("\n\n")
                if boundary > 0:
                    prefix = encoding.encode(content[:boundary], disallowed_special=())
                    if len(prefix) >= 450:
                        end = start + len(prefix)
                        content = encoding.decode(tokens[start:end])
            if content.strip():
                chunks.append(ChunkData(text=content.strip(), page=section.page,
                                        heading=section.heading, chunk_index=len(chunks),
                                        content_hash=hashlib.sha256(content.strip().encode()).hexdigest()))
            if end == len(tokens):
                break
            start = end - settings.chunk_overlap
    return chunks
