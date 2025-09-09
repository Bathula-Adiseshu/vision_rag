from typing import Iterable, List


def sliding_window_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


