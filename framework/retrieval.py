"""
Offline text retrieval using BM25 or TF-IDF.

No external dependencies required. Falls back from BM25 to TF-IDF if rank_bm25 not installed.
"""

from typing import List, Tuple
import re
import math
from collections import Counter


def _tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase + split on non-word chars."""
    return re.findall(r'\w+', text.lower())


def retrieve_top_k_tfidf(
    query: str,
    chunks: List[str],
    top_k: int = 6
) -> List[Tuple[int, float]]:
    """
    Retrieve top-k chunks using TF-IDF cosine similarity.

    Args:
        query: Search query
        chunks: List of text chunks
        top_k: Number of chunks to retrieve

    Returns:
        List of (chunk_index, score) tuples, sorted by score descending
    """
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    chunk_tokens = [_tokenize(ch) for ch in chunks]

    # Build vocabulary
    vocab = set()
    for tokens in chunk_tokens + [query_tokens]:
        vocab.update(tokens)
    vocab = sorted(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # Document frequency
    df = Counter()
    for tokens in chunk_tokens:
        for word in set(tokens):
            df[word] += 1

    N = len(chunks)

    def tfidf_vector(tokens: List[str]) -> List[float]:
        vec = [0.0] * len(vocab)
        tf = Counter(tokens)
        for word, count in tf.items():
            if word in word_to_idx:
                idx = word_to_idx[word]
                idf = math.log((N + 1) / (df.get(word, 0) + 1)) + 1
                vec[idx] = count * idf
        return vec

    query_vec = tfidf_vector(query_tokens)
    chunk_vecs = [tfidf_vector(tokens) for tokens in chunk_tokens]

    # Cosine similarity
    def cosine(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    scores = [(i, cosine(query_vec, ch_vec)) for i, ch_vec in enumerate(chunk_vecs)]
    # REPRODUCIBILITY: Stable sort by score (descending) then chunk_index (ascending)
    # This ensures deterministic ordering when scores are tied
    scores.sort(key=lambda x: (-x[1], x[0]))

    return scores[:top_k]


def retrieve_top_k_bm25(
    query: str,
    chunks: List[str],
    top_k: int = 6
) -> List[Tuple[int, float]]:
    """
    Retrieve top-k chunks using BM25 (if rank_bm25 is installed).

    Args:
        query: Search query
        chunks: List of text chunks
        top_k: Number of chunks to retrieve

    Returns:
        List of (chunk_index, score) tuples, sorted by score descending
    """
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]

        tokenized_chunks = [_tokenize(ch) for ch in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        query_tokens = _tokenize(query)
        scores = bm25.get_scores(query_tokens)

        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        # REPRODUCIBILITY: Stable sort by score (descending) then chunk_index (ascending)
        # This ensures deterministic ordering when scores are tied
        indexed_scores.sort(key=lambda x: (-x[1], x[0]))

        return indexed_scores[:top_k]
    except ImportError:
        # Fallback to TF-IDF (always available, no external dependencies)
        return retrieve_top_k_tfidf(query, chunks, top_k)


def retrieve_chunks(
    query: str,
    chunks: List[str],
    top_k: int = 6,
    method: str = "bm25"
) -> List[str]:
    """
    Retrieve top-k relevant chunks for a query.

    Args:
        query: Search query
        chunks: List of text chunks
        top_k: Number of chunks to retrieve
        method: 'bm25' (tries BM25, falls back to TF-IDF) or 'tfidf'

    Returns:
        List of retrieved chunks (in relevance order)
    """
    if not chunks:
        return []

    if method == "bm25":
        results = retrieve_top_k_bm25(query, chunks, top_k)
    else:
        results = retrieve_top_k_tfidf(query, chunks, top_k)

    # Return chunks in relevance order
    return [chunks[idx] for idx, _ in results]
