"""
Ретривер по Qdrant на BGE-M3 (dense) + MMR.
Зависимости: qdrant-client, sentence-transformers, numpy.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# Конфиг через env
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rasa_mvp")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

TOPK_FETCH = int(os.getenv("TOPK_FETCH", "20"))
TOPK_RETURN = int(os.getenv("TOPK_RETURN", "8"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.4"))

_client: Optional[QdrantClient] = None
_model: Optional[SentenceTransformer] = None

def _client_singleton() -> QdrantClient:
    """Lazily instantiate the Qdrant client."""

    global _client
    if _client is None:
        # Настройки берём из ENV, чтобы их можно было переопределить без деплоя.
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    return _client

def _model_singleton() -> SentenceTransformer:
    """Return a shared SentenceTransformer instance."""

    global _model
    if _model is None:
        # Loading the embedding model can take noticeable time and memory, so we
        # keep it around for the duration of the process instead of reloading it
        # on every request.
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def _embed(texts: List[str]) -> np.ndarray:
    """Vectorise the provided texts with the configured embedding model."""

    m = _model_singleton()
    vecs = m.encode(texts, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
    # Cast to float32 to keep memory usage predictable across different models.
    return np.asarray(vecs, dtype=np.float32)

def _mmr(q: np.ndarray, C: np.ndarray, k: int, lam: float) -> List[int]:
    """Select ``k`` diverse results using Maximal Marginal Relevance."""

    k = min(k, C.shape[0])
    rel = C @ q  # косинусная близость при условии нормированных векторов
    sel = []  # индексы уже выбранных кандидатов
    avail = set(range(C.shape[0]))
    for _ in range(k):
        if not sel:
            i = int(np.argmax(rel))
            sel.append(i); avail.remove(i); continue
        S = C[sel]
        div = S @ C.T
        max_div = div.max(axis=0)
        mmr = lam * rel - (1 - lam) * max_div
        i = max(avail, key=lambda j: mmr[j])
        sel.append(i); avail.remove(i)
    return sel

def search_hybrid(query: str, lang_filter: Optional[List[str]] = None, topk: int = TOPK_RETURN) -> List[Dict[str, Any]]:
    """Perform a dense similarity search with an optional language filter."""

    cli = _client_singleton()
    # эмбеддинг запроса
    qvec = _embed([query])[0]
    # фильтр
    flt = None
    if lang_filter:
        flt = qm.Filter(must=[qm.FieldCondition(key="lang", match=qm.MatchAny(any=lang_filter))])
    # поиск
    res = cli.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        query_filter=flt,
        limit=max(TOPK_FETCH, topk),
        with_payload=True,
        with_vectors=True,
    )
    if not res:
        # Qdrant возвращает None или пустой список, если совпадений нет.
        return []
    # Split the structured response into primitive containers so we can apply
    # numpy-powered post-processing without keeping references to the rich
    # client objects returned by qdrant-client.
    C, ids, payloads, scores = [], [], [], []
    for r in res:
        # qdrant-client может возвращать вектор в словаре (multi-vector search),
        # поэтому аккуратно достаём «dense» представление либо первое попавшееся.
        v = r.vector if not isinstance(r.vector, dict) else (r.vector.get("dense") or next(iter(r.vector.values())))
        C.append(np.asarray(v, dtype=np.float32))
        # Храним id/score/payload отдельно, чтобы вернуть лёгкие словари.
        ids.append(str(r.id)); payloads.append(r.payload or {}); scores.append(float(r.score))
    C = np.vstack(C)
    # Apply MMR over the candidate vectors to reduce redundancy between
    # neighbouring passages that may share identical sentences.
    idx = _mmr(qvec, C, k=topk, lam=MMR_LAMBDA)
    out = [{"id": ids[i], "score": scores[i], "payload": payloads[i]} for i in idx]
    return out
