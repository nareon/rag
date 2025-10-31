"""Tests for helpers in full.ingest."""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Provide stubs for optional heavy dependencies so the module can be imported.
if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _DummyResponse:  # pragma: no cover - helper stub
        text = ""

    requests_stub.get = lambda *_args, **_kwargs: _DummyResponse()
    sys.modules["requests"] = requests_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")

    class _DummySoup:  # pragma: no cover - helper stub
        def __init__(self, text: str, *_args, **_kwargs) -> None:
            self._text = text

        def __call__(self, *_args, **_kwargs):
            return []

        def get_text(self, separator: str = " ", strip: bool = False) -> str:
            return self._text

    bs4_stub.BeautifulSoup = _DummySoup  # type: ignore[attr-defined]
    sys.modules["bs4"] = bs4_stub

if "qdrant_client" not in sys.modules:
    qdrant_stub = types.ModuleType("qdrant_client")

    class _DummyQdrantClient:  # pragma: no cover - helper stub
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    qdrant_stub.QdrantClient = _DummyQdrantClient  # type: ignore[attr-defined]
    sys.modules["qdrant_client"] = qdrant_stub

if "qdrant_client.http" not in sys.modules:
    qdrant_http_stub = types.ModuleType("qdrant_client.http")
    sys.modules["qdrant_client.http"] = qdrant_http_stub

if "qdrant_client.http.models" not in sys.modules:
    qdrant_models_stub = types.ModuleType("qdrant_client.http.models")

    class _DummyVectorParams:  # pragma: no cover - helper stub
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _DummyDistance:  # pragma: no cover - helper stub
        COSINE = "cosine"

    qdrant_models_stub.VectorParams = _DummyVectorParams  # type: ignore[attr-defined]
    qdrant_models_stub.Distance = _DummyDistance  # type: ignore[attr-defined]
    sys.modules["qdrant_client.http.models"] = qdrant_models_stub
    setattr(sys.modules["qdrant_client.http"], "models", qdrant_models_stub)

if "sentence_transformers" not in sys.modules:
    st_stub = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:  # pragma: no cover - helper stub
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def encode(self, texts, **_kwargs):
            return [[0.0] * 3 for _ in texts]

    st_stub.SentenceTransformer = _DummySentenceTransformer  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = st_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable, **_kwargs: iterable  # pragma: no cover - helper stub
    sys.modules["tqdm"] = tqdm_stub


ingest = importlib.import_module("full.ingest")


def test_chunk_text_handles_large_overlap():
    text = " ".join(f"w{i}" for i in range(10))

    chunks = list(ingest.chunk_text(text, max_len=3, overlap=5))

    assert chunks[:3] == ["w0 w1 w2", "w1 w2 w3", "w2 w3 w4"]
    assert chunks[-1] == "w9"
