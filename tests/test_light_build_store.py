"""Tests for helper utilities in light.build_store."""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Provide lightweight stubs for optional heavy dependencies used in the module.
if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.asarray = lambda data, dtype=None: data
    numpy_stub.float32 = float
    numpy_stub.save = lambda *_args, **_kwargs: None
    numpy_stub.load = lambda *_args, **_kwargs: None
    sys.modules["numpy"] = numpy_stub

if "sentence_transformers" not in sys.modules:
    st_stub = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:  # pragma: no cover - helper stub
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def encode(self, texts, **_kwargs):
            return [[1.0] * 3 for _ in texts]

    st_stub.SentenceTransformer = _DummySentenceTransformer  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = st_stub

build_store = importlib.import_module("light.build_store")


def test_chunk_text_respects_overlap():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = list(build_store._chunk_text(text, max_len=4, overlap=1))

    assert chunks == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
        "w9",
    ]


def test_looks_like_russian_detects_script():
    assert build_store._looks_like_russian("Привет, world!") is True
    assert build_store._looks_like_russian("Hello world") is False
    assert build_store._looks_like_russian("") is True
