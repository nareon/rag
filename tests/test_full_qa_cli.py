"""Tests for utilities in full.qa_cli."""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stub heavy dependencies so full.qa_cli can be imported without optional packages.
if "full.actions.retriever" not in sys.modules:
    retriever_stub = types.ModuleType("full.actions.retriever")

    def _fake_search(*_args, **_kwargs):  # pragma: no cover - helper stub
        return []

    retriever_stub.search_hybrid = _fake_search  # type: ignore[attr-defined]
    sys.modules["full.actions.retriever"] = retriever_stub
    sys.modules.setdefault("actions.retriever", retriever_stub)

if "full.actions.llm_client" not in sys.modules:
    llm_stub = types.ModuleType("full.actions.llm_client")

    class _DummyLLMClient:  # pragma: no cover - helper stub
        def generate(self, *_, **__):
            return ""

    llm_stub.LLMClient = _DummyLLMClient  # type: ignore[attr-defined]
    sys.modules["full.actions.llm_client"] = llm_stub
    sys.modules.setdefault("actions.llm_client", llm_stub)

qa_cli = importlib.import_module("full.qa_cli")


def test_merge_hits_prefers_highest_score_per_id():
    hits_primary = [
        {"id": "A", "score": 0.4, "payload": {"text": "one"}},
        {"id": "B", "score": 0.3, "payload": {"text": "two"}},
    ]
    hits_secondary = [
        {"id": "A", "score": 0.9, "payload": {"text": "one (better)"}},
        {"id": None, "score": 1.2, "payload": {}},  # ignored: missing id
        {"id": "C", "score": "bad", "payload": {}},  # ignored: score is not numeric
    ]

    merged = qa_cli._merge_hits(hits_primary, hits_secondary, limit=3)

    assert [item["id"] for item in merged] == ["A", "B"]
    assert merged[0]["score"] == 0.9


def test_merge_hits_respects_limit_and_sorting():
    hits_one = [
        {"id": "A", "score": 0.2, "payload": {}},
        {"id": "B", "score": 0.6, "payload": {}},
    ]
    hits_two = [
        {"id": "C", "score": 0.7, "payload": {}},
        {"id": "D", "score": 0.1, "payload": {}},
    ]

    merged = qa_cli._merge_hits(hits_one, hits_two, limit=2)

    assert [item["id"] for item in merged] == ["C", "B"]
    assert all(isinstance(item["score"], float) for item in merged)
