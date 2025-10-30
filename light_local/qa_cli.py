"""
Лёгкий CLI-пайплайн без Qdrant и без Argos Translate во время запроса.
Использует локально подготовленные файлы (embeddings.npy + payloads.jsonl).

Перед запуском необходимо выполнить подготовку:
  python -m light_local.build_store

Запуск самого CLI:
  python -m light_local.qa_cli "Как подключить бота?" --llm
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from full_qdrant.actions.llm_client import LLMClient

STORE_DIR = Path(__file__).resolve().parent / "store"
EMB_PATH = STORE_DIR / "embeddings.npy"
PAYLOAD_PATH = STORE_DIR / "payloads.jsonl"
META_PATH = STORE_DIR / "metadata.json"

SYSTEM_PROMPT = (
    "Ты русскоязычный ассистент. Отвечай только по предоставленным выдержкам. Если данных мало — скажи об этом."
)


def _load_payloads() -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    with PAYLOAD_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payloads.append(json.loads(line))
    return payloads


def _load_metadata() -> Dict[str, Any]:
    with META_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_store() -> None:
    missing = [p for p in (EMB_PATH, PAYLOAD_PATH, META_PATH) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise SystemExit(
            "Не найдено локальное хранилище. Сначала выполните подготовку: "
            "python -m light_local.build_store. \nОтсутствуют файлы: "
            + missing_str
        )


def _search(query: str, embeddings: np.ndarray, model: SentenceTransformer, topk: int) -> List[Dict[str, Any]]:
    query_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    scores = embeddings @ query_vec
    top_indices = np.argsort(-scores)[:topk]
    return [
        {"index": int(i), "score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str, help="Вопрос на русском языке")
    ap.add_argument("--llm", action="store_true", help="Получить финальный ответ от LLM")
    ap.add_argument("--k", type=int, default=4, help="Количество возвращаемых фрагментов")
    args = ap.parse_args()

    _ensure_store()
    payloads = _load_payloads()
    metadata = _load_metadata()
    embeddings = np.load(EMB_PATH)

    model = SentenceTransformer(metadata["model_name"])
    hits = _search(args.question.strip(), embeddings, model, args.k)

    if not hits:
        print("Контекст не найден")
        return

    print("=== КОНТЕКСТЫ ===")
    contexts: List[Dict[str, Any]] = []
    for pos, hit in enumerate(hits, start=1):
        payload = payloads[hit["index"]]
        contexts.append({"payload": payload, "score": hit["score"]})
        snippet = payload.get("text", "")[:500].replace("\n", " ")
        print(f"[{pos}] score={hit['score']:.3f} src={payload.get('source','')}")
        print(textwrap.fill(snippet, width=100))
        print("-" * 80)

    if not args.llm:
        return

    ctx = "\n\n".join(
        f"[{idx}] {c['payload'].get('text', '')[:1000]}" for idx, c in enumerate(contexts, start=1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Вопрос: {args.question.strip()}\n\nКонтекст:\n{ctx}\n\nОтвечай по-русски",
        },
    ]

    llm = LLMClient()
    answer = llm.generate(messages, max_tokens=600, temperature=0.2)
    print("\n=== ОТВЕТ LLM ===")
    print(textwrap.fill(answer, width=100))


if __name__ == "__main__":
    main()
