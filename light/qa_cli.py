"""
Лёгкий CLI-пайплайн без Qdrant и без Argos Translate во время запроса.
Использует локально подготовленные файлы (embeddings.npy + payloads.jsonl).

Перед запуском необходимо выполнить подготовку:
  python -m light.build_store

Запуск самого CLI:
  python -m light.qa_cli "Как подключить бота?" --llm
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from full.actions.llm_client import LLMClient

STORE_DIR = Path(__file__).resolve().parent / "store"
# Файлы, которые формируются при помощи ``light.build_store``.
EMB_PATH = STORE_DIR / "embeddings.npy"
PAYLOAD_PATH = STORE_DIR / "payloads.jsonl"
META_PATH = STORE_DIR / "metadata.json"

SYSTEM_PROMPT = (
    "Ты русскоязычный ассистент. Отвечай только по предоставленным выдержкам. Если данных мало — скажи об этом."
)
# Локальный промпт упрощён, но сохраняет требование ссылаться на контекст.


def _load_payloads() -> List[Dict[str, Any]]:
    """Загружает метаданные контекстов из JSONL."""

    payloads: List[Dict[str, Any]] = []
    with PAYLOAD_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                # JSONL позволяет обрабатывать большие датасеты стримингово.
                payloads.append(json.loads(line))
    return payloads


def _load_metadata() -> Dict[str, Any]:
    """Читает параметры построения стора (модель, размер чанков, и т.д.)."""

    with META_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_store() -> None:
    """Проверяет наличие всех обязательных артефактов локального стора."""

    missing = [p for p in (EMB_PATH, PAYLOAD_PATH, META_PATH) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise SystemExit(
            "Не найдено локальное хранилище. Сначала выполните подготовку: "
            "python -m light.build_store. \nОтсутствуют файлы: "
            + missing_str
        )


def _search(query: str, embeddings: np.ndarray, model: SentenceTransformer, topk: int) -> List[Dict[str, Any]]:
    """Считает косинусное сходство между запросом и хранилищем."""

    query_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    scores = embeddings @ query_vec
    top_indices = np.argsort(-scores)[:topk]
    return [
        {"index": int(i), "score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0  # отрицательные значения считаем шумом
    ]


def main() -> None:
    """Основной CLI, который ищет по локальному хранилищу и опционально вызывает LLM."""

    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str, help="Вопрос на русском языке")
    ap.add_argument("--llm", action="store_true", help="Получить финальный ответ от LLM")
    ap.add_argument("--k", type=int, default=4, help="Количество возвращаемых фрагментов")
    args = ap.parse_args()

    _ensure_store()
    payloads = _load_payloads()
    metadata = _load_metadata()
    # Эмбеддинги подгружаем лениво, только когда уверены, что файлы есть.
    embeddings = np.load(EMB_PATH)

    # CLI использует ту же модель, что была применена при построении стора,
    # иначе косинусные расстояния будут некорректными.
    model = SentenceTransformer(metadata["model_name"])
    hits = _search(args.question.strip(), embeddings, model, args.k)

    if not hits:
        # Даже без LLM информируем пользователя о пустой выдаче.
        print("Контекст не найден")
        return

    print("=== КОНТЕКСТЫ ===")
    contexts: List[Dict[str, Any]] = []
    for pos, hit in enumerate(hits, start=1):
        payload = payloads[hit["index"]]
        # Сохраняем весь payload для передачи в LLM на следующем шаге.
        contexts.append({"payload": payload, "score": hit["score"]})
        snippet = payload.get("text", "")[:500].replace("\n", " ")
        # Формат вывода совпадает с «тяжёлой» версией для единообразия UX.
        print(f"[{pos}] score={hit['score']:.3f} src={payload.get('source','')}")
        print(textwrap.fill(snippet, width=100))
        print("-" * 80)

    if not args.llm:
        return

    # Формируем подсказку в стиле RAG: объединяем топ-k контекстов и просим LLM
    # ссылаться на них в ответе.
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

    # Используем тот же клиент, что и в «тяжёлой» версии, чтобы не дублировать
    # настройки и логику ретраев.
    llm = LLMClient()
    # Температуру держим низкой, чтобы ответы были более детерминированными.
    answer = llm.generate(messages, max_tokens=600, temperature=0.2)
    print("\n=== ОТВЕТ LLM ===")
    # Выводим текст с обрезкой по ширине для удобства чтения в терминале.
    print(textwrap.fill(answer, width=100))


if __name__ == "__main__":
    main()
