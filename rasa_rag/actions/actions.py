"""Custom actions powering the intelligent assistant pipeline."""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import EventType

from .llm_client import LLMClient
from .retriever import search_hybrid

logger = logging.getLogger(__name__)

# Optional local translation via Argos Translate. The dependency is optional, so we
# guard the import to keep the action server lightweight by default.
try:  # pragma: no cover - optional dependency
    from argostranslate import translate as argos_translate

    _langs = argos_translate.get_installed_languages()
    _ru = next((lang for lang in _langs if lang.code == "ru"), None)
    _en = next((lang for lang in _langs if lang.code == "en"), None)
    _to_en = _ru.get_translation(_en) if (_ru and _en) else None
except Exception:  # pragma: no cover - optional dependency
    _to_en = None

SYSTEM_PROMPT = (
    "Ты интеллектуальный ассистент компании. Общайся по-русски, дружелюбно и технологично. "
    "Используй только предоставленные выдержки и ссылайся на них по номерам. Если сведений мало, честно скажи об этом."
)


def maybe_translate_ru2en(text: str) -> str:
    """Translate Russian text into English if a local translator is available."""

    if not text:
        return ""
    if _to_en is None:
        return text

    try:  # pragma: no cover - relies on optional dependency
        return _to_en.translate(text)
    except Exception:  # pragma: no cover - optional dependency may fail at runtime
        # In case of translation issues we continue with the original query so the
        # pipeline still produces a meaningful result.
        return text


def _merge_hits(*hit_groups: Iterable[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
    """Merge retriever results, preserving only the highest score per id."""

    by_id: Dict[str, Dict[str, Any]] = {}
    for hit in (item for group in hit_groups for item in group):
        hit_id = hit.get("id")
        if hit_id is None:
            continue

        score = hit.get("score")
        if not isinstance(score, (int, float)):
            continue

        key = str(hit_id)
        stored = by_id.get(key)
        if stored is None or float(score) > float(stored.get("score", float("-inf"))):
            by_id[key] = hit

    # Sort by score descending and keep the requested number of items.
    return sorted(by_id.values(), key=lambda item: float(item["score"]), reverse=True)[:limit]


class ActionRAGAnswer(Action):
    """Generate an answer using the retrieval-augmented generation pipeline."""

    def __init__(self, *, contexts_to_return: int = 4) -> None:
        self._llm: Optional[LLMClient] = None
        self._contexts_to_return = contexts_to_return

    def name(self) -> Text:
        return "action_rag_answer"

    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        query = (tracker.latest_message.get("text") or "").strip()
        if not query:
            dispatcher.utter_message(text="Не удалось распознать вопрос. Попробуйте сформулировать его иначе.")
            return []

        logger.debug("RAG pipeline received query: %s", query)

        query_en = maybe_translate_ru2en(query)

        try:
            hits_ru = search_hybrid(query, lang_filter=["ru", "en"], topk=self._contexts_to_return)
            hits_en: List[Dict[str, Any]] = []
            if query_en and query_en != query:
                hits_en = search_hybrid(query_en, lang_filter=["ru", "en"], topk=self._contexts_to_return)
        except Exception as exc:  # pragma: no cover - depends on external services
            logger.exception("Retriever call failed: %%s", exc)
            dispatcher.utter_message(text="Не удалось получить контекст из базы знаний. Попробуйте позже.")
            return []

        contexts = _merge_hits(hits_ru, hits_en, limit=self._contexts_to_return)
        if not contexts:
            dispatcher.utter_message(text="К сожалению, мне не удалось найти информацию по вашему вопросу.")
            return []

        stitched_context = "\n\n".join(
            [f"[{i}] {c['payload'].get('text', '')[:1000]}" for i, c in enumerate(contexts, 1)]
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    Вопрос: {query}

                    Контекст:
                    {stitched_context}

                    Отвечай по-русски, ссылайся на выдержки по номерам при необходимости.
                    """
                ).strip(),
            },
        ]

        try:
            answer = self._ensure_llm().generate(messages, max_tokens=600, temperature=0.2)
        except Exception as exc:  # pragma: no cover - depends on external services
            logger.exception("LLM generation failed: %%s", exc)
            dispatcher.utter_message(text="Я нашёл контекст, но не смог сформировать ответ. Попробуйте позже.")
            return []

        dispatcher.utter_message(text=answer)

        # Provide lightweight metadata with the sources so that channels capable
        # of rendering it (e.g. custom frontends) can highlight the passages.
        dispatcher.utter_message(
            text="Источники:",
            metadata={
                "contexts": [
                    {
                        "rank": idx,
                        "id": ctx.get("id"),
                        "score": ctx.get("score"),
                        "source": ctx.get("payload", {}).get("source"),
                        "lang": ctx.get("payload", {}).get("lang"),
                    }
                    for idx, ctx in enumerate(contexts, 1)
                ]
            },
        )

        return []


__all__ = ["ActionRAGAnswer"]
