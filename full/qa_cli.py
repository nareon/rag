"""
CLI-пайплайн без Rasa:
- принимает RU-вопрос,
- делает ретрив по Qdrant (RU и RU→EN),
- печатает найденные фрагменты,
- опционально вызывает LLM и печатает ответ.

Запуск:
  python -m full.qa_cli "Как подключить Telegram к Rasa?" --llm
"""

from __future__ import annotations

import argparse
import textwrap
from typing import Dict, Iterable, List

try:
    from .actions.retriever import search_hybrid
    from .actions.llm_client import LLMClient
except ImportError:  # pragma: no cover - fallback for direct script execution
    import pathlib
    import sys

    pkg_root = pathlib.Path(__file__).resolve().parent
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from actions.retriever import search_hybrid  # type: ignore
    from actions.llm_client import LLMClient  # type: ignore

# опциональный локальный перевод Argos
try:
    from argostranslate import translate as argos_translate

    # Собираем язык RU и EN один раз, чтобы не создавать переводчик в цикле.
    _langs = argos_translate.get_installed_languages()
    _ru = next((l for l in _langs if l.code == "ru"), None)
    _en = next((l for l in _langs if l.code == "en"), None)
    _to_en = _ru.get_translation(_en) if (_ru and _en) else None
except Exception:
    _to_en = None

SYSTEM_PROMPT = (
    "Ты ассистент по Rasa. Отвечай кратко и по-русски. Используй только предоставленные выдержки. "
    "Если сведений мало, скажи об этом."
)
# Такой же промпт используется и в боевом боте, чтобы поведение CLI совпадало.


def maybe_translate_ru2en(q: str) -> str:
    """Translate the Russian query into English if Argos Translate is available."""

    if _to_en:
        try:
            return _to_en.translate(q)
        except Exception:
            # Переводчик может не справиться с длинными или «шумными» запросами;
            # в таком случае оставляем оригинал.
            return q
    return q


def _merge_hits(
    *hit_groups: Iterable[Dict[str, object]], limit: int = 4
) -> List[Dict[str, object]]:
    """Объединяет результаты ретрива по id и сортирует по score."""

    # Функция принимает несколько списков результатов (например, ru и en
    # запросы). Для каждого уникального ``id`` мы сохраняем лучший фрагмент и в
    # конце сортируем по итоговому скору, чтобы показать пользователю наиболее
    # релевантный контекст.

    by_id: Dict[str, Dict[str, object]] = {}
    for hit in (h for group in hit_groups for h in group):
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

    return sorted(by_id.values(), key=lambda h: float(h["score"]), reverse=True)[:limit]

def main():
    """Entry-point for the CLI utility."""

    ap = argparse.ArgumentParser()
    # Опции минимальны: вопрос, флаг LLM и количество возвращаемых контекстов.
    ap.add_argument("question", type=str, help="Вопрос на русском")
    ap.add_argument("--llm", action="store_true", help="Сгенерировать финальный ответ с LLM")
    ap.add_argument("--k", type=int, default=4, help="Количество контекстов после MMR")
    args = ap.parse_args()

    q_ru = args.question.strip()
    # Переводим вопрос заранее, чтобы не дергать переводчик дважды.
    q_en = maybe_translate_ru2en(q_ru)

    # два ретрива и объединение по id
    # Сначала ищем по русскому запросу, затем повторяем попытку на переведённой
    # версии.  Это помогает, когда в базе много англоязычных выдержек.
    hits_ru = search_hybrid(q_ru, lang_filter=["ru", "en"], topk=args.k)
    hits_en = search_hybrid(q_en, lang_filter=["ru", "en"], topk=args.k) if q_en else []
    contexts = _merge_hits(hits_en, hits_ru, limit=args.k)
    if not contexts:
        # В случае отсутствия результатов не имеет смысла обращаться к LLM.
        print("Нет результатов.")
        return

    print("=== КОНТЕКСТЫ ===")
    for i, h in enumerate(contexts, 1):
        src = h["payload"].get("source", "")
        lang = h["payload"].get("lang", "")
        txt = h["payload"].get("text", "")[:500].replace("\n", " ")
        # Показываем базовые метаданные, чтобы оператор мог оценить релевантность.
        print(f"[{i}] score={h['score']:.3f} lang={lang} src={src}")
        print(textwrap.fill(txt, width=100))
        print("-" * 80)

    if not args.llm:
        return

    # Сшивка контекста и генерация ответа
    # Модель получает «сырой» контекст, поэтому дополнительно проставляем
    # номера чанков, чтобы ассистент мог ссылаться на источник в ответе.
    ctx = "\n\n".join(
        [f"[{i}] {c['payload'].get('text', '')[:1000]}" for i, c in enumerate(contexts, 1)]
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Вопрос: {q_ru}\n\nКонтекст:\n{ctx}\n\nОтвечай по-русски.",
        },
    ]
    # Используем один и тот же клиент, что и в Rasa-экшенах, чтобы поведение
    # CLI и полноценного бота совпадало.
    llm = LLMClient()
    # Температуру держим низкой, чтобы ответы были более детерминированными.
    ans = llm.generate(messages, max_tokens=600, temperature=0.2)
    print("\n=== ОТВЕТ LLM ===")
    # Выводим текст с обрезкой по ширине для удобства чтения в терминале.
    print(textwrap.fill(ans, width=100))

if __name__ == "__main__":
    main()
