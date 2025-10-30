"""
CLI-пайплайн без Rasa:
- принимает RU-вопрос,
- делает ретрив по Qdrant (RU и RU→EN),
- печатает найденные фрагменты,
- опционально вызывает LLM и печатает ответ.

Запуск:
  python qa_cli.py "Как подключить Telegram к Rasa?" --llm
"""

from __future__ import annotations
import argparse, textwrap
from actions.retriever import search_hybrid
from actions.llm_client import LLMClient

# опциональный локальный перевод Argos
try:
    from argostranslate import translate as argos_translate
    _langs = argos_translate.get_installed_languages()
    _ru = next((l for l in _langs if l.code=="ru"), None)
    _en = next((l for l in _langs if l.code=="en"), None)
    _to_en = _ru.get_translation(_en) if (_ru and _en) else None
except Exception:
    _to_en = None

SYSTEM_PROMPT = "Ты ассистент по Rasa. Отвечай кратко и по-русски. Используй только предоставленные выдержки. Если сведений мало, скажи об этом."

def maybe_translate_ru2en(q: str) -> str:
    if _to_en:
        try: return _to_en.translate(q)
        except Exception: return q
    return q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str, help="Вопрос на русском")
    ap.add_argument("--llm", action="store_true", help="Сгенерировать финальный ответ с LLM")
    ap.add_argument("--k", type=int, default=4, help="Количество контекстов после MMR")
    args = ap.parse_args()

    q_ru = args.question.strip()
    q_en = maybe_translate_ru2en(q_ru)

    # два ретрива и объединение по id
    hits_ru = search_hybrid(q_ru, lang_filter=["ru","en"], topk=args.k)
    hits_en = search_hybrid(q_en, lang_filter=["ru","en"], topk=args.k) if q_en else []
    by_id = {}
    for h in hits_en + hits_ru:
        by_id.setdefault(h["id"], h)

    contexts = list(by_id.values())[:args.k]
    if not contexts:
        print("Нет результатов.")
        return

    print("=== КОНТЕКСТЫ ===")
    for i, h in enumerate(contexts, 1):
        src = h["payload"].get("source","")
        lang = h["payload"].get("lang","")
        txt = h["payload"].get("text","")[:500].replace("\n"," ")
        print(f"[{i}] score={h['score']:.3f} lang={lang} src={src}")
        print(textwrap.fill(txt, width=100))
        print("-"*80)

    if not args.llm:
        return

    # Сшивка контекста и генерация ответа
    ctx = "\n\n".join([f"[{i}] {c['payload'].get('text','')[:1000]}" for i,c in enumerate(contexts,1)])
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"Вопрос: {q_ru}\n\nКонтекст:\n{ctx}\n\nОтвечай по-русски."}
    ]
    llm = LLMClient()
    ans = llm.generate(messages, max_tokens=600, temperature=0.2)
    print("\n=== ОТВЕТ LLM ===")
    print(textwrap.fill(ans, width=100))

if __name__ == "__main__":
    main()
