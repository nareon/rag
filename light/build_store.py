"""
Построение локального RAG-хранилища без Qdrant.
- читает документы из папки data (предполагается русский язык),
- при необходимости переводит их в русский с помощью Argos Translate,
- разбивает тексты на чанки,
- векторизует моделью SentenceTransformer,
- сохраняет эмбеддинги и метаданные на диск для дальнейшего оффлайн-поиска.

Запуск:
  python -m light.build_store
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

# Попытка подключить Argos только для подготовки хранилища
try:
    from argostranslate import translate as argos_translate

    _langs = argos_translate.get_installed_languages()
    _ru_lang = next((lang for lang in _langs if lang.code == "ru"), None)
    _translators_to_ru = []
    if _ru_lang:
        for lang in _langs:
            if lang.code == "ru":
                continue
            translator = lang.get_translation(_ru_lang)
            if translator:
                _translators_to_ru.append((lang.code, translator))
except Exception:
    _ru_lang = None
    _translators_to_ru = []

# ----------- Настройки хранилища -----------
LOCAL_DIR = Path("data")
STORE_DIR = Path(__file__).resolve().parent / "store"
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
# -------------------------------------------


def _chunk_text(text: str, max_len: int, overlap: int) -> Iterable[str]:
    words = text.split()
    step = max(1, max_len - overlap)
    for start in range(0, len(words), step):
        yield " ".join(words[start : start + max_len])


def _looks_like_russian(text: str) -> bool:
    if not text:
        return True
    cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text if ch.isalpha())
    lat = sum("a" <= ch.lower() <= "z" for ch in text if ch.isalpha())
    return cyr >= lat


def _translate_to_ru(text: str) -> str:
    if not text or _looks_like_russian(text) or not _translators_to_ru:
        return text
    for code, translator in _translators_to_ru:
        try:
            return translator.translate(text)
        except Exception:
            continue
    return text


def _load_local_documents() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for file in sorted(LOCAL_DIR.glob("*.txt")):
        try:
            text = file.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[build_store] Ошибка чтения {file}: {exc}")
            continue
        text = _translate_to_ru(text)
        for chunk in _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            docs.append({"text": chunk, "source": str(file), "lang": "ru"})
    return docs


def main() -> None:
    docs = _load_local_documents()
    if not docs:
        raise SystemExit("Не найдено документов в папке data/*.txt")

    texts = [doc["text"] for doc in docs]
    print(f"Собрано {len(texts)} фрагментов, подготавливаю эмбеддинги ...")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = STORE_DIR / "embeddings.npy"
    np.save(emb_path, np.asarray(embeddings, dtype=np.float32))

    payload_path = STORE_DIR / "payloads.jsonl"
    with payload_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            json.dump(doc, f, ensure_ascii=False)
            f.write("\n")

    metadata = {
        "model_name": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "count": len(docs),
        "translate_used": bool(_translators_to_ru),
    }
    with (STORE_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Хранилище готово: {emb_path} ({len(docs)} фрагментов)")


if __name__ == "__main__":
    main()
