"""
Полный пайплайн подготовки данных для Qdrant.
Собирает тексты (документацию Rasa и локальные файлы),
разбивает их на чанки, векторизует моделью BAAI/bge-m3
и загружает в коллекцию Qdrant.

Запуск:
  python -m full.ingest
"""

from __future__ import annotations

from pathlib import Path

import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------- НАСТРОЙКИ ----------------

# Размер вектора должен совпадать с размерностью эмбеддингов модели BGE-M3
VECTOR_SIZE = 1024

# Имя коллекции в Qdrant
COLLECTION_NAME = "rasa_mvp"

# Стартовый URL для загрузки документации
BASE_URL = "https://rasa.com/docs/rasa/"

# Папка с локальными текстовыми файлами
LOCAL_DIR = Path("data")

# -------------------------------------------


def fetch_text_from_url(url: str) -> str:
    """Загружает HTML и возвращает очищенный текст."""

    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        [s.extract() for s in soup(["script", "style", "nav", "footer"])]
        return soup.get_text(separator=" ", strip=True)
    except Exception as exc:
        print(f"[fetch_text_from_url] Ошибка при загрузке {url}: {exc}")
        return ""


def chunk_text(text: str, max_len: int = 800, overlap: int = 120):
    """Делит текст на перекрывающиеся чанки."""

    words = text.split()
    step = max(1, max_len - overlap)
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + max_len])


def collect_documents():
    """Возвращает список документов из сети и локальных файлов."""

    docs = []

    urls = [BASE_URL]
    for url in tqdm(urls, desc="Сбор документации Rasa"):
        text = fetch_text_from_url(url)
        for chunk in chunk_text(text):
            docs.append({"text": chunk, "source": url, "lang": "en"})

    for file in LOCAL_DIR.glob("*.txt"):
        try:
            txt = file.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[collect_documents] Ошибка при чтении {file}: {exc}")
            continue
        for chunk in chunk_text(txt):
            docs.append({"text": chunk, "source": str(file), "lang": "ru"})

    print(f"Собрано {len(docs)} фрагментов текста")
    return docs


def ensure_collection(client: QdrantClient):
    """Создаёт коллекцию, если она отсутствует."""

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"Создаю коллекцию {COLLECTION_NAME} ...")
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        print(f"Коллекция {COLLECTION_NAME} уже существует.")


def main():
    client = QdrantClient(host="localhost", port=6333)
    ensure_collection(client)

    print("Загрузка модели BAAI/bge-m3 ...")
    model = SentenceTransformer("BAAI/bge-m3")

    docs = collect_documents()
    texts = [d["text"] for d in docs]

    print("Создание эмбеддингов ...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=16)

    print("Загрузка данных в Qdrant ...")
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=docs,
        ids=None,
        batch_size=64,
    )

    print(f"✅ Загружено {len(docs)} документов в коллекцию '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
