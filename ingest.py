# ingest/ingest.py
"""
Назначение:
  Скрипт собирает тексты (документацию Rasa и локальные файлы),
  разбивает их на смысловые фрагменты (чанки),
  векторизует моделью BAAI/bge-m3
  и записывает результаты в векторную базу Qdrant.

Результат:
  Создаётся коллекция 'rasa_mvp' с эмбеддингами и метаданными.
"""

import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

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
    """
    Загружает HTML со страницы и очищает его от служебных тегов.
    Возвращает чистый текст.
    """
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        # Удаляем ненужные блоки (скрипты, стили, навигацию)
        [s.extract() for s in soup(["script", "style", "nav", "footer"])]

        # Преобразуем в плоский текст
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"[fetch_text_from_url] Ошибка при загрузке {url}: {e}")
        return ""


def chunk_text(text: str, max_len=800, overlap=120):
    """
    Делит длинный текст на перекрывающиеся куски (чанки),
    чтобы модель могла работать с контекстом оптимального размера.
    """
    words = text.split()
    for i in range(0, len(words), max_len - overlap):
        yield " ".join(words[i : i + max_len])


def collect_documents():
    """
    Собирает документы из сети (Rasa docs) и из локальных файлов.
    Каждый документ — словарь: {'text': ..., 'source': ..., 'lang': ...}
    """
    docs = []

    # 1. Загрузка Rasa документации
    urls = [BASE_URL]
    for url in tqdm(urls, desc="Сбор документации Rasa"):
        text = fetch_text_from_url(url)
        for chunk in chunk_text(text):
            docs.append({"text": chunk, "source": url, "lang": "en"})

    # 2. Загрузка локальных текстов из папки data/
    for file in LOCAL_DIR.glob("*.txt"):
        try:
            txt = file.read_text(encoding="utf-8")
            for chunk in chunk_text(txt):
                docs.append({"text": chunk, "source": str(file), "lang": "ru"})
        except Exception as e:
            print(f"[collect_documents] Ошибка при чтении {file}: {e}")

    print(f"Собрано {len(docs)} фрагментов текста")
    return docs


def ensure_collection(client: QdrantClient):
    """
    Проверяет, существует ли коллекция в Qdrant.
    Если нет — создаёт новую с параметрами модели BGE-M3.
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"Создаю коллекцию {COLLECTION_NAME} ...")
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )
    else:
        print(f"Коллекция {COLLECTION_NAME} уже существует.")


def main():
    # --- 1. Подключение к Qdrant ---
    client = QdrantClient(host="localhost", port=6333)
    ensure_collection(client)

    # --- 2. Инициализация модели эмбеддингов ---
    # BGE-M3 — мультиязычная, поддерживает dense и sparse-вектора
    print("Загрузка модели BAAI/bge-m3 ...")
    model = SentenceTransformer("BAAI/bge-m3")

    # --- 3. Сбор текстов ---
    docs = collect_documents()
    texts = [d["text"] for d in docs]

    # --- 4. Векторизация ---
    print("Создание эмбеддингов ...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=16)

    # --- 5. Загрузка в Qdrant ---
    print("Загрузка данных в Qdrant ...")
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=docs,  # сюда записываются тексты и метаданные
        ids=None,
        batch_size=64,
    )

    print(f"✅ Загружено {len(docs)} документов в коллекцию '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
